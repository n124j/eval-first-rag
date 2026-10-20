"""Command line: ingest, ask, eval, refresh-recordings and gate."""

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import zipfile
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

from evalrag.config import ConfigError, Settings
from evalrag.corpus import CorpusError, load_corpus
from evalrag.errors import CacheMismatchError, IndexDataError
from evalrag.evals.dataset import DatasetError, load_golden
from evalrag.evals.gate import (
    BaselineError,
    MetricsError,
    dump_metrics,
    gate,
    load_metrics,
)
from evalrag.evals.report import METRIC_ORDER, render_markdown
from evalrag.evals.runner import run_eval
from evalrag.generation.llm import (
    LLMClient,
    LLMError,
    OpenAICompatClient,
    RecordingClient,
    ReplayClient,
)
from evalrag.ingest import ingest, open_retriever
from evalrag.pipeline import RagPipeline
from evalrag.retrieval.dense import EmptyIndex
from evalrag.retrieval.embedder import (
    Embedder,
    EmbeddingError,
    OpenAICompatEmbedder,
    RecordingEmbedder,
    ReplayEmbedder,
    ReplayMiss,
)
from evalrag.sut import make_sut

RUNTIME_ERRORS = (
    ReplayMiss,
    FileNotFoundError,
    EmbeddingError,
    LLMError,
    DatasetError,
    CorpusError,
    BaselineError,
    MetricsError,
    EmptyIndex,
    ConfigError,
    CacheMismatchError,
    IndexDataError,
    OSError,
    sqlite3.Error,
    UnicodeDecodeError,
    zipfile.BadZipFile,
    json.JSONDecodeError,
)


class ComponentFactory:
    """Builds the embedder and LLM client; tests override `live_*` with fakes."""

    def live_embedder(self, settings: Settings) -> Embedder:
        return OpenAICompatEmbedder(settings.embed_base_url, settings.embed_model)

    def live_llm(self, settings: Settings) -> LLMClient:
        return OpenAICompatClient(settings.llm_base_url, settings.llm_model)

    def embedder(self, settings: Settings, record: Path | None, replay: Path | None) -> Embedder:
        if replay is not None:
            return ReplayEmbedder(replay)
        inner = self.live_embedder(settings)
        if record is not None:
            return RecordingEmbedder(inner, record, model=settings.embed_model)
        return inner

    def llm(self, settings: Settings, record: Path | None, replay: Path | None) -> LLMClient:
        if replay is not None:
            return ReplayClient(replay)
        inner = self.live_llm(settings)
        if record is not None:
            return RecordingClient(inner, record)
        return inner


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}")
    return value


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _flush(component: object) -> None:
    flush = getattr(component, "flush", None)
    if callable(flush):
        flush()


def _entry_count(component: object) -> int:
    return int(getattr(component, "entry_count", 0))


def _exclusive(parser: argparse.ArgumentParser, record: str, replay: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(record, metavar="PATH")
    group.add_argument(replay, metavar="PATH")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalrag", description="Eval-first RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk the corpus and build the indexes")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--data-dir")
    _exclusive(p, "--record-embeddings", "--replay-embeddings")

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--data-dir")
    _exclusive(p, "--record-llm", "--replay-llm")
    p.add_argument("--replay-embeddings", metavar="PATH")

    p = sub.add_parser("eval", help="run the golden set and write metrics and a report")
    p.add_argument("--golden", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--metrics-out", required=True)
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--data-dir")
    p.add_argument("--k", type=_positive_int, default=5)
    _exclusive(p, "--record-llm", "--replay-llm")
    _exclusive(p, "--record-embeddings", "--replay-embeddings")
    p.add_argument("--model")
    p.add_argument("--hardware")
    p.add_argument("--date")

    p = sub.add_parser(
        "refresh-recordings", help="re-record embeddings and completions against live endpoints"
    )
    p.add_argument("--golden", required=True)
    p.add_argument("--embeddings-out", required=True)
    p.add_argument("--llm-out", required=True)
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--data-dir")
    p.add_argument("--k", type=_positive_int, default=5)
    p.add_argument("--report-out", default="evals/reports/live.md")
    p.add_argument("--metrics-out", default="evals/reports/live-metrics.json")
    p.add_argument("--model")
    p.add_argument("--hardware")
    p.add_argument("--date")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("gate", help="fail when metrics regress past the baseline")
    p.add_argument("--metrics", required=True)
    p.add_argument("--baseline", required=True)
    return parser


def _settings(env: Mapping[str, str], *replayed: object) -> Settings:
    """Endpoints are required unless every network component is replayed."""
    return Settings.from_env(env, require_endpoints=not all(replayed))


def _data_dir(args: argparse.Namespace, settings: Settings) -> Path:
    return Path(args.data_dir) if args.data_dir else settings.data_dir


def _ingest_step(args, settings, embedder) -> None:
    """Build the index and flush the embedder; shared by `ingest` and `refresh-recordings`."""
    summary = ingest(settings, Path(args.corpus), _data_dir(args, settings), embedder)
    _flush(embedder)
    print(f"ingested {summary.documents} documents, {summary.chunks} chunks, dim {summary.dim}")


def _cmd_ingest(args, env, factory) -> int:
    settings = _settings(env, args.replay_embeddings)
    embedder = factory.embedder(
        settings, _path(args.record_embeddings), _path(args.replay_embeddings)
    )
    _ingest_step(args, settings, embedder)
    return 0


@contextlib.contextmanager
def open_pipeline(settings, data_dir, k, embedder, llm) -> Iterator[RagPipeline]:
    retriever, chunks_by_id = open_retriever(settings, data_dir, embedder)
    try:
        yield RagPipeline(retriever, llm, chunks_by_id, k)
    finally:
        retriever.close()


def _cmd_ask(args, env, factory) -> int:
    settings = _settings(env, args.replay_llm, args.replay_embeddings)
    embedder = factory.embedder(settings, None, _path(args.replay_embeddings))
    llm = factory.llm(settings, _path(args.record_llm), _path(args.replay_llm))
    with open_pipeline(settings, _data_dir(args, settings), settings.top_k, embedder, llm) as p:
        answer = p.ask(args.question)
    _flush(llm)
    print(answer.text)
    for citation in answer.citations:
        print(f'  cited {citation.chunk_id}: "{citation.quote}"')
    print(f"refused: {str(answer.refused).lower()}")
    print(f"citations valid: {str(answer.report.valid).lower()}")
    return 0


def _eval_step(args, env, settings, embedder, llm) -> None:
    """Run the golden set, flush both components, write outputs; shared with refresh."""
    known = {doc.doc_id for doc in load_corpus(Path(args.corpus))}
    items = load_golden(Path(args.golden), known_sources=known)
    with open_pipeline(settings, _data_dir(args, settings), args.k, embedder, llm) as pipeline:
        result = run_eval(make_sut(pipeline), items, k=args.k)
    _flush(llm)
    _flush(embedder)

    meta = {
        "model": args.model or str(llm.params.get("model", "")),
        "hardware": args.hardware or env.get("HARDWARE") or "unspecified",
        "date": args.date or datetime.now(UTC).date().isoformat(),
    }
    metrics_out, report_out = Path(args.metrics_out), Path(args.report_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    dump_metrics(result.metrics, metrics_out)
    report_out.write_text(render_markdown(result, meta), encoding="utf-8", newline="\n")
    for name in METRIC_ORDER:
        print(f"{name}: {result.metrics[name]}")


def _cmd_eval(args, env, factory) -> int:
    settings = _settings(env, args.replay_llm, args.replay_embeddings)
    embedder = factory.embedder(
        settings, _path(args.record_embeddings), _path(args.replay_embeddings)
    )
    llm = factory.llm(settings, _path(args.record_llm), _path(args.replay_llm))
    _eval_step(args, env, settings, embedder, llm)
    return 0


def _cmd_refresh(args, env, factory) -> int:
    emb_out, llm_out = Path(args.embeddings_out), Path(args.llm_out)
    for path in (emb_out, llm_out):
        if path.exists() and not args.overwrite:
            print(
                f"error: {path} already exists; pass --overwrite to start a fresh recording",
                file=sys.stderr,
            )
            return 1
    settings = Settings.from_env(env, require_endpoints=True)
    if args.overwrite:
        emb_out.unlink(missing_ok=True)
        llm_out.unlink(missing_ok=True)
    embedder = factory.embedder(settings, emb_out, None)
    llm = factory.llm(settings, llm_out, None)
    _ingest_step(args, settings, embedder)
    _eval_step(args, env, settings, embedder, llm)
    print(
        f"recorded {_entry_count(embedder)} embeddings and {_entry_count(llm)} completions"
    )
    return 0


def _cmd_gate(args, env, factory) -> int:
    return gate(load_metrics(Path(args.metrics)), Path(args.baseline))


_COMMANDS = {
    "ingest": _cmd_ingest,
    "ask": _cmd_ask,
    "eval": _cmd_eval,
    "refresh-recordings": _cmd_refresh,
    "gate": _cmd_gate,
}


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    factory: ComponentFactory | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    env = os.environ if env is None else env
    factory = factory or ComponentFactory()
    try:
        return _COMMANDS[args.command](args, env, factory)
    except RUNTIME_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
