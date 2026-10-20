"""Command line: ingest, ask, eval, refresh-recordings and gate."""

import argparse
import contextlib
import json
import math
import os
import sqlite3
import sys
import zipfile
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

from evalrag.config import ConfigError, Settings, is_valid_min_dense_score
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


def _min_dense_score(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not is_valid_min_dense_score(value):
        raise argparse.ArgumentTypeError(f"must be a finite number in [0.0, 1.0], got {text!r}")
    return value


def _add_min_dense_score(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-dense-score",
        type=_min_dense_score,
        metavar="FLOAT",
        help="abstention threshold on the dense rank-1 cosine; default: the MIN_DENSE_SCORE "
        "setting; 0 disables",
    )


def _min_score(args: argparse.Namespace, settings: Settings) -> float:
    return args.min_dense_score if args.min_dense_score is not None else settings.min_dense_score


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
    _add_min_dense_score(p)

    p = sub.add_parser("eval", help="run the golden set and write metrics and a report")
    p.add_argument("--golden", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--metrics-out", required=True)
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--data-dir")
    p.add_argument("--k", type=_positive_int, help="default: the TOP_K setting")
    _exclusive(p, "--record-llm", "--replay-llm")
    _exclusive(p, "--record-embeddings", "--replay-embeddings")
    _add_min_dense_score(p)
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
    p.add_argument("--k", type=_positive_int, help="default: the TOP_K setting")
    p.add_argument("--report-out", default="evals/reports/live.md")
    p.add_argument("--metrics-out", default="evals/reports/live-metrics.json")
    p.add_argument("--model")
    p.add_argument("--hardware")
    p.add_argument("--date")
    p.add_argument("--overwrite", action="store_true")
    _add_min_dense_score(p)

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
    """Build the index; shared by `ingest` and `refresh-recordings`. Callers flush."""
    summary = ingest(settings, Path(args.corpus), _data_dir(args, settings), embedder)
    print(f"ingested {summary.documents} documents, {summary.chunks} chunks, dim {summary.dim}")


def _cmd_ingest(args, env, factory) -> int:
    settings = _settings(env, args.replay_embeddings)
    embedder = factory.embedder(
        settings, _path(args.record_embeddings), _path(args.replay_embeddings)
    )
    _ingest_step(args, settings, embedder)
    _flush(embedder)
    return 0


@contextlib.contextmanager
def open_pipeline(
    settings, data_dir, k, embedder, llm, min_dense_score: float | None = None
) -> Iterator[RagPipeline]:
    """`min_dense_score` defaults to the setting; pass a value to override it."""
    if min_dense_score is None:
        min_dense_score = settings.min_dense_score
    retriever, chunks_by_id = open_retriever(settings, data_dir, embedder)
    try:
        yield RagPipeline(retriever, llm, chunks_by_id, k, min_dense_score=min_dense_score)
    finally:
        retriever.close()


def _cmd_ask(args, env, factory) -> int:
    settings = _settings(env, args.replay_llm, args.replay_embeddings)
    embedder = factory.embedder(settings, None, _path(args.replay_embeddings))
    llm = factory.llm(settings, _path(args.record_llm), _path(args.replay_llm))
    data_dir, min_score = _data_dir(args, settings), _min_score(args, settings)
    with open_pipeline(settings, data_dir, settings.top_k, embedder, llm, min_score) as p:
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
    # One k for both the pipeline and the metrics; it defaults to the TOP_K setting.
    k = args.k if args.k is not None else settings.top_k
    data_dir, min_score = _data_dir(args, settings), _min_score(args, settings)
    with open_pipeline(settings, data_dir, k, embedder, llm, min_score) as pipeline:
        result = run_eval(make_sut(pipeline), items, k=k)
    _flush(llm)
    _flush(embedder)

    meta = {
        "model": args.model or str(llm.params.get("model", "")),
        "hardware": args.hardware or env.get("HARDWARE") or "unspecified",
        "date": args.date or datetime.now(UTC).date().isoformat(),
        # The effective abstention threshold (flag or setting); 0 means disabled.
        "min_dense_score": format(min_score, "g"),
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
    known = {doc.doc_id for doc in load_corpus(Path(args.corpus))}
    load_golden(Path(args.golden), known_sources=known)
    # Record into temporary files; the finals are replaced only after everything succeeded.
    emb_tmp = emb_out.with_name(emb_out.name + ".refresh-tmp")
    llm_tmp = llm_out.with_name(llm_out.name + ".refresh-tmp")
    for tmp in (emb_tmp, llm_tmp):
        tmp.unlink(missing_ok=True)
    try:
        embedder = factory.embedder(settings, emb_tmp, None)
        llm = factory.llm(settings, llm_tmp, None)
        _ingest_step(args, settings, embedder)
        _eval_step(args, env, settings, embedder, llm)  # flushes both recorders once
        os.replace(emb_tmp, emb_out)
        os.replace(llm_tmp, llm_out)
    except BaseException:
        for tmp in (emb_tmp, llm_tmp):
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        raise
    print(f"recorded {_entry_count(embedder)} embeddings and {_entry_count(llm)} completions")
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
