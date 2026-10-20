import json
import socket
from pathlib import Path

import numpy as np
import pytest

from evalrag.cli import ComponentFactory, _build_parser, main
from evalrag.generation.answer import REFUSAL_TEXT
from evalrag.generation.llm import LLMError
from evalrag.ingest import load_chunks
from evalrag.retrieval.dense import DenseIndex
from evalrag.retrieval.embedder import cache_key
from tests.unit.fakes import HashEmbedder, ScriptedLLM, make_corpus

ENV = {
    "LLM_BASE_URL": "http://localhost:1/v1",
    "LLM_MODEL": "fake-llm",
    "EMBED_BASE_URL": "http://localhost:2/v1",
    "EMBED_MODEL": "fake-embed",
    "CHUNK_SIZE": "12",
    "CHUNK_OVERLAP": "2",
}

METRICS = {
    "recall_at_k",
    "mrr",
    "citation_validity_rate",
    "correct_refusal_rate",
    "over_refusal_rate",
    "latency_p50_s",
    "latency_p95_s",
    "prompt_tokens_total",
    "completion_tokens_total",
}


class FakeFactory(ComponentFactory):
    """Live components are fakes; record/replay wrapping is the real default logic."""

    def live_embedder(self, settings):
        return HashEmbedder()

    def live_llm(self, settings):
        return ScriptedLLM()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return make_corpus(tmp_path)


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    rows = [
        {
            "id": "g1",
            "question": "How does the alpha module sort lists?",
            "expected_sources": ["alpha"],
            "answerable": True,
            "kind": "single",
        },
        {
            "id": "g2",
            "question": "What do beta and gamma do?",
            "expected_sources": ["beta", "gamma"],
            "answerable": True,
            "kind": "multi",
        },
        {
            "id": "g3",
            "question": "An unanswerable question about the weather",
            "expected_sources": [],
            "answerable": False,
            "kind": "unanswerable",
        },
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def run(argv, env=ENV, factory=None):
    return main([str(a) for a in argv], env=env, factory=factory or FakeFactory())


def ingest_live(tmp_path: Path, corpus: Path) -> Path:
    data = tmp_path / "data"
    assert run(["ingest", "--corpus", corpus, "--data-dir", data]) == 0
    return data


def test_cli_ingest_then_ask_roundtrip(tmp_path, corpus, capsys):
    data = ingest_live(tmp_path, corpus)
    out = capsys.readouterr().out
    n = len(load_chunks(data / "chunks.jsonl"))
    assert out.strip() == f"ingested 3 documents, {n} chunks, dim 16"
    assert run(["ask", "How does the alpha module sort lists?", "--data-dir", data]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Answer. [")
    assert any(line.startswith("  cited ") for line in lines)
    assert lines[-2:] == ["refused: false", "citations valid: true"]


def test_cli_ingest_record_then_replay_gives_identical_index_files(tmp_path, corpus):
    cache = tmp_path / "emb.npz"
    a, b = tmp_path / "a", tmp_path / "b"
    assert run(["ingest", "--corpus", corpus, "--data-dir", a, "--record-embeddings", cache]) == 0
    assert cache.exists()
    code = main(
        ["ingest", "--corpus", str(corpus), "--data-dir", str(b), "--replay-embeddings", str(cache)],
        env={"CHUNK_SIZE": "12", "CHUNK_OVERLAP": "2"},
    )
    assert code == 0
    assert (a / "chunks.jsonl").read_bytes() == (b / "chunks.jsonl").read_bytes()
    ia, ib = DenseIndex.load(a / "dense.npz"), DenseIndex.load(b / "dense.npz")
    probe = HashEmbedder().embed(["alpha module"], kind="query")[0]
    assert ia.search(probe, 5) == ib.search(probe, 5)
    assert np.array_equal(ia._matrix, ib._matrix)


def eval_args(tmp_path, corpus, golden, data, *extra):
    return [
        "eval",
        "--golden",
        golden,
        "--corpus",
        corpus,
        "--data-dir",
        data,
        "--report-out",
        tmp_path / "out" / "report.md",
        "--metrics-out",
        tmp_path / "out" / "metrics.json",
        *extra,
    ]


def test_cli_eval_writes_metrics_and_report(tmp_path, corpus, golden, capsys):
    data = ingest_live(tmp_path, corpus)
    capsys.readouterr()
    assert run(eval_args(tmp_path, corpus, golden, data, "--date", "2026-01-02")) == 0
    printed = capsys.readouterr().out.splitlines()
    assert {line.split(":")[0] for line in printed} == METRICS
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == METRICS
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Evaluation report")
    assert "fake-llm" in report and "2026-01-02" in report and "unspecified" in report


def record_everything(tmp_path, corpus, golden):
    emb, llm = tmp_path / "emb.npz", tmp_path / "llm.jsonl"
    data = tmp_path / "data"
    assert run(["ingest", "--corpus", corpus, "--data-dir", data, "--record-embeddings", emb]) == 0
    args = eval_args(
        tmp_path, corpus, golden, data, "--record-llm", llm, "--record-embeddings", emb
    )
    assert run(args) == 0
    return data, emb, llm


def test_cli_eval_in_replay_mode_never_opens_a_network_connection(
    tmp_path, corpus, golden, monkeypatch, capsys
):
    data, emb, llm = record_everything(tmp_path, corpus, golden)
    recorded = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    capsys.readouterr()

    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    args = eval_args(
        tmp_path, corpus, golden, data, "--replay-llm", llm, "--replay-embeddings", emb
    )
    assert main([str(a) for a in args], env={}) == 0
    replayed = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert replayed == recorded


def test_cli_eval_replay_miss_returns_1_with_message(tmp_path, corpus, golden, capsys):
    data, emb, llm = record_everything(tmp_path, corpus, golden)
    other = tmp_path / "golden2.jsonl"
    other.write_text(
        json.dumps(
            {
                "id": "z",
                "question": "A brand new alpha question never recorded",
                "expected_sources": ["alpha"],
                "answerable": True,
                "kind": "single",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    capsys.readouterr()
    args = eval_args(tmp_path, corpus, other, data, "--replay-llm", llm, "--replay-embeddings", emb)
    assert main([str(a) for a in args], env={}) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "no recorded" in err


def test_cli_gate_exit_codes_zero_and_one(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "model": "m",
                "date": "2026-01-01",
                "reason": "test",
                "metrics": {"mrr": {"value": 0.8, "tolerance": 0.05, "direction": "higher"}},
            }
        ),
        encoding="utf-8",
    )
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps({"mrr": 0.8}), encoding="utf-8")
    bad.write_text(json.dumps({"mrr": 0.5}), encoding="utf-8")
    base = ["gate", "--baseline", baseline, "--metrics"]
    assert run([*base, good]) == 0
    assert run([*base, bad]) == 1
    assert run([*base, tmp_path / "missing.json"]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_conflicting_record_and_replay_flags_exit_2(tmp_path, corpus):
    with pytest.raises(SystemExit) as exc:
        run(
            [
                "ingest",
                "--corpus",
                corpus,
                "--record-embeddings",
                tmp_path / "a",
                "--replay-embeddings",
                tmp_path / "b",
            ]
        )
    assert exc.value.code == 2


def test_cli_missing_index_returns_1_with_remedy(tmp_path, capsys):
    assert run(["ask", "anything?", "--data-dir", tmp_path / "none"]) == 1
    err = capsys.readouterr().err
    assert err == f'error: no index in {tmp_path / "none"}; run "evalrag ingest" first\n'


def test_cli_eval_missing_golden_returns_1(tmp_path, corpus, capsys):
    data = ingest_live(tmp_path, corpus)
    args = eval_args(tmp_path, corpus, tmp_path / "nope.jsonl", data)
    assert run(args) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_replay_modes_need_no_endpoint_env(tmp_path, corpus, golden):
    data, emb, llm = record_everything(tmp_path, corpus, golden)
    ask = ["ask", "How does the alpha module sort lists?", "--data-dir", str(data)]
    assert main([*ask, "--replay-llm", str(llm), "--replay-embeddings", str(emb)], env={}) == 0
    args = eval_args(
        tmp_path, corpus, golden, data, "--replay-llm", llm, "--replay-embeddings", emb
    )
    # empty env means the default chunk settings, which do not matter once the index exists
    assert main([str(a) for a in args], env={}) == 0


def test_refusal_text_is_what_the_fake_uses():
    assert ScriptedLLM().complete(
        [{"role": "user", "content": "[a#000]\nx\n\nQuestion: unanswerable?"}]
    ).text == REFUSAL_TEXT


def test_cli_ask_record_llm_with_different_params_returns_1(tmp_path, corpus, capsys):
    data = ingest_live(tmp_path, corpus)
    cache = tmp_path / "llm.jsonl"
    q = ["ask", "How does the alpha module sort lists?", "--data-dir", data, "--record-llm", cache]
    assert run(q) == 0

    class Other(FakeFactory):
        def live_llm(self, settings):
            llm = ScriptedLLM()
            llm.params = {**llm.params, "temperature": 0.9}
            return llm

    capsys.readouterr()
    assert run(q, factory=Other()) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_cli_malformed_chunks_line_returns_1_naming_line(tmp_path, corpus, capsys):
    data = ingest_live(tmp_path, corpus)
    path = data / "chunks.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8")
    n = len(path.read_text(encoding="utf-8").splitlines())
    capsys.readouterr()
    assert run(["ask", "alpha?", "--data-dir", data]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and f"line {n}" in err


def test_cli_corrupt_dense_index_returns_1(tmp_path, corpus, capsys):
    data = ingest_live(tmp_path, corpus)
    (data / "dense.npz").write_bytes(b"garbage")
    capsys.readouterr()
    assert run(["ask", "alpha?", "--data-dir", data]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_cli_ask_and_eval_release_the_sqlite_handle(tmp_path, corpus, golden):
    data = ingest_live(tmp_path, corpus)
    assert run(["ask", "How does the alpha module sort lists?", "--data-dir", data]) == 0
    (data / "keyword.sqlite").unlink()
    data = ingest_live(tmp_path, corpus)
    assert run(eval_args(tmp_path, corpus, golden, data)) == 0
    (data / "keyword.sqlite").unlink()
    assert run(["ingest", "--corpus", corpus, "--data-dir", data]) == 0


def test_cli_closes_connection_when_eval_raises(tmp_path, corpus, golden):
    data = ingest_live(tmp_path, corpus)

    class Boom(ScriptedLLM):
        def complete(self, messages):
            raise RuntimeError("boom")

    class BoomFactory(FakeFactory):
        def live_llm(self, settings):
            return Boom()

    with pytest.raises(RuntimeError):
        run(eval_args(tmp_path, corpus, golden, data), factory=BoomFactory())
    (data / "keyword.sqlite").unlink()


@pytest.mark.parametrize("bad", ["0", "-3", "x"])
def test_cli_eval_k_must_be_positive(tmp_path, corpus, golden, bad):
    args = eval_args(tmp_path, corpus, golden, tmp_path / "d", "--k", bad)
    with pytest.raises(SystemExit) as exc:
        run(args)
    assert exc.value.code == 2


def refresh_args(tmp_path, corpus, golden, data, *extra):
    return [
        "refresh-recordings",
        "--golden",
        golden,
        "--corpus",
        corpus,
        "--data-dir",
        data,
        "--embeddings-out",
        tmp_path / "rec" / "emb.npz",
        "--llm-out",
        tmp_path / "rec" / "llm.jsonl",
        "--report-out",
        tmp_path / "out" / "live.md",
        "--metrics-out",
        tmp_path / "out" / "live-metrics.json",
        "--date",
        "2026-01-02",
        *extra,
    ]


def test_refresh_refuses_existing_outputs_without_overwrite(tmp_path, corpus, golden, capsys):
    (tmp_path / "rec").mkdir()
    emb, llm = tmp_path / "rec" / "emb.npz", tmp_path / "rec" / "llm.jsonl"
    data = tmp_path / "d"
    llm.write_text("keep-llm", encoding="utf-8")
    capsys.readouterr()
    assert run(refresh_args(tmp_path, corpus, golden, data)) == 1
    err = capsys.readouterr().err
    assert err == f"error: {llm} already exists; pass --overwrite to start a fresh recording\n"
    assert llm.read_text(encoding="utf-8") == "keep-llm"
    assert not emb.exists() and not data.exists()
    # both exist: the embeddings path is named first
    emb.write_bytes(b"keep-emb")
    assert run(refresh_args(tmp_path, corpus, golden, data)) == 1
    assert str(emb) in capsys.readouterr().err
    assert emb.read_bytes() == b"keep-emb" and llm.read_text(encoding="utf-8") == "keep-llm"
    assert not data.exists() and not (tmp_path / "out").exists()


def test_refresh_overwrite_starts_fresh_and_drops_stale_entries(tmp_path, corpus, golden):
    data = ingest_live(tmp_path, corpus)
    llm = tmp_path / "rec" / "llm.jsonl"
    stale = ["ask", "A stale question outside the golden set", "--data-dir", data]
    assert run([*stale, "--record-llm", llm]) == 0
    stale_entry = llm.read_text(encoding="utf-8").splitlines()[1]
    assert run(refresh_args(tmp_path, corpus, golden, data, "--overwrite")) == 0
    lines = llm.read_text(encoding="utf-8").splitlines()
    assert stale_entry not in lines
    assert len(lines) - 1 == 3  # one completion per golden question


def test_refresh_records_documents_and_queries_in_one_embedding_cache(
    tmp_path, corpus, golden, capsys
):
    data = tmp_path / "data"
    capsys.readouterr()
    assert run(refresh_args(tmp_path, corpus, golden, data)) == 0
    out = capsys.readouterr().out.splitlines()
    chunks = load_chunks(data / "chunks.jsonl")
    questions = [json.loads(line)["question"] for line in golden.read_text().splitlines()]
    expected = {cache_key("document", c.text) for c in chunks} | {
        cache_key("query", q) for q in questions
    }
    with np.load(tmp_path / "rec" / "emb.npz") as cache:
        keys = set(cache["keys"].tolist())
    assert keys == expected
    # identical chunk texts share one cache entry (the key is kind + text)
    assert len(keys) == len({c.text for c in chunks}) + len(set(questions))
    assert out[0].startswith("ingested 3 documents")
    assert {line.split(":")[0] for line in out[1:10]} == METRICS
    assert out[-1] == f"recorded {len(keys)} embeddings and 3 completions"


def test_refresh_outputs_replay_to_identical_metrics(tmp_path, corpus, golden, monkeypatch):
    assert run(refresh_args(tmp_path, corpus, golden, tmp_path / "data")) == 0
    written = (tmp_path / "out" / "live-metrics.json").read_bytes()
    emb, llm = tmp_path / "rec" / "emb.npz", tmp_path / "rec" / "llm.jsonl"

    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    env = {"CHUNK_SIZE": "12", "CHUNK_OVERLAP": "2"}
    replay_data = tmp_path / "replay-data"
    ingest = ["ingest", "--corpus", corpus, "--data-dir", replay_data, "--replay-embeddings", emb]
    assert main([str(a) for a in ingest], env=env) == 0
    args = eval_args(
        tmp_path, corpus, golden, replay_data, "--replay-llm", llm, "--replay-embeddings", emb
    )
    assert main([str(a) for a in args], env=env) == 0
    assert (tmp_path / "out" / "metrics.json").read_bytes() == written


def test_refresh_writes_report_and_metrics_to_given_paths(tmp_path, corpus, golden):
    assert run(refresh_args(tmp_path, corpus, golden, tmp_path / "data")) == 0
    metrics = json.loads((tmp_path / "out" / "live-metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == METRICS
    report = (tmp_path / "out" / "live.md").read_text(encoding="utf-8")
    assert report.startswith("# Evaluation report") and "2026-01-02" in report


def test_refresh_requires_endpoints_in_env(tmp_path, corpus, golden, capsys):
    capsys.readouterr()
    assert run(refresh_args(tmp_path, corpus, golden, tmp_path / "data"), env={}) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "LLM_BASE_URL" in err
    assert not (tmp_path / "rec").exists()


def test_refresh_defaults_for_report_and_metrics_paths():
    args = _build_parser().parse_args(
        ["refresh-recordings", "--golden", "g", "--embeddings-out", "e", "--llm-out", "l"]
    )
    assert args.report_out == "evals/reports/live.md"
    assert args.metrics_out == "evals/reports/live-metrics.json"
    assert args.overwrite is False


class CountingFactory(FakeFactory):
    def __init__(self):
        self.embedder_calls = 0
        self.llm_calls = 0

    def embedder(self, settings, record, replay):
        self.embedder_calls += 1
        return super().embedder(settings, record, replay)

    def llm(self, settings, record, replay):
        self.llm_calls += 1
        return super().llm(settings, record, replay)


def _old_caches(tmp_path):
    (tmp_path / "rec").mkdir(exist_ok=True)
    emb, llm = tmp_path / "rec" / "emb.npz", tmp_path / "rec" / "llm.jsonl"
    emb.write_bytes(b"old-emb")
    llm.write_text("old-llm", encoding="utf-8")
    return emb, llm


def _assert_untouched(tmp_path, emb, llm):
    assert emb.read_bytes() == b"old-emb" and llm.read_text(encoding="utf-8") == "old-llm"
    assert list(tmp_path.rglob("*.refresh-tmp")) == []


def test_refresh_overwrite_with_missing_golden_keeps_old_recordings(
    tmp_path, corpus, capsys
):
    emb, llm = _old_caches(tmp_path)
    args = refresh_args(tmp_path, corpus, tmp_path / "nope.jsonl", tmp_path / "d", "--overwrite")
    assert run(args) == 1
    assert capsys.readouterr().err.startswith("error: ")
    _assert_untouched(tmp_path, emb, llm)


def test_refresh_overwrite_with_failing_llm_keeps_old_recordings(tmp_path, corpus, golden):
    emb, llm = _old_caches(tmp_path)

    class Failing(ScriptedLLM):
        def complete(self, messages):
            raise LLMError("server stopped")

    class FailingFactory(FakeFactory):
        def live_llm(self, settings):
            return Failing()

    args = refresh_args(tmp_path, corpus, golden, tmp_path / "d", "--overwrite")
    assert run(args, factory=FailingFactory()) == 1
    _assert_untouched(tmp_path, emb, llm)


def test_refresh_overwrite_success_replaces_finals_and_leaves_no_temp_files(
    tmp_path, corpus, golden
):
    emb, llm = _old_caches(tmp_path)
    factory = CountingFactory()
    assert run(refresh_args(tmp_path, corpus, golden, tmp_path / "d", "--overwrite"), factory=factory) == 0
    assert factory.embedder_calls == 1 and factory.llm_calls == 1
    assert b"old-emb" != emb.read_bytes() and "old-llm" not in llm.read_text(encoding="utf-8")
    assert len(llm.read_text(encoding="utf-8").splitlines()) - 1 == 3
    assert list(tmp_path.rglob("*.refresh-tmp")) == []


def test_refresh_refusal_leaves_everything_untouched(tmp_path, corpus, golden):
    emb, llm = _old_caches(tmp_path)
    assert run(refresh_args(tmp_path, corpus, golden, tmp_path / "d")) == 1
    _assert_untouched(tmp_path, emb, llm)
    assert not (tmp_path / "d").exists()


class SourceCountingLLM(ScriptedLLM):
    """Records how many sources each real (final) user message carried."""

    def __init__(self) -> None:
        super().__init__()
        self.source_counts: list[int] = []

    def complete(self, messages):
        self.source_counts.append(messages[-1]["content"].count("Source id: "))
        return super().complete(messages)


class SourceCountingFactory(FakeFactory):
    def __init__(self) -> None:
        self.llm_instance = SourceCountingLLM()

    def live_llm(self, settings):
        return self.llm_instance


@pytest.mark.parametrize("command", ["eval", "refresh-recordings"])
def test_cli_k_has_no_hard_coded_default(command):
    required = {
        "eval": ["--golden", "g", "--report-out", "r", "--metrics-out", "m"],
        "refresh-recordings": ["--golden", "g", "--embeddings-out", "e", "--llm-out", "l"],
    }[command]
    assert _build_parser().parse_args([command, *required]).k is None


@pytest.mark.parametrize(
    ("env_extra", "argv_extra", "expected_k"),
    [
        ({}, [], 3),  # Settings default top_k
        ({"TOP_K": "2"}, [], 2),  # TOP_K setting
        ({"TOP_K": "2"}, ["--k", "1"], 1),  # explicit --k wins
    ],
)
def test_cli_eval_k_defaults_to_top_k_setting_for_pipeline_and_metrics(
    tmp_path, corpus, golden, env_extra, argv_extra, expected_k
):
    data = ingest_live(tmp_path, corpus)
    factory = SourceCountingFactory()
    args = eval_args(tmp_path, corpus, golden, data, *argv_extra)
    assert run(args, env={**ENV, **env_extra}, factory=factory) == 0
    assert factory.llm_instance.source_counts
    assert set(factory.llm_instance.source_counts) == {expected_k}
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert f"| k | {expected_k} |" in report


def test_refresh_k_defaults_to_top_k_setting(tmp_path, corpus, golden):
    factory = SourceCountingFactory()
    args = refresh_args(tmp_path, corpus, golden, tmp_path / "data")
    assert run(args, env={**ENV, "TOP_K": "2"}, factory=factory) == 0
    assert set(factory.llm_instance.source_counts) == {2}
    report = (tmp_path / "out" / "live.md").read_text(encoding="utf-8")
    assert "| k | 2 |" in report
