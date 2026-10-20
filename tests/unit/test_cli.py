import json
import socket
from pathlib import Path

import numpy as np
import pytest

from evalrag.cli import ComponentFactory, main
from evalrag.generation.answer import REFUSAL_TEXT
from evalrag.ingest import load_chunks
from evalrag.retrieval.dense import DenseIndex
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
    assert np.allclose(ia._matrix, ib._matrix, atol=1e-2)


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
