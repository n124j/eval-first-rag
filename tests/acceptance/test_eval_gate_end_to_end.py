"""Acceptance tests: committed recordings, baseline and gate work end to end, offline."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import socket
from pathlib import Path

import pytest

from evalrag.cli import ComponentFactory, main
from evalrag.evals.dataset import load_golden
from evalrag.evals.gate import compare, load_baseline, load_metrics
from evalrag.generation.llm import ReplayClient, request_key
from evalrag.retrieval.embedder import ReplayEmbedder, cache_key

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus"
GOLDEN = ROOT / "evals" / "golden" / "golden.jsonl"
EMBEDDINGS = ROOT / "evals" / "recordings" / "embeddings.npz"
LLM = ROOT / "evals" / "recordings" / "llm.jsonl"
BASELINE = ROOT / "evals" / "baseline.json"
GUARD_ON_METRICS = ROOT / "evals" / "reports" / "golden-guard-on-metrics.json"
GUARD_OFF_METRICS = ROOT / "evals" / "reports" / "golden-guard-off-metrics.json"
MODEL = "qwen2.5-7b-instruct-1m"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

DIRECTIONS = {
    "recall_at_k": "higher",
    "mrr": "higher",
    "citation_validity_rate": "higher",
    "correct_refusal_rate": "higher",
    "over_refusal_rate": "lower",
    "latency_p50_s": "lower",
    "latency_p95_s": "lower",
    "prompt_tokens_total": "lower",
    "completion_tokens_total": "lower",
}


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted during an offline replay")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _replay(
    tmp_path: Path, factory: ComponentFactory | None = None, extra: list[str] | None = None
) -> Path:
    """Ingest and eval purely from the committed recordings; return the metrics path."""
    data = tmp_path / "data"
    metrics = tmp_path / "metrics.json"
    code = main(
        [
            "ingest",
            "--corpus", str(CORPUS),
            "--data-dir", str(data),
            "--replay-embeddings", str(EMBEDDINGS),
        ],
        env={},
        factory=factory,
    )
    assert code == 0
    code = main(
        [
            "eval",
            "--golden", str(GOLDEN),
            "--corpus", str(CORPUS),
            "--data-dir", str(data),
            "--replay-llm", str(LLM),
            "--replay-embeddings", str(EMBEDDINGS),
            "--report-out", str(tmp_path / "report.md"),
            "--metrics-out", str(metrics),
            "--model", MODEL,
            "--hardware", "test",
            "--date", "2026-10-20",
            *(extra or []),
        ],
        env={},
        factory=factory,
    )
    assert code == 0
    return metrics


def test_baseline_has_all_nine_metrics_with_expected_directions_and_finite_values():
    baseline = load_baseline(BASELINE)
    assert set(baseline.metrics) == set(DIRECTIONS)
    for name, direction in DIRECTIONS.items():
        entry = baseline.metrics[name]
        assert entry.direction == direction, name
        assert math.isfinite(entry.value), name
        assert math.isfinite(entry.tolerance), name


def test_baseline_tolerances_follow_the_stated_rules():
    baseline = load_baseline(BASELINE)
    tol = {name: entry.tolerance for name, entry in baseline.metrics.items()}
    for name in ("recall_at_k", "mrr", "citation_validity_rate"):
        assert tol[name] == 0.02, name
    for name in ("correct_refusal_rate", "over_refusal_rate"):
        assert tol[name] == 0.10, name
    for name in ("latency_p50_s", "latency_p95_s"):
        assert abs(tol[name] - 0.25 * baseline.metrics[name].value) < 0.001, name
    for name in ("prompt_tokens_total", "completion_tokens_total"):
        assert abs(tol[name] - 0.05 * baseline.metrics[name].value) < 0.01, name
    assert baseline.model.strip() and "qwen2.5-7b-instruct-1m" in baseline.model
    assert baseline.date.strip()
    assert "re-record" in baseline.reason


def test_baseline_matches_committed_guard_on_metrics_file():
    baseline = load_baseline(BASELINE)
    committed = load_metrics(GUARD_ON_METRICS)
    assert set(committed) == set(baseline.metrics)
    for name, entry in baseline.metrics.items():
        assert entry.value == committed[name], name
        assert entry.tolerance >= 0, name


def test_recordings_meta_names_the_models():
    header = json.loads(LLM.read_text(encoding="utf-8").splitlines()[0])
    assert header["meta"]["model"] == MODEL
    assert ReplayClient(LLM).params["model"] == MODEL
    assert ReplayEmbedder(EMBEDDINGS).model == EMBED_MODEL


def test_end_to_end_replay_eval_with_default_guard_passes_gate(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metrics = _replay(tmp_path)
    assert metrics.read_bytes() == GUARD_ON_METRICS.read_bytes()
    assert main(["gate", "--metrics", str(metrics), "--baseline", str(BASELINE)], env={}) == 0


def test_gate_fails_when_recall_degrades_beyond_tolerance(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metrics = load_metrics(_replay(tmp_path))
    baseline = load_baseline(BASELINE)
    degraded = dict(metrics)
    degraded["recall_at_k"] = metrics["recall_at_k"] - baseline.metrics["recall_at_k"].tolerance - 0.05
    path = tmp_path / "degraded.json"
    path.write_text(json.dumps(degraded), encoding="utf-8")
    assert main(["gate", "--metrics", str(path), "--baseline", str(BASELINE)], env={}) == 1


def test_replay_with_guard_disabled_reproduces_the_live_run_exactly(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metrics = _replay(tmp_path, extra=["--min-dense-score", "0"])
    assert metrics.read_bytes() == GUARD_OFF_METRICS.read_bytes()


def test_gate_fails_when_citation_validity_degrades_beyond_tolerance(tmp_path):
    metrics = dict(load_metrics(GUARD_ON_METRICS))
    tolerance = load_baseline(BASELINE).metrics["citation_validity_rate"].tolerance
    metrics["citation_validity_rate"] -= tolerance + 0.05
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert main(["gate", "--metrics", str(path), "--baseline", str(BASELINE)], env={}) == 1
    assert [r.name for r in compare(metrics, load_baseline(BASELINE))] == ["citation_validity_rate"]


def test_gate_fails_when_over_refusal_rises_beyond_tolerance(tmp_path):
    metrics = dict(load_metrics(GUARD_ON_METRICS))
    baseline = load_baseline(BASELINE)
    assert baseline.metrics["over_refusal_rate"].direction == "lower"
    metrics["over_refusal_rate"] += baseline.metrics["over_refusal_rate"].tolerance + 0.05
    path = tmp_path / "refuses.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert main(["gate", "--metrics", str(path), "--baseline", str(BASELINE)], env={}) == 1
    assert [r.name for r in compare(metrics, baseline)] == ["over_refusal_rate"]


def test_gate_fails_when_a_metric_is_missing(tmp_path):
    metrics = copy.deepcopy(load_metrics(GUARD_ON_METRICS))
    del metrics["mrr"]
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert main(["gate", "--metrics", str(path), "--baseline", str(BASELINE)], env={}) == 1


def test_gate_fails_when_latency_regresses_beyond_tolerance(tmp_path):
    metrics = dict(load_metrics(GUARD_ON_METRICS))
    tolerance = load_baseline(BASELINE).metrics["latency_p95_s"].tolerance
    metrics["latency_p95_s"] = metrics["latency_p95_s"] + tolerance + 1.0
    path = tmp_path / "slow.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert main(["gate", "--metrics", str(path), "--baseline", str(BASELINE)], env={}) == 1
    assert [r.name for r in compare(metrics, load_baseline(BASELINE))] == ["latency_p95_s"]


class _CountingFactory(ComponentFactory):
    """Default factory, but records which prompts the replay client was asked for."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def llm(self, settings, record, replay):
        inner = super().llm(settings, record, replay)
        keys = self.keys

        class Counting:
            params = inner.params

            def complete(self, messages):
                keys.append(request_key(
                    str(inner.params["model"]), messages, inner.params["temperature"],
                    inner.params["max_tokens"], inner.params.get("seed"),
                ))
                return inner.complete(messages)

        return Counting()


def test_every_golden_question_has_a_recorded_completion(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    factory = _CountingFactory()
    # Guard forced off: every question asks for a completion; a ReplayMiss makes main() fail.
    _replay(tmp_path, factory, extra=["--min-dense-score", "0"])
    items = load_golden(GOLDEN, known_sources=_doc_ids())
    recorded = len(LLM.read_text(encoding="utf-8").splitlines()) - 1  # minus the meta header
    assert len(factory.keys) == len(items) == 60
    assert recorded == 60


def _doc_ids() -> set[str]:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    return {entry["doc_id"] for entry in manifest}


def test_request_key_is_pinned():
    # Changing request_key orphans every committed recording; re-record and re-baseline if so.
    key = request_key("m", [{"role": "user", "content": "hi"}], 0.0, 512, None)
    assert key == "e99dc1f41c5b607d6d0fcd5eebc44a943e97bf78c4c833a9949d19c435f1cd5a"


def test_embedder_cache_key_is_pinned():
    # Changing the embedder cache key orphans every committed embedding recording.
    literal = "0a7bf51fb1cd023e5ebeeb56f0fd53781c5883548e7bbb844d4178dfb7737399"
    assert hashlib.sha256(b"query\nhello").hexdigest() == literal
    assert cache_key("query", "hello") == literal
