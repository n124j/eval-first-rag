"""Acceptance tests: the retrieval ablation is reproducible and consistent with the harness."""
from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pytest

from evalrag.evals.gate import load_metrics

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "evals" / "analysis" / "retrieval_ablation.py"
REPORT = ROOT / "evals" / "reports" / "retrieval-ablation.md"
GUARD_ON_METRICS = ROOT / "evals" / "reports" / "golden-guard-on-metrics.json"
GUARD_OFF_METRICS = ROOT / "evals" / "reports" / "golden-guard-off-metrics.json"
RETRIEVERS = ("keyword", "dense", "hybrid")


@pytest.fixture(scope="module")
def ablation():
    calls: list[str] = []

    def refuse(*args, **kwargs):
        calls.append("network")
        raise AssertionError("network access attempted during an offline ablation")

    mp = pytest.MonkeyPatch()
    mp.setattr(socket, "socket", refuse)
    mp.setattr(socket, "getaddrinfo", refuse)
    mp.setattr(socket, "create_connection", refuse)
    try:
        spec = importlib.util.spec_from_file_location("retrieval_ablation", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules["retrieval_ablation"] = module
        spec.loader.exec_module(module)
        analysis = module.analyse()
        text = module.render(analysis)
    finally:
        mp.undo()
    return module, analysis, text, calls


def test_no_network_call_is_attempted(ablation):
    *_, calls = ablation
    assert calls == []


def test_output_equals_the_committed_markdown_byte_for_byte(ablation):
    _, _, text, _ = ablation
    assert text.encode("utf-8") == REPORT.read_bytes()


def test_all_three_retrievers_are_scored_on_the_50_answerable_items(ablation):
    _, analysis, _, _ = ablation
    for name in RETRIEVERS:
        per_item = analysis["retrievers"][name]["items"]
        assert len(per_item) == 50
        assert all(i.startswith("g") for i in per_item)


@pytest.mark.parametrize("metrics_path", [GUARD_ON_METRICS, GUARD_OFF_METRICS])
def test_hybrid_row_matches_the_harness_recall_and_mrr(ablation, metrics_path):
    # The harness averages recall_at_k and mrr over the answerable items only, at k=3,
    # with MRR computed over the top 3 retrieved chunks the pipeline sees.
    _, analysis, _, _ = ablation
    committed = load_metrics(metrics_path)
    hybrid = analysis["retrievers"]["hybrid"]
    assert round(hybrid["recall@3"], 4) == round(committed["recall_at_k"], 4)
    assert round(hybrid["mrr"], 4) == round(committed["mrr"], 4)


def test_recall_is_monotone_in_k(ablation):
    _, analysis, _, _ = ablation
    for name in RETRIEVERS:
        row = analysis["retrievers"][name]
        assert row["recall@1"] <= row["recall@3"] <= row["recall@5"]
