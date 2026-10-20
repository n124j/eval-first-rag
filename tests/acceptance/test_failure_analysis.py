"""Acceptance tests: the golden failure analysis is reproducible from the committed recordings."""
from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pytest

from evalrag.evals.gate import load_metrics

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "evals" / "analysis" / "failure_taxonomy.py"
REPORT = ROOT / "evals" / "reports" / "golden-failure-analysis.md"
GUARD_ON_METRICS = ROOT / "evals" / "reports" / "golden-guard-on-metrics.json"


@pytest.fixture(scope="module")
def taxonomy():
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted during an offline analysis")

    mp = pytest.MonkeyPatch()
    mp.setattr(socket, "socket", refuse)
    mp.setattr(socket, "getaddrinfo", refuse)
    spec = importlib.util.spec_from_file_location("failure_taxonomy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["failure_taxonomy"] = module
    spec.loader.exec_module(module)
    analysis = module.analyse()
    text = module.render(analysis)
    mp.undo()
    return module, analysis, text


def test_output_equals_the_committed_markdown_byte_for_byte(taxonomy):
    _, _, text = taxonomy
    assert text.encode("utf-8") == REPORT.read_bytes()


def test_answerable_class_counts_sum_to_50_with_no_item_in_two_classes(taxonomy):
    _, analysis, _ = taxonomy
    ids = [i for members in analysis["classes"].values() for i in members]
    assert len(ids) == 50
    assert len(set(ids)) == 50


def test_valid_class_matches_the_committed_citation_validity_rate(taxonomy):
    module, analysis, _ = taxonomy
    rate = load_metrics(GUARD_ON_METRICS)["citation_validity_rate"]
    assert len(analysis["classes"][module.VALID]) == 15
    assert len(analysis["classes"][module.VALID]) / 50 == rate


def test_unanswerable_split_matches_the_committed_correct_refusal_rate(taxonomy):
    _, analysis, _ = taxonomy
    split = analysis["unanswerable"]
    total = sum(len(v) for v in split.values())
    assert total == 10
    refused = len(split["guard"]) + len(split["model"])
    assert refused / total == load_metrics(GUARD_ON_METRICS)["correct_refusal_rate"]
