"""Acceptance tests: the first live measurement (v0 prompt, 1B model) is kept as evidence.

The evidence is a measured failure, not a CI fixture: it must live under evals/reports/
and never under evals/recordings/, which is reserved for the final replayable fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

from evalrag.evals.gate import load_metrics

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "evals" / "reports"
REPORT = REPORTS / "v0-prompt-1b-report.md"
METRICS = REPORTS / "v0-prompt-1b-metrics.json"
COMPLETIONS = REPORTS / "v0-prompt-1b-completions.jsonl"
README = REPORTS / "README.md"
MASQUERADE = ROOT / "evals" / "recordings" / "llm.jsonl"

METRIC_KEYS = {
    "recall_at_k",
    "mrr",
    "citation_validity_rate",
    "correct_refusal_rate",
    "latency_p50_s",
    "latency_p95_s",
    "prompt_tokens_total",
    "completion_tokens_total",
}
V0_REFUSAL = "Not found in the provided documents."


def test_the_three_v0_files_exist():
    for path in (REPORT, METRICS, COMPLETIONS):
        assert path.is_file(), path


def test_v0_metrics_parse_and_show_the_failure():
    metrics = load_metrics(METRICS)
    assert set(metrics) == METRIC_KEYS
    assert metrics["citation_validity_rate"] == 0.0
    assert metrics["recall_at_k"] == 0.9


def test_v0_completions_are_60_json_lines_with_text():
    # The first line is the cache's meta header (model, temperature, ...); 60 completions follow.
    lines = COMPLETIONS.read_text(encoding="utf-8").splitlines()
    assert set(json.loads(lines[0])) == {"meta"}
    records = [json.loads(line) for line in lines[1:]]
    assert len(records) == 60
    for record in records:
        assert isinstance(record["completion"]["text"], str)


def test_readme_states_the_finding_and_the_refusal_text():
    text = README.read_text(encoding="utf-8")
    assert "0 of 60" in text
    assert V0_REFUSAL in text


def test_v0_evidence_is_not_masquerading_as_a_fixture():
    assert not MASQUERADE.exists()
