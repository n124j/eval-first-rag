"""Acceptance tests: the README is the project's front door and must match the evidence.

Display convention checked here (the README's results table must follow it):
- rates (recall_at_k, mrr, citation_validity_rate, correct_refusal_rate, over_refusal_rate)
  are shown with 4 decimals (`f"{value:.4f}"`);
- latencies (latency_p50_s, latency_p95_s) with 3 decimals (`f"{value:.3f}"`);
- token totals as integers (`f"{value:.0f}"`).
Each displayed value must equal the committed guard-on metric formatted that way.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
ADR_DIR = ROOT / "docs" / "adr"
GUARD_ON_METRICS = ROOT / "evals" / "reports" / "golden-guard-on-metrics.json"
NEW_ADRS = ("0002-hybrid-retrieval.md", "0003-no-rag-framework.md",
            "0004-recorded-responses-in-ci.md")
FORMATS = {
    "recall_at_k": ".4f",
    "mrr": ".4f",
    "citation_validity_rate": ".4f",
    "correct_refusal_rate": ".4f",
    "over_refusal_rate": ".4f",
    "latency_p50_s": ".3f",
    "latency_p95_s": ".3f",
    "prompt_tokens_total": ".0f",
    "completion_tokens_total": ".0f",
}
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


@pytest.fixture(scope="module")
def text() -> str:
    assert README.is_file(), "README.md is missing"
    return README.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n{heading}\n")
    end = text.find("\n## ", start + len(heading) + 2)
    return text[start : end if end != -1 else len(text)]


def _results_table(text: str) -> tuple[list[str], dict[str, list[str]]]:
    rows = [line for line in _section(text, "## Results").splitlines() if line.startswith("|")]
    assert len(rows) >= 3, "no Markdown table under ## Results"
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], [c for c in cells[2:] if c]
    return header, {row[0].strip("`* "): row[1:] for row in body}


@pytest.mark.parametrize(
    "heading", ["## Results", "## Quick start", "## Known limitations", "## Architecture"]
)
def test_required_headings(text, heading):
    assert f"\n{heading}\n" in text


def test_results_table_has_the_nine_metrics_with_the_guard_on_values(text):
    header, table = _results_table(text)
    assert "on" in header[1].lower(), f"second column must be the guard-on result: {header}"
    metrics = json.loads(GUARD_ON_METRICS.read_text(encoding="utf-8"))
    assert set(FORMATS) == set(metrics)
    for name, fmt in FORMATS.items():
        assert name in table, f"metric {name} missing from the results table"
        assert table[name][0] == format(metrics[name], fmt), name


@pytest.mark.parametrize(
    "needle",
    ["qwen2.5-7b-instruct-1m", "Ryzen 9 3900X", "2026-10-20", "held-out", "did not transfer",
     "0.30", "stop floor", "docker compose up"],
)
def test_readme_states(text, needle):
    assert needle in text



def test_every_relative_link_points_to_an_existing_file(text):
    targets = LINK.findall(text)
    assert targets, "README has no links"
    for target in targets:
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        path = ROOT / target.split("#", 1)[0]
        assert path.exists(), f"broken link: {target}"


@pytest.mark.parametrize("name", NEW_ADRS)
def test_new_adrs_exist_and_list_rejected_alternatives(name):
    path = ADR_DIR / name
    assert path.is_file(), f"{name} missing"
    assert "Rejected" in path.read_text(encoding="utf-8")


def test_quick_start_has_the_offline_gate(text):
    quick = _section(text, "## Quick start")
    assert "pip install -e . --no-deps" in quick
    assert "evalrag gate" in quick


def test_links_the_docker_readme(text):
    assert "docker/README.md" in LINK.findall(text)


def test_has_a_mermaid_diagram(text):
    assert "```mermaid\n" in text
