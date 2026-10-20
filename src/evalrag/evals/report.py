"""Deterministic Markdown report for an `EvalResult`."""

import math
from collections.abc import Mapping

from evalrag.evals.dataset import source_of
from evalrag.evals.judge import JudgeScore
from evalrag.evals.runner import EvalResult, ItemResult

REQUIRED_META = ("model", "hardware", "date")
_RATIO = ("recall_at_k", "mrr", "citation_validity_rate", "correct_refusal_rate")
_LATENCY = ("latency_p50_s", "latency_p95_s")
_TOKENS = ("prompt_tokens_total", "completion_tokens_total")
METRIC_ORDER = _RATIO + _LATENCY + _TOKENS
_SOURCES_SHOWN = 3


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(header: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return lines


def _format_metric(name: str, value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if name in _RATIO:
        return f"{value:.4f}"
    if name in _LATENCY:
        return f"{value:.3f}"
    return str(round(value))


def _failure_line(row: ItemResult) -> str:
    sources: list[str] = []
    for chunk_id in row.sut.retrieved_ids:
        source = source_of(chunk_id)
        if source not in sources:
            sources.append(source)
    shown = ", ".join(sources[:_SOURCES_SHOWN]) or "none"
    cited = ", ".join(row.sut.cited_ids) or "none"
    return (
        f"- {row.item.id} ({row.item.kind}): {row.item.question} | "
        f"failures: {', '.join(row.failures)} | retrieved sources: {shown} | cited: {cited}"
    )


def render_markdown(
    result: EvalResult,
    meta: Mapping[str, str],
    judge_scores: Mapping[str, JudgeScore] | None = None,
) -> str:
    """Render the report. Same inputs always give the same text."""
    required = list(REQUIRED_META) + (["judge_model"] if judge_scores is not None else [])
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"meta is missing required keys: {missing}")

    meta_rows: list[tuple[object, ...]] = [(key, meta[key]) for key in REQUIRED_META]
    meta_rows.extend((key, meta[key]) for key in sorted(set(meta) - set(REQUIRED_META)))
    meta_rows.append(("k", result.k))
    meta_rows.append(("items", len(result.items)))

    metric_rows = [(name, _format_metric(name, result.metrics[name])) for name in METRIC_ORDER]
    failing = [row for row in result.items if row.failures]

    lines = ["# Evaluation report", ""]
    lines += _table(("Field", "Value"), meta_rows)
    lines += ["", "## Metrics", ""]
    lines += _table(("Metric", "Value"), metric_rows)
    lines += ["", "## Failures", ""]
    lines += [_failure_line(row) for row in failing] if failing else ["No failures."]

    if judge_scores is not None:
        known = {row.item.id for row in result.items}
        unknown = sorted(set(judge_scores) - known)
        if unknown:
            raise ValueError(f"judge_scores has unknown item ids: {unknown}")
        score_rows = [
            (
                row.item.id,
                judge_scores[row.item.id].faithfulness,
                judge_scores[row.item.id].relevance,
            )
            for row in result.items
            if row.item.id in judge_scores
        ]
        lines += ["", "## LLM-judged scores (noisy)", ""]
        lines.append(
            f"These scores were produced by `{meta['judge_model']}`, are noisy, "
            "and are not used by the gate."
        )
        lines.append("")
        lines += _table(("Item", "Faithfulness", "Relevance"), score_rows)
    return "\n".join(lines) + "\n"
