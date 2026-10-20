"""Run a system under test over golden items and aggregate metrics."""

from collections.abc import Callable
from dataclasses import dataclass

from evalrag.evals.dataset import GoldenItem, source_of
from evalrag.evals.metrics import mrr, percentile, recall_at_k

NAN = float("nan")


@dataclass(frozen=True)
class SutResult:
    answer: str
    retrieved_ids: list[str]
    cited_ids: list[str]
    citation_ok: bool
    refused: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


SystemUnderTest = Callable[[str], SutResult]


@dataclass
class ItemResult:
    """Per-item scores; None where a score does not apply.

    `citation_ok` is set for answerable items only: True iff the item was answered (not
    refused) with valid citations. `correct_refusal` is set for unanswerable items only.
    """

    item: GoldenItem
    sut: SutResult
    recall: float | None
    rr: float | None
    citation_ok: bool | None
    correct_refusal: bool | None
    failures: list[str]


@dataclass
class EvalResult:
    items: list[ItemResult]
    metrics: dict[str, float]
    k: int


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else NAN


def _rate(hits: int, total: int) -> float:
    return hits / total if total else NAN


def _score(item: GoldenItem, sut: SutResult, k: int) -> ItemResult:
    failures: list[str] = []
    if not item.answerable:
        if not sut.refused:
            failures.append("answered unanswerable question")
        return ItemResult(item, sut, None, None, None, sut.refused, failures)

    sources = [source_of(c) for c in sut.retrieved_ids]
    expected = set(item.expected_sources)
    recall = recall_at_k(expected, sources, k)
    rr = mrr(expected, sources)
    valid = (not sut.refused) and sut.citation_ok
    if recall == 0.0:
        failures.append("recall 0")
    if sut.refused:
        failures.append("refused answerable question")
    elif not sut.citation_ok:
        failures.append("invalid citations")
    return ItemResult(item, sut, recall, rr, valid, None, failures)


def run_eval(sut: SystemUnderTest, items: list[GoldenItem], k: int = 5) -> EvalResult:
    """Call `sut` once per item, in order, and aggregate. Exceptions from `sut` propagate."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not items:
        raise ValueError("items must not be empty")

    rows = [_score(item, sut(item.question), k) for item in items]
    answerable = [r for r in rows if r.item.answerable]
    unanswerable = [r for r in rows if not r.item.answerable]
    latencies = [r.sut.latency_s for r in rows]

    metrics = {
        "recall_at_k": _mean([r.recall for r in answerable if r.recall is not None]),
        "mrr": _mean([r.rr for r in answerable if r.rr is not None]),
        "citation_validity_rate": _rate(
            sum(bool(r.citation_ok) for r in answerable), len(answerable)
        ),
        "correct_refusal_rate": _rate(
            sum(bool(r.correct_refusal) for r in unanswerable), len(unanswerable)
        ),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "prompt_tokens_total": float(sum(r.sut.prompt_tokens for r in rows)),
        "completion_tokens_total": float(sum(r.sut.completion_tokens for r in rows)),
    }
    return EvalResult(rows, metrics, k)
