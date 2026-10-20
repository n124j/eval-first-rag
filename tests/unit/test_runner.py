import math

import pytest

from evalrag.evals.dataset import GoldenItem
from evalrag.evals.runner import SutResult, run_eval


def gi(i: str, kind: str, sources: tuple[str, ...]) -> GoldenItem:
    return GoldenItem(i, f"question {i}?", sources, kind != "unanswerable", kind)


def res(
    retrieved: list[str],
    *,
    ok: bool = True,
    refused: bool = False,
    latency: float = 1.0,
    pt: int = 0,
    ct: int = 0,
) -> SutResult:
    return SutResult("ans", retrieved, [], ok, refused, latency, pt, ct)


class Scripted:
    def __init__(self, script: dict[str, SutResult]) -> None:
        self.script = script
        self.calls: list[str] = []

    def __call__(self, question: str) -> SutResult:
        self.calls.append(question)
        return self.script[question]


ITEMS = [
    gi("1", "single", ("argparse",)),
    gi("2", "single", ("os.path",)),
    gi("3", "multi", ("a", "b")),
    gi("4", "multi", ("a", "b")),
    gi("5", "unanswerable", ()),
    gi("6", "unanswerable", ()),
]


def scripted() -> Scripted:
    return Scripted(
        {
            "question 1?": res(["argparse#001", "os#002"], latency=1, pt=10, ct=1),
            "question 2?": res(
                ["argparse#001", "re#002", "os.path#003"], ok=False, latency=2, pt=20, ct=2
            ),
            "question 3?": res(["a#001", "x#001", "b#001"], refused=True, latency=3, pt=30, ct=3),
            "question 4?": res(["b#001", "a#002"], latency=4, pt=40, ct=4),
            "question 5?": res([], refused=True, latency=5, pt=50, ct=5),
            "question 6?": res(["x#001"], latency=6, pt=60, ct=6),
        }
    )


def test_run_eval_aggregates_match_hand_computed_values() -> None:
    out = run_eval(scripted(), ITEMS, k=2)
    m = out.metrics
    assert out.k == 2
    # recall: 1, 0 (os.path only at rank 3), 0.5, 1 -> 2.5 / 4
    assert m["recall_at_k"] == 0.625
    # rr: 1, 1/3, 1 (item 3 refused but retrieval still scored), 1 -> (10/3) / 4
    assert m["mrr"] == pytest.approx(10 / 12)
    # valid and not refused: items 1 and 4 out of 4 answerable
    assert m["citation_validity_rate"] == 0.5
    # item 5 refused, item 6 answered
    assert m["correct_refusal_rate"] == 0.5
    # latencies 1..6: nearest rank p50 -> 3rd, p95 -> ceil(5.7) = 6th
    assert m["latency_p50_s"] == 3.0
    assert m["latency_p95_s"] == 6.0
    assert m["prompt_tokens_total"] == 210.0
    assert m["completion_tokens_total"] == 21.0
    assert isinstance(m["prompt_tokens_total"], float)


def test_metrics_keys_are_exact() -> None:
    out = run_eval(scripted(), ITEMS, k=2)
    assert set(out.metrics) == {
        "recall_at_k",
        "mrr",
        "citation_validity_rate",
        "correct_refusal_rate",
        "latency_p50_s",
        "latency_p95_s",
        "prompt_tokens_total",
        "completion_tokens_total",
    }


def test_failures_strings() -> None:
    rows = run_eval(scripted(), ITEMS, k=2).items
    assert rows[0].failures == []
    assert rows[1].failures == ["recall 0", "invalid citations"]
    assert rows[2].failures == ["refused answerable question"]
    assert rows[3].failures == []
    assert rows[4].failures == []
    assert rows[5].failures == ["answered unanswerable question"]
    assert rows[0].recall == 1.0 and rows[0].rr == 1.0 and rows[0].citation_ok is True
    assert rows[1].rr == pytest.approx(1 / 3) and rows[1].citation_ok is False
    assert rows[2].citation_ok is False  # refused answerable: not a valid answer
    assert rows[4].recall is None and rows[4].rr is None and rows[4].citation_ok is None
    assert rows[4].correct_refusal is True and rows[5].correct_refusal is False
    assert rows[0].correct_refusal is None


def test_over_refusal_lowers_citation_validity() -> None:
    items = [gi("1", "single", ("a",)), gi("2", "single", ("a",))]
    sut = Scripted(
        {
            "question 1?": res(["a#001"]),
            "question 2?": res(["a#001"], refused=True),
        }
    )
    assert run_eval(sut, items).metrics["citation_validity_rate"] == 0.5


def test_unanswerable_answered_counts_incorrect() -> None:
    items = [gi("1", "unanswerable", ())]
    sut = Scripted({"question 1?": res(["a#001"], refused=False)})
    out = run_eval(sut, items)
    assert out.metrics["correct_refusal_rate"] == 0.0
    assert out.items[0].correct_refusal is False


def test_empty_denominator_gives_nan() -> None:
    only_answerable = [gi("1", "single", ("a",))]
    m = run_eval(Scripted({"question 1?": res(["a#001"])}), only_answerable).metrics
    assert math.isnan(m["correct_refusal_rate"])
    assert m["recall_at_k"] == 1.0

    only_unanswerable = [gi("1", "unanswerable", ())]
    m = run_eval(Scripted({"question 1?": res([], refused=True)}), only_unanswerable).metrics
    for key in ("recall_at_k", "mrr", "citation_validity_rate"):
        assert math.isnan(m[key])
    assert m["correct_refusal_rate"] == 1.0


def test_sut_called_once_per_item_in_order() -> None:
    sut = scripted()
    run_eval(sut, ITEMS, k=2)
    assert sut.calls == [f"question {i}?" for i in range(1, 7)]


def test_sut_exception_propagates() -> None:
    class Boom(RuntimeError):
        pass

    def sut(question: str) -> SutResult:
        raise Boom(question)

    with pytest.raises(Boom):
        run_eval(sut, ITEMS)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        run_eval(scripted(), [], k=2)
    with pytest.raises(ValueError):
        run_eval(scripted(), ITEMS, k=0)
