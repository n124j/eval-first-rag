import pytest

from evalrag.evals.dataset import GoldenItem
from evalrag.evals.judge import JudgeScore
from evalrag.evals.report import render_markdown
from evalrag.evals.runner import EvalResult, SutResult, run_eval

META = {"model": "test-model", "hardware": "cpu", "date": "2026-01-02"}

Q1 = GoldenItem("q1", "How to parse?", ("argparse",), True, "single")
Q2 = GoldenItem("q2", "Pipe | question?", ("a", "b"), True, "multi")
SCRIPT = {
    Q1.question: SutResult(
        "ans", ["argparse#001", "argparse#002"], ["argparse#001"], True, False, 1.0, 10, 5
    ),
    Q2.question: SutResult(
        "ans", ["a#001", "a#002", "c#001", "d#001"], [], False, False, 3.0, 20, 7
    ),
}


def make_result() -> EvalResult:
    return run_eval(lambda q: SCRIPT[q], [Q1, Q2], k=5)


EXPECTED_BASE = """\
# Evaluation report

| Field | Value |
| --- | --- |
| model | test-model |
| hardware | cpu |
| date | 2026-01-02 |
| k | 5 |
| items | 2 |

## Metrics

| Metric | Value |
| --- | --- |
| recall_at_k | 0.7500 |
| mrr | 1.0000 |
| citation_validity_rate | 0.5000 |
| correct_refusal_rate | n/a |
| over_refusal_rate | 0.0000 |
| latency_p50_s | 1.000 |
| latency_p95_s | 3.000 |
| prompt_tokens_total | 30 |
| completion_tokens_total | 12 |

## Failures

- q2 (multi): Pipe | question? | failures: invalid citations | retrieved sources: a, c, d | cited: none
"""


def test_report_requires_meta_keys() -> None:
    result = make_result()
    with pytest.raises(ValueError, match="hardware") as exc:
        render_markdown(result, {"model": "m", "date": "d"})
    assert "model" not in str(exc.value)
    with pytest.raises(ValueError, match="judge_model"):
        render_markdown(result, META, {"q1": JudgeScore(5, 5, "x")})


def test_report_is_deterministic_and_exact() -> None:
    result = make_result()
    out = render_markdown(result, META)
    assert out == render_markdown(result, META)
    assert out == EXPECTED_BASE
    assert "\r" not in out


def test_report_lists_failures() -> None:
    result = make_result()
    out = render_markdown(result, META)
    assert "- q2 (multi)" in out
    assert "- q1" not in out


def test_report_no_failures_sentence() -> None:
    result = run_eval(lambda q: SCRIPT[Q1.question], [Q1], k=5)
    out = render_markdown(result, META)
    assert out.endswith("## Failures\n\nNo failures.\n")


def test_report_failure_line_with_cited_ids() -> None:
    item = GoldenItem("u1", "Unknown?", (), False, "unanswerable")
    sut = SutResult("x", [], ["a#001", "b#002"], False, False, 1.0, 1, 1)
    out = render_markdown(run_eval(lambda q: sut, [item], k=5), META)
    assert (
        "- u1 (unanswerable): Unknown? | failures: answered unanswerable question"
        " | retrieved sources: none | cited: a#001, b#002\n"
    ) in out


def test_report_escapes_pipes() -> None:
    meta = dict(META, hardware="a|b")
    out = render_markdown(make_result(), meta)
    assert "| hardware | a\\|b |" in out


def test_report_judge_section_only_when_scores_given() -> None:
    result = make_result()
    assert "LLM-judged" not in render_markdown(result, META)
    meta = dict(META, judge_model="judge-m")
    out = render_markdown(
        result, meta, {"q1": JudgeScore(5, 4, "good"), "q2": JudgeScore(2, 3, "x")}
    )
    assert out.startswith("# Evaluation report\n")
    assert out.endswith(
        "\n## LLM-judged scores (noisy)\n\n"
        "These scores were produced by `judge-m`, are noisy, and are not used by the gate.\n\n"
        "| Item | Faithfulness | Relevance |\n"
        "| --- | --- | --- |\n"
        "| q1 | 5 | 4 |\n"
        "| q2 | 2 | 3 |\n"
    )
    assert "| judge_model | judge-m |" in out


def test_report_rejects_scores_for_unknown_items() -> None:
    meta = dict(META, judge_model="j")
    with pytest.raises(ValueError, match="zzz"):
        render_markdown(make_result(), meta, {"zzz": JudgeScore(1, 1, "x")})


def test_report_meta_order_required_then_k_then_extras_alphabetically() -> None:
    meta = dict(META, min_dense_score="0.67", alpha="x")
    expected = EXPECTED_BASE.replace(
        "| k | 5 |\n", "| k | 5 |\n| alpha | x |\n| min_dense_score | 0.67 |\n"
    )
    assert render_markdown(make_result(), meta) == expected
