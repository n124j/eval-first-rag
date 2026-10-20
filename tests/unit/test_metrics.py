import pytest

from evalrag.evals.dataset import source_of
from evalrag.evals.metrics import mrr, percentile, recall_at_k


def test_recall_at_k_exact_value() -> None:
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], 2) == 0.5
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], 3) == 1.0


def test_recall_ignores_duplicates_in_retrieved() -> None:
    assert recall_at_k({"a", "b"}, ["a", "a", "a"], 3) == 0.5


def test_recall_empty_expected_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k(set(), ["a"], 1)


def test_recall_nonpositive_k_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k({"a"}, ["a"], 0)
    with pytest.raises(ValueError):
        recall_at_k({"a"}, ["a"], -1)


def test_mrr_exact_values() -> None:
    assert mrr({"b"}, ["a", "b"]) == 0.5
    assert mrr({"a", "b"}, ["x", "y", "b", "a"]) == pytest.approx(1 / 3)
    assert mrr({"a"}, ["a"]) == 1.0
    assert mrr({"a"}, ["x", "y"]) == 0.0
    assert mrr({"a"}, []) == 0.0


def test_mrr_empty_expected_raises() -> None:
    with pytest.raises(ValueError):
        mrr(set(), ["a"])


def test_percentile_nearest_rank_exact() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 50) == 2
    assert percentile(values, 95) == 4
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 4


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], -0.1)
    with pytest.raises(ValueError):
        percentile([1.0], 100.1)


def test_source_of_handles_dotted_ids_and_missing_hash() -> None:
    assert source_of("argparse#016") == "argparse"
    assert source_of("os.path#003") == "os.path"
    assert source_of("os.path") == "os.path"
    assert source_of("a#b#007") == "a#b"
