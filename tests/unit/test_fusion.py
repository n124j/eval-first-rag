import itertools

import pytest

from evalrag.retrieval.fusion import rrf


def test_rrf_known_values():
    result = rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)
    expected = {
        "a": 1 / 61 + 1 / 63,
        "b": 1 / 62 + 1 / 61,
        "c": 1 / 63,
        "d": 1 / 62,
    }
    assert [cid for cid, _ in result] == ["b", "a", "d", "c"]
    for cid, score in result:
        assert score == pytest.approx(expected[cid], abs=1e-12)


def test_rrf_tie_break_is_deterministic():
    # a and b each appear once at rank 1 in separate rankings: equal scores.
    result = rrf([["b"], ["a"]])
    assert [cid for cid, _ in result] == ["a", "b"]
    assert result[0][1] == result[1][1]


def test_item_in_both_lists_beats_item_in_one():
    result = rrf([["x", "shared"], ["y", "shared"]])
    assert result[0][0] == "shared"


def test_rrf_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        rrf([["a"]], k=0)
    with pytest.raises(ValueError):
        rrf([["a"]], k=-5)


def test_rrf_rejects_duplicate_id_in_one_ranking():
    with pytest.raises(ValueError, match="dup"):
        rrf([["a", "dup", "dup"]])


def test_rrf_empty_inputs_return_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_score_is_independent_of_ranking_order():
    rankings = [["a", "b", "c"], ["c", "a", "d"], ["d", "c", "b", "a"]]
    baseline = rrf(rankings)
    for perm in itertools.permutations(rankings):
        assert rrf(list(perm)) == baseline
