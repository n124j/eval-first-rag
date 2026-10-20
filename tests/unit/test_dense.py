from pathlib import Path

import numpy as np
import pytest

from evalrag.retrieval.dense import DenseIndex, DimensionMismatch, EmptyIndex


def build() -> DenseIndex:
    idx = DenseIndex()
    idx.add(["x", "y", "z"], np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    return idx


def test_cosine_order():
    hits = build().search(np.array([1.0, 0.1], dtype=np.float32), 3)
    assert [h.chunk_id for h in hits] == ["x", "z", "y"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score > hits[2].score
    assert isinstance(hits[0].score, float)


def test_save_load_roundtrip(tmp_path: Path):
    idx = build()
    idx.save(tmp_path / "d.npz")
    loaded = DenseIndex.load(tmp_path / "d.npz")
    q = np.array([0.3, 0.9], dtype=np.float32)
    assert len(loaded) == 3
    assert loaded.search(q, 3) == idx.search(q, 3)


def test_dimension_mismatch_raises():
    idx = build()
    with pytest.raises(DimensionMismatch):
        idx.add(["w"], np.ones((1, 3), dtype=np.float32))
    with pytest.raises(DimensionMismatch):
        idx.search(np.ones(3, dtype=np.float32), 1)


def test_empty_index_search_raises():
    with pytest.raises(EmptyIndex):
        DenseIndex().search(np.ones(2, dtype=np.float32), 1)


def test_k_larger_than_corpus_returns_all():
    assert len(build().search(np.array([1.0, 1.0], dtype=np.float32), 50)) == 3


def test_normalisation_makes_scale_irrelevant():
    idx = DenseIndex()
    idx.add(["a", "b"], np.array([[100, 0], [0, 0.001]], dtype=np.float32))
    hits = idx.search(np.array([5.0, 0.0], dtype=np.float32), 2)
    assert hits[0].chunk_id == "a"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)


def test_duplicate_ids_rejected():
    idx = build()
    with pytest.raises(ValueError):
        idx.add(["x"], np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        DenseIndex().add(["p", "p"], np.ones((2, 2), dtype=np.float32))
    assert len(idx) == 3


def test_length_mismatch_and_zero_vectors_rejected():
    with pytest.raises(ValueError):
        DenseIndex().add(["a"], np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        DenseIndex().add(["a"], np.zeros((1, 2), dtype=np.float32))


def test_bad_search_arguments():
    idx = build()
    with pytest.raises(ValueError):
        idx.search(np.ones(2, dtype=np.float32), 0)
    with pytest.raises(ValueError):
        idx.search(np.zeros(2, dtype=np.float32), 1)


def test_ties_broken_by_chunk_id():
    idx = DenseIndex()
    idx.add(["b", "a", "c"], np.ones((3, 2), dtype=np.float32))
    hits = idx.search(np.ones(2, dtype=np.float32), 3)
    assert [h.chunk_id for h in hits] == ["a", "b", "c"]


def test_near_ties_below_rounding_step_order_by_chunk_id():
    # "a" has a raw cosine just below 1.0 (about 1 - 1.8e-7) that rounds to 1.0;
    # a raw-score tie-break would wrongly put "b" first.
    idx = DenseIndex()
    idx.add(["b", "a"], np.array([[1.0, 0.0], [1.0, 6e-4]], dtype=np.float32))
    hits = idx.search(np.array([1.0, 0.0], dtype=np.float32), 2)
    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert [h.score for h in hits] == [1.0, 1.0]


def test_scores_are_rounded_to_6_decimals():
    idx = DenseIndex()
    idx.add(["a"], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    score = idx.search(np.array([3.0, 1.0, 2.0], dtype=np.float32), 1)[0].score
    assert score == round(score, 6)


def test_failed_save_keeps_previous_file_and_leaves_no_tmp(tmp_path: Path, monkeypatch):
    path = tmp_path / "d.npz"
    idx = build()
    idx.save(path)
    before = path.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d.npz"]

    def failing(fh, **arrays):
        fh.write(b"partial garbage")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", failing)
    with pytest.raises(OSError):
        idx.save(path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d.npz"]
