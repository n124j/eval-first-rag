import sys
import types

import numpy as np
import pytest

from evalrag.models import Chunk, Hit
from evalrag.retrieval.dense import DenseIndex, EmptyIndex
from evalrag.retrieval.hybrid import HybridRetriever
from evalrag.retrieval.keyword import KeywordIndex
from evalrag.retrieval.rerank import (
    CrossEncoderReranker,
    NoopReranker,
    RerankerUnavailable,
)


def hits(*ids: str) -> list[Hit]:
    return [Hit(chunk_id=i, score=1.0 / r, rank=r) for r, i in enumerate(ids, start=1)]


class FakeKeyword:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.calls.append((query, k))
        return hits(*self.ids)[:k]


class FakeDense:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[int] = []

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        self.calls.append(k)
        return hits(*self.ids)[:k]


class FakeEmbedder:
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        self.kinds.append(kind)
        return np.ones((len(texts), 2), dtype=np.float32)


def make(kw, dn, **kwargs) -> HybridRetriever:
    return HybridRetriever(kw, dn, FakeEmbedder(), **kwargs)


def test_hybrid_merges_keyword_and_dense():
    r = make(FakeKeyword(["a", "b", "c"]), FakeDense(["b", "d", "a"]))
    out = r.retrieve("q", 10)
    assert [h.chunk_id for h in out] == ["b", "a", "d", "c"]
    # b = 1/62 + 1/61, a = 1/61 + 1/63, d = 1/62, c = 1/63, rounded to 6 places.
    assert [h.score for h in out] == [0.032522, 0.032266, 0.016129, 0.015873]
    assert [h.rank for h in out] == [1, 2, 3, 4]


def test_hybrid_with_empty_keyword_result_uses_dense():
    r = make(FakeKeyword([]), FakeDense(["x", "y"]))
    out = r.retrieve("q", 5)
    assert out == [
        Hit(chunk_id="x", score=0.016393, rank=1),
        Hit(chunk_id="y", score=0.016129, rank=2),
    ]


def test_hybrid_query_uses_query_kind():
    emb = FakeEmbedder()
    r = HybridRetriever(FakeKeyword(["a"]), FakeDense(["a"]), emb)
    r.retrieve("q", 1)
    assert emb.kinds == ["query"]


def test_hybrid_truncates_to_k_and_renumbers_ranks():
    r = make(FakeKeyword(["a", "b", "c"]), FakeDense(["b", "d", "a"]))
    out = r.retrieve("q", 2)
    assert [(h.chunk_id, h.rank) for h in out] == [("b", 1), ("a", 2)]


def test_hybrid_rejects_nonpositive_k():
    r = make(FakeKeyword(["a"]), FakeDense(["a"]))
    with pytest.raises(ValueError):
        r.retrieve("q", 0)


def test_hybrid_passes_candidate_k_to_indexes():
    kw, dn = FakeKeyword(["a"]), FakeDense(["a"])
    make(kw, dn, candidate_k=7).retrieve("q", 1)
    assert kw.calls == [("q", 7)]
    assert dn.calls == [7]


def test_dense_empty_index_propagates():
    r = HybridRetriever(FakeKeyword(["a"]), DenseIndex(), FakeEmbedder())
    with pytest.raises(EmptyIndex):
        r.retrieve("q", 3)


def test_hybrid_with_real_indexes(tmp_path):
    chunks = [
        Chunk("c1", "d", 0, "alpha beta", "u"),
        Chunk("c2", "d", 1, "beta gamma", "u"),
        Chunk("c3", "d", 2, "delta", "u"),
    ]
    kw = KeywordIndex(tmp_path / "k.db")
    kw.add(chunks)
    dn = DenseIndex()
    dn.add(["c1", "c2", "c3"], np.array([[1, 1], [1, 0.5], [0, 1]], dtype=np.float32))
    out = HybridRetriever(kw, dn, FakeEmbedder()).retrieve("beta", 3)
    assert {h.chunk_id for h in out} == {"c1", "c2", "c3"}
    assert [h.rank for h in out] == [1, 2, 3]
    kw.close()


class ReverseReranker:
    def rerank(self, query, hits, texts):
        ordered = sorted(hits, key=lambda h: h.chunk_id, reverse=True)
        return [Hit(h.chunk_id, float(len(texts[h.chunk_id])), r) for r, h in enumerate(ordered, 1)]


TEXTS = {"a": "x", "b": "yy", "c": "zzz", "d": "w"}


def test_reranker_reorders_but_keeps_membership():
    plain = make(FakeKeyword(["a", "b", "c"]), FakeDense(["b", "d", "a"])).retrieve("q", 10)
    r = make(
        FakeKeyword(["a", "b", "c"]),
        FakeDense(["b", "d", "a"]),
        reranker=ReverseReranker(),
        texts=TEXTS,
    )
    out = r.retrieve("q", 10)
    assert [h.chunk_id for h in out] == ["d", "c", "b", "a"]
    assert {h.chunk_id for h in out} == {h.chunk_id for h in plain}
    assert [h.rank for h in out] == [1, 2, 3, 4]
    assert out[1].score == 3.0


def test_reranker_output_truncated_to_k():
    r = make(
        FakeKeyword(["a", "b", "c"]),
        FakeDense(["b", "d", "a"]),
        reranker=ReverseReranker(),
        texts=TEXTS,
    )
    assert [h.chunk_id for h in r.retrieve("q", 2)] == ["d", "c"]


def test_reranker_requires_texts():
    with pytest.raises(ValueError):
        make(FakeKeyword([]), FakeDense([]), reranker=NoopReranker())


def test_noop_reranker_returns_new_equal_list():
    hs = hits("a", "b")
    out = NoopReranker().rerank("q", hs, {})
    assert out == hs
    assert out is not hs


def _fake_st(monkeypatch, scores: dict[str, float]):
    mod = types.ModuleType("sentence_transformers")

    class CrossEncoder:
        def __init__(self, name: str) -> None:
            self.name = name

        def predict(self, pairs):
            return [scores[text] for _, text in pairs]

    mod.CrossEncoder = CrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)


def test_cross_encoder_with_fake_module_orders_by_score(monkeypatch):
    _fake_st(monkeypatch, {"ta": 0.1, "tb": 0.9123456789, "tc": 0.5})
    rr = CrossEncoderReranker("m")
    out = rr.rerank("q", hits("a", "b", "c"), {"a": "ta", "b": "tb", "c": "tc"})
    assert out == [
        Hit("b", 0.912346, 1),
        Hit("c", 0.5, 2),
        Hit("a", 0.1, 3),
    ]


def test_cross_encoder_missing_text_raises_keyerror(monkeypatch):
    _fake_st(monkeypatch, {})
    with pytest.raises(KeyError, match="zz"):
        CrossEncoderReranker("m").rerank("q", hits("zz"), {})


def test_cross_encoder_unavailable_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankerUnavailable, match="pip install sentence-transformers"):
        CrossEncoderReranker("m")
