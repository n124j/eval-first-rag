"""Hybrid retrieval: keyword + dense candidates fused with RRF."""

from collections.abc import Mapping
from dataclasses import dataclass

from evalrag.models import Hit
from evalrag.retrieval.dense import DenseIndex
from evalrag.retrieval.embedder import Embedder
from evalrag.retrieval.fusion import rrf
from evalrag.retrieval.keyword import KeywordIndex
from evalrag.retrieval.rerank import Reranker


@dataclass(frozen=True)
class RetrievalResult:
    """Final hits plus the cosine of the dense rank-1 candidate, taken before fusion.

    `top_dense_score` is the retrieval-confidence signal used for abstention (ADR 0006); the
    fused RRF scores in `hits` are rank-based and carry no such meaning.
    """

    hits: list[Hit]
    top_dense_score: float


class HybridRetriever:
    def __init__(
        self,
        keyword: KeywordIndex,
        dense: DenseIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        texts: Mapping[str, str] | None = None,
        candidate_k: int = 50,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(f"candidate_k must be >= 1, got {candidate_k}")
        if reranker is not None and texts is None:
            raise ValueError("texts is required when a reranker is given")
        self._keyword = keyword
        self._dense = dense
        self._embedder = embedder
        self._reranker = reranker
        self._texts = texts
        self._candidate_k = candidate_k

    def retrieve(self, query: str, k: int) -> list[Hit]:
        return self.retrieve_detailed(query, k).hits

    def retrieve_detailed(self, query: str, k: int) -> RetrievalResult:
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        keyword_hits = self._keyword.search(query, self._candidate_k)
        vector = self._embedder.embed([query], kind="query")[0]
        dense_hits = self._dense.search(vector, self._candidate_k)
        # No dense candidate at all means no confidence: report 0.0.
        top_dense_score = round(dense_hits[0].score, 6) if dense_hits else 0.0
        fused = rrf([[h.chunk_id for h in keyword_hits], [h.chunk_id for h in dense_hits]])[
            : self._candidate_k
        ]
        hits = [
            Hit(chunk_id=cid, score=round(score, 6), rank=rank)
            for rank, (cid, score) in enumerate(fused, start=1)
        ]
        if self._reranker is not None and self._texts is not None:
            hits = self._reranker.rerank(query, hits, self._texts)
        final = [
            Hit(chunk_id=h.chunk_id, score=round(h.score, 6), rank=rank)
            for rank, h in enumerate(hits[:k], start=1)
        ]
        return RetrievalResult(hits=final, top_dense_score=top_dense_score)

    def close(self) -> None:
        """Close the keyword index; safe to call more than once."""
        self._keyword.close()
