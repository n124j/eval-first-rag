"""Reranker hook: a no-op default and an optional cross-encoder."""

from collections.abc import Mapping
from typing import Protocol

from evalrag.models import Hit


class RerankerUnavailable(RuntimeError):
    """The optional reranking dependency is not installed."""


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], texts: Mapping[str, str]) -> list[Hit]: ...


class NoopReranker:
    def rerank(self, query: str, hits: list[Hit], texts: Mapping[str, str]) -> list[Hit]:
        return list(hits)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailable(
                "CrossEncoderReranker needs sentence-transformers; "
                "install it with: pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], texts: Mapping[str, str]) -> list[Hit]:
        pairs = []
        for hit in hits:
            if hit.chunk_id not in texts:
                raise KeyError(f"no text for chunk_id: {hit.chunk_id}")
            pairs.append((query, texts[hit.chunk_id]))
        if not pairs:
            return []
        scores = [round(float(s), 6) for s in self._model.predict(pairs)]
        ranked = sorted(zip(hits, scores, strict=True), key=lambda p: (-p[1], p[0].chunk_id))
        return [
            Hit(chunk_id=hit.chunk_id, score=score, rank=rank)
            for rank, (hit, score) in enumerate(ranked, start=1)
        ]
