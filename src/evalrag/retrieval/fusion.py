"""Reciprocal rank fusion."""

import math


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists; score = sum of 1/(k + rank), rank 1-based.

    Sorted by (-score, chunk_id) so the result is deterministic; ``fsum`` makes
    each score independent of the order the rankings are supplied in.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    contributions: dict[str, list[float]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                raise ValueError(f"duplicate chunk_id in one ranking: {chunk_id}")
            seen.add(chunk_id)
            contributions.setdefault(chunk_id, []).append(1.0 / (k + rank))
    scored = [(cid, math.fsum(parts)) for cid, parts in contributions.items()]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored
