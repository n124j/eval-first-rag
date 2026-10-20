"""Pure metric functions. Inputs are plain strings/floats, so any system can be scored."""

import math


def recall_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of `expected` found among the first `k` entries of `retrieved`."""
    if not expected:
        raise ValueError("expected must not be empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return len(expected & set(retrieved[:k])) / len(expected)


def mrr(expected: set[str], retrieved: list[str]) -> float:
    """Reciprocal of the 1-based rank of the first expected entry; 0.0 if none is present."""
    if not expected:
        raise ValueError("expected must not be empty")
    for rank, item in enumerate(retrieved, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the value at rank max(1, ceil(p/100 * n)) of the sorted values."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be within [0, 100], got {p}")
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered) / 100))
    return ordered[rank - 1]
