"""Dense (cosine) vector index held in memory with .npz persistence."""

import contextlib
import os
from pathlib import Path

import numpy as np

from evalrag.models import Hit


class DimensionMismatch(ValueError):
    """A vector's dimension differs from the index's dimension."""


class EmptyIndex(ValueError):
    """Search was attempted on an index with no vectors."""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError("vectors must be finite and non-zero")
    return vectors / norms


class DenseIndex:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if len(chunk_ids) != len(vectors):
            raise ValueError(f"{len(chunk_ids)} ids but {len(vectors)} vectors")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("duplicate chunk ids within call")
        clash = self._id_set.intersection(chunk_ids)
        if clash:
            raise ValueError(f"chunk ids already indexed: {sorted(clash)[:3]}")
        if not chunk_ids:
            return
        if self._matrix is not None and vectors.shape[1] != self._matrix.shape[1]:
            raise DimensionMismatch(
                f"index dimension is {self._matrix.shape[1]}, got {vectors.shape[1]}"
            )
        normed = _normalise(vectors).astype(np.float32)
        self._matrix = normed if self._matrix is None else np.vstack([self._matrix, normed])
        self._ids.extend(chunk_ids)
        self._id_set.update(chunk_ids)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if self._matrix is None:
            raise EmptyIndex("cannot search an empty index")
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._matrix.shape[1]:
            raise DimensionMismatch(
                f"index dimension is {self._matrix.shape[1]}, got {query.shape[1]}"
            )
        query = _normalise(query)[0]
        scores = self._matrix @ query
        # Round so last-bit BLAS differences cannot reorder ties; report what we sorted by.
        rounded = [round(float(s), 6) for s in scores]
        order = sorted(range(len(self._ids)), key=lambda i: (-rounded[i], self._ids[i]))
        return [
            Hit(chunk_id=self._ids[i], score=rounded[i], rank=rank)
            for rank, i in enumerate(order[:k], start=1)
        ]

    def save(self, path: Path) -> None:
        if self._matrix is None:
            matrix = np.zeros((0, 0), dtype=np.float32)
        else:
            matrix = self._matrix
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:
                np.savez(fh, ids=np.array(self._ids, dtype=str), vectors=matrix)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        index = cls()
        with np.load(Path(path), allow_pickle=False) as data:
            ids = [str(i) for i in data["ids"]]
            vectors = np.array(data["vectors"], dtype=np.float32)
        if len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError(f"corrupt dense index file: {path}")
        if ids:
            # Saved vectors are already normalised; keep them bit-for-bit.
            index._ids = ids
            index._id_set = set(ids)
            index._matrix = vectors
        return index
