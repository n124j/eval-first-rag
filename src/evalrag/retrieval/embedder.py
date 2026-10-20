"""Embedders: a live OpenAI-compatible client plus record/replay wrappers.

Record/replay lets the eval run offline and deterministically: vectors are
recorded once against the live server, then replayed from an ``.npz`` cache.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Protocol

import httpx
import numpy as np

from evalrag.errors import CacheMismatchError

Kind = Literal["document", "query"]


class EmbeddingError(RuntimeError):
    """The embedding server failed or returned an unusable response."""


class ReplayMiss(KeyError):
    """A (kind, text) pair was not recorded in the replay cache."""

    def __str__(self) -> str:
        # KeyError.__str__ would repr() the message; show it plainly instead.
        return str(self.args[0]) if self.args else ""


class Embedder(Protocol):
    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        """Return float32 vectors of shape (n, dim); empty input gives shape (0, 0)."""
        ...


def cache_key(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\n{text}".encode()).hexdigest()


def _empty() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float32)


class OpenAICompatEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        client: httpx.Client | None = None,
        document_prefix: str = "search_document: ",
        query_prefix: str = "search_query: ",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client if client is not None else httpx.Client(timeout=60.0)
        self.prefixes = {"document": document_prefix, "query": query_prefix}
        self.batch_size = batch_size

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        if not texts:
            return _empty()
        prefix = self.prefixes[kind]
        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[start : start + self.batch_size]]
            batches.append(self._embed_batch(batch))
        dims = {b.shape[1] for b in batches}
        if len(dims) != 1:
            raise EmbeddingError(f"inconsistent embedding dimensions across batches: {dims}")
        return np.vstack(batches).astype(np.float32, copy=False)

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        try:
            response = self.client.post(
                f"{self.base_url}/embeddings", json={"model": self.model, "input": batch}
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        if not response.is_success:
            raise EmbeddingError(
                f"embedding server returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = np.array([item["embedding"] for item in items], dtype=np.float32)
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"malformed embedding response: {exc!r}: {response.text[:200]}"
            ) from exc
        if vectors.ndim != 2 or len(vectors) != len(batch):
            raise EmbeddingError(
                f"expected {len(batch)} vectors, got shape {vectors.shape}: {response.text[:200]}"
            )
        return vectors


class RecordingEmbedder:
    def __init__(self, inner: Embedder, cache_path: Path, model: str = "") -> None:
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.model = model
        self._vectors: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            keys, vectors, meta = _load_cache(self.cache_path)
            cached_model = meta.get("model", "")
            if model and cached_model and model != cached_model:
                raise CacheMismatchError(
                    f"{self.cache_path}: cache was recorded with model {cached_model!r}, "
                    f"not {model!r}"
                )
            self._vectors = dict(zip(keys, vectors, strict=True))
            self.model = self.model or cached_model

    @property
    def entry_count(self) -> int:
        return len(self._vectors)

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        if not texts:
            return _empty()
        keys = [cache_key(kind, t) for t in texts]
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in self._vectors:
                missing.setdefault(key, text)
        if missing:
            fresh = self.inner.embed(list(missing.values()), kind=kind)
            if self._vectors and len(fresh):
                existing_dim = int(next(iter(self._vectors.values())).shape[0])
                if int(fresh.shape[1]) != existing_dim:
                    raise CacheMismatchError(
                        f"{self.cache_path}: cache has dimension {existing_dim}, "
                        f"new vectors have dimension {int(fresh.shape[1])}"
                    )
            for key, vector in zip(missing, fresh, strict=True):
                self._vectors[key] = vector.astype(np.float16)
        return np.stack([self._vectors[k] for k in keys]).astype(np.float32)

    def flush(self) -> None:
        if not self._vectors:
            return
        keys = sorted(self._vectors)
        vectors = np.stack([self._vectors[k] for k in keys]).astype(np.float16)
        meta = json.dumps({"model": self.model, "dim": int(vectors.shape[1])})
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:
                np.savez(fh, keys=np.array(keys, dtype=str), vectors=vectors, meta=np.array(meta))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.cache_path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise


class ReplayEmbedder:
    def __init__(self, cache_path: Path) -> None:
        keys, vectors, meta = _load_cache(Path(cache_path))
        self._index = {key: i for i, key in enumerate(keys)}
        self._vectors = vectors
        self.model: str = meta.get("model", "")
        self.dim: int = int(meta.get("dim", vectors.shape[1] if vectors.size else 0))

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        if not texts:
            return _empty()
        rows = []
        for text in texts:
            key = cache_key(kind, text)
            if key not in self._index:
                raise ReplayMiss(f"no recorded embedding for key {key} (text: {text[:60]!r})")
            rows.append(self._index[key])
        return self._vectors[rows].astype(np.float32)


def _load_cache(path: Path) -> tuple[list[str], np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        keys = [str(k) for k in data["keys"]]
        vectors = np.array(data["vectors"], dtype=np.float16)
        meta = json.loads(str(data["meta"]))
    return keys, vectors, meta
