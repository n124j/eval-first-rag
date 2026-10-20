"""Build and open the on-disk index: keyword (FTS5), dense vectors and the chunk table."""

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evalrag.chunking import chunk_document
from evalrag.config import Settings
from evalrag.corpus import load_corpus
from evalrag.models import Chunk
from evalrag.retrieval.dense import DenseIndex
from evalrag.retrieval.embedder import Embedder
from evalrag.retrieval.hybrid import HybridRetriever
from evalrag.retrieval.keyword import KeywordIndex

KEYWORD_FILE = "keyword.sqlite"
DENSE_FILE = "dense.npz"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "ingest.json"
EMBED_BATCH = 64
CANDIDATE_K = 50
_CHUNK_KEYS = ("chunk_id", "doc_id", "ordinal", "text", "url")


@dataclass(frozen=True)
class IngestSummary:
    documents: int
    chunks: int
    dim: int
    chunk_size: int
    chunk_overlap: int


def _write_atomic(path: Path, data: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def ingest(
    settings: Settings, corpus_dir: Path, data_dir: Path, embedder: Embedder
) -> IngestSummary:
    """Replace the index in `data_dir` with one built from `corpus_dir`."""
    documents = load_corpus(corpus_dir)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, settings.chunk_size, settings.chunk_overlap))

    parts = [
        embedder.embed([c.text for c in chunks[i : i + EMBED_BATCH]], kind="document")
        for i in range(0, len(chunks), EMBED_BATCH)
    ]
    vectors = np.vstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)
    dim = int(vectors.shape[1]) if len(chunks) else 0

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (KEYWORD_FILE, DENSE_FILE, CHUNKS_FILE, META_FILE):
        (data_dir / name).unlink(missing_ok=True)

    keyword = KeywordIndex(data_dir / KEYWORD_FILE)
    try:
        keyword.add(chunks)
    finally:
        keyword.close()

    dense = DenseIndex()
    dense.add([c.chunk_id for c in chunks], vectors)
    dense.save(data_dir / DENSE_FILE)

    lines = [
        json.dumps(
            {key: getattr(c, key) for key in _CHUNK_KEYS}, sort_keys=True, ensure_ascii=False
        )
        + "\n"
        for c in chunks
    ]
    _write_atomic(data_dir / CHUNKS_FILE, "".join(lines))

    summary = IngestSummary(
        documents=len(documents),
        chunks=len(chunks),
        dim=dim,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    meta = {
        "chunk_size": summary.chunk_size,
        "chunk_overlap": summary.chunk_overlap,
        "chunks": summary.chunks,
        "dim": summary.dim,
        "documents": summary.documents,
        "embed_model": str(getattr(embedder, "model", "")),
    }
    _write_atomic(data_dir / META_FILE, json.dumps(meta, sort_keys=True, indent=2) + "\n")
    return summary


def load_chunks(path: Path) -> list[Chunk]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"chunks file not found: {path}")
    chunks: list[Chunk] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict) or set(raw) != set(_CHUNK_KEYS):
                    raise ValueError("expected an object with keys " + ", ".join(_CHUNK_KEYS))
                chunks.append(
                    Chunk(
                        chunk_id=str(raw["chunk_id"]),
                        doc_id=str(raw["doc_id"]),
                        ordinal=int(raw["ordinal"]),
                        text=str(raw["text"]),
                        url=str(raw["url"]),
                    )
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: line {lineno}: malformed chunk: {exc}") from exc
    return chunks


def open_retriever(
    settings: Settings,
    data_dir: Path,
    embedder: Embedder,
    *,
    check_same_thread: bool = True,
) -> tuple[HybridRetriever, dict[str, Chunk]]:
    data_dir = Path(data_dir)
    if not (data_dir / META_FILE).exists():
        raise FileNotFoundError(f'no index in {data_dir}; run "evalrag ingest" first')
    chunks = load_chunks(data_dir / CHUNKS_FILE)
    dense = DenseIndex.load(data_dir / DENSE_FILE)
    keyword = KeywordIndex(data_dir / KEYWORD_FILE, check_same_thread=check_same_thread)
    retriever = HybridRetriever(keyword, dense, embedder, candidate_k=CANDIDATE_K)
    return retriever, {c.chunk_id: c for c in chunks}
