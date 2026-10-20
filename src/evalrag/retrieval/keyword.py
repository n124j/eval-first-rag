"""BM25 keyword index backed by SQLite FTS5."""

import re
import sqlite3
from pathlib import Path

from evalrag.models import Chunk, Hit

_TOKEN = re.compile(r"\w+")


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 query: quoted tokens joined by OR."""
    tokens = list(dict.fromkeys(_TOKEN.findall(query.lower())))
    return " OR ".join(f'"{token}"' for token in tokens)


class KeywordIndex:
    def __init__(self, db_path: Path, *, check_same_thread: bool = True) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "chunk_id UNINDEXED, text, tokenize='porter unicode61')"
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS chunk_ids (chunk_id TEXT PRIMARY KEY)")
        self._conn.commit()

    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks atomically; a duplicate chunk_id raises and adds nothing."""
        with self._conn:  # one transaction; rolls back if anything raises
            for chunk in chunks:
                try:
                    self._conn.execute(
                        "INSERT INTO chunk_ids (chunk_id) VALUES (?)", (chunk.chunk_id,)
                    )
                except sqlite3.IntegrityError:
                    raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}") from None
                self._conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                    (chunk.chunk_id, chunk.text),
                )

    def search(self, query: str, k: int) -> list[Hit]:
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        match = _fts_query(query)
        if not match:
            return []
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score, chunk_id LIMIT ?",
            (match, k),
        ).fetchall()
        return [
            Hit(chunk_id=chunk_id, score=-score, rank=rank)
            for rank, (chunk_id, score) in enumerate(rows, start=1)
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunk_ids").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
