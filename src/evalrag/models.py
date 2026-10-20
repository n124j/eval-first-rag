"""Domain models shared across the pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    url: str
    version: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    url: str


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    rank: int
