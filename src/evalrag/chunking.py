"""Word-window chunking."""

from evalrag.models import Chunk, Document


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split a document into overlapping windows of ``size`` words.

    Windows step by ``size - overlap``. A trailing window that would add no
    new words is not emitted.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got {overlap}")

    words = doc.text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = start + size
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{ordinal:03d}",
                doc_id=doc.doc_id,
                ordinal=ordinal,
                text=" ".join(words[start:end]),
                url=doc.url,
            )
        )
        if end >= len(words):
            break
        start += step
    return chunks
