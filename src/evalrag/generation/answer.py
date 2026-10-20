"""Prompt construction and citation parsing/verification for grounded answers."""

import re
from dataclasses import dataclass

from evalrag.models import Chunk

REFUSAL_TEXT = "Not found in the provided documents."
MIN_QUOTE_CHARS = 8

# Part of cached request keys: keep deterministic (no timestamps, no randomness).
SYSTEM_PROMPT = (
    "You answer questions using only the context provided in the user message. "
    "Do not use outside knowledge.\n"
    "Cite every claim as [chunk_id: \"exact quote\"], where chunk_id is the id shown in "
    "square brackets above a context block and the quote is copied verbatim from that "
    f"chunk and is at least {MIN_QUOTE_CHARS} characters long.\n"
    "If the context does not contain the answer, reply with exactly "
    f"{REFUSAL_TEXT} and nothing else."
)

_CITATION_RE = re.compile(r'\[([\w.\-]+#\d{3}):\s*"([^"]+)"\]')
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class CitationReport:
    valid: bool
    unknown_ids: list[str]
    missing_quotes: list[str]


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    if not question.strip():
        raise ValueError("question must not be empty")
    blocks = [f"[{c.chunk_id}]\n{c.text}" for c in chunks]
    blocks.append(f"Question: {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def parse_citations(text: str) -> list[Citation]:
    seen: set[Citation] = set()
    found: list[Citation] = []
    for match in _CITATION_RE.finditer(text):
        citation = Citation(chunk_id=match.group(1), quote=match.group(2))
        if citation not in seen:
            seen.add(citation)
            found.append(citation)
    return found


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text)


def verify_citations(citations: list[Citation], chunks: list[Chunk]) -> CitationReport:
    texts = {c.chunk_id: _normalise(c.text) for c in chunks}
    unknown_ids: list[str] = []
    missing_quotes: list[str] = []
    for citation in citations:
        if citation.chunk_id not in texts:
            if citation.chunk_id not in unknown_ids:
                unknown_ids.append(citation.chunk_id)
            continue
        quote = citation.quote.strip()
        if len(quote) < MIN_QUOTE_CHARS or _normalise(quote) not in texts[citation.chunk_id]:
            missing_quotes.append(f"{citation.chunk_id}: {citation.quote}")
    valid = bool(citations) and not unknown_ids and not missing_quotes
    return CitationReport(valid=valid, unknown_ids=unknown_ids, missing_quotes=missing_quotes)


def is_refusal(text: str) -> bool:
    return text.strip() == REFUSAL_TEXT
