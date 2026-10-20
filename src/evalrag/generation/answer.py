"""Prompt construction and citation parsing/verification for grounded answers."""

import re
from dataclasses import dataclass

from evalrag.models import Chunk

REFUSAL_TEXT = "Not found in the provided documents."
MIN_QUOTE_CHARS = 8

# Part of cached request keys: keep deterministic (no timestamps, no randomness).
SYSTEM_PROMPT = (
    "You answer questions using only the sources in the user message. "
    "Do not use outside knowledge and do not copy the sources.\n"
    "Answer in one or two short sentences. End each sentence with a citation: the source id, "
    "a colon, and a quote copied word for word from that source, all in square brackets, "
    'like [widgets#004: "returns the new angle in degrees"]. '
    f"The quote must be at least {MIN_QUOTE_CHARS} characters long.\n"
    "If the sources do not answer the question, reply with exactly "
    f"{REFUSAL_TEXT} and nothing else."
)

# Few-shot examples, sent as earlier user/assistant turns in the same layout as the real
# question. Their sources are made up (no corpus text). Also part of the cache keys.
# Chosen on the dev set in ADR 0005.
_EXAMPLE_ANSWER_CHUNKS = (
    ("widgets#004", "The spin() function rotates a widget. It returns the new angle in degrees."),
    ("widgets#007", "The hide() function removes a widget from the screen until show() is called."),
)
_EXAMPLE_REFUSAL_CHUNKS = (
    ("widgets#002", "A widget has a name and a size. The size is given in pixels."),
    ("gadgets#010", "The gadget registry keeps one entry for every gadget that was created."),
)
FEW_SHOT = (
    (
        "What does spin() return?",
        _EXAMPLE_ANSWER_CHUNKS,
        'spin() returns the new angle of the widget [widgets#004: "returns the new angle in degrees"].',
    ),
    ("How do I paint a widget blue?", _EXAMPLE_REFUSAL_CHUNKS, REFUSAL_TEXT),
)

# The quote may contain double quotes, because the corpus wraps identifiers in them
# ("Fraction"). It runs to the first `"]` on the same line and never contains `"]`, so an
# empty quote cannot swallow the next citation. verify_citations still requires the quote
# (at least MIN_QUOTE_CHARS) to occur in the cited retrieved chunk. See ADR 0005, 12h.
_CITATION_RE = re.compile(r'\[([\w.\-]+#\d{3}):\s*"((?:(?!"\]).)+?)"\]')
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


def _user_content(question: str, sources: list[tuple[str, str]]) -> str:
    parts = [f"Question: {question}", "Sources:"]
    parts.extend(f"Source id: {chunk_id}\nText: {text}" for chunk_id, text in sources)
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    # The question and chunk texts are collapsed to one line each, so neither can start a
    # line with "Source id:", "Text:", "Question:" or "Answer:" and forge a block. Marker
    # text inside a single line is still possible; see ADR 0005, "Prompt-format injection".
    # chunk_document already emits single-spaced text, so real prompts are unchanged.
    question = _one_line(question)
    if not question:
        raise ValueError("question must not be empty")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_question, example_sources, example_answer in FEW_SHOT:
        messages.append(
            {"role": "user", "content": _user_content(example_question, list(example_sources))}
        )
        messages.append({"role": "assistant", "content": example_answer})
    sources = [(c.chunk_id, _one_line(c.text)) for c in chunks]
    messages.append({"role": "user", "content": _user_content(question, sources)})
    return messages


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
        quote = _normalise(citation.quote.strip())
        if len(quote) < MIN_QUOTE_CHARS or quote not in texts[citation.chunk_id]:
            missing_quotes.append(f"{citation.chunk_id}: {citation.quote}")
    valid = bool(citations) and not unknown_ids and not missing_quotes
    return CitationReport(valid=valid, unknown_ids=unknown_ids, missing_quotes=missing_quotes)


def is_refusal(text: str) -> bool:
    return text.strip() == REFUSAL_TEXT
