import pytest

from evalrag.generation.answer import (
    MIN_QUOTE_CHARS,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    Citation,
    CitationReport,
    build_messages,
    is_refusal,
    parse_citations,
    verify_citations,
)
from evalrag.models import Chunk


def chunk(chunk_id: str, text: str) -> Chunk:
    doc_id, ordinal = chunk_id.split("#")
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, ordinal=int(ordinal), text=text, url="u")


A = chunk("argparse#016", "The add_argument method defines how a single argument is parsed.")
B = chunk("os.path#003", "os.path.join joins\n  path   components intelligently.")


def test_constants():
    assert REFUSAL_TEXT == "Not found in the provided documents."
    assert MIN_QUOTE_CHARS == 8
    assert REFUSAL_TEXT in SYSTEM_PROMPT


def test_build_messages_is_deterministic_and_ordered():
    chunks = [A, B]
    first = build_messages("How do I join paths?", chunks)
    assert first == build_messages("How do I join paths?", chunks)
    assert first == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "[argparse#016]\n"
                "The add_argument method defines how a single argument is parsed.\n"
                "\n"
                "[os.path#003]\n"
                "os.path.join joins\n  path   components intelligently.\n"
                "\n"
                "Question: How do I join paths?"
            ),
        },
    ]


def test_build_messages_rejects_empty_question():
    with pytest.raises(ValueError):
        build_messages("   \n", [A])


def test_parse_citations_handles_multiple_and_ignores_malformed():
    text = (
        'Yes [argparse#016: "defines how a single argument"] and '
        '[os.path#003: "path components"]. Bad: [argparse#16: "x y z w"] '
        '[argparse#016: no quotes] [argparse#016: ""] [nohash: "abc def gh"]'
    )
    assert parse_citations(text) == [
        Citation("argparse#016", "defines how a single argument"),
        Citation("os.path#003", "path components"),
    ]


def test_parse_citations_dedupes_and_keeps_order():
    text = '[b#002: "second one"] [a#001: "first one"] [b#002: "second one"] [a#001: "other one"]'
    assert parse_citations(text) == [
        Citation("b#002", "second one"),
        Citation("a#001", "first one"),
        Citation("a#001", "other one"),
    ]


def test_parse_citations_supports_dotted_doc_ids():
    text = '[os.path#003: "join paths"] [urllib.parse#012: "split the URL"]'
    assert parse_citations(text) == [
        Citation("os.path#003", "join paths"),
        Citation("urllib.parse#012", "split the URL"),
    ]


def test_verify_accepts_valid_citations():
    report = verify_citations([Citation("argparse#016", "defines how a single argument")], [A, B])
    assert report == CitationReport(valid=True, unknown_ids=[], missing_quotes=[])


def test_verify_rejects_unknown_chunk_id():
    report = verify_citations([Citation("zzz#001", "whatever quote")], [A])
    assert report == CitationReport(valid=False, unknown_ids=["zzz#001"], missing_quotes=[])


def test_verify_rejects_quote_not_in_chunk():
    report = verify_citations([Citation("argparse#016", "invented sentence here")], [A])
    assert report == CitationReport(
        valid=False, unknown_ids=[], missing_quotes=["argparse#016: invented sentence here"]
    )


def test_verify_requires_at_least_one_citation():
    assert verify_citations([], [A]) == CitationReport(False, [], [])


def test_verify_normalises_whitespace_in_quote_and_chunk():
    report = verify_citations([Citation("os.path#003", "joins path\ncomponents")], [B])
    assert report.valid is True


def test_verify_rejects_short_quote():
    short = "defines"
    assert len(short) < MIN_QUOTE_CHARS
    report = verify_citations([Citation("argparse#016", short)], [A])
    assert report == CitationReport(False, [], ["argparse#016: defines"])


def test_verify_is_case_sensitive():
    report = verify_citations([Citation("argparse#016", "THE ADD_ARGUMENT method")], [A])
    assert report.valid is False
    assert report.missing_quotes == ["argparse#016: THE ADD_ARGUMENT method"]


def test_verify_reports_unknown_and_missing_separately():
    citations = [
        Citation("zzz#001", "nothing to see"),
        Citation("argparse#016", "not in the chunk"),
        Citation("zzz#001", "again unknown"),
        Citation("argparse#016", "defines how a single argument"),
    ]
    report = verify_citations(citations, [A])
    assert report == CitationReport(
        valid=False,
        unknown_ids=["zzz#001"],
        missing_quotes=["argparse#016: not in the chunk"],
    )


def test_refusal_must_match_exactly():
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal("  " + REFUSAL_TEXT + "\n")
    assert not is_refusal(REFUSAL_TEXT.lower())
    assert not is_refusal(REFUSAL_TEXT.rstrip("."))
    assert not is_refusal("Sorry. " + REFUSAL_TEXT)
    assert not is_refusal("")


def test_refusal_with_extra_words_is_not_refusal():
    assert not is_refusal(REFUSAL_TEXT + " But maybe it is X.")
