"""Acceptance tests: the golden evaluation set is well-formed and honest about the corpus."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from evalrag.evals.dataset import load_golden

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "evals" / "golden"
GOLDEN = GOLDEN_DIR / "golden.jsonl"
ABSENT = GOLDEN_DIR / "absent_terms.json"
NOTES = GOLDEN_DIR / "review-notes.md"
CORPUS = ROOT / "corpus"
META_WORDS = ("corpus", "document", "chunk", "page", "context")
CANONICAL_KEYS = ["id", "question", "expected_sources", "answerable", "kind"]


def _doc_ids() -> set[str]:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    return {entry["doc_id"] for entry in manifest}


def _items():
    return load_golden(GOLDEN, known_sources=_doc_ids())


def _corpus_texts() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8").casefold() for p in sorted(CORPUS.glob("*.txt"))}


def test_golden_loads_with_corpus_sources():
    items = _items()
    assert items


def test_golden_counts_and_kinds():
    items = _items()
    counts = Counter(item.kind for item in items)
    assert abs(counts["single"] - 40) <= 2, counts
    assert abs(counts["multi"] - 10) <= 2, counts
    assert abs(counts["unanswerable"] - 10) <= 2, counts
    assert 56 <= len(items) <= 64


def test_ids_are_sequential_and_unique():
    ids = [item.id for item in _items()]
    assert len(ids) == len(set(ids))
    assert ids == [f"g{n:03d}" for n in range(1, len(ids) + 1)]


def test_unanswerable_items_have_no_expected_sources():
    for item in _items():
        if item.kind == "unanswerable":
            assert not item.answerable, item.id
            assert item.expected_sources == (), item.id


def test_answerable_items_have_expected_sources_matching_kind():
    for item in _items():
        if item.kind == "single":
            assert item.answerable and len(item.expected_sources) == 1, item.id
        elif item.kind == "multi":
            assert item.answerable, item.id
            assert len(item.expected_sources) == 2, item.id
            assert len(set(item.expected_sources)) == 2, item.id


def test_covers_at_least_30_distinct_docs():
    docs = {src for item in _items() if item.answerable for src in item.expected_sources}
    assert len(docs) >= 30, sorted(docs)


def test_no_doc_has_more_than_3_single_questions():
    counts = Counter(item.expected_sources[0] for item in _items() if item.kind == "single")
    over = {doc: n for doc, n in counts.items() if n > 3}
    assert not over, over


def test_unanswerable_absent_terms_are_absent_from_corpus():
    absent = json.loads(ABSENT.read_text(encoding="utf-8"))
    unanswerable = {item.id for item in _items() if item.kind == "unanswerable"}
    assert set(absent) == unanswerable
    texts = _corpus_texts()
    assert len(texts) == len(_doc_ids())
    for ident, terms in absent.items():
        assert 2 <= len(terms) <= 6, ident
        for term in terms:
            assert term.strip(), ident
            hits = [name for name, text in texts.items() if term.casefold() in text]
            assert not hits, f"{ident}: term {term!r} occurs in {hits}"


def test_questions_avoid_meta_words():
    pattern = re.compile(r"\b(" + "|".join(META_WORDS) + r")s?\b", re.IGNORECASE)
    offenders = [item.id for item in _items() if pattern.search(item.question)]
    assert not offenders


def test_review_notes_has_an_entry_for_every_item():
    notes = NOTES.read_text(encoding="utf-8")
    missing = [item.id for item in _items() if not re.search(rf"\b{item.id}\b", notes)]
    assert not missing


def test_golden_file_is_utf8_lf_with_trailing_newline():
    data = GOLDEN.read_bytes()
    data.decode("utf-8")
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")


def test_golden_file_keys_in_canonical_order():
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        assert list(json.loads(line)) == CANONICAL_KEYS
