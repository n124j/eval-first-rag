"""Acceptance tests: the dev (tuning) set is well-formed and disjoint from the golden set."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from evalrag.evals.dataset import load_golden

ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = ROOT / "evals" / "dev"
DEV = DEV_DIR / "dev.jsonl"
DEV_ABSENT = DEV_DIR / "absent_terms.json"
GOLDEN_DIR = ROOT / "evals" / "golden"
GOLDEN = GOLDEN_DIR / "golden.jsonl"
GOLDEN_ABSENT = GOLDEN_DIR / "absent_terms.json"
CORPUS = ROOT / "corpus"
META_WORDS = ("corpus", "document", "chunk", "page", "context")
CANONICAL_KEYS = ["id", "question", "expected_sources", "answerable", "kind"]
JACCARD_LIMIT = 0.5


def _doc_ids() -> set[str]:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    return {entry["doc_id"] for entry in manifest}


def _dev_items():
    return load_golden(DEV, known_sources=_doc_ids())


def _golden_items():
    return load_golden(GOLDEN, known_sources=_doc_ids())


def _corpus_texts() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8").casefold() for p in sorted(CORPUS.glob("*.txt"))}


def _normalise(question: str) -> str:
    return " ".join(question.split()).casefold()


def _tokens(question: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.]+", question.casefold()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def test_dev_loads_with_corpus_sources():
    items = _dev_items()
    assert items


def test_dev_counts_and_kinds():
    items = _dev_items()
    counts = Counter(item.kind for item in items)
    assert counts == {"single": 10, "multi": 2, "unanswerable": 8}, counts
    assert len(items) == 20
    for item in items:
        if item.kind == "multi":
            assert len(item.expected_sources) == 2, item.id
            assert len(set(item.expected_sources)) == 2, item.id
        elif item.kind == "single":
            assert len(item.expected_sources) == 1, item.id
        else:
            assert item.expected_sources == () and not item.answerable, item.id


def test_ids_are_d001_to_d020_in_order():
    ids = [item.id for item in _dev_items()]
    assert ids == [f"d{n:03d}" for n in range(1, 21)]


def test_dev_questions_are_disjoint_from_golden():
    golden = _golden_items()
    golden_keys = {_normalise(item.question) for item in golden}
    for item in _dev_items():
        assert _normalise(item.question) not in golden_keys, item.id
        for gold in golden:
            score = _jaccard(item.question, gold.question)
            assert score < JACCARD_LIMIT, f"{item.id} vs {gold.id}: jaccard {score:.2f}"


def test_dev_unanswerable_absent_terms_are_absent_from_corpus():
    absent = json.loads(DEV_ABSENT.read_text(encoding="utf-8"))
    unanswerable = {item.id for item in _dev_items() if item.kind == "unanswerable"}
    assert set(absent) == unanswerable
    texts = _corpus_texts()
    assert len(texts) == len(_doc_ids())
    for ident, terms in absent.items():
        assert 2 <= len(terms) <= 6, ident
        for term in terms:
            assert term.strip(), ident
            assert term == term.casefold(), f"{ident}: term {term!r} is not lowercase"
            hits = [name for name, text in texts.items() if term.casefold() in text]
            assert not hits, f"{ident}: term {term!r} occurs in {hits}"


def test_dev_questions_avoid_meta_words():
    pattern = re.compile(r"\b(" + "|".join(META_WORDS) + r")s?\b", re.IGNORECASE)
    offenders = [item.id for item in _dev_items() if pattern.search(item.question)]
    assert not offenders


def test_dev_file_is_utf8_lf_with_trailing_newline():
    data = DEV.read_bytes()
    data.decode("utf-8")
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")


def test_dev_file_keys_in_canonical_order():
    for line in DEV.read_text(encoding="utf-8").splitlines():
        assert list(json.loads(line)) == CANONICAL_KEYS


def test_dev_unanswerable_topics_differ_from_golden_unanswerable():
    dev_terms = {t for terms in json.loads(DEV_ABSENT.read_text(encoding="utf-8")).values()
                 for t in terms}
    golden_terms = {t for terms in json.loads(GOLDEN_ABSENT.read_text(encoding="utf-8")).values()
                    for t in terms}
    assert dev_terms, "dev absent terms are empty"
    overlap = dev_terms & golden_terms
    assert not overlap, sorted(overlap)
