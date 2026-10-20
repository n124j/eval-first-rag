import sqlite3
from pathlib import Path

import pytest

from evalrag.models import Chunk
from evalrag.retrieval.keyword import KeywordIndex


def make(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], ordinal=0, text=text, url="u")


@pytest.fixture
def index(tmp_path: Path):
    idx = KeywordIndex(tmp_path / "kw.db")
    yield idx
    idx.close()


def test_fts5_is_available():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    except sqlite3.OperationalError as exc:
        pytest.fail(f"This Python's sqlite3 build lacks FTS5: {exc}")
    finally:
        conn.close()


def test_finds_exact_term(index):
    index.add(
        [make("a#000", "argparse builds command line parsers"), make("b#000", "lists and tuples")]
    )
    hits = index.search("argparse", 5)
    assert [h.chunk_id for h in hits] == ["a#000"]


def test_ranks_more_relevant_chunk_first(index):
    index.add(
        [
            make("a#000", "the cat sat on the mat with many other unrelated words here today"),
            make("b#000", "cat cat cat"),
            make("c#000", "dogs only"),
        ]
    )
    hits = index.search("cat", 5)
    assert [h.chunk_id for h in hits] == ["b#000", "a#000"]


@pytest.mark.parametrize("query", ['"unterminated', "a AND OR NEAR(*", "foo*bar"])
def test_special_characters_do_not_raise(index, query):
    index.add([make("a#000", "foo bar a near")])
    index.search(query, 5)


@pytest.mark.parametrize("query", ["", "   ", "\t\n", "!!! ???"])
def test_empty_and_whitespace_query_return_empty(index, query):
    index.add([make("a#000", "something")])
    assert index.search(query, 5) == []


def test_k_limits_results(index):
    index.add([make(f"d#{i:03d}", "shared term") for i in range(10)])
    assert len(index.search("shared", 3)) == 3


def test_persists_to_disk(tmp_path):
    path = tmp_path / "kw.db"
    first = KeywordIndex(path)
    first.add([make("a#000", "persistent text")])
    first.close()
    second = KeywordIndex(path)
    try:
        assert second.count() == 1
        assert [h.chunk_id for h in second.search("persistent", 5)] == ["a#000"]
    finally:
        second.close()


def test_stemming_matches_plural_and_singular(index):
    index.add([make("a#000", "writing custom parsers")])
    assert [h.chunk_id for h in index.search("parser", 5)] == ["a#000"]


def test_duplicate_chunk_id_raises_and_adds_nothing(index):
    index.add([make("a#000", "first")])
    with pytest.raises(ValueError, match="a#000"):
        index.add([make("b#000", "new"), make("a#000", "dup")])
    assert index.count() == 1
    assert index.search("new", 5) == []


def test_duplicate_within_one_call_raises(index):
    with pytest.raises(ValueError, match="a#000"):
        index.add([make("a#000", "x"), make("a#000", "y")])
    assert index.count() == 0


def test_nonpositive_k_raises(index):
    with pytest.raises(ValueError):
        index.search("x", 0)
    with pytest.raises(ValueError):
        index.search("x", -1)


def test_scores_non_increasing_and_ranks_start_at_one(index):
    index.add(
        [
            make("a#000", "alpha beta gamma delta"),
            make("b#000", "alpha alpha beta"),
            make("c#000", "alpha"),
        ]
    )
    hits = index.search("alpha beta", 5)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_ordering_is_deterministic_on_ties(index):
    index.add([make(f"d#{i:03d}", "same words") for i in (3, 1, 2, 0)])
    ids = [h.chunk_id for h in index.search("same", 10)]
    assert ids == ["d#000", "d#001", "d#002", "d#003"]


def test_operator_words_are_treated_as_plain_terms(index):
    index.add([make("a#000", "this is NOT near AND ok"), make("b#000", "unrelated")])
    hits = index.search("NOT near AND", 5)
    assert [h.chunk_id for h in hits] == ["a#000"]


def test_unicode_query(index):
    index.add([make("a#000", "café über naïve"), make("b#000", "plain")])
    assert [h.chunk_id for h in index.search("Café", 5)] == ["a#000"]
