import pytest

from evalrag.chunking import chunk_document
from evalrag.models import Document


def make_doc(text: str, doc_id: str = "doc-a", url: str = "https://example.test/a") -> Document:
    return Document(doc_id=doc_id, title="T", url=url, version="1", text=text)


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_chunk_ids_are_stable_and_ordered():
    chunks = chunk_document(make_doc(words(30)), size=10, overlap=0)
    assert [c.chunk_id for c in chunks] == ["doc-a#000", "doc-a#001", "doc-a#002"]
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    again = chunk_document(make_doc(words(30)), size=10, overlap=0)
    assert chunks == again


def test_overlap_repeats_exact_words():
    chunks = chunk_document(make_doc(words(10)), size=5, overlap=2)
    assert len(chunks) == 3
    parts = [c.text.split(" ") for c in chunks]
    assert parts[0] == ["w0", "w1", "w2", "w3", "w4"]
    assert parts[1] == ["w3", "w4", "w5", "w6", "w7"]
    assert parts[2] == ["w6", "w7", "w8", "w9"]
    assert parts[0][-2:] == parts[1][:2]
    assert parts[1][-2:] == parts[2][:2]


def test_empty_text_yields_no_chunks():
    assert chunk_document(make_doc(""), size=5, overlap=2) == []
    assert chunk_document(make_doc("   \n\t "), size=5, overlap=2) == []


def test_invalid_overlap_raises_ValueError():
    with pytest.raises(ValueError):
        chunk_document(make_doc(words(10)), size=5, overlap=5)
    with pytest.raises(ValueError):
        chunk_document(make_doc(words(10)), size=5, overlap=-1)
    with pytest.raises(ValueError):
        chunk_document(make_doc(words(10)), size=0, overlap=0)


def test_no_redundant_trailing_chunk():
    # 8 words, size 5, overlap 2: windows start 0, 3; a window at 6 would be
    # w6 w7 only, adding nothing new beyond w7 -> not emitted... but start 3
    # already covers through w7, so exactly 2 chunks.
    chunks = chunk_document(make_doc(words(8)), size=5, overlap=2)
    assert len(chunks) == 2
    assert chunks[-1].text.split(" ")[-1] == "w7"
    # Exact fit: 5 words, size 5 -> one chunk only.
    assert len(chunk_document(make_doc(words(5)), size=5, overlap=2)) == 1


def test_chunk_url_and_doc_id_copied():
    doc = make_doc(words(12), doc_id="kb-7", url="https://example.test/kb-7")
    chunks = chunk_document(doc, size=5, overlap=1)
    assert chunks
    assert all(c.doc_id == "kb-7" for c in chunks)
    assert all(c.url == "https://example.test/kb-7" for c in chunks)
    assert all(c.chunk_id == f"kb-7#{c.ordinal:03d}" for c in chunks)


def test_whitespace_normalised_to_single_spaces():
    chunks = chunk_document(make_doc("a  b\n\nc\td e"), size=10, overlap=0)
    assert chunks[0].text == "a b c d e"
