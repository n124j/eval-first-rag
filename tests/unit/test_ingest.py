import json
from pathlib import Path

import pytest

from evalrag.config import Settings
from evalrag.generation.answer import REFUSAL_TEXT, Citation, CitationReport
from evalrag.generation.llm import Completion
from evalrag.ingest import ingest, load_chunks, open_retriever
from evalrag.models import Chunk
from evalrag.pipeline import Answer
from evalrag.sut import make_sut
from tests.unit.fakes import HashEmbedder, make_corpus


def settings() -> Settings:
    return Settings(
        llm_base_url="x",
        llm_model="m",
        embed_base_url="y",
        embed_model="e",
        chunk_size=12,
        chunk_overlap=2,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return make_corpus(tmp_path)


def test_ingest_builds_files_and_summary(tmp_path: Path, corpus: Path):
    data = tmp_path / "data"
    emb = HashEmbedder()
    summary = ingest(settings(), corpus, data, emb)
    assert summary.documents == 3
    assert summary.dim == 16
    assert summary.chunk_size == 12 and summary.chunk_overlap == 2
    chunks = load_chunks(data / "chunks.jsonl")
    assert summary.chunks == len(chunks) > 3
    for name in ("keyword.sqlite", "dense.npz", "chunks.jsonl", "ingest.json"):
        assert (data / name).exists()
    meta = json.loads((data / "ingest.json").read_text(encoding="utf-8"))
    assert meta["embed_model"] == "fake-embed" and meta["chunks"] == summary.chunks
    assert all(n <= 64 for n in emb.calls)
    assert chunks[0].doc_id == "alpha"


def test_ingest_overwrites_previous_index(tmp_path: Path, corpus: Path):
    data = tmp_path / "data"
    ingest(settings(), corpus, data, HashEmbedder())
    first = len(load_chunks(data / "chunks.jsonl"))
    other = tmp_path / "other"
    other.mkdir()
    sub = make_corpus(other)
    manifest = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))[:1]
    (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = ingest(settings(), sub, data, HashEmbedder())
    assert summary.documents == 1 and summary.chunks < first
    retriever, by_id = open_retriever(settings(), data, HashEmbedder())
    assert len(by_id) == summary.chunks
    assert {h.chunk_id for h in retriever.retrieve("module", 50)} <= set(by_id)


def test_chunks_jsonl_roundtrip_and_bytes_are_stable(tmp_path: Path, corpus: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    ingest(settings(), corpus, a, HashEmbedder())
    ingest(settings(), corpus, b, HashEmbedder())
    for name in ("chunks.jsonl", "ingest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    raw = (a / "chunks.jsonl").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    first = json.loads(raw.decode("utf-8").splitlines()[0])
    assert list(first) == ["chunk_id", "doc_id", "ordinal", "text", "url"]
    chunks = load_chunks(a / "chunks.jsonl")
    assert isinstance(chunks[0], Chunk) and chunks[0].chunk_id == "alpha#000"


def test_load_chunks_malformed_line_names_line_number(tmp_path: Path):
    good = json.dumps({"chunk_id": "a#000", "doc_id": "a", "ordinal": 0, "text": "t", "url": "u"})
    path = tmp_path / "chunks.jsonl"
    path.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunks(path)
    path.write_text(good + "\n" + json.dumps({"chunk_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunks(path)
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        load_chunks(tmp_path / "nope.jsonl")


def test_open_retriever_without_index_names_remedy(tmp_path: Path):
    missing = tmp_path / "empty"
    with pytest.raises(FileNotFoundError) as exc:
        open_retriever(settings(), missing, HashEmbedder())
    assert str(exc.value) == f'no index in {missing}; run "evalrag ingest" first'


def test_make_sut_maps_answer_fields():
    report = CitationReport(valid=True, unknown_ids=[], missing_quotes=[])
    answer = Answer(
        text='yes [a#000: "some quote"]',
        citations=[Citation("a#000", "some quote")],
        report=report,
        retrieved_ids=["a#000", "b#001"],
        refused=False,
        completion=Completion("t", prompt_tokens=11, completion_tokens=5, latency_s=0.25),
    )

    class P:
        def ask(self, q):
            assert q == "q?"
            return answer

    result = make_sut(P())("q?")  # type: ignore[arg-type]
    assert result.answer == answer.text
    assert result.retrieved_ids == ["a#000", "b#001"]
    assert result.cited_ids == ["a#000"]
    assert result.citation_ok is True and result.refused is False
    assert (result.latency_s, result.prompt_tokens, result.completion_tokens) == (0.25, 11, 5)

    empty = Answer(REFUSAL_TEXT, [], CitationReport(False, [], []), [], True, None)

    class Q:
        def ask(self, q):
            return empty

    r2 = make_sut(Q())("q")  # type: ignore[arg-type]
    assert (r2.latency_s, r2.prompt_tokens, r2.completion_tokens) == (0.0, 0, 0)
    assert r2.refused and not r2.citation_ok
