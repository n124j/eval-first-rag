import pytest

from evalrag.generation.answer import (
    REFUSAL_TEXT,
    Citation,
    CitationReport,
    build_messages,
)
from evalrag.generation.llm import Completion
from evalrag.models import Chunk, Hit
from evalrag.pipeline import Answer, RagPipeline

A = Chunk("argparse#016", "argparse", 16, "The add_argument method defines an argument.", "ua")
B = Chunk("os.path#003", "os.path", 3, "os.path.join joins path components.", "ub")
CHUNKS = {A.chunk_id: A, B.chunk_id: B}


class FakeRetriever:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[Hit]:
        self.calls.append((query, k))
        return [Hit(chunk_id=i, score=1.0 / r, rank=r) for r, i in enumerate(self.ids, start=1)]


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.params: dict[str, object] = {}
        self.completion = Completion(text=text, prompt_tokens=11, completion_tokens=7, latency_s=0.5)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        self.calls.append(messages)
        return self.completion


def test_pipeline_ask_wires_retrieval_prompt_and_verification():
    retriever = FakeRetriever(["os.path#003", "argparse#016"])
    llm = FakeLLM('Join with os.path.join [os.path#003: "joins path components"].')
    answer = RagPipeline(retriever, llm, CHUNKS, k=2).ask("How to join?")
    assert retriever.calls == [("How to join?", 2)]
    assert llm.calls == [build_messages("How to join?", [B, A])]
    assert answer.retrieved_ids == ["os.path#003", "argparse#016"]
    assert answer.citations == [Citation("os.path#003", "joins path components")]
    assert answer.report == CitationReport(True, [], [])
    assert answer.refused is False
    assert answer.text == 'Join with os.path.join [os.path#003: "joins path components"].'


def test_refusal_answer_marks_refused_and_skips_citation_requirement():
    llm = FakeLLM(REFUSAL_TEXT)
    answer = RagPipeline(FakeRetriever(["argparse#016"]), llm, CHUNKS, k=1).ask("Unrelated?")
    assert answer.refused is True
    assert answer.citations == []
    assert answer.report == CitationReport(False, [], [])
    assert answer.text == REFUSAL_TEXT


def test_pipeline_unknown_hit_id_raises_KeyError():
    pipeline = RagPipeline(FakeRetriever(["ghost#001"]), FakeLLM("x"), CHUNKS, k=1)
    with pytest.raises(KeyError, match="ghost#001"):
        pipeline.ask("q?")


def test_pipeline_zero_hits_refuses_without_calling_llm():
    llm = FakeLLM("should not be used")
    answer = RagPipeline(FakeRetriever([]), llm, CHUNKS, k=3).ask("q?")
    assert llm.calls == []
    assert answer == Answer(
        text=REFUSAL_TEXT,
        citations=[],
        report=CitationReport(False, [], []),
        retrieved_ids=[],
        refused=True,
        completion=None,
    )


def test_pipeline_passes_completion_through():
    llm = FakeLLM(REFUSAL_TEXT)
    answer = RagPipeline(FakeRetriever(["argparse#016"]), llm, CHUNKS, k=1).ask("q?")
    assert answer.completion is llm.completion


def test_pipeline_strips_answer_text():
    llm = FakeLLM('\n  See [argparse#016: "defines an argument"].  \n')
    answer = RagPipeline(FakeRetriever(["argparse#016"]), llm, CHUNKS, k=1).ask("q?")
    assert answer.text == 'See [argparse#016: "defines an argument"].'
    assert answer.report.valid is True


def test_pipeline_extra_words_with_refusal_text_is_not_refused():
    llm = FakeLLM(REFUSAL_TEXT + " However, maybe.")
    answer = RagPipeline(FakeRetriever(["argparse#016"]), llm, CHUNKS, k=1).ask("q?")
    assert answer.refused is False
    assert answer.report.valid is False


@pytest.mark.parametrize("k", [0, -1])
def test_pipeline_rejects_nonpositive_k(k):
    with pytest.raises(ValueError):
        RagPipeline(FakeRetriever([]), FakeLLM(""), CHUNKS, k=k)
