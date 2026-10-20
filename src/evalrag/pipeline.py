"""End-to-end RAG pipeline: retrieve, prompt, answer, verify citations."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from evalrag.generation.answer import (
    REFUSAL_TEXT,
    Citation,
    CitationReport,
    build_messages,
    is_refusal,
    parse_citations,
    verify_citations,
)
from evalrag.generation.llm import Completion, LLMClient
from evalrag.models import Chunk, Hit


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Hit]: ...


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    report: CitationReport
    retrieved_ids: list[str]
    refused: bool
    completion: Completion | None


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        chunks_by_id: Mapping[str, Chunk],
        k: int,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        self._retriever = retriever
        self._llm = llm
        self._chunks_by_id = chunks_by_id
        self._k = k

    def ask(self, question: str) -> Answer:
        hits = self._retriever.retrieve(question, self._k)
        retrieved_ids = [h.chunk_id for h in hits]
        chunks = []
        for chunk_id in retrieved_ids:
            if chunk_id not in self._chunks_by_id:
                raise KeyError(f"retrieved chunk id not in chunks_by_id: {chunk_id}")
            chunks.append(self._chunks_by_id[chunk_id])
        if not chunks:
            return Answer(
                text=REFUSAL_TEXT,
                citations=[],
                report=CitationReport(valid=False, unknown_ids=[], missing_quotes=[]),
                retrieved_ids=[],
                refused=True,
                completion=None,
            )
        completion = self._llm.complete(build_messages(question, chunks))
        text = completion.text.strip()
        citations = parse_citations(text)
        return Answer(
            text=text,
            citations=citations,
            report=verify_citations(citations, chunks),
            retrieved_ids=retrieved_ids,
            refused=is_refusal(text),
            completion=completion,
        )
