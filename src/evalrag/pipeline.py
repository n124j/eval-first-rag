"""End-to-end RAG pipeline: retrieve, prompt, answer, verify citations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from evalrag.config import is_valid_min_dense_score
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


class ScoredRetrieval(Protocol):
    hits: list[Hit]
    top_dense_score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    report: CitationReport
    retrieved_ids: list[str]
    refused: bool
    completion: Completion | None
    # True when the pipeline refused on low retrieval confidence, without calling the LLM.
    abstained: bool = False


def _refusal(retrieved_ids: list[str], abstained: bool) -> Answer:
    return Answer(
        text=REFUSAL_TEXT,
        citations=[],
        report=CitationReport(valid=False, unknown_ids=[], missing_quotes=[]),
        retrieved_ids=retrieved_ids,
        refused=True,
        completion=None,
        abstained=abstained,
    )


class RagPipeline:
    """Answers a question from retrieved chunks.

    With `min_dense_score > 0.0` the pipeline abstains (refuses without calling the LLM) when
    the retriever's dense rank-1 cosine is strictly below the threshold (ADR 0006). The
    retriever must then provide `retrieve_detailed`. 0.0 disables the check.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        chunks_by_id: Mapping[str, Chunk],
        k: int,
        min_dense_score: float = 0.0,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        if not is_valid_min_dense_score(min_dense_score):
            raise ValueError(
                f"min_dense_score must be a finite number in [0.0, 1.0], got {min_dense_score!r}"
            )
        detailed: Callable[[str, int], ScoredRetrieval] | None = None
        if min_dense_score > 0.0:
            detailed = getattr(retriever, "retrieve_detailed", None)
            if not callable(detailed):
                raise TypeError(
                    "a positive min_dense_score needs a retriever with retrieve_detailed(query, k)"
                )
        self._retriever = retriever
        self._llm = llm
        self._chunks_by_id = chunks_by_id
        self._k = k
        self._min_dense_score = float(min_dense_score)
        self._retrieve_detailed = detailed  # None when the check is disabled

    def ask(self, question: str) -> Answer:
        if self._retrieve_detailed is not None:
            result = self._retrieve_detailed(question, self._k)
            hits = result.hits
            if result.top_dense_score < self._min_dense_score:
                return _refusal([h.chunk_id for h in hits], abstained=True)
        else:
            hits = self._retriever.retrieve(question, self._k)
        retrieved_ids = [h.chunk_id for h in hits]
        chunks = []
        for chunk_id in retrieved_ids:
            if chunk_id not in self._chunks_by_id:
                raise KeyError(f"retrieved chunk id not in chunks_by_id: {chunk_id}")
            chunks.append(self._chunks_by_id[chunk_id])
        if not chunks:
            return _refusal([], abstained=False)
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
