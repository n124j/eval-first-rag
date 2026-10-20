"""Adapt a RagPipeline to the eval harness's system-under-test callable."""

from evalrag.evals.runner import SutResult, SystemUnderTest
from evalrag.pipeline import RagPipeline


def make_sut(pipeline: RagPipeline) -> SystemUnderTest:
    def sut(question: str) -> SutResult:
        answer = pipeline.ask(question)
        completion = answer.completion
        return SutResult(
            answer=answer.text,
            retrieved_ids=answer.retrieved_ids,
            cited_ids=[c.chunk_id for c in answer.citations],
            citation_ok=answer.report.valid,
            refused=answer.refused,
            latency_s=completion.latency_s if completion else 0.0,
            prompt_tokens=completion.prompt_tokens if completion else 0,
            completion_tokens=completion.completion_tokens if completion else 0,
        )

    return sut
