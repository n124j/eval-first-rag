"""HTTP API: `GET /health` and `POST /ask` over a `RagPipeline`.

Run with `uvicorn --factory evalrag.api:app_from_env`. The API is unauthenticated, has no
CORS, and is for local use only.
"""

import contextlib
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from evalrag.cli import ComponentFactory
from evalrag.config import Settings
from evalrag.generation.llm import LLMError
from evalrag.ingest import open_retriever
from evalrag.pipeline import RagPipeline
from evalrag.retrieval.dense import EmptyIndex
from evalrag.retrieval.embedder import EmbeddingError

MAX_QUESTION_CHARS = 2000
log = logging.getLogger("evalrag.api")


class AskRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]


class CitationOut(BaseModel):
    chunk_id: str
    quote: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    refused: bool
    citations_valid: bool
    retrieved_ids: list[str]


class HealthResponse(BaseModel):
    status: str


def create_app(pipeline: RagPipeline, close: Callable[[], None] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if close is not None:
                try:
                    close()
                except Exception:
                    log.exception("error while closing the pipeline")

    app = FastAPI(title="eval-first-rag", lifespan=lifespan)
    lock = threading.Lock()

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        try:
            with lock:
                answer = pipeline.ask(request.question)
        except LLMError as exc:
            log.warning("language model service error: %s", exc)
            raise HTTPException(status_code=502, detail="language model service error") from exc
        except EmbeddingError as exc:
            log.warning("embedding service error: %s", exc)
            raise HTTPException(status_code=502, detail="embedding service error") from exc
        except EmptyIndex as exc:
            raise HTTPException(status_code=503, detail="index is empty") from exc
        return AskResponse(
            answer=answer.text,
            citations=[CitationOut(chunk_id=c.chunk_id, quote=c.quote) for c in answer.citations],
            refused=answer.refused,
            citations_valid=answer.report.valid,
            retrieved_ids=list(answer.retrieved_ids),
        )

    return app


def app_from_env(
    env: Mapping[str, str] | None = None, factory: ComponentFactory | None = None
) -> FastAPI:
    """uvicorn app factory: live components, index from DATA_DIR."""
    env = os.environ if env is None else env
    factory = factory or ComponentFactory()
    settings = Settings.from_env(env)
    embedder = factory.embedder(settings, None, None)
    llm = factory.llm(settings, None, None)
    retriever, chunks_by_id = open_retriever(
        settings, settings.data_dir, embedder, check_same_thread=False
    )
    try:
        pipeline = RagPipeline(retriever, llm, chunks_by_id, settings.top_k)
        return create_app(pipeline, close=retriever.close)
    except BaseException:
        with contextlib.suppress(Exception):
            retriever.close()
        raise
