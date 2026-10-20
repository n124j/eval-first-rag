import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from evalrag.api import app_from_env, create_app
from evalrag.cli import main
from evalrag.generation.answer import REFUSAL_TEXT, Citation, CitationReport
from evalrag.generation.llm import LLMError
from evalrag.pipeline import Answer
from evalrag.retrieval.dense import EmptyIndex
from evalrag.retrieval.embedder import EmbeddingError
from tests.unit.fakes import make_corpus
from tests.unit.test_cli import ENV, FakeFactory, QuestionRecordingFactory

QUOTED = 'Cited. [a#000: "quote"]'


def _answer(text=QUOTED, refused=False, citations=None, valid=True):
    return Answer(
        text=text,
        citations=[Citation("a#000", "quote")] if citations is None else citations,
        report=CitationReport(valid=valid, unknown_ids=[], missing_quotes=[]),
        retrieved_ids=["a#000", "b#001"],
        refused=refused,
        completion=None,
    )


class StubPipeline:
    def __init__(self, result=None, error=None):
        self.result = result or _answer()
        self.error = error
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return self.result


def test_health():
    client = TestClient(create_app(StubPipeline()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_citations_and_shape():
    pipeline = StubPipeline()
    client = TestClient(create_app(pipeline))
    response = client.post("/ask", json={"question": "  What is a?  "})
    assert response.status_code == 200
    assert response.json() == {
        "answer": QUOTED,
        "citations": [{"chunk_id": "a#000", "quote": "quote"}],
        "refused": False,
        "citations_valid": True,
        "retrieved_ids": ["a#000", "b#001"],
        "abstained": False,
    }
    assert pipeline.questions == ["What is a?"]


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_whitespace_question_is_422(question):
    pipeline = StubPipeline()
    client = TestClient(create_app(pipeline))
    assert client.post("/ask", json={"question": question}).status_code == 422
    assert pipeline.questions == []


def test_overlong_question_is_422():
    client = TestClient(create_app(StubPipeline()))
    assert client.post("/ask", json={"question": "x" * 2000}).status_code == 200
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422


def test_missing_body_and_wrong_types_are_422():
    client = TestClient(create_app(StubPipeline()))
    assert client.post("/ask").status_code == 422
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": 5}).status_code == 422
    assert client.post("/ask", json={"question": None}).status_code == 422
    assert client.post("/ask", json=["q"]).status_code == 422
    assert client.post("/ask", content=b"not json").status_code == 422


def test_refusal_response_shape():
    refusal = _answer(text=REFUSAL_TEXT, refused=True, citations=[], valid=False)
    client = TestClient(create_app(StubPipeline(refusal)))
    body = client.post("/ask", json={"question": "weather?"}).json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert body["answer"] == REFUSAL_TEXT
    assert body["citations_valid"] is False


def test_llm_error_maps_to_502_with_message(caplog):
    client = TestClient(create_app(StubPipeline(error=LLMError("llm down at secret-host"))))
    with caplog.at_level(logging.WARNING, logger="evalrag.api"):
        response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert response.json() == {"detail": "language model service error"}
    assert "secret-host" not in response.text
    assert "llm down at secret-host" in caplog.text


def test_embedding_error_maps_to_502(caplog):
    client = TestClient(create_app(StubPipeline(error=EmbeddingError("embed down at secret-host"))))
    with caplog.at_level(logging.WARNING, logger="evalrag.api"):
        response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert response.json() == {"detail": "embedding service error"}
    assert "secret-host" not in response.text
    assert "embed down at secret-host" in caplog.text


def test_empty_index_maps_to_503():
    client = TestClient(create_app(StubPipeline(error=EmptyIndex("nothing"))))
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert response.json() == {"detail": "index is empty"}


def test_other_exceptions_propagate():
    app = create_app(StubPipeline(error=RuntimeError("bug")))
    client = TestClient(app, raise_server_exceptions=True)
    with pytest.raises(RuntimeError, match="bug"):
        client.post("/ask", json={"question": "q"})


def test_requests_are_serialised_by_lock():
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    class Overlapping:
        def ask(self, question):
            with guard:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.02)
            with guard:
                state["active"] -= 1
            return _answer()

    client = TestClient(create_app(Overlapping()))
    statuses = []

    def fire():
        statuses.append(client.post("/ask", json={"question": "q"}).status_code)

    threads = [threading.Thread(target=fire) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses == [200] * 8
    assert state["max"] == 1


def test_close_called_once_on_shutdown():
    calls = []
    app = create_app(StubPipeline(), close=lambda: calls.append(1))
    with TestClient(app) as client:
        client.get("/health")
        assert calls == []
    assert calls == [1]


def test_close_failure_does_not_break_shutdown():
    calls = []

    def bad_close():
        calls.append(1)
        raise OSError("close failed")

    app = create_app(StubPipeline(), close=bad_close)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    assert calls == [1]


def _tiny_index(tmp_path):
    corpus = make_corpus(tmp_path)
    data = tmp_path / "data"
    argv = ["ingest", "--corpus", str(corpus), "--data-dir", str(data)]
    assert main(argv, env=ENV, factory=FakeFactory()) == 0
    return data


def test_app_from_env_serves_ask_from_a_real_tiny_index(tmp_path):
    data = _tiny_index(tmp_path)
    app = app_from_env(env={**ENV, "DATA_DIR": str(data)}, factory=FakeFactory())
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "How does the alpha module sort lists?"})
        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is False and body["citations_valid"] is True
        assert body["citations"] and body["retrieved_ids"]

        other = {}

        def second():
            other["response"] = client.post("/ask", json={"question": "What do beta and gamma do?"})

        thread = threading.Thread(target=second)
        thread.start()
        thread.join()
        assert other["response"].status_code == 200


def test_app_from_env_without_index_names_remedy(tmp_path):
    missing = tmp_path / "none"
    with pytest.raises(FileNotFoundError) as exc:
        app_from_env(env={**ENV, "DATA_DIR": str(missing)}, factory=FakeFactory())
    assert str(exc.value) == f'no index in {missing}; run "evalrag ingest" first'


def test_openapi_lists_routes_and_schemas():
    schema = TestClient(create_app(StubPipeline())).get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/ask"}
    assert "post" in schema["paths"]["/ask"] and "get" in schema["paths"]["/health"]
    assert {"AskRequest", "AskResponse"} <= set(schema["components"]["schemas"])
    response = schema["components"]["schemas"]["AskResponse"]
    assert (
        set(response["properties"])
        == set(response["required"])
        == {
            "answer",
            "citations",
            "refused",
            "citations_valid",
            "retrieved_ids",
            "abstained",
        }
    )
    assert response["properties"]["abstained"]["type"] == "boolean"


def test_app_from_env_passes_min_dense_score_and_abstains_without_llm(tmp_path):
    data = _tiny_index(tmp_path)
    factory = QuestionRecordingFactory()
    env = {**ENV, "DATA_DIR": str(data), "MIN_DENSE_SCORE": "1"}
    app = app_from_env(env=env, factory=factory)
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "How does the alpha module sort lists?"})
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True and body["answer"] == REFUSAL_TEXT
    assert body["citations"] == [] and body["retrieved_ids"]
    assert body["abstained"] is True
    assert factory.llm_instance.questions == []


def test_abstained_answer_response_is_exact():
    abstained = Answer(
        REFUSAL_TEXT, [], CitationReport(False, [], []), ["a#000"], True, None, abstained=True
    )
    body = TestClient(create_app(StubPipeline(abstained))).post("/ask", json={"question": "q"})
    assert body.json() == {
        "answer": REFUSAL_TEXT,
        "citations": [],
        "refused": True,
        "citations_valid": False,
        "retrieved_ids": ["a#000"],
        "abstained": True,
    }


def test_answered_question_reports_not_abstained(tmp_path):
    data = _tiny_index(tmp_path)
    app = app_from_env(env={**ENV, "DATA_DIR": str(data)}, factory=FakeFactory())
    with TestClient(app) as client:
        body = client.post("/ask", json={"question": "How does the alpha module sort lists?"})
    assert body.json()["abstained"] is False and body.json()["refused"] is False
