import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from evalrag.retrieval.embedder import (
    EmbeddingError,
    OpenAICompatEmbedder,
    RecordingEmbedder,
    ReplayEmbedder,
    ReplayMiss,
)


def fake_vector(text: str, dim: int = 4) -> list[float]:
    seed = sum(ord(c) for c in text) % 97
    return [float(seed + i) for i in range(dim)]


def make_transport(requests: list[dict], shuffle: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"url": str(request.url), "body": body})
        items = [{"index": i, "embedding": fake_vector(t)} for i, t in enumerate(body["input"])]
        if shuffle:
            items.reverse()
        return httpx.Response(200, json={"data": items})

    return httpx.MockTransport(handler)


def make_embedder(requests: list[dict], shuffle: bool = False, **kwargs) -> OpenAICompatEmbedder:
    client = httpx.Client(transport=make_transport(requests, shuffle))
    return OpenAICompatEmbedder("http://lm.test/v1", "nomic", client=client, **kwargs)


def fixed_embedder(handler) -> OpenAICompatEmbedder:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatEmbedder("http://lm.test/v1", "m", client=client)


def test_openai_compat_posts_and_parses():
    requests: list[dict] = []
    out = make_embedder(requests, shuffle=True).embed(["alpha", "beta"])
    assert out.dtype == np.float32
    assert out.shape == (2, 4)
    assert requests[0]["url"] == "http://lm.test/v1/embeddings"
    assert requests[0]["body"]["model"] == "nomic"
    np.testing.assert_allclose(out[0], fake_vector("search_document: alpha"))
    np.testing.assert_allclose(out[1], fake_vector("search_document: beta"))


def test_batches_over_32():
    requests: list[dict] = []
    out = make_embedder(requests).embed([f"t{i}" for i in range(70)])
    assert out.shape == (70, 4)
    assert [len(r["body"]["input"]) for r in requests] == [32, 32, 6]


def test_query_prefix_differs_from_document_prefix():
    requests: list[dict] = []
    emb = make_embedder(requests)
    emb.embed(["x"], kind="document")
    emb.embed(["x"], kind="query")
    assert requests[0]["body"]["input"] == ["search_document: x"]
    assert requests[1]["body"]["input"] == ["search_query: x"]


def test_empty_prefixes_disable_prefixing():
    requests: list[dict] = []
    emb = make_embedder(requests, document_prefix="", query_prefix="")
    emb.embed(["x"])
    emb.embed(["y"], kind="query")
    assert [r["body"]["input"] for r in requests] == [["x"], ["y"]]


def test_empty_input_embeds_to_empty_array():
    requests: list[dict] = []
    out = make_embedder(requests).embed([])
    assert out.shape == (0, 0)
    assert out.dtype == np.float32
    assert requests == []


def test_http_error_raises_EmbeddingError_with_status():
    emb = fixed_embedder(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(EmbeddingError, match="500.*boom"):
        emb.embed(["a"])


def test_malformed_response_raises_EmbeddingError():
    emb = fixed_embedder(lambda r: httpx.Response(200, json={"nope": 1}))
    with pytest.raises(EmbeddingError):
        emb.embed(["a"])


def test_wrong_vector_count_raises():
    emb = fixed_embedder(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})
    )
    with pytest.raises(EmbeddingError):
        emb.embed(["a", "b"])


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts, kind="document"):
        self.calls.append((list(texts), kind))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.array([fake_vector(t) for t in texts], dtype=np.float32)


def test_recording_then_replay_roundtrip(tmp_path: Path):
    path = tmp_path / "cache.npz"
    rec = RecordingEmbedder(CountingEmbedder(), path, model="nomic")
    live = rec.embed(["a", "b"])
    rec.embed(["a"], kind="query")
    rec.flush()
    replay = ReplayEmbedder(path)
    got = replay.embed(["a", "b"])
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, live, rtol=1e-2)
    assert replay.model == "nomic"
    assert replay.dim == 4
    assert replay.embed(["a"], kind="query").shape == (1, 4)
    assert replay.embed([]).shape == (0, 0)


def test_recording_reuses_cached_vectors_without_calling_inner(tmp_path: Path):
    path = tmp_path / "cache.npz"
    rec = RecordingEmbedder(CountingEmbedder(), path)
    rec.embed(["a", "b"])
    rec.flush()
    second = CountingEmbedder()
    rec2 = RecordingEmbedder(second, path)
    rec2.embed(["a", "b"])
    assert second.calls == []
    rec2.embed(["a", "c"])
    assert second.calls == [(["c"], "document")]
    rec2.flush()
    assert ReplayEmbedder(path).embed(["c"]).shape == (1, 4)


def test_replay_miss_names_key_and_never_calls_inner(tmp_path: Path):
    path = tmp_path / "cache.npz"
    inner = CountingEmbedder()
    rec = RecordingEmbedder(inner, path)
    rec.embed(["seen"])
    rec.flush()
    inner.calls.clear()
    with pytest.raises(ReplayMiss):
        ReplayEmbedder(path).embed(["unseen"])
    assert inner.calls == []


def test_replay_miss_message_has_key_and_text_prefix(tmp_path: Path):
    path = tmp_path / "cache.npz"
    rec = RecordingEmbedder(CountingEmbedder(), path)
    rec.embed(["seen"])
    rec.flush()
    text = "z" * 100
    with pytest.raises(ReplayMiss) as info:
        ReplayEmbedder(path).embed([text], kind="query")
    key = hashlib.sha256(f"query\n{text}".encode()).hexdigest()
    msg = str(info.value)
    assert key in msg
    assert "z" * 60 in msg
    assert "z" * 61 not in msg


def test_replay_never_calls_network(tmp_path: Path, monkeypatch):
    path = tmp_path / "cache.npz"
    rec = RecordingEmbedder(CountingEmbedder(), path)
    rec.embed(["a"])
    rec.flush()

    def boom(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(httpx.Client, "send", boom)
    assert ReplayEmbedder(path).embed(["a"]).shape == (1, 4)
