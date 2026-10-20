import json
from pathlib import Path

import httpx
import pytest

from evalrag.generation import llm as llm_module
from evalrag.generation.llm import (
    Completion,
    LLMError,
    OpenAICompatClient,
    RecordingClient,
    ReplayClient,
    request_key,
)
from evalrag.retrieval.embedder import ReplayMiss

MSGS = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is the capital of France?"},
]


def ok_body(text: str = "Paris", usage: object = "default") -> dict:
    body: dict = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage == "default":
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 3}
    elif usage is not None:
        body["usage"] = usage
    return body


def make_client(handler, **kwargs) -> OpenAICompatClient:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatClient("http://lm.test/v1/", "m1", client=client, **kwargs)


class FakeInner:
    def __init__(self, params: dict | None = None) -> None:
        self.params: dict[str, object] = params or {
            "model": "m1",
            "temperature": 0.0,
            "max_tokens": 512,
            "seed": None,
        }
        self.calls = 0

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        self.calls += 1
        return Completion(f"echo:{messages[-1]['content']}", 5, 2, 0.25)


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_parses_text_and_usage():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=ok_body())

    c = make_client(handler)
    out = c.complete(MSGS)
    assert out.text == "Paris"
    assert (out.prompt_tokens, out.completion_tokens) == (11, 3)
    assert out.latency_s >= 0.0
    assert str(seen[0].url) == "http://lm.test/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body == {
        "model": "m1",
        "messages": MSGS,
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }
    assert c.params == {"model": "m1", "temperature": 0.0, "max_tokens": 512, "seed": None}


def test_request_includes_seed_only_when_set():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    make_client(handler).complete(MSGS)
    make_client(handler, seed=7).complete(MSGS)
    assert "seed" not in bodies[0]
    assert bodies[1]["seed"] == 7


def test_missing_usage_defaults_to_zero():
    c = make_client(lambda r: httpx.Response(200, json=ok_body(usage=None)))
    out = c.complete(MSGS)
    assert (out.prompt_tokens, out.completion_tokens) == (0, 0)
    c = make_client(lambda r: httpx.Response(200, json=ok_body(usage={"prompt_tokens": 4})))
    out = c.complete(MSGS)
    assert (out.prompt_tokens, out.completion_tokens) == (4, 0)


def test_non_string_content_raises_LLMError():
    body = {"choices": [{"message": {"content": None}}]}
    with pytest.raises(LLMError):
        make_client(lambda r: httpx.Response(200, json=body)).complete(MSGS)


def test_no_choices_raises_LLMError():
    for body in ({}, {"choices": []}):
        with pytest.raises(LLMError):
            make_client(lambda r, b=body: httpx.Response(200, json=b)).complete(MSGS)


def test_invalid_json_raises_LLMError():
    with pytest.raises(LLMError):
        make_client(lambda r: httpx.Response(200, content=b"not json")).complete(MSGS)


def test_http_error_raises_LLMError_with_status():
    c = make_client(lambda r: httpx.Response(503, text="model not loaded"))
    with pytest.raises(LLMError, match="503") as info:
        c.complete(MSGS)
    assert "model not loaded" in str(info.value)


def test_transport_error_raises_LLMError():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError):
        make_client(handler).complete(MSGS)


def test_request_key_is_order_insensitive_for_dict_keys():
    a = [{"role": "user", "content": "hi"}]
    b = [{"content": "hi", "role": "user"}]
    assert request_key("m", a, 0.0, 10) == request_key("m", b, 0.0, 10)


def test_request_key_ignores_dict_key_order():
    a = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    b = [{"content": "s", "role": "system"}, {"content": "hi", "role": "user"}]
    assert request_key("m", a, 0.0, 10, 3) == request_key("m", b, 0.0, 10, 3)


def test_request_key_changes_with_each_parameter():
    base = request_key("m", user("hi"), 0.0, 10, None)
    variants = [
        request_key("m2", user("hi"), 0.0, 10, None),
        request_key("m", user("hi"), 0.5, 10, None),
        request_key("m", user("hi"), 0.0, 11, None),
        request_key("m", user("hi"), 0.0, 10, 1),
        request_key("m", user("hello"), 0.0, 10, None),
    ]
    assert len({base, *variants}) == 6
    assert len(base) == 64


def test_record_then_replay_returns_identical_completion(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    first = rec.complete(user("a"))
    rec.flush()
    replay = ReplayClient(path)
    assert replay.complete(user("a")) == first
    assert replay.params == rec.params


def test_replay_returns_recorded_latency(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    rec.complete(user("a"))
    rec.flush()
    assert ReplayClient(path).complete(user("a")).latency_s == 0.25


def test_replay_miss_never_calls_inner(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    inner = FakeInner()
    rec = RecordingClient(inner, path)
    rec.complete(user("a"))
    rec.flush()
    calls = inner.calls
    with pytest.raises(ReplayMiss):
        ReplayClient(path).complete(user("unseen"))
    assert inner.calls == calls


def test_replay_miss_message_has_key_and_prompt_prefix(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    rec.complete(user("a"))
    rec.flush()
    long_prompt = "Q" * 200
    with pytest.raises(ReplayMiss) as info:
        ReplayClient(path).complete(user(long_prompt))
    msg = str(info.value)
    key = request_key("m1", user(long_prompt), 0.0, 512, None)
    assert key in msg
    assert "Q" * 80 in msg
    assert "Q" * 81 not in msg


def test_recording_reuses_cached_entries_without_calling_inner(tmp_path: Path):
    inner = FakeInner()
    rec = RecordingClient(inner, tmp_path / "llm.jsonl")
    one = rec.complete(user("a"))
    two = rec.complete(user("a"))
    assert one == two
    assert inner.calls == 1


def test_recording_extends_existing_cache_and_reuses_entries(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    first = RecordingClient(FakeInner(), path)
    first.complete(user("a"))
    first.flush()
    inner = FakeInner()
    second = RecordingClient(inner, path)
    second.complete(user("a"))
    assert inner.calls == 0
    second.complete(user("b"))
    assert inner.calls == 1
    second.flush()
    replay = ReplayClient(path)
    assert replay.complete(user("a")).text == "echo:a"
    assert replay.complete(user("b")).text == "echo:b"


def test_recording_rejects_cache_with_different_meta(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    rec.complete(user("a"))
    rec.flush()
    other = FakeInner({"model": "m1", "temperature": 0.7, "max_tokens": 512, "seed": None})
    with pytest.raises(ValueError, match="temperature"):
        RecordingClient(other, path)


def test_recording_params_mirror_inner(tmp_path: Path):
    inner = FakeInner()
    assert RecordingClient(inner, tmp_path / "x.jsonl").params == inner.params


def test_cache_file_format(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    rec.complete(user("a"))
    rec.flush()
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert json.loads(lines[0]) == {"meta": rec.params}
    entry = json.loads(lines[1])
    assert entry["key"] == request_key("m1", user("a"), 0.0, 512, None)
    assert entry["completion"] == {
        "text": "echo:a",
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "latency_s": 0.25,
    }


def test_cache_file_is_sorted_and_stable(tmp_path: Path):
    prompts = ["a", "b", "c", "d"]
    p1, p2 = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    r1 = RecordingClient(FakeInner(), p1)
    for p in prompts:
        r1.complete(user(p))
    r1.flush()
    r2 = RecordingClient(FakeInner(), p2)
    for p in reversed(prompts):
        r2.complete(user(p))
    r2.flush()
    assert p1.read_bytes() == p2.read_bytes()
    keys = [json.loads(line)["key"] for line in p1.read_text("utf-8").splitlines()[1:]]
    assert keys == sorted(keys)


def test_flush_is_atomic_on_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    rec.complete(user("a"))
    rec.flush()
    before = path.read_bytes()
    rec.complete(user("b"))

    def boom(fh, lines):
        fh.write("partial garbage")
        raise OSError("disk full")

    monkeypatch.setattr(llm_module, "_write_lines", boom)
    with pytest.raises(OSError, match="disk full"):
        rec.flush()
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["llm.jsonl"]


def test_replay_missing_file_names_path(tmp_path: Path):
    path = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        ReplayClient(path)


def test_replay_malformed_line_names_line_number(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    rec.complete(user("a"))
    rec.flush()
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("{not json\n")
    with pytest.raises(LLMError, match="line 3"):
        ReplayClient(path)


def test_replay_malformed_header_raises(tmp_path: Path):
    path = tmp_path / "llm.jsonl"
    path.write_text('{"key": "x"}\n', encoding="utf-8")
    with pytest.raises(LLMError, match="line 1"):
        ReplayClient(path)


def test_replay_never_calls_network(tmp_path: Path, monkeypatch):
    path = tmp_path / "llm.jsonl"
    rec = RecordingClient(FakeInner(), path)
    rec.complete(user("a"))
    rec.flush()

    def forbidden(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.Client, "__init__", forbidden)
    assert ReplayClient(path).complete(user("a")).text == "echo:a"
