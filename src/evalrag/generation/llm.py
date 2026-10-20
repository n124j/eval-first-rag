"""LLM clients: a live OpenAI-compatible chat client plus record/replay wrappers.

Recorded completions feed the CI regression gate, so the cache is a
deterministic JSONL file (sorted by request key) written atomically:

    {"meta": {"model": ..., "temperature": ..., "max_tokens": ..., "seed": ...}}
    {"key": "<sha256>", "completion": {"text": ..., "prompt_tokens": ..., ...}}
"""

import contextlib
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Protocol

import httpx

from evalrag.retrieval.embedder import ReplayMiss


class LLMError(RuntimeError):
    """The LLM server failed, returned an unusable response, or a cache is malformed."""


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient(Protocol):
    params: dict[str, object]

    def complete(self, messages: list[dict[str, str]]) -> Completion: ...


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    seed: int | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _key_for(params: dict[str, object], messages: list[dict[str, str]]) -> str:
    return request_key(
        str(params["model"]),
        messages,
        params["temperature"],  # type: ignore[arg-type]
        params["max_tokens"],  # type: ignore[arg-type]
        params.get("seed"),  # type: ignore[arg-type]
    )


def _token_count(usage: object, field: str) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        client: httpx.Client | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = client if client is not None else httpx.Client(timeout=timeout_s)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    @property
    def params(self) -> dict[str, object]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        start = time.perf_counter()
        try:
            response = self.client.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        latency = time.perf_counter() - start
        if not response.is_success:
            raise LLMError(f"LLM server returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed LLM response: {exc!r}: {response.text[:200]}") from exc
        if not isinstance(text, str):
            raise LLMError(f"LLM response content is not a string: {response.text[:200]}")
        usage = data.get("usage")
        return Completion(
            text=text,
            prompt_tokens=_token_count(usage, "prompt_tokens"),
            completion_tokens=_token_count(usage, "completion_tokens"),
            latency_s=latency,
        )


def _read_cache(path: Path) -> tuple[dict[str, object], dict[str, Completion]]:
    meta: dict[str, object] | None = None
    entries: dict[str, Completion] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for number, line in enumerate(fh, start=1):
            try:
                obj = json.loads(line)
                if meta is None:
                    if not isinstance(obj, dict) or not isinstance(obj.get("meta"), dict):
                        raise ValueError("expected a header object with a 'meta' dict")
                    meta = obj["meta"]
                else:
                    key = obj["key"]
                    if not isinstance(key, str):
                        raise ValueError("key must be a string")
                    entries[key] = Completion(**obj["completion"])
            except (ValueError, KeyError, TypeError) as exc:
                raise LLMError(f"{path}: malformed cache at line {number}: {exc!r}") from exc
    if meta is None:
        raise LLMError(f"{path}: malformed cache at line 1: file is empty")
    return meta, entries


def _write_lines(fh: IO[str], lines: list[str]) -> None:
    for line in lines:
        fh.write(line + "\n")


class RecordingClient:
    def __init__(self, inner: LLMClient, cache_path: Path) -> None:
        self.inner = inner
        self.cache_path = Path(cache_path)
        self._entries: dict[str, Completion] = {}
        if self.cache_path.exists():
            meta, self._entries = _read_cache(self.cache_path)
            if meta != inner.params:
                differing = sorted(
                    k for k in meta.keys() | inner.params.keys() if meta.get(k) != inner.params.get(k)
                )
                raise ValueError(
                    f"{self.cache_path}: cache was recorded with different parameters "
                    f"(differing keys: {', '.join(differing)})"
                )

    @property
    def params(self) -> dict[str, object]:
        return self.inner.params

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        key = _key_for(self.params, messages)
        if key not in self._entries:
            self._entries[key] = self.inner.complete(messages)
        return self._entries[key]

    def flush(self) -> None:
        lines = [_canonical({"meta": self.params})]
        for key in sorted(self._entries):
            lines.append(_canonical({"key": key, "completion": asdict(self._entries[key])}))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                _write_lines(fh, lines)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.cache_path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise


class ReplayClient:
    def __init__(self, cache_path: Path) -> None:
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"LLM replay cache not found: {path}")
        self.params, self._entries = _read_cache(path)

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        key = _key_for(self.params, messages)
        if key not in self._entries:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            raise ReplayMiss(f"no recorded completion for key {key} (prompt: {last_user[:80]!r})")
        return self._entries[key]
