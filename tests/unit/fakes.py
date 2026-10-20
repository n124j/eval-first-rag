"""Shared deterministic fakes for the ingest and CLI tests (no network)."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from evalrag.generation.answer import REFUSAL_TEXT
from evalrag.generation.llm import Completion

DOCS = {
    "alpha": "The alpha module sorts lists quickly. " * 8,
    "beta": "The beta module parses command line arguments carefully. " * 6,
    "gamma": "Gamma handles temporary files and directories safely. " * 7,
}


class HashEmbedder:
    """Vectors derived from sha256(kind + text): dim 16, float32."""

    model = "fake-embed"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        self.calls.append(len(texts))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows = []
        for text in texts:
            digest = hashlib.sha256(f"{kind}\n{text}".encode()).digest()
            rows.append([b / 255.0 - 0.5 for b in digest[:16]])
        return np.array(rows, dtype=np.float32)


class ScriptedLLM:
    """Cites the first chunk in the prompt, or refuses when the question says 'unanswerable'."""

    def __init__(self) -> None:
        self.params: dict[str, object] = {
            "model": "fake-llm",
            "temperature": 0.0,
            "max_tokens": 64,
            "seed": None,
        }

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        user = messages[-1]["content"]
        if "unanswerable" in user.rsplit("Question:", 1)[-1]:
            return Completion(REFUSAL_TEXT, 10, 3, 0.1)
        match = re.search(r"Source id: ([\w.\-]+#\d{3})\nText: (.+)", user)
        assert match is not None
        quote = " ".join(match.group(2).split()[:4])
        return Completion(f'Answer. [{match.group(1)}: "{quote}"]', 10, 5, 0.1)


def make_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    entries = []
    for doc_id, text in DOCS.items():
        data = text.encode("utf-8")
        (corpus / f"{doc_id}.txt").write_bytes(data)
        entries.append(
            {
                "doc_id": doc_id,
                "title": doc_id,
                "url": f"https://example.test/{doc_id}",
                "version": "1",
                "file": f"{doc_id}.txt",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (corpus / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return corpus
