"""Golden dataset: JSONL loading with strict validation.

Source ids are DOC ids (for example ``argparse`` or ``os.path``). Chunk ids follow the
convention ``<doc_id>#<nnn>`` (for example ``argparse#016``); `source_of` maps one to the other.
"""

import json
from dataclasses import dataclass
from pathlib import Path

KINDS = ("single", "multi", "unanswerable")
FIELDS = {"id", "question", "expected_sources", "answerable", "kind"}


class DatasetError(ValueError):
    """The golden dataset is malformed or inconsistent."""


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    expected_sources: tuple[str, ...]
    answerable: bool
    kind: str


def source_of(chunk_id: str) -> str:
    """Doc id of a chunk id (``<doc_id>#<nnn>``); an id without ``#`` is returned unchanged."""
    return chunk_id.rsplit("#", 1)[0]


def _normalise(question: str) -> str:
    return " ".join(question.split()).casefold()


def _check_consistency(kind: str, answerable: bool, sources: tuple[str, ...]) -> str | None:
    if kind == "unanswerable":
        if answerable:
            return "unanswerable item must have answerable=false"
        if sources:
            return "unanswerable item must have no expected_sources"
        return None
    if not answerable:
        return f"{kind} item must have answerable=true"
    if not sources:
        return "answerable item has no expected_sources"
    if kind == "single" and len(sources) != 1:
        return "single item needs exactly 1 expected source"
    if kind == "multi" and len(set(sources)) < 2:
        return "multi item needs at least 2 distinct expected sources"
    return None


def load_golden(path: Path, known_sources: set[str] | None = None) -> list[GoldenItem]:
    """Read and validate a golden JSONL file; items are returned in file order."""
    items: list[GoldenItem] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"line {lineno}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise DatasetError(f"line {lineno}: expected a JSON object")
            ident = raw.get("id")

            def fail(msg: str, lineno: int = lineno, ident: object = ident) -> DatasetError:
                return DatasetError(f"line {lineno} (id {ident!r}): {msg}")

            if set(raw) != FIELDS:
                missing = sorted(FIELDS - set(raw))
                extra = sorted(set(raw) - FIELDS)
                raise fail(
                    f"fields must be exactly {sorted(FIELDS)}; missing {missing}, extra {extra}"
                )
            question, sources = raw["question"], raw["expected_sources"]
            answerable, kind = raw["answerable"], raw["kind"]
            if not isinstance(ident, str) or not isinstance(question, str):
                raise fail("id and question must be strings")
            if not isinstance(answerable, bool):
                raise fail("answerable must be a boolean")
            if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
                raise fail("expected_sources must be a list of strings")
            if not ident.strip():
                raise fail("empty id")
            if not question.strip():
                raise fail("empty question")
            if ident in seen_ids:
                raise fail("duplicate id")
            if kind not in KINDS:
                raise fail(f"kind must be one of {list(KINDS)}, got {kind!r}")
            key = _normalise(question)
            if key in seen_questions:
                raise fail("duplicate question")
            problem = _check_consistency(kind, answerable, tuple(sources))
            if problem:
                raise fail(problem)
            if known_sources is not None:
                unknown = sorted(set(sources) - known_sources)
                if unknown:
                    raise fail(f"unknown sources {unknown}")
            seen_ids.add(ident)
            seen_questions.add(key)
            items.append(GoldenItem(ident, question, tuple(sources), answerable, kind))
    return items
