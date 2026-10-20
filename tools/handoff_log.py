"""Append-only JSONL trail of handoffs between the workforce roles."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

ROLES = ("manager", "developer", "tester", "deployer")
VERDICTS = ("submitted", "approved", "rejected", "blocked")


class HandoffError(ValueError):
    pass


@dataclass(frozen=True)
class Handoff:
    project: str
    from_role: str
    to_role: str
    summary: str
    verdict: str
    ts: str


def _validate_ts(ts: str) -> None:
    if not isinstance(ts, str):
        raise HandoffError("ts must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise HandoffError(f"invalid ts {ts!r}: {exc}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise HandoffError(f"ts {ts!r} must be UTC (trailing Z or +00:00)")


def _validate(h: Handoff) -> None:
    if h.from_role not in ROLES:
        raise HandoffError(f"unknown from_role {h.from_role!r}")
    if h.to_role not in ROLES:
        raise HandoffError(f"unknown to_role {h.to_role!r}")
    if h.verdict not in VERDICTS:
        raise HandoffError(f"unknown verdict {h.verdict!r}")
    if not isinstance(h.summary, str) or not h.summary.strip():
        raise HandoffError("summary must not be empty")
    if not isinstance(h.project, str) or not h.project.strip():
        raise HandoffError("project must not be empty")
    _validate_ts(h.ts)


def read_all(log_path: Path) -> list[Handoff]:
    path = Path(log_path)
    if not path.exists():
        return []
    entries: list[Handoff] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entries.append(Handoff(**json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise HandoffError(f"malformed handoff at line {n}: {exc}") from exc
    return entries


def append(log_path: Path, h: Handoff) -> None:
    _validate(h)
    path = Path(log_path)
    if h.from_role == "deployer" and h.verdict == "submitted":
        signed = any(
            e.from_role == "tester" and e.verdict == "approved" and e.project == h.project
            for e in read_all(path)
        )
        if not signed:
            raise HandoffError(
                f"deployer cannot submit project {h.project!r} without a prior tester sign-off"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(asdict(h), ensure_ascii=False) + "\n")
