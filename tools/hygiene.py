"""Repo hygiene checker: flags secret-like strings and missing safety files."""

import re
import sys
from pathlib import Path

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private-key": re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "generic-api-key": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"
    ),
}

# Design and plan documents quote fixture strings; CI's gitleaks job covers them.
# The gitleaks config must quote the fake key it allowlists.
SKIP_PREFIXES = ("docs/superpowers/", ".gitleaks.toml")

SKIP_DIRS = {".git", ".superpowers", ".venv", "node_modules"}


def find_secret_like(text: str) -> list[str]:
    """Return the kinds of secret-like tokens found in text."""
    return [kind for kind, pattern in SECRET_PATTERNS.items() if pattern.search(text)]


def _gitignore_covers_env(gitignore: Path) -> bool:
    if not gitignore.is_file():
        return False
    lines = {line.strip() for line in gitignore.read_text(errors="replace").splitlines()}
    return bool(lines & {".env", "/.env", ".env*", "*.env"})


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check_repo(root: Path) -> list[str]:
    """Return human-readable hygiene problems; an empty list means clean."""
    root = Path(root)
    problems: list[str] = []
    if not _gitignore_covers_env(root / ".gitignore"):
        problems.append(".gitignore: must ignore .env")
    if not (root / ".env.example").is_file():
        problems.append(".env.example: missing")

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if SKIP_DIRS & set(rel.parts) or not path.is_file():
            continue
        if rel.as_posix().startswith(SKIP_PREFIXES):
            continue
        text = _read_text(path)
        if text is None:
            continue
        for kind in find_secret_like(text):
            problems.append(f"{rel.as_posix()}: {kind}")
    return problems


def main() -> int:
    problems = check_repo(Path("."))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
