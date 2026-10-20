"""Role path checker: each commit declares `Role: x`; its diff must stay within that role's globs."""

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

ROLE_RE = re.compile(r"^\s*role:[ \t]*(\S+)[ \t]*$", re.IGNORECASE | re.MULTILINE)
ZERO_SHA = re.compile(r"^0+$")


def load_policy(path: Path) -> dict[str, list[str]]:
    """Load role -> glob list from a policy TOML file."""
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return {role: list(globs) for role, globs in data.get("roles", {}).items()}


def role_from_trailer(message: str) -> str | None:
    """Return the lowercased value of the last `Role: x` line, or None."""
    matches = ROLE_RE.findall(message)
    return matches[-1].lower() if matches else None


def _glob_to_regex(glob: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    while i < len(glob):
        char = glob[i]
        if glob.startswith("/**/", i):
            out.append("(?:/|/.*/)")
            i += 4
            continue
        if glob.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(char))
        i += 1
    return re.compile("".join(out) + r"\Z")


def _matches(path: str, globs: list[str]) -> bool:
    return any(_glob_to_regex(glob).match(path) for glob in globs)


def violations(role: str, changed: list[str], policy: dict[str, list[str]]) -> list[str]:
    """Return changed paths the role may not touch (all of them for an unknown role)."""
    globs = policy.get(role, [])
    return [path for path in changed if not _matches(path.replace("\\", "/"), globs)]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _resolves(repo: Path, rev: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0


def _changed_paths(repo: Path, sha: str) -> list[str]:
    out = _git(repo, "diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "-M", "-z", sha)
    fields = out.split("\0")
    paths: list[str] = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i][0]
        count = 2 if status in "RC" else 1
        paths.extend(fields[i + 1 : i + 1 + count])
        i += 1 + count
    return paths


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="role_paths")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--policy", default="workforce/policy.toml")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args(argv)

    repo = Path(args.repo)
    policy = load_policy(Path(args.policy))

    if ZERO_SHA.match(args.base) or not _resolves(repo, args.base):
        rev_range = args.head
    else:
        rev_range = f"{args.base}..{args.head}"
    shas = _git(repo, "rev-list", "--no-merges", rev_range).split()

    failed = False
    for sha in shas:
        short = sha[:7]
        role = role_from_trailer(_git(repo, "log", "-1", "--format=%B", sha))
        if role is None:
            print(f"{short} (missing) Role trailer not found")
            failed = True
            continue
        if role not in policy:
            print(f"{short} {role} unknown role")
            failed = True
            continue
        for path in violations(role, _changed_paths(repo, sha), policy):
            print(f"{short} {role} {path}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
