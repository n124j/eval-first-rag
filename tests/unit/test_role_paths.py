import subprocess
from pathlib import Path

import pytest

from tools.role_paths import load_policy, main, role_from_trailer, violations

POLICY_PATH = Path(__file__).resolve().parents[2] / "workforce" / "policy.toml"


@pytest.fixture(scope="module")
def policy() -> dict[str, list[str]]:
    return load_policy(POLICY_PATH)


def test_trailer_parsed():
    assert role_from_trailer("feat: x\n\nRole: Tester") == "tester"


def test_trailer_with_trailing_coauthor_line():
    msg = "feat: x\n\nRole: developer\nCo-Authored-By: Claude <noreply@anthropic.com>"
    assert role_from_trailer(msg) == "developer"


def test_last_trailer_wins():
    assert role_from_trailer("x\n\nRole: manager\nRole: Deployer\n") == "deployer"


def test_missing_trailer_is_none():
    assert role_from_trailer("feat: x") is None


def test_developer_cannot_touch_acceptance_tests(policy):
    assert violations("developer", ["tests/acceptance/test_a.py"], policy) == [
        "tests/acceptance/test_a.py"
    ]


def test_tester_cannot_touch_product_source(policy):
    assert violations("tester", ["tools/hygiene.py"], policy) == ["tools/hygiene.py"]


def test_tester_may_write_evals(policy):
    assert violations("tester", ["evals/rag/golden.jsonl"], policy) == []


def test_windows_separators(policy):
    assert violations("developer", ["tools\\hygiene.py"], policy) == []


def test_root_file_matches(policy):
    assert violations("developer", ["README.md"], policy) == []


def test_unknown_role_all_violations(policy):
    assert violations("ceo", ["README.md"], policy) == ["README.md"]


def test_double_star_matches_zero_dirs_and_deep(policy):
    assert violations("manager", ["docs/superpowers/specs/a.md", "docs/superpowers/b.md"], policy) == []


def test_single_star_stays_in_segment(policy):
    assert violations("deployer", ["Dockerfile.dev", "sub/Dockerfile"], policy) == ["sub/Dockerfile"]
    assert violations("deployer", ["compose.prod.yaml", "x/compose.yaml"], policy) == ["x/compose.yaml"]


def test_policy_file_loads(policy):
    assert {"manager", "developer", "tester", "deployer"} <= set(policy)
    assert violations("manager", ["workforce/policy.toml", ".claude/agents/a.md"], policy) == []


# --- CLI ---------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, rel: str, message: str, content: str = "x\n") -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "T")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _commit(tmp_path, "README.md", "init\n\nRole: developer")
    return tmp_path


def _run(repo: Path, base: str, head: str = "HEAD") -> int:
    return main(["--base", base, "--head", head, "--policy", str(POLICY_PATH), "--repo", str(repo)])


def test_cli_passes_clean_commit(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "tools/a.py", "feat: a\n\nRole: developer\nCo-Authored-By: C <c@example.com>")
    assert _run(repo, base) == 0


def test_cli_fails_on_missing_trailer(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "tools/a.py", "feat: a without trailer")
    assert _run(repo, base) == 1


def test_cli_fails_on_violation(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "tests/acceptance/t.py", "test: t\n\nRole: developer")
    assert _run(repo, base) == 1


def test_cli_fails_on_unknown_role(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "tools/a.py", "feat: a\n\nRole: ceo")
    assert _run(repo, base) == 1


def test_cli_skips_merge_commits(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "side")
    _commit(repo, "tools/side.py", "feat: side\n\nRole: developer")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "tools/main.py", "feat: main\n\nRole: developer")
    # Merge commit has no Role trailer; it must be skipped.
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge side", "side")
    assert _run(repo, base) == 0


def test_cli_checks_both_sides_of_rename(repo):
    _commit(repo, "tests/acceptance/old.py", "test: old\n\nRole: tester")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "tools").mkdir()
    _git(repo, "mv", "tests/acceptance/old.py", "tools/new.py")
    _git(repo, "commit", "-q", "-m", "refactor: move\n\nRole: developer")
    # Developer may write the new path but not the old (acceptance) path.
    assert _run(repo, base) == 1


def test_cli_zero_base_checks_all_commits(repo):
    _commit(repo, "tests/acceptance/t.py", "test: t\n\nRole: developer")
    assert _run(repo, "0" * 40) == 1


def test_cli_unresolvable_base_checks_all_commits(repo):
    assert _run(repo, "deadbeef" * 5) == 0
    _commit(repo, "tests/acceptance/t.py", "test: t\n\nRole: developer")
    assert _run(repo, "deadbeef" * 5) == 1


def test_mid_pattern_double_star():
    p = {"r": ["a/**/b"]}
    assert violations("r", ["a/b", "a/x/b", "a/x/y/b"], p) == []
    assert violations("r", ["a/xb", "ab"], p) == ["a/xb", "ab"]


def test_trailing_double_star_and_root_only_still_hold(policy):
    assert violations("manager", ["docs/superpowers/a/b/c.md"], policy) == []
    assert violations("deployer", ["sub/Dockerfile"], policy) == ["sub/Dockerfile"]


def test_non_ascii_commit_message_is_decoded(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "tools/a.py", "feat: café Łódź\n\nRole: developer")
    assert _run(repo, base) == 0
