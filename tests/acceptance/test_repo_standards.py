"""Acceptance tests: the repo carries the workforce standards."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from tools.hygiene import check_repo

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("manager", "developer", "tester", "deployer")
POLICY = tomllib.loads((ROOT / "workforce" / "policy.toml").read_text(encoding="utf-8"))["roles"]
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_policy_has_exactly_the_four_roles():
    assert set(POLICY) == set(ROLES)


@pytest.mark.parametrize("glob", ["src/**", "corpus/**", "scripts/**"])
def test_developer_owns_project_dirs(glob):
    assert glob in POLICY["developer"]


def test_tester_owns_evals():
    assert "evals/**" in POLICY["tester"]


def test_no_site_globs_in_policy():
    assert not [g for globs in POLICY.values() for g in globs if g.startswith("site")]


def test_handoff_log_writable_by_all_roles():
    for role in ROLES:
        assert "workforce/status/handoffs.jsonl" in POLICY[role]


@pytest.mark.parametrize("role", ROLES)
def test_agent_file_matches_policy(role):
    path = ROOT / ".claude" / "agents" / f"{role}.md"
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) < 60
    assert f"Role: {role}" in text
    missing = [g for g in POLICY[role] if f"`{g}`" not in text]
    assert not missing, f"{role} omits policy globs: {missing}"
    assert "site/" not in text


def test_env_example_declares_llm_settings():
    keys = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line
    }
    assert {"LLM_BASE_URL", "EMBED_MODEL"} <= keys


def test_hygiene_is_clean():
    assert check_repo(ROOT) == []


def test_package_version():
    import evalrag

    assert evalrag.__version__ == "0.1.0"


def test_ci_jobs_are_python_and_secrets_only():
    assert set(CI["jobs"]) == {"python", "secrets"}
    assert "site" not in CI["jobs"]


def test_ci_uses_are_sha_pinned():
    uses = [
        step["uses"]
        for job in CI["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]
    assert uses
    for ref in uses:
        _, _, version = ref.partition("@")
        assert SHA_RE.match(version), f"not pinned to a 40-hex SHA: {ref}"


def test_ci_runs_ruff_before_pytest():
    runs = [s["run"] for s in CI["jobs"]["python"]["steps"] if "run" in s]
    ruff = next(i for i, r in enumerate(runs) if "ruff check" in r)
    pytest_ = next(i for i, r in enumerate(runs) if "pytest" in r)
    assert ruff < pytest_
    assert CI["permissions"] == {"contents": "read"}
