"""Acceptance tests: the CI eval gate and the Docker packaging are wired as designed."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
STEPS = CI["jobs"]["python"]["steps"]
POLICY = tomllib.loads((ROOT / "workforce" / "policy.toml").read_text(encoding="utf-8"))["roles"]
USES_RE = re.compile(r"@[0-9a-f]{40}$")


def _compose() -> dict:
    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


def _dockerfile() -> list[tuple[str, str]]:
    """Instruction lines as (INSTRUCTION, arguments), continuations joined, comments dropped."""
    lines, pending = [], ""
    for raw in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not pending and (not text or text.startswith("#")):
            continue
        if text.endswith("\\"):
            pending += text[:-1] + " "
            continue
        lines.append((pending + text).strip())
        pending = ""
    return [(ln.split(None, 1)[0].upper(), ln.split(None, 1)[1] if " " in ln else "") for ln in lines]


def _step_index(fragment: str) -> int:
    for i, step in enumerate(STEPS):
        if fragment in step.get("run", "") or fragment in step.get("uses", ""):
            return i
    raise AssertionError(f"no CI step contains {fragment!r}")


def _run(fragment: str) -> str:
    return STEPS[_step_index(fragment)]["run"]


def test_ci_step_order():
    order = [
        _step_index("ruff check"),
        _step_index("pytest"),
        _step_index("evalrag ingest"),
        _step_index("evalrag eval"),
        _step_index("evalrag gate"),
        _step_index("actions/upload-artifact"),
    ]
    assert order == sorted(set(order))


def test_ci_install_steps():
    runs = [s.get("run", "") for s in STEPS]
    assert any("-r requirements.txt" in r and "-r requirements-dev.txt" in r for r in runs)
    assert any("pip install -e . --no-deps" in r for r in runs)
    assert _step_index("pip install -e") < _step_index("evalrag ingest")


@pytest.mark.parametrize("sub", ["ingest", "eval", "gate"])
def test_ci_eval_steps_need_no_endpoint(sub):
    step = STEPS[_step_index(f"evalrag {sub}")]
    assert "LLM_BASE_URL" not in step["run"]
    assert "env" not in step


def test_ci_replays_recordings_and_gates_on_baseline():
    assert "--replay-embeddings evals/recordings/embeddings.npz" in _run("evalrag ingest")
    ev = _run("evalrag eval")
    assert "--replay-llm evals/recordings/llm.jsonl" in ev
    assert "--replay-embeddings evals/recordings/embeddings.npz" in ev
    assert "evals/baseline.json" in _run("evalrag gate")


def test_every_action_is_sha_pinned_in_all_jobs():
    uses = [s["uses"] for job in CI["jobs"].values() for s in job["steps"] if "uses" in s]
    assert uses
    assert all(USES_RE.search(u) for u in uses), uses


def test_workflow_permissions_are_read_only():
    assert CI["permissions"] == {"contents": "read"}


def test_upload_step_always_runs_and_uploads_both_files():
    step = STEPS[_step_index("actions/upload-artifact")]
    assert step["if"] == "always()"
    assert step["with"]["name"] == "eval-report"
    assert step["with"]["retention-days"] == 14
    assert "ci-report.md" in step["with"]["path"]
    assert "ci-metrics.json" in step["with"]["path"]


def test_compose_services():
    services = _compose()["services"]
    assert {"init", "api", "ollama"} <= set(services)
    assert services["api"]["depends_on"]["init"]["condition"] == "service_completed_successfully"
    assert any(str(p).startswith("8000:8000") for p in services["api"]["ports"])
    assert services["ollama"]["profiles"] == ["ollama"]
    for name in ("init", "api"):
        assert "profiles" not in services[name]


def test_ollama_image_is_pinned():
    image = _compose()["services"]["ollama"]["image"]
    tag = image.split(":", 1)[1] if ":" in image else "latest"
    assert tag != "latest"
    assert re.search(r"\d", tag)


def test_dockerfile_base_pinned_and_single_stage():
    froms = [args for ins, args in _dockerfile() if ins == "FROM"]
    assert len(froms) == 1
    image = froms[0].split()[0]
    assert ":" in image
    assert image.split(":", 1)[1] != "latest"


def test_dockerfile_runs_as_non_root():
    users = [args.split(":")[0].strip() for ins, args in _dockerfile() if ins == "USER"]
    assert users
    assert users[-1] not in {"root", "0"}


def test_dockerfile_cmd_uses_the_app_factory():
    cmds = [args for ins, args in _dockerfile() if ins == "CMD"]
    assert len(cmds) == 1
    assert "--factory" in cmds[0]
    assert "evalrag.api:app_from_env" in cmds[0]


def test_dockerignore_excludes_the_essentials():
    lines = {
        ln.strip()
        for ln in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }
    assert {".git", ".venv", "data/", ".env"} <= lines


def test_docker_readme_covers_the_caveats():
    text = (ROOT / "docker" / "README.md").read_text(encoding="utf-8")
    assert "host.docker.internal" in text
    assert "--profile ollama" in text or "profiles" in text
    assert "CPU" in text


def test_deployer_owns_dockerignore():
    assert ".dockerignore" in POLICY["deployer"]
