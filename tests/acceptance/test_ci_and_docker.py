"""Acceptance tests: the CI eval gate and the Docker packaging are wired as designed."""
from __future__ import annotations

import copy
import fnmatch
import re
import subprocess
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
    assert any(str(p) == "127.0.0.1:8000:8000" for p in services["api"]["ports"])
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
    assert {".git", ".venv", "data/", ".env", ".superpowers/", "evals/reports/"} <= lines


def test_docker_readme_covers_the_caveats():
    text = (ROOT / "docker" / "README.md").read_text(encoding="utf-8")
    assert "host.docker.internal" in text
    assert "--profile ollama" in text or "profiles" in text
    assert "CPU" in text


def test_deployer_owns_dockerignore():
    assert ".dockerignore" in POLICY["deployer"]


# --- hardening: failures must fail the job, /data must be writable, build context must be intact ---

FAIL_MASKS = ("|| true", "; true", "set +e")


def _masking_problems(job: dict, step: dict) -> list[str]:
    problems = []
    if "continue-on-error" in job:
        problems.append("job has continue-on-error")
    if "continue-on-error" in step:
        problems.append("step has continue-on-error")
    problems += [f"run contains {m!r}" for m in FAIL_MASKS if m in step.get("run", "")]
    return problems


@pytest.mark.parametrize("sub", ["ingest", "eval", "gate"])
def test_ci_eval_steps_cannot_mask_failure(sub):
    job = CI["jobs"]["python"]
    assert _masking_problems(job, STEPS[_step_index(f"evalrag {sub}")]) == []


def test_failure_masking_detector_catches_mutations():
    job = CI["jobs"]["python"]
    step = STEPS[_step_index("evalrag gate")]
    for key, value in (("continue-on-error", True),):
        mutated = copy.deepcopy(step)
        mutated[key] = value
        assert _masking_problems(job, mutated)
    for mask in FAIL_MASKS:
        mutated = copy.deepcopy(step)
        mutated["run"] += f" {mask}"
        assert _masking_problems(job, mutated)
    assert _masking_problems({**job, "continue-on-error": True}, step)


def _data_dir_prepared_before_last_user(instructions: list[tuple[str, str]]) -> bool:
    last_user = max(i for i, (ins, _) in enumerate(instructions) if ins == "USER")
    return any(
        ins == "RUN" and "mkdir" in args and "/data" in args and "chown" in args
        for ins, args in instructions[:last_user]
    )


def test_dockerfile_prepares_data_dir_before_dropping_privileges():
    instructions = _dockerfile()
    assert _data_dir_prepared_before_last_user(instructions)
    # mutation: moving the USER instruction ahead of the RUN must be detected
    users = [x for x in instructions if x[0] == "USER"]
    rest = [x for x in instructions if x[0] != "USER"]
    assert not _data_dir_prepared_before_last_user(users + rest)


def test_compose_init_does_not_restart_and_api_has_healthcheck():
    services = _compose()["services"]
    assert services["init"]["restart"] == "no"
    assert services["api"]["healthcheck"]["test"]


def _ignored(path: str, patterns: list[str]) -> bool:
    for pat in patterns:
        base = pat.rstrip("/")
        if path == base or path.startswith(base + "/") or fnmatch.fnmatch(path, base):
            return True
    return False


def _dockerignore_patterns() -> list[str]:
    return [
        ln.strip()
        for ln in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#") and not ln.startswith("!")
    ]


def test_dockerfile_copy_sources_are_exact_and_not_ignored():
    sources = set()
    for ins, args in _dockerfile():
        if ins == "COPY":
            parts = [p for p in args.split() if not p.startswith("--")]
            sources.update(p.rstrip("/") for p in parts[:-1])
    assert sources == {"requirements.txt", "pyproject.toml", "src", "corpus", "evals/recordings"}
    patterns = _dockerignore_patterns()
    assert [s for s in sources if _ignored(s, patterns)] == []
    assert _ignored("src", [*patterns, "src/"])  # matcher sanity: it can flag an exclusion


def test_env_example_localhost_urls_are_flagged_for_docker():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    for key in ("LLM_BASE_URL", "EMBED_BASE_URL"):
        idx = next((i for i, ln in enumerate(lines) if ln.startswith(f"{key}=")), None)
        assert idx is not None, f"{key} missing from .env.example"
        value = lines[idx].split("=", 1)[1]
        if "localhost" in value or "127.0.0.1" in value:
            block = []
            j = idx - 1
            while j >= 0 and (lines[j].startswith("#") or lines[j].startswith("LLM_BASE_URL=")):
                block.append(lines[j])
                j -= 1
            assert any("host.docker.internal" in ln for ln in block), (
                f"{key} is a localhost URL without an adjacent host.docker.internal comment"
            )


# --- final whole-branch review fixes (I4, I6) ----------------------------------------------

RECORDING_PATHS = {"evals/recordings/llm.jsonl", "evals/reports/v0-prompt-1b-completions.jsonl"}
CACHE_KEY = "047b03ba82d86ec7a09dfdc00f03aa2797f489743fc7865b0d55126f26275143"
FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"  # split so the hygiene check does not flag this file


def _gitleaks() -> dict:
    return tomllib.loads((ROOT / ".gitleaks.toml").read_text(encoding="utf-8"))


def _gitleaks_allowlists() -> list[dict]:
    config = _gitleaks()
    entries = list(config.get("allowlists", []))
    if "allowlist" in config:
        entries.append(config["allowlist"])
    return entries


def test_every_published_compose_port_is_bound_to_loopback():
    published = [
        (name, str(port))
        for name, service in _compose()["services"].items()
        for port in service.get("ports", [])
    ]
    assert {name for name, _ in published} >= {"api", "ollama"}
    assert [p for p in published if not p[1].startswith("127.0.0.1:")] == []


def test_docker_readme_security_mentions_the_loopback_binding():
    text = (ROOT / "docker" / "README.md").read_text(encoding="utf-8")
    start = text.index("\n## Security\n")
    end = text.find("\n## ", start + 1)
    assert "127.0.0.1" in text[start : end if end != -1 else len(text)]


def test_gitleaks_keeps_default_rules_and_the_fake_key_allowlist():
    config = _gitleaks()
    assert config["extend"]["useDefault"] is True
    # gitleaks rejects the deprecated [allowlist] table alongside [[allowlists]].
    assert not ("allowlist" in config and "allowlists" in config)
    assert any(
        re.search(rx, FAKE_AWS_KEY)
        for entry in _gitleaks_allowlists()
        for rx in entry.get("regexes", [])
    )


def _cache_key_allowlist() -> dict:
    entries = [e for e in _gitleaks_allowlists() if e.get("paths")]
    assert len(entries) == 1, "expected exactly one path-scoped allowlist (the cache keys)"
    return entries[0]


def test_gitleaks_cache_key_allowlist_is_scoped_to_the_two_recordings():
    entry = _cache_key_allowlist()
    assert entry.get("condition") == "AND", "paths AND regexes must both match"
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    matched = {f for f in tracked if any(re.search(p, f) for p in entry["paths"])}
    assert matched == RECORDING_PATHS
    assert len(entry["paths"]) == 2


def test_gitleaks_cache_key_regex_matches_only_cache_keys():
    entry = _cache_key_allowlist()
    assert entry.get("regexTarget", "secret") in {"match", "line"}
    regexes = entry["regexes"]
    assert len(regexes) == 1
    rx = re.compile(regexes[0])
    line = '{"completion":{"text":"Not found."},"key":"' + CACHE_KEY + '"}'
    assert rx.search(line)
    assert rx.search('key":"' + CACHE_KEY + '"')  # the generic-api-key rule's match text
    for text in (
        '"api_key": "' + FAKE_AWS_KEY + '"',
        '"key":"' + FAKE_AWS_KEY + '"',
        '"key":"' + CACHE_KEY[:-1] + '"',
        '"key":"' + CACHE_KEY.upper() + '"',
        '"token":"' + CACHE_KEY + '"',
    ):
        assert not rx.search(text), text
