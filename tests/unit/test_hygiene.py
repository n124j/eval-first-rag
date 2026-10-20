import subprocess
import sys
from pathlib import Path

from tools.hygiene import check_repo, find_secret_like

REPO_ROOT = Path(__file__).resolve().parents[2]

# Built by concatenation so this file does not trip the hygiene scan itself.
AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
LEAKED_KEY = "AKIA" + "QWERTYUIOPASDFGH"
PRIVATE_KEY_HEADER = "-----BEGIN RSA " + "PRIVATE KEY-----"


def make_repo(tmp_path: Path, gitignore: str = ".env\n", env_example: bool = True) -> Path:
    (tmp_path / ".gitignore").write_text(gitignore)
    if env_example:
        (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n")
    return tmp_path


def test_flags_private_key_header():
    assert "private-key" in find_secret_like(PRIVATE_KEY_HEADER)


def test_flags_aws_key():
    assert "aws-access-key" in find_secret_like(AWS_KEY)


def test_ignores_plain_text():
    assert find_secret_like("hello world") == []


def test_check_repo_requires_env_ignored(tmp_path):
    make_repo(tmp_path, gitignore="__pycache__/\n")
    problems = check_repo(tmp_path)
    assert any(".env" in p and ".gitignore" in p for p in problems)


def test_check_repo_requires_env_example(tmp_path):
    make_repo(tmp_path, env_example=False)
    problems = check_repo(tmp_path)
    assert any(".env.example" in p for p in problems)


def test_check_repo_flags_secret_in_tracked_text_file(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "config.txt").write_text(f"key = {LEAKED_KEY}\n")
    problems = check_repo(tmp_path)
    assert any("config.txt" in p for p in problems)


def test_check_repo_clean(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "notes.md").write_text("benign text\n")
    assert check_repo(tmp_path) == []


def test_check_repo_skips_ignored_dirs(tmp_path):
    make_repo(tmp_path)
    for d in (".superpowers", ".venv", ".git", "node_modules"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "leak.txt").write_text(LEAKED_KEY)
    assert check_repo(tmp_path) == []


def test_check_repo_skips_binary_files(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01" + LEAKED_KEY.encode())
    assert check_repo(tmp_path) == []


def run_cli(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tools.hygiene"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )


def test_cli_exits_zero_when_clean(tmp_path):
    make_repo(tmp_path)
    result = run_cli(tmp_path)
    assert result.returncode == 0
    assert result.stdout == ""


def test_cli_exits_one_and_prints_problems(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "config.txt").write_text(LEAKED_KEY)
    result = run_cli(tmp_path)
    assert result.returncode == 1
    assert "config.txt: aws-access-key" in result.stdout.splitlines()


def test_check_repo_skips_docs_superpowers(tmp_path):
    make_repo(tmp_path)
    doc = tmp_path / "docs" / "superpowers" / "plans"
    doc.mkdir(parents=True)
    (doc / "plan.md").write_text(f"example: {AWS_KEY}\n")
    assert check_repo(tmp_path) == []


def test_check_repo_reports_same_content_elsewhere(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "README.md").write_text(PRIVATE_KEY_HEADER + "\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "other.md").write_text(f"{AWS_KEY}\n")
    problems = check_repo(tmp_path)
    assert "README.md: private-key" in problems
    assert "docs/other.md: aws-access-key" in problems


def test_gitleaks_config_is_exempt(tmp_path):
    root = make_repo(tmp_path)
    (root / ".gitleaks.toml").write_text("regexes = ['''" + AWS_KEY + "''']\n")
    assert check_repo(root) == []
