"""The eval harness must score any system, so it may not depend on the RAG internals."""

import ast
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[2] / "src" / "evalrag" / "evals"
FORBIDDEN = ("evalrag.retrieval", "evalrag.generation", "evalrag.pipeline")
REQUIRED_MODULES = {"metrics", "dataset", "runner"}


def _is_forbidden(name: str) -> bool:
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


def _resolve_relative(path: Path, level: int, module: str | None) -> str:
    # Package of the file: evalrag.evals (+ subpackages), then climb `level - 1` levels.
    rel = path.relative_to(EVALS_DIR.parent.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        pkg = parts
    else:
        pkg = parts[:-1]
    base = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
    return ".".join(base + ([module] if module else []))


def _imported_names(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative(path, node.level, node.module)
            else:
                base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


def test_evals_package_imports_nothing_from_retrieval_or_generation() -> None:
    assert EVALS_DIR.is_dir(), f"missing package directory {EVALS_DIR}"
    modules = {p.stem for p in EVALS_DIR.glob("*.py")}
    assert REQUIRED_MODULES <= modules, f"missing modules: {REQUIRED_MODULES - modules}"

    offenders = []
    for path in sorted(EVALS_DIR.rglob("*.py")):
        for name in _imported_names(path):
            if _is_forbidden(name):
                offenders.append(f"{path.name}: {name}")
    assert not offenders, f"forbidden imports: {offenders}"
