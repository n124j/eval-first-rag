"""Build the pinned corpus from the official Python 3.12 text documentation archive.

Source: the Python documentation (PSF documentation license).
Run once, after approval: python scripts/fetch_corpus.py --out corpus --version 3.12
"""

import argparse
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

DEFAULT_URL = "https://docs.python.org/3.12/archives/python-3.12-docs-text.zip"

MODULES: tuple[str, ...] = (
    "argparse", "bisect", "calendar", "collections", "contextlib", "copy", "csv",
    "dataclasses", "datetime", "decimal", "enum", "fractions", "functools", "glob",
    "hashlib", "heapq", "itertools", "json", "logging", "math", "operator", "os.path",
    "pathlib", "pickle", "random", "re", "secrets", "shutil", "sqlite3", "statistics",
    "string", "subprocess", "tempfile", "textwrap", "threading", "time", "typing",
    "unittest", "urllib.parse", "uuid", "zipfile",
)  # fmt: skip


def _find_library_prefix(names: Sequence[str]) -> str:
    for name in names:
        parts = name.split("/")
        if "library" in parts[:-1]:
            return "/".join(parts[: parts.index("library") + 1]) + "/"
    raise ValueError("archive has no library/ directory")


def extract_pages(
    zip_path: Path, out_dir: Path, modules: Sequence[str], version: str
) -> list[dict]:
    out_dir = Path(out_dir)
    seen: set[str] = set()
    for module in modules:
        if not module or "/" in module or "\\" in module or ".." in module:
            raise ValueError(f"unsafe module name: {module!r}")
        if module in seen:
            raise ValueError(f"duplicate module name: {module}")
        seen.add(module)
    entries: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        prefix = _find_library_prefix(names)
        available = set(names)
        for module in modules:
            member = f"{prefix}{module}.txt"
            if member not in available:
                raise ValueError(f"module {module} not found in archive ({member})")
        out_dir.mkdir(parents=True, exist_ok=True)
        for module in modules:
            data = zf.read(f"{prefix}{module}.txt")
            (out_dir / f"{module}.txt").write_bytes(data)
            text = data.decode("utf-8")
            title = next((ln.strip() for ln in text.splitlines() if ln.strip()), module)
            entries.append(
                {
                    "doc_id": module,
                    "title": title,
                    "url": f"https://docs.python.org/{version}/library/{module}.html",
                    "version": version,
                    "file": f"{module}.txt",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return entries


def write_manifest(entries: list[dict], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    parser.add_argument("--version", default="3.12")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    if not args.url.startswith("https://"):
        parser.error("--url must be an https:// URL")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "docs.zip"
        with urllib.request.urlopen(args.url, timeout=60) as resp, archive.open("wb") as fh:
            if not resp.geturl().startswith("https://"):
                raise ValueError(f"download redirected to non-https URL: {resp.geturl()}")
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        data = archive.read_bytes()
        print(f"archive sha256: {hashlib.sha256(data).hexdigest()}")
        print(f"archive size: {len(data)} bytes")
        entries = extract_pages(archive, args.out, MODULES, args.version)
    write_manifest(entries, args.out)
    print(f"wrote {len(entries)} pages and manifest to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
