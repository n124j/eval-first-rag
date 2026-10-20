import hashlib
import importlib.util
import json
import urllib.request
import zipfile
from pathlib import Path

import pytest

from evalrag.corpus import CorpusError, load_corpus
from evalrag.models import Document

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_corpus", ROOT / "scripts" / "fetch_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"{doc_id} title",
        "url": f"https://docs.python.org/3.12/library/{doc_id}.html",
        "version": "3.12",
        "file": f"{doc_id}.txt",
        "sha256": _sha(text.encode("utf-8")),
    }


def _write_corpus(tmp_path: Path, texts: dict[str, str]) -> list[dict]:
    entries = []
    for doc_id, text in texts.items():
        (tmp_path / f"{doc_id}.txt").write_bytes(text.encode("utf-8"))
        entries.append(_entry(doc_id, text))
    (tmp_path / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return entries


def _write_manifest(tmp_path: Path, entries: list[dict]) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")


def _make_zip(path: Path, modules: dict[str, str], root: str = "python-3.12-docs-text") -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{root}/about.txt", "ignored")
        for name, text in modules.items():
            zf.writestr(f"{root}/library/{name}.txt", text)


def test_load_corpus_roundtrip(tmp_path):
    _write_corpus(tmp_path, {"b": "B text é\n", "a": "A text\n"})
    docs = load_corpus(tmp_path)
    assert [d.doc_id for d in docs] == ["b", "a"]
    assert docs[0] == Document(
        doc_id="b",
        title="b title",
        url="https://docs.python.org/3.12/library/b.html",
        version="3.12",
        text="B text é\n",
    )


def test_hash_mismatch_raises(tmp_path):
    _write_corpus(tmp_path, {"a": "A text"})
    (tmp_path / "a.txt").write_bytes(b"tampered")
    with pytest.raises(CorpusError, match="a"):
        load_corpus(tmp_path)


def test_missing_file_raises(tmp_path):
    _write_corpus(tmp_path, {"a": "A text"})
    (tmp_path / "a.txt").unlink()
    with pytest.raises(CorpusError, match="a"):
        load_corpus(tmp_path)


@pytest.mark.parametrize("field", ["doc_id", "title", "url", "version", "file", "sha256"])
def test_manifest_requires_all_fields(tmp_path, field):
    entries = _write_corpus(tmp_path, {"a": "A text"})
    del entries[0][field]
    _write_manifest(tmp_path, entries)
    with pytest.raises(CorpusError, match=field):
        load_corpus(tmp_path)


def test_extra_manifest_field_rejected(tmp_path):
    entries = _write_corpus(tmp_path, {"a": "A text"})
    entries[0]["bonus"] = "x"
    _write_manifest(tmp_path, entries)
    with pytest.raises(CorpusError, match="bonus"):
        load_corpus(tmp_path)


def test_duplicate_doc_id_rejected(tmp_path):
    entries = _write_corpus(tmp_path, {"a": "A text"})
    _write_manifest(tmp_path, entries + entries)
    with pytest.raises(CorpusError, match="a"):
        load_corpus(tmp_path)


def test_modules_list_has_41_unique_entries():
    modules = _load_script().MODULES
    assert len(modules) == 41
    assert len(set(modules)) == 41


def test_extract_pages_roundtrips_through_load_corpus(tmp_path):
    script = _load_script()
    zip_path = tmp_path / "docs.zip"
    _make_zip(zip_path, {"json": "\n\njson module\n===\nbody\n", "re": "re module\nbody é\n"})
    out = tmp_path / "out"
    entries = script.extract_pages(zip_path, out, ["json", "re"], "3.12")
    assert [e["doc_id"] for e in entries] == ["json", "re"]
    assert entries[0]["title"] == "json module"
    assert entries[0]["url"] == "https://docs.python.org/3.12/library/json.html"
    assert entries[0]["file"] == "json.txt"
    script.write_manifest(entries, out)
    docs = load_corpus(out)
    assert [d.doc_id for d in docs] == ["json", "re"]
    assert docs[1].text == "re module\nbody é\n"


def test_missing_module_in_archive_names_module(tmp_path):
    script = _load_script()
    zip_path = tmp_path / "docs.zip"
    _make_zip(zip_path, {"json": "json\n"})
    with pytest.raises(Exception, match="nosuchmod"):
        script.extract_pages(zip_path, tmp_path / "out", ["json", "nosuchmod"], "3.12")


def test_no_network_calls(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    script = _load_script()
    zip_path = tmp_path / "docs.zip"
    _make_zip(zip_path, {"json": "json\n"})
    entries = script.extract_pages(zip_path, tmp_path / "out", ["json"], "3.12")
    script.write_manifest(entries, tmp_path / "out")
    assert len(load_corpus(tmp_path / "out")) == 1
