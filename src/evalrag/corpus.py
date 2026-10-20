"""Load the pinned documentation corpus from a manifest, verifying integrity."""

import hashlib
import json
from pathlib import Path

from evalrag.models import Document

FIELDS = ("doc_id", "title", "url", "version", "file", "sha256")


class CorpusError(ValueError):
    """The corpus manifest or one of its files is invalid."""


def load_corpus(corpus_dir: Path) -> list[Document]:
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"cannot read manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f"manifest {manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise CorpusError("manifest must be a JSON list of objects")

    documents: list[Document] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CorpusError(f"manifest entry {index} is not an object")
        label = entry.get("doc_id", f"#{index}")
        for field in FIELDS:
            if field not in entry:
                raise CorpusError(f"manifest entry {label}: missing field {field}")
        for field in entry:
            if field not in FIELDS:
                raise CorpusError(f"manifest entry {label}: unknown field {field}")
        for field in FIELDS:
            if not isinstance(entry[field], str):
                raise CorpusError(f"manifest entry {label}: field {field} must be a string")
        doc_id = entry["doc_id"]
        if doc_id in seen:
            raise CorpusError(f"duplicate doc_id {doc_id}")
        seen.add(doc_id)

        file_path = corpus_dir / entry["file"]
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            raise CorpusError(f"{doc_id}: cannot read file {entry['file']}: {exc}") from exc
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise CorpusError(f"{doc_id}: sha256 mismatch for {entry['file']}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CorpusError(f"{doc_id}: file {entry['file']} is not UTF-8: {exc}") from exc
        documents.append(
            Document(
                doc_id=doc_id,
                title=entry["title"],
                url=entry["url"],
                version=entry["version"],
                text=text,
            )
        )
    return documents
