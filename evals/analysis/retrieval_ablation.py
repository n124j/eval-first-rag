"""Retrieval ablation on the golden set, replayed offline from the committed recordings.

Usage: python evals/analysis/retrieval_ablation.py [--out PATH]
(needs the package importable: PYTHONPATH=src or an editable install).

Compares keyword-only (SQLite FTS5 BM25), dense-only (cosine) and the shipped hybrid
retriever (RRF of both, no confidence guard) on the 50 answerable golden items. Scoring is at
DOCUMENT level, exactly as the harness scores it: each retrieved chunk id is mapped to its
document with `source_of`, then `recall_at_k` and `mrr` from `evalrag.evals.metrics` are
applied. MRR is computed over the top 3 retrieved chunks, because the shipped pipeline (k=3)
passes only those to the harness.

No model, no network, no tuning: default settings (empty env), the recorded query and
document embeddings only. The golden set is used here for measurement only.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from evalrag.config import Settings
from evalrag.corpus import load_corpus
from evalrag.evals.dataset import load_golden, source_of
from evalrag.evals.metrics import mrr, recall_at_k
from evalrag.ingest import DENSE_FILE, KEYWORD_FILE, ingest, open_retriever
from evalrag.retrieval.dense import DenseIndex
from evalrag.retrieval.embedder import ReplayEmbedder
from evalrag.retrieval.keyword import KeywordIndex

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus"
GOLDEN = ROOT / "evals" / "golden" / "golden.jsonl"
EMBEDDINGS = ROOT / "evals" / "recordings" / "embeddings.npz"

DEPTH = 5  # retrieve this many chunks per query; recall is read at 1, 3 and 5
MRR_DEPTH = 3  # the shipped k; the harness computes MRR over these chunks
RETRIEVERS = {
    "keyword": "keyword-only (FTS5 BM25)",
    "dense": "dense-only (cosine)",
    "hybrid": "hybrid RRF k=60, no guard (shipped)",
}


def _score(expected: set[str], chunk_ids: list[str]) -> dict[str, float]:
    sources = [source_of(c) for c in chunk_ids]
    return {
        "r1": recall_at_k(expected, sources, 1),
        "r3": recall_at_k(expected, sources, 3),
        "r5": recall_at_k(expected, sources, 5),
        "rr": mrr(expected, sources[:MRR_DEPTH]),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def analyse() -> dict:
    settings = Settings.from_env({}, require_endpoints=False)
    items = load_golden(GOLDEN, known_sources={d.doc_id for d in load_corpus(CORPUS)})
    answerable = [item for item in items if item.answerable]
    per: dict[str, dict[str, dict[str, float]]] = {name: {} for name in RETRIEVERS}
    kinds = {item.id: item.kind for item in answerable}
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        embedder = ReplayEmbedder(EMBEDDINGS)
        ingest(settings, CORPUS, data_dir, embedder)
        hybrid, _ = open_retriever(settings, data_dir, embedder)
        keyword = KeywordIndex(data_dir / KEYWORD_FILE)
        dense = DenseIndex.load(data_dir / DENSE_FILE)
        try:
            for item in answerable:
                expected = set(item.expected_sources)
                vector = embedder.embed([item.question], kind="query")[0]
                ranked = {
                    "keyword": [h.chunk_id for h in keyword.search(item.question, DEPTH)],
                    "dense": [h.chunk_id for h in dense.search(vector, DEPTH)],
                    "hybrid": [h.chunk_id for h in hybrid.retrieve(item.question, DEPTH)],
                }
                for name, chunk_ids in ranked.items():
                    per[name][item.id] = _score(expected, chunk_ids)
        finally:
            keyword.close()
            hybrid.close()

    retrievers = {}
    for name, rows in per.items():
        retrievers[name] = {
            "items": rows,
            "recall@1": _mean([r["r1"] for r in rows.values()]),
            "recall@3": _mean([r["r3"] for r in rows.values()]),
            "recall@5": _mean([r["r5"] for r in rows.values()]),
            "mrr": _mean([r["rr"] for r in rows.values()]),
            "zero@3": [i for i, r in rows.items() if r["r3"] == 0.0],
            "partial@3": [i for i, r in rows.items() if 0.0 < r["r3"] < 1.0],
        }
    return {"n": len(answerable), "kinds": kinds, "retrievers": retrievers}


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


def _hits(a: dict, name: str) -> set[str]:
    """Items where at least one expected document is in the top 3."""
    return {i for i, r in a["retrievers"][name]["items"].items() if r["r3"] > 0.0}


def _only(a: dict, wins: str, loses: str) -> list[str]:
    return sorted(_hits(a, wins) - _hits(a, loses))


def _compare(x: float, y: float) -> str:
    if round(x, 4) == round(y, 4):
        return "ties"
    return "beats" if x > y else "trails"


def _reading(a: dict) -> list[str]:
    """Plain statements computed from the numbers, so they cannot drift from the table."""
    k, d, h = (a["retrievers"][name] for name in ("keyword", "dense", "hybrid"))
    n = a["n"]
    lines = []
    for metric in ("recall@1", "recall@3", "recall@5", "mrr"):
        lines.append(
            f"- {metric}: dense-only {_compare(d[metric], h[metric])} hybrid "
            f"({d[metric]:.4f} against {h[metric]:.4f}, difference "
            f"{abs(d[metric] - h[metric]):.4f}; one single-document item is "
            f"{1 / n:.4f}); keyword-only "
            f"{_compare(k[metric], h[metric])} hybrid ({k[metric]:.4f})."
        )
    lines += [
        ("- Counting items with no expected document in the top 3: hybrid recovers "
         f"{len(_only(a, 'hybrid', 'dense'))} dense-only miss(es) "
         f"({_ids(_only(a, 'hybrid', 'dense'))}) and loses "
         f"{len(_only(a, 'dense', 'hybrid'))} item(s) that dense-only finds "
         f"({_ids(_only(a, 'dense', 'hybrid'))})."),
        ("- Keyword-only finds an expected document for "
         f"{_ids(_only(a, 'keyword', 'dense'))}, which dense-only misses; it misses "
         f"{len(_only(a, 'dense', 'keyword'))} items that dense-only finds."),
    ]
    return lines


def render(a: dict | None = None) -> str:
    a = a or analyse()
    n, rs = a["n"], a["retrievers"]
    out = [
        "# Retrieval ablation (golden set, answerable items)",
        "",
        ("Generated by `evals/analysis/retrieval_ablation.py` by replaying the committed "
         "embeddings (`evals/recordings/embeddings.npz`) offline: no model, no network, default "
         "settings (chunk size 200 words, overlap 40). Every number below is computed by that "
         "script; the tests re-run it and compare byte for byte."),
        "",
        "## Method",
        "",
        (f"- Items: the {n} answerable golden items (40 single-document, 10 multi-document). "
         "The 10 unanswerable items have no expected source and are not scored here."),
        ("- Scoring is per DOCUMENT, exactly as the harness scores it: a retrieved chunk counts "
         "if its source document is an expected source. recall@k is the fraction of an item's "
         "expected documents found in the top k chunks (0.5 is possible for a multi item), "
         "averaged over items."),
        (f"- MRR is computed over the top {MRR_DEPTH} chunks, because the shipped pipeline "
         f"(k={MRR_DEPTH}) passes only those to the harness."),
        ("- Retrievers: keyword-only is the SQLite FTS5 BM25 index (porter tokenizer, query "
         "tokens joined by OR); dense-only is cosine similarity over the recorded "
         "`nomic-embed-text-v1.5` vectors; hybrid is the shipped `HybridRetriever` (top 50 "
         "candidates from each, reciprocal rank fusion with k=60). The retrieval-confidence "
         "guard is not applied: it decides whether to answer, not what is retrieved."),
        ("- The golden set is used here for measurement only. Nothing was tuned on these "
         "numbers."),
        "",
        "## Results",
        "",
        "| Retriever | recall@1 | recall@3 | recall@5 | MRR (top 3) | items with recall@3 = 0 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, label in RETRIEVERS.items():
        r = rs[name]
        out.append(
            f"| {label} | {r['recall@1']:.4f} | {r['recall@3']:.4f} | {r['recall@5']:.4f} | "
            f"{r['mrr']:.4f} | {len(r['zero@3'])} |"
        )
    out += [
        "",
        ("The hybrid row's recall@3 and MRR are the harness's `recall_at_k` and `mrr` for the "
         "published golden run (both guard on and guard off; the guard does not change "
         "retrieval). The tests check that they match."),
        "",
        "## Items with recall@3 below 1.0",
        "",
        "| Retriever | recall@3 = 0 | 0 < recall@3 < 1 (multi items, one of two documents) |",
        "| --- | --- | --- |",
    ]
    for name, label in RETRIEVERS.items():
        r = rs[name]
        out.append(f"| {label} | {_ids(r['zero@3'])} | {_ids(r['partial@3'])} |")
    out += [
        "",
        "## Where the retrievers disagree",
        "",
        ("Success here means at least one expected document is in the top 3 (recall@3 above "
         "0); failure means none is."),
        "",
        f"- Hybrid fails, dense-only succeeds: {_ids(_only(a, 'dense', 'hybrid'))}.",
        f"- Dense-only fails, hybrid succeeds: {_ids(_only(a, 'hybrid', 'dense'))}.",
        f"- Hybrid fails, keyword-only succeeds: {_ids(_only(a, 'keyword', 'hybrid'))}.",
        f"- Keyword-only fails, hybrid succeeds: {_ids(_only(a, 'hybrid', 'keyword'))}.",
        f"- Keyword-only succeeds, dense-only fails: {_ids(_only(a, 'keyword', 'dense'))}.",
        f"- Dense-only succeeds, keyword-only fails: {_ids(_only(a, 'dense', 'keyword'))}.",
        "",
        "## What this shows",
        "",
    ]
    out += _reading(a)
    out += [
        ("- Plain reading: on this set, adding BM25 to the dense ranking through RRF did not "
         "improve document-level retrieval over dense-only; it is level on recall@3 and "
         "recall@5 and slightly behind on recall@1 and MRR. The gaps are one or two items, "
         "within the noise of 50 items, so this does not show that dense-only is better "
         "either. Keyword-only is the weakest of the three on every column."),
        "",
        "## Limits of this measurement",
        "",
        (f"- {n} items written by one author. One single-document item moves a recall by 0.02; "
         "differences of one or two items are not evidence of anything."),
        ("- Document level, not passage level: a chunk from the right page counts even if it "
         "does not contain the answer, so these numbers overstate how precisely the answering "
         "passage is found."),
        ("- One embedding model at one quantisation, one chunking (200/40), one RRF constant "
         "(60) and one candidate depth (50). None of these was varied."),
        "",
    ]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    text = render()
    if len(argv) == 2 and argv[0] == "--out":
        Path(argv[1]).write_bytes(text.encode("utf-8"))
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
