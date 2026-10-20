"""Failure analysis of the golden run, replayed offline from the committed recordings.

Usage: python evals/analysis/failure_taxonomy.py [--out PATH]
(needs the package importable: PYTHONPATH=src or an editable install).

No model, no network, no tuning: the default settings (guard 0.67, k=3, empty env) and the
recorded embeddings and completions only. Everything printed is computed here.

Caveat on the class "quote not found: differs only in case, spacing or punctuation": it is
defined by a loose deterministic check (lowercase, remove every non-alphanumeric character,
then a substring test). It therefore also catches quotes that are not verbatim and may not
support the claim (g015 is an example). The real cause of each difference (for example markup
such as backticks or asterisks in the stored chunk text) was NOT individually confirmed.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from evalrag.config import Settings
from evalrag.corpus import load_corpus
from evalrag.evals.dataset import load_golden, source_of
from evalrag.evals.metrics import recall_at_k
from evalrag.generation.answer import MIN_QUOTE_CHARS, _normalise
from evalrag.generation.llm import ReplayClient
from evalrag.ingest import ingest, open_retriever
from evalrag.pipeline import RagPipeline
from evalrag.retrieval.embedder import ReplayEmbedder

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus"
GOLDEN = ROOT / "evals" / "golden" / "golden.jsonl"
EMBEDDINGS = ROOT / "evals" / "recordings" / "embeddings.npz"
LLM = ROOT / "evals" / "recordings" / "llm.jsonl"

VALID = "valid answer with valid citation"
GUARD = "refused by the guard (abstained)"
MODEL_REFUSED = "refused by the model"
NO_CITATION = "answered without any parsable citation"
UNKNOWN_ID = "citation id not among the retrieved chunks"
SHORT = f"quote shorter than {MIN_QUOTE_CHARS} normalised characters"
SWAP = "quote not found: quote-character swap only"
ELIDED = "quote not found: truncated or elided quote"
WRONG_ID = "quote not found: quote is in a different retrieved chunk (wrong id cited)"
CASE = "quote not found: differs only in case, spacing or punctuation"
PARAPHRASE = "quote not found: paraphrase (token overlap above 0.6)"
OTHER = "quote not found: other"
CLASSES = (VALID, GUARD, MODEL_REFUSED, NO_CITATION, UNKNOWN_ID, SHORT, SWAP, ELIDED,
           WRONG_ID, CASE, PARAPHRASE, OTHER)

_QUOTES = "'\"`‘’“”"
_QUOTE_TABLE = {ord(c): "'" for c in _QUOTES}
_TOKEN = re.compile(r"\w+")
_NONALNUM = re.compile(r"[\W_]+")


def _squash(text: str) -> str:
    return _NONALNUM.sub("", text.lower())


def _not_found_class(quote: str, cited_text: str, other_texts: list[str]) -> str:
    if quote.translate(_QUOTE_TABLE) in cited_text.translate(_QUOTE_TABLE):
        return SWAP
    if "..." in quote or "…" in quote:
        return ELIDED
    if any(quote in text for text in other_texts):
        return WRONG_ID
    if _squash(quote) and _squash(quote) in _squash(cited_text):
        return CASE
    tokens = _TOKEN.findall(quote.lower())
    chunk_tokens = set(_TOKEN.findall(cited_text.lower()))
    if tokens and sum(t in chunk_tokens for t in tokens) / len(tokens) > 0.6:
        return PARAPHRASE
    return OTHER


def _classify(answer, chunks_by_id) -> tuple[str, str | None]:
    """(class, offending quote). Priority: guard, model refusal, valid, no citation, then the
    first citation problem in the order unknown id, short quote, quote not found."""
    if answer.abstained:
        return GUARD, None
    if answer.refused:
        return MODEL_REFUSED, None
    if answer.report.valid:
        return VALID, None
    if not answer.citations:
        return NO_CITATION, None
    retrieved = {i: _normalise(chunks_by_id[i].text) for i in answer.retrieved_ids}
    unknown = [c for c in answer.citations if c.chunk_id not in retrieved]
    if unknown:
        return UNKNOWN_ID, unknown[0].quote
    shorts = [c for c in answer.citations if len(_normalise(c.quote.strip())) < MIN_QUOTE_CHARS]
    if shorts:
        return SHORT, shorts[0].quote
    for citation in answer.citations:
        quote = _normalise(citation.quote.strip())
        if quote not in retrieved[citation.chunk_id]:
            others = [t for i, t in retrieved.items() if i != citation.chunk_id]
            return _not_found_class(quote, retrieved[citation.chunk_id], others), citation.quote
    raise AssertionError("invalid answer with no identifiable problem")


def analyse() -> dict:
    settings = Settings.from_env({}, require_endpoints=False)
    k = settings.top_k
    items = load_golden(GOLDEN, known_sources={d.doc_id for d in load_corpus(CORPUS)})
    classes: dict[str, list[str]] = {name: [] for name in CLASSES}
    rows: dict[str, dict] = {}
    unanswerable: dict[str, list[str]] = {"guard": [], "model": [], "answered": []}
    answers: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        embedder = ReplayEmbedder(EMBEDDINGS)
        ingest(settings, CORPUS, data_dir, embedder)
        retriever, chunks_by_id = open_retriever(settings, data_dir, embedder)
        try:
            pipeline = RagPipeline(
                retriever, ReplayClient(LLM), chunks_by_id, k,
                min_dense_score=settings.min_dense_score,
            )
            for item in items:
                answer = pipeline.ask(item.question)
                answers[item.id] = answer
                if not item.answerable:
                    key = "guard" if answer.abstained else "model" if answer.refused else "answered"
                    unanswerable[key].append(item.id)
                    continue
                name, quote = _classify(answer, chunks_by_id)
                classes[name].append(item.id)
                sources = [source_of(c) for c in answer.retrieved_ids]
                expected = set(item.expected_sources)
                rows[item.id] = {
                    "class": name, "quote": quote, "kind": item.kind, "answer": answer.text,
                    "question": item.question,
                    "recall": recall_at_k(expected, sources, k),
                    "hit": bool(expected & set(sources)), "sources": sources,
                    "expected": list(item.expected_sources),
                }
        finally:
            retriever.close()
    return {"k": k, "guard": settings.min_dense_score, "classes": classes, "rows": rows,
            "unanswerable": unanswerable, "answers": answers}


def _cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


def render(a: dict | None = None) -> str:
    a = a or analyse()
    classes, rows = a["classes"], a["rows"]
    n = sum(len(v) for v in classes.values())
    out = [
        "# Golden-set failure analysis",
        "",
        ("Generated by `evals/analysis/failure_taxonomy.py` by replaying the committed recordings "
        f"offline (guard {a['guard']:g}, k={a['k']}, no model, no network). Every number below "
        "is computed by that script; the tests re-run it and compare byte for byte."),
        "",
        f"## (a) Outcome classes of the {n} answerable items",
        "",
        ("Each answerable item is in exactly one class. Priority: refused by the guard, refused by "
        "the model, valid, no parsable citation, then the first citation problem found (unknown "
        "id, short quote, quote not found). Quotes are compared after whitespace "
        "normalisation, as the verifier does. For a quote not found, the first matching check "
        "wins: quote-character swap, elision (`...`), quote present in another retrieved "
        "chunk, case or punctuation only, paraphrase (more than 0.6 of the quote's word tokens "
        "occur in the cited chunk), other."),
        "",
        ("Caveat: the class \"differs only in case, spacing or punctuation\" is defined by a "
        "loose deterministic check (lowercase, remove every non-alphanumeric character, then a "
        "substring test). It therefore also catches quotes that are not verbatim and may not "
        "support the claim (g015 is an example). The real cause of each difference (for example "
        "markup such as backticks or asterisks in the stored chunk text) was NOT individually "
        "confirmed."),
        "",
        "| Class | Count | Items |",
        "| --- | --- | --- |",
    ]
    out += [f"| {name} | {len(ids)} | {_ids(ids)} |" for name, ids in classes.items()]
    out += [f"| **Total** | **{n}** | |", ""]

    out += [
        "## (b) By kind and by retrieval outcome", "",
        ("Retrieval hit means at least one expected source document is among the top "
        f"{a['k']} retrieved chunks."), "",
        "| Class | single | multi | expected source retrieved | not retrieved |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, ids in classes.items():
        cells = [
            sum(rows[i]["kind"] == "single" for i in ids),
            sum(rows[i]["kind"] == "multi" for i in ids),
            sum(rows[i]["hit"] for i in ids),
            sum(not rows[i]["hit"] for i in ids),
        ]
        out.append(f"| {name} | " + " | ".join(map(str, cells)) + " |")
    totals = [
        sum(r["kind"] == "single" for r in rows.values()),
        sum(r["kind"] == "multi" for r in rows.values()),
        sum(r["hit"] for r in rows.values()),
        sum(not r["hit"] for r in rows.values()),
    ]
    out += ["| **Total** | " + " | ".join(f"**{t}**" for t in totals) + " |", ""]

    low = [i for i in rows if rows[i]["recall"] < 1.0]
    out += [
        f"## (c) Answerable items with recall below 1.0 ({len(low)})", "",
        "| Item | Kind | Recall | Expected | Retrieved sources | Outcome class |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i in low:
        r = rows[i]
        out.append(
            f"| {i} | {r['kind']} | {r['recall']:.2f} | {', '.join(r['expected'])} | "
            f"{', '.join(r['sources'])} | {r['class']} |"
        )
    out.append("")

    un = a["unanswerable"]
    out += [
        f"## (d) Unanswerable items ({sum(len(v) for v in un.values())})", "",
        f"- Refused by the guard: {len(un['guard'])} ({_ids(un['guard'])}).",
        f"- Refused by the model: {len(un['model'])} ({_ids(un['model'])}).",
        f"- Answered (failure): {len(un['answered'])} ({_ids(un['answered'])}).", "",
    ]
    for i in un["answered"]:
        out.append(f"- {i} answered: {_cut(a['answers'][i].text, 200)}")
    out.append("")

    failing = [
        (name, ids) for name, ids in classes.items()
        if ids and name not in (VALID, GUARD, MODEL_REFUSED)
    ]
    top_name, top_ids = max(failing, key=lambda pair: len(pair[1]))
    out += [
        "## (e) Examples of the most common failure class", "",
        (f'The most common failure class is "{top_name}" ({len(top_ids)} of {n} answerable '
        "items). Up to 5 examples, answers cut at 300 characters."), "",
    ]
    for i in top_ids[:5]:
        r = rows[i]
        out += [
            f"**{i}**: {r['question']}", "",
            "```text", _cut(r["answer"], 300), "```", "",
            f"Offending quote: `{_cut(r['quote'] or '', 200)}`", "",
        ]

    valid = len(classes[VALID])
    invalid = n - valid
    misses = sum(1 for r in rows.values() if r["class"] != VALID and not r["hit"])
    out += [
        "## (f) What this suggests",
        "",
        (f"Only {valid} of {n} answerable items ended with a valid citation. Most of the rest "
        "are citation or refusal outcomes; see the retrieval columns in (b). Nothing below was tested; these are readings of the "
        "counts, not claims that a fix works."),
        "",
        ("- Classes that look addressable by the system: quote-character swaps and "
        "case, spacing or punctuation differences could be absorbed by a more tolerant "
        "comparison, but a looser check also weakens what a valid citation guarantees, so "
        "that is a trade-off to decide on purpose. Elided quotes, paraphrases and wrong ids "
        "might respond to a different label format or to one citation per sentence. Each "
        "would need its own measurement."),
        ("- Classes that look like model behaviour: copying a quote word for word from a "
        "small quantised model, refusing answerable questions, and answering unanswerable "
        "ones from memory. A larger model could change them, which is also untested."),
        (f"- Of the {invalid} answerable items without a valid citation, {misses} had no "
        f"expected source in the top {a['k']}, so retrieval explains at most {misses} of "
        f"them; the other {invalid - misses} had an expected source retrieved."),
        ("- The dev set (20 items, 12 answerable) gave citation validity 0.67 for the same "
        f"configuration and the golden set gives {valid / n:.2f}. That is not surprising: "
        "with 12 answerable dev items one item moves the rate by 0.08, and the prompt, labels, "
        "k and parser were tuned on the 20-item dev set (see "
        "docs/adr/0005-prompt-iteration.md), not on the golden items, so some of the dev "
        "result reflects fitting to those 20 items. The dev result was a development signal, "
        "not a prediction."),
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
