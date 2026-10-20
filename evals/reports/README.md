# Evaluation reports

## The v0 evidence (first live measurement)

Note: the v0 description below (k=5, bare `[chunk_id]` context header, 1B model) is the historical v0 configuration; see `docs/adr/0005-prompt-iteration.md` for what replaced it.

The three `v0-prompt-1b-*` files are the first live evaluation of this system. They are
kept on purpose: this project is "eval-first", so a measured failure and the fix that
follows are part of the story.

| File | What it is |
| --- | --- |
| `v0-prompt-1b-report.md` | The generated evaluation report (settings, metrics, one line per failing item). |
| `v0-prompt-1b-metrics.json` | The eight metrics as machine-readable JSON (the format the regression gate reads). |
| `v0-prompt-1b-completions.jsonl` | The 60 recorded model completions (plus the cache's meta header line). This is the evidence of the failure. |

Measurement conditions:

- Date: 2026-10-20.
- Generation model: `llama-3.2-1b-instruct` (Q8_0), served by LM Studio, temperature 0.
- Embedding model: `text-embedding-nomic-embed-text-v1.5` (Q4_K_M), served by LM Studio.
- Hardware: AMD Ryzen 9 3900X, 128 GB RAM, Radeon RX 6600 XT 8 GB (GPU offload setting not recorded).
- Prompt: v0, the `SYSTEM_PROMPT` as of commit `e79d4fb` (reproduced below).
- Retrieval: top `k=5` chunks. Golden set: 60 items (40 single-document, 10 multi-document, 10 unanswerable).

### Metrics

| Metric | Value |
| --- | --- |
| recall_at_k | 0.9000 |
| mrr | 0.9017 |
| citation_validity_rate | 0.0000 |
| correct_refusal_rate | 0.0000 |
| latency_p50_s | 1.966 |
| latency_p95_s | 5.525 |
| prompt_tokens_total | 103293 |
| completion_tokens_total | 8527 |

### Finding

Retrieval works; generation with this prompt and this 1B model does not follow the
citation contract.

- Retrieval recall@5 is 0.90 and MRR is 0.90, so the right document is usually in the
  context the model sees.
- 0 of 60 completions contained a parsable `[chunk_id: "quote"]` citation, so
  `citation_validity_rate` is 0.0. The model never produced the required format.
- Only 2 of 60 completions were the exact refusal text (g033 and g040), and both were
  for answerable questions, so they count as wrong refusals.
- All 10 unanswerable questions (g051 to g060) were answered rather than refused, so
  `correct_refusal_rate` is 0.0. The model fills the gap from outside knowledge (see g054).
- 42 of 60 completions began by echoing a bare `[chunk_id]` header copied from the
  context (for example `[urllib.parse#023]`) instead of a citation. (An earlier
  working note said 50; counting the recorded completions gives 42.)

### Examples

The completions below are copied verbatim from `v0-prompt-1b-completions.jsonl`. Each was
tied to its question by replaying the recorded completions offline through the
evaluation pipeline (no model was run) and matching the request's final
`Question: ...` line, which the cache does not store, against the golden set. The
replay reproduced `v0-prompt-1b-metrics.json` byte for byte.

**g001** (answerable) asks: "When a value is already in a sorted list, where does bisect_left place the insertion point relative to the existing equal entries?"

```text
by default, the insertion point will be before (to the left of) any existing entries.
```

Roughly right content, but no citation of any kind. It is not a valid answer under the contract.

**g003** (answerable) asks: "What kind of string does secrets.token_hex return?"

```text
secrets.token_hex([nbytes=None]) Return a random text string, in hexadecimal. The string has *nbytes* random bytes, each byte converted to two hex digits. If *nbytes* is "None" or not supplied, a reasonable default is used. >>> token_hex(16) 'f9bf78b9a18ce6d46a0cd2b0b86df9da'
```

It pastes the documentation text rather than answering, again with no citation.

**g054** (unanswerable) asks: "How do I parse an XML file and loop over all elements with a given tag name?"

````text
[urllib.parse#023]
sequence of two-element tuples is used as the *query* argument, the first element of each tuple is a key and the second is a value. The value element in itself can be a sequence and in that case, if the optional parameter *doseq* evaluates to "True", individual "key=value" pairs separated by "'&'" are generated for each element of the value sequence for the key.

You can use this information to parse an XML file and loop over all elements with a given tag name. Here's an example:
```
import xml.etree.ElementTree as ET

def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Loop over all elements with the 'tag' attribute
    for elem in root.iter():
        if elem.tag == 'tag_name':
            print(elem.text)

# Call the function with the file path
parse_xml_file('example.xml')
```
This code will parse the XML file and loop over all elements with a tag name that matches `'tag_name'`. The `iter()` method is used to iterate over all elements in the tree, and the `getroot()` method is used to get the root element of the tree.
````

The corpus has nothing on XML, so the correct answer is the refusal. The model echoes an
unrelated chunk header, then answers from outside knowledge.

### The v0 system prompt

This is the resolved text of `SYSTEM_PROMPT` at commit `e79d4fb` (with `MIN_QUOTE_CHARS`
= 8 and the refusal text filled in). Each line break below is a single newline character,
exactly as in the source; the sentences within a line are separated by single spaces.

```text
You answer questions using only the context provided in the user message. Do not use outside knowledge.
Cite every claim as [chunk_id: "exact quote"], where chunk_id is the id shown in square brackets above a context block and the quote is copied verbatim from that chunk and is at least 8 characters long.
If the context does not contain the answer, reply with exactly Not found in the provided documents. and nothing else.
```

## Why this is kept, and what it is not

The project's claim is that behaviour changes are driven by measurement. The v0 run is the
measurement that showed a small local model, given this prompt, cannot be trusted to cite
or to refuse. Keeping it makes the later improvement checkable against a real starting point.

These files are **not** CI fixtures. The completions are cached against the exact request
(model, messages, temperature, token limit), so they can only be replayed with the v0
prompt. A different prompt produces different request keys and finds nothing in this cache.
For that reason `v0-prompt-1b-completions.jsonl` lives here and not under
`evals/recordings/`, which is reserved for the final recordings that CI replays.
`evals/baseline.json` is not derived from v0; it comes from the final golden recording (next section).

## What comes next

1. Iterate on the prompt using a development split of questions, not the frozen golden set,
   so the golden set stays an honest held-out measure.
2. Add retrieval-confidence abstention: when retrieval scores are weak, refuse without
   asking the model, which addresses the unanswerable questions directly.
3. Record the final run on the frozen golden set, commit it under `evals/recordings/`
   together with the baseline, and let CI replay it offline through the regression gate
   (done, see the next section; items 1 and 2 are also done).

## Final golden-set results (qwen2.5-7b-instruct-1m)

- Date: 2026-10-20.
- Generation model: `qwen2.5-7b-instruct-1m` (Qwen2.5-7B-Instruct-1M, Q4_K_M, 4.68 GB), temperature 0, served by LM Studio.
- Embedding model: `text-embedding-nomic-embed-text-v1.5` (Q4_K_M), served by LM Studio.
- Hardware: AMD Ryzen 9 3900X, 128 GB RAM, Radeon RX 6600 XT 8 GB; served by LM Studio (GPU offload setting not recorded).
- Setup: frozen golden set, 60 items (50 answerable, 10 unanswerable), `k=3`, committed prompt and parser (ADR 0005).
- Recording: ONE live run with the retrieval-confidence guard disabled (`--min-dense-score 0`), so a completion exists for every question. It recorded 1349 embeddings and 60 completions in about 6 minutes (363 s wall clock, including embedding the 1289 chunks). These are the CI fixtures: `evals/recordings/embeddings.npz` and `evals/recordings/llm.jsonl`.
- The same recordings were then replayed with the guard on (default 0.67). Both result sets are published: `golden-guard-off{.md,-metrics.json}` and `golden-guard-on{.md,-metrics.json}`.

| Metric | Guard off (0) | Guard on (0.67, default) |
| --- | --- | --- |
| recall_at_k | 0.8800 | 0.8800 |
| mrr | 0.8967 | 0.8967 |
| citation_validity_rate | 0.3000 | 0.3000 |
| correct_refusal_rate | 0.7000 | 0.8000 |
| over_refusal_rate | 0.1800 | 0.1800 |
| latency_p50_s | 4.373 | 4.373 |
| latency_p95_s | 7.113 | 7.113 |
| prompt_tokens_total | 81651 | 76449 |
| completion_tokens_total | 3037 | 2970 |

Latency and token rows are not comparable between the columns as measures of model speed: with the guard on, an abstained question makes no LLM call and counts as zero latency and zero tokens (ADR 0006).

### Stop floor

`citation_validity_rate` 0.30 (15 of 50) sits EXACTLY at the pre-agreed stop floor. The rule was: a value below 0.30 stops the run and blocks the baseline; 0.30 continues. One fewer valid answer (14 of 50, 0.28) would have blocked the baseline.

### Pre-registered rule (ADR 0006)

The guard may stay the default unless it raises golden `over_refusal_rate` by more than 0.05. Guard off: 0.1800. Guard on: 0.1800. Difference: 0.0000. Verdict: the rule does not fire, so the shipped default stays 0.67. The guard changed one outcome: `correct_refusal_rate` rose from 0.7 to 0.8 (g051, an asyncio question that the model had answered, is now refused by the guard). The guard also abstained on one answerable item, g025, which the model had refused in the guard-off run as well, so no answerable item's outcome changed and over-refusal is 0.18 either way (see `golden-failure-analysis.md`).

### Variance

The LLM side was re-run live 3 more times with the guard disabled (scratch runs, not committed; embeddings replayed). Quality metrics were identical in all four runs (recorded run plus 3 re-runs): recall_at_k 0.88, mrr 0.8967, citation_validity_rate 0.30, correct_refusal_rate 0.70, over_refusal_rate 0.18, prompt tokens 81651, completion tokens 3037. The spread (max minus min) is 0 for all of them, as expected at temperature 0 on one server. Only timing varies:

| Metric | Recorded | Re-run 1 | Re-run 2 | Re-run 3 | Spread |
| --- | --- | --- | --- | --- | --- |
| latency_p50_s | 4.373 | 4.361 | 4.380 | 4.378 | 0.019 |
| latency_p95_s | 7.113 | 7.056 | 7.034 | 7.055 | 0.079 |

Zero spread means "no sampling noise on this machine and server build", not "robust to a different GPU offload, server version or quantisation". The baseline tolerances therefore use their floors (see `evals/baseline.json`).

### Replay determinism

Replaying the guard-off recording offline with `--min-dense-score 0` reproduces `golden-guard-off-metrics.json` (and the guard-off report) byte for byte. Replaying with the default guard reproduces `golden-guard-on-metrics.json`. Both checks run in CI as acceptance tests with the network disabled. `evals/baseline.json` is built from the guard-on row.

### What the CI gate does and does not show

The CI gate replays a FIXED recording. A passing gate shows that the code still reproduces the scores of the recorded responses (retrieval, parsing, citation verification, abstention, metrics). It cannot detect model drift or a change in live quality; those require a new recording.

Gate sensitivity: the `over_refusal_rate` tolerance is 0.10 on a baseline of 0.18, so the gate passes up to 0.28 and fails above it (with 50 answerable items the next attainable value is 0.30). The `citation_validity_rate` tolerance is 0.02 on 0.30, so a one-item drop (0.28) passes and a two-item drop (0.26) fails.

### What these numbers do and do not show

- The golden set is 60 items written by one author. One answerable item moves a rate by 0.02, one unanswerable item by 0.10. Differences of a few items are not evidence of anything.
- Retrieval is scored per document (is the right document among the top 3), not per passage, so recall_at_k overstates how precisely the right passage is found.
- `citation_validity_rate` only checks that each quoted string occurs in a retrieved chunk. A valid citation is not a supported answer: a model can quote a real sentence that does not back its claim. Only 0.30 of answerable questions got a fully valid citation, so most answers are not verifiably grounded by this metric.
- On the dev set the same model and committed code scored citation_validity_rate 0.6667, over_refusal_rate 0.1667 and correct_refusal_rate 0.6250 (ADR 0005). On the golden set validity is much lower (0.30), over-refusal is similar (0.18) and correct refusal is higher (0.70 without the guard). Dev prompt iteration therefore did not transfer fully to held-out questions, which is what a held-out set is for; with 20 and 60 items the gap is measured but its causes are not.
- The guard's effect is narrow: it only helps on questions whose best dense similarity is below 0.67. It refused one more unanswerable item (g051). It changed no answerable item's outcome: the one answerable item it refused, g025, had also been refused by the model in the guard-off run, so over-refusal is 0.18 either way. It does not address near-miss questions on topics the corpus covers.
- Results depend on this recording. Re-recording after any change to the prompt, parser, chunking, retrieval, model or threshold can move every number, and requires a new baseline.

A reproducible breakdown of the golden failures (outcome classes of the 50 answerable items, unanswerable items, retrieval versus citation failures) is in `golden-failure-analysis.md`; it is generated offline from the committed recordings by `evals/analysis/failure_taxonomy.py`.
