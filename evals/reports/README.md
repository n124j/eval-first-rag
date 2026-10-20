# Evaluation reports

## The v0 evidence (first live measurement)

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
`evals/baseline.json` does not exist yet and is not derived from v0.

## What comes next

1. Iterate on the prompt using a development split of questions, not the frozen golden set,
   so the golden set stays an honest held-out measure.
2. Add retrieval-confidence abstention: when retrieval scores are weak, refuse without
   asking the model, which addresses the unanswerable questions directly.
3. Record the final run on the frozen golden set, commit it under `evals/recordings/`
   together with the baseline, and let CI replay it offline through the regression gate.
