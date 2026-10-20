# ADR 0005: Prompt iteration on the dev set

Status: accepted (winner committed; the result is negative, see "Outcome")

## Context

The first live evaluation (`evals/reports/README.md`, the `v0-prompt-1b-*` files) found that
retrieval works (golden recall@5 0.90, MRR 0.90) but that `llama-3.2-1b-instruct` with the v0
prompt produced no valid citation of the form `[chunk_id: "exact quote"]` and no correct
refusal. 42 of 60 answers began by echoing a bare `[chunk_id]` header copied from the context.
The director approved a short, measured tuning loop on a separate dev set before adding a
retrieval-confidence abstention check (a later task) and recording once on the frozen golden set.

## Method

- Measurement set: `evals/dev/dev.jsonl` only (20 items: 10 single, 2 multi, 8 unanswerable).
  The golden set was not opened, read or evaluated. It stays held out.
- Fixed: model `llama-3.2-1b-instruct` (Q8_0, LM Studio, localhost), temperature 0.0,
  `max_tokens` 512, `seed` unset, embedding model `text-embedding-nomic-embed-text-v1.5`, the
  index in `data/` (chunk size 200 words, overlap 40, 1289 chunks), the refusal text
  `Not found in the provided documents.` and exact-match `is_refusal`.
- Levers: system prompt wording and a made-up few-shot example; how the context is labelled
  in the user message; `k` (5 or 3); and, only after two prompt-only rounds failed, the
  citation parser. Chunking was not changed.
- Budget: at most 4 candidate rounds, each measured once on the dev set and all reported.
  Round 0 is the unchanged v0 configuration measured on the dev set as the baseline the
  stopping rule refers to. It is not one of the 4 candidates.
- Stopping rule: stop when `citation_validity_rate >= 0.60` while `recall_at_k` is not more
  than 0.10 below round 0's. Winner: highest `citation_validity_rate`, ties broken by higher
  `recall_at_k`, then lower `prompt_tokens_total`.
- Each round ran `python -m evalrag eval --golden evals/dev/dev.jsonl --k <k>` live against
  localhost with `--record-llm` into a scratch directory outside the repo. Query embeddings
  were cached across rounds in a scratch `qemb.npz`. Every completion was then read, item by
  item, by replaying the scratch recordings offline through the pipeline.
- Date 2026-10-20. Hardware: AMD Ryzen 9 3900X, 128 GB RAM, Radeon RX 6600 XT 8 GB; served by
  LM Studio (GPU offload setting not recorded).

## Results

| Round | Change from the previous round | k | recall_at_k | mrr | citation_validity_rate | correct_refusal_rate | latency_p50_s | latency_p95_s | prompt_tokens_total | completion_tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | v0 baseline: v0 prompt, `[chunk_id]` header above each chunk | 5 | 0.9167 | 0.9583 | 0.0000 | 0.0000 | 2.416 | 4.897 | 34170 | 2809 |
| 1 | New system prompt with an inline example; `Source id:` / `Text:` labels; question before and after the sources | 5 | 0.9167 | 0.9583 | 0.0000 | 0.1250 | 2.514 | 8.224 | 37442 | 3583 |
| 2 | Example moved into two few-shot chat turns (answer, then refusal); shorter system prompt without a placeholder template; k 5 to 3 | 3 | 0.9167 | 0.9583 | 0.0000 | 0.6250 | 0.479 | 1.363 | 26791 | 343 |
| 3 | Few-shot order reversed (refusal, then answer); example texts use `"quoted"` identifiers like the corpus; parser accepts `"` inside a quote | 3 | 0.9167 | 0.9583 | 0.0000 | 0.1250 | 0.906 | 1.303 | 27251 | 730 |
| 4 | As round 3, but the prompt asks for a short quote with no quotation marks, and the example quote has none | 3 | 0.9167 | 0.9583 | 0.0000 | 0.1250 | 0.665 | 1.277 | 27291 | 531 |

`recall_at_k` and `mrr` are computed at each round's own k. On this dev set, recall at 3 equals
recall at 5: every expected page found in the top 5 is already in the top 3. The two multi
items reach only 0.5 recall at either k (`shutil` and `csv` are never retrieved).

Behaviour counts (read from the recorded completions; 12 answerable and 8 unanswerable items):

| Round | Answerable refused | Answerable with a parsable citation | Unanswerable refused | Begins with a copied id or label line | Copies the made-up example |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 15 | 0 |
| 1 | 0 | 0 | 1 | 16 | 0 |
| 2 | 10 | 2 | 5 | 0 | 2 |
| 3 | 5 | 5 | 1 | 0 | at least 2 |
| 4 | 4 | 7 | 1 | 0 | 7 |

The stopping rule never triggered: every round has `citation_validity_rate` 0.0.

## Prompts

Each block is the resolved message list. `<question>`, `<id>` and `<text>` stand for the real
values, and one source block is repeated per retrieved chunk. Line breaks are single newlines,
exactly as sent.

### Round 0 (v0 baseline)

```text
--- system
You answer questions using only the context provided in the user message. Do not use outside knowledge.
Cite every claim as [chunk_id: "exact quote"], where chunk_id is the id shown in square brackets above a context block and the quote is copied verbatim from that chunk and is at least 8 characters long.
If the context does not contain the answer, reply with exactly Not found in the provided documents. and nothing else.
--- user
[<id>]
<text>

Question: <question>
```

Parser: `\[([\w.\-]+#\d{3}):\s*"([^"]+)"\]` (unchanged in rounds 0 to 2).

### Round 1

```text
--- system
You answer questions about Python documentation using only the sources in the user message. Do not use outside knowledge.
Write a short answer of one to three sentences. After each sentence put a citation in this exact form: [source id: "exact quote"]. The source id is the value on the "Source id:" line of a source, for example widgets#004. The quote is copied word for word from that source's text and is at least 8 characters long. Do not copy the sources; answer the question.
If the sources do not answer the question, reply with exactly Not found in the provided documents. and nothing else.

Example with made-up sources:
Source id: widgets#004
Text: The spin() function rotates a widget. It returns the new angle in degrees.
Question: What does spin() return?
Answer: spin() returns the new angle [widgets#004: "returns the new angle in degrees"].
Question: How do I paint a widget blue?
Answer: Not found in the provided documents.
--- user
Question: <question>

Sources:

Source id: <id>
Text: <text>

Question: <question>
Answer:
```

### Round 2 (winner, committed)

```text
--- system
You answer questions using only the sources in the user message. Do not use outside knowledge and do not copy the sources.
Answer in one or two short sentences. End each sentence with a citation: the source id, a colon, and a quote copied word for word from that source, all in square brackets, like [widgets#004: "returns the new angle in degrees"]. The quote must be at least 8 characters long.
If the sources do not answer the question, reply with exactly Not found in the provided documents. and nothing else.
--- user
Question: What does spin() return?

Sources:

Source id: widgets#004
Text: The spin() function rotates a widget. It returns the new angle in degrees.

Source id: widgets#007
Text: The hide() function removes a widget from the screen until show() is called.

Question: What does spin() return?
Answer:
--- assistant
spin() returns the new angle of the widget [widgets#004: "returns the new angle in degrees"].
--- user
Question: How do I paint a widget blue?

Sources:

Source id: widgets#002
Text: A widget has a name and a size. The size is given in pixels.

Source id: gadgets#010
Text: The gadget registry keeps one entry for every gadget that was created.

Question: How do I paint a widget blue?
Answer:
--- assistant
Not found in the provided documents.
--- user
Question: <question>

Sources:

Source id: <id>
Text: <text>

Question: <question>
Answer:
```

### Round 3

```text
--- system
You answer questions using only the sources in the user message. Do not use outside knowledge and do not copy the sources.
Answer in one or two short sentences. End each sentence with a citation: the source id, a colon, and a quote copied word for word from that source, all in square brackets, like [widgets#004: "returns the new angle in degrees"]. The quote must be at least 8 characters long. Copy it exactly, including any quotation marks.
If the sources do not answer the question, reply with exactly Not found in the provided documents. and nothing else.
--- user
Question: How do I paint a widget blue?

Sources:

Source id: widgets#002
Text: A widget has a "name" and a "size". The size is given in pixels.

Source id: gadgets#010
Text: The gadget registry keeps one entry for every gadget that was created.

Question: How do I paint a widget blue?
Answer:
--- assistant
Not found in the provided documents.
--- user
Question: What does spin() return?

Sources:

Source id: widgets#004
Text: The "spin()" function rotates a widget and returns the new angle in degrees.

Source id: widgets#007
Text: The "hide()" function removes a widget from the screen until "show()" is called.

Question: What does spin() return?
Answer:
--- assistant
spin() returns the new angle of the widget [widgets#004: "The "spin()" function rotates a widget and returns the new angle in degrees"].
--- user
Question: <question>

Sources:

Source id: <id>
Text: <text>

Question: <question>
Answer:
```

Parser in rounds 3 and 4: `\[([\w.\-]+#\d{3}):\s*"(.+?)"\]` (a quote may contain `"` and ends at
the first `"]`). Verification was unchanged.

### Round 4

```text
--- system
You answer questions using only the sources in the user message. Do not use outside knowledge and do not copy the sources.
Answer in one or two short sentences. End each sentence with a citation: the source id, a colon, and a quote copied word for word from that source, all in square brackets, like [widgets#004: "returns the new angle in degrees"]. Pick a short quote of a few words that supports your sentence and contains no quotation marks; it must be at least 8 characters long and copied exactly.
If the sources do not answer the question, reply with exactly Not found in the provided documents. and nothing else.
--- user
Question: How do I paint a widget blue?

Sources:

Source id: widgets#002
Text: A widget has a "name" and a "size". The size is given in pixels.

Source id: gadgets#010
Text: The gadget registry keeps one entry for every gadget that was created.

Question: How do I paint a widget blue?
Answer:
--- assistant
Not found in the provided documents.
--- user
Question: What does spin() return?

Sources:

Source id: widgets#004
Text: The "spin()" function rotates a widget and returns the new angle in degrees.

Source id: widgets#007
Text: The "hide()" function removes a widget from the screen until "show()" is called.

Question: What does spin() return?
Answer:
--- assistant
spin() returns the new angle of the widget [widgets#004: "returns the new angle in degrees"].
--- user
Question: <question>

Sources:

Source id: <id>
Text: <text>

Question: <question>
Answer:
```

## Analysis per round

**Round 0 (baseline).** The dev set reproduces the v0 failure exactly. 15 of 20 completions
begin with a copied `[id]` line, sometimes an id that was not retrieved (d001 echoed
`fractions#005`). Most then paste chunk text or answer from memory (d020: "Leonardo da
Vinci."). No completion contains a parsable citation. One answerable item was refused (d002).
d018 came close to a correct refusal but put an `[id]` line before the refusal text.

**Round 1.** Replacing the bracketed header with `Source id:` / `Text:` labels did not stop
the echo. The model now echoes the new labels instead: it pastes whole `Source id: ... Text: ...`
blocks (16 of 20). It also copied the placeholder in the instruction literally and wrote
`[source id: fractions#004]` with no quote. And it continued the inline example as if it were
part of the conversation (d009 and d011 end with "Question: How do I paint a widget blue?
Answer: Not found ..."). Lesson: with a 1B model, every literal string in the prompt, the
placeholder template included, is a candidate for copying, and an inline example in the system
prompt reads as text to continue.

**Round 2.** Moving the example into real user/assistant turns changed the behaviour the most
of any round. Header and label echo went to 0. Completions are short: 343 completion tokens
against 2809 in round 0, and latency fell about fivefold. 5 completions use the exact
`[id: "quote"]` form, but only 2 of those 5 are on answerable items (d001, d009). The other 3
are on unanswerable items (d015, d016, d017), where a citation does not count. The model also
over-refuses: 10 of 12 answerable questions get the
refusal text. The last example turn was the refusal, and the model imitates the most recent turn.
The two answerable citations both fail verification. d001 rewrote the corpus's `"limit_denominator()"`
as `'limit_denominator()'`. d009 invented a regular expression as the "quote". The 5 refusals of
unanswerable items (0.625) are a side effect of the over-refusal, not real discrimination: on
answerable items the refusal rate is even higher (10 of 12).

**Round 3.** Two problems were visible after round 2. One was the over-refusal (recency).
The other was that the corpus wraps identifiers in double quotes (`"Fraction"`,
`"numbers.Rational"`), which the `[^"]+` quote pattern cannot carry. Rounds 1 and 2 had both
scored 0.0, so the parser lever was allowed. Round 3 put the answer example last, made the
example texts use quoted identifiers, asked for exact copying "including any quotation marks",
and let the parser accept `"` inside a quote, ending at the first `"]`. Verification was
unchanged. Answerable refusals fell from 10 to 5, and 5 answerable items carried parsable
citations with retrieved ids. All 5 still failed verification. The model systematically
rewrites `"x"` inside a quote as `'x'` or `\"x\"`, and several quotes are unrelated to the claim
(d008 cites a `ValueError` sentence for an answer about `auto`). Refusal of unanswerable items
fell back to 1 of 8.

**Round 4.** This tried to avoid the quote-mark problem from the prompt side: pick a short quote
with no quotation marks, and show such a quote in the example. Parsable citations rose to 7
answerable items, but the model mostly ignored the no-quotation-marks instruction. Worse, it
copied the made-up example answer verbatim
(`spin() returns the new angle ... [widgets#004: "returns the new angle in degrees"]`) in 7 of
20 items, including d020 ("The painter of the Mona Lisa was Leonardo da Vinci [widgets#004: ...]").
Round 4 is worse than round 3 in substance even though the metrics tie.

**Diagnostic, not a round.** Offline, on the recorded completions only, I checked how many
answerable items would have passed if `'` and `\"` were treated as `"`. The answer is 1 in
round 2, 3 in round 3 (d008, d010, d011) and 3 in round 4. That is still below 0.30, and in
several of those cases the quote does not support the sentence it is attached to. Normalising
quote marks would therefore raise the number without making the answers grounded. It was not
adopted (see Rejected).

## Outcome

- Winner: **round 2**. All four candidates (and round 0) tie at `citation_validity_rate` 0.0
  and `recall_at_k` 0.9167, so the tie-break on `prompt_tokens_total` decides: round 2 has
  26791, against 27251 (round 3) and 27291 (round 4). It is also the only candidate that removes
  the header and label echo without the model copying the example into most answers.
- Committed configuration: the round 2 system prompt and few-shot turns
  (`SYSTEM_PROMPT`, `FEW_SHOT` in `src/evalrag/generation/answer.py`), the
  question / `Source id:` / `Text:` / question / `Answer:` layout, and `top_k` 3
  (`DEFAULT_TOP_K` in `src/evalrag/config.py`). The `eval` and `refresh-recordings` commands now
  take `--k` from the `TOP_K` setting unless it is given, so the pipeline and the metrics use the
  same k. The parser is unchanged from v0, because the round 3 parser change was not part of the
  winner. That the committed code reproduces the round 2 metrics was verified locally against
  the scratch recordings (not committed).
- Known cost of the winner: on the dev set it refuses 10 of 12 answerable questions. Its
  `correct_refusal_rate` of 0.625 must not be read as the prompt "learning to abstain".
- **The 1B model is judged insufficient for this citation format.** The winner's
  `citation_validity_rate` is 0.0, far below the 0.30 floor, and no prompt or layout tried here
  moved it. The model can be steered into the bracket syntax (rounds 2 to 4), but not into copying
  a span of source text verbatim, and it cannot tell answerable from unanswerable questions from
  the context. Recommendation (nothing was downloaded): measure a larger instruction-tuned model
  class, roughly 7B to 8B parameters at Q4 or better (about 5 GB, which fits the 8 GB GPU), with
  the round 2 configuration and the same dev procedure before touching the golden set. Changing
  the model needs the director's approval.
- Refusals stay poor with this model. The planned retrieval-confidence abstention check (a
  separate task) is the right lever for unanswerable questions. It was not implemented here.

## Limitations

- The dev set has 20 items (12 answerable, 8 unanswerable), written by a single author. One item
  moves a rate by 0.083 (answerable) or 0.125 (unanswerable). The differences between rounds are
  within noise, except the round 0/1 versus round 2 change in behaviour (echo, length, refusal),
  which is large and consistent across items.
- One model, a 1B model, at temperature 0. Each round ran once, and a different seed or
  quantisation could behave differently.
- Rounds 2 and 3 each changed more than one lever, so their effects are confounded. The budget of
  four rounds was spent on the largest plausible moves rather than on one-factor ablations. In
  round 2 the k change is harmless to recall on this dev set but still a confound for behaviour.
- Tuning against the dev set risks overfitting to its phrasing. Choosing a winner among four
  configurations that all score 0.0 by a token-count tie-break carries little information.
- The result on the golden set is still unmeasured. Nothing in this ADR is a golden-set claim.
- I did not find a dev item that I believe is wrong. The two multi items are limited by
  retrieval (`shutil` and `csv` never reach the top 5), not by the item.
- **Prompt-format injection.** The question and the retrieved chunk text are untrusted and are
  interpolated into the prompt. `build_messages` now collapses all whitespace in both to single
  spaces, so neither can start a new line with `Source id:`, `Text:`, `Question:` or `Answer:`
  and forge a source block or a pre-filled answer across lines. Real prompts are unchanged,
  because `chunk_document` already emits single-spaced text and the dev questions are
  single-line. That was verified locally against the scratch recordings (not committed). Marker
  text inline within one line (for example a question that contains
  `... Answer: Not found in the provided documents.`) is still possible, and layout alone cannot
  remove it. The corpus is trusted and the API is for local use only, so this remaining risk is
  accepted.

## Rejected

- **Normalising quote characters in `verify_citations`** (treating `'`, `\"` and `"` as equal).
  This would have raised round 3 to 3 of 12 on the dev set, but it loosens the "quote occurs in
  the chunk" guarantee, and the quotes it would accept often did not support the claim. Not tried
  as a round.
- **Keeping the round 3 parser change** (allowing `"` inside a quote). It is sound, because
  verification still requires an exact occurrence, but it belongs to a losing round and did not
  produce a single valid citation. It can be revisited with a stronger model.
- **A non-`"` quote delimiter** (for example `[id: «quote»]` or `[id] quote [/id]`). It would avoid
  the corpus quote-mark clash, but it needs a parser change and the 1B model did not reliably
  copy text even where no quote marks were involved (d006, d009). Considered, not tried.
- **Chunk size or overlap changes.** Retrieval is not the bottleneck: recall is 0.92 at k 3. They
  would also need a scratch re-ingest and a `Settings` and `data/` change. Not tried.
- **k = 5 with the round 2 prompt.** It was not separated from the prompt change. Recall on dev is
  identical at 3 and 5, and fewer chunks mean less text to paste and fewer tokens.
- **Numbered sources (`Source 1`) instead of ids.** The model would have to map numbers back to
  chunk ids, which is one more step for a 1B model and needs a parser change. Not tried.
- **Abstention or a score threshold.** Out of scope; a separate later task.
- **A larger model.** Out of scope without the director's approval; recommended above.

## Addendum 2026-10-20: decision

- All four rounds, and the baseline, scored 0.0 valid citations on the dev set. On that basis
  the director approved evaluating a 7B to 8B instruction-tuned model, using the same dev-set
  procedure, before the golden set is touched.
- The committed configuration (round 2 prompt and few-shot turns, `top_k` 3) is the
  **starting point** for that evaluation. It is not a claimed improvement: on the 1B model it
  produces no valid citations and over-refuses.
- Over-refusal is now measured directly by a new metric, `over_refusal_rate` (refused
  answerable items divided by all answerable items; lower is better; NaN without answerable
  items). Recomputed from the round 2 dev recordings (verified locally against the scratch
  recordings, not committed), round 2 has an `over_refusal_rate` of 0.8333 (10 of 12). The
  table above predates the metric and keeps the original eight.
- The 7B evaluation is recorded in "Addendum: evaluation with qwen2.5-7b-instruct-1m"
  below.

## Addendum: evaluation with qwen2.5-7b-instruct-1m (2026-10-20)

### Why

With `llama-3.2-1b-instruct`, every round scored 0.0 valid citations. The director approved
evaluating a larger model class and loaded it in LM Studio. This addendum measures it with the
same dev-set procedure. The golden set was not used.

### Setup

- Generation model: `qwen2.5-7b-instruct-1m` (lmstudio-community Qwen2.5-7B-Instruct-1M, Q4_K_M,
  4.68 GB, a long-context variant of Qwen2.5-7B-Instruct), served by LM Studio on localhost.
  Temperature 0.0, `max_tokens` 512 and `seed` unset were unchanged.
- Embeddings, `data/` index (200/40), dev set and hardware are as above.
- Each round ran live on the dev set with the retrieval-confidence guard disabled
  (`--min-dense-score 0`, ADR 0006), recording completions and query embeddings into a scratch
  directory outside the repo. The guard-on row (`--min-dense-score 0.67`, the default) was then
  computed by replay from those recordings, with no live calls.
- Budget: round 8B-0 (the committed configuration unchanged) plus at most two more. Stop rule:
  8B-0 guard off with `citation_validity_rate >= 0.60` and `over_refusal_rate <= 0.25`.
  Winner: highest guard-off `citation_validity_rate`, ties broken by lower `over_refusal_rate`,
  then higher `correct_refusal_rate`, then lower `prompt_tokens_total`.

### Results

| Round | Guard | k | recall_at_k | mrr | citation_validity_rate | correct_refusal_rate | over_refusal_rate | latency_p50_s | latency_p95_s | prompt_tokens_total | completion_tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8B-0 | off | 3 | 0.9167 | 0.9583 | 0.4167 | 0.6250 | 0.1667 | 4.628 | 6.111 | 27453 | 961 |
| 8B-0 | on (0.67, replay) | 3 | 0.9167 | 0.9583 | 0.4167 | 0.6250 | 0.1667 | 4.628 | 6.111 | 24811 | 947 |
| 8B-1 | off | 3 | 0.9167 | 0.9583 | 0.5000 | 0.5000 | 0.1667 | 4.626 | 6.706 | 28033 | 1006 |
| 8B-1 | on (0.67, replay) | 3 | 0.9167 | 0.9583 | 0.5000 | 0.5000 | 0.1667 | 4.626 | 6.706 | 25333 | 992 |
| 8B-2 | off | 3 | 0.9167 | 0.9583 | 0.6667 | 0.6250 | 0.1667 | 4.511 | 6.034 | 27453 | 961 |
| 8B-2 | on (0.67, replay) | 3 | 0.9167 | 0.9583 | 0.6667 | 0.6250 | 0.1667 | 4.511 | 6.034 | 24811 | 947 |

Wall-clock time per live round (20 questions): 115 s for 8B-0 (this includes embedding the 20
queries for the first time), 94 s for 8B-1 and 90 s for 8B-2. Mean generation latency per
question with the guard off: 4.52 s, 4.59 s and 4.45 s. With the guard on, abstained items
count as 0 s, which gives 4.26 s (8B-0), 4.32 s (8B-1) and 4.19 s (8B-2).

In every round the guard abstained on exactly two items, d019 and d020 (the out-of-domain
questions). The model had already refused both, so the guard changes only the token and latency
totals, not the headline rates. Guard-on and guard-off latency and token figures are not
comparable as model cost (ADR 0006).

### Prompts

8B-0 and 8B-2 use the committed round 2 prompt and few-shot turns unchanged (see "Round 2
(winner, committed)" above). 8B-1 changed only the system prompt:

```text
--- system
You answer questions using only the sources in the user message. Do not use outside knowledge and do not copy the sources.
Answer in one or two short sentences. End each sentence with a citation: the source id, a colon, and a quote copied word for word from that source, all in square brackets, like [widgets#004: "returns the new angle in degrees"]. The quote must be at least 8 characters long. Copy it character for character, with the same capitalisation and punctuation, and choose a part of the source that contains no double quotation marks (").
If the sources do not answer the question, reply with exactly Not found in the provided documents. and nothing else.
```

The few-shot turns and the user-message layout in 8B-1 are the round 2 ones.

8B-2 changed only the citation parser, not the prompt. The quote pattern went from
`\[([\w.\-]+#\d{3}):\s*"([^"]+)"\]` to `\[([\w.\-]+#\d{3}):\s*"((?:(?!"\]).)+?)"\]`. A quote may
now contain `"`. It ends at the first `"]` on the same line and never contains `"]`, so an empty
quote cannot swallow the next citation. `verify_citations` is unchanged: the id must be among
the retrieved chunks, and the quote must be at least 8 normalised characters and occur in that
chunk.

### Analysis per round

**8B-0 (committed configuration, unchanged).** The 7B model follows the contract far better
than the 1B. Every non-refusal is one or two sentences ending in a single citation of the exact
`[id: "quote"]` form. There is no header echo, no pasted context, and no copying of the
few-shot example. Of 12 answerable items, 2 were refused (d002, d010) and 10 were answered with
a citation. 5 of those 10 verified. The other 5 broke down as follows:

- Three were parse failures only (d006, d008, d009). The model copied the quote verbatim,
  including the corpus's double-quoted identifiers (`"TextWrapper"`, `"auto"`, `"Match"`), and
  the old `[^"]+` pattern cannot carry a `"`. This is a new format problem that did not show
  with the 1B model, which never copied verbatim.
- Two were real copy errors. d001 rewrote `"self"` as `'self'` and dropped the quotes around
  `Fraction`. d005 changed `Return the low median` to `returns the low median`.

Of the 8 unanswerable items, 5 were refused. d013 answered from memory and cited an id that was
never retrieved (`struct#003`). d014 answered `ast.literal_eval` from memory, citing an
unrelated `typing` chunk. d016 answered "Yes" from memory with a real but irrelevant quote,
which is a verified citation on an unanswerable item.

Validity is not support. d007's quote really does occur in the chunk, but the leap-year rule in
the answer comes from memory. d012's quote is real, but the answer's CSV step is not in the
sources. The metric checks that the quote exists, not that it supports the sentence.

Below the stop thresholds (validity 0.4167 < 0.60), so tuning continued.

**8B-1 (prompt-only: copy exactly, avoid double quotes).** The rule allows the parser lever
only after a prompt-only variant fails, so this tried to steer the quote choice from the prompt
side. Results:

- It fixed d005: the model picked `the smaller of the two middle values is returned`, which
  verifies. Validity rose to 0.5000.
- The model ignored the no-double-quotes instruction where it mattered. d006 and d009 still
  quote text containing `"`. d008 avoided quotes but paraphrased (`ValueError is raised with the
  details`), and d001 still writes `'self'`.
- It got worse elsewhere. d013 cited the made-up example id `widgets#004`, and d017 speculated
  instead of refusing, so `correct_refusal_rate` fell to 0.5000.

Prompt-only steering clearly failed on the inner-quote problem. An offline check (diagnostic,
not a round) parsed the 8B-0 and 8B-1 recordings with an inner-quote parser. Both reached 8 of
12, which showed that the parser, not the prompt, was the limit.

**8B-2 (committed prompt plus inner-quote parser).** The prompt was identical to 8B-0, so every
recorded request matched, and at temperature 0 the 20 live completions were byte-identical to
8B-0's. The difference is entirely the parser. d006, d008 and d009 now verify:
`citation_validity_rate` 0.6667 (8 of 12), with `over_refusal_rate` (0.1667) and
`correct_refusal_rate` (0.6250) unchanged. The 8B-0 problems other than parsing all remain:
d001 and d005 copy errors, d002 and d010 over-refusals, d013, d014 and d016 answered from
memory, and validity not implying support.

### Winner

**8B-2.** It has the highest guard-off `citation_validity_rate` (0.6667, against 0.5000 and
0.4167), the lowest `over_refusal_rate` (0.1667, tied with the others), and the highest
`correct_refusal_rate` (0.6250, tied with 8B-0). It also meets the thresholds of the stop rule
(validity >= 0.60, over-refusal <= 0.25). Committed: the parser change and its tests. The
prompt, few-shot turns, layout and `top_k` 3 are unchanged. That the committed code reproduces
the 8B-2 metrics, with the guard off and on, was verified locally against the scratch
recordings (not committed).

### Limitations

- 20 dev items (12 answerable, 8 unanswerable) from one author. One item moves a rate by 0.083
  or 0.125. The 8B-0 to 8B-2 gain is 3 items, and the cause is directly visible (parse failures
  on identical completions). The 8B-0 to 8B-1 differences are within noise.
- Dev set only. The golden-set result is still unmeasured, and nothing here is a golden-set
  claim.
- The served model is the long-context `-1m` variant of Qwen2.5-7B-Instruct at Q4_K_M. Its
  behaviour can differ from the standard 7B instruct release or other quantisations. Each round
  ran once at temperature 0.
- `citation_validity_rate` checks that a quote exists in a retrieved chunk, not that it supports
  the claim. Several valid citations decorate answers drawn from the model's own knowledge
  (d007, d012, and d016 on an unanswerable item).
- 3 of 8 unanswerable items were still answered from memory. The guard at 0.67 does not catch
  them, because their retrieval scores are above the threshold. Only d019 and d020 abstain, and
  the model already refused both.

### Recommendation

The 7B class is judged sufficient on the dev set to proceed: 8B-2 meets the
pre-agreed thresholds (validity 0.6667 >= 0.60, over-refusal 0.1667 <= 0.25), so no larger model
is requested at this point. Next, record the golden set once with this configuration, guard off
at recording time (ADR 0006), and report guard-on and guard-off results. If the golden results
fall clearly below the dev numbers, the next candidates are a quote-support check (beyond
existence) or a 14B-class model (about 9 GB, partly offloaded to the CPU), each needing the
director's approval.
