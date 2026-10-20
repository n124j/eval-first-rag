# ADR 0006: Retrieval-confidence abstention

Status: accepted (default `MIN_DENSE_SCORE` 0.67, calibrated on the dev set only). Amended
2026-10-20 with the golden result (see "Golden result (2026-10-20)"): the pre-registered rule
did not fire, so the default stays 0.67, but the golden benefit is one item and the dev-derived
safety margin did not hold. The golden set was not used for tuning; the threshold was fixed
before the golden measurement.

## Context

The first live evaluations showed that `llama-3.2-1b-instruct` cannot decide when it does not
know. With the v0 prompt it answered every unanswerable question. In the dev-set prompt
iteration (ADR 0005) no round was good at both: rounds 0, 1, 3 and 4 refused at most 1 of 8
unanswerable dev items, and round 2, the committed prompt, refused 5 of 8 only by also refusing
10 of 12 answerable items (`over_refusal_rate` 0.8333). Whether to refuse was, in effect, left
to a model that cannot make that call.

A classic guardrail does not ask the LLM. It refuses before generation when retrieval itself is
not confident. The director approved this as part of the plan, with the threshold calibrated on
the dev set and never on the golden set. Over-refusal is its price, and the harness measures it
with `over_refusal_rate` (refused answerable items / all answerable items).

## Design

- **Signal: the dense rank-1 cosine.** `HybridRetriever.retrieve_detailed(query, k)` returns a
  frozen `RetrievalResult(hits, top_dense_score)`. `top_dense_score` is the score of the rank-1
  hit of the dense candidate list, the cosine that `DenseIndex.search` reports (rounded to 6
  places), taken before fusion. `retrieve(query, k)` returns `retrieve_detailed(query, k).hits`,
  so there is one implementation. The cosine between the query and the closest chunk is a
  bounded similarity in [-1, 1] that is comparable across queries for one embedding model. That
  makes a single global threshold meaningful, which is not true of the fused scores (see
  Rejected).
- **No LLM in the decision.** `RagPipeline(..., min_dense_score)` checks the score. If
  `top_dense_score < min_dense_score` (strictly below; equal does not abstain), `ask` returns
  `REFUSAL_TEXT` with `refused=True`, no citations, an invalid empty citation report, the
  retrieved ids, `completion=None`, and the new `Answer.abstained=True`, all without calling the
  LLM. An abstention therefore costs no generation latency or tokens. `make_sut` reports it as
  a refusal with zero latency and tokens. The `abstained` flag lets a reader tell a guard
  refusal from a model refusal, which both have the same text.
- **Disabled is unchanged.** `min_dense_score` must be a finite number in [0.0, 1.0]. 0.0
  disables the check: the pipeline then calls `retrieve` exactly as before. A positive value
  needs a retriever with `retrieve_detailed`, and this is checked at construction (`TypeError`).
- **Configuration.** `MIN_DENSE_SCORE` (default `DEFAULT_MIN_DENSE_SCORE`, used by both the
  `Settings` dataclass default and `from_env`) reaches `ask`, `eval`, `refresh-recordings` and
  the API. The three commands also take `--min-dense-score FLOAT` for experiments. Invalid values
  are a `ConfigError` naming the key, or an argparse error (exit 2) for the flag.
- **No dense candidate.** If the dense candidate list is empty, `retrieve_detailed` reports
  `top_dense_score` 0.0. With a positive threshold that is below it, so the guard refuses with
  `abstained=True` (the retrieved ids are whatever keyword-only hits remain). This differs from
  the pipeline's older no-hit path: when retrieval returns no hits at all and the guard did not
  fire (threshold 0.0, or a retriever that reports a score at or above the threshold with no
  hits), `ask` refuses without an LLM call but with `abstained=False` and empty
  `retrieved_ids`. So `abstained` means "refused by the confidence guard", not "refused without
  calling the LLM". With the real indexes neither case occurs in practice: `DenseIndex.search`
  raises `EmptyIndex` on an empty index instead of returning nothing, and a non-empty index
  always returns at least one candidate. The two paths are therefore left as they are.
- **Replay and the final recording.** The guard decides from dense scores computed from
  whatever vectors the embedder returns. On replay those are the recorded vectors, stored as
  float16, so replay against the same recordings is deterministic. A plain live run uses
  float32 vectors (`RecordingEmbedder` already returns the float16-rounded vectors while
  recording, so a recording run sees the same values as its replay, but an unrecorded live run
  does not). A live float32 score within about 1e-5 of the threshold could therefore decide
  differently live than on replay. If such a question was recorded with the guard on and
  abstained, it would have no recorded completion, and a replay that does not abstain would
  fail with a replay miss. **Design for the final recording:** the final golden recordings are
  made with the guard DISABLED (`refresh-recordings --min-dense-score 0`), so that a completion
  is recorded for every question. The guard is then applied at replay time (`eval` with the
  default or an explicit `--min-dense-score`). An abstained question simply does not use its
  recorded completion. This removes the replay-miss risk and lets the same recordings be
  replayed with the guard on and off. The dev calibration below keeps a margin for the same
  float16 reason.
- **Metrics side effects.** `make_sut` reports an abstained item, which has no completion, with
  `latency_s` 0.0 and zero prompt and completion tokens. With the guard on, abstentions therefore
  pull `latency_p50_s`, `latency_p95_s`, `prompt_tokens_total` and `completion_tokens_total`
  down compared with a guard-off run of the same items. Latency and token figures from guard-on
  and guard-off runs are not comparable as measures of model speed or cost per answered
  question. The behaviour is intentional (an abstention really costs no generation) and is
  documented in `src/evalrag/sut.py`.
- **Reports.** `eval` and `refresh-recordings` write the effective threshold (flag or setting,
  `0` when disabled) as `min_dense_score` in the report metadata table, and the `/ask` response
  carries `abstained`.

## Calibration (dev set only)

Method. A throwaway script outside the repository embedded each of the 20 questions of
`evals/dev/dev.jsonl` live (`OpenAICompatEmbedder`, `kind="query"`, `EMBED_BASE_URL`
`http://localhost:1234/v1`, `EMBED_MODEL` `text-embedding-nomic-embed-text-v1.5`), searched the
real dense index in `data/dense.npz` (1289 chunks, chunk size 200 words, overlap 40), and
recorded the rank-1 cosine and the document of the rank-1 chunk. Only the embeddings endpoint
was used; no chat model was called, loaded or changed. The golden set was not opened, read or
evaluated. Date 2026-10-20; hardware as in ADR 0005 (AMD Ryzen 9 3900X, LM Studio on localhost).

All 20 dev items (unanswerable subtypes from `evals/dev/README.md`: near-miss on a stdlib module
that is not in the corpus, feature that an included module's page does not document, or out of
domain):

| id | kind | top dense score | expected doc | top-hit doc |
| --- | --- | --- | --- | --- |
| d001 | single | 0.839119 | fractions | fractions |
| d002 | single | 0.763545 | time | time |
| d003 | single | 0.790862 | pathlib | pathlib |
| d004 | single | 0.781836 | collections | collections |
| d005 | single | 0.770194 | statistics | statistics |
| d006 | single | 0.746542 | textwrap | textwrap |
| d007 | single | 0.760142 | calendar | datetime (calendar is dense rank 6) |
| d008 | single | 0.774663 | enum | enum |
| d009 | single | 0.750620 | re | re |
| d010 | single | 0.756795 | logging | logging |
| d011 | multi | 0.768308 | textwrap, shutil | textwrap |
| d012 | multi | 0.691150 | csv, statistics | statistics |
| d013 | unanswerable (missing module: struct) | 0.687502 | none | random |
| d014 | unanswerable (missing module: ast) | 0.706440 | none | re |
| d015 | unanswerable (missing module: queue) | 0.797336 | none | threading |
| d016 | unanswerable (missing module: inspect) | 0.718062 | none | typing |
| d017 | unanswerable (undocumented feature: zipfile Zstandard) | 0.769463 | none | zipfile |
| d018 | unanswerable (undocumented feature: calendar holidays) | 0.748009 | none | calendar |
| d019 | unanswerable (out of domain) | 0.618419 | none | fractions |
| d020 | unanswerable (out of domain) | 0.558583 | none | datetime |

Answerable range 0.691150 to 0.839119. Unanswerable range 0.558583 to 0.797336. The ranges
overlap heavily: 5 of the 8 unanswerable items (d014, d015, d016, d017, d018) score above the
lowest answerable item (0.691150, d012), and d015 scores above 11 of the 12 answerable items.
d013 (0.687502) is just below the lowest answerable score, by 0.0036, which is inside the 0.02
margin required below.

Threshold grid (abstain when the score is strictly below the threshold). Thresholds 0.30 to
0.85 in steps of 0.05, plus values in the observed gaps (0.56, 0.62, 0.67, 0.68, 0.69):

| threshold | answerable abstained (over-refusal, of 12) | unanswerable caught (of 8) | caught ids |
| --- | --- | --- | --- |
| 0.30 to 0.55 | 0 | 0 | none |
| 0.56 | 0 | 1 | d020 |
| 0.60 | 0 | 1 | d020 |
| 0.62 | 0 | 2 | d019, d020 |
| 0.65 | 0 | 2 | d019, d020 |
| **0.67** | **0** | **2** | **d019, d020** |
| 0.68 | 0 | 2 | d019, d020 |
| 0.69 | 0 | 3 | d013, d019, d020 |
| 0.70 | 1 (d012) | 3 | d013, d019, d020 |
| 0.75 | 2 (d012, d006) | 6 | d013, d014, d016, d018, d019, d020 |
| 0.80 | 11 (all but d001) | 8 | all |
| 0.85 | 12 | 8 | all |

Selection rule, applied exactly: choose the largest threshold, rounded down to two decimals,
that abstains on zero answerable dev items and keeps a margin of at least 0.02 between the
threshold and the lowest answerable top score. The margin absorbs the difference between
replayed float16 embeddings and live ones (about 1e-3 was assumed; in this run, rounding the
query vectors alone to float16 moved the 20 scores by at most 1.4e-5) and run-to-run noise.

The lowest answerable score is 0.691150 (d012), so the threshold must be at most 0.671150,
which rounds down to **0.67**. 0.68 (margin 0.0112) and 0.69 (margin 0.0012, which would also
catch d013) abstain on no answerable dev item but fail the margin.

## Decision

`DEFAULT_MIN_DENSE_SCORE = 0.67`. On the dev set it:

- over-refuses nothing (0 of 12 answerable items abstain);
- catches 2 of 8 unanswerable items: the two out-of-domain questions, d019 (boiling point on
  Everest, 0.618) and d020 (Mona Lisa, 0.559). These are exactly the items the 1B model got
  most visibly wrong (in round 4 it answered d020 with a made-up citation);
- misses 6 of 8: all four missing-module near-misses (d013 struct 0.688, d014 ast 0.706, d015
  queue 0.797, d016 inspect 0.718) and both undocumented features of included modules (d017
  zipfile Zstandard 0.769, d018 calendar holidays 0.748).

The near-miss pattern is the expected weakness of a similarity guard. A question about a missing
module looks like a page that is included: queue resembles threading, ast resembles re, inspect
resembles typing, struct resembles random. A question about a missing feature of an included
module matches that module's own page well (zipfile, calendar). These questions are on topic for
the corpus, so their best chunk is close in embedding space even though it does not contain the
answer. Five of the six missed items score above the lowest answerable item. The sixth, d013,
scores just below it (0.687502 against 0.691150) but above the threshold. Only the
out-of-domain questions fall clearly below the answerable range.

Judgement: the guard is narrow but useful. It costs no over-refusal on the dev set, removes the
LLM call for clearly off-topic questions, and the code is opt-out (`MIN_DENSE_SCORE=0`). It does
not solve abstention: 6 of 8 dev unanswerable items still reach the LLM, and refusing those is
left to the model or to a later mechanism.

**Pre-registered rule for the golden set.** The golden set has NOT been used for tuning, and the
threshold above was fixed on the dev set before any golden measurement. At the final recording
(made with the guard disabled, see "Replay and the final recording"), the same recordings will
be replayed twice: with the guard on (default 0.67) and with it off (0.0). BOTH results are
reported. The shipped default stays 0.67 unless the guard raises golden `over_refusal_rate` by
more than 0.05 compared with guard-off. In that case the default reverts to 0.0 and this ADR is
amended with the measured result.

## Golden result (2026-10-20)

This is the amendment the pre-registered rule above calls for. Sources:
`evals/reports/golden-guard-on-metrics.json`, `evals/reports/golden-guard-off-metrics.json`,
`evals/reports/golden-failure-analysis.md` and `evals/reports/README.md`. The two top dense
scores below were recomputed for this amendment with a read-only snippet that replays the
committed embeddings (`evals/recordings/embeddings.npz`) through `retrieve_detailed`, as
`evals/analysis/failure_taxonomy.py` does; no model and no network were used.

- **Rule outcome.** Golden `over_refusal_rate` is 0.18 with the guard on and 0.18 with it off:
  difference 0.0000, not more than 0.05. The rule does not fire and the default stays 0.67.
  `correct_refusal_rate` rose from 0.70 (guard off) to 0.80 (guard on).
- **The whole golden benefit is one item.** The 0.70 to 0.80 rise is g051 (how to run several
  asyncio coroutines concurrently), whose top dense score is 0.667983, only 0.0020 below the
  threshold. The model had answered it in the guard-off run.
- **The guard also abstained on an answerable item.** g025 (sending log messages to a file),
  top dense score 0.667614, is below 0.67. This was harmless only because the model had refused
  g025 in the guard-off run too, so no answerable outcome changed and over-refusal is 0.18
  either way. The 0.02 safety margin derived on the dev set (threshold at least 0.02 below the
  lowest answerable dev score) did NOT hold on the golden set: an answerable golden question
  scored below the threshold itself.
- **Unanswerable items.** Of the 10 golden unanswerable items, 3 were refused by the guard
  (g051, g054, g055), 5 were refused by the model (g052, g056, g057, g058, g060) and 2 were
  answered (g053, g059). g054 and g055 had also been refused by the model in the guard-off run;
  only g051 changed.
- **Conclusion.** The benefit is narrow and fragile: one item, decided by a margin of 0.0020,
  with a threshold calibrated on 20 dev items and pinned by one of them (d012). The guard stays
  on because the pre-registered rule says so and it cost no answerable outcome on this set, not
  because it is shown to help in general. A small shift in the embeddings, the chunking or the
  question wording could move g051 or g025 to the other side of 0.67.

## Limitations

- 20 dev items (12 answerable, 8 unanswerable), written by one author. One item moves a rate by
  0.083 (answerable) or 0.125 (unanswerable). The threshold is pinned by one item, d012, the
  lowest-scoring answerable question, and that is a multi item whose second page (csv) is never
  retrieved.
- The float16 margin of 0.02 is an assumption sized above the observed drift, not a measured
  bound for every query.
- The threshold did not fully transfer to the held-out golden set: one answerable golden item,
  g025, scores below 0.67 (see "Golden result (2026-10-20)"). Apart from that section, nothing
  in this ADR is a golden-set claim.
- Near-misses on included modules, and on modules similar to included ones, will score high.
  A cosine guard cannot catch them at any threshold that does not also refuse most answerable
  questions (at 0.75 it catches 6 of 8 but over-refuses 2 of 12; at 0.80 it over-refuses 11).
- The value is specific to `text-embedding-nomic-embed-text-v1.5` with the `search_query: ` and
  `search_document: ` prefixes and to the current chunking. Changing the embedding model, the
  prefixes or the chunking requires recalibrating on the dev set.
- The fake-embedding unit tests set `MIN_DENSE_SCORE=0`, because hash vectors have no semantic
  scale. Abstention is tested there with explicit thresholds.

## Rejected

- **LLM self-assessment** (ask the model whether the context answers the question, or rely on
  its refusal). That is exactly what failed: the 1B model either answered everything or refused
  almost everything (ADR 0005). It also costs a generation call per question.
- **A BM25 score threshold.** FTS5 BM25 scores are unbounded and depend on query length, term
  rarity and document length, so one global threshold is not comparable across queries. Near-miss
  questions also share many keywords with included pages (for example "zipfile", "calendar").
- **A fused RRF score threshold.** RRF scores are rank-based: `1 / (60 + rank)` summed over the
  lists. They encode only where a chunk ranks, not how similar it is, so the rank-1 fused score
  is nearly the same for a perfect match and for an irrelevant question, because something is
  always ranked first. They carry no confidence.
- **Choosing 0.68 or 0.69.** Both have zero dev over-refusal, and 0.69 would also catch d013, but
  they leave a margin of 0.011 and 0.001 to the lowest answerable score, below the 0.02 the rule
  requires. Choosing them would be fitting the threshold to a single dev item.
- **Shipping the guard disabled (0.0).** The rule allows this only when the chosen threshold
  catches nothing or is useless. It catches 2 of 8 at zero dev over-refusal, so it ships enabled.
