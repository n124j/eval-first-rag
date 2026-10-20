# Dev (tuning) set

## Purpose

`dev.jsonl` is the set used for TUNING: prompt wording, context layout, the number of
retrieved chunks `k`, and the abstention (retrieval-confidence) threshold. Iterate on it as
often as needed.

The golden set (`evals/golden/`) is the held-out test. It must not be used for tuning: do not
look at per-item golden results while choosing a prompt, `k` or a threshold. Only the final,
already-tuned configuration is measured on it.

## Composition

20 items: 10 `single` (one expected page), 2 `multi` (exactly two pages, neither sufficient
alone) and 8 `unanswerable` (4 near-misses on stdlib modules that are not in the corpus, 2
features that an included module's page does not document, 2 out-of-domain questions).

It is deliberately small and written by a single author, so scores on it are noisy and may
reflect that author's phrasing. Treat dev numbers as a direction for tuning, not as a result.

## Schema and validation

Same schema and conventions as the golden set (see `evals/golden/README.md`): ids `d001` ..
`d020`, `absent_terms.json` maps each unanswerable id to 2-6 lowercase terms that occur in no
`corpus/*.txt`. `tests/acceptance/test_dev_set.py` checks the format, the counts, the absent
terms, and that every dev question is disjoint from every golden question (no normalised
equality, token-set Jaccard similarity below 0.5, no shared unanswerable absent terms).

## Change rule

Changing this set after tuning has started is allowed, but the change and its reason must be
noted in the ADR that logs the tuning, since earlier dev scores are then not comparable.
