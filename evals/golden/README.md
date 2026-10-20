# Golden evaluation set

## Purpose

`golden.jsonl` is the fixed set of questions the RAG system is measured against: 40
`single` questions (answerable from one page), 10 `multi` questions (need two pages) and 10
`unanswerable` questions (the system should decline). Every metric in the baseline is
computed over this set. `review-notes.md` is the human review pack with a supporting excerpt
or an unanswerability reason for every item.

## Schema

UTF-8, `\n` line endings, one JSON object per line, keys in this order:

```json
{"id": "g001", "question": "...", "expected_sources": ["bisect"], "answerable": true, "kind": "single"}
```

- `id`: `g001`, `g002`, ... sequential and unique.
- `expected_sources`: doc ids from `corpus/manifest.json` (module names such as `os.path`).
  `single` has exactly one, `multi` exactly two different ones, `unanswerable` none.
- `answerable`: `false` only for `kind: "unanswerable"`.

`absent_terms.json` maps each unanswerable id to 2-6 lowercase terms an answer would need;
none of them may occur in any `corpus/*.txt`.

## Validation

- `evalrag.evals.dataset.load_golden(path, known_sources=...)` enforces the schema, kind and
  source consistency, unknown sources and duplicate ids or questions.
- `tests/acceptance/test_golden_set.py` adds the set-level rules: counts per kind, sequential
  ids, coverage of at least 30 docs, at most 3 `single` items per doc, absent terms really
  absent from the corpus, no meta words in questions, a review-notes entry per item, and the
  file format (UTF-8, LF, trailing newline, canonical key order).

## Change rule

Once the baseline is frozen, any change to this set (adding, removing or rewording an item,
or changing its expected sources) invalidates the baseline. Such a change requires
re-recording, a new baseline, and a recorded reason for the change in the commit message and
in the baseline's `reason` field.
