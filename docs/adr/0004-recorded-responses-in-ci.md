# ADR 0004: Recorded embeddings and responses make the CI gate deterministic

Status: accepted (the recordings exist and replay byte for byte on the Windows machine that
made them; replay on the Linux CI runner is UNVERIFIED until the first CI run, because the
repository has no remote yet)

## Context

The project's regression gate must fail a change that makes retrieval, citation checking,
abstention or the metrics worse. Running the real system needs two models, a 7B chat model
and an embedding model, served by LM Studio on a desktop GPU (AMD Radeon RX 6600 XT 8 GB).
GitHub-hosted runners have no GPU, a paid model API would cost money per run and add network
flakiness, and a live model makes every run slightly different unless sampling, server build
and quantisation are all pinned.

## Decision

Record once, replay in CI.

- `RecordingEmbedder` and `RecordingClient` wrap the live clients and store every response:
  embeddings in `evals/recordings/embeddings.npz` (float16, keyed by kind and text) and chat
  completions in `evals/recordings/llm.jsonl` (keyed by a SHA-256 of the canonical request:
  model, messages, temperature, `max_tokens`, seed). Both are written deterministically and
  atomically.
- `ReplayEmbedder` and `ReplayClient` serve those responses and raise on any request that was
  not recorded (a replay miss), so a changed prompt or chunking cannot silently fall back to
  something else.
- `evalrag refresh-recordings` makes a new recording against live endpoints: it ingests the
  corpus and runs the golden set while recording, writes to temporary files and replaces the
  finals only if everything succeeded, and refuses to overwrite existing recordings unless
  `--overwrite` is given. The published recording was made with the confidence guard disabled
  (`--min-dense-score 0`) so that every question has a completion; the guard is applied at
  replay time (ADR 0006).
- CI (`.github/workflows/ci.yml`) runs `evalrag ingest` with `--replay-embeddings`, then
  `evalrag eval` with `--replay-llm` and `--replay-embeddings`, then `evalrag gate` against
  `evals/baseline.json`. Acceptance tests replay the same recordings with the network disabled
  and compare the metrics with the published files byte for byte.
- `evals/baseline.json` holds the guard-on values, a per-metric tolerance and direction, and a
  `reason` field that records why the baseline is what it is. Changing the prompt, parser,
  chunking, retrieval, model or threshold requires re-recording and a new baseline with a new
  reason.

## Consequences

- The gate is deterministic, needs no GPU and no model server, costs nothing, and runs in
  seconds (the golden replay is a few seconds of the test suite).
- A pass means: this code, given these exact recorded model outputs, still reproduces the
  recorded scores. It covers retrieval, prompt construction (any prompt change causes replay
  misses), citation parsing and verification, abstention and the metric code.
- **It cannot detect model drift** or measure live quality. A new model version, a different
  quantisation, another server build or a different GPU offload can change every number; only
  a new recording shows that.
- Recordings are large-ish binary and text files in Git (1349 embeddings, 60 completions).
- Replayed latencies are the recorded ones, so the latency rows in CI are not a performance
  test.

## Rejected

- **A live model in CI.** Needs a GPU runner or a paid API; slower, costs money per run, makes
  CI depend on an external service, and gives slightly different results between runs unless
  everything is pinned. It would also make a red build ambiguous (code regression or model
  noise).
- **A tiny CPU model in CI** (for example a 0.5B to 1B model). Possible on a free runner, but it
  measures a different system: the first live measurement showed a 1B model produced 0 valid
  citations of 60 with the v0 prompt (`evals/reports/README.md`), so a gate on such a model
  would guard numbers that say nothing about the shipped configuration. It would also be slow
  and still nondeterministic across runner hardware.
- **A mocked LLM** (hand-written canned answers). Good for unit tests, which use fakes, but a
  gate on invented answers checks only the plumbing; recorded real answers also exercise the
  parser and verifier on real model output, including its real failures.

## Limitations

- The gate measures a FIXED recording. It is a regression test for the code, not for the model.
- The recording was made on Windows (AMD Ryzen 9 3900X, LM Studio). CI replays it on Linux.
  Dense scores are rounded to 6 places and ties broken by id to keep ordering identical across
  BLAS builds, but cross-platform byte-for-byte replay has not been observed yet; it is
  unverified until the first CI run.
- Tolerances use floors (0.02 for retrieval and citation rates, 0.10 for refusal rates)
  because the observed spread over the recording and 3 live re-runs was 0 at temperature 0.
  Zero spread on one machine is not robustness to other hardware or server builds.
- Re-recording requires the same models served locally; the published models' availability
  and licences are outside this repository.
