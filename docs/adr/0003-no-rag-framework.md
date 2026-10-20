# ADR 0003: No RAG framework; a small explicit pipeline

Status: accepted (a design rationale; no framework-based implementation was built or measured
for comparison)

## Context

The project's claim is "eval-first": every behaviour change is driven by a measurement, and the
measurements must be reproducible offline in CI (ADR 0004). That puts specific demands on the
code:

- every step a question goes through (chunking, both indexes, fusion, the confidence guard, the
  prompt, the citation parser and verifier) must be visible and testable on its own;
- the exact request sent to the model must be stable, because recorded completions are keyed by
  the request (model, messages, temperature, token limit); any hidden change to a prompt
  template invalidates the recording;
- the evaluation harness must be able to score *any* system, not only this one, so it must not
  depend on the pipeline's internals.

## Decision

Write the pipeline directly, in plain Python, with no RAG or agent framework:

- `src/evalrag/` is about 2,500 lines across small modules: `chunking.py`, `ingest.py`,
  `retrieval/keyword.py` (SQLite FTS5), `retrieval/dense.py` (NumPy), `retrieval/fusion.py`
  (RRF), `retrieval/hybrid.py`, `generation/answer.py` (prompt, citation parser and verifier),
  `generation/llm.py` (an OpenAI-compatible HTTP client over `httpx`, with record and replay),
  `pipeline.py`, `api.py` (FastAPI) and `cli.py`.
- Runtime dependencies are five pinned packages: `numpy`, `httpx`, `fastapi`, `uvicorn`,
  `pydantic`.
- The harness (`src/evalrag/evals/`) takes a plain callable `question -> SutResult` and pure
  metric functions over strings and floats. `tests/acceptance/test_harness_independence.py`
  checks that it imports nothing from `evalrag.retrieval`, `evalrag.generation` or
  `evalrag.pipeline`.
- "Agentic" here means a small fixed loop, retrieve, decide whether to abstain, answer, verify
  the citations, not an open-ended agent that chooses tools.

## Consequences

- A reader can follow one question from `RagPipeline.ask` to the verified answer in a few
  files, and every step has unit tests with fakes (no model needed).
- The prompt is a literal in `generation/answer.py`, so a change to it is a visible diff that
  also breaks the recorded-request keys, which is the intended behaviour (re-record, ADR 0004).
- Record and replay are a thin wrapper around two small client interfaces, which is what makes
  the CI gate deterministic and free.
- The cost: features a framework would give for free (document loaders, many vector stores,
  streaming, tool calling, tracing) are absent, and anything added must be written and tested
  here.

## Rejected

- **LangChain.** Broad and fast-moving; its chains and prompt templates add layers between the
  code and the exact request, which works against request-keyed recordings and against reading
  the pipeline in one sitting. Its abstractions would also be a large dependency for about a
  dozen operations.
- **LlamaIndex.** Strong indexing and retrieval abstractions, but the same concerns: hidden
  default prompts and retrieval settings, and a heavy dependency tree for a 41-page corpus.
- **A vector database server** (for example Qdrant, Weaviate, Chroma in server mode, pgvector).
  1289 chunks fit in a NumPy matrix; a server would be one more service to run in CI and
  Docker with no measured benefit at this size.
- **A framework's evaluation module** (for example a RAG evaluation library with LLM-judged
  metrics). The project needed deterministic, offline, string-level metrics that CI can replay
  without a model; an LLM judge makes the gate depend on a model. An optional LLM-as-judge
  scorer exists in `src/evalrag/evals/judge.py` for reports, but it is not wired into the CLI
  and the gate never reads it.

## Limitations

- No comparison was measured. This ADR does not claim the pipeline is faster, cheaper or more
  accurate than a framework-based one; it claims it is easier to read, test and replay, which
  is a judgement.
- Some things a framework handles well (other file formats, streaming answers, many backends)
  would have to be built if needed.
