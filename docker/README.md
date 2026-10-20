# Running eval-first-rag in Docker

## What `docker compose up` does

1. Builds one image (`python:3.12-slim-bookworm`, non-root user) containing the package, the corpus
   and the recorded embeddings.
2. Runs the one-shot `init` service. It builds the search index into the named volume
   `evalrag-data` from the **recorded** document embeddings, so no model server is needed for this
   step.
3. Starts `api` on <http://localhost:8000> (`GET /health`, `POST /ask`) once `init` has finished
   successfully. Its healthcheck calls `/health` with Python's standard library.

```
docker compose up --build
```

## The model server is not in the default stack

Answering a question needs an LLM, and embedding the question needs an embedding model. Both come
from an OpenAI-compatible server that you run yourself:

- **LM Studio on the host (default).** Start its local server on port 1234. The containers reach it
  at `http://host.docker.internal:1234/v1` (`extra_hosts` maps that name on Linux too).
- **Another server or an API key.** Create a `.env` file containing ONLY the variables you want to
  override (`LLM_BASE_URL`, `EMBED_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, ...). Do not copy
  `.env.example` wholesale: its `localhost` URLs are for running evalrag directly on your machine,
  and inside a container `localhost` is the container itself. Use `host.docker.internal` (or a
  service name) in Docker.
- **Precedence.** A value in `.env` overrides the compose defaults. The defaults, used when a
  variable is unset, are `http://host.docker.internal:1234/v1` for both URLs, `qwen2.5-7b-instruct-1m`
  and `text-embedding-nomic-embed-text-v1.5`. The `.env` file is optional.

## CPU only

The container path is CPU-only. Docker on Windows cannot use the AMD GPU, so GPU inference needs the
model server running natively on the host (LM Studio) rather than in a container.

## The `ollama` profile

An optional Ollama container, off by default (image tag pinned in `compose.yaml`):

```
docker compose --profile ollama up --build
docker compose exec ollama ollama pull <llm-model>
docker compose exec ollama ollama pull <embedding-model>
```

Then set `LLM_BASE_URL=http://ollama:11434/v1`, `EMBED_BASE_URL=http://ollama:11434/v1` and the
matching `LLM_MODEL` / `EMBED_MODEL` in `.env`. Models live in the `ollama-models` volume. Pulling
models is a large download and runs on CPU here, so answers are slow.

## Embedding model caveat

The document vectors were recorded with LM Studio's `nomic-embed-text-v1.5` at Q4_K_M. A different
embedding model or quantisation (for example Ollama's) gives different vectors. Whether retrieval
still works with Ollama embeddings is UNVERIFIED: the `ollama` profile was never run, and query
vectors from a different build may not line up with the recorded document vectors. The refusal guard threshold
(`MIN_DENSE_SCORE`, 0.67) was calibrated on the recorded vectors, so scores and refusals may
shift.

## Security

The API has **no authentication** and is for local use only. `compose.yaml` therefore publishes
its ports on the loopback interface only (`"127.0.0.1:8000:8000"` for the API and
`"127.0.0.1:11434:11434"` for the optional Ollama server), so they are not reachable from other
machines. Do not change that binding to expose port 8000 or 11434 to a network you do not
control.

## First run

The first `docker compose up --build` downloads the base image and installs the dependencies. On
the verified machine build plus start took about 29 s, with a warm base image pull (see
"Verified"); on another machine, with a cold pull and PyPI downloads, the first run may take a few
minutes. The `init` step then builds the index; later runs reuse the `evalrag-data`
volume and the build cache. The first question after startup can be slow while the model server
loads its model. Run `docker compose down -v` to delete the volumes and start clean.

## Verified

Verified on 2026-10-20 with Docker 27.4.0 and Docker Compose v2.31.0-desktop.2 (Windows 11 host,
Docker Desktop), with no `.env` file, so the compose defaults were used as shipped.

What was run: `docker compose up -d --build` (default profile: `init` and `api`), a check of the
runtime user and the `/data` volume, `GET /health`, two `POST /ask` calls against LM Studio on the
host (`http://host.docker.internal:1234/v1`, `qwen2.5-7b-instruct-1m`), then `docker compose down -v`.

Observed:

- Build plus start took about 29 s on this machine's first build of the image, with a warm base
  image pull; a first run elsewhere may take minutes. The image `eval-first-rag:local` is 326 MB.
- `init` exited 0 and printed `ingested 41 documents, 1289 chunks, dim 768`. The four index files
  (`chunks.jsonl`, `dense.npz`, `ingest.json`, `keyword.sqlite`) were in `/data`, owned by the
  runtime user.
- `api` ran as uid 10001 (`evalrag`, not root); the container healthcheck reported `healthy` and
  `GET /health` returned `{"status":"ok"}`.
- The smoke test re-ran golden item g004 ("How does shutil.copy2 differ from shutil.copy?") live:
  HTTP 200 in about 4 s, one citation (`shutil#006`), `refused: false`. It reproduced the recorded
  outcome: the model quoted with single quotes where the chunk has double quotes, so the verbatim
  check rejected it (`citations_valid: false`). This confirms that the Docker path reproduces
  recorded behaviour (same held-out item, same model, temperature 0); it is not new evidence.
- Dev item d020 ("Who painted the Mona Lisa?", off-topic) was abstained by the
  retrieval-confidence guard (`abstained: true`, `refused: true`, answered in about 0.07 s, i.e.
  without an LLM call), as the dev calibration in ADR 0006 predicts.
- `docker compose down -v` removed the containers, the network and the `evalrag-data` volume.

Limitations: one machine, a Windows host, CPU-only container (no GPU), the model server was the
host's LM Studio (not part of the stack), and only two questions were asked, both items already
in the golden or dev set, so this shows the packaging works end to end, not answer quality. The
`ollama` profile was not run. The loopback port binding was added after this run and has only
been checked statically (`docker compose config`).
