# ADR 0001: Corpus

Status: accepted

## Context

The project needs a public, permissively licensed, technical corpus that is
unrelated to any employer.

## Decision

Use a pinned subset of 41 Python 3.12 standard-library module pages from the
official text documentation archive.

- URL: https://docs.python.org/3.12/archives/python-3.12-docs-text.zip
- Filename: python-3.12-docs-text.zip
- Size: 4066492 bytes
- sha256: e99a9d0200f871d4f1d0ca8137149e11737c3c342ed1f8beacef2cfdf89cbecf
- Downloaded: 2026-10-19 (one time, via scripts/fetch_corpus.py)

## Consequences

- manifest.json records a sha256 per file; load_corpus verifies each one.
- .gitattributes marks `corpus/**` as `-text` so line endings never alter hashes.
- Chunking happens later, at index time, not in the stored corpus.

## Rejected

- Wikipedia: noisy, and licensing varies per article.
- Kubernetes docs: larger and less familiar to reviewers.
- The full Python docs: too large for a recorded-response CI.
