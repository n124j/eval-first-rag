# Workforce

This repository was built by AI coding agents working in four roles under the repository
owner's direction. The role definitions are in `.claude/agents/`; this page summarises the
rules they share and states what the role check does and does not prove.

## Roles and owned paths

The source of truth is [`workforce/policy.toml`](../workforce/policy.toml). At the time of
writing it gives:

| Role | Owns |
| --- | --- |
| manager | `docs/superpowers/**`, `docs/workforce.md`, `workforce/status/**`, `workforce/backlog/**`, `workforce/policy.toml`, `.claude/agents/**` |
| developer | `src/**`, `corpus/**`, `scripts/**`, `tests/unit/**`, `docs/adr/**`, `tools/**`, `tests/conftest.py`, `requirements*.txt`, `.gitattributes`, `README.md`, `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, `workforce/status/handoffs.jsonl` |
| tester | `tests/acceptance/**`, `evals/**`, `workforce/status/handoffs.jsonl` |
| deployer | `.github/**`, `docker/**`, `Dockerfile*`, `compose*.yaml`, `.dockerignore`, `.gitleaks*`, `workforce/status/handoffs.jsonl` |

If this table and `workforce/policy.toml` disagree, the policy file wins.

## The commit-trailer rule

Every commit message carries one `Role: <role>` line (the last one counts), and every path the
commit changes must match one of that role's globs. A commit that touches paths of two roles is
split into two commits.

## Running the check

From the repository root:

```bash
python -m tools.role_paths --base <base-sha> --head HEAD
```

To check the whole history, pass the all-zero SHA as the base:

```bash
python -m tools.role_paths --base 0000000000000000000000000000000000000000 --head HEAD
```

It prints each violation and exits non-zero if there is any.

## Known limits

- **`Role:` trailers are self-declared.** The check proves that the declared role and the paths
  a commit touches are consistent. It does not prove which agent, or which person, did the work.
- **One git identity.** All commits share the same author identity, so authorship cannot
  distinguish the roles either.
- **The manager can change the rules.** The manager role owns `workforce/policy.toml`, so it can
  edit the very policy the check enforces.
- **`workforce/status/handoffs.jsonl` is empty in this repository.** The agent files describe
  appending handoffs to it, but the orchestration record (task ledger, rulings, reviews) lives
  outside this repository, in the separate `ai-workforce` repository.
- **Uniform co-author trailer.** Every commit carries
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`, as the build plan mandated,
  although some tasks ran on other model tiers. The trailer does not record which model did a
  given task.
- **No server-side enforcement yet.** The repository has no remote, so nothing on GitHub (branch
  protection, required checks) enforces the role check or CI. Until a remote with branch
  protection exists, the check is only as strong as running it locally.
