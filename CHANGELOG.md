# Changelog

## Unreleased

- Persist enabled Skill learning across explicit session resume, including failed attempts and retryable evidence. `/skill-learning` exposes state and `/extract_now` retries unfinished work.
- Bind candidates to their workspace, serialize native Skill writes with OS locks, and preserve evaluation receipts and previous bytes before activation. Add `/skill-promote`, `/skill-history` and guarded `/skill-rollback`.
- Archive removed conversation prefixes before folding. Valid history references survive session resume and support paged reads through the native observation tool.
- Publish [lifecycle validation](experiment/published/20261023-memory-lifecycle/README.md): 1,001 Windows regressions passed, 1 skipped; CLI, offline smoke and fresh wheel installation passed. The real extraction request was blocked by provider credits; its failed window survived process restart.
- Skill promotion installs the exact candidate bytes checked against the evaluation receipt, using a unique temporary file with failure cleanup.
- Interrupted chats and failed manual Skill extraction retain the preceding pending learning window; successful extraction cannot discard a newer window.
- Research folding loops can continue after an auxiliary summary reservation is denied when the complete original solve request remains admissible. Unknown usage and exhausted budgets still prevent continuation.
- DeepSeek side queries on CommandCode use the native `thinking` toggle. Provider support for the toggle must be checked from returned usage.
- Skill extraction resolves original receipt IDs through unique mappings in the trusted execution window. Unknown or ambiguous references remain rejected, and repair claims still require a matching failure-to-pass transition.

## 0.1.0 — 2026-10-17

First versioned runtime package from the public research snapshot.

- Native OpenAI and Anthropic loops reject tool batches from incomplete responses. Truncation recovery is bounded to two attempts per Agent instance and consumes existing call limits. Unknown finish reasons remain incomplete. Benchmark loops share this policy.
- Install the `sans` command from the checkout. The wheel contains runtime modules and source notices; frozen evidence and benchmark workspaces stay in the repository.
- Configure solve-call, tool-call and output limits and request timeouts from the CLI. Inspect the version with `sans --version`.
- Inspect staged Skill candidates with `/skill-candidates`. Embedders can supply a task evaluator through `Agent(skill_task_evaluator=...)`; promotion retains its existing task, oracle and snapshot checks.
- Windows/Linux CI checks the built wheel in a fresh environment outside the source import path and retains regression reports, installation receipts and wheels.

Historical benchmark results remain recorded against their original source. This release adds runtime and installation verification; it does not establish new Skill or context-folding task scores. [Evaluation](docs/evaluation.md) · [Release procedure](docs/releases.md)
