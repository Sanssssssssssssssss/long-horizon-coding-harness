# Changelog

## Unreleased

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
