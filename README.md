# Long-Horizon Coding Harness

**Procedural memory and context folding for coding agents that work across many turns.**

[简体中文](README.zh-CN.md) · [Architecture](docs/architecture.md) · [Skills](docs/skills.md) · [Context folding](docs/context-folding.md) · [Evaluation](docs/evaluation.md)

[![Checks](https://github.com/Sanssssssssssssssss/long-horizon-coding-harness/actions/workflows/checks.yml/badge.svg)](https://github.com/Sanssssssssssssssss/long-horizon-coding-harness/actions/workflows/checks.yml)

Long tasks accumulate stale assumptions, repeated exploration and verification results that lose their original scope. This Python harness studies how to retain useful working context and reuse executable experience with a fixed model.

## Two research directions

| Mechanism | Implementation |
|---|---|
| **Skills as procedural memory** | Extract procedures from feedback and execution receipts; keep provenance outside the instruction body; merge local corrections into versioned snapshots. Load a selected Skill against its source hash and a bounded context budget. Evaluate candidates through fresh baseline and candidate replay on frozen development/test splits. |
| **Context folding** | Preserve task updates and complete tool exchanges; keep runtime evidence separate from generated summaries. Admit checkpoints against the full request budget, retain recoverable observations, and invalidate source views when execution or content changes. Experimental continuation policies preserve coverage gaps and the next edit. |

The runtime uses Python 3.12, asyncio, OpenAI/Anthropic SDKs, typed execution records, JSONL traces and SHA-256 artifact identities. Rich powers the terminal interface; pytest, Git and container adapters support evaluation.

## Recorded evidence

| Cohort | Result | Boundary |
|---|---|---|
| SCB `circuit_eval`, continuous 8-stage development | Historical best **561/566** checks; latest candidate **527/566**, 240 solve responses | One development task; earlier candidate 510/566. Late-stage stagnation remains unresolved. |
| SkillLearnBench subset | **12/20** tasks passed official checks; 5/20 finished naturally | Five task families; a Skill loaded in 8/20 test runs. No fresh no-Skill control. |
| Truncated-summary recovery | Same recorded input completed with an 8192-token retry cap; 5888 output tokens | One real API replay, plus 35 offline regressions; separate from the full-task score. |

[Protocols, per-stage scores, usage and retained failures](docs/evaluation.md) · [Published trace metadata and receipts](experiment/published/20261016/README.md)

## Run

```powershell
git clone https://github.com/Sanssssssssssssssss/long-horizon-coding-harness.git
cd long-horizon-coding-harness
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
# Configure one provider in .env, then:
.venv/Scripts/python.exe -B -m agents.main --thinking
```

On Linux/macOS, use `.venv/bin/python` and `cp`. The CLI is named `sans`; `/plan`, `/skills`, `/compact`, `/trace` and `/exit` are available. [Configuration and operation](docs/running.md)

## Repository

```text
agents/       Runtime, tools, execution state and Skill lifecycle
benchmarks/   Isolated task runners and experimental folding policies
tests/        Offline regression suites
scripts/      Repository checks and evidence export
docs/         Maintainer and user documentation
experiment/   Published evidence, regression fixtures and ignored local runs
```

The default CLI uses the native runtime. `ContinuationFoldingLoop` is an experimental benchmark policy; its results do not imply that it is the CLI default. Candidate promotion and executable grading require the configured research adapters. Permissions constrain tool use; they are not an OS sandbox.

[Testing](TESTING.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Source provenance](SNAPSHOT_NOTICE.md) · [Attribution](THIRD_PARTY_NOTICES.md)

The repository is publicly viewable. Upstream attribution is retained; no project-wide redistribution license is asserted.
