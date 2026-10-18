# Long-Horizon Coding Harness

**Context folding, Skill evolution and trace-driven evaluation for long-running coding tasks.**

[简体中文](README.zh-CN.md) · [Architecture](docs/architecture.md) · [Skills](docs/skills.md) · [Context folding](docs/context-folding.md) · [Evaluation](docs/evaluation.md)

[![Checks](https://github.com/Sanssssssssssssssss/long-horizon-coding-harness/actions/workflows/checks.yml/badge.svg)](https://github.com/Sanssssssssssssssss/long-horizon-coding-harness/actions/workflows/checks.yml)

The harness preserves task constraints and execution evidence across long coding sessions, then turns task feedback into reusable, versioned Skills. Its research focuses on improving task execution with a fixed model through procedural memory and inference-time context management.

## Core mechanisms

| Mechanism | Implementation |
|---|---|
| **RSI: experience distillation and Skill evolution** | Match failure and repair observations using execution IDs and test fingerprints. Distill reusable procedures with applicability conditions and verification rules; maintain versioned Skills through incremental merges, frozen replay and held-out promotion checks. Load relevant Skill bodies on demand with source-hash validation. |
| **Long-horizon context folding** | Reproduce and adapt the inference-time mechanism from [Context Folding, ICML 2026](https://proceedings.mlr.press/v306/sun26x.html). Combine branch isolation, execution checkpoints and version-bound observation recovery; preserve the next action and task constraints within the full model-request budget. |
| **Trace and layered evaluation** | Connect asynchronous spans with ContextVar; retain payloads and evidence through SHA-256 content addressing and atomic writes. Turn trace findings into [seven regression groups](tests/INDEX.md), API replays and independent task grading to guide Skill and folding updates. |

Technology: **Python 3.12 / asyncio / native Agent runtime / OpenAI and Anthropic client SDKs / typed execution records / JSONL / SHA-256 / pytest / Git / Docker adapters**. Rich powers the terminal interface.

## Recorded task results

| Cohort | Result | Scope |
|---|---|---|
| SCB `circuit_eval`, continuous 8-stage development | **527/566 (93.1%)** checks passed | Code and conversation carried across eight stages in one repository; independent official checks after each stage. |
| Four SkillLearnBench families | **12/16 (75%)** tasks passed official checks | DBSCAN tuning, earthquake/plate calculation, offer generation and scheduling; 16 test instances. |

The ordinary-summary SCB reference scored **500/566 (88.3%)**, a **4.8 percentage-point** difference. This is a comparison across configurations: candidate/reference token allocations were **15M/5M**, with different stage-response limits and request timeouts. [Evaluation protocol and per-stage records](docs/evaluation.md)

Evaluation covers code repair with SWE-bench Verified/Pro, terminal tasks with Terminal-Bench, continuous development with SCB, and Skill transfer with SkillLearnBench. Trace findings connect task outcomes to reproducible regression cases and candidate evaluation.

[Published trace metadata and receipts](experiment/published/20261016/README.md) · [Skill cohort](benchmarks/cohorts/skilllearn-v1.json) · [Published method comparisons](docs/skills.md#comparison-with-published-methods)

## Run

```powershell
git clone https://github.com/Sanssssssssssssssss/long-horizon-coding-harness.git
cd long-horizon-coding-harness
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
.venv/Scripts/python.exe -m pip install --no-deps -e .
Copy-Item .env.example .env
# Configure one provider in .env, then:
.venv/Scripts/sans.exe --thinking
```

On Linux/macOS, use `.venv/bin/python`, `.venv/bin/sans` and `cp`. `/plan`, `/skills`, `/skill-candidates`, `/compact`, `/trace` and `/exit` are available. Use `sans --version` to identify an installation. [Configuration and operation](docs/running.md)

The CLI runs the native Agent. Benchmark callers select folding policies and task evaluators; [Skill configuration](docs/skills.md) describes candidate staging and promotion. [Security](SECURITY.md) documents tool permissions and process isolation.

## Repository

```text
agents/       Runtime, tools, execution state and Skill lifecycle
benchmarks/   Isolated task runners and experimental folding policies
tests/        Offline regression suites
scripts/      Repository checks and evidence export
docs/         Maintainer and user documentation
experiment/   Published evidence, regression fixtures and ignored local runs
```

[Testing](TESTING.md) · [Changelog](CHANGELOG.md) · [Releases](docs/releases.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Source provenance](SNAPSHOT_NOTICE.md) · [Attribution](THIRD_PARTY_NOTICES.md)

Source provenance, publication scope and licensing status are documented in the source and attribution notices above.
