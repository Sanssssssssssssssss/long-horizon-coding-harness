# Skills as procedural memory

A Skill stores reusable operating instructions with a versioned identity. The mechanism separates selection, loading, application, learning and evaluation so each stage can be inspected.

| Stage | Mechanism | Source |
|---|---|---|
| Extract | Use user feedback or typed execution receipts; require execution evidence IDs for execution-derived claims; preserve the current task and recent feedback in the extraction window | [online_skill_evolution.py](../agents/online_skill_evolution.py) |
| Maintain | Merge local corrections into the existing procedure, preserve version history, and keep provenance and evolution reasons outside the instruction body; equivalent updates do not create a new version | [skill_evolution.py](../agents/skill_evolution.py) |
| Select and load | Discover metadata, choose a relevant Skill, verify its source identity and SHA-256, then load a complete body within the 12 KB limit; a rejected load preserves the previous active Skill | [skills.py](../agents/skills.py) |
| Apply | Inject the loaded version into model requests and record actual exposure and usage. Explicit model Skill calls use the checked loader; the CLI slash-command path has a separate chat flow | [agent.py](../agents/agent.py) |
| Compare | Freeze development/promotion splits, generate fresh baseline and candidate responses on matched samples, exclude historical answers from prompts, and retain rejection/promotion decisions | [online_skill_eval.py](../agents/online_skill_eval.py) |

The executable-memory research adapter can retain validated helper attachments and replay their source dependencies. Reusing a parser function is insufficient when its imported globals change; dependencies belong to the frozen artifact identity.

Evaluation must distinguish a discovered Skill, a loaded body, an observed invocation and a correctly completed task. A stored high score cannot replace fresh matched replay. Automatic extraction and an available promotion interface do not establish a fully validated autonomous learning deployment.

The example configuration disables automatic evolution for initial use. Set `SANS_AUTO_SKILL_EVOLUTION=1` only when feedback-driven local updates are intended. Evolution and session state remain local; see [running](running.md).

## Current evidence

The [active cohort](../benchmarks/cohorts/skilllearn-v1.json) retains four families: DBSCAN tuning, earthquake/plate calculation, offer generation and scheduling. Existing records for these families give **12/16** official passes, **5/16** natural finishes and **8/16** loaded Skills. These test runs used **6,517,258 tokens**; TRAIN and distillation are excluded from that subtotal.

GDP was excluded after the runs at the maintainer's request on 2026-10-17. The original five-family record remains **12/20**, with **9,996,940 tokens / 767 requests** including TRAIN, distillation and failures. The scope change is not a new experiment or a measured improvement. No fresh no-Skill control has been run.

[Per-task scores, activation and usage](../experiment/published/20261016/skill/summary.json) · [Evaluation boundaries](evaluation.md) · [Regression index](../tests/INDEX.md)

Run `python -B scripts/skill_cohort.py` to reconstruct the selected result from the frozen source receipt. [Selected cohort and explicit forecasts](../experiment/published/20261017-scope/README.md)
