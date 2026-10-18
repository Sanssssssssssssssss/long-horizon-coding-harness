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

Dialogue and exported evidence can use different receipt IDs. Extraction resolves an original ID only when the trusted execution window maps it to exactly one scoped receipt. Unknown and ambiguous IDs remain rejected; resolving an ID does not establish a repair without a matching verification transition.

Evaluation must distinguish a discovered Skill, a loaded body, an observed invocation and a correctly completed task. A stored high score cannot replace fresh matched replay. Automatic extraction and an available promotion interface do not establish a fully validated autonomous learning deployment.

The example configuration disables automatic evolution for initial use. Set `SANS_AUTO_SKILL_EVOLUTION=1` only when feedback-driven local updates are intended. Evolution and session state remain local; see [running](running.md).

Native extraction stages candidates. `/skill-candidates` reads persisted receipts and checks their candidate and active-file hashes, showing pending, active or changed identities without model calls. An embedding application can pass `Agent(skill_task_evaluator=async_evaluate)`; the callback receives the staged snapshot and must return the complete task-level receipt required by `promote_staged_skill`. It owns the isolated task execution and budget. Missing, incomplete or mismatched evaluation leaves the candidate staged. `/skill-eval` is response-replay analysis and does not activate staged candidates.

## Current evidence

The [active cohort](../benchmarks/cohorts/skilllearn-v1.json) covers four families: DBSCAN tuning, earthquake/plate calculation, offer generation and scheduling. Across these 16 test instances, **12/16 (75%)** passed official task checks. Test execution used **6,517,258 tokens**, excluding TRAIN and distillation. The [evaluation record](evaluation.md) documents the protocol, cohort selection, historical runs and usage.

[Per-task scores, activation and usage](../experiment/published/20261016/skill/summary.json) · [Evaluation protocol](evaluation.md) · [Regression index](../tests/INDEX.md)

Run `python -B scripts/skill_cohort.py` to reconstruct the selected result from the frozen source receipt. [Selected cohort and explicit forecasts](../experiment/published/20261017-scope/README.md)

## Comparison with published methods

The four-family comparison below covers DBSCAN tuning, earthquake/plate calculation, offer generation and scheduling. Each family receives equal weight. Published references are recomputed from [SkillLearnBench, Table 8](https://arxiv.org/html/2604.20087v1#A6.T8).

| System or Skill generation method | Four-family accuracy | Evidence |
|---|---:|---|
| **Long-Horizon Coding Harness** | **75.00%** | Recorded execution: 12/16 test instances |
| One-Shot | 49.45% | Mean of four published family scores |
| Self Feedback | 48.75% | Mean of four published family scores |
| Teacher Feedback | 36.81% | Mean of four published family scores |
| Skill Creator | 40.00% | Mean of four published family scores |
| Human-authored Skills | 81.67% | Mean of four published family scores |

This is a comparison across studies. Our run uses DeepSeek V4.1 Flash on instances 2–5 of each family (16 instances); the paper uses Claude Sonnet 4.6 to solve all 22 instances in these families, including seed instances, and averages generated-Skill results across six generation models. Budgets and instance coverage differ, so these rows provide context rather than a controlled estimate of Skill improvement.

For example, One-Shot is `(63.33 + 52.78 + 25.00 + 56.67) / 4 = 49.445%`, rounded to 49.45%. These four-family averages are derived from the paper's rounded cells; the [official leaderboard](https://cxcscmu.github.io/SkillLearnBench/leaderboard.html) reports the full 20-family benchmark.
