# Evaluation and evidence

All published results below come from recorded runs. Scores, natural completion, Skill exposure and model usage have separate denominators. CI verifies code and evidence integrity; paid task execution is explicitly launched by the maintainer.

## Long-task cohort

SCB `circuit_eval` is one repository developed over eight consecutive stages. Each stage retains the previous code and conversation. The public task and visible tests are available to the solver; full official checks run independently after each stage.

| Version | Final checks | Stages fully passing | Late new checks | Natural finish | Known tokens |
|---|---:|---:|---:|---:|---:|
| Ordinary-summary baseline | 500/566 | 3/8 | 76/136 | 4/8 | 3,695,940 |
| Historical best candidate | 561/566 | 4/8 | 131/136 | 6/8 | 11,757,046 |
| Previous candidate | 510/566 | 2/8 | 96/136 | 6/8 | 12,110,585 |
| Latest candidate | 527/566 | 2/8 | 98/136 | 5/8 | 12,889,060 |

The three candidate runs share a fixed `deepseek/deepseek-v4.1-flash` model, 48 responses per stage, a 15M-token allocation and a 600-second request timeout. The ordinary-summary baseline used 24 responses, 5M tokens and 180 seconds; its score difference is not a matched-budget folding effect. The task has been used during development. There is no held-out generalization result or isolated component ablation here.

The latest run contains 240 solve responses and 36 installed folds. Its 39 remaining failures are one parse-error location check, one randomized equivalence check and 37 optimization-command checks. The latest repair round additionally includes four summary probes and one capacity replay: **13,076,209 known tokens / 304 requests**, with one request of unknown usage. Its 373,316-token reservation is not measured consumption.

## Skill cohort

The current [cohort manifest](../benchmarks/cohorts/skilllearn-v1.json) retains four families and 16 recorded test instances. They passed **12/16 (75%)**, finished naturally in **5/16**, and loaded a Skill in **8/16**. Test-only usage is **6,517,258 tokens**. GDP was removed from the active scope after the recorded runs; the original evidence below remains available.

SkillLearnBench: **20 test instances, five families**, four test instances per family. Published TRAIN records cover four distinct instances; the GDP instance appears twice because the second record is an offline regrade. The two recorded revisions complete one combined subset; shared usage is counted once.

| Measure | Result |
|---|---:|
| Official task checks fully passing | 12/20 |
| Natural completion | 5/20 |
| Skill body actually loaded | 8/20 |
| TRAIN, distillation and evaluation usage | 9,996,940 tokens / 767 requests |

No fresh no-Skill control was run for this subset. GDP grading corrections were offline regrades, with the original run identity retained. Schedule tasks used a positive scoring-contract fixture rather than an official reference solver. These limitations travel with the score.

[2026-10-17 scope and forecast](../experiment/published/20261017-scope/README.md) records the selected result, the full upstream inventory and family-level planning estimates. Forecasts are not included in measured benchmark totals.

## Verification layers

| Layer | Question | Evidence |
|---|---|---|
| Runtime regressions | Are tool order, state, loading and budget contracts preserved? | [Suite index](../tests/INDEX.md), commit-specific CI |
| Mechanism replay | Does a recorded failure recover under the intended intervention? | [Summary capacity replay](../experiment/published/20261016/context/capacity-replay/receipt.json) |
| Task execution | Does the generated artifact pass the official grader? | Per-task and per-stage receipts |
| Long-task behavior | What is completed late, lost after earlier success, or left unfinished? | Per-stage curves, natural finish and tool events |
| Accounting | Are failed requests and unknown usage retained? | Trace-derived ledgers and explicit reservations |

[Published packet](../experiment/published/20261016/README.md) includes scores, event metadata, original artifact hashes and a byte-level manifest. Full requests, outputs and frozen workspaces remain in local experiment storage. The export is an audit projection; it does not contain all inputs needed to rerun every historical experiment.
