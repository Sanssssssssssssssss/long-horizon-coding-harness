# Core reliability: real business tasks

Thirteen development runs on four SkillLearnBench instances across document generation, PDF scheduling and numerical optimization. Baseline: `945aa21`; retained runtime: `4a40bc7`. Each revision, task input, image, Skill and grading qualification was frozen before execution. Model: `deepseek/deepseek-v4.1-flash` through CommandCode. Each task received the same 500,000-token admission allowance, 36 solve decisions, 48 total model calls and a 32,768-token input ceiling. Independent grading ran after the solver stopped.

## Retained repairs

- **Skill receipt references.** A real extraction cited the original execution ID visible in the dialogue. The trusted extraction window had added run/task scope, causing a valid procedure to be rejected. Exact, unique mappings now resolve original IDs to scoped receipts. Unknown or ambiguous IDs and unsupported repair claims still fail validation. Revalidating the same recorded API output produced a staged Skill without another model call.
- **Provider parameters.** DeepSeek side queries use the native `thinking.type=disabled` field on CommandCode. A live replay still returned reasoning tokens, so this study does not claim the provider honored the requested mode. The parameter follows the [DeepSeek API contract](https://api-docs.deepseek.com/guides/thinking_mode/).

The retained runtime changes touch two existing files and add 14 net lines. Targeted verification: **110 passed**. Full validation runs through the repository's Windows/Linux CI.

## Task evidence

The Skill was extracted only from `offer-letter-generator-1`, which passed 25/25 official checks. In the predeclared matched pair on instance 4, no Skill scored **22/29**; the frozen extracted Skill scored **29/29**, with natural completion. The control's tests inspected the document body only; the Skill run also inspected headers and footers. Tokens increased from **201,795 to 346,319**. After reverting the experimental folding changes, the retained runtime again scored **29/29**, with natural completion, using **379,078 tokens**.

These are development repeats, not independent benchmark samples. Another earlier no-Skill run also scored 29/29. The pair demonstrates one observed transfer and coverage improvement; it does not establish an overall accuracy gain.

| Round | Task | Revision / Skill | Official checks | Natural completion | Tokens |
|---|---|---|---:|:---:|---:|
| r01 | schedule 4 | frozen baseline / historical Skill | 2/2 | No | 392,704 |
| r02 | offer 1 | frozen baseline / none; TRAIN | 25/25 | Yes | 377,808 |
| r03 | schedule 4 | non-thinking request candidate | 0/2 | No | 226,480 |
| r04 | offer 1 | same candidate / none | 25/25 | No | 447,055 |
| r05 | schedule 4 | native provider toggle | 2/2 | No | 296,975 |
| r06 | offer 1 | native provider toggle / none | 25/25 | Yes | 137,003 |
| r07 | offer 4 | native provider toggle / none | 29/29 | Yes | 154,094 |
| r08 | offer 4 | expanded retry / no Skill control | 22/29 | Yes | 201,795 |
| r09 | offer 4 | same runtime / extracted Skill | 29/29 | Yes | 346,319 |
| r10 | schedule 4 | expanded retry / historical Skill | 2/2 | No | 456,845 |
| r11 | DBSCAN 2 | expanded retry / historical Skill | 7/8 | No | 455,817 |
| r12 | schedule 4 | factual handoff role / historical Skill | 0/2 | No | 448,026 |
| r13 | offer 4 | retained runtime / extracted Skill | 29/29 | Yes | 379,078 |

## Folding decision

The schedule baseline spent ten summary calls on truncated output and used no completed fold. Expanding retry capacity alone did not produce natural task completion. Adding an explicit factual-handoff role made a retained failed request return a complete 4,916-token summary, and r12 used four folds, but the full task still failed after a duration-parsing error and budget exhaustion. Both folding candidates were reverted. The numerical task retains a failure in Pareto frontier completeness. These outcomes remain in the record.

The source archive and raw traces remain local under `experiment/`. r01 loaded the byte-identical checkout before any runtime edits; r02 onward imported dedicated frozen runtime directories and recorded their module locations. No test-instance feedback was included in Skill extraction, and the candidate was not automatically promoted into the active library.

**Total provider usage: 4,388,478 tokens, 315 physical requests, zero unknown-usage requests.** This includes all task outcomes, extraction and both diagnostic replays. A probe-construction error made zero API calls and was retained locally before retry.

[Summary and source identities](summary.json) · [Frozen protocols](protocols.json) · [Skill revalidation](skill-revalidation.json) · [Artifact manifest](manifest.json)
