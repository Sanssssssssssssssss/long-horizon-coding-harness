# Benchmark code

| Modules | Purpose |
|---|---|
| `runner`, `live`, `smoke` | Isolated execution, trace requirements, patch replay and offline smoke |
| `context_folding`, `folding_checkpoint`, `continuation_folding` | Experimental context policies |
| `skillsbench_trial`, `skillsbench_prepare`, `skillsbench_learning` | Dialogue-parser task preparation, execution and TRAIN-only extraction |
| `skill_compiler`, `skill_transfer_eval`, `skill_transfer_fixtures` | Executable helper packaging and constructed transfer checks |
| `panel_run`, `panels/` | Frozen SWE-bench and terminal task panels |
| [cohorts/skilllearn-v1.json](cohorts/skilllearn-v1.json) | Current four-family SkillLearnBench selection; summarize with `python -B scripts/skill_cohort.py` |

Historical task launchers may require local frozen inputs. Their presence does not make old downloads available in a fresh clone. Active Python dependencies are maintained here; historical comparison implementations used only by tests are in `experiment/fixtures/`.

Outputs belong under `experiment/`. See [evaluation](../docs/evaluation.md) for task scores, protocol differences and forecast scope.
