# Tests and validation

Use the project interpreter and run the offline CI suite:

```text
python -B scripts/check_repository.py
python -B -m pytest tests experiment/research/20261007-two-directions experiment/research/20261007-small-capability/test_user_updates.py experiment/research/20261008-auto-research experiment/research/20261009-skill-folding/test_goal_anchor.py -q -W error::pytest.PytestUnhandledThreadExceptionWarning --basetemp=experiment/cache/t
python -B -m agents.main --help
python -B -m benchmarks.smoke --runs-root experiment/runs --run-id offline-smoke
```

CI runs on Windows and Linux. Test reports and temporary repositories belong under `experiment/`. The [suite index](tests/INDEX.md) groups regression sources by behavior; selected historical fixtures keep their original paths and bytes.

`check_repository.py` checks the tracked-file boundary, Python syntax, documentation links, test-index coverage, published manifests, per-case score totals and trace-derived usage. It makes no model calls.

Historical task scores are documented in [evaluation](docs/evaluation.md). A local or CI regression pass does not constitute a new benchmark result. Paid runs require a frozen protocol, qualified environment, fresh output directory and an explicit budget. Preserve failed requests and missing usage receipts.

For new failures: retain the triggering trace and source identity, reproduce the defect, repair the existing module, run relevant regressions, and rerun the affected real case when needed. Keep before/after records and distinguish a mechanism replay from full-task scoring.
