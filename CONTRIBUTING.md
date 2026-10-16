# Contributing

Runtime code belongs in `agents/`, benchmark adapters in `benchmarks/`, regression suites in `tests/`, developer commands in `scripts/`, and maintained documentation in `docs/`. Experimental outputs stay under `experiment/`.

1. Describe the observable failure and the behavior the change should establish.
2. Prefer a small change in the existing module. Add a regression when it protects a meaningful contract or reproduces a real failure.
3. Run the relevant checks in [TESTING.md](TESTING.md), update [tests/INDEX.md](tests/INDEX.md), and keep generated files out of source directories.
4. Freeze model, task, source, environment, grading and budget before a paid experiment. Retain failures and unknown usage.
5. Publish a compact evidence packet with a manifest. Keep full model payloads, credentials, downloaded workspaces, personal notes and teaching materials local.

Use a pull request for `main` and complete the Windows/Linux checks. Review the [source provenance](SNAPSHOT_NOTICE.md) and [security boundaries](SECURITY.md) before contributing or publishing artifacts.

Shared provider and tool guarantees belong in `agents/`; benchmark policies should reuse them. Verify the installed CLI for entrypoint or dependency changes. Keep user-facing behavior changes in [CHANGELOG.md](CHANGELOG.md) and follow the [release procedure](docs/releases.md).
