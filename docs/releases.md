# Maintaining and releasing

The package version is defined in [agents/__init__.py](../agents/__init__.py). Python 3.12 on Windows and Linux is the tested environment. The runtime wheel contains `agents/`; benchmark protocols, datasets and research workspaces are managed separately in the source repository.

For a release:

1. Reproduce a runtime failure before changing its behavior. Keep regression sources under `tests/` and diagnostics under `experiment/`.
2. Update [CHANGELOG.md](../CHANGELOG.md), affected user documentation and the version when preparing a new release. Keep direct dependencies in `requirements.txt` and `pyproject.toml` aligned; preserve the tested lock file unless deliberately upgrading dependencies.
3. Run [TESTING.md](../TESTING.md), including `python -B scripts/check_install.py`. This builds a wheel, checks its contents, installs it with locked dependencies in a fresh virtual environment, and invokes `sans --version` and `sans --help` from an isolated working directory. All output remains under `experiment/cache/install-*/`.
4. Merge through a PR after Windows/Linux CI passes. Each workflow retains the regression XML, installation receipt and wheel for 14 days. The receipt includes the wheel SHA-256. Store evidence that must outlive CI retention in a reviewed publication packet.
5. If creating a GitHub release, tag the tested commit and attach the reviewed artifact with its matching checksum. Preserve [source and rights notices](../SNAPSHOT_NOTICE.md); building a wheel does not establish redistribution rights or authorize a PyPI publication.

Runtime changes and research claims have different checks. A provider recovery fix needs the native provider tests and integration checks. A claimed Skill or folding improvement needs a frozen real-task protocol and preserved failures, matched conditions and measured usage. Optional task evaluators must use explicit budgets and isolated task workspaces.

Current research follow-ups are post-fold execution stagnation, task-specific completion contracts and measured Skill transfer. They remain open in [evaluation](evaluation.md); package releases do not imply those questions are solved.

Engineering references: [mini-swe-agent packaging and entrypoints](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/pyproject.toml), [Hermes contribution and platform checks](https://github.com/NousResearch/hermes-agent/blob/848fc0ad8be766b9dc10fe42bd80b5fc2d45d95c/CONTRIBUTING.md). These informed the workflow; external runtime code was not copied for this release.
