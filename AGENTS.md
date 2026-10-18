# Repository layout

- Keep runtime code in `agents/`, benchmark implementation in `benchmarks/`, regression test source in `tests/`, reusable developer commands in `scripts/`, and maintained project documentation in `docs/`.
- Put all experiment outputs, logs, traces, patches, downloaded experiment data, generated documents, test caches, temporary workspaces and maintenance receipts under `experiment/`. Do not create new `.pytest-*`, `.test-tmp`, `benchmark-results`, `tmp`, or output folders at the repository root, or new sibling audit directories outside this checkout.
- Use `experiment/runs/` for new runs, `experiment/cache/` for disposable files, `experiment/research/` for curated research notes, and `experiment/archive/` for preserved historical material. Frozen artifacts are immutable; keep their historical embedded paths unchanged and consult the relocation receipt.
- Run tests with `.venv/Scripts/python.exe -B -m pytest` on Windows. `pytest.ini` provides the artifact paths; `-B` avoids bytecode beside source files. A separate `--basetemp` must also stay under `experiment/cache/`.
- Keep credentials local and ignored. Do not stage generated output wholesale. Retain the existing `.venv` and standard root configuration needed by Python, Git, Docker and VS Code.
- Publish compact, verified evidence under `experiment/published/`. Keep teaching notes, CV material, full model payloads and local research workspaces out of the tracked publication. Preserve their local copies and frozen artifact bytes.
