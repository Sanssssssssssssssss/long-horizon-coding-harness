# Experiments

| Location | Contents |
|---|---|
| [published/20261016](published/20261016/README.md) | Current scores, trace metadata, source hashes and manifest |
| [fixtures/](fixtures/README.md) | Historical comparison implementations and recorded folding inputs for offline regressions |
| `research/` | Ignored local research history; regression entrypoints live in `tests/` |
| `runs/`, `runtime/` | Ignored local execution records, sessions and complete payloads |
| `cache/`, `env/` | Ignored downloads, disposable workspaces and environments |
| `archive/`, `maintenance/` | Ignored historical material, teaching notes and maintenance receipts |

The maintainer retains the original frozen experiments locally. Repository cleanup changes tracking, not historical artifact bytes. New published packets must use a fresh directory; manifests bind the exact exported bytes. Full payloads and unqualified benchmark downloads are excluded from publication.

`scripts/export_evidence.py` rebuilds the recorded packet only where the original local runs are available. It performs no API calls. The public packet contains audit projections, not a complete copy of every historical task environment.
