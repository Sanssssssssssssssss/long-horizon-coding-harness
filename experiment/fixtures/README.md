# Regression fixtures

`context/` contains historical comparison implementations used by offline counterexamples. They are test fixtures; the maintained runtime is in `agents/` and `benchmarks/`.

`folding/` contains three recorded compaction inputs and receipts. These JSON files retain their original bytes. Paths embedded in historical data describe the original execution.

The cleanup copied these fixtures from local research directories. Original research files remain local and unchanged. Fixture relocation records are in `manifest.json`; executable fixture imports may be adjusted to the new layout, with both original and relocated source hashes recorded.

All regression entrypoints are in `tests/`. No new outputs belong in this directory.
