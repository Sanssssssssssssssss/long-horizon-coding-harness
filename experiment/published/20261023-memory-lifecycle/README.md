# Recoverable learning and folded history

This update adds three runtime capabilities:

- Persist learning windows before extraction; resume failed work with the same window ID and retained execution evidence. Session generation checks prevent late callbacks from applying to a cleared task.
- Bind candidates to their workspace, retain evaluated activation snapshots, and coordinate native Skill writes with OS locks and hash checks. `/skill-promote` imports a trusted local task evaluation; `/skill-history` and `/skill-rollback` expose version history and guarded recovery.
- Archive removed conversation prefixes by content hash before folding. Resumed sessions can page through registered history using `read_observation`, while historical results remain separate from current verification.

Windows regression: **1,001 passed, 1 skipped**, including **38 new cases**. Repository checks, CLI startup, offline benchmark smoke and installation of the wheel in a fresh environment passed. The skipped adapter test requires POSIX. [Validation receipt](validation.json) · [New regression cases](regression-cases.json) · [Historical trace replay](fold-replay.json)

SkillLearnBench environment checks passed for the selected document task: TRAIN empty/reference **0/25 and 25/25**; TEST empty/reference **0/29 and 29/29**. The real extraction request was rejected by the provider for insufficient credits. A new process restored the same failed learning window and execution evidence. No candidate or new task score was produced; the matched business runs did not start. Usage: **1 physical request, 0 known tokens, 1 request without reported usage**. The **56,978-token reservation** is a conservative ledger charge, not measured consumption.

Activation requires the existing task evaluation gate; automated evaluation is supplied by the embedding host, and rollback is explicit. Native writers share the lock; external edits and simultaneous writers to one session require coordination. A crash after API success but before the completion checkpoint may repeat extraction. History archives retain original conversation text and previews; they do not reload old execution blobs or establish current test success.

[Skill lifecycle](../../../docs/skills.md) · [Context folding](../../../docs/context-folding.md) · [CLI](../../../docs/running.md)
