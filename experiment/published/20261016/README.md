# Research evidence · 2026-10-16

This packet publishes scores and event metadata from completed experiments. Model/tool payloads and task workspaces remain in local storage. [Manifest](manifest.json) covers the data files; this navigation page is outside the frozen data manifest.

| Evidence | Entry |
|---|---|
| Four continuous 8-stage SCB runs | [Comparison](context/comparison.json) |
| Ordinary-summary baseline | [Receipt](context/baseline/receipt.json), [events](context/baseline/events.jsonl) |
| Historical best candidate | [Receipt](context/best/receipt.json), [events](context/best/events.jsonl) |
| Previous candidate | [Receipt](context/previous/receipt.json), [events](context/previous/events.jsonl) |
| Latest candidate | [Receipt](context/current/receipt.json), [events](context/current/events.jsonl) |
| Truncated-summary recovery | [Receipt](context/capacity-replay/receipt.json), [events](context/capacity-replay/events.jsonl) |
| Late-stage failure | [Recorded case](context/diagnostics/stage8-stall-case.json) |
| Latest repair-round accounting | [All requests and unknown usage](context/diagnostics/ROUND-USAGE.json) |
| SkillLearnBench subset | [20 test instances and TRAIN records](skill/summary.json) |
| Skill protocol revisions | [Original](skill/20261016-skilllearn-20-protocol.json), [continuation](skill/20261016-skilllearn-20-v2-protocol.json) |
| Original artifact identities | [Provenance and export boundary](provenance.json) |

SCB checks are assertions within one task, not independent benchmark tasks. The latest candidate scores 527/566; the best recorded candidate scores 561/566. The ordinary baseline has a smaller budget. SkillLearnBench scores 12/20 with five task families and no fresh no-Skill control. [Conditions and limitations](../../../docs/evaluation.md)

Validate the packet from the repository root with `python -B scripts/check_repository.py`. Exported event metadata preserves the fields needed to reconstruct measured usage and unknown-request reservations. Failed and incomplete outcomes remain included.

The five TRAIN rows represent four distinct instances. The GDP row is retained in both protocol revisions; the second is an offline regrade of the original execution. The cumulative ledger counts shared API usage once.
