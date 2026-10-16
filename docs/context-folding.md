# Context folding

The continuation problem has two parts: retain enough implementation context to make the next edit, and preserve the authority and scope of execution evidence. These are represented separately.

| Layer | Behavior |
|---|---|
| Native conversation boundary | Preserve the canonical request, later user updates and whole assistant/tool exchanges. Budget both the summary and retained tail. Reject incomplete or inadmissible replacements while retaining raw history. |
| Execution checkpoint | Select required receipts deterministically using execution and repository identity. Summarized hypotheses and implementation notes remain unverified proposals. |
| Recoverable observations | Store large outputs by reference. Track observed source pages against execution identity and content SHA; invalidate views when content changes and admit bodies against the complete request budget. |
| Experimental continuation | Carry next actions, edit anchors, unresolved entrypoints and accurate retained-tail metadata. Avoid a new paid summary for an identical validated input. Retry truncated summaries with bounded extra output capacity. |

Native entrypoints are `Agent._compact_conversation`, `_compaction_history_parts` and `_generate_folded_session_memory` in [agent.py](../agents/agent.py); serialization is in [session_memory.py](../agents/session_memory.py). Research policies are [EvidenceFoldingLoop](../benchmarks/folding_checkpoint.py) and [ContinuationFoldingLoop](../benchmarks/continuation_folding.py).

The continuation policy also observes consecutive model responses without repository changes. It can advise the model to implement a small working path. This is a prompt-level intervention; it does not force an edit or prove that reading is unnecessary.

## Recorded failure and repair

In the latest SCB run, the summary retry flag was accepted but ignored by the continuation override: both attempts still requested 4096 output tokens. The fix doubles only the existing retry capacity, respects the explicit model cap and includes the effective cap in the cache key. The original failing input subsequently returned 5888 output tokens with an 8192 cap, `finish_reason=stop`, and one paid request. Thirty-five related offline checks passed.

That fix was applied **after** the complete 527/566 run. The full sequence's last stage still had **0/37** new checks: no implementation change landed in 48 responses, despite summaries retaining a concrete edit plan. The final attempted tool call was truncated after reasoning consumed 32667 of 32768 output tokens. This remains a failed long-task outcome.

[Run comparison](../experiment/published/20261016/context/comparison.json) · [Failure record](../experiment/published/20261016/context/diagnostics/stage8-stall-case.json) · [Capacity replay](../experiment/published/20261016/context/capacity-replay/receipt.json)

Future experiments need to measure executable progress after a fold, coverage of the required entrypoints and time left for integration. More detailed summaries alone have not established an improvement in late completion.
