# Context folding

The continuation problem has two parts: retain enough implementation context to make the next edit, and preserve the authority and scope of execution evidence. These are represented separately.

The research basis is [Scaling Long-Horizon Agent via Context Folding (ICML 2026)](https://proceedings.mlr.press/v306/sun26x.html). This repository reproduces and adapts its inference-time branch/return mechanism: a subtask runs in a separate conversation and returns a compact outcome while workspace changes persist. The paper also introduces FoldGRPO training; our recorded experiments use fixed models and focus on runtime context management.

Our extensions bind recoverable observations to source versions, retain authoritative execution checkpoints, and admit the complete next request against the available budget. [ContextFoldingAgent](../benchmarks/context_folding.py) implements branch orchestration; the continuation policies below build on the shared folding loop.

| Layer | Behavior |
|---|---|
| Native conversation boundary | Preserve the canonical request, later user updates and whole assistant/tool exchanges. Budget both the summary and retained tail. Reject incomplete or inadmissible replacements while retaining raw history. |
| Execution checkpoint | Select required receipts deterministically using execution and repository identity. Summarized hypotheses and implementation notes remain unverified proposals. |
| Recoverable observations | Store large outputs by reference. Track observed source pages against execution identity and content SHA; invalidate views when content changes and admit bodies against the complete request budget. |
| Experimental continuation | Carry next actions, edit anchors, unresolved entrypoints and accurate retained-tail metadata. Avoid a new paid summary for an identical validated input. Retry truncated summaries with bounded extra output capacity. |

Native entrypoints are `Agent._compact_conversation`, `_compaction_history_parts` and `_generate_folded_session_memory` in [agent.py](../agents/agent.py); serialization is in [session_memory.py](../agents/session_memory.py). Research policies are [EvidenceFoldingLoop](../benchmarks/folding_checkpoint.py) and [ContinuationFoldingLoop](../benchmarks/continuation_folding.py).

Provider response completion is now a shared native runtime policy: incomplete tool batches are rejected, truncated responses receive at most two recovery attempts per Agent instance, and missing finish markers cannot certify completion. Benchmark loops reuse the same protection while retaining their own decision and token admission. Summary-capacity retry has its own output budget and cache identity.

In the shared research folding loop, an auxiliary summary can be denied before dispatch because its reservation exceeds the remaining token budget. If the original complete solve request still fits both input and token limits, the loop preserves raw history and restores the termination state from before that summary attempt. Unknown usage, a dispatched request or an inadmissible solve request prevents this recovery. The check does not charge or refund usage; continuation still passes the normal request admission checks.

DeepSeek side queries use the native thinking toggle on CommandCode. Returned reasoning usage establishes whether the provider honored it; requesting a mode does not establish that the provider used it.

The continuation policy also observes consecutive model responses without repository changes. It can advise the model to implement a small working path. This is a prompt-level intervention; it does not force an edit or prove that reading is unnecessary.

## Recorded results

The best recorded SCB `circuit_eval` run passed **561/566 (99.1%)** official checks across one continuous eight-stage development task. [Evaluation](evaluation.md) records the stage scores, budgets and run history.

In a separate summary-capacity replay, a previously truncated input completed with **5888 output tokens** under an **8192-token cap**, `finish_reason=stop`, and one paid request. The repair makes the continuation override apply the retry capacity, respects the explicit model cap and includes the effective cap in the cache key. Thirty-five related offline checks passed.

[Run comparison](../experiment/published/20261016/context/comparison.json) · [Execution diagnostics](../experiment/published/20261016/context/diagnostics/stage8-stall-case.json) · [Capacity replay](../experiment/published/20261016/context/capacity-replay/receipt.json)

Long-task evaluation tracks executable progress after each fold, coverage of required entrypoints and time available for integration. The retained run records support analysis of each checkpoint.
