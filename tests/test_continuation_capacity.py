"""A truncated summary retry must have useful, bounded output capacity."""
import asyncio
from types import SimpleNamespace as NS

from agents.agent import _IncompleteSideResult
from benchmarks.continuation_folding import ContinuationFoldingLoop
from test_folding_checkpoint import agent, receipt, state


def test_native_length_retry_increases_capacity_and_preserves_failure(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop,
              checkpoint_summary_tokens=4096)
    current = a._repair_state = state()
    red = receipt(current)
    before = current.snapshot()
    limits = []
    async def summarize(*args, **kwargs):
        limits.append(kwargs['max_tokens'])
        if kwargs['max_tokens'] < 8192:
            raise _IncompleteSideResult('length')
        return 'Next action: connect the incomplete output path and test its required behavior.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    result = asyncio.run(a._generate_folded_session_memory([], None, []))
    assert result is not None and red.id in result
    assert limits == [4096, 8192]
    assert current.snapshot() == before and not current.completion_decision().complete


def test_retry_cannot_exceed_explicit_model_output_cap(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop,
              checkpoint_summary_tokens=4096)
    a.max_output_tokens = 3000
    a._repair_state = state()
    before = list(a._openai_messages)
    limits = []
    async def truncated(*args, **kwargs):
        limits.append(kwargs['max_tokens'])
        return NS(choices=[NS(message=NS(content='partial'), finish_reason='length')],
                  usage={'prompt_tokens':10,'completion_tokens':3000,'total_tokens':3010})
    monkeypatch.setattr(a, '_openai_client', NS(chat=NS(completions=NS(create=truncated))))
    assert asyncio.run(a._generate_folded_session_memory([], None, [])) is None
    assert limits == [3000, 3000]
    assert a._openai_messages == before


def test_cached_summary_does_not_mask_changed_retry_capacity(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop,
              checkpoint_summary_tokens=4096)
    a._repair_state = state()
    limits = []
    async def summarize(*args, **kwargs):
        limits.append(kwargs['max_tokens'])
        return 'Pending: inspect the exact output contract.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    asyncio.run(a._run_compaction_query([], 'Summarize'))
    asyncio.run(a._run_compaction_query([], 'Summarize', retry_with_full_output=True))
    asyncio.run(a._run_compaction_query([], 'Summarize', retry_with_full_output=True))
    assert limits == [4096, 8192]
