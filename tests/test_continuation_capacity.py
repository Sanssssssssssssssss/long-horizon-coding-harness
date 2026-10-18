"""A truncated summary retry must have useful, bounded output capacity."""
import asyncio
from types import SimpleNamespace as NS
import pytest

from agents.agent import _IncompleteSideResult
from benchmarks.continuation_folding import ContinuationFoldingLoop
from test_folding_checkpoint import agent, receipt, state


def test_summary_requests_text_capacity_without_changing_main_reasoning(tmp_path, monkeypatch):
    """Check the local request contract; real provider compliance needs live validation."""
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, thinking=True)
    a.model = 'deepseek/deepseek-v4.1-flash'
    a._api_base = 'https://api.commandcode.ai/provider/v1'
    a._openai_extra_body = {'reasoning_effort': 'low'}
    current = a._repair_state = state()
    red = receipt(current)
    before = current.snapshot()
    main_options = a._openai_request_extra_body()
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        reasoning = kwargs.get('extra_body', {}).get('thinking') != {'type': 'disabled'}
        return NS(choices=[NS(message=NS(content='' if reasoning else
                  'Next action: execute solve.py, then verify all exported artifacts.'),
                  finish_reason='length' if reasoning else 'stop')],
                  usage={'prompt_tokens': 100, 'completion_tokens': kwargs['max_tokens'] if reasoning else 25,
                         'total_tokens': 100 + (kwargs['max_tokens'] if reasoning else 25)})

    monkeypatch.setattr(a, '_openai_client', NS(chat=NS(completions=NS(create=complete))))
    result = asyncio.run(a._generate_folded_session_memory([], None, []))
    assert result is not None and red.id in result
    assert len(requests) == 1
    assert requests[0]['extra_body']['thinking'] == {'type': 'disabled'}
    assert a._openai_request_extra_body() == main_options
    assert main_options['reasoning_effort'] == 'low'
    assert current.snapshot() == before and not current.completion_decision().complete


@pytest.mark.parametrize('initial_capacity', [2048, 4096])
def test_native_length_retry_increases_capacity_and_preserves_failure(tmp_path, monkeypatch, initial_capacity):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop,
              checkpoint_summary_tokens=initial_capacity)
    a.max_output_tokens = 8192
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
    assert limits == [initial_capacity, 8192]
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
