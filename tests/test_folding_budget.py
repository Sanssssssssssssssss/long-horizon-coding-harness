"""Summary retries must leave one model decision, and expose the active solve cap."""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.agent import Agent, _BudgetExceeded
from benchmarks.context_folding import _FoldingLoop


@pytest.mark.parametrize('remaining', [1, 2])
def test_failed_summary_does_not_spend_the_final_solve_call(remaining):
    agent = Agent(api_base='http://unused.invalid', api_key='test', custom_system_prompt='fixed',
                  custom_tools=[], is_sub_agent=True, max_model_calls=10)
    agent._operation_counts['model_calls'] = 10 - remaining
    calls = []
    async def query(*args, **kwargs):
        agent._reserve_call('model_calls', agent.max_model_calls)
        calls.append(kwargs)
        raise ValueError('Constructed incomplete summary')
    agent._run_compaction_query = query
    result = asyncio.run(agent._generate_folded_session_memory([{'role': 'assistant', 'content': 'Earlier execution evidence.'}]))
    assert result is None and len(calls) == remaining - 1
    assert agent._operation_counts['model_calls'] == 9


def test_completed_summary_can_use_the_penultimate_call():
    agent = Agent(api_base='http://unused.invalid', api_key='test', custom_system_prompt='fixed',
                  custom_tools=[], is_sub_agent=True, max_model_calls=10)
    agent._operation_counts['model_calls'] = 8
    async def query(*args, **kwargs):
        agent._reserve_call('model_calls', agent.max_model_calls)
        return 'The user constraints and current test failure are preserved in this completed checkpoint.'
    agent._run_compaction_query = query
    assert asyncio.run(agent._generate_folded_session_memory([]))
    assert agent._operation_counts['model_calls'] == 9


def test_stage_budget_is_visible_alongside_total_api_call_budget():
    clock = {'completed': 9, 'last_periodic': 8}
    agent = _FoldingLoop(token_budget={'limit': 100000, 'charged': 0}, max_decisions=10, decision_clock=clock,
        api_base='http://unused.invalid', api_key='test', custom_system_prompt='fixed', custom_tools=[],
        is_sub_agent=True, max_model_calls=50)
    agent._operation_counts['model_calls'] = 12
    raw = [{'role': 'user', 'content': 'Complete this stage.'}]
    projected = agent._messages_with_runtime_state(raw)
    assert 'Solve responses remaining: 1' in projected[-1]['content']
    assert raw[-1]['content'] == 'Complete this stage.'


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
@pytest.mark.parametrize('remaining_path', ['admissible', 'token_limit', 'input_limit', 'unknown_usage'])
def test_rejected_summary_retry_preserves_an_admissible_solve(monkeypatch, provider, remaining_path):
    agent = _FoldingLoop(token_budget={'limit': 15000 if remaining_path == 'token_limit' else 17000,
                                     'charged': 0}, auto_compact=True,
        api_base='http://unused.invalid' if provider == 'openai' else None, api_key='test', model='test',
        custom_system_prompt='fixed', custom_tools=[], skills_enabled=False, is_sub_agent=True,
        max_input_tokens_per_call=32768, max_output_tokens=8192)
    messages = agent._openai_messages if provider == 'openai' else agent._anthropic_messages
    messages.extend([{'role': 'user', 'content': 'Implement parser.'},
                     {'role': 'assistant', 'content': 'Old observed code: ' + 'x' * 450},
                     {'role': 'assistant', 'content': 'Latest observed code: ' + 'y' * 100}])
    original = deepcopy(messages)
    calls = []

    async def truncated(**kwargs):
        calls.append(kwargs['max_tokens'])
        assert len(calls) == 1, 'The second summary must be rejected before the provider.'
        usage = ({'prompt_tokens': 500, 'completion_tokens': kwargs['max_tokens']}
                 if provider == 'openai' else {'input_tokens': 500, 'output_tokens': kwargs['max_tokens']})
        if remaining_path == 'unknown_usage':
            usage = None
        if remaining_path == 'input_limit':
            agent.max_input_tokens_per_call = 1
        if provider == 'openai':
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Partial'),
                                   finish_reason='length')], usage=usage)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='Partial')],
                               stop_reason='max_tokens', usage=usage)

    client = agent._openai_client.chat.completions if provider == 'openai' else agent._anthropic_client.messages
    monkeypatch.setattr(client, 'create', truncated)

    async def exercise():
        assert not await agent._check_and_compact()
        assert messages == original and agent._fold_count == 0
        assert len(calls) == 1 and agent._operation_counts['model_calls'] == 1
        if remaining_path == 'admissible':
            assert agent.token_budget['charged'] == 500 + calls[0]
            assert agent.token_budget['charged'] + agent._next_request_reservation() <= agent.token_budget['limit']
            assert agent.termination_reason is None
            assert not await agent._check_and_compact()
            assert len(calls) == 1
            projected = agent._messages_with_runtime_state(messages)
            agent._preflight_model_input(messages=projected, tools=[],
                system=None if provider == 'openai' else agent._system_prompt)
        elif remaining_path == 'input_limit':
            # Input rejection already has its own recovery path; the raw solve must still fail admission.
            with pytest.raises(_BudgetExceeded, match='Input budget exceeded'):
                agent._preflight_model_input(messages=agent._messages_with_runtime_state(messages), tools=[])
        else:
            assert agent.termination_reason == {'kind': 'token_budget', 'detail': (
                'missing_usage' if remaining_path == 'unknown_usage' else 'request_reservation')}
            if remaining_path == 'unknown_usage':
                assert agent.token_budget['blocked'] == 'missing_usage'
        await agent.shutdown()

    asyncio.run(exercise())
