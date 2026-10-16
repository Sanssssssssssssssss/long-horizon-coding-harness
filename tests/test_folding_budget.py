"""Summary retries must leave one model decision, and expose the active solve cap."""
import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.agent import Agent
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
