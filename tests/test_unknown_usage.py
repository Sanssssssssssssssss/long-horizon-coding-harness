import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import httpx
import openai
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.agent import _BudgetExceeded, _is_retryable
from benchmarks.context_folding import _FoldingLoop


def test_provider_connection_error_is_retryable():
    assert _is_retryable(openai.APIConnectionError(request=httpx.Request('POST','https://example.invalid/v1')))


@pytest.mark.parametrize('limit',[None,100000])
def test_unmetered_error_blocks_only_a_bounded_budget(limit):
    agent=_FoldingLoop(token_budget={'limit':limit,'charged':0},api_base='https://example.invalid/v1',
                       api_key='test',custom_system_prompt='solver',custom_tools=[],is_sub_agent=True)
    async def failed():
        agent._preflight_model_input(messages=[{'role':'user','content':'task'}],tools=[])
        raise RuntimeError('response lost after submission')
    async def run():
        with pytest.raises(RuntimeError): await agent._await_model_call(failed())
        assert agent.token_budget['unknown_usage_calls']==1
        assert agent.token_budget['charged']>0
        if limit is None:
            assert not agent.token_budget.get('blocked')
            agent._preflight_model_input(messages=[{'role':'user','content':'continue'}],tools=[])
        else:
            with pytest.raises(_BudgetExceeded):
                agent._preflight_model_input(messages=[{'role':'user','content':'continue'}],tools=[])
        await agent.shutdown()
    asyncio.run(run())
