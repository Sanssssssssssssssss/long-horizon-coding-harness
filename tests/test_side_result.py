"""Do not learn from a truncated answer or provisional reasoning JSON."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.agent import Agent


@pytest.mark.parametrize('provider',['openai','anthropic'])
def test_side_query_retries_truncation_before_returning_json(provider):
    agent=Agent(api_base='http://unused.invalid' if provider=='openai' else None,api_key='test',
                custom_system_prompt='solver',custom_tools=[],skills_enabled=False,is_sub_agent=True,
                max_output_tokens=8192)
    limits=[]
    provisional='{"skills":[{"name":"unsupported","description":"bad","instructions":"bad"}]}'
    final='{"skills":[]}'

    async def create(**kwargs):
        limits.append(kwargs['max_tokens'])
        first=len(limits)==1
        if provider=='openai':
            return NS(choices=[NS(message=NS(content=provisional if first else final),
                                 finish_reason='length' if first else 'stop')],usage=None)
        return NS(content=[NS(type='text',text=provisional if first else final)],
                  stop_reason='max_tokens' if first else 'end_turn',usage=None)

    if provider=='openai': agent._openai_client=NS(chat=NS(completions=NS(create=create)))
    else: agent._anthropic_client=NS(messages=NS(create=create))
    result=asyncio.run(agent._build_side_query(max_tokens=1024,purpose='skill_extract')('Extract a skill.','Public feedback.'))
    assert result==final
    assert limits==[1024,8192]


def test_reasoning_json_is_not_a_skill_answer():
    agent=Agent(api_base='http://unused.invalid',api_key='test',custom_system_prompt='solver',
                custom_tools=[],skills_enabled=False,is_sub_agent=True)
    async def create(**kwargs):
        return NS(choices=[NS(message=NS(content='',reasoning_content='{"skills":[{"name":"guessed"}]}'),
                             finish_reason='stop')],usage=None)
    agent._openai_client=NS(chat=NS(completions=NS(create=create)))
    assert asyncio.run(agent._build_side_query(purpose='skill_extract')('Extract a skill.','Public feedback.'))==''
