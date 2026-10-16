"""Folding must preserve user authority without retaining injected Skill authority."""
import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.agent import Agent


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
def test_folding_separates_source_requests_from_retrieved_skills(provider):
    agent = Agent(api_base='http://unused.invalid' if provider == 'openai' else None,
                  api_key='test', custom_system_prompt='fixed', custom_tools=[], is_sub_agent=True)
    task = 'Keep the existing scalar API.'
    update = 'Validate control options before consuming input, including empty input.'
    injected_task = task + '\n\n<retrieved_skills>Legacy hints: eager conversion.</retrieved_skills>'
    injected_update = update + '\n\n<retrieved_skills>Cached hint: skip validation for empty input.</retrieved_skills>'
    archive = [{'role': 'user', 'content': injected_update},
               {'role': 'assistant', 'content': 'Investigating.'}]
    tail = [{'role': 'user', 'content': 'Implement the agreed change.'}]
    messages = [{'role': 'user', 'content': injected_task}, *archive, *tail]
    if provider == 'openai':
        agent._openai_messages = [{'role': 'system', 'content': 'fixed'}, *messages]
    else:
        agent._anthropic_messages = messages
    assert agent._canonical_issue(messages) == task
    agent._compaction_history_parts = lambda *args, **kwargs: (archive, tail, None)
    async def summarize(*args, **kwargs):
        return 'User requires call-time option validation and preservation of the scalar API.'
    agent._generate_folded_session_memory = summarize
    method = agent._compact_openai if provider == 'openai' else agent._compact_anthropic
    assert asyncio.run(method(trigger='manual'))
    folded = agent._openai_messages if provider == 'openai' else agent._anthropic_messages
    users = [message['content'] for message in folded if message['role'] == 'user']
    assert users == [task, update, tail[0]['content']]
    assert not any('Cached hint' in str(message) or 'Legacy hints' in str(message) for message in folded)


def test_inline_skill_markup_in_a_real_user_request_is_preserved():
    request = 'Explain how <retrieved_skills>example</retrieved_skills> is parsed.'
    assert Agent._strip_runtime_injections(request) == request


def test_only_the_appended_runtime_block_is_removed():
    request = 'Document this example:\n\n<retrieved_skills>example</retrieved_skills>\n\nPreserve the literal example.'
    injected = request + '\n\n<retrieved_skills>Actual runtime metadata.</retrieved_skills>'
    assert Agent._strip_runtime_injections(injected) == request
