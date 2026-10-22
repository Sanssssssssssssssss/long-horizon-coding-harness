"""Folded implementation context remains recoverable without restoring verification."""
import asyncio
from copy import deepcopy
import json

import pytest

from agents import agent as agent_module, session
from agents.agent import Agent
from agents.session_memory import folded_memory_summary
from agents.tools import get_active_tool_definitions, get_solver_tool_definitions


def make_agent(monkeypatch, tmp_path, provider='openai', *, custom_tools=()):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 300)
    agent = Agent(api_base='http://unused.invalid' if provider == 'openai' else None,
        api_key='test', model='test', custom_system_prompt='fixed',
        custom_tools=None if custom_tools is None else list(custom_tools),
        skills_enabled=False, is_sub_agent=True, max_input_tokens_per_call=32768,
        max_output_tokens=8192)
    agent._session_persistence_enabled = True
    return agent


def exchange(provider, identifier, text):
    if provider == 'openai':
        return [{'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': identifier, 'type': 'function',
                    'function': {'name': 'run_shell', 'arguments': json.dumps({'command': 'inspect ' + identifier})}}]},
                {'role': 'tool', 'tool_call_id': identifier, 'content': text}]
    return [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': identifier,
                'name': 'run_shell', 'input': {'command': 'inspect ' + identifier}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': identifier, 'content': text}]}]


def add_history(agent, provider):
    messages = agent._openai_messages if provider == 'openai' else agent._anthropic_messages
    messages.extend([{'role': 'user', 'content': 'Implement the decoder without losing invalid-input handling.'},
        *exchange(provider, 'old-read', 'EXACT_RECOVERY_ANCHOR\n' + 'old implementation detail\n' * 150),
        *exchange(provider, 'latest-check', 'FAIL: decoding the empty input still fails.')])
    return messages


async def summary(*_args, **_kwargs):
    return json.dumps({'schema_version': 'example', 'implementation_summary':
                      'The decoder exists, but its invalid-input handling is not verified.'})


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
def test_fold_source_survives_resume_and_preserves_complete_tool_exchanges(monkeypatch, tmp_path, provider):
    agent = make_agent(monkeypatch, tmp_path, provider)
    original = deepcopy(add_history(agent, provider))
    monkeypatch.setattr(agent, '_generate_folded_session_memory', summary)

    async def exercise():
        assert await agent._compact_conversation(trigger='manual')
        refs = getattr(agent, '_folded_history_refs', [])
        assert len(refs) == 1, 'A successful fold must retain an addressable original history.'
        identifier = refs[0]['id']
        current = agent._openai_messages if provider == 'openai' else agent._anthropic_messages
        assert current[-2:] == original[-2:]
        checkpoint = next(folded_memory_summary(m) for m in current if folded_memory_summary(m))
        assert json.loads(checkpoint)['schema_version'] == 'example'
        assert identifier in json.dumps(current)
        agent._auto_save()
        saved = session.load_session(agent.session_id)
        resumed = make_agent(monkeypatch, tmp_path, provider)
        resumed.restore_session(saved)
        text = resumed._read_observation({'evidence_id': identifier, 'limit': 1000})
        assert 'EXACT_RECOVERY_ANCHOR' in text
        assert 'inspect old-read' in text and 'id=old-read' in text
        assert 'current_verification=false' in text
        assert 'latest-check' not in text, 'The retained tail must not be archived a second time.'
        assert resumed._repair_state is None and resumed._observations == {}
        assert resumed._read_observation({'evidence_id': 'old-read'}).startswith('Error:')
        await resumed.shutdown()
        await agent.shutdown()

    asyncio.run(exercise())


def test_archive_write_failure_preserves_raw_context(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    original = deepcopy(add_history(agent, 'openai'))
    monkeypatch.setattr(agent, '_generate_folded_session_memory', summary)
    def reject(*_args, **_kwargs):
        raise OSError('constructed archive write failure')
    monkeypatch.setattr(session, 'atomic_write_json', reject, raising=False)

    async def exercise():
        assert not await agent._compact_conversation(trigger='auto')
        assert agent._openai_messages == original and agent._fold_count == 0
        assert not getattr(agent, '_folded_history_refs', [])
        await agent.shutdown()
    asyncio.run(exercise())


def test_repeated_folds_keep_history_chain_and_paged_reads(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    add_history(agent, 'openai')
    monkeypatch.setattr(agent, '_generate_folded_session_memory', summary)

    async def exercise():
        assert await agent._compact_conversation(trigger='manual')
        first = getattr(agent, '_folded_history_refs', [])
        assert len(first) == 1
        first_id = first[0]['id']
        agent._openai_messages.extend(exchange('openai', 'second-read', 'SECOND_ANCHOR\n' * 180))
        agent._openai_messages.extend(exchange('openai', 'new-tail', 'New unresolved error.'))
        assert await agent._compact_conversation(trigger='manual')
        second_id = agent._folded_history_refs[-1]['id']
        assert second_id != first_id
        latest = agent._read_observation({'evidence_id': second_id, 'limit': 1000})
        assert 'SECOND_ANCHOR' in latest and first_id in latest
        old = agent._read_observation({'evidence_id': first_id, 'limit': 1000})
        assert 'EXACT_RECOVERY_ANCHOR' in old
        assert len(agent._read_observation({'evidence_id': first_id, 'limit': 1})) < len(old)
        await agent.shutdown()
    asyncio.run(exercise())


def test_reading_archived_success_claim_cannot_change_current_failed_verification(monkeypatch, tmp_path):
    from test_folding_checkpoint import state, receipt
    agent = make_agent(monkeypatch, tmp_path)
    add_history(agent, 'openai')
    current = agent._repair_state = state()
    agent._repair_task = current.task
    receipt(current, passed=False)
    agent._openai_messages[3]['content'] += '\nA previous model claimed that all tests passed.'
    before = current.snapshot()
    monkeypatch.setattr(agent, '_generate_folded_session_memory', summary)

    async def exercise():
        assert await agent._compact_conversation(trigger='manual')
        text = agent._read_observation({'evidence_id': agent._folded_history_refs[-1]['id'], 'limit': 1000})
        assert 'all tests passed' in text and 'current_verification=false' in text
        assert current.snapshot() == before
        assert not current.completion_decision().complete
        await agent.shutdown()
    asyncio.run(exercise())


@pytest.mark.parametrize('damage', ['contents', 'workspace', 'unregistered'])
def test_archive_reads_refuse_tampering_scope_changes_and_unregistered_ids(monkeypatch, tmp_path, damage):
    agent = make_agent(monkeypatch, tmp_path)
    add_history(agent, 'openai')
    monkeypatch.setattr(agent, '_generate_folded_session_memory', summary)

    async def exercise():
        assert await agent._compact_conversation(trigger='manual')
        refs = getattr(agent, '_folded_history_refs', [])
        assert len(refs) == 1
        identifier = refs[0]['id']
        archive = session.get_project_session_dir() / 'history' / (refs[0]['sha256'] + '.json')
        if damage == 'contents':
            archive.write_text('{}', encoding='utf-8')
        elif damage == 'workspace':
            other = tmp_path / 'other-workspace'
            other.mkdir()
            monkeypatch.chdir(other)
            target = session.get_project_session_dir() / 'history' / archive.name
            target.parent.mkdir()
            target.write_bytes(archive.read_bytes())
        else:
            agent._folded_history_refs = []
        assert agent._read_observation({'evidence_id': identifier}).startswith('Error:')
        assert agent._repair_state is None
        await agent.shutdown()
    asyncio.run(exercise())


@pytest.mark.parametrize('profile', ['native', 'B0', 'B1', 'restricted'])
def test_history_reader_is_exposed_by_native_and_solver_profiles_without_widening_custom_tools(monkeypatch, tmp_path, profile):
    tools = (None if profile == 'native' else
             [] if profile == 'restricted' else get_solver_tool_definitions(profile))
    agent = make_agent(monkeypatch, tmp_path, custom_tools=tools)
    add_history(agent, 'openai')
    monkeypatch.setattr(agent, '_generate_folded_session_memory', summary)

    async def exercise():
        names = [tool['name'] for tool in get_active_tool_definitions(agent.tools)]
        assert names.count('read_observation') == (0 if profile == 'restricted' else 1)
        assert await agent._compact_conversation(trigger='manual')
        identifier = agent._folded_history_refs[-1]['id']
        request = json.dumps(agent._openai_request_messages())
        if profile == 'restricted':
            assert 'read_observation(evidence_id=' not in request
            assert agent.tools == []
        else:
            assert 'read_observation(evidence_id=' in request
            result = await agent._execute_tool_call('read_observation', {'evidence_id': identifier, 'limit': 1000})
            assert 'EXACT_RECOVERY_ANCHOR' in str(result)
        await agent.shutdown()
    asyncio.run(exercise())
