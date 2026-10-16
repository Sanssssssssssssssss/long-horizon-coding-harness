"""Native pressure checks for source projection; all execution stays local."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agents.agent import Agent
from agents.tools import format_file_window


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'experiment/research'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=['before', 'after'])
def rig(request, tmp_path):
    patched = request.param == 'after'
    directory = '20261015-context-pressure-admission' if patched else '20261015-observed-window-packing'
    projector = load('pressure_'+request.param, RESEARCH/directory/'working_set.py')
    projector.ORDER = ('code.py',)
    pages = load('pressure_pages', RESEARCH/'20261015-observed-window-packing/observed_pages.py').ObservedPages()
    source = '\n'.join(f'value_{index:04d} = "observed implementation content with distinct value {index:04d}"' for index in range(2200))
    path = tmp_path/'code.py'
    path.write_text(source, encoding='utf-8')
    state = NS(execution_id='execution')
    view = None
    for number, (offset, limit) in enumerate(((0, 1000), (1000, 1000), (2000, 1000), (1700, 10))):
        inp = {'file_path': 'code.py', 'offset': offset, 'limit': limit}
        evidence = NS(tool='read_file', status='ok', id=f'execution:e{number}',
                      input_summary=json.dumps(inp), repo_generation=0, repo_after='current')
        view = pages.observe(state, evidence, format_file_window(source, inp), path, 'code.py')
    assert view['complete']
    agent = NS(_repair_state=state, cell={'workspace': str(tmp_path)}, use_openai=True,
               context_window=65536, max_input_tokens_per_call=65536, max_output_tokens=32768,
               token_budget={'limit': 2000000, 'charged': 0}, last_input_token_count=0)
    agent._compaction_reserve_tokens = lambda: Agent._compaction_reserve_tokens(agent)
    estimate = lambda _, messages: Agent._estimate_model_input_tokens(messages=messages, tools=[])
    digest = lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest()

    def project(messages=None):
        messages = messages or [{'role': 'user', 'content': 'Continue the current public task.'}]
        result = projector.project(agent, messages, [view], checked_path=lambda work, name: work/name,
            digest=digest, cost=lambda _, rows: len(json.dumps(rows)), estimate=estimate)
        row = json.loads(result[-1]['content'].split(projector.TAG+'\n')[1].split('\n'+projector.END)[0])['files'][0]
        return result, row

    def triggers(messages):
        calls = []
        agent._openai_messages = messages
        agent._conversation_input_estimate = lambda rows: estimate(agent, rows)
        async def record(*, trigger):
            calls.append(trigger)
        agent._compact_conversation = record
        asyncio.run(Agent._check_and_compact(agent))
        return calls
    return patched, agent, view, project, triggers


def test_optional_source_cannot_induce_another_native_auto_fold(rig):
    patched, agent, view, project, triggers = rig
    messages, row = project()
    assert Agent._estimate_model_input_tokens(messages=messages, tools=[]) < agent.max_input_tokens_per_call
    assert triggers(messages) == ([] if patched else ['auto'])
    assert row['body_scope'] == ('recent_read' if patched else 'observed_file')
    if patched:
        assert row['body_ranges'] == [view['recent_range']]
        assert row['body_evidence_ids'] == [view['evidence_id']]
        assert not row['body_complete']


def test_recent_tool_exchange_survives_optional_projection_without_duplicate_body(rig):
    patched, _, view, project, triggers = rig
    original = [{'role': 'user', 'content': 'Current task'},
                {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'read', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'read', 'content': view['recent_output']}]
    snapshot = json.dumps(original)
    messages, row = project(original)
    assert json.dumps(original) == snapshot
    assert messages[1] == original[1] and messages[2]['content'].startswith(original[2]['content'])
    assert triggers(messages) == ([] if patched else ['auto'])
    if patched:
        assert row['body_omitted']


def test_mandatory_context_pressure_still_triggers_native_compaction(rig):
    patched, _, _, project, triggers = rig
    messages, row = project([{'role': 'user', 'content': 'required context ' * 11000}])
    if patched:
        assert row['body_omitted']
    assert triggers(messages) == ['auto']


@pytest.mark.parametrize('boundary', ['hard_input', 'remaining_budget'])
def test_existing_request_limits_still_block_optional_source(rig, boundary):
    _, agent, _, project, _ = rig
    if boundary == 'hard_input':
        agent.max_input_tokens_per_call = 1
    else:
        agent.token_budget['limit'] = 4096
    assert project()[1]['body_omitted']
