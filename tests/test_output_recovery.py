"""Local SDK streams with explicitly synthetic usage; all pytest commands execute for real."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import openai
import pytest

from agents.agent import _BudgetExceeded
from agents.telemetry import TraceRecorder, read_trace
from agents.tools import get_solver_tool_definitions, tool_definitions
from benchmarks.context_folding import _FoldingLoop
from benchmarks.research_budget import usage_from_trace

COMMAND = 'python -B -m pytest test_draft.py -q -c pytest.ini'
BROKEN = 'def double(value):\n    return value + 1\n'
FIXED = 'def double(value):\n    return value * 2\n'


def tool(name, arguments):
    return {'name': name, 'arguments': arguments if isinstance(arguments, str) else json.dumps(arguments)}


def fixture(tmp_path, monkeypatch, replies):
    work = tmp_path / 'workspace'
    work.mkdir()
    (work / 'helper.py').write_text(BROKEN, encoding='utf-8', newline='')
    (work / 'test_draft.py').write_text('from helper import double\ndef test_value():\n    assert double(3) == 6\n', encoding='utf-8', newline='')
    (work / 'pytest.ini').write_text('[pytest]\naddopts = --rootdir=. -p no:cacheprovider\n', encoding='utf-8', newline='')
    (work / '.gitignore').write_text('__pycache__/\n.pytest_cache/\nexperiment/\n', encoding='utf-8', newline='')
    for args in (['init', '-q'], ['config', 'core.autocrlf', 'false'], ['add', '.'],
                 ['-c', 'user.name=Research', '-c', 'user.email=research@localhost', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Public fixture']):
        subprocess.run(['git', *args], cwd=work, capture_output=True, check=True)
    monkeypatch.chdir(work)
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    monkeypatch.setenv('PYTEST_ADDOPTS', '--rootdir=. -p no:cacheprovider')
    requests = []
    async def respond(request):
        index = len(requests)
        requests.append(json.loads(request.content))
        assert index < len(replies), 'Unexpected extra physical provider request'
        finish, calls, usage = replies[index]
        if index == 3 and len(replies) == 6:
            assert (work / 'helper.py').read_text() == BROKEN  # Truncated write did not mutate delivery.
        delta = {'content': 'partial output' if finish == 'length' else ''}
        if calls:
            delta['tool_calls'] = [{'index': i, 'id': f'call-{index}-{i}', 'type': 'function',
                                   'function': call} for i, call in enumerate(calls)]
        chunk = {'id': 'local-stream', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'test',
                 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
        if usage:
            chunk['usage'] = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text='data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n')
    a = _FoldingLoop(token_budget={'charged': 0, 'limit': 100000}, auto_compact=False,
        api_base='https://example.invalid/v1', api_key='test', model='test', custom_system_prompt='Repair the public function and verify it.',
        custom_tools=[t for t in get_solver_tool_definitions('B1') if t['name'] in {'read_file', 'write_file', 'run_shell'}],
        skills_enabled=False, is_sub_agent=True, permission_mode='bypassPermissions', repair_task=None,
        max_model_calls=12, max_tool_calls=20, max_output_tokens=2048, max_input_tokens_per_call=32768,
        trace=TraceRecorder(tmp_path / 'trace', run_id='mock', task_id='recovery', attempt_id='synthetic'))
    return a, work, requests, respond


async def connect(a, respond):
    await a._openai_client.close()
    a._openai_client = openai.AsyncOpenAI(api_key='test', base_url='https://example.invalid/v1', max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))


def pytest_exits(a):
    exits = []
    for event in read_trace(a.trace.trace_path):
        if event['event_type'] in {'tool.call.end', 'tool.call.error'} and event.get('purpose') == 'run_shell':
            data = json.loads((a.trace.blob_dir / event['payload_ref']['sha256']).read_bytes())
            execution = data.get('execution') or {}
            if execution.get('pytest_report'):
                exits.append(execution['exit_code'])
    return exits


@pytest.mark.parametrize('complete_prefix', [False, True])
def test_truncated_write_recovers_with_real_failed_to_passed_pytest(tmp_path, monkeypatch, complete_prefix):
    clipped = [tool('read_file', {'file_path': 'helper.py'})] if complete_prefix else []
    clipped.append(tool('write_file', '{"file_path":"helper.py","content":"def double(value):'))
    replies = [('tool_calls', [tool('read_file', {'file_path': 'helper.py'})], True),
               ('tool_calls', [tool('run_shell', {'command': COMMAND})], True), ('length', clipped, True),
               ('tool_calls', [tool('write_file', {'file_path': 'helper.py', 'content': FIXED})], True),
               ('tool_calls', [tool('run_shell', {'command': COMMAND})], True), ('stop', [], True)]
    a, work, requests, respond = fixture(tmp_path, monkeypatch, replies)
    async def run():
        await connect(a, respond)
        try:
            result = await a.run_once('Repair double(value) to return twice the value. Keep the test unchanged.')
            assert result['termination_reason'] is None and a.finish_reason == 'stop'
            assert pytest_exits(a) == [1, 0] and (work / 'helper.py').read_text() == FIXED
            assert len(requests) == 6 and a._operation_counts['model_calls'] == 6
            assert a.token_budget['charged'] == 72  # Synthetic provider usage, not paid tokens.
            assert a._length_recoveries == 1
            events = read_trace(a.trace.trace_path)
            writes = [e for e in events if e['event_type'] == 'tool.call.start' and e.get('purpose') == 'write_file']
            assert len(writes) == 1  # Incomplete JSON never reached execution.
            reads = [e for e in events if e['event_type'] == 'tool.call.end' and e.get('purpose') == 'read_file']
            assert len(reads) == 1  # The entire truncated batch is rejected, including its complete prefix.
            truncated = [e for e in events if e['event_type'] == 'model.call.end' and
                         e.get('attributes', {}).get('raw_finish_reason') == 'length']
            assert len(truncated) == 1
            raw = json.loads((a.trace.blob_dir / truncated[0]['payload_ref']['sha256']).read_bytes())
            assert len(raw['tool_calls']) == (2 if complete_prefix else 1)  # Original provider trace is retained.
            assert a._verification_history[0]['success'] is False
        finally:
            await a.shutdown()
    asyncio.run(run())


def test_three_truncations_stop_at_two_recoveries_and_retain_failure(tmp_path, monkeypatch):
    replies = [('tool_calls', [tool('run_shell', {'command': COMMAND})], True)] + [('length', [], True)] * 3
    a, work, requests, respond = fixture(tmp_path, monkeypatch, replies)
    async def run():
        await connect(a, respond)
        try:
            result = await a.run_once('Diagnose the current failure.')
            assert result['termination_reason'] == {'kind': 'incomplete_response', 'detail': 'length'}
            assert a._length_recoveries == 2 and len(requests) == 4 and a.token_budget['charged'] == 48
            assert pytest_exits(a) == [1] and (work / 'helper.py').read_text() == BROKEN
            assert a._verification_history[0]['success'] is False
            assert sum(e['event_type'] == 'model.truncation_recovery' for e in read_trace(a.trace.trace_path)) == 2
        finally:
            await a.shutdown()
    asyncio.run(run())


def test_unknown_usage_truncation_blocks_another_request(tmp_path, monkeypatch):
    a, _, requests, respond = fixture(tmp_path, monkeypatch, [('length', [], False)])
    async def run():
        await connect(a, respond)
        try:
            with pytest.raises(RuntimeError, match='missing usage'):
                await a.run_once('Inspect the failure.')
            assert a.token_budget['blocked'] == 'unknown_usage_after_error' and a.token_budget['charged'] > 0
            assert a._length_recoveries == 0
            with pytest.raises(_BudgetExceeded):
                await a._call_openai_stream()
            assert len(requests) == 1 and usage_from_trace(read_trace(a.trace.trace_path))['unknown_usage_requests'] == 1
        finally:
            await a.shutdown()
    asyncio.run(run())


def test_recovery_allowance_does_not_reset_between_run_once_phases(tmp_path, monkeypatch):
    a, _, requests, respond = fixture(tmp_path, monkeypatch,
        [('length', [], True), ('max_tokens', [], True), ('stop', [], True), ('length', [], True)])
    async def run():
        await connect(a, respond)
        try:
            first = await a.run_once('First read-only checkpoint.')
            assert first['termination_reason'] is None and a._length_recoveries == 2
            second = await a.run_once('Next read-only checkpoint.')
            assert second['termination_reason'] == {'kind': 'incomplete_response', 'detail': 'length'}
            assert a._length_recoveries == 2 and len(requests) == 4 and a.token_budget['charged'] == 48
        finally:
            await a.shutdown()
    asyncio.run(run())


@pytest.mark.parametrize('bad_arguments', ['{"file_path":', '[]', 'null'])
def test_normal_malformed_arguments_keep_call_balance_and_allow_zero_arg_tool(tmp_path, monkeypatch, bad_arguments):
    a, work, requests, respond = fixture(tmp_path, monkeypatch, [
        ('stop', [tool('write_file', bad_arguments)], True),
        ('tool_calls', [tool('compact_context', {})], True), ('stop', [], True)])
    a.tools.append(next(t for t in tool_definitions if t['name'] == 'compact_context'))
    async def run():
        await connect(a, respond)
        try:
            await a.run_once('Inspect files only.')
            assert len(requests) == 3 and a.token_budget['charged'] == 36
            assert (work / 'helper.py').read_text() == BROKEN and a._length_recoveries == 0
            matching = [m for m in requests[1]['messages'] if m.get('role') == 'tool' and m.get('tool_call_id') == 'call-0-0']
            assert len(matching) == 1 and 'no tool was executed' in matching[0]['content']
            events = read_trace(a.trace.trace_path)
            assert not any(e['event_type'] == 'tool.call.start' and e.get('purpose') == 'write_file' for e in events)
            assert sum(e['event_type'] == 'tool.call.start' and e.get('purpose') == 'compact_context' for e in events) == 1
            assert a._operation_counts == {'model_calls': 3, 'tool_calls': 1}
        finally:
            await a.shutdown()
    asyncio.run(run())
