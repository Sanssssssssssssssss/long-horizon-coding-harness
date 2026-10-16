"""Native execution counterexamples for mutable actor tests and public receipts."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agents.repair_state import TaskEnvelope
from agents.tools import get_solver_tool_definitions


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def write(agent, inp):
    await agent._execute_tool_call('read_file', {'file_path': inp['file_path']})
    result = await agent._execute_tool_call('write_file', inp)
    assert (agent.workspace/inp['file_path']).read_text(encoding='utf-8') == inp['content'], result


@pytest.fixture
def rig(tmp_path, monkeypatch):
    caller = load('split_test_caller', ROOT/'experiment/fixtures/context/caller.py')
    split = load('split_test_adapter', ROOT/'experiment/fixtures/context/split_checks.py')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    monkeypatch.setattr('agents.agent.Path.home', lambda: tmp_path)
    caller.legacy.CASES[1] = [('P', 'Keep public behavior', 'Keep public behavior')]
    (tmp_path/'target.py').write_text('VALUE = 0\n', encoding='utf-8')
    (tmp_path/'test_checkpoint_1.py').write_text('from target import VALUE\ndef test_public():\n    assert VALUE == 1\n', encoding='utf-8')
    # Match the benchmark scaffold, including platform text newline conversion.
    (tmp_path/'test_actor.py').write_text(split.EMPTY_ACTOR.decode('utf-8'), encoding='utf-8')
    empty_actor_sha256 = hashlib.sha256((tmp_path/'test_actor.py').read_bytes()).hexdigest()
    (tmp_path/'circopt.py').write_text('# source window fixture\n', encoding='utf-8')
    (tmp_path/'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path/'.gitignore').write_text('__pycache__/\n.pytest_cache/\nexperiment/\n', encoding='utf-8')
    for args in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'baseline']):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)

    def construct(patched):
        cls = (split.install(caller, caller.TextCaller, empty_actor_sha256=empty_actor_sha256)
               if patched else caller.TextCaller)
        agent = cls(workspace=tmp_path, api_base='http://unused.invalid', api_key='offline',
            token_budget={'limit': 0, 'charged': 0},
            is_sub_agent=True, skills_enabled=False, custom_system_prompt='Offline execution only',
            custom_tools=get_solver_tool_definitions('B0'),
            repair_task=TaskEnvelope('split', 'Keep public behavior', str(tmp_path),
                                     requires_patch=False, requires_repair_witness=False))
        async def forbidden(*args, **kwargs):
            raise AssertionError('No API call is allowed in this regression')
        monkeypatch.setattr(agent._openai_client.chat.completions, 'create', forbidden)
        return agent
    return tmp_path, caller, split, construct


@pytest.mark.parametrize('patched', [False, True])
def test_adding_passing_actor_assertions_preserves_original_public_failure_resolution(rig, patched):
    work, _, _, construct = rig
    agent = construct(patched)
    async def run():
        await agent._verify_current()
        assert not agent._repair_state.completion_decision().complete
        await write(agent, {'file_path': 'target.py', 'content': 'VALUE = 1\n'})
        assert agent._repair_state.completion_decision().complete
        await write(agent, {'file_path': 'test_actor.py',
            'content': 'from target import VALUE\ndef test_actor():\n    assert VALUE > 0\n'})
        assert agent._repair_state.completion_decision().complete is patched
        public = [v for v in agent._repair_state.verifications if v.acceptance_ids]
        assert (public[0].check_fingerprint == public[-1].check_fingerprint) is patched
        if patched:
            frame = next(m['content'] for m in reversed(agent._pending_public_notes)
                         if 'runtime-actor-test' in m['content'])
            note = json.loads(frame.split('\n')[1])
            assert note['success'] and note['runner_recognized']
            assert 'verification metadata missing' not in note['actual_output']
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['public_regression', 'actor_failure', 'weakened_public'])
def test_split_checks_keep_real_failures_and_changed_conditions_blocking(rig, failure):
    work, _, _, construct = rig
    agent = construct(True)
    async def run():
        await agent._verify_current()
        await write(agent, {'file_path': 'target.py', 'content': 'VALUE = 1\n'})
        assert agent._repair_state.completion_decision().complete
        if failure == 'public_regression':
            await write(agent, {'file_path': 'target.py', 'content': 'VALUE = 0\n'})
        elif failure == 'actor_failure':
            await write(agent, {'file_path': 'test_actor.py',
                'content': 'from target import VALUE\ndef test_actor():\n    assert VALUE == 2\n'})
        else:
            # Inject tampering outside the actor guard to exercise the native evidence boundary.
            (work/'test_checkpoint_1.py').write_text('def test_public():\n    assert True\n', encoding='utf-8')
            agent._repair_state.observe_repo_state(await agent._probe_repo_state())
            await agent._verify_current()
        assert not agent._repair_state.completion_decision().complete
    asyncio.run(run())


def test_empty_actor_execution_is_never_reported_as_successful_assertions(rig):
    _, _, _, construct = rig
    agent = construct(True)
    async def run():
        await agent._verify_current()
        assert not any('runtime-actor-test' in m['content'] for m in agent._pending_public_notes)
        await write(agent, {'file_path': 'test_actor.py', 'content': '# still no tests\n'})
        frame = next(m['content'] for m in reversed(agent._pending_public_notes) if 'runtime-actor-test' in m['content'])
        note = json.loads(frame.split('\n')[1])
        assert note['success'] is None and note['verification_id'] is None
        assert not note['test_cases'] and 'inconclusive' in note['classification']
    asyncio.run(run())


def test_eight_public_files_get_native_fingerprint_and_actor_cannot_claim_acceptance(rig):
    work, caller, split, construct = rig
    original = caller.legacy.public_command(8)
    agent = construct(True)
    for number in range(2, 9):
        (work/f'test_checkpoint_{number}.py').write_text('def test_extra():\n    assert True\n', encoding='utf-8')
    async def run():
        assert await agent._check_fingerprint(original) is None
        fingerprint = await agent._check_fingerprint(caller.legacy.public_command(8))
        assert fingerprint is not None and len(fingerprint) == 64
    asyncio.run(run())
    assert caller.legacy.scoped('run_shell', {'command': split.ACTOR_COMMAND}, work, 8, {})
    for forbidden in ({'acceptance_ids': ['P']}, {'diagnostic': True}):
        with pytest.raises(AssertionError):
            caller.legacy.scoped('run_shell', {'command': split.ACTOR_COMMAND, **forbidden}, work, 8, {})
