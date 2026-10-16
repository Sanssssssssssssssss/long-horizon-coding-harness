import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agents import working_set as current


@pytest.fixture
def project(tmp_path, monkeypatch):
    module = current
    # The same behavioral counterexamples can run against the frozen predecessor.
    if os.environ.get('WORKING_SET_COUNTEREXAMPLE_SOURCE'):
        spec = importlib.util.spec_from_file_location('frozen_projector', os.environ['WORKING_SET_COUNTEREXAMPLE_SOURCE'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ORDER', ('code.py',))
    source = 'def pending_feature():\n    raise NotImplementedError("public requirement")\n'
    (tmp_path / 'code.py').write_text(source, encoding='utf-8')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    agent = NS(_repair_state=NS(execution_id='current'), cell={'workspace': str(tmp_path)},
               token_budget={'limit': 100000, 'charged': 0}, max_input_tokens_per_call=100000)
    view = dict(file='code.py', execution_id='current', source_sha256=digest(tmp_path / 'code.py'),
                evidence_id='current:read', actual_output=source)
    def call(messages=None, **kwargs):
        checker = kwargs.pop('checked_path', lambda work, name: work / name)
        return module.project(agent, messages or [{'role': 'user', 'content': 'Complete every public requirement.'}],
            [view], checked_path=checker, digest=digest,
            cost=lambda _, rows: len(json.dumps(rows)), estimate=lambda _, rows: len(json.dumps(rows)), **kwargs)
    return agent, view, call


def test_green_subset_keeps_current_source_with_unimplemented_work(project):
    _, view, call = project
    result = call(finish_ready=True, completion={'formal_id': 'public', 'diff_id': 'diff'})
    assert view['actual_output'] in json.loads(result[-1]['content'].split(current.TAG+'\n')[1].split('\n'+current.END)[0])['files'][0]['actual_output']


def test_green_subset_does_not_issue_an_end_instruction(project):
    _, _, call = project
    result = call(finish_ready=True)
    info = json.loads(result[-1]['content'].split(current.TAG+'\n')[1].split('\n'+current.END)[0])
    assert 'runtime_next_action' not in info
    assert info['public_verification']['registered_checks_ready']
    assert 'does not establish' in info['public_verification']['limitation']


@pytest.mark.parametrize('stale', ['execution', 'source'])
def test_stale_observed_code_is_never_reinjected(project, stale):
    _, view, call = project
    view['execution_id' if stale == 'execution' else 'source_sha256'] = 'stale'
    result = call(finish_ready=True)
    assert 'pending_feature' not in json.dumps(result)


def test_whole_request_limits_still_control_source_admission(project):
    agent, _, call = project
    agent.max_input_tokens_per_call = 1
    assert 'pending_feature' not in json.dumps(call(finish_ready=True))
    agent.max_input_tokens_per_call = 100000
    agent.token_budget['limit'] = 4096
    assert 'pending_feature' not in json.dumps(call(finish_ready=True))


def test_denied_source_path_has_no_body_and_does_not_interrupt_projection(project):
    _, _, call = project
    def denied(work, name):
        raise AssertionError('outside permitted scope')
    assert 'pending_feature' not in json.dumps(call(finish_ready=True, checked_path=denied))


def test_refresh_replaces_only_own_frames_without_mutating_history(project):
    _, _, call = project
    messages = [{'role': 'user', 'content': '<runtime-working-set>\n{"user_owned": true}\n</runtime-working-set>'}]
    first = call(messages, finish_ready=False)
    before = json.dumps(first)
    refreshed = call(first, finish_ready=True)
    assert json.dumps(first) == before
    assert json.dumps(refreshed).count('registered_checks_ready') == 1
    assert 'user_owned' in json.dumps(refreshed)
