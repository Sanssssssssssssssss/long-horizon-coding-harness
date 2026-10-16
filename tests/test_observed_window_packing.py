"""Coverage and actual request-delivery counterexamples from the CP8 read loop."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agents.tools import format_file_window


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT/'experiment/fixtures/context'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(params=['before', 'after'])
def observed(request, tmp_path):
    patched = request.param == 'after'
    folder = DRAFT if patched else ROOT/'agents'
    pages = module('observed_packing_'+request.param, folder/'observed_pages.py').ObservedPages()
    projector = module('projected_packing_'+request.param, folder/('window_packing.py' if patched else 'working_set.py'))
    projector.ORDER = ('code.py',)
    text = '\n'.join(f'var_{index:04d} = "distinct observed source value {index:04d}"' for index in range(1500))
    path = tmp_path/'code.py'
    path.write_text(text, encoding='utf-8')
    state = NS(execution_id='native')
    count = 0
    def read(offset, limit):
        nonlocal count
        count += 1
        inp = {'file_path': 'code.py', 'offset': offset, 'limit': limit}
        receipt = NS(tool='read_file', status='ok', id=f'native:{count}', input_summary=json.dumps(inp),
                     repo_generation=2, repo_after='current')
        return pages.observe(state, receipt, format_file_window(text, inp), path, 'code.py')
    agent = NS(_repair_state=state, cell={'workspace': str(tmp_path)},
               token_budget={'limit': 1000000, 'charged': 0}, max_input_tokens_per_call=6000)
    digest = lambda value: hashlib.sha256(Path(value).read_bytes()).hexdigest()
    def project(view, messages=None):
        messages = messages or [{'role': 'user', 'content': 'Implement remaining requirements.'}]
        result = projector.project(agent, messages, [view], checked_path=lambda work, name: work/name,
            digest=digest, cost=lambda _, rows: len(json.dumps(rows)), estimate=lambda _, rows: len(json.dumps(rows)))
        info = json.loads(result[-1]['content'].split(projector.TAG+'\n')[1].split('\n'+projector.END)[0])
        return result, info['files'][0]
    return patched, pages, read, agent, project, path


def test_redundant_reads_do_not_erase_complete_observed_revision(observed):
    patched, pages, read, _, _, _ = observed
    read(0, 1000)
    assert read(1000, 1000)['complete']
    for offset in range(10, 100, 10):
        view = read(offset, 10)
    assert view['complete'] is patched
    assert len(pages.files['code.py']['pages']) <= 8
    if patched:
        assert view['observed_ranges'] == [[0, 1500]]
        assert {row['evidence_id'] for row in view['source_receipts']} == {'native:1', 'native:2'}


def test_over_capacity_file_delivers_bounded_recent_native_window(observed):
    patched, _, read, agent, project, _ = observed
    read(0, 1000)
    read(1000, 1000)
    view = read(1100, 10)
    messages, row = project(view)
    assert len(json.dumps(messages)) <= agent.max_input_tokens_per_call
    assert row['body_omitted'] is (not patched)
    if patched:
        assert row['body_scope'] == 'recent_read' and row['body_ranges'] == [[1100, 1110]]
        assert not row['body_complete'] and row['body_evidence_ids'] == [view['evidence_id']]
        assert 'var_1100' in row['actual_output'] and 'var_1109' in row['actual_output']
        assert 'var_1099' not in row['actual_output'] and 'var_1110' not in row['actual_output']


@pytest.mark.parametrize('stale', ['execution', 'source', 'file_removed'])
def test_recent_window_cannot_bypass_source_validity(observed, stale):
    patched, _, read, _, project, path = observed
    view = read(1100, 10)
    if stale == 'file_removed':
        path.unlink()
    else:
        view['execution_id' if stale == 'execution' else 'source_sha256'] = 'other'
    messages, row = project(view)
    assert row['body_omitted'] and 'var_1100' not in json.dumps(messages)


def test_recent_window_respects_both_request_limits_and_does_not_duplicate_tool_body(observed):
    patched, _, read, agent, project, _ = observed
    read(0, 1000)
    read(1000, 1000)
    view = read(1100, 10)
    if not patched:
        return
    original = [{'role': 'user', 'content': view['recent_output']}]
    _, row = project(view, original)
    assert row['body_omitted'] and original[0]['content'] == view['recent_output']
    agent.max_input_tokens_per_call = 1
    assert project(view)[1]['body_omitted']
    agent.max_input_tokens_per_call = 6000
    agent.token_budget['limit'] = 4096
    assert project(view)[1]['body_omitted']
