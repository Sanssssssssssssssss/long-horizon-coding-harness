"""Observed source must remain usable when its full body exceeds request capacity."""
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS
import hashlib

import pytest

from agents.agent import Agent
from agents.tools import format_file_window
from benchmarks.context_folding import _FoldingLoop


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'experiment/research'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scenario(tmp_path):
    if not (RESEARCH / '20261015-observed-page-admission/working_set.py').exists():
        pytest.skip('This isolated research candidate is not present in this checkout.')
    modules = [load('old_admission', RESEARCH / '20261015-fold-overhead/working_set.py'),
               load('new_admission', RESEARCH / '20261015-observed-page-admission/working_set.py')]
    pages = load('admission_pages', RESEARCH / '20261015-observed-window-packing/observed_pages.py').ObservedPages()
    path = tmp_path / 'code.py'
    source = '\n'.join(f'value_{i:04d} = {repr("observed implementation text " * 2)}' for i in range(1200))
    path.write_text(source, encoding='utf-8')
    state = NS(execution_id='current')
    for offset, limit, identifier in ((0, 400, 'prefix'), (800, 400, 'tail'), (1190, 10, 'recent')):
        inp = {'file_path': 'code.py', 'offset': offset, 'limit': limit}
        evidence = NS(tool='read_file', status='ok', id='current:' + identifier,
                      input_summary=json.dumps(inp), repo_generation=1, repo_after='current')
        view = pages.observe(state, evidence, format_file_window(source, inp), path, 'code.py')
    messages = [{'role': 'user', 'content': 'Continue the implementation.'},
                {'role': 'tool', 'content': view['recent_output']}]
    agent = NS(_repair_state=state, cell={'workspace': str(tmp_path)},
               token_budget={'limit': 1000000, 'charged': 0}, context_window=20000,
               max_input_tokens_per_call=20000, max_output_tokens=2048,
               _compaction_reserve_tokens=lambda: 4000)
    estimate = lambda a, m: Agent._estimate_model_input_tokens(messages=m, tools=[])
    cost = lambda a, m: _FoldingLoop._request_reservation(a, messages=m, tools=[])
    for module in modules:
        module.ORDER = ('code.py',)

    def run(module=modules[1], msg=None):
        result = module.project(agent, messages if msg is None else msg, [view],
            checked_path=lambda root, name: root / name,
            digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), cost=cost, estimate=estimate)
        frame = next(json.loads(match[1]) for m in result for match in
            re.finditer(r'<runtime-working-set>\n([^\n]*)\n</runtime-working-set>', m['content']))
        return result, frame['files'][0]

    return NS(agent=agent, view=view, path=path, source=source, messages=messages,
              old=modules[0], new=modules[1], run=run, estimate=estimate, cost=cost)


def test_oversized_source_keeps_missing_observed_windows_with_exact_receipts(scenario):
    s = scenario
    _, before = s.run(s.old)
    assert before['body_omitted']
    result, row = s.run()
    assert not row['body_omitted'] and row['body_scope'] == 'observed_windows'
    assert s.estimate(s.agent, result) <= 16000
    assert s.cost(s.agent, result) <= s.agent.token_budget['limit'] - 4096
    assert not row['body_complete'] and row['verification'] is False
    expected = {i for a, b in row['body_ranges'] for i in range(a, b)}
    actual = {}
    for line in row['actual_output'].split('\n'):
        match = re.match(r'^ *(\d+) \| ', line)
        if match:
            index = int(match[1]) - 1
            assert line == format_file_window(s.source, {'offset': index, 'limit': 1})
            actual[index] = line
    assert set(actual) == expected
    assert expected and not expected.intersection(range(400, 800))
    for i in expected:
        assert any(r['offset'] <= i < r['offset'] + r['line_count']
                   and r['evidence_id'] in row['body_evidence_ids'] for r in row['source_receipts'])


@pytest.mark.parametrize('boundary', ['sha', 'execution', 'required_only', 'budget', 'hard_limit', 'soft_limit'])
def test_invalid_or_unaffordable_source_is_never_admitted(scenario, boundary):
    s = scenario
    if boundary == 'sha':
        s.path.write_text('changed', encoding='utf-8')
    elif boundary == 'execution':
        s.agent._repair_state = NS(execution_id='next')
    elif boundary == 'required_only':
        s.agent._compaction_required_only = True
    elif boundary == 'budget':
        s.agent.token_budget['limit'] = 4096
    elif boundary == 'hard_limit':
        s.agent.max_input_tokens_per_call = 100
    else:
        s.agent.context_window = 4100
    _, row = s.run()
    assert row['body_omitted'] and 'actual_output' not in row


def test_already_visible_body_is_not_duplicated(scenario):
    s = scenario
    _, row = s.run(msg=[{'role': 'user', 'content': s.view['actual_output']}])
    assert row['body_in_recent_messages'] and row['body_omitted']


def test_projection_refresh_preserves_history_and_replaces_only_own_frame(scenario):
    s = scenario
    original = json.dumps(s.messages)
    result, first = s.run()
    saved = json.dumps(result)
    refreshed, second = s.run(msg=result)
    assert json.dumps(s.messages) == original and json.dumps(result) == saved
    assert first == second
    assert sum(m['content'].count('<runtime-working-set>') for m in refreshed) == 1


def test_fully_fitting_source_keeps_existing_whole_body_path(scenario):
    s = scenario
    s.agent.context_window = s.agent.max_input_tokens_per_call = 100000
    _, row = s.run()
    assert row['body_scope'] == 'observed_file'
    assert row['actual_output'] == s.view['actual_output']
