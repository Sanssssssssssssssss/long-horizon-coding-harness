import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from agents.observed_pages import ObservedPages
from agents.tools import format_file_window


def test_partial_current_revision_survives_without_exposing_unread_lines(tmp_path):
    path = tmp_path/'large.py'
    text = '\n'.join(f'value_{i} = {i}' for i in range(1351))
    path.write_text(text, encoding='utf-8')
    state = NS(execution_id='task')
    cache = ObservedPages()

    def read(offset, limit, identifier):
        current = path.read_text(encoding='utf-8')
        inp = {'file_path': 'large.py', 'offset': offset, 'limit': limit}
        evidence = NS(tool='read_file', status='ok', id='task:'+identifier,
                      input_summary=json.dumps(inp), repo_generation=1, repo_after='current')
        return cache.observe(state, evidence, format_file_window(current, inp), path, 'large.py')

    before = read(1200, 100, 'old-tail')
    assert before is not None and 'value_1200' in before['actual_output']
    path.write_text(text.replace('value_1 = 1', 'changed_value_1 = 1'), encoding='utf-8')
    after = read(0, 1000, 'current-prefix')
    assert after is not None and not after['complete']
    assert after['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'changed_value_1 = 1' in after['actual_output']
    assert 'value_1200' not in after['actual_output']
    assert 'value_1300' not in after['actual_output']
    assert all(r['evidence_id'] != 'task:old-tail' for r in after['source_receipts'])
    tail = read(1200, 100, 'current-tail')
    assert 'value_1200' in tail['actual_output'] and not tail['verification']
    assert 'value_1100' not in tail['actual_output']


@pytest.mark.parametrize('scenario', ['large_complete_read', 'shorter_reread'])
def test_observed_unchanged_source_does_not_lose_previously_read_lines(tmp_path, scenario):
    path = tmp_path/'source.py'
    lines = 780 if scenario == 'large_complete_read' else 80
    text = '\n'.join(f'value_{i:04d} = "already observed implementation detail"' for i in range(lines))
    path.write_text(text, encoding='utf-8')
    state = NS(execution_id='task')
    cache = ObservedPages()

    def read(limit, identifier):
        inp = {'file_path': 'source.py', 'offset': 0, 'limit': limit}
        evidence = NS(tool='read_file', status='ok', id='task:'+identifier,
                      input_summary=json.dumps(inp), repo_generation=1, repo_after='current')
        return cache.observe(state, evidence, format_file_window(text, inp), path, 'source.py')

    view = read(1000, 'full')
    if scenario == 'shorter_reread':
        view = read(2, 'short')
    assert view['complete']
    assert view['observed_ranges'] == [[0, lines]]
    assert view['actual_output'] == format_file_window(text, {'offset': 0, 'limit': 1000})
    assert view['source_receipts'] == [{'evidence_id': 'task:full', 'offset': 0, 'line_count': lines}]
