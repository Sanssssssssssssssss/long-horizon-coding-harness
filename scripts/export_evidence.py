"""Export the recorded research cohort as scores and payload-free event logs.

Requires the maintainer's original local runs. Makes no provider calls.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.telemetry import read_trace
from benchmarks.research_budget import usage_from_trace

OUT = ROOT / 'experiment/published/20261016'
RESEARCH = ROOT / 'experiment/research'
sources = {}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    sources[path.relative_to(ROOT).as_posix()] = digest(path)
    return json.loads(path.read_bytes())


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def events(run, target):
    path = run / 'trace.jsonl'
    sources[path.relative_to(ROOT).as_posix()] = digest(path)
    rows = read_trace(path)
    allowed = {'provider', 'model', 'reasoning_mode', 'finish_reason', 'raw_finish_reason',
               'scope', 'trigger', 'checkpoint', 'tool_name', 'state_changed', 'exit_code',
               'error_type', 'reserved'}
    exported = []
    for row in rows:
        item = {key: row[key] for key in ('seq', 'ts_utc', 'event_type', 'component', 'purpose',
                 'status', 'span_id', 'parent_span_id', 'metrics') if key in row}
        item['attributes'] = {key: val for key, val in row.get('attributes', {}).items() if key in allowed}
        if row.get('payload_ref'):
            item['original_payload_sha256'] = row['payload_ref']['sha256']
        exported.append(item)
    dest = OUT / target / 'events.jsonl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in exported), encoding='utf-8')
    assert usage_from_trace(exported) == usage_from_trace(rows)
    return usage_from_trace(rows)


def main():
    assert not (OUT / 'manifest.json').exists(), 'Published packets are immutable; choose a new date.'
    comparison = read(RESEARCH / '20261016-folding-action-repair/RESULT.json')
    write('context/comparison.json', comparison)
    paths = {'baseline': RESEARCH / '20261014-official-long-horizon/runs/baseline',
             'best': RESEARCH / '20261015-page-admission-full/runs/candidate',
             'previous': ROOT / 'experiment/runs/20261016-folding-long-compare/candidate',
             'current': ROOT / 'experiment/runs/20261016-folding-action-repair-v2/candidate'}
    for label, path in paths.items():
        original = read(path / 'receipt.json')
        usage = events(path, 'context/' + label)
        assert usage == original['usage']
        stages = []
        for row in original['checkpoints']:
            xml = path / f"grades/checkpoint-{row['checkpoint']}/official/pytest.xml"
            assert digest(xml) == row['official']['xml_sha256']
            sources[xml.relative_to(ROOT).as_posix()] = digest(xml)
            cases = [{'id': [n.get('classname'), n.get('name')],
                      'status': next((tag for tag in ('failure', 'error', 'skipped')
                                      if n.find(tag) is not None), 'passed')}
                     for n in ET.parse(xml).getroot().findall('.//testcase')]
            assert cases == row['official']['case_status']
            stages.append({key: row[key] for key in ('checkpoint', 'natural_finish', 'termination_reason',
                'solve_responses', 'fold_count', 'usage', 'snapshot_files')} | {
                'official': {'passed': row['official']['passed'], 'tests': row['official']['tests'],
                             'skipped': row['official']['skipped'], 'cases': cases,
                             'original_xml_sha256': digest(xml)}})
        write(f'context/{label}/receipt.json', {'usage': usage, 'stages': stages,
            'original_receipt_sha256': digest(path / 'receipt.json')})
    for name in ('stage8-stall-case.json', 'summary-costs.json', 'ROUND-USAGE.json',
                 'capacity-application.json', 'truncated-summary-case.json', 'trace-audit.json'):
        write('context/diagnostics/' + name, read(RESEARCH / '20261016-folding-action-repair' / name))
    replay = ROOT / 'experiment/runs/20261016-folding-capacity-live'
    write('context/capacity-replay/receipt.json', read(replay / 'result.json'))
    events(replay, 'context/capacity-replay')

    all_rows = []
    for cohort in ('20261016-skilllearn-20', '20261016-skilllearn-20-v2'):
        result = read(RESEARCH / cohort / 'RESULT.json')
        protocol = read(RESEARCH / cohort / 'protocol.json')
        write(f'skill/{cohort}-protocol.json', {key: protocol[key] for key in (
            'upstream', 'model', 'families', 'train', 'test_instances', 'limits', 'source_files',
            'source_archive_sha256', 'skill_policy', 'folding', 'adaptations', 'limitations')})
        for row in result['rows']:
            path = ROOT / row['path'].replace('\\', '/')
            original = read(path / 'receipt.json')
            record = {key: original[key] for key in ('task', 'family', 'instance', 'split', 'model',
                'protocol_sha256', 'skill_sha256', 'termination', 'decisions', 'grade', 'usage',
                'natural_finish', 'completion_complete', 'environment_status')}
            trace_run = path if (path / 'trace.jsonl').exists() else (
                ROOT / original['snapshot'].replace('\\', '/')).parent.parent
            record['trace_source_run'] = trace_run.relative_to(ROOT).as_posix()
            record['offline_regrade'] = trace_run != path
            assert events(trace_run, 'skill/runs/' + row['split'] + '-' + row['task']) == record['usage']
            write('skill/runs/' + row['split'] + '-' + row['task'] + '/receipt.json', record)
            all_rows.append(row)
    test = [row for row in all_rows if row['split'] == 'test']
    assert len(test) == len({row['task'] for row in test}) == 20
    ledger = result['ledger']  # v2 includes the earlier cohort; count it once.
    write('skill/summary.json', {'tasks': 20, 'passed': sum(row['pass'] for row in test),
        'natural_finish': sum(row['natural'] for row in test), 'skill_loaded': sum(row['skill'] for row in test),
        'families': dict(Counter(row['family'] for row in test)), 'rows': all_rows,
        'usage_including_train_and_distillation': {key: ledger[key] for key in
            ('known_api_tokens', 'physical_requests', 'unknown_usage_requests')},
        'limitations': 'Five families with four test instances each; not 20 independent families. '
            'No fresh no-Skill control. Only eight test tasks loaded a Skill. No causal uplift claim.'})
    write('provenance.json', {'sources': sources,
        'projection': 'Event identifiers, timestamps, metrics and selected attributes. Model/tool payloads excluded. '
            'Scores extracted from original receipts; context case statuses checked against original XML.',
        'source_scope': 'Frozen experiment versions; published runtime may contain later fixes.'})
    manifest = [{'path': path.relative_to(OUT).as_posix(), 'sha256': digest(path), 'bytes': path.stat().st_size}
                for path in sorted(OUT.rglob('*')) if path.is_file()]
    write('manifest.json', {'entries': manifest})
    print(json.dumps({'files': len(manifest), 'bytes': sum(row['bytes'] for row in manifest),
                      'skill_tasks': len(test), 'api_calls': 0}))


if __name__ == '__main__':
    main()
