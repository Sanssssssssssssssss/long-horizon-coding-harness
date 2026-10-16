"""Check tracked source, documentation and published evidence without model calls."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agents.telemetry import read_trace
from benchmarks.research_budget import usage_from_trace
from skill_cohort import summarize as summarize_skill_cohort


def main():
    files = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().strip('\0').split('\0'))
    required = {'README.md', 'README.zh-CN.md', 'docs/README.md', 'docs/architecture.md',
        'docs/skills.md', 'docs/context-folding.md', 'docs/evaluation.md', 'docs/running.md',
        'TESTING.md', 'tests/INDEX.md', 'CONTRIBUTING.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md',
        'SNAPSHOT_NOTICE.md', '.env.example', 'requirements.lock.txt', '.github/workflows/checks.yml'}
    assert required <= files, f'Missing publication files: {sorted(required - files)}'
    for name in files:
        assert name not in {'.env', '.mcp.json'}, name
        assert not any(part in {'.venv', '__pycache__', '.pytest_cache'} for part in Path(name).parts), name
        assert not name.startswith(('wiki/', 'experiment/research/', 'experiment/archive/', 'experiment/cache/',
            'experiment/runs/', 'experiment/runtime/', 'experiment/maintenance/')), name
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8'), filename=name)
    index = (ROOT / 'tests/INDEX.md').read_text(encoding='utf-8')
    suites = [name for name in files if name.startswith('tests/')
              and Path(name).name.startswith('test_') and name.endswith('.py')]
    for name in suites:
        assert Path(name).name in index, f'Test index missing {name}'

    documents = {name for name in required if name.endswith('.md')}
    documents |= {name for name in files if name.startswith('docs/') and name.endswith('.md')}
    documents |= {'experiment/README.md', 'experiment/published/20261016/README.md'}
    documents |= {name for name in files if name.startswith(('experiment/published/', 'experiment/fixtures/'))
                  and name.endswith('/README.md')}
    links = 0
    for name in documents:
        path = ROOT / name
        text = path.read_text(encoding='utf-8')
        targets = re.findall(r'\]\(([^)]+)\)', text) + re.findall(r'(?:href|src)="([^"]+)"', text)
        for target in targets:
            if target.startswith(('http:', 'https:', '#')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.exists(), f'Broken link: {name}: {target}'
            assert resolved.is_dir() or resolved.relative_to(ROOT).as_posix() in files, target
            links += 1

    artifacts = 0
    for name in files:
        if not name.startswith('experiment/published/') or not name.endswith('/manifest.json'):
            continue
        path = ROOT / name
        for entry in json.loads(path.read_bytes())['entries']:
            item = (path.parent / entry['path']).resolve()
            assert item.is_relative_to(path.parent.resolve())
            assert item.relative_to(ROOT).as_posix() in files
            assert sha256(item.read_bytes()).hexdigest() == entry['sha256'], str(item)
            assert item.stat().st_size == entry['bytes']
            artifacts += 1

    fixtures = json.loads((ROOT / 'experiment/fixtures/manifest.json').read_bytes())
    for entry in fixtures['entries']:
        assert entry['current'] in files
        assert sha256((ROOT / entry['current']).read_bytes()).hexdigest() == entry['current_sha256']
    scope = ROOT / 'experiment/published/20261017-scope'
    assert summarize_skill_cohort() == json.loads((scope / 'selection.json').read_bytes())
    forecast = json.loads((scope / 'forecast.json').read_bytes())
    inventory = json.loads((scope / 'upstream-inventory.json').read_bytes())
    assert forecast['kind'] == 'judgment_forecast' and forecast['new_model_calls'] == 0
    assert sum(inventory['families'].values()) == inventory['catalog_instances'] == 100
    for kind in ('catalog', 'heldout'):
        for key in ('instances', 'unseen', 'expected', 'low', 'high'):
            assert abs(sum(row[kind][key] for row in forecast['rows']) - forecast[kind][key]) < 1e-6
    assert forecast['catalog']['instances'] == 94 and forecast['heldout']['instances'] == 75

    packet = ROOT / 'experiment/published/20261016'
    runs = [*packet.glob('context/*/receipt.json'), *packet.glob('skill/runs/*/receipt.json')]
    stages = 0
    for path in runs:
        receipt = json.loads(path.read_bytes())
        assert usage_from_trace(read_trace(path.parent / 'events.jsonl')) == receipt['usage'], str(path)
        grades = [row['official'] for row in receipt.get('stages', [])]
        if 'grade' in receipt:
            grades.append(receipt['grade'])
        for grade in grades:
            cases = grade['cases']
            assert sum(case['status'] == 'passed' for case in cases) == grade['passed'], str(path)
            assert len(cases) == grade.get('tests', grade.get('total')), str(path)
            stages += 1
    skill = json.loads((packet / 'skill/summary.json').read_bytes())
    test = [row for row in skill['rows'] if row['split'] == 'test']
    assert len(test) == len({row['task'] for row in test}) == skill['tasks'] == 20
    assert sum(row['pass'] for row in test) == skill['passed'] == 12
    for row in skill['rows']:
        path = packet / 'skill/runs' / (row['split'] + '-' + row['task']) / 'receipt.json'
        receipt = json.loads(path.read_bytes())
        assert receipt['grade']['task_pass'] == row['pass']
        assert receipt['usage']['known_tokens'] == row['tokens']
    print(json.dumps({'tracked_files': len(files), 'test_suites': len(suites), 'documentation_links': links,
        'published_artifacts': artifacts, 'metered_runs': len(runs), 'verified_grade_sets': stages,
        'repository_checks': 'passed'}))


if __name__ == '__main__':
    main()
