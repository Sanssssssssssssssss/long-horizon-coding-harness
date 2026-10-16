"""Freeze one small official three-checkpoint task and calibrate its native graders; no model calls."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_HOME = ROOT / 'experiment/research/20261011-scb-preparation-001'
SOURCE = SOURCE_HOME / 'source/circuit_eval'
FROZEN = HERE / 'context-frozen'
OUT = HERE / 'context-calibration-001'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def public_projection(source, destination):
    text = source.read_text(encoding='utf-8')
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    blocks, ranges = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
            markers = [ast.unparse(dec) for dec in node.decorator_list]
            if any('pytest.mark.functionality' in marker or 'pytest.mark.error' in marker for marker in markers):
                continue
        start = min([node.lineno] + [item.lineno for item in getattr(node, 'decorator_list', [])])
        block = ''.join(lines[start - 1:node.end_lineno])
        blocks.append(block)
        ranges.append({'name': getattr(node, 'name', type(node).__name__), 'start': start, 'end': node.end_lineno,
                       'sha256': hashlib.sha256(block.encode()).hexdigest()})
    projected = '\n\n'.join(blocks) + '\n'
    assert 'novel_circuits' not in projected and 'pytest.mark.functionality' not in projected and 'pytest.mark.error' not in projected
    destination.write_text(projected, encoding='utf-8', newline='')
    return ranges


def grade(cp):
    stage = OUT / f'checkpoint-{cp}'
    stage.mkdir(parents=True)
    actor = stage / 'reference'
    shutil.copytree(SOURCE / 'solutions' / f'checkpoint_{cp}', actor)
    before = {path.relative_to(actor).as_posix(): digest(path) for path in actor.rglob('*') if path.is_file()}
    entrypoint = '"' + Path(sys.executable).as_posix() + '" -B -X utf8 "' + (actor / 'circopt.py').as_posix() + '"'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    reports = {}
    for mode, tests in (('official', FROZEN / 'official'), ('public', FROZEN / 'public')):
        report = stage / f'{mode}.xml'
        args = [sys.executable, '-B', '-X', 'utf8', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                '-c', str(FROZEN / 'pytest.ini'), '--rootdir', str(tests), '--basetemp', str(stage / f'{mode}-tmp'),
                '--junitxml', str(report), '--entrypoint', entrypoint, '--checkpoint', f'checkpoint_{cp}']
        args += [str(tests / f'test_checkpoint_{i}.py') for i in range(1, cp + 1)]
        done = subprocess.run(args, cwd=tests, env=env, capture_output=True, text=True, encoding='utf-8', timeout=120)
        (stage / f'{mode}-stdout.txt').write_text(done.stdout, encoding='utf-8')
        (stage / f'{mode}-stderr.txt').write_text(done.stderr, encoding='utf-8')
        xml = ET.parse(report).getroot()
        cases = xml.findall('.//testcase')
        ids = sorted([(node.get('classname'), node.get('name')) for node in cases])
        reports[mode] = {'exit_code': done.returncode, 'tests': len(cases), 'failures': len(xml.findall('.//failure')),
            'errors': len(xml.findall('.//error')), 'skipped': len(xml.findall('.//skipped')), 'identities': ids,
            'command': args, 'xml_sha256': digest(report)}
        assert done.returncode == 0 and reports[mode]['failures'] == reports[mode]['errors'] == reports[mode]['skipped'] == 0
        assert len(ids) == len(set(ids))
    assert reports['official']['tests'] == (36, 99, 205)[cp - 1]
    assert reports['public']['tests'] == (9, 14, 32)[cp - 1]
    assert all(digest(actor / name) == sha for name, sha in before.items())
    save(stage / 'receipt.json', reports)
    return {'checkpoint': cp, **reports}


def main():
    assert not FROZEN.exists() and not OUT.exists()
    FROZEN.mkdir()
    OUT.mkdir()
    shutil.copytree(SOURCE / 'tests', FROZEN / 'official')
    for cp in range(4, 9):
        # Only the selected three test modules participate; unused source remains in the immutable upstream checkout.
        unused = FROZEN / 'official' / f'test_checkpoint_{cp}.py'
        if unused.exists():
            unused.unlink()
    public = FROZEN / 'public'
    public.mkdir()
    for name in ('helpers.py', 'conftest.py'):
        shutil.copyfile(SOURCE / 'tests' / name, public / name)
    projections = {}
    for cp in (1, 2, 3):
        projections[str(cp)] = public_projection(SOURCE / 'tests' / f'test_checkpoint_{cp}.py', public / f'test_checkpoint_{cp}.py')
        text = (SOURCE / f'checkpoint_{cp}.md').read_text(encoding='utf-8')
        (FROZEN / f'checkpoint_{cp}.md').write_text(text.replace('%%%ENTRYPOINT:entry_command%%%', 'python -B circopt.py'), encoding='utf-8', newline='')
    (FROZEN / 'pytest.ini').write_text('[pytest]\nmarkers =\n    functionality: original extended functionality\n    error: original error cases\n', encoding='utf-8', newline='')
    save(FROZEN / 'public-projection.json', projections)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(grade, (1, 2, 3)))
    bindings = {path.relative_to(ROOT).as_posix(): digest(path) for path in FROZEN.rglob('*') if path.is_file()}
    original = json.loads((SOURCE_HOME / 'preparation-receipt-001.json').read_text(encoding='utf-8'))
    sources = {path.relative_to(ROOT).as_posix(): digest(path) for folder in ('agents', 'benchmarks')
               for path in (ROOT / folder).rglob('*.py')}
    sources[Path(__file__).relative_to(ROOT).as_posix()] = digest(__file__)
    with zipfile.ZipFile(HERE / 'context-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
    protocol = {'task': 'SCB circuit_eval CP1..3', 'repository': original['repository'], 'commit': original['commit'],
        'checkpoint_order': [1, 2, 3], 'official_cumulative_counts': [36, 99, 205], 'public_cumulative_counts': [9, 14, 32],
        'oracle_identity_sets': {str(row['checkpoint']): row['official']['identities'] for row in results},
        'public_identity_sets': {str(row['checkpoint']): row['public']['identities'] for row in results},
        'data_files': bindings, 'source_files': sources, 'source_zip_sha256': digest(HERE / 'context-source.zip'),
        'model': 'deepseek/deepseek-v4.1-flash', 'api_base': 'https://api.commandcode.ai/provider/v1',
        'arms': {'baseline': 'benchmarks.context_folding._FoldingLoop existing textual compaction',
                 'candidate': 'benchmarks.folding_checkpoint.EvidenceFoldingLoop typed receipt selection'},
        'allocation_per_arm': 350000, 'planned_max_total_tokens': 700000, 'checkpoint_budget_shares': [110000, 120000, 120000],
        'common_parameters': {'skills_enabled': False, 'model_retry_limit': 0, 'reasoning_effort': 'low', 'max_output_tokens': 8192,
            'max_input_tokens': 65536, 'fold_every': 4, 'keep_recent_tokens': 2048, 'summary_output_tokens': 4096},
        'common_runtime': 'Same Solver system instructions, tool definitions, permissions, runtime state/gates/observation getter and ResearchBudget ledger. Only fold-summary method differs.',
        'progression': 'One persistent model/workspace per arm. Deliver only current public spec plus retained prior specs/core tests. '
                       'Keep conversation/observations and prior code across stage changes; never reset a failing artifact or restore a reference. '
                       'At each checkpoint snapshot the actual executable and grade unchanged official tests CP1..CP_N outside model context.',
        'feedback': 'Public core assertions only. Official hidden/error/functionality results are never sent to the model; grade once per snapshot.',
        'curves': ['checkpoint index vs cumulative official pass fraction', 'cumulative solve responses vs completed official checkpoints',
                   'cumulative provider tokens vs completed checkpoints', 'prior public and official behavior retained after later code edits'],
        'trace': ['typed runtime evidence/execution_id/generation', 'model purpose/actual usage', 'fold trigger/selected refs/fallback',
                  'state and assertion drift', 'natural finish vs budget exit', 'historical observations read', 'snapshot code hashes'],
        'split': {'TRAIN': ['dialogue procedure traces', 'accounting selection public traces now TRAIN'],
                  'DEV': ['log_query CP1 already used with observed private grade; excluded from this comparator'],
                  'heldout_candidate': ['circuit_eval CP1..3; prior activity was model-free reference calibration'],
                  'check': 'rg circuit_eval across experiment/research trace.jsonl returned no recorded model trace; final source review must confirm no overlapping task in external runs'},
        'limitations': 'One task/three checkpoints and one run per arm; no statistical generalization or official Docker score. '
                       'Natural solve length arises from actual implementation/verification, with no token padding or forced redundant loops.',
        'launch_state': 'DATA_AND_ORACLE_READY; paired API runner still requires explicit source binding and independent review before model calls'}
    save(HERE / 'context-protocol.json', protocol)
    save(HERE / 'context-readiness.json', {'API_calls': 0, 'provider_tokens': 0, 'environment': {'interpreter': sys.executable,
        'python_version': sys.version}, 'checkpoints': [{key: row[key] for key in ('checkpoint', 'official', 'public')} for row in results],
        'source_and_data_bound': True, 'native_pytest_invocations': 6, 'scope': 'Reference/public projection calibration, not model completion or benefit'})
    print(json.dumps({'API_calls': 0, 'reference_official': [row['official']['tests'] for row in results],
                      'reference_public': [row['public']['tests'] for row in results], 'ready': str(HERE)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
