"""Paired persistent SCB sequence. Preparation/spy are offline; root owns paid launches."""
from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prepare_context as data
from agents.agent import _BudgetExceeded
from agents.repair_state import TaskEnvelope
from agents.telemetry import TraceRecorder, read_trace
from agents.tools import get_solver_tool_definitions
from benchmarks.context_folding import _FoldingLoop
from benchmarks.folding_checkpoint import EvidenceFoldingLoop
from benchmarks.harbor_agent import _isolated_home
from benchmarks.research_budget import ResearchBudget, save_json, usage_from_trace

OUT = HERE / 'context-paired-001'
PARAMS = dict(allocation=350000,
              phase_solve_responses=20, max_model_calls=80, max_tool_calls=240,
              summary_calls=18, max_output_tokens=8192, max_input_tokens=65536,
              fold_every=4, keep_recent_tokens=2048, summary_output_tokens=4096)
SYSTEM = '''You implement executable code from a public contract in one isolated repository.
Work on the current checkpoint, preserving earlier behavior unless the new specification supersedes it.
Register the supplied acceptance cases together with update_work_state before editing or Shell.
Use runtime observations for facts; plans, printed values and summaries are not test evidence.
Implement, then run the supplied public pytest command. Bind only acceptance IDs actually checked.
Validate new self-authored assertions against the public contract in standalone diagnostic_*.py first:
run those files with diagnostic=true and no acceptance_ids; use test_actor.py for validated assertions.
Delete a finished draft with exactly del diagnostic_NAME.py (one root file, absent from Git baseline/index).
Keep supplied tests/config/TASK.md unchanged. Do not make intentionally wrong edits to live code.
Use ordinary tools inside this repository; no network, external/reference/grader files, or Skills.
When current acceptance has recognized green checks and the runtime is ready, return the final response.
Finish with the actual outcome and any remaining public-contract limitations.'''
CASES = {
 1: [('C1-cli', '--json — Output exactly one JSON object to stdout',
       '--json` — Output exactly one JSON object to stdout (no other stdout text).'),
     ('C1-check', 'Parse and validate scalar circuits and report their interface',
       "Parses and validates a `.circ` file. On success, reports the circuit's interface."),
     ('C1-errors', 'Preserve scalar grammar and graph validation errors',
       'The assignment graph must be acyclic (no circular dependencies).')],
 2: [('C2-eval', 'Evaluate given scalar inputs with Boolean operators',
       'Evaluates the circuit with given inputs. Inputs must be 0 or 1.'),
     ('C2-flags', 'Implement missing/extra/default input policies',
       '[--set name=value ...] [--default 0|1] [--allow-extra] [--json]'),
     ('C2-output', 'Return sorted text and the declared JSON output envelope',
       'Text mode: `name=value` per line, sorted by name.')],
 3: [('C3-language', 'Implement vector declarations, widths, slices and literals',
       'Vector signals in `.circ`, width checking, slicing/indexing/concatenation, richer literal formats'),
     ('C3-operators', 'Implement vector Boolean operators, MUX, reductions and equality',
       'All operators use call syntax `OP(...)` with case-insensitive names.'),
     ('C3-io', 'Validate input widths and format requested output radix',
       'Runtime literal width must exactly equal declared input width.')],
}


def tree(folder):
    return {p.relative_to(folder).as_posix(): data.digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and not {'.git', '__pycache__', 'experiment', '.pytest_cache'}.intersection(p.relative_to(folder).parts)}


def env():
    value = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                 PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', SANS_AUTO_SKILL_EVOLUTION='0')
    for name in ('PYTEST_ADDOPTS', 'PYTHONPATH', 'PYTEST_PLUGINS'):
        value.pop(name, None)
    value['PATH'] = str(Path(sys.executable).parent) + os.pathsep + value.get('PATH', '')
    return value


def public_command(cp):
    return 'python -B -m pytest ' + ' '.join(f'test_checkpoint_{i}.py' for i in range(1, cp+1)) + ' test_actor.py -q -c pytest.ini'


def publish(work, cp):
    shutil.copyfile(data.FROZEN / f'checkpoint_{cp}.md', work / 'TASK.md')
    shutil.copyfile(data.FROZEN / 'public' / f'test_checkpoint_{cp}.py', work / f'test_checkpoint_{cp}.py')
    (work / 'pytest.ini').write_text((data.FROZEN / 'pytest.ini').read_text(encoding='utf-8') +
        f'addopts = -p no:cacheprovider --entrypoint "python -B circopt.py" --checkpoint checkpoint_{cp}\n', encoding='utf-8')
    for args in (['add', '.', ':(glob,exclude)**/diagnostic_*.py'], ['-c', 'user.name=Research', '-c', 'user.email=research@example.invalid',
                              '-c', 'commit.gpgsign=false', 'commit', '-qm', f'publish checkpoint {cp}', '--allow-empty']):
        subprocess.run(['git', *args], cwd=work, capture_output=True, check=True)
    return {name: data.digest(work / name) for name in ['TASK.md', 'pytest.ini', 'helpers.py', 'conftest.py', '.gitignore',
            *[f'test_checkpoint_{i}.py' for i in range(1, cp+1)]]}


def prompt(cp):
    spec = (data.FROZEN / f'checkpoint_{cp}.md').read_text(encoding='utf-8')
    cases = [dict(id=i, behavior=b, source_quote=q, source_kind='issue') for i,b,q in CASES[cp]]
    assert all(case['source_quote'] in spec for case in cases)
    return ('Current checkpoint: ' + str(cp) + '. Python/pytest are prepared; cwd is the isolated repository.\n'
            'Public command: ' + public_command(cp) + '\n'
            'Temporary drafts use root diagnostic_NAME.py. Remove finished drafts with del diagnostic_NAME.py.\n'
            'Register these cases in one update_work_state call; hypotheses may be empty:\n' +
            json.dumps(cases, ensure_ascii=False) + '\n\n' + spec)


def scoped(name, inp, work, cp, protected):
    result = dict(inp)
    def path(value):
        lexical = work / str(value)
        assert not any(part.is_symlink() or part.is_junction() for part in [lexical, *lexical.parents])
        p = lexical.resolve()
        rel = p.relative_to(work.resolve()).as_posix()
        assert not {'.git', '__pycache__', 'experiment'}.intersection(Path(rel).parts)
        assert not any(part.is_symlink() for part in [p, *p.parents])
        return rel
    if name in {'read_file', 'write_file', 'edit_file'}:
        result['file_path'] = path(inp.get('file_path', ''))
        if name != 'read_file':
            assert result['file_path'] not in protected and result['file_path'] not in {'helpers.py', 'conftest.py', '.gitignore'}
            assert Path(result['file_path']).suffix in {'.py', '.circ', '.txt', '.json'}
            if Path(result['file_path']).name.startswith('diagnostic_'):
                assert re.fullmatch(r'diagnostic_[A-Za-z0-9_-]+\.py', result['file_path'])
    elif name == 'grep_search':
        result['path'] = path(inp.get('path', '.')) or '.'
        assert not any(p.is_symlink() or p.is_junction() for p in (work / result['path']).rglob('*'))
    elif name == 'list_files':
        pattern = str(inp.get('pattern', '**/*')).replace('\\', '/')
        assert not Path(pattern).is_absolute() and ':' not in pattern and '..' not in pattern.split('/')
    elif name == 'run_shell':
        command = str(inp.get('command', ''))
        assert not any(char in command for char in '&|<>\r\n%!`')
        tokens = shlex.split(command)
        if command == public_command(cp):
            pass
        elif tokens[:4] == ['python', '-B', '-m', 'pytest']:
            assert tokens[-3:] == ['-q', '-c', 'pytest.ini']
            files = tokens[4:-3]
            assert files and all(re.fullmatch(r'diagnostic_[A-Za-z0-9_-]+\.py', item) and path(item)==item for item in files)
            assert not inp.get('acceptance_ids') and inp.get('diagnostic') is True
        elif tokens[:1] == ['del']:
            assert len(tokens)==2 and re.fullmatch(r'diagnostic_[A-Za-z0-9_-]+\.py', tokens[1])
            filename=tokens[1]
            assert command=='del '+filename and path(filename)==filename and (work/filename).is_file()
            assert filename not in protected
            baseline=subprocess.run(['git','ls-tree','-r','--name-only','HEAD','--',filename],cwd=work,capture_output=True)
            indexed=subprocess.run(['git','ls-files','--error-unmatch','--',filename],cwd=work,capture_output=True)
            assert baseline.returncode==0 and not baseline.stdout.strip() and indexed.returncode==1
        elif tokens[:3] == ['python', '-B', 'circopt.py']:
            assert len(tokens) == 3 or tokens[3] in {'check', 'eval', '--help', '--version', '--json'}
            for item in tokens[4:]:
                if item.endswith('.circ') or '/' in item or '\\' in item or (len(item)>1 and item[1]==':'):
                    path(item)
        elif tokens[:2] == ['git', 'status']:
            assert tokens == ['git', 'status', '--short']
        elif tokens[:2] == ['git', 'diff']:
            assert all(item in {'--', '--check', '--stat', '--name-only'} for item in tokens[2:])
        else:
            raise ValueError('Shell command outside documented public commands')
    elif name not in {'update_work_state', 'update_plan', 'read_observation', 'compact_context'}:
        raise ValueError('Tool outside common experiment scope')
    return result


class CommonScope:
    def __init__(self, *, workspace, **kwargs):
        self.workspace, self.checkpoint, self.protected = workspace, 1, {}
        self.summary_attempts = 0
        super().__init__(**kwargs)

    def _compaction_token_budgets(self):
        summary, recent, overhead = super()._compaction_token_budgets()
        return min(summary, PARAMS['summary_output_tokens']), min(recent, PARAMS['keep_recent_tokens']), overhead

    async def _run_compaction_query(self, *args, **kwargs):
        if self.summary_attempts >= PARAMS['summary_calls']:
            raise _BudgetExceeded('Frozen summary allowance exhausted')
        self.summary_attempts += 1
        return await super()._run_compaction_query(*args, **kwargs)

    async def _execute_tool_call(self, name, inp, *, runtime_owned=False, **kwargs):
        if not runtime_owned:
            try:
                inp = scoped(name, inp, self.workspace, self.checkpoint, self.protected)
            except (AssertionError, ValueError, TypeError) as error:
                self.trace.event('tool.scope_denied', component='tool', purpose=name, status='error',
                                 attributes={'checkpoint': self.checkpoint, 'reason': type(error).__name__}, payload={'input': inp})
                return 'Error: work inside this repository; supplied tests/config/TASK.md are protected. Use public pytest, diagnostic root-file pytest, del diagnostic_NAME.py for a new untracked draft, circopt CLI or read-only Git commands.'
        return await super()._execute_tool_call(name, inp, runtime_owned=runtime_owned, **kwargs)


class Baseline(CommonScope, _FoldingLoop):
    pass


class Candidate(CommonScope, EvidenceFoldingLoop):
    pass


def grade(snapshot, cp, destination, mode):
    destination.mkdir(parents=True, exist_ok=False)
    report = destination / 'pytest.xml'
    tests = data.FROZEN / mode
    entry = f'"{Path(sys.executable).as_posix()}" -B -X utf8 "{(snapshot / "circopt.py").as_posix()}"'
    args = [sys.executable, '-B', '-X', 'utf8', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-c', str(data.FROZEN / 'pytest.ini'),
            '--rootdir', str(tests), '--basetemp', str(destination / 'tmp'), '--junitxml', str(report),
            '--entrypoint', entry, '--checkpoint', f'checkpoint_{cp}',
            *[str(tests / f'test_checkpoint_{i}.py') for i in range(1,cp+1)]]
    done = subprocess.run(args, cwd=tests, env=env(), capture_output=True, text=True, encoding='utf-8', timeout=180)
    (destination / 'stdout.txt').write_text(done.stdout, encoding='utf-8')
    (destination / 'stderr.txt').write_text(done.stderr, encoding='utf-8')
    xml = ET.parse(report).getroot()
    cases = xml.findall('.//testcase')
    ids = sorted([(node.get('classname'), node.get('name')) for node in cases])
    expected = json.loads((HERE / 'context-protocol.json').read_text(encoding='utf-8'))[('oracle' if mode=='official' else 'public') + '_identity_sets'][str(cp)]
    assert ids == [tuple(row) for row in expected], 'Frozen grader collection changed'
    result = dict(exit_code=done.returncode, tests=len(cases), failures=len(xml.findall('.//failure')),
                  errors=len(xml.findall('.//error')), skipped=len(xml.findall('.//skipped')),
                  passed_ids=[(n.get('classname'), n.get('name')) for n in cases if not any(n.find(tag) is not None for tag in ('failure','error','skipped'))],
                  xml_sha256=data.digest(report), command=args)
    result['passed'] = len(result['passed_ids'])
    save_json(destination / 'receipt.json', result)
    return result


def prepare():
    OUT.mkdir(exist_ok=False)
    for arm in ('baseline', 'candidate'):
        run = OUT / arm
        work = run / 'workspace'
        work.mkdir(parents=True)
        (work / 'experiment/cache/pytest').mkdir(parents=True)
        (work / 'circopt.py').write_text('# Implement the public circuit evaluator here.\n', encoding='utf-8')
        (work / 'test_actor.py').write_text('# Validated public-contract assertions may be added here.\n', encoding='utf-8')
        (work / '.gitignore').write_text('__pycache__/\n.pytest_cache/\nexperiment/\n', encoding='utf-8')
        for name in ('helpers.py', 'conftest.py'):
            shutil.copyfile(data.FROZEN / 'public' / name, work / name)
        for args in (['init','-q'],['config','core.autocrlf','false']):
            subprocess.run(['git',*args], cwd=work, capture_output=True, check=True)
        protected = publish(work, 1)
        save_json(run / 'prepared.json', {'API_calls': 0, 'workspace_files': tree(work), 'protected': protected})
    assert tree(OUT / 'baseline/workspace') == tree(OUT / 'candidate/workspace')
    for cp in (1,2,3):
        prompt(cp)
    save_json(OUT / 'preparation.json', {'API_calls':0, 'provider_tokens':0, 'identical_inputs': True,
                                      'parameters':PARAMS, 'source_sealed':False})
    print(json.dumps({'prepared':str(OUT), 'API_calls':0}))


def seal():
    assert not any((OUT / arm / 'trace.jsonl').exists() for arm in ('baseline','candidate'))
    protocol = json.loads((HERE / 'context-protocol.json').read_text(encoding='utf-8'))
    protocol['data_files'][(HERE / 'context-protocol.json').relative_to(ROOT).as_posix()] = data.digest(HERE / 'context-protocol.json')
    files = {p.relative_to(ROOT).as_posix():data.digest(p) for folder in ('agents','benchmarks') for p in (ROOT/folder).rglob('*.py')}
    for p in (Path(__file__), Path(data.__file__)):
        files[p.relative_to(ROOT).as_posix()] = data.digest(p)
    files[(HERE/'context_smoke.py').relative_to(ROOT).as_posix()] = data.digest(HERE/'context_smoke.py')
    files[(HERE/'context_diagnostic_smoke.py').relative_to(ROOT).as_posix()] = data.digest(HERE/'context_diagnostic_smoke.py')
    with zipfile.ZipFile(OUT / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT/name,name)
    protocol.pop('checkpoint_budget_shares', None)
    protocol.update(source_files=files, source_zip_sha256=data.digest(OUT/'source.zip'), parameters=PARAMS,
                    common_runtime='One shared small caller prompt, common tools/guard/state/gates/historical reader. Skills disabled. Only native fold-summary strategy differs.',
                    system_prompt=SYSTEM, acceptance_cases=CASES, launch_state='PREPARED; independent launch review required',
                    phase_budget_policy='One shared350k ledger allocation per complete three-checkpoint episode, with no per-checkpoint token cap. Requests retain production byte reservation. Next public checkpoint follows a partial decision-cap return while episode allowance remains; exhaustion or unmetered usage stops the arm, and later checkpoints are reported not attempted.')
    save_json(OUT/'protocol.json', protocol)
    print(json.dumps({'sealed':str(OUT/'protocol.json'),'API_calls':0}))


def calibrate_runner():
    reference=data.OUT/'checkpoint-3/reference'
    before=tree(reference)
    rows={mode:grade(reference,3,HERE/'context-runner-grader-calibration-001'/mode,mode) for mode in ('public','official')}
    assert tree(reference)==before
    assert rows['public']['passed']==32 and rows['official']['passed']==205
    assert all(row['skipped']==row['errors']==row['failures']==0 for row in rows.values())
    save_json(HERE/'context-runner-grader-calibration-001/receipt.json',dict(API_calls=0,provider_tokens=0,
        scope='Native final runner grading route on immutable CP3 reference; not a model score',grades=rows))
    print(json.dumps({'API_calls':0,'public':32,'official':205,'reference_unchanged':True}))


async def run(arm):
    from dotenv import dotenv_values
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    run = OUT/arm
    work = run/'workspace'
    initial = json.loads((run/'prepared.json').read_text(encoding='utf-8'))
    assert tree(work)==initial['workspace_files'] and not (run/'trace.jsonl').exists()
    assert all(data.digest(ROOT/name)==sha for name,sha in p['source_files'].items())
    assert all(data.digest(ROOT/name)==sha for name,sha in p['data_files'].items())
    assert data.digest(OUT/'source.zip')==p['source_zip_sha256'] and p['parameters']==PARAMS
    settings=dotenv_values(ROOT/'.env')
    assert str(settings.get('OPENAI_BASE_URL','')).rstrip('/')==p['api_base']
    key=os.environ.get('SANS_BENCHMARK_API_KEY') or settings.get('OPENAI_API_KEY')
    assert key
    trace=TraceRecorder(run,run_id='circuit-'+arm,task_id='circuit-eval-long',attempt_id='001')
    rows=[]
    with ResearchBudget(run,trace,allocation=PARAMS['allocation']) as budget, _isolated_home(run):
        for name,value in list(os.environ.items()):
            if key in value:
                os.environ.pop(name)
        configured=env()
        os.environ.clear()
        os.environ.update(configured)
        os.chdir(work)
        params=dict(workspace=work,token_budget=budget.token_budget,before_model_call=budget.before_call,
            fold_every_decisions=PARAMS['fold_every'],auto_compact=False,max_decisions=PARAMS['phase_solve_responses'],
            model=p['model'],api_base=p['api_base'],api_key=key,thinking=True,openai_extra_body={'reasoning_effort':'low'},
            model_retry_limit=0,custom_system_prompt=SYSTEM,custom_tools=get_solver_tool_definitions('B0'),
            skills_enabled=False,is_sub_agent=True,permission_mode='bypassPermissions',trace=trace,
            max_model_calls=PARAMS['max_model_calls'],max_tool_calls=PARAMS['max_tool_calls'],
            max_output_tokens=PARAMS['max_output_tokens'],max_input_tokens_per_call=PARAMS['max_input_tokens'],model_call_timeout_seconds=180)
        if arm=='candidate':
            params['checkpoint_summary_tokens']=PARAMS['summary_output_tokens']
        agent=({'baseline':Baseline,'candidate':Candidate}[arm])(**params)
        total_limit=budget.token_budget['limit']
        try:
            for cp in (1,2,3):
                if budget.token_budget.get('blocked'):
                    break
                before_calls=agent._operation_counts['model_calls']
                before_decisions=agent.decision_clock['completed']
                agent.checkpoint=cp
                agent.protected=initial['protected'] if cp==1 else publish(work,cp)
                agent._repair_task=TaskEnvelope(f'circuit-checkpoint-{cp}',(data.FROZEN/f'checkpoint_{cp}.md').read_text(encoding='utf-8'),str(work),requires_patch=False,requires_repair_witness=False)
                agent.max_decisions=agent.decision_clock['completed']+PARAMS['phase_solve_responses']
                start_seq=len(read_trace(trace.trace_path))
                trace.event('sequence.checkpoint.start',component='benchmark',purpose='phase',attributes={'checkpoint':cp,'decision_clock':dict(agent.decision_clock),'token_limit':budget.token_budget['limit']})
                error=None
                result={}
                with (run/'console.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                    try:
                        result=await agent.run_once(prompt(cp))
                    except Exception as exc:
                        error={'type':type(exc).__name__,'message':str(exc)[:1000]}
                events=read_trace(trace.trace_path)
                natural=any(e['event_type']=='hook.stop' and e['status']=='ok' for e in events[start_seq:]) and agent.termination_reason is None and agent.finish_reason=='stop'
                unchanged=all(data.digest(work/name)==sha for name,sha in agent.protected.items())
                assert unchanged, 'Protected public input changed'
                if agent.decision_clock['completed']==before_decisions and (agent.termination_reason or {}).get('kind')=='token_budget':
                    save_json(run/f'checkpoint-{cp}-not-attempted.json',dict(checkpoint=cp,termination_reason=agent.termination_reason,
                              usage=usage_from_trace(events),model_calls_delta=agent._operation_counts['model_calls']-before_calls,
                              solve_responses=0,reason='Episode allowance could not admit a solve response'))
                    break
                snapshot=run/'snapshots'/f'checkpoint-{cp}'
                snapshot.mkdir(parents=True)
                for name in tree(work):
                    dest=snapshot/name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(work/name,dest)
                snapshot_hashes=tree(snapshot)
                public=grade(snapshot,cp,run/'grades'/f'checkpoint-{cp}'/'public','public')
                official=grade(snapshot,cp,run/'grades'/f'checkpoint-{cp}'/'official','official')
                assert tree(snapshot)==snapshot_hashes
                row=dict(checkpoint=cp,natural_finish=natural,completion_decision=agent._repair_state.completion_decision().complete,
                    termination_reason=agent.termination_reason,error=error,result=result,repair_state=agent._repair_state.snapshot(),
                    solve_responses=agent.decision_clock['completed'],summary_attempts=agent.summary_attempts,fold_count=agent._fold_count,
                    usage=usage_from_trace(events),snapshot_files=snapshot_hashes,protected=agent.protected,public=public,official=official)
                previous=set(tuple(i) for i in rows[-1]['official']['passed_ids']) if rows else set()
                row['prior_passed_cases_lost']=sorted(previous-set(tuple(i) for i in official['passed_ids']))
                rows.append(row)
                save_json(run/'progress.json',rows)
                trace.event('sequence.checkpoint.end',component='benchmark',purpose='phase',attributes={'checkpoint':cp,'natural_finish':natural,'solve_responses':agent.decision_clock['completed']})
                print(json.dumps({'arm':arm,'checkpoint':cp,'official':f"{official['passed']}/{official['tests']}",'natural_finish':natural,'tokens':row['usage']['known_tokens']}),flush=True)
                if error or usage_from_trace(events)['unknown_usage_requests'] or (agent.termination_reason or {}).get('kind')=='token_budget':
                    break
        finally:
            budget.token_budget['limit']=total_limit
            await agent.shutdown()
            receipt=dict(arm=arm,checkpoints=rows,usage=usage_from_trace(read_trace(trace.trace_path)),
                source_unchanged=all(data.digest(ROOT/name)==sha for name,sha in p['source_files'].items()),
                data_unchanged=all(data.digest(ROOT/name)==sha for name,sha in p['data_files'].items()),
                finished_official_checkpoints=sum(row['official']['passed']==row['official']['tests'] and not row['official']['skipped'] for row in rows),
                not_attempted_checkpoints=[cp for cp in (1,2,3) if cp not in {row['checkpoint'] for row in rows}],
                scope='One sequence/one arm. No hidden-score feedback or retries. Official completion and natural return recorded separately.')
            save_json(run/'receipt.json',receipt)
            curve=[{key:row[key] for key in ('checkpoint','solve_responses','fold_count','natural_finish','completion_decision','prior_passed_cases_lost')} |
                   dict(tokens=row['usage']['known_tokens'],official_passed=row['official']['passed'],official_total=row['official']['tests']) for row in rows]
            save_json(run/'curve.json',curve)


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        prepare()
    elif sys.argv[1]=='seal':
        seal()
    elif sys.argv[1]=='calibrate':
        calibrate_runner()
    elif sys.argv[1]=='run' and sys.argv[2] in {'baseline','candidate'}:
        asyncio.run(run(sys.argv[2]))
    else:
        raise SystemExit('Expected prepare | seal | run baseline|candidate')
