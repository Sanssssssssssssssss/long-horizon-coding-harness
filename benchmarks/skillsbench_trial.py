"""Frozen SkillsBench dialogue trials with visible banks and isolated official grading."""
from __future__ import annotations

import argparse
import asyncio
from copy import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
import tarfile
import zipfile

from agents.agent import _BudgetExceeded
from agents.prompt import ARTIFACT_SOLVER_SYSTEM_PROMPT, SOLVER_SYSTEM_PROMPT
from agents.repair_state import TaskEnvelope
from agents.telemetry import TraceRecorder, read_trace, summarize_trace
from agents.tools import get_solver_tool_definitions
from benchmarks import checkpoints, research_trial as trial
from benchmarks.context_folding import _FoldingLoop
from benchmarks.harbor_agent import EnvironmentTools, _FULL_DIFF_CHECK, _isolated_home
from benchmarks.research_budget import ResearchBudget, usage_from_trace


ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / 'benchmarks/skillsbench_prepare.py'
BANK_ROOT = '/app/.sans/skills'
MODEL, BASE_URL = trial.MODEL, trial.BASE_URL
LIMITS = {'solve_calls': 12, 'summary_calls': 4, 'total_model_calls': 16,
          'fold_every_decisions': 6, 'case_tokens': 120000, 'max_output_tokens': 8192,
          'checkpoint_summary_tokens': 2048, 'max_input_tokens': 32768,
          'max_tool_calls': 80, 'request_timeout_seconds': 120}
BUDGET_FIELDS = {'case_tokens', 'solve_calls', 'summary_calls'}
WORKFLOW_PROMPTS = {'repair': SOLVER_SYSTEM_PROMPT, 'build': ARTIFACT_SOLVER_SYSTEM_PROMPT}
TRAIN_STRATEGIES = {
    'grammar_first': 'Derive a grammar from the public task and a few input lines. Build a single-pass '
                     'lexer or state machine. Export JSON and DOT from the same graph representation. '
                     'Test small constructed inputs before processing the provided script.',
    'contract_first': 'Define the public output contract and boundary assertions first. Build a minimal '
                      'parser skeleton. Iterate control flow and sequence consistency, then generate '
                      'outputs from the provided script.',
}
ARMS = {'train': tuple(TRAIN_STRATEGIES), 'canonical': ('no_skill', 'curated', 'learned')}
WorkspaceBase = trial.WindowsWSLWorkspace if os.name == 'nt' else checkpoints.DockerWorkspace


def preparer():
    spec = importlib.util.spec_from_file_location('skillsbench_prepare', PREPARER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return checkpoints.digest(path)


def save(path, value):
    trial.save(path, value)


def artifact(path):
    path = Path(path).resolve()
    path.relative_to(ROOT / 'experiment')
    return path


def tree_hashes(folder):
    files = {}
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():
            raise ValueError('Frozen bank contains a symbolic link')
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = digest(path)
    return files


def validate_limits(limits):
    if not isinstance(limits, dict) or set(limits) != set(LIMITS):
        raise ValueError('Frozen protocol limit fields changed')
    for name, default in LIMITS.items():
        value = limits[name]
        minimum = 0 if name in {'summary_calls', 'fold_every_decisions'} else 1
        if type(value) is not int or value < minimum:
            raise ValueError('Protocol limits must be positive integers (summary_calls may be zero): ' + name)
        expected = (0 if limits['summary_calls'] == 0 else default) if name == 'fold_every_decisions' else default
        if name not in BUDGET_FIELDS | {'total_model_calls'} and value != expected:
            raise ValueError('Only case_tokens, solve_calls and summary_calls may change: ' + name)
    if limits['total_model_calls'] != limits['solve_calls'] + limits['summary_calls']:
        raise ValueError('Total model calls must equal solve plus summary calls')
    return dict(limits)


def workflow_profile(protocol):
    workflow = protocol.get('workflow', 'repair')
    if not isinstance(workflow, str) or workflow not in WORKFLOW_PROMPTS:
        raise ValueError('Unknown frozen workflow')
    prompt = WORKFLOW_PROMPTS[workflow]
    prompt_sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if ('workflow' in protocol or 'solver_prompt_sha256' in protocol) and (
            protocol.get('solver_prompt_sha256') != prompt_sha or 'workflow' not in protocol):
        raise ValueError('Frozen workflow prompt changed')
    return workflow, prompt, prompt_sha


def freeze(ready_path, output, *, case_tokens=None, solve_calls=None, summary_calls=None, workflow='repair'):
    if not isinstance(workflow, str) or workflow not in WORKFLOW_PROMPTS:
        raise ValueError('Unknown frozen workflow')
    prompt_sha = hashlib.sha256(WORKFLOW_PROMPTS[workflow].encode('utf-8')).hexdigest()
    limits = dict(LIMITS)
    for name, value in (('case_tokens', case_tokens), ('solve_calls', solve_calls), ('summary_calls', summary_calls)):
        if value is not None:
            limits[name] = value
    limits['total_model_calls'] = limits['solve_calls'] + limits['summary_calls']
    limits['fold_every_decisions'] = 0 if limits['summary_calls'] == 0 else LIMITS['fold_every_decisions']
    limits = validate_limits(limits)
    ready_path, output = artifact(ready_path), artifact(output)
    prepare = preparer()
    ready = prepare.validate_ready(ready_path)
    tasks, _, _ = prepare.official_inputs()
    output.mkdir(parents=True, exist_ok=False)
    sources = {path.relative_to(ROOT).as_posix(): digest(path) for directory in ('agents', 'benchmarks')
               for path in sorted((ROOT / directory).glob('*.py'))}
    with zipfile.ZipFile(output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
    cases = {}
    for split, task in tasks.items():
        instruction = output / (split + '-instruction.md')
        shutil.copyfile(task / 'instruction.md', instruction)
        cases[split] = {'image': ready['image_ids'][split], 'task_root': task.relative_to(ROOT).as_posix(),
                        'instruction': instruction.relative_to(ROOT).as_posix(),
                        'instruction_sha256': digest(instruction),
                        'input_sha256': ready['images'][split]['input_sha256']}
    protocol = {'version': 1, 'model': MODEL, 'api_base': BASE_URL, 'limits': limits,
                'workflow': workflow, 'solver_prompt_sha256': prompt_sha,
                'auto_compact': limits['summary_calls'] > 0,
                'arms': {key: list(value) for key, value in ARMS.items()}, 'train_strategies': TRAIN_STRATEGIES,
                'ready_receipt': ready_path.relative_to(ROOT).as_posix(), 'ready_sha256': digest(ready_path),
                'source_files': sources, 'source_archive': (output / 'source.zip').relative_to(ROOT).as_posix(),
                'source_archive_sha256': digest(output / 'source.zip'), 'cases': cases,
                'solver_root': '/app', 'bank_root': BANK_ROOT, 'hidden_feedback_to_solver': False,
                'production_skill_learning': False, 'folding': 'disabled' if limits['summary_calls'] == 0 else 'plain_periodic_6',
                'scope': 'Two author-train strategies and three canonical Skill arms; single-family pilot, not full benchmark.'}
    save(output / 'protocol.json', protocol)
    return output / 'protocol.json'


def validate_protocol(path, split, arm):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_limits(protocol.get('limits'))
    if protocol.get('auto_compact', True) is not (protocol['limits']['summary_calls'] > 0):
        raise ValueError('Frozen automatic folding policy changed')
    workflow_profile(protocol)
    if (protocol.get('version') != 1 or protocol.get('model') != MODEL or protocol.get('api_base') != BASE_URL
            or protocol.get('train_strategies') != TRAIN_STRATEGIES
            or protocol.get('bank_root') != BANK_ROOT or protocol.get('solver_root') != '/app'
            or protocol.get('hidden_feedback_to_solver') is not False
            or protocol.get('production_skill_learning') is not False or split not in ARMS or arm not in ARMS[split]):
        raise ValueError('Frozen protocol or selected arm changed')
    for name, sha in protocol['source_files'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    if digest(ROOT / protocol['source_archive']) != protocol['source_archive_sha256']:
        raise ValueError('Frozen source archive changed')
    ready_path = ROOT / protocol['ready_receipt']
    if digest(ready_path) != protocol['ready_sha256']:
        raise ValueError('Preparation receipt changed')
    ready = preparer().validate_ready(ready_path)
    case = protocol['cases'][split]
    if (case['image'] != ready['image_ids'][split]
            or case['input_sha256'] != ready['images'][split]['input_sha256']
            or digest(ROOT / case['instruction']) != case['instruction_sha256']
            or case['instruction_sha256'] != digest(ROOT / case['task_root'] / 'instruction.md')):
        raise ValueError('Prepared task identity changed')
    return protocol, ready, case


def stage_bank(case, split, arm, output, bank=None, hint_file=None):
    if split == 'train' and bank is not None:
        raise ValueError('Training uses an empty bank and cannot read canonical or human Skills')
    if arm in {'no_skill', 'curated'} and hint_file is not None:
        raise ValueError('Canonical NoSkill and Curated do not accept extra strategy hints')
    if arm == 'no_skill' and bank is not None:
        raise ValueError('NoSkill requires an empty bank')
    if arm == 'curated':
        if bank is not None:
            raise ValueError('Curated uses the original official bank')
        source = ROOT / case['task_root'] / 'environment/skills'
    elif arm == 'learned':
        if bank is None:
            raise ValueError('Learned requires a frozen research bank folder')
        source = artifact(bank)
        if not any(source.is_relative_to(ROOT / 'experiment' / name) for name in ('research', 'runs')):
            raise ValueError('Learned bank must be a research artifact, not an official task cache')
    else:
        source = None
    target = output / 'bank'
    target.mkdir()
    if source is not None:
        tree_hashes(source)
        if (source / 'SKILL.md').is_file():
            shutil.copytree(source, target / source.name)
        else:
            shutil.copytree(source, target, dirs_exist_ok=True)
        if not list(target.rglob('SKILL.md')):
            raise ValueError('A nonempty Skill arm requires SKILL.md in its real bank')
    hint = TRAIN_STRATEGIES.get(arm, '')
    hint_ref = None
    if hint_file is not None:
        hint_file = artifact(hint_file)
        if hint_file.is_relative_to(ROOT / 'experiment/cache'):
            raise ValueError('A research hint cannot come from official cached tasks, gold, tests or human Skills')
        hint_ref = {'source': hint_file.relative_to(ROOT).as_posix(), 'sha256': digest(hint_file)}
        extra = hint_file.read_text(encoding='utf-8')
        (output / 'hint.md').write_text(extra, encoding='utf-8')
        hint = hint + '\n\n' + extra if hint else extra
    manifest = {'arm': arm, 'split': split, 'container_path': BANK_ROOT, 'read_only': True,
                'files': tree_hashes(target), 'hint': hint_ref, 'strategy_text': hint,
                'source': source.relative_to(ROOT).as_posix() if source else None}
    save(output / 'bank-manifest.json', manifest)
    return target, manifest


def mount(source, target, *, readonly=True):
    source = trial.linux_path(source) if os.name == 'nt' else str(Path(source).resolve())
    return ['--mount', f'type=bind,source={source},target={target}' + (',readonly' if readonly else '')]


class BankWorkspace(WorkspaceBase):
    def __init__(self, image, bank, *, extra_mounts=()):
        super().__init__(image, root='/app', memory='4g', cpus=2)
        self.bank_mounts = [*mount(bank, BANK_ROOT), *extra_mounts]

    def __enter__(self):
        checkpoints.docker('run', '-d', '--name', self.name, '--network', 'none', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '128', '--memory', self.memory,
            '--cpus', str(self.cpus), '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--entrypoint', 'sleep', *self.bank_mounts, self.image, 'infinity')
        return self


async def checked_exec(env, command, timeout=60):
    result = await env.exec(command, cwd='/app', timeout_sec=timeout)
    if result.return_code:
        raise RuntimeError('Owned workspace command failed: ' + str(result.stderr or result.stdout)[:1500])
    return result


async def register_bank(env, manifest):
    scripts = sorted({str(PurePosixPath(BANK_ROOT) / PurePosixPath(name).parent)
                      for name in manifest['files'] if PurePosixPath(name).parent.name == 'scripts'
                      and name.endswith('.py')})
    script = 'import json,site,sys; from pathlib import Path; Path(site.getsitepackages()[0],"skillsbench_research.pth").write_text("\\n".join(json.loads(sys.argv[1]))+"\\n")'
    await checked_exec(env, 'python -c ' + shlex.quote(script) + ' ' + shlex.quote(json.dumps(['/app', *scripts])))
    verify = '''import hashlib,importlib.util,json,sys
from pathlib import Path
files=json.loads(sys.argv[1]); modules={}
for name,expected in files.items():
    path=Path(sys.argv[2])/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    if path.parent.name=='scripts' and path.suffix=='.py' and path.stem!='__init__':
        spec=importlib.util.find_spec(path.stem)
        assert spec and Path(spec.origin).resolve()==path.resolve(),name
        modules[path.stem]=spec.origin
print(json.dumps({'bank_files':len(files),'importable_modules':modules}))'''
    result = await checked_exec(env, 'python -c ' + shlex.quote(verify) + ' '
                                + shlex.quote(json.dumps(manifest['files'])) + ' ' + shlex.quote(BANK_ROOT))
    return json.loads(result.stdout)


async def initialize_solver(env, manifest, case):
    await checked_exec(env, 'command -v git && mkdir -p /app/experiment && git init -q && '
        'git config user.email study@example.invalid && git config user.name SkillsBenchResearch && '
        "printf '/experiment/\\n' > .gitignore && git add -A -- . && git commit -q -m solver-baseline && git diff --exit-code")
    probe = '''import hashlib,json
from pathlib import Path
assert not any(Path(path).exists() for path in ('/tests','/solution','/calibration_support'))
print(json.dumps({'input_sha256':hashlib.sha256(Path('/app/script.txt').read_bytes()).hexdigest(),'hidden_artifacts':False}))'''
    result = json.loads((await checked_exec(env, 'python -c ' + shlex.quote(probe))).stdout)
    if result['input_sha256'] != case['input_sha256']:
        raise ValueError('Solver image public input differs from prepared task')
    result.update(await register_bank(env, manifest))
    return result


def candidate_replay(snapshot, output):
    """Keep original input and the immutable bank outside the candidate replay archive."""
    input_sha = None
    target = output / 'candidate-replay.tar'
    with tarfile.open(snapshot) as original, tarfile.open(target, 'w') as replay:
        for member in original:
            parts = list(PurePosixPath(member.name).parts)
            if parts and parts[0] == 'app':
                parts.pop(0)
            if not parts:
                continue
            if '..' in parts or PurePosixPath(member.name).is_absolute():
                raise ValueError('Candidate archive escapes the owned application root')
            name = '/'.join(parts)
            if name == 'script.txt':
                if not member.isfile() or input_sha is not None:
                    raise ValueError('Candidate has an invalid public input file')
                input_sha = hashlib.sha256(original.extractfile(member).read()).hexdigest()
                continue
            if name == '.sans/skills' or name.startswith('.sans/skills/') or name == '.git' or name.startswith('.git/'):
                continue
            updated = copy(member)
            updated.name = name
            replay.addfile(updated, original.extractfile(member) if member.isfile() else None)
    return target, input_sha


def score(output, calibration, *, source_integrity):
    ctrf = json.loads((output / 'ctrf.json').read_text(encoding='utf-8'))['results']
    full = json.loads((output / 'pytest-cases.json').read_text(encoding='utf-8'))
    reward = float((output / 'reward.txt').read_text(encoding='utf-8').strip())
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError('Invalid official reward')
    expected = {case['name'] for case in calibration['cases']}
    cases = full.get('cases', [])
    names = [case.get('name') for case in cases]
    expected_functions = {case['name'] for case in calibration['official_ctrf_functions']}
    functions = [case.get('name') for case in ctrf['tests']]
    summary = ctrf['summary']
    passed_phases = all(all(case.get('phases', {}).get(phase) == 'passed'
                            for phase in ('setup', 'call', 'teardown'))
                        for case in cases if case.get('status') == 'passed')
    valid = (set(names) == expected and len(names) == len(set(names)) and full.get('collected') == names
             and set(functions) == expected_functions and len(functions) == len(set(functions))
             and all(case.get('status') in {'passed', 'failed', 'error', 'skipped'} for case in cases)
             and type(full.get('exitstatus')) is int and full['exitstatus'] in {0, 1}
             and type(summary.get('tests')) is int and summary['tests'] == len(expected_functions)
             and type(summary.get('passed')) is int and 0 <= summary['passed'] <= summary['tests']
             and passed_phases
             and reward == round(summary['passed'] / summary['tests'], 3))
    resolved = bool(valid and source_integrity and reward == 1 and full['exitstatus'] == 0
                    and all(case['status'] == 'passed' for case in cases))
    return {'reward': reward, 'valid_evaluation': valid, 'resolved': resolved,
            'source_integrity': source_integrity, 'official_ctrf_summary': summary,
            'official_ctrf_functions': ctrf['tests'], 'pytest_exitstatus': full['exitstatus'], 'cases': cases,
            'summary': {'tests': len(cases), **{status: sum(case['status'] == status for case in cases)
                       for status in ('passed', 'failed', 'error', 'skipped', 'incomplete')}}}


async def grade(case, ready, bank, manifest, snapshot, output, ready_path):
    output.mkdir()
    replay, input_sha = candidate_replay(snapshot, output)
    task = ROOT / case['task_root']
    support = ready_path.parent / 'calibration-support'
    mounts = [*mount(task / 'tests', '/tests'), *mount(support, '/calibration_support'),
              *mount(output, '/logs/verifier', readonly=False), *mount(task / 'environment/script.txt', '/app/script.txt')]
    with BankWorkspace(case['image'], bank, extra_mounts=mounts) as env:
        await env.upload_file(replay, '/candidate-replay.tar')
        await checked_exec(env, 'tar -xf /candidate-replay.tar -C /app')
        await register_bank(env, manifest)
        command = ('PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTEST_PLUGINS=calibration_identity '
                   'PYTHONPATH=/calibration_support bash /tests/test.sh')
        result = await env.exec(command, cwd='/app', timeout_sec=240)
        (output / 'execution.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
        value = score(output, ready['calibration'][manifest['split']]['gold'],
                      source_integrity=input_sha == case['input_sha256'])
        value.update(shell_exit_code=result.return_code, candidate_input_sha256=input_sha,
                     original_input_sha256=case['input_sha256'], snapshot_sha256=digest(snapshot),
                     replay_sha256=digest(replay), network='none', human_gold_exposed=False)
        save(output / 'score.json', value)
        return value


class CallLimitsMixin(trial.SummaryLimitMixin):
    _trial_limits = LIMITS

    async def _await_model_call(self, call):
        summary = bool(self._trial_call_purpose and self._trial_call_purpose.startswith('session_fold'))
        limits = self._trial_limits
        counter, limit = ('_trial_summary_calls', limits['summary_calls']) if summary else ('_trial_solve_attempts', limits['solve_calls'])
        if getattr(self, counter) >= limit:
            call.close()
            if not summary:
                self.termination_reason = {'kind': 'solve_calls'}
            raise _BudgetExceeded('Separate SkillsBench ' + ('summary' if summary else 'solve') + ' allowance exhausted')
        # Skip the parent 24/6 allowance; retain its side-query purpose bookkeeping.
        setattr(self, counter, getattr(self, counter) + 1)
        return await _FoldingLoop._await_model_call(self, call)


async def run(protocol_path, split, arm, output, *, bank=None, hint_file=None, folding='plain'):
    output = artifact(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'split': split, 'arm': arm, 'status': 'incomplete', 'paid_model_calls': 0}
    trace = None
    try:
        protocol, ready, case = validate_protocol(protocol_path, split, arm)
        limits = protocol['limits']
        workflow, solver_prompt, prompt_sha = workflow_profile(protocol)
        save(output / 'protocol.json', protocol)
        study_bank, manifest = stage_bank(case, split, arm, output, bank, hint_file)
        receipt.update(protocol_sha256=digest(protocol_path), ready_sha256=protocol['ready_sha256'],
                       bank_manifest_sha256=digest(output / 'bank-manifest.json'), folding=folding,
                       workflow=workflow, solver_prompt_sha256=prompt_sha)
        from dotenv import dotenv_values
        settings = dotenv_values(ROOT / '.env')
        if str(settings.get('OPENAI_BASE_URL', '')).rstrip('/') != BASE_URL:
            raise ValueError('Local provider differs from the frozen model route')
        key = os.environ.get('SANS_BENCHMARK_API_KEY') or settings.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('Missing local API credential')
        for name, value in list(os.environ.items()):
            if key in value:
                os.environ.pop(name)
        trace = TraceRecorder(output, run_id=output.name, task_id='skillsbench/dialogue-parser/' + split, attempt_id='001')
        base = _FoldingLoop
        if folding == 'evidence':
            from benchmarks.folding_checkpoint import EvidenceFoldingLoop
            base = EvidenceFoldingLoop
        elif folding != 'plain':
            raise ValueError('Unknown folding mode')
        loop = type('SkillsBenchLoop', (CallLimitsMixin, base), {})
        instruction = (ROOT / case['instruction']).read_text(encoding='utf-8')
        bank_index = [name for name in manifest['files'] if name.endswith('/SKILL.md')]
        system = solver_prompt + ('\nResearch Skill files are available at ' + BANK_ROOT
            + '. Inspect their SKILL.md and scripts when useful; these are procedural hints, not user requirements. '
            'The bank is frozen and read-only. Copy a helper into application code before adapting it. '
            'Available SKILL.md paths: ' + json.dumps(bank_index) + '.\n')
        if manifest['strategy_text']:
            system += '\n<research_strategy>Unverified procedural guidance:\n' + manifest['strategy_text'] + '\n</research_strategy>\n'
        with trial.transport():
            with BankWorkspace(case['image'], study_bank) as env, _isolated_home(output):
                receipt['solver_initialization'] = await initialize_solver(env, manifest, case)
                tools = EnvironmentTools(env, '/app')
                with ResearchBudget(output, trace, allocation=limits['case_tokens']) as budget:
                    agent = loop(token_budget=budget.token_budget, before_model_call=budget.before_call,
                        auto_compact=protocol.get('auto_compact', True),
                        fold_every_decisions=limits['fold_every_decisions'] or None, max_decisions=limits['solve_calls'],
                        model=MODEL, api_base=BASE_URL, api_key=key, thinking=True,
                        openai_extra_body={'reasoning_effort': 'low'}, custom_system_prompt=system,
                        custom_tools=get_solver_tool_definitions('B0'), tool_executor=tools, repo_state_probe=tools.repo_state,
                        repair_task=TaskEnvelope('skillsbench/dialogue-parser/' + split, instruction, '/app',
                                                requires_repair_witness=workflow == 'repair'),
                        diff_check_command=_FULL_DIFF_CHECK, skills_enabled=False, is_sub_agent=True,
                        permission_mode='bypassPermissions', trace=trace, max_model_calls=limits['total_model_calls'],
                        max_tool_calls=limits['max_tool_calls'], max_output_tokens=limits['max_output_tokens'],
                        max_input_tokens_per_call=limits['max_input_tokens'], model_call_timeout_seconds=limits['request_timeout_seconds'],
                        model_retry_limit=0)
                    agent._trial_limits = limits
                    agent._trial_summary_calls = agent._trial_solve_attempts = 0
                    agent._trial_call_purpose = None
                    try:
                        receipt['result'] = await agent.run_once('Complete the public task in /app. '
                            'Only this split input is provided. Hidden tests, gold and external task archives are unavailable.\n\n' + instruction)
                    except Exception as error:
                        receipt['solve_error'] = {'type': type(error).__name__, 'message': str(error)[:1500]}
                    finally:
                        try:
                            snapshot = await asyncio.to_thread(env.snapshot, output / 'snapshots')
                            receipt.update(snapshot=str(snapshot.relative_to(ROOT)), termination=agent.termination_reason,
                                           solve_calls=agent.decision_clock['completed'], summary_calls=agent._trial_summary_calls,
                                           solve_attempts=agent._trial_solve_attempts, repair_state=agent._repair_state.snapshot())
                        finally:
                            await agent.shutdown()
            receipt['grade'] = await grade(case, ready, study_bank, manifest, snapshot, output / 'grading', ROOT / protocol['ready_receipt'])
            receipt['status'] = 'interrupted' if receipt.get('solve_error') else 'completed'
        receipt['bank_unchanged'] = tree_hashes(study_bank) == manifest['files']
        receipt['source_unchanged'] = all(digest(ROOT / name) == sha for name, sha in protocol['source_files'].items())
        if not receipt['bank_unchanged'] or not receipt['source_unchanged']:
            receipt['status'] = 'frozen_inputs_changed'
    except BaseException as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)[:1500]}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if trace:
            events = read_trace(trace.trace_path)
            receipt['usage'] = usage_from_trace(events)
            receipt['paid_model_calls'] = receipt['usage']['physical_requests']
            receipt['natural_finish'] = any(row['event_type'] == 'hook.stop' and row['status'] == 'ok' for row in events)
            receipt['trace_summary'] = summarize_trace(events)
        save(output / 'receipt.json', receipt)
        print(json.dumps({name: receipt.get(name) for name in ('split', 'arm', 'status', 'paid_model_calls', 'error')}, ensure_ascii=False), flush=True)
    return receipt


async def smoke(protocol_path, output, arm='no_skill'):
    """Exercise snapshot, read-only bank imports and grading without constructing a model client."""
    output = artifact(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol, ready, case = validate_protocol(protocol_path, 'canonical', arm)
    study_bank, manifest = stage_bank(case, 'canonical', arm, output)
    with trial.transport():
        with BankWorkspace(case['image'], study_bank) as env:
            initialized = await initialize_solver(env, manifest, case)
            snapshot = await asyncio.to_thread(env.snapshot, output / 'snapshots')
        result = await grade(case, ready, study_bank, manifest, snapshot, output / 'grading', ROOT / protocol['ready_receipt'])
    if not result['valid_evaluation'] or not result['source_integrity'] or result['reward'] != 0:
        raise ValueError('Empty solver-state smoke differs from the calibrated baseline')
    receipt = {'status': 'passed', 'model_api_calls': 0, 'arm': arm, 'solver_initialization': initialized,
               'grade': result, 'protocol_sha256': digest(protocol_path)}
    save(output / 'receipt.json', receipt)
    print(json.dumps({'status': 'passed', 'model_api_calls': 0, 'arm': arm, 'reward': result['reward']}), flush=True)
    return receipt


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    frozen = commands.add_parser('freeze')
    frozen.add_argument('--ready', type=Path, required=True)
    frozen.add_argument('--out', type=Path, required=True)
    frozen.add_argument('--case-tokens', type=int)
    frozen.add_argument('--solve-calls', type=int)
    frozen.add_argument('--summary-calls', type=int)
    frozen.add_argument('--workflow', choices=WORKFLOW_PROMPTS, default='repair')
    dry = commands.add_parser('smoke')
    dry.add_argument('--protocol', type=Path, required=True)
    dry.add_argument('--out', type=Path, required=True)
    dry.add_argument('--arm', choices=('no_skill', 'curated'), default='no_skill')
    live = commands.add_parser('run')
    live.add_argument('--protocol', type=Path, required=True)
    live.add_argument('--split', choices=ARMS, required=True)
    live.add_argument('--arm', choices=[*TRAIN_STRATEGIES, *ARMS['canonical']], required=True)
    live.add_argument('--out', type=Path, required=True)
    live.add_argument('--bank', type=Path)
    live.add_argument('--hint-file', type=Path)
    live.add_argument('--folding', choices=('plain', 'evidence'), default='plain')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(freeze(args.ready, args.out, case_tokens=args.case_tokens,
                     solve_calls=args.solve_calls, summary_calls=args.summary_calls, workflow=args.workflow))
    elif args.command == 'smoke':
        asyncio.run(smoke(args.protocol, args.out, args.arm))
    else:
        asyncio.run(run(args.protocol, args.split, args.arm, args.out,
                        bank=args.bank, hint_file=args.hint_file, folding=args.folding))


if __name__ == '__main__':
    main()
