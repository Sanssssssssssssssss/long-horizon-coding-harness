"""Offline Skill-bank, artifact-integrity and official-scoring contracts; no provider calls."""
import asyncio
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tarfile

import pytest

from benchmarks import skillsbench_trial as driver
from agents.agent import _BudgetExceeded


def calibration():
    return {'cases': [{'name': 'test_outputs.py::shape[a]'}, {'name': 'test_outputs.py::shape[b]'}],
            'official_ctrf_functions': [{'name': 'test_outputs.py::shape'}]}


def grading(folder, *, passed=True):
    folder.mkdir(exist_ok=True)
    names = ['test_outputs.py::shape[a]', 'test_outputs.py::shape[b]']
    cases = [{'name': name, 'status': 'passed' if passed else 'failed',
              'phases': {'setup': 'passed', 'call': 'passed' if passed else 'failed', 'teardown': 'passed'}} for name in names]
    (folder / 'pytest-cases.json').write_text(json.dumps({'exitstatus': 0 if passed else 1, 'collected': names, 'cases': cases}))
    (folder / 'ctrf.json').write_text(json.dumps({'results': {'summary': {'tests': 1, 'passed': int(passed)},
                                                           'tests': [{'name': 'test_outputs.py::shape', 'status': 'passed' if passed else 'failed'}]}}))
    (folder / 'reward.txt').write_text('1' if passed else '0')


def test_zero_shell_exit_does_not_override_official_failure_or_modified_input(tmp_path):
    grading(tmp_path, passed=False)
    assert not driver.score(tmp_path, calibration(), source_integrity=True)['resolved']
    grading(tmp_path)
    assert driver.score(tmp_path, calibration(), source_integrity=True)['resolved']
    result = driver.score(tmp_path, calibration(), source_integrity=False)
    assert result['reward'] == 1 and not result['resolved'] and not result['source_integrity']


@pytest.mark.parametrize('damage', ['missing_node', 'extra_node', 'missing_call', 'reward_mismatch'])
def test_incomplete_or_unbound_oracle_results_cannot_be_full_pass(tmp_path, damage):
    grading(tmp_path)
    path = tmp_path / 'pytest-cases.json'
    full = json.loads(path.read_text())
    if damage == 'missing_node':
        full['cases'].pop()
        full['collected'].pop()
    elif damage == 'extra_node':
        full['cases'].append({'name': 'unknown', 'status': 'passed', 'phases': {}})
        full['collected'].append('unknown')
    elif damage == 'missing_call':
        full['cases'][0]['phases'].pop('call')
    else:
        (tmp_path / 'reward.txt').write_text('0.5')
    path.write_text(json.dumps(full))
    result = driver.score(tmp_path, calibration(), source_integrity=True)
    assert not result['valid_evaluation'] and not result['resolved']


def test_snapshot_replay_excludes_input_bank_and_git_but_retains_candidate(tmp_path):
    snapshot = tmp_path / 'snapshot.tar'
    with tarfile.open(snapshot, 'w') as archive:
        for name, content in {'app/script.txt': b'changed input', 'app/.sans/skills/teacher/SKILL.md': b'skill',
                              'app/.git/config': b'git', 'app/solution.py': b'def parse_script(text): return {}'}.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    replay, input_sha = driver.candidate_replay(snapshot, tmp_path)
    assert input_sha != driver.hashlib.sha256(b'original input').hexdigest()
    with tarfile.open(replay) as archive:
        assert archive.getnames() == ['solution.py']


def test_train_and_noskill_cannot_receive_external_or_curated_banks(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    output = tmp_path / 'experiment/runs/test'
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match='Training uses an empty bank'):
        driver.stage_bank({}, 'train', 'grammar_first', output, bank=tmp_path / 'human')
    with pytest.raises(ValueError, match='NoSkill requires an empty bank'):
        driver.stage_bank({}, 'canonical', 'no_skill', output, bank=tmp_path / 'human')
    with pytest.raises(ValueError, match='do not accept extra'):
        driver.stage_bank({}, 'canonical', 'no_skill', output, hint_file=tmp_path / 'hint.md')


def test_whole_learned_folder_is_copied_frozen_with_actual_script(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    source = tmp_path / 'experiment/research/learned-dialogue'
    (source / 'scripts').mkdir(parents=True)
    (source / 'SKILL.md').write_text('Read the grammar and export one graph.')
    (source / 'scripts/parser_helper.py').write_text('def parse(text): return {}\n')
    output = tmp_path / 'experiment/runs/learned'
    output.mkdir(parents=True)
    bank, manifest = driver.stage_bank({}, 'canonical', 'learned', output, bank=source)
    assert set(manifest['files']) == {'learned-dialogue/SKILL.md', 'learned-dialogue/scripts/parser_helper.py'}
    assert manifest['read_only'] and manifest['container_path'] == '/app/.sans/skills'
    (source / 'SKILL.md').write_text('Changed source')
    assert driver.tree_hashes(bank) == manifest['files']


def test_owned_solver_mounts_only_readonly_bank_with_symmetric_permissions(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver.checkpoints, 'docker', lambda *args: calls.append(args))
    monkeypatch.setattr(driver, 'mount', lambda source, target, readonly=True: ['--mount', f'{target}:readonly={readonly}'])
    env = driver.BankWorkspace('prepared-image', tmp_path)
    env.__enter__()
    call = calls[0]
    assert '/app/.sans/skills:readonly=True' in call
    assert '--network' in call and call[call.index('--network') + 1] == 'none'
    assert env.root == '/app' and env.memory == '4g' and env.cpus == 2


@pytest.mark.parametrize('solve_limit', [12, 20])
def test_frozen_solve_and_summary_requests_have_separate_caps(solve_limit):
    class FakeAwait:
        async def _await_model_call(self, call):
            return await call
    async def accept():
        return 'accepted'
    original = driver._FoldingLoop._await_model_call
    driver._FoldingLoop._await_model_call = FakeAwait._await_model_call
    class Loop(driver.CallLimitsMixin):
        _trial_summary_calls = _trial_solve_attempts = 0
        _trial_call_purpose = None
        _trial_limits = {**driver.LIMITS, 'solve_calls': solve_limit, 'total_model_calls': solve_limit + 4}
    async def exercise():
        loop = Loop()
        for _ in range(solve_limit):
            assert await loop._await_model_call(accept()) == 'accepted'
        with pytest.raises(_BudgetExceeded, match='solve'):
            await loop._await_model_call(accept())
        loop._trial_call_purpose = 'session_fold'
        for _ in range(4):
            assert await loop._await_model_call(accept()) == 'accepted'
        with pytest.raises(_BudgetExceeded, match='summary'):
            await loop._await_model_call(accept())
        assert loop._trial_solve_attempts == solve_limit and loop._trial_summary_calls == 4
    try:
        asyncio.run(exercise())
    finally:
        driver._FoldingLoop._await_model_call = original


def test_admission_rejection_writes_receipt_without_constructing_client(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    def reject(*args):
        raise ValueError('Frozen preparation changed')
    monkeypatch.setattr(driver, 'validate_protocol', reject)
    output = tmp_path / 'experiment/runs/rejected'
    receipt = asyncio.run(driver.run(tmp_path / 'protocol.json', 'train', 'grammar_first', output))
    assert receipt['paid_model_calls'] == 0 and receipt['error']['message'] == 'Frozen preparation changed'
    assert (output / 'receipt.json').is_file()


@pytest.mark.parametrize('change', [
    {'case_tokens': True}, {'case_tokens': 0}, {'solve_calls': -1}, {'solve_calls': 20.0},
    {'summary_calls': '4'}, {'max_output_tokens': 4096}, {'max_input_tokens': 65536},
    {'max_tool_calls': 160}, {'total_model_calls': 24}, {'new_budget_field': 1},
])
def test_budget_configuration_rejects_invalid_or_nonbudget_changes(change):
    with pytest.raises(ValueError):
        driver.validate_limits({**driver.LIMITS, **change})


def test_freeze_explicit_budget_preserves_default_and_other_conditions(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    ready_path = tmp_path / 'experiment/research/ready.json'
    ready_path.parent.mkdir(parents=True)
    ready_path.write_text('{}')
    tasks = {}
    for split in ('train', 'canonical'):
        tasks[split] = tmp_path / 'experiment/cache' / split
        tasks[split].mkdir(parents=True)
        (tasks[split] / 'instruction.md').write_text('Public ' + split + ' task')
    ready = {'image_ids': {'train': 'train-image', 'canonical': 'canonical-image'},
             'images': {split: {'input_sha256': split + '-input'} for split in tasks}}
    monkeypatch.setattr(driver, 'preparer', lambda: SimpleNamespace(validate_ready=lambda path: ready,
        official_inputs=lambda: (tasks, {}, {})))
    for options, expected in (({}, driver.LIMITS),
        ({'case_tokens': 240000, 'solve_calls': 20, 'summary_calls': 4},
         {**driver.LIMITS, 'case_tokens': 240000, 'solve_calls': 20, 'total_model_calls': 24}),
        ({'workflow': 'build'}, driver.LIMITS),
        ({'solve_calls': 8, 'summary_calls': 0}, {**driver.LIMITS, 'solve_calls': 8, 'summary_calls': 0, 'total_model_calls': 8, 'fold_every_decisions': 0})):
        label = 'zero_summary' if options.get('summary_calls') == 0 else options.get('workflow', 'diagnostic' if options else 'default')
        path = driver.freeze(ready_path, tmp_path / 'experiment/runs' / label, **options)
        protocol, _, _ = driver.validate_protocol(path, 'train', 'grammar_first')
        assert protocol['limits'] == expected
        assert protocol['model'] == driver.MODEL and protocol['api_base'] == driver.BASE_URL
        assert protocol['hidden_feedback_to_solver'] is False
        assert protocol['auto_compact'] is (expected['summary_calls'] > 0)
        workflow = options.get('workflow', 'repair')
        assert protocol['workflow'] == workflow
        assert protocol['solver_prompt_sha256'] == driver.hashlib.sha256(driver.WORKFLOW_PROMPTS[workflow].encode()).hexdigest()
    assert driver.LIMITS['case_tokens'] == 120000 and driver.LIMITS['total_model_calls'] == 16


@pytest.mark.parametrize('workflow, summaries', [('repair', 4), ('build', 4), (None, 4), ('build', 0)])
def test_runtime_uses_frozen_budget_for_admission_agent_and_separate_caps(tmp_path, monkeypatch, workflow, summaries):
    from types import SimpleNamespace
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    instruction = tmp_path / 'instruction.md'
    instruction.write_text('Public task')
    (tmp_path / '.env').write_text('OPENAI_BASE_URL=' + driver.BASE_URL + '\nOPENAI_API_KEY=offline-test-key\n')
    limits = {**driver.LIMITS, 'case_tokens': 240000, 'solve_calls': 20, 'summary_calls': summaries,
              'total_model_calls': 20 + summaries, 'fold_every_decisions': 6 if summaries else 0}
    protocol = {'limits': limits, 'ready_sha256': 'ready', 'ready_receipt': 'ready.json', 'source_files': {},
                'auto_compact': bool(summaries)}
    if workflow:
        protocol.update(workflow=workflow, solver_prompt_sha256=driver.hashlib.sha256(driver.WORKFLOW_PROMPTS[workflow].encode()).hexdigest())
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(driver, 'validate_protocol', lambda *args: (protocol, {}, {'image': 'image', 'instruction': instruction.name}))
    captured = {}
    class Workspace:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def snapshot(self, output):
            output.mkdir()
            path = output / 'snapshot.tar'
            path.write_bytes(b'owned fixture')
            return path
    class Budget:
        def __init__(self, output, trace, *, allocation):
            captured['allocation'] = allocation
            self.token_budget = {'limit': allocation, 'charged': 0}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        async def before_call(self):
            raise AssertionError('Offline wiring check must never reach a provider')
    class Agent:
        def __init__(self, **options):
            captured['options'] = options
            self.decision_clock = {'completed': 0}
            self.termination_reason = None
            self._repair_state = SimpleNamespace(snapshot=lambda: {})
        async def run_once(self, prompt):
            assert self._trial_limits == limits
            return {'text': '', 'tokens': {'input': 0, 'output': 0}}
        async def shutdown(self):
            pass
    async def initialize(*args):
        return {}
    async def grade(*args):
        return {'reward': 0}
    monkeypatch.setattr(driver, 'BankWorkspace', Workspace)
    monkeypatch.setattr(driver, 'ResearchBudget', Budget)
    monkeypatch.setattr(driver, '_FoldingLoop', Agent)
    monkeypatch.setattr(driver, 'EnvironmentTools', lambda *args: SimpleNamespace(repo_state=None))
    monkeypatch.setattr(driver, 'initialize_solver', initialize)
    monkeypatch.setattr(driver, 'grade', grade)
    monkeypatch.setattr(driver.trial, 'transport', nullcontext)
    monkeypatch.setattr(driver, '_isolated_home', lambda output: nullcontext())
    receipt = asyncio.run(driver.run(protocol_path, 'train', 'grammar_first', tmp_path / 'experiment/runs/wiring'))
    assert receipt['status'] == 'completed' and receipt['paid_model_calls'] == 0
    assert captured['allocation'] == 240000
    assert captured['options']['max_decisions'] == 20 and captured['options']['max_model_calls'] == 20 + summaries
    assert captured['options']['auto_compact'] is bool(summaries)
    assert captured['options']['fold_every_decisions'] == (6 if summaries else None)
    expected_workflow = workflow or 'repair'
    assert captured['options']['custom_system_prompt'].startswith(driver.WORKFLOW_PROMPTS[expected_workflow])
    assert captured['options']['repair_task'].requires_patch is True
    assert captured['options']['repair_task'].requires_repair_witness is (expected_workflow == 'repair')
    assert receipt['workflow'] == expected_workflow
    assert all(captured['options'][argument] == limits[field] for argument, field in (
        ('max_tool_calls', 'max_tool_calls'),
        ('max_output_tokens', 'max_output_tokens'), ('max_input_tokens_per_call', 'max_input_tokens'),
        ('model_call_timeout_seconds', 'request_timeout_seconds')))


@pytest.mark.parametrize('profile', [
    {'workflow': 'other'}, {'workflow': True}, {'workflow': 'build'},
    {'workflow': 'repair'}, {'solver_prompt_sha256': 'unbound'},
    {'workflow': 'build', 'solver_prompt_sha256': driver.hashlib.sha256(driver.SOLVER_SYSTEM_PROMPT.encode()).hexdigest()},
])
def test_unbound_or_changed_workflow_prompt_is_rejected(profile):
    with pytest.raises(ValueError):
        driver.workflow_profile(profile)


def test_zero_summary_refuses_provider_and_preserves_solve_allowance(monkeypatch):
    async def accept(self, call):
        return await call
    monkeypatch.setattr(driver._FoldingLoop, '_await_model_call', accept)
    class Loop(driver.CallLimitsMixin):
        _trial_limits = {**driver.LIMITS, 'solve_calls': 8, 'summary_calls': 0, 'total_model_calls': 8, 'fold_every_decisions': 0}
        _trial_summary_calls = _trial_solve_attempts = 0
        _trial_call_purpose = 'session_fold'
    entered = []
    async def response():
        entered.append(True)
        return 'ok'
    async def exercise():
        loop = Loop()
        with pytest.raises(_BudgetExceeded, match='summary'):
            await loop._await_model_call(response())
        assert not entered and loop._trial_summary_calls == 0 and loop._trial_solve_attempts == 0
        loop._trial_call_purpose = None
        for _ in range(8):
            assert await loop._await_model_call(response()) == 'ok'
        assert len(entered) == 8
        with pytest.raises(_BudgetExceeded, match='solve'):
            await loop._await_model_call(response())
    asyncio.run(exercise())


def test_disabled_folding_never_attempts_summary_at_period_or_pressure_boundary(monkeypatch):
    loop = driver._FoldingLoop(token_budget={'limit': 1, 'charged': 1}, auto_compact=False,
        fold_every_decisions=None, max_decisions=8, api_base='https://example.invalid/v1',
        api_key='offline', custom_tools=[], skills_enabled=False, is_sub_agent=True,
        max_model_calls=8, max_input_tokens_per_call=32768, max_output_tokens=8192)
    loop.decision_clock = {'completed': 6, 'last_periodic': 0}
    loop.last_input_token_count = 100000
    async def forbidden(*args, **kwargs):
        raise AssertionError('Disabled folding cannot start a summary')
    monkeypatch.setattr(loop, '_compact_conversation', forbidden)
    async def exercise():
        try:
            assert await loop._check_and_compact() is False
        finally:
            await loop.shutdown()
    asyncio.run(exercise())
