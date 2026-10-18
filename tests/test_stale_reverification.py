"""Stale test anchors stay historical and require a real current rerun; API0."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agents.repair_state import RepairState, TaskEnvelope
from agents.verification import python_command
from agents.telemetry import TraceRecorder
from agents.tools import get_solver_tool_definitions
from benchmarks.folding_checkpoint import EvidenceFoldingLoop, _required_evidence_ids, render_checkpoint

ISSUE='Return seven. Preserve deterministic output. Validate input.'

def state():
    r=RepairState(TaskEnvelope('stale-test',ISSUE,'/repo',False,False),'base')
    for identifier,quote in [('AC1','Return seven.'),('AC2','Preserve deterministic output.'),('AC3','Validate input.')]:
        r.register_acceptance_case(identifier,quote,quote)
    return r

def check(r,ids=('AC1','AC2','AC3'),*,passed=True,recognized=True,scope='full',diagnostic=False):
    e=r.record_evidence(tool='run_shell',status='ok' if passed else 'error',exit_code=0 if passed else 1,
        input_summary='python -B -m pytest test_public.py -q',repo_before=r.current_repo_state,repo_after=r.current_repo_state,
        output_preview='1 passed' if passed else '1 failed',test_pass_observed=passed,test_failure_observed=not passed,
        diagnostic=diagnostic,execution_started=True)
    v=r.record_verification(e.id,ids,scope=scope,oracle='diff_check' if scope=='diff' else 'existing_test',runner_recognized=recognized)
    return e,v

def picked(r,ids):
    return dict(evidence_refs=[dict(execution_id=r.execution_id,evidence_id=i) for i in ids],hypotheses=[],pending_actions=[])

def rendered(r):
    return json.loads(render_checkpoint(r,picked(r,_required_evidence_ids(r))))

def stale(r):
    r.observe_repo_state('diagnostic-file'); r.observe_repo_state('base')

@pytest.mark.parametrize('condition',['failed','unrecognized','diff','different_bytes','current_generation','no_pending','diagnostic','unscoped','missing_observation'])
def test_ineligible_checks_do_not_create_reverification_anchor(condition):
    r=state()
    e,_=check(r,ids=() if condition in ('diff','unscoped') else ('AC1','AC2','AC3'),
        passed=condition!='failed',recognized=condition!='unrecognized',scope='diff' if condition=='diff' else 'full',
        diagnostic=condition=='diagnostic')
    if condition=='different_bytes':r.observe_repo_state('other-code')
    elif condition!='current_generation':stale(r)
    if condition=='no_pending':check(r)
    if condition=='missing_observation':r._evidence.clear()
    assert rendered(r)['runtime_required_reverification'] is None

def test_latest_eligible_check_only_covers_current_unverified_intersection():
    r=state(); earlier,_=check(r); latest,v=check(r,('AC3','AC1','AC2')); stale(r)
    current,_=check(r,('AC2',))
    before=r.snapshot(); payload=rendered(r); anchor=payload['runtime_required_reverification']
    assert anchor['verification_id']==v.id and anchor['evidence_id']==latest.id
    assert anchor['acceptance_ids']==['AC3','AC1']
    assert anchor['current_repo_generation']==2 and anchor['verified_repo_generation']==0
    assert anchor['scope']=='full' and anchor['oracle']=='existing_test'
    assert anchor['input_preview']==latest.input_summary and anchor['observation_reader']=={'evidence_id':latest.id,'part':'input'}
    assert 'actually rerun' in anchor['instruction'] and latest.id in payload['pinned_evidence_ids']
    assert earlier.id not in payload['pinned_evidence_ids'] and current.id not in payload['pinned_evidence_ids']
    assert not r.completion_decision().complete and r.snapshot()==before
    assert payload['recorded_verifications'][0]['current_verification'] is False
    with pytest.raises(ValueError,match='omitted required'):
        render_checkpoint(r,picked(r,[current.id]))

def test_reverification_anchor_preserves_twelve_reference_capacity(tmp_path,monkeypatch):
    r=state(); green,_=check(r); stale(r); pins=[]
    for i in range(12):
        e=r.record_evidence(tool='read_file',status='ok',input_summary=f'helper{i}.py',repo_before='base',repo_after='base')
        pins.append(e.id)
    r.upsert_hypothesis('cause','Twelve observations constrain the implementation.','All constraints hold.','One is contradicted.',evidence_ids=pins)
    assert len(_required_evidence_ids(r))==13 and green.id in _required_evidence_ids(r)
    a=EvidenceFoldingLoop(token_budget={'charged':0,'limit':None},api_base='https://example.invalid/v1',api_key='test',model='test',
        custom_system_prompt='fixed',custom_tools=[],skills_enabled=False,is_sub_agent=True,repair_task=None)
    a._repair_state=r
    async def forbidden(*args,**kwargs):pytest.fail('Capacity failure must happen before any summary request')
    monkeypatch.setattr(a,'_side_query_once',forbidden)
    async def exercise():
        try:
            with pytest.raises(ValueError,match='12-reference'):
                await a._run_compaction_query([],'Return selection')
            assert a._operation_counts['model_calls']==0 and a.token_budget['charged']==0
        finally:await a.shutdown()
    asyncio.run(exercise())

def test_real_formal_pytest_must_rerun_after_diagnostic_creation_and_removal(tmp_path,monkeypatch):
    work=tmp_path/'work'; work.mkdir()
    files={'helper.py':'def answer():\n    return 7\n',
        'test_public.py':'from helper import answer\ndef test_answer():\n    assert answer() == 7\n',
        'pytest.ini':'[pytest]\naddopts = --rootdir=. -p no:cacheprovider\n',
        '.gitignore':'__pycache__/\n.pytest_cache/\nexperiment/\n'}
    for name,body in files.items():(work/name).write_text(body,encoding='utf-8',newline='')
    for args in (['init','-q'],['config','core.autocrlf','false'],['add','.'],
        ['-c','user.name=Test','-c','user.email=test@localhost','-c','commit.gpgsign=false','commit','-qm','Public fixture']):
        subprocess.run(['git',*args],cwd=work,capture_output=True,check=True)
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip()
    monkeypatch.chdir(work); monkeypatch.setenv('PATH',str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1'); monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')
    monkeypatch.setenv('PYTEST_ADDOPTS','--rootdir=. -p no:cacheprovider')
    a=EvidenceFoldingLoop(token_budget={'charged':0,'limit':None},api_base='https://example.invalid/v1',api_key='test',model='test',
        custom_system_prompt='Verify current output.',custom_tools=get_solver_tool_definitions('B0'),skills_enabled=False,is_sub_agent=True,
        permission_mode='bypassPermissions',repair_task=TaskEnvelope('actual-stale','Return seven.',str(work),False,False),
        trace=TraceRecorder(tmp_path/'trace',run_id='stale',task_id='actual-stale',attempt_id='local'),max_model_calls=1)
    async def forbidden_model(**kwargs):pytest.fail('This verification must never call a model')
    monkeypatch.setattr(a._openai_client.chat.completions,'create',forbidden_model)
    a._baseline_ref=base; r=a._repair_state; r.register_acceptance_case('AC1','Return seven.','Return seven.')
    formal={'command':'python -B -m pytest test_public.py -q -c pytest.ini','acceptance_ids':['AC1'],'scope':'full','oracle':'existing_test'}
    async def exercise():
        try:
            await a._execute_tool_call('run_shell',formal); green=r.evidence[-1]; verification=r.verifications[-1]
            assert green.execution_started and green.test_pass_observed and green.test_cases and r.completion_decision().complete
            before_state=r.current_repo_state
            await a._execute_tool_call('write_file',{'file_path':'draft_diagnostic_stale.py','content':'def test_probe():\n    assert True\n'})
            await a._execute_tool_call('run_shell',{'command':'python -B -m pytest draft_diagnostic_stale.py -q -c pytest.ini','diagnostic':True})
            diagnostic=r.evidence[-1]; assert diagnostic.diagnostic and diagnostic.execution_started and diagnostic.test_pass_observed
            await a._execute_tool_call('run_shell',{'command':python_command("from pathlib import Path; Path('draft_diagnostic_stale.py').unlink()",[],str(work),sys.executable)})
            assert not (work/'draft_diagnostic_stale.py').exists() and r.current_repo_state==before_state
            assert r.repo_generation==2 and not r.completion_decision().complete
            snapshot=r.snapshot(); payload=rendered(r); anchor=payload['runtime_required_reverification']
            assert anchor['verification_id']==verification.id and anchor['evidence_id']==green.id and anchor['acceptance_ids']==['AC1']
            assert payload['recorded_verifications'][0]['current_verification'] is False
            assert green.id in payload['pinned_evidence_ids'] and diagnostic.id not in payload['pinned_evidence_ids'] and r.snapshot()==snapshot
            with pytest.raises(ValueError,match='omitted required'):
                render_checkpoint(r,picked(r,[diagnostic.id]))
            await a._execute_tool_call('run_shell',formal); current=r.evidence[-1]
            assert current.id!=green.id and current.execution_started and current.test_pass_observed and current.test_cases==green.test_cases
            assert current.repo_generation==2 and r.completion_decision().complete
            assert rendered(r)['runtime_required_reverification'] is None and a._operation_counts['model_calls']==0
        finally:await a.shutdown()
    asyncio.run(exercise())
