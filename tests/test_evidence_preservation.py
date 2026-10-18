"""Bounded typed testcase outcomes survive folding; API0 and unchanged gates."""
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import agents.agent as agent_module
from agents.repair_state import RepairState, TaskEnvelope
from agents.telemetry import TraceRecorder
from agents.tools import get_solver_tool_definitions
from benchmarks.folding_checkpoint import EvidenceFoldingLoop, _receipt_view, render_checkpoint

ISSUE='Repair CLI JSON output while preserving eval behavior.'

def repair(root):
    r=RepairState(TaskEnvelope('typed-outcomes',ISSUE,str(root),False,False),'base')
    r.register_acceptance_case('AC1',ISSUE,ISSUE)
    return r

def observed(r,cases,preview='All tests failed and eval is not implemented.'):
    e=r.record_evidence(tool='run_shell',status='error',exit_code=1,input_summary='python -B -m pytest test_public.py -q',
        repo_before=r.current_repo_state,repo_after=r.current_repo_state,output_preview=preview,
        test_failure_observed=True,test_pass_observed=False,test_cases=tuple(cases),execution_started=True)
    r.record_verification(e.id,['AC1'],scope='full',oracle='existing_test',runner_recognized=True)
    return e

def selection(r,e):
    return dict(evidence_refs=[dict(execution_id=r.execution_id,evidence_id=e.id)],
        hypotheses=['Eval may not be implemented.'],pending_actions=[])

def agent(root,trace,task=None):
    return EvidenceFoldingLoop(token_budget={'charged':0,'limit':None},api_base='https://example.invalid/v1',api_key='test',model='test',
        custom_system_prompt='Follow actual testcase evidence.',custom_tools=get_solver_tool_definitions('B0'),skills_enabled=False,is_sub_agent=True,
        permission_mode='bypassPermissions',repair_task=task,trace=TraceRecorder(trace,run_id='typed',task_id='typed-outcomes',attempt_id='API0'),
        max_output_tokens=8192,max_input_tokens_per_call=32768,max_model_calls=4)

def test_fourteen_mixed_outcomes_survive_selector_actual_fold_and_provider_wire(tmp_path,monkeypatch):
    r=repair(tmp_path)
    cases=[(f'test_public::test_eval_{i}','passed','') for i in range(5)]
    cases += [(f'test_public::test_cli_{i}','passed','') for i in range(7)]
    cases += [('test_public::test_unknown_command','failure','bad JSON'),('test_public::test_check_interface','failure','bad JSON')]
    e=observed(r,cases); before=r.snapshot(); a=agent(tmp_path,tmp_path.parent/'t'/'fold',r.task); a._repair_state=r
    a._openai_messages += [{'role':'user','content':ISSUE}, {'role':'assistant','content':'Earlier exploration. '*5000},
                          {'role':'assistant','content':'Latest observation remains available.'}]
    seen=[]
    async def select(_system,text,**kwargs):
        catalog=json.loads(text)['receipt_catalog']; facts=next(row['pytest_outcomes'] for row in catalog if row['evidence_id']==e.id)
        seen.append(facts); return json.dumps(selection(r,e))
    monkeypatch.setattr(a,'_side_query_once',select); monkeypatch.setattr(agent_module,'COMPACTION_KEEP_RECENT_TOKENS',200)
    async def exercise():
        try:
            assert await a._compact_conversation(trigger='manual') and a._fold_count==1
            facts=seen[0]; assert facts['total']==14 and facts['status_counts']=={'passed':12,'failure':2}
            assert facts['omitted_count']==0 and facts['omitted_status_counts']=={} and len(facts['sample'])==14
            assert sum(row['status']=='passed' and 'test_eval_' in row['id'] for row in facts['sample'])==5
            for provider in ('openai','anthropic'):
                a.use_openai=provider=='openai'
                if not a.use_openai:a._anthropic_messages=[m for m in a._openai_messages if m['role']!='system']
                wire=a._openai_request_messages() if a.use_openai else a._anthropic_request_messages()
                text=json.dumps(wire); assert 'pytest_outcomes' in text
                assert all(name in text for name,_status,_signature in cases)
            assert r.snapshot()==before and not r.completion_decision().complete and a._operation_counts['model_calls']==0
        finally:await a.shutdown()
    asyncio.run(exercise())

def test_bound_priority_names_and_omission_counts_keep_raw_status_labels(tmp_path):
    r=repair(tmp_path); long='test_public::'+('多'*300)
    cases=[('p00','passed',''),('skip00','skipped',''),('f00','failure',''),('p01','passed',''),('e00','error',''),
        (long,'failure',''),('e01','error','')]+[(f'p{i:02}','passed','') for i in range(2,20)]+[('skip01','skipped',''),('other','custom-status','')]
    e=observed(r,cases,preview='All tests passed!'); before=r.snapshot(); facts=_receipt_view(e)['pytest_outcomes']
    assert facts['total']==27 and facts['status_counts']==dict(Counter(status for _name,status,_signature in cases))
    assert len(facts['sample'])==16 and [x['id'] for x in facts['sample'][:4]]==['f00','e00',long[:240],'e01']
    assert [x['id'] for x in facts['sample'][4:]]==[f'p{i:02}' for i in range(12)]
    assert facts['sample'][2]['name_clipped'] is True and len(facts['sample'][2]['id'])==240
    assert facts['omitted_count']==11 and facts['omitted_status_counts']=={'passed':8,'skipped':2,'custom-status':1}
    assert r.snapshot()==before and not r.completion_decision().complete

def test_non_test_observation_does_not_gain_testcase_metadata(tmp_path):
    r=repair(tmp_path); e=r.record_evidence(tool='read_file',status='ok',input_summary='helper.py',repo_before='base',repo_after='base',
        output_preview='12 passed, 2 failed')
    assert 'pytest_outcomes' not in _receipt_view(e) and not r.completion_decision().complete

def test_real_mixed_pytest_remains_red_and_historical_after_generation_change(tmp_path,monkeypatch):
    work=tmp_path/'work'; work.mkdir()
    files={'helper.py':'def evaluate(value):\n    return value + 1\ndef cli_error():\n    return ""\n',
        'test_public.py':'from helper import evaluate, cli_error\ndef test_eval_zero():\n    assert evaluate(0) == 1\ndef test_eval_one():\n    assert evaluate(1) == 2\ndef test_cli_json():\n    assert cli_error() == \'{"error":"unknown"}\'\n',
        'pytest.ini':'[pytest]\naddopts = --rootdir=. -p no:cacheprovider\n','.gitignore':'__pycache__/\n.pytest_cache/\nexperiment/\n'}
    for name,body in files.items():(work/name).write_text(body,encoding='utf-8',newline='')
    for args in (['init','-q'],['config','core.autocrlf','false'],['add','.'],
        ['-c','user.name=Test','-c','user.email=test@localhost','-c','commit.gpgsign=false','commit','-qm','Public fixture']):
        subprocess.run(['git',*args],cwd=work,capture_output=True,check=True)
    monkeypatch.chdir(work); monkeypatch.setenv('PATH',str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1'); monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')
    monkeypatch.setenv('PYTEST_ADDOPTS','--rootdir=. -p no:cacheprovider')
    task=TaskEnvelope('real-mixed',ISSUE,str(work),False,False); a=agent(work,tmp_path.parent/'t'/'real',task)
    a._repair_state.register_acceptance_case('AC1',ISSUE,ISSUE); before=(work/'helper.py').read_bytes()
    async def forbidden(**kwargs):pytest.fail('No provider call belongs in this regression')
    monkeypatch.setattr(a._openai_client.chat.completions,'create',forbidden)
    async def exercise():
        try:
            await a._execute_tool_call('run_shell',{'command':'python -B -m pytest test_public.py -q -c pytest.ini',
                'acceptance_ids':['AC1'],'scope':'full','oracle':'existing_test'})
            r=a._repair_state; e=r.evidence[-1]; assert e.execution_started and e.exit_code==1 and len(e.test_cases)==3
            assert r.verifications[-1].runner_recognized and not r.verifications[-1].success
            payload=json.loads(render_checkpoint(r,selection(r,e))); facts=payload['selected_runtime_receipts'][0]['pytest_outcomes']
            assert facts['total']==3 and facts['status_counts']=={'passed':2,'failure':1}
            assert [x['status'] for x in facts['sample']]==['failure','passed','passed'] and not r.completion_decision().complete
            a._openai_messages.append({'role':'assistant','content':json.dumps(payload)})
            assert 'pytest_outcomes' in json.dumps(a._openai_request_messages())
            r.observe_repo_state('changed-epoch'); snapshot=r.snapshot(); historical=json.loads(render_checkpoint(r,selection(r,e)))
            assert historical['selected_runtime_receipts'][0]['state_scope']=='historical_only'
            assert historical['recorded_verifications'][0]['current_verification'] is False
            assert historical['selected_runtime_receipts'][0]['pytest_outcomes']==facts
            assert r.snapshot()==snapshot and not r.completion_decision().complete and (work/'helper.py').read_bytes()==before
            assert a._operation_counts['model_calls']==0
        finally:await a.shutdown()
    asyncio.run(exercise())
