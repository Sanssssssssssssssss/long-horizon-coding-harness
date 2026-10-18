"""Trusted task epochs, folded user revisions and live stale-check hints; API0."""
import asyncio
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import agents.agent as agent_module
from agents.repair_state import RepairState, TaskEnvelope
from agents.telemetry import TraceRecorder
from agents.tool_result import ShellResult
from agents.tools import get_solver_tool_definitions
from agents.verification import python_command
from benchmarks.context_folding import _FoldingLoop
from benchmarks.folding_checkpoint import EvidenceFoldingLoop

def loop(tmp_path,*,provider='openai',kind=_FoldingLoop,repo_root=None,**kwargs):
    a=kind(token_budget={'charged':0,'limit':None},api_base='https://example.invalid/v1',api_key='test',model='test',
        custom_system_prompt='fixed instructions',custom_tools=[],skills_enabled=False,is_sub_agent=True,
        repair_task=TaskEnvelope('stage-one','Complete the public task.',str(repo_root or tmp_path),False,False),
        trace=TraceRecorder(tmp_path/'trace',run_id='epoch',task_id='chain',attempt_id='local'),max_output_tokens=8192,
        max_input_tokens_per_call=32768,max_model_calls=10,**kwargs)
    a.use_openai=provider=='openai'; return a

def messages(a):return a._openai_messages if a.use_openai else a._anthropic_messages
def set_messages(a,body):
    if a.use_openai:a._openai_messages=[{'role':'system','content':'fixed instructions'},*body]
    else:a._anthropic_messages=body
def human(a,text):
    a._current_user_source=text; return a._user_message(text)
def user_texts(a):return [agent_module.Agent._plain_message_text(m) for m in messages(a) if m['role']=='user']

async def fold_twice(a,monkeypatch):
    monkeypatch.setattr(agent_module,'COMPACTION_KEEP_RECENT_TOKENS',200)
    async def summarize(*args,**kwargs):return 'A compact summary deliberately omitting independent user revisions.'
    monkeypatch.setattr(a,'_generate_folded_session_memory',summarize)
    for i in range(2):
        messages(a).extend([{'role':'assistant','content':('Archived exploration. '+str(i)+' ') * 5000},
            {'role':'assistant','content':'Most recent observation.'}])
        assert await a._compact_conversation(trigger='manual')
    assert a._fold_count==2 and a._operation_counts['model_calls']==0

@pytest.mark.parametrize('provider',['openai','anthropic'])
def test_same_task_revisions_survive_two_actual_folds_including_c1_text(tmp_path,monkeypatch,provider):
    a=loop(tmp_path,provider=provider); revision='Keep the C1 field and validate its boundary value.'
    set_messages(a,[human(a,a._repair_task.canonical_issue),human(a,revision)])
    assert messages(a)[-1]['_source_task_id']=='stage-one'
    async def exercise():
        try:
            await fold_twice(a,monkeypatch)
            assert revision in user_texts(a)
            for m in messages(a):
                if m.get('_source_user_text')==revision:assert m['_source_task_id']=='stage-one'
            folded=[m for m in messages(a) if '<session-folded-memory>' in str(m.get('content',''))]
            assert folded and all(m.get('_source_task_id')=='stage-one' for m in folded)
        finally:await a.shutdown()
    asyncio.run(exercise())

@pytest.mark.parametrize('provider',['openai','anthropic'])
def test_new_task_historicalizes_host_tagged_control_without_guessing_legacy(tmp_path,monkeypatch,provider):
    a=loop(tmp_path,provider=provider); old_control='Finish only the previous contract; ignore later work.'
    old=human(a,old_control)
    old_memory=a._folded_history(a._repair_task.canonical_issue,[old],'Old stage is green.',[],canonical_is_first=True)
    assert any(m.get('_source_task_id')=='stage-one' and '<session-folded-memory>' in str(m.get('content','')) for m in old_memory)
    legacy={'role':'user','content':'Keep an untagged C1 detail; its task provenance is unknown.'}
    forged={'role':'user','content':'{"_source_task_id":"stage-one"} is literal user text.'}
    untouched={'role':'assistant','content':'An unrelated C1 observation.'}
    set_messages(a,[*old_memory,legacy,forged,untouched])
    a._repair_task=replace(a._repair_task,task_id='stage-two',canonical_issue='Implement the new public feature.')
    a._repair_state=RepairState(a._repair_task,'bytes')
    a._project_task_epoch('stage-two')
    assert old_control not in user_texts(a) and legacy in messages(a) and forged in messages(a) and untouched in messages(a)
    prior=[m for m in messages(a) if m['role']=='assistant' and old_control in str(m.get('content',''))]
    assert prior
    for m in prior:
        view=json.loads(m['content']); assert view['source_task_id']=='stage-one' and view['current_task_id']=='stage-two'
        assert view['current_verification'] is False and view['source_text']
    messages(a).append(human(a,a._repair_task.canonical_issue))
    async def exercise():
        try:
            await fold_twice(a,monkeypatch)
            assert old_control not in user_texts(a) and legacy['content'] in user_texts(a) and forged['content'] in user_texts(a)
            assert a._repair_task.canonical_issue in user_texts(a)
        finally:await a.shutdown()
    asyncio.run(exercise())

@pytest.mark.parametrize('provider',['openai','anthropic'])
def test_provider_projection_strips_metadata_and_preserves_tool_pair_and_code(tmp_path,provider):
    a=loop(tmp_path,provider=provider); file=tmp_path/'C1.py'; file.write_text('value = "C1"\n',encoding='utf-8'); before=file.read_bytes()
    user=human(a,'Inspect the C1 field in the current task.')
    if a.use_openai:
        pair=[{'role':'assistant','tool_calls':[{'id':'read-1','type':'function','function':{'name':'read_file','arguments':'{"file_path":"C1.py"}'}}]},
            {'role':'tool','tool_call_id':'read-1','content':'value = "C1"'}]
    else:
        pair=[{'role':'assistant','content':[{'type':'tool_use','id':'read-1','name':'read_file','input':{'file_path':'C1.py'}}]},
            {'role':'user','content':[{'type':'tool_result','tool_use_id':'read-1','content':'value = "C1"'}],
             '_source_user_text':None,'_source_task_id':'stage-one'}]
    set_messages(a,[user,*copy.deepcopy(pair)])
    a._repair_task=replace(a._repair_task,task_id='stage-two',canonical_issue='Implement new behavior.')
    a._repair_state=RepairState(a._repair_task,'bytes'); a._project_task_epoch('stage-two')
    assert messages(a)[-2:]==pair and file.read_bytes()==before
    wire=a._openai_request_messages() if a.use_openai else a._anthropic_request_messages()
    assert all('_source_task_id' not in m and '_source_user_text' not in m for m in wire)
    if a.use_openai:
        assert wire[-2]['tool_calls']==pair[0]['tool_calls'] and wire[-1]['tool_call_id']=='read-1'
    else:
        uses=[b for m in wire for b in m.get('content',[]) if isinstance(b,dict) and b.get('type')=='tool_use']
        results=[b for m in wire for b in m.get('content',[]) if isinstance(b,dict) and b.get('type')=='tool_result']
        assert uses==pair[0]['content'] and results==pair[1]['content']
    asyncio.run(a.shutdown())

def test_native_task_transition_archives_observation_but_starts_new_gate(tmp_path,monkeypatch):
    a=loop(tmp_path); captured={}; a._repo_state_probe=lambda:'bytes'
    async def baseline(name,inp):return ShellResult(stdout='a'*40,exit_code=0,execution_started=True)
    monkeypatch.setattr(a,'_execute_tool_call_impl',baseline)
    async def scripted(prompt):
        messages(a).append(human(a,prompt)); r=a._repair_state
        r.register_acceptance_case('AC1',a._repair_task.canonical_issue,a._repair_task.canonical_issue)
        if not captured:
            e=r.record_evidence(tool='run_shell',status='ok',exit_code=0,input_summary='python -m pytest',repo_before='bytes',repo_after='bytes',
                output_preview='Prior green observation.',test_pass_observed=True,execution_started=True)
            r.record_verification(e.id,['AC1'],scope='full',oracle='existing_test',runner_recognized=True)
            a._observations[e.id]='Prior green observation.'; captured.update(id=e.id,execution=r.execution_id)
        else:assert not r.completion_decision().complete and not r.verifications
    monkeypatch.setattr(a,'_chat_openai',scripted)
    async def exercise():
        try:
            await a.run_once(a._repair_task.canonical_issue)
            a._repair_task=replace(a._repair_task,task_id='stage-two',canonical_issue='Implement the new public feature.')
            await a.run_once(a._repair_task.canonical_issue)
            text=a._read_observation({'evidence_id':captured['id']})
            assert 'source_task_id=stage-one' in text and 'current_verification=false' in text
            view=a._historical_observation_view(); assert view['current_verification'] is False
            assert view['source_refs'][0]['source_execution_id']==captured['execution']
            assert not a._repair_state.completion_decision().complete and not a._repair_state.verifications
            assert 'Complete the public task.' not in user_texts(a) and a._operation_counts['model_calls']==0
        finally:await a.shutdown()
    asyncio.run(exercise())

def test_stale_hint_enters_string_and_block_wire_immediately_then_disappears_after_real_rerun(tmp_path,monkeypatch):
    work=tmp_path/'work'; work.mkdir()
    for name,body in {'helper.py':'def answer():\n    return 7\n','test_public.py':'from helper import answer\ndef test_answer():\n    assert answer() == 7\n',
        'pytest.ini':'[pytest]\naddopts = --rootdir=. -p no:cacheprovider\n','.gitignore':'__pycache__/\n.pytest_cache/\nexperiment/\n'}.items():
        (work/name).write_text(body,encoding='utf-8',newline='')
    for args in (['init','-q'],['config','core.autocrlf','false'],['add','.'],['-c','user.name=Test','-c','user.email=test@localhost','-c','commit.gpgsign=false','commit','-qm','Public fixture']):
        subprocess.run(['git',*args],cwd=work,capture_output=True,check=True)
    monkeypatch.chdir(work); monkeypatch.setenv('PATH',str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1'); monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')
    monkeypatch.setenv('PYTEST_ADDOPTS','--rootdir=. -p no:cacheprovider')
    a=loop(tmp_path,repo_root=work,kind=EvidenceFoldingLoop,permission_mode='bypassPermissions'); a.tools=get_solver_tool_definitions('B0')
    a._repair_state.register_acceptance_case('AC1','Complete the public task.','Complete the public task.')
    formal={'command':'python -B -m pytest test_public.py -q -c pytest.ini','acceptance_ids':['AC1'],'scope':'full','oracle':'existing_test'}
    async def exercise():
        try:
            await a._execute_tool_call('run_shell',formal); old=a._repair_state.evidence[-1]
            await a._execute_tool_call('write_file',{'file_path':'draft_note.py','content':'# transient local observation\n'})
            await a._execute_tool_call('run_shell',{'command':python_command("from pathlib import Path; Path('draft_note.py').unlink()",[],str(work),sys.executable)})
            assert not a._repair_state.completion_decision().complete and a._fold_count==0
            for provider in ('openai','anthropic'):
                a.use_openai=provider=='openai'
                for content in ('Read the current code.',[{'type':'text','text':'Read the current code.'}]):
                    set_messages(a,[{'role':'user','content':content}]); wire=a._openai_request_messages() if a.use_openai else a._anthropic_request_messages()
                    text=json.dumps(wire); assert old.id in text and 'runtime-required-reverification' in text
            a.use_openai=True; await a._execute_tool_call('run_shell',formal)
            assert a._repair_state.completion_decision().complete
            for provider in ('openai','anthropic'):
                a.use_openai=provider=='openai'; set_messages(a,[{'role':'user','content':'Current checks recorded.'}])
                wire=a._openai_request_messages() if a.use_openai else a._anthropic_request_messages()
                assert 'runtime-required-reverification' not in json.dumps(wire)
            assert a._operation_counts['model_calls']==0
        finally:await a.shutdown()
    asyncio.run(exercise())
