import asyncio
import json
from pathlib import Path
import sys
from dataclasses import replace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.online_skill_evolution import extract_online_skill_candidate
from agents.agent import Agent
from agents.repair_state import EvidenceReceipt, RepairState, TaskEnvelope


def test_execution_skill_requires_known_receipts():
    evidence=[{'id':'case:e001','status':'error','test_failure_observed':True},
              {'id':'case:e002','status':'ok','test_pass_observed':True}]
    async def query(system,payload):
        assert json.loads(payload)['execution_evidence']==evidence
        return json.dumps({'skills':[{'name':'validation','description':'Validate options before processing',
                   'instructions':'Validate options before consuming even an empty iterable.',
                   'source':'execution','evidence_ids':['case:e001','case:e002']}]})
    candidate=asyncio.run(extract_online_skill_candidate(messages=[{'role':'user','content':'Fix configuration validation.'}],
                      side_query=query,execution_evidence=evidence))
    assert candidate is not None and candidate.evidence_ids==['case:e001','case:e002']
    async def guessed(system,payload):
        return json.dumps({'skills':[{'name':'guess','description':'Guess','instructions':'Always assume tests pass.',
                   'source':'execution','evidence_ids':['made-up-receipt']}]})
    with pytest.raises(ValueError, match='unregistered receipt'):
        asyncio.run(extract_online_skill_candidate(messages=[{'role':'user','content':'Fix validation.'}],
                    side_query=guessed,execution_evidence=evidence))


def test_failed_execution_survives_without_final_answer(tmp_path):
    agent=Agent(api_base='http://unused.invalid',api_key='test',custom_system_prompt='fixed',
                custom_tools=[],is_sub_agent=True)
    state=RepairState(TaskEnvelope('task-a','Fix invalid control options.',str(tmp_path)),None)
    failure=EvidenceReceipt(id='e1',sequence=1,tool='run_shell',status='error',exit_code=1,
            duration_ms=1,repo_before=None,repo_after=None,repo_generation=0,
            input_summary='pytest test_control.py',operation_key='op1',output_preview='1 failed',
            check_fingerprint='trusted-check',test_failure_observed=True,
            test_cases=(('test_control::test_empty','failure','invalid option accepted'),))
    state._evidence=[failure,replace(failure,id='e2',sequence=2,diagnostic=True),
                     replace(failure,id='e3',sequence=3,check_fingerprint=None)]
    agent._repair_state=state
    agent._set_pending_skill_extraction_window(original_user_message='Fix invalid control options.',
                        assistant_text='',retrieved_reference=None)
    window=agent._pop_pending_skill_extraction_window('')
    assert window is not None and window['latest_assistant']==''
    assert [row['id'] for row in window['execution_evidence']]==[f'{agent.session_id}:{state.execution_id}:task-a:e1']
    assert window['execution_evidence'][0]['test_failure_observed']


@pytest.mark.parametrize('ambiguous', [False, True])
def test_original_receipt_reference_requires_unique_trusted_mapping(ambiguous):
    """A real TRAIN extraction cited the receipt shown in dialogue instead of its export scope."""
    evidence = [{'id': 'run-a:task:execution:e24', 'original_evidence_id': 'execution:e24'}]
    if ambiguous:
        evidence.append({'id': 'run-b:task:execution:e24', 'original_evidence_id': 'execution:e24'})
    async def query(system, payload):
        return json.dumps({'skills': [{'name': 'document-template', 'description': 'Fill document templates',
            'instructions': 'Inspect saved headers and footers for unresolved tokens.',
            'source': 'execution', 'evidence_ids': ['execution:e24'], 'claim_kind': 'procedure'}]})
    call = extract_online_skill_candidate(messages=[], side_query=query, execution_evidence=evidence)
    if ambiguous:
        with pytest.raises(ValueError, match='unregistered receipt'):
            asyncio.run(call)
    else:
        candidate = asyncio.run(call)
        assert candidate.evidence_ids == ['run-a:task:execution:e24']


def test_receipt_alias_does_not_certify_a_repair():
    async def query(system, payload):
        return json.dumps({'skills': [{'name': 'repair', 'description': 'Repair template traversal',
            'instructions': 'Traverse headers.', 'source': 'execution',
            'evidence_ids': ['execution:e24'], 'claim_kind': 'check_repair'}]})
    with pytest.raises(ValueError, match='matching trusted transition'):
        asyncio.run(extract_online_skill_candidate(messages=[], side_query=query,
            execution_evidence=[{'id': 'run-a:execution:e24', 'original_evidence_id': 'execution:e24'}]))
