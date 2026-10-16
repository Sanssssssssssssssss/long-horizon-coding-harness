"""Receipt authority, freshness and inherited folding boundaries; no live model calls."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

import agents.agent as agent_module
from agents.repair_state import RepairState, TaskEnvelope
from agents.telemetry import TraceRecorder, read_trace
from agents.tool_result import PytestReport, ShellResult
from benchmarks.context_folding import _FoldingLoop
from benchmarks.folding_checkpoint import EvidenceFoldingLoop, render_checkpoint


def agent(tmp_path, *, loop_class=EvidenceFoldingLoop, **options):
    return loop_class(
        token_budget={'limit': None, 'charged': 0}, api_base='https://example.invalid/v1',
        api_key='test', model='test', custom_system_prompt='fixed instructions', custom_tools=[],
        skills_enabled=False, is_sub_agent=True, max_input_tokens_per_call=32768,
        max_output_tokens=8192, max_model_calls=6,
        repair_task=TaskEnvelope('task', 'fix value', '/repo'),
        trace=TraceRecorder(tmp_path, run_id='run', task_id='task', attempt_id='attempt'),
        **options,
    )


def state():
    repair = RepairState(TaskEnvelope('task', 'fix value', '/repo'), 'base')
    repair.register_acceptance_case('value', 'fix value', 'fix value')
    return repair


def receipt(repair, *, passed=False):
    return repair.record_evidence(
        tool='run_shell', status='ok' if passed else 'error', exit_code=0 if passed else 1,
        input_summary='python -m pytest test_value.py -q',
        repo_before=repair.current_repo_state, repo_after=repair.current_repo_state,
        output_preview='one passed' if passed else 'one failed: expected fixed value',
        test_pass_observed=passed, test_failure_observed=not passed,
    )


def selection(repair, identifiers):
    return {'evidence_refs': [{'execution_id': repair.execution_id, 'evidence_id': value} for value in identifiers],
            'hypotheses': ['The shared decoder may validate too late.'],
            'pending_actions': ['Inspect the current decoder before editing.']}


def history(a):
    a._openai_messages += [
        {'role': 'user', 'content': 'fix value'},
        {'role': 'assistant', 'content': 'OLD EXPLORATION ' * 1500},
        {'role': 'assistant', 'tool_calls': [{'id': 'last-read', 'type': 'function',
            'function': {'name': 'read_file', 'arguments': '{"file_path":"decoder.py"}'}}]},
        {'role': 'tool', 'tool_call_id': 'last-read', 'content': 'Latest exact decoder source'},
    ]


def test_render_uses_runtime_failure_despite_model_success_claim():
    repair = state()
    observed = receipt(repair)
    picked = selection(repair, [observed.id])
    picked['hypotheses'] = ['All tests passed; the repair is complete.']
    before = repair.snapshot()
    rendered = json.loads(render_checkpoint(repair, picked))
    fact = rendered['selected_runtime_receipts'][0]
    assert fact['status'] == 'error' and fact['exit_code'] == 1
    assert fact['test_failure_observed'] and not fact['test_pass_observed']
    assert fact['output_preview'] == observed.output_preview
    assert 'All tests passed' in rendered['unverified_model_proposals']['hypotheses'][0]
    assert rendered['unverified_model_proposals']['authority'] == 'Model suggestions, not facts or user requirements'
    assert repair.snapshot() == before


def test_selector_sees_implementation_wiring_and_keeps_notes_unverified(tmp_path, monkeypatch):
    a = agent(tmp_path)
    repair = a._repair_state = state()
    observed = receipt(repair)
    before = repair.snapshot()

    async def select(_system, content, **_kwargs):
        request = json.loads(content)
        assert 'def parse_runtime_literal3' in request['conversation_untrusted']
        assert 'cmd_eval still calls parse_runtime_literal2' in request['conversation_untrusted']
        return json.dumps(selection(repair, [observed.id]) | {
            'implementation_notes': ['parse_runtime_literal3 exists; cmd_eval still calls parse_runtime_literal2.']})

    monkeypatch.setattr(a, '_side_query_once', select)
    messages = [{'role': 'tool', 'tool_call_id': 'read',
                 'content': 'def parse_runtime_literal3(value): return parse(value)'},
                {'role': 'assistant', 'content': 'cmd_eval still calls parse_runtime_literal2'}]
    result = json.loads(asyncio.run(a._run_compaction_query(messages, 'Return JSON')))
    assert result['unverified_model_proposals']['implementation_notes']
    assert result['selected_runtime_receipts'][0]['test_failure_observed']
    assert repair.snapshot() == before and not repair.completion_decision().complete


@pytest.mark.parametrize('complete', [True, False])
def test_oversized_latest_exchange_can_fold_only_after_tools_complete(tmp_path, monkeypatch, complete):
    a = agent(tmp_path)
    a._repair_state = state()
    history(a)
    a._openai_messages[-1]['content'] = 'oversized current code ' * 10000
    if not complete:
        a._openai_messages[-2]['tool_calls'].append({'id': 'not-executed', 'type': 'function',
            'function': {'name': 'edit_file', 'arguments': '{}'}})

    async def select(_system, content, **_kwargs):
        request = json.loads(content)
        assert len(request['conversation_untrusted'].encode('utf-8')) <= 64000
        return json.dumps(selection(a._repair_state, []))

    monkeypatch.setattr(a, '_side_query_once', select)
    before = copy.deepcopy(a._openai_messages)
    folded = asyncio.run(a._compact_conversation(trigger='auto'))
    assert folded is complete
    if complete:
        assert a._conversation_input_estimate(a._openai_messages) <= a.context_window
        assert not any(m.get('role') == 'tool' for m in a._openai_messages)
        assert a._fold_count == 1
    else:
        assert a._openai_messages == before


def test_old_green_remains_historical_after_repository_returns_to_same_bytes():
    repair = state()
    green = receipt(repair, passed=True)
    repair.record_verification(green.id, ['value'], scope='focused', oracle='existing_test', runner_recognized=True)
    repair.observe_repo_state('candidate')
    repair.observe_repo_state('base')
    rendered = json.loads(render_checkpoint(repair, selection(repair, [green.id])))
    assert rendered['repo_generation'] == 2 and rendered['current_repo_state'] == 'base'
    assert rendered['selected_runtime_receipts'][0]['state_scope'] == 'historical_only'
    assert not rendered['recorded_verifications'][0]['current_verification']


@pytest.mark.parametrize('mutation', ['unknown_id', 'other_execution', 'new_fact', 'new_state',
                                      'reference_state', 'duplicate', 'too_many', 'claimed_proposal'])
def test_selection_rejects_unbound_or_fabricated_fields(mutation):
    repair = state()
    observed = receipt(repair)
    picked = selection(repair, [observed.id])
    if mutation == 'unknown_id':
        picked['evidence_refs'][0]['evidence_id'] = 'e9999'
    elif mutation == 'other_execution':
        picked['evidence_refs'][0]['execution_id'] = state().execution_id
    elif mutation == 'new_fact':
        picked['facts'] = {'tests_passed': True}
    elif mutation == 'new_state':
        picked['repo_generation'] = 900
    elif mutation == 'reference_state':
        picked['evidence_refs'][0]['status'] = 'ok'
    elif mutation == 'duplicate':
        picked['evidence_refs'] *= 2
    elif mutation == 'too_many':
        picked['evidence_refs'] *= 13
    else:
        picked['hypotheses'] = [{'claim': 'fixed', 'verified': True}]
    with pytest.raises(ValueError):
        render_checkpoint(repair, picked)


def test_side_query_charges_usage_and_does_not_consume_a_solve_decision(tmp_path):
    a = agent(tmp_path)
    a._repair_state = state()
    observed = receipt(a._repair_state)
    calls = []

    async def create(**params):
        calls.append(params)
        return NS(choices=[NS(message=NS(content=json.dumps(selection(a._repair_state, [observed.id]))),
                             finish_reason='stop')],
                  usage=NS(prompt_tokens=100, completion_tokens=25, total_tokens=125,
                           prompt_tokens_details=None, completion_tokens_details=None, cost=0.01))

    a._openai_client = NS(chat=NS(completions=NS(create=create)))
    rendered = asyncio.run(a._run_compaction_query([], 'Return JSON'))
    assert json.loads(rendered)['selected_runtime_receipts'][0]['evidence_id'] == observed.id
    assert calls[0]['max_tokens'] == 2048 and calls[0]['messages'][0]['role'] == 'system'
    assert a._operation_counts['model_calls'] == 1 and a.decision_clock['completed'] == 0
    assert a.total_input_tokens == 100 and a.total_output_tokens == 25
    assert a.total_provider_reported_cost == pytest.approx(0.01)
    assert a.token_budget['charged'] == 125
    ended = [row for row in read_trace(a.trace.trace_path) if row['event_type'] == 'model.call.end']
    assert ended[0]['purpose'] == 'session_fold_evidence'
    assert ended[0]['metrics']['provider_usage']['total_tokens'] == 125


@pytest.mark.parametrize('prior_scope', ['current', 'other_execution', 'other_task', 'oversized'])
def test_selector_carries_only_current_task_proposals_not_prior_receipt_facts(tmp_path, monkeypatch, prior_scope):
    a = agent(tmp_path)
    repair = a._repair_state = state()
    observed = receipt(repair)
    previous = json.loads(render_checkpoint(repair, selection(repair, [observed.id])))
    # An evicted or foreign receipt in a checkpoint must not expand the next catalog.
    previous['selected_runtime_receipts'] = [{'evidence_id': 'old-execution:e0001',
                                            'output_preview': 'OLD FACT MUST NOT ENTER SELECTOR'}]
    if prior_scope == 'other_execution':
        previous['execution_id'] = 'old-execution'
    elif prior_scope == 'other_task':
        previous['canonical_task_sha256'] = 'another-task'
    elif prior_scope == 'oversized':
        previous['unverified_model_proposals']['pending_actions'] = ['x' * 401]
    before = repair.snapshot()

    async def select(_system, content, **_kwargs):
        request = json.loads(content)
        assert 'old-execution:e0001' not in content and 'OLD FACT' not in content
        assert {r['evidence_id'] for r in request['receipt_catalog']} == {observed.id}
        proposals = request['previous_proposals_untrusted']
        if prior_scope == 'current':
            assert proposals['pending_actions'] == ['Inspect the current decoder before editing.']
        else:
            assert proposals is None
        return json.dumps(selection(repair, [observed.id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    rendered = json.loads(asyncio.run(a._run_compaction_query([], 'Return JSON', json.dumps(previous))))
    assert rendered['selected_runtime_receipts'][0]['evidence_id'] == observed.id
    assert rendered['selected_runtime_receipts'][0]['test_failure_observed']
    assert not repair.completion_decision().complete and repair.snapshot() == before


def test_actual_fold_preserves_canonical_task_and_complete_recent_tool_exchange(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path)
    a._repair_state = state()
    observed = receipt(a._repair_state)
    history(a)
    original_tail = copy.deepcopy(a._openai_messages[-2:])

    async def select(*_args, **_kwargs):
        return json.dumps(selection(a._repair_state, [observed.id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    assert asyncio.run(a._compact_conversation(trigger='periodic'))
    assert a._openai_messages[1]['content'] == 'fix value'
    assert a._openai_messages[-2:] == original_tail
    assert 'historical_only' not in a._openai_messages[2]['content']
    assert 'OLD EXPLORATION' not in json.dumps(a._openai_messages)
    assert a._fold_count == 1 and a._repair_state.evidence == (observed,)


def test_invalid_selection_preserves_raw_history_and_final_solve_call(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path)
    a.max_model_calls = 2
    a._repair_state = state()
    receipt(a._repair_state)
    history(a)
    original = copy.deepcopy(a._openai_messages)
    calls = []

    async def select(*_args, **kwargs):
        calls.append(kwargs)
        a._operation_counts['model_calls'] += 1
        return '{"facts":{"all_tests_passed":true}}'

    monkeypatch.setattr(a, '_side_query_once', select)
    assert not asyncio.run(a._compact_conversation(trigger='periodic'))
    assert a._openai_messages == original and a._fold_count == 0
    assert len(calls) == 1 and a._operation_counts['model_calls'] == 1
    assert calls[0]['max_tokens'] == 2048


def test_receipt_selection_fails_if_repository_changes_during_query(tmp_path, monkeypatch):
    a = agent(tmp_path)
    a._repair_state = state()
    observed = receipt(a._repair_state)

    async def select(*_args, **_kwargs):
        picked = selection(a._repair_state, [observed.id])
        a._repair_state.observe_repo_state('external-change')
        return json.dumps(picked)

    monkeypatch.setattr(a, '_side_query_once', select)
    with pytest.raises(ValueError, match='runtime state changed'):
        asyncio.run(a._run_compaction_query([], 'Return JSON'))


def test_no_repair_state_uses_plain_fallback_with_fixed_budget_and_trace_scope(tmp_path, monkeypatch):
    a = agent(tmp_path)
    a._repair_state = None
    a._openai_messages.append({'role': 'user', 'content': 'fix value'})
    calls = []

    async def query(system, text, **kwargs):
        calls.append((system, text, kwargs))
        return 'The decoder cause is unresolved; inspect the source before choosing a change.'

    monkeypatch.setattr(a, '_side_query_once', query)
    summary = asyncio.run(a._run_compaction_query([{'role': 'assistant', 'content': 'unresolved investigation'}],
                                                 'Return checkpoint'))
    assert 'unresolved' in summary and 'selected_runtime_receipts' not in summary
    assert '<canonical-task>\nfix value' in calls[0][1]
    assert calls[0][2]['max_tokens'] == 2048
    assert calls[0][2]['purpose'] == 'session_fold_plain_fallback'
    assert read_trace(a.trace.trace_path)[-1]['attributes']['scope'] == 'plain_fallback_no_repair_state'


def add_reads(repair, count):
    for index in range(count):
        repair.record_evidence(tool='read_file', status='ok', input_summary=f'helper_{index}.py',
                               output_preview=f'Unrelated helper {index}',
                               repo_before=repair.current_repo_state, repo_after=repair.current_repo_state)


@pytest.mark.parametrize('hypothesis_status', ['active', 'supported'])
def test_old_cause_and_unresolved_witness_survive_more_than_64_observations(tmp_path, monkeypatch, hypothesis_status):
    a = agent(tmp_path)
    repair = a._repair_state = state()
    cause, witness = receipt(repair), receipt(repair)
    repair.upsert_hypothesis('cause', 'Validation happens after the empty return.',
                            'Empty invalid options fail validation.', 'Empty invalid options pass validation.',
                            status=hypothesis_status, evidence_ids=[cause.id])
    repair.record_verification(witness.id, ['value'], scope='focused', oracle='existing_test', runner_recognized=True)
    repair.observe_repo_state('candidate')
    add_reads(repair, 70)
    assert cause not in repair.evidence[-64:] and witness not in repair.evidence[-64:]
    requests = []

    async def select(_system, text, **_kwargs):
        requests.append(json.loads(text))
        return json.dumps(selection(repair, [cause.id, witness.id, repair.evidence[-1].id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    rendered = json.loads(asyncio.run(a._run_compaction_query([], 'Return JSON')))
    catalog = requests[0]['receipt_catalog']
    assert len(catalog) == 64 and {cause.id, witness.id}.issubset({row['evidence_id'] for row in catalog})
    assert {row['evidence_id'] for row in requests[0]['required_evidence_refs']} == {cause.id, witness.id}
    assert rendered['pinned_evidence_ids'] == [cause.id, witness.id]
    assert [row['state_scope'] for row in rendered['selected_runtime_receipts']] == [
        'historical_only', 'historical_only', 'current_repository']
    assert not rendered['recorded_verifications'][0]['current_verification']


def test_missing_obligation_pin_refuses_fold_and_preserves_raw_history(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path)
    repair = a._repair_state = state()
    cause = receipt(repair)
    repair.upsert_hypothesis('cause', 'Validation is late.', 'The empty test fails.', 'The empty test passes.',
                            evidence_ids=[cause.id])
    add_reads(repair, 70)
    history(a)
    original = copy.deepcopy(a._openai_messages)
    calls = []

    async def select(*_args, **_kwargs):
        calls.append(True)
        return json.dumps(selection(repair, [repair.evidence[-1].id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    assert not asyncio.run(a._compact_conversation(trigger='periodic'))
    assert a._openai_messages == original and a._fold_count == 0
    assert len(calls) == 2
    with pytest.raises(ValueError, match='omitted required'):
        render_checkpoint(repair, selection(repair, [repair.evidence[-1].id]))


def test_more_than_12_pins_preserves_history_before_any_summary_call(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path)
    repair = a._repair_state = state()
    observed = [receipt(repair) for _ in range(13)]
    repair.upsert_hypothesis('cause', 'Several cases may share a cause.', 'The cases fail together.',
                            'The cases fail independently.', evidence_ids=[row.id for row in observed])
    history(a)
    original = copy.deepcopy(a._openai_messages)

    async def forbidden(*_args, **_kwargs):
        pytest.fail('An oversized obligation set must not spend a summary call')

    monkeypatch.setattr(a, '_side_query_once', forbidden)
    assert not asyncio.run(a._compact_conversation(trigger='periodic'))
    assert a._openai_messages == original and a._fold_count == 0
    assert a._operation_counts['model_calls'] == 0
    events = [row for row in read_trace(a.trace.trace_path) if row['event_type'] == 'context.checkpoint']
    assert events and all(row['attributes']['scope'] == 'pinned_capacity_exceeded' for row in events)
    assert len(events[0]['attributes']['required_ids']) == 13


def test_rejected_hypothesis_does_not_pin_obsolete_evidence(tmp_path, monkeypatch):
    a = agent(tmp_path)
    repair = a._repair_state = state()
    obsolete = receipt(repair)
    repair.upsert_hypothesis('cause', 'Validation is late.', 'The empty test fails.', 'The empty test passes.',
                            status='rejected', evidence_ids=[obsolete.id])
    add_reads(repair, 70)

    async def select(_system, text, **_kwargs):
        request = json.loads(text)
        assert request['required_evidence_refs'] == []
        assert obsolete.id not in {row['evidence_id'] for row in request['receipt_catalog']}
        return json.dumps(selection(repair, [repair.evidence[-1].id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    assert json.loads(asyncio.run(a._run_compaction_query([], 'Return JSON')))['pinned_evidence_ids'] == []


def test_unresolved_verification_pins_without_a_hypothesis(tmp_path, monkeypatch):
    a = agent(tmp_path)
    repair = a._repair_state = state()
    failed = receipt(repair)
    repair.record_verification(failed.id, ['value'], scope='focused', oracle='existing_test', runner_recognized=True)
    add_reads(repair, 70)

    async def select(_system, text, **_kwargs):
        request = json.loads(text)
        assert request['unverified_existing_hypotheses'] == []
        assert request['required_evidence_refs'][0]['evidence_id'] == failed.id
        return json.dumps(selection(repair, [failed.id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    assert json.loads(asyncio.run(a._run_compaction_query([], 'Return JSON')))['pinned_evidence_ids'] == [failed.id]


def test_changed_obligation_set_during_query_refuses_commit(tmp_path, monkeypatch):
    a = agent(tmp_path)
    repair = a._repair_state = state()
    first, second = receipt(repair), receipt(repair)
    repair.upsert_hypothesis('first', 'The first cause is possible.', 'First prediction.', 'First falsifier.',
                            evidence_ids=[first.id])

    async def select(*_args, **_kwargs):
        repair.upsert_hypothesis('second', 'A second cause is possible.', 'Second prediction.', 'Second falsifier.',
                                evidence_ids=[second.id])
        return json.dumps(selection(repair, [first.id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    with pytest.raises(ValueError, match='runtime state changed'):
        asyncio.run(a._run_compaction_query([], 'Return JSON'))


def short_fold_history(a, provider):
    a.use_openai = provider == 'openai'
    if a.use_openai:
        history(a)
        a._openai_messages[2]['content'] = 'Archived observation. ' * 150
    else:
        a._anthropic_messages = [
            {'role': 'user', 'content': 'fix value'},
            {'role': 'assistant', 'content': 'Archived observation. ' * 150},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'last-read',
                'name': 'read_file', 'input': {'file_path': 'decoder.py'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'last-read',
                'content': 'Latest exact decoder source'}]},
        ]


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
def test_minimum_checkpoint_cost_skips_selector_without_poisoning_cache(tmp_path, monkeypatch, provider):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path)
    repair = a._repair_state = state()
    pinned = [receipt(repair) for _ in range(12)]
    repair.upsert_hypothesis('cause', 'Several observations constrain this change.',
                            'All constraints hold.', 'A constraint is contradicted.',
                            evidence_ids=[row.id for row in pinned])
    short_fold_history(a, provider)
    before = copy.deepcopy(a._openai_messages if a.use_openai else a._anthropic_messages)
    before_state = repair.snapshot()
    a._failed_compaction_signature = 'an earlier failure'

    async def forbidden(*_args, **_kwargs):
        pytest.fail('A minimum checkpoint larger than raw history must not call a selector')

    monkeypatch.setattr(a, '_side_query_once', forbidden)
    assert not asyncio.run(a._compact_conversation(trigger='periodic'))
    assert (a._openai_messages if a.use_openai else a._anthropic_messages) == before
    assert repair.snapshot() == before_state and a._fold_count == 0
    assert a._failed_compaction_signature == 'an earlier failure'
    assert a._operation_counts['model_calls'] == 0 and a.token_budget['charged'] == 0
    events = [row for row in read_trace(a.trace.trace_path) if row['event_type'] == 'context.checkpoint']
    assert events[-1]['attributes']['scope'] == 'minimum_checkpoint_not_smaller'
    assert events[-1]['attributes']['candidate_reservation'] >= events[-1]['attributes']['raw_reservation']


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
@pytest.mark.parametrize('trigger', ['periodic', 'manual'])
def test_growing_selected_checkpoint_preserves_raw_automatic_history(tmp_path, monkeypatch, provider, trigger):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path)
    repair = a._repair_state = state()
    observed = [receipt(repair) for _ in range(12)]
    short_fold_history(a, provider)
    before = copy.deepcopy(a._openai_messages if a.use_openai else a._anthropic_messages)
    calls = []

    async def select(*_args, **_kwargs):
        calls.append(True)
        return json.dumps(selection(repair, [row.id for row in observed]))

    monkeypatch.setattr(a, '_side_query_once', select)
    compacted = asyncio.run(a._compact_conversation(trigger=trigger))
    assert calls == [True]
    if trigger == 'periodic':
        assert not compacted
        assert (a._openai_messages if a.use_openai else a._anthropic_messages) == before
        assert a._fold_count == 0 and a._failed_compaction_signature is None
        events = [row for row in read_trace(a.trace.trace_path) if row['event_type'] == 'context.checkpoint']
        assert events[-1]['attributes']['scope'] == 'selected_checkpoint_not_smaller'
    else:
        assert compacted and a._fold_count == 1


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
def test_cost_rejected_periodic_selection_does_not_block_budget_fold(tmp_path, monkeypatch, provider):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path)
    repair = a._repair_state = state()
    observed = [receipt(repair) for _ in range(12)]
    short_fold_history(a, provider)
    calls = []

    async def select(*_args, **_kwargs):
        calls.append(True)
        return json.dumps(selection(repair, [row.id for row in observed] if len(calls) == 1 else [observed[0].id]))

    monkeypatch.setattr(a, '_side_query_once', select)
    assert not asyncio.run(a._compact_conversation(trigger='periodic'))
    assert a._failed_compaction_signature is None and a._fold_count == 0
    assert asyncio.run(a._compact_conversation(trigger='budget'))
    assert len(calls) == 2 and a._fold_count == 1
    assert a._repair_state is repair and a._failed_compaction_signature is None


def lifecycle_agent(tmp_path, monkeypatch, *, with_trace=True, loop_class=EvidenceFoldingLoop):
    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.chdir(repo)
    a = agent(tmp_path / 'trace', loop_class=loop_class)
    if not with_trace:
        a.trace = None
    a._repair_task = TaskEnvelope('first', 'fix value', str(repo), requires_patch=False)

    async def probe():
        return 'base'

    async def bootstrap(*_args, **_kwargs):
        return ShellResult(stdout='a' * 40, exit_code=0)

    async def forbidden(*_args, **_kwargs):
        pytest.fail('Observation lifecycle regression must not call a provider')

    monkeypatch.setattr(a, '_probe_repo_state', probe)
    monkeypatch.setattr(a, '_execute_tool_call_impl', bootstrap)
    monkeypatch.setattr(a._openai_client.chat.completions, 'create', forbidden)
    return a, repo


def observed_green(a):
    a._repair_state.register_acceptance_case('value', 'fix value', 'fix value')
    command = 'python -m pytest test_value.py -q'
    output = 'FIRST\n' + 'X' * 20000 + '\nLAST'
    identifier, error = a._record_repair_evidence(
        'run_shell', {'command': command, 'acceptance_ids': ['value'], 'scope': 'focused', 'oracle': 'existing_test'},
        ShellResult(stdout=output, exit_code=0, pytest_report=PytestReport(1)),
        failed=False, duration_ms=0, before='base', after='base', check_fingerprint='unchanged-check',
    )
    assert error is None and a._repair_state.completion_decision().complete
    return identifier, command


@pytest.mark.parametrize('with_trace', [False, True])
@pytest.mark.parametrize('loop_class', [_FoldingLoop, EvidenceFoldingLoop])
def test_prior_execution_observation_survives_real_two_turn_initialization(tmp_path, monkeypatch, with_trace, loop_class):
    a, repo = lifecycle_agent(tmp_path, monkeypatch, with_trace=with_trace, loop_class=loop_class)
    seen = {}

    async def scripted_chat(_prompt):
        repair = a._repair_state
        if not seen:
            identifier, command = observed_green(a)
            seen.update(id=identifier, command=command, execution=repair.execution_id)
            return
        repair.register_acceptance_case('value', 'fix value', 'fix value')
        assert repair.execution_id != seen['execution']
        before = repair.snapshot()
        for part in ('input', 'output'):
            page = a._read_observation({'evidence_id': seen['id'], 'part': part})
            assert 'historical_scope=prior_execution' in page
            assert 'current_verification=false' in page
            assert seen['execution'] in page and repair.execution_id in page
            assert (seen['command'] if part == 'input' else 'FIRST') in page
        middle = a._read_observation({'evidence_id': seen['id'], 'offset': 1, 'limit': 1, 'char_offset': 16000})
        assert 'next_char_offset=None' in middle and 'X' * 3000 in middle
        assert 'char_offset' in a._read_observation({'evidence_id': seen['id'], 'char_offset': -1})
        assert 'part must be input or output' in a._read_observation({'evidence_id': seen['id'], 'part': []})
        assert repair.snapshot() == before and not repair.verifications
        assert not repair.completion_decision().complete
        with pytest.raises(ValueError, match='unknown'):
            repair.upsert_hypothesis('old', 'claim', 'prediction', 'falsifier', evidence_ids=[seen['id']])
        with pytest.raises((KeyError, ValueError)):
            repair.record_verification(seen['id'], ['value'], scope='focused', oracle='existing_test', runner_recognized=True)
        current, _ = a._record_repair_evidence('read_file', {'file_path': 'current.py'}, 'CURRENT_OUTPUT',
            failed=False, duration_ms=0, before='base', after='base')
        assert current != seen['id']
        assert 'CURRENT_OUTPUT' in a._read_observation({'evidence_id': current})
        assert 'historical_scope' not in a._read_observation({'evidence_id': current})

    monkeypatch.setattr(a, '_chat_openai', scripted_chat)

    async def exercise():
        try:
            await a.run_once('fix value')
            a._repair_task = TaskEnvelope('second', 'fix value', str(repo), requires_patch=False)
            await a.run_once('fix value')
            a._repair_task = TaskEnvelope('third', 'fix value', str(repo), requires_patch=False)
            await a.run_once('fix value')
            assert a._operation_counts['model_calls'] == 0
        finally:
            await a.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize('invalidation', ['repo', 'cwd', 'trace', 'trace_id', 'trace_task_id', 'clear', 'internal_clear'])
@pytest.mark.parametrize('loop_class', [_FoldingLoop, EvidenceFoldingLoop])
def test_historical_observation_scope_expires_without_resurrection(tmp_path, monkeypatch, invalidation, loop_class):
    a, repo = lifecycle_agent(tmp_path, monkeypatch, loop_class=loop_class)
    seen = {}

    async def scripted_chat(_prompt):
        if not seen:
            seen['id'], _ = observed_green(a)

    monkeypatch.setattr(a, '_chat_openai', scripted_chat)

    async def exercise():
        try:
            await a.run_once('fix value')
            await a.run_once('fix value')
            assert 'FIRST' in a._read_observation({'evidence_id': seen['id']})
            original_task, original_trace = a._repair_task, a.trace
            original_trace_id = a.trace.trace_id
            original_trace_task_id = a.trace.task_id
            if invalidation == 'repo':
                a._repair_task = TaskEnvelope('elsewhere', 'fix value', str(tmp_path / 'other'))
            elif invalidation == 'cwd':
                monkeypatch.chdir(tmp_path)
            elif invalidation == 'trace':
                a.trace = TraceRecorder(tmp_path / 'other-trace', run_id='other', task_id='other', attempt_id='1')
            elif invalidation == 'trace_id':
                a.trace.trace_id = 'a-new-trace-identity'
            elif invalidation == 'trace_task_id':
                a.trace.task_id = 'a-different-task-chain'
            elif invalidation == 'clear':
                a.clear_history()
            else:
                a._clear_history_keep_system()
            for part in ('input', 'output'):
                assert a._read_observation({'evidence_id': seen['id'], 'part': part}).startswith('Error:')
            a._repair_task, a.trace = original_task, original_trace
            a.trace.trace_id = original_trace_id
            a.trace.task_id = original_trace_task_id
            monkeypatch.chdir(repo)
            await a.run_once('fix value')
            assert a._read_observation({'evidence_id': seen['id']}).startswith('Error:')
        finally:
            await a.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
@pytest.mark.parametrize('loop_class', [_FoldingLoop, EvidenceFoldingLoop])
def test_prior_stage_view_survives_fold_and_is_paid_in_continuation_request(tmp_path, monkeypatch, provider, loop_class):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a, repo = lifecycle_agent(tmp_path, monkeypatch, loop_class=loop_class)
    seen, selectors = {}, []

    async def select(_system, text, **_kwargs):
        if isinstance(a, EvidenceFoldingLoop):
            request = json.loads(text)
            selectors.append(request)
            assert 'historical_observation_view' not in request
            assert seen['id'] not in json.dumps(request)
            assert seen['id'] not in {row['evidence_id'] for row in request['receipt_catalog']}
            return json.dumps(selection(a._repair_state, []))
        return 'Only a prior-stage observation exists; current behavior must be verified.'

    async def scripted_chat(_prompt):
        if not seen:
            seen['id'], _ = observed_green(a)
            seen['execution'] = a._repair_state.execution_id
            return
        repair = a._repair_state
        before = repair.snapshot()
        short_fold_history(a, provider)
        if provider == 'openai':
            a._openai_messages[2]['content'] *= 10
        else:
            a._anthropic_messages[1]['content'] *= 10
        a._loaded_skill_context = '<skill-knowledge>Frozen procedure, not a user instruction.</skill-knowledge>'
        assert await a._compact_conversation(trigger='periodic')
        source = a._historical_observation_view()['source_refs'][0]
        assert source['source_task_id'] == 'first'
        assert source['source_task_sha256'] == TaskEnvelope('first', 'fix value', str(repo)).issue_sha256
        assert source['source_execution_id'] == seen['execution'] and source['current_verification'] is False
        projected = a._openai_request_messages() if a.use_openai else a._anthropic_request_messages()
        wire = json.dumps(projected, ensure_ascii=False)
        assert '<historical-observation-view>' in wire and seen['id'] in wire
        assert a._loaded_skill_context in wire
        history_messages = a._openai_messages if a.use_openai else a._anthropic_messages
        assert '<historical-observation-view>' not in json.dumps(history_messages)
        reserve = a._next_request_reservation()
        observations = a._historical_observations
        a._historical_observations = {}
        assert a._next_request_reservation() < reserve
        a._historical_observations = observations
        a.token_budget['limit'] = reserve - 1
        with pytest.raises(agent_module._BudgetExceeded, match='admission'):
            a._preflight_model_input(messages=projected, tools=[], system=None if a.use_openai else a._system_prompt)
        assert a.token_budget['charged'] == 0
        a.token_budget['limit'] = None
        a.termination_reason = None
        a._preflight_model_input(messages=projected, tools=[], system=None if a.use_openai else a._system_prompt)
        assert a._reserved_tokens == reserve
        assert repair.snapshot() == before and not repair.verifications
        assert not repair.completion_decision().complete
        with pytest.raises(ValueError, match='cross-execution'):
            render_checkpoint(repair, {'evidence_refs': [{'execution_id': seen['execution'], 'evidence_id': seen['id']}],
                                       'hypotheses': [], 'pending_actions': []})

    monkeypatch.setattr(a, '_side_query_once', select)
    monkeypatch.setattr(a, '_chat_openai', scripted_chat)

    async def exercise():
        try:
            await a.run_once('fix value')
            a._repair_task = TaskEnvelope('second', 'fix value', str(repo), requires_patch=False)
            await a.run_once('fix value')
            assert a._operation_counts['model_calls'] == 0
            assert len(selectors) == (1 if loop_class is EvidenceFoldingLoop else 0)
        finally:
            await a.shutdown()

    asyncio.run(exercise())


def test_bounded_prior_obligations_remain_auxiliary_and_do_not_block_new_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a, repo = lifecycle_agent(tmp_path, monkeypatch)
    identifiers = []

    async def select(*_args, **_kwargs):
        return json.dumps(selection(a._repair_state, []))

    async def scripted_chat(_prompt):
        if not identifiers:
            for index in range(20):
                identifier, error = a._record_repair_evidence('read_file', {'file_path': f'实体{index}.py'},
                    '历史观察' * 200, failed=False, duration_ms=0, before='base', after='base')
                assert error is None
                identifiers.append(identifier)
            a._repair_state.upsert_hypothesis('many', 'Prior entities may share a cause.', 'Cases fail together.',
                                             'Cases fail independently.', evidence_ids=identifiers[:12])
            return
        repair = a._repair_state
        view = a._historical_observation_view()
        assert len(view['source_refs']) <= 8
        assert len(json.dumps(view, ensure_ascii=False).encode('utf-8')) <= 4096
        shown = {row['evidence_id'] for row in view['source_refs']}
        assert identifiers[0] in shown and view['omitted_count'] == 20 - len(shown)
        omitted = next(identifier for identifier in identifiers if identifier not in shown)
        assert '历史观察' in a._read_observation({'evidence_id': omitted})
        assert not repair.hypotheses and not repair.verifications
        history(a)
        assert await a._compact_conversation(trigger='periodic')
        assert a._fold_count == 1 and not repair.completion_decision().complete

    monkeypatch.setattr(a, '_side_query_once', select)
    monkeypatch.setattr(a, '_chat_openai', scripted_chat)

    async def exercise():
        try:
            await a.run_once('fix value')
            a._repair_task = TaskEnvelope('second', 'fix value', str(repo), requires_patch=False)
            await a.run_once('fix value')
        finally:
            await a.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize('loop_class', [_FoldingLoop, EvidenceFoldingLoop])
def test_historical_trace_blob_integrity_still_checked(tmp_path, monkeypatch, loop_class):
    a, _repo = lifecycle_agent(tmp_path, monkeypatch, loop_class=loop_class)
    seen = {}

    async def scripted_chat(_prompt):
        if not seen:
            seen['id'], _ = observed_green(a)
            seen['blob'] = dict(a._observations[seen['id']])

    monkeypatch.setattr(a, '_chat_openai', scripted_chat)

    async def exercise():
        try:
            await a.run_once('fix value')
            await a.run_once('fix value')
            (a.trace.blob_dir / seen['blob']['sha256']).write_text('tampered', encoding='utf-8')
            assert 'integrity check' in a._read_observation({'evidence_id': seen['id']})
            assert seen['id'] not in {item.id for item in a._repair_state.evidence}
            assert not a._repair_state.verifications
        finally:
            await a.shutdown()

    asyncio.run(exercise())
