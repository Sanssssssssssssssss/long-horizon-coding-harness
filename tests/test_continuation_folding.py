"""Regression boundaries for actual continuation failures; no paid calls."""
import asyncio
import json

import pytest

from benchmarks.continuation_folding import ContinuationFoldingLoop
from benchmarks.folding_checkpoint import render_checkpoint
from test_folding_checkpoint import agent, receipt, selection, state, history
from agents.session_memory import folded_memory_summary


def test_cross_stage_implementation_survives_without_old_receipt_authority(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    old = state()
    green = receipt(old, passed=True)
    checkpoint = json.loads(render_checkpoint(old, selection(old, [green.id])))
    checkpoint['unverified_model_proposals']['implementation_summary'] = 'decode_vector parses widths; cmd_eval consumes VectorValue.'
    current = a._repair_state = state()
    red = receipt(current)
    before = current.snapshot()

    async def summarize(_system, content, **kwargs):
        assert kwargs['purpose'] == 'session_fold_continuation'
        request = json.loads(content)
        assert 'decode_vector' in request['previous_continuation_untrusted']
        assert green.id not in content
        assert 'cmd_eval' in request['conversation_untrusted']
        return 'Progress: decode_vector is present. Pending: cmd_eval must wire the new evaluator. All tests passed.'

    monkeypatch.setattr(a, '_side_query_once', summarize)
    result = json.loads(asyncio.run(a._run_compaction_query([
        {'role':'assistant','content':'cmd_eval currently calls old evaluator'}], 'Summarize', json.dumps(checkpoint))))
    assert result['selected_runtime_receipts'][0]['evidence_id'] == red.id
    assert result['selected_runtime_receipts'][0]['test_failure_observed']
    assert 'All tests passed' in result['unverified_model_proposals']['implementation_summary']
    assert current.snapshot() == before and not current.completion_decision().complete


@pytest.mark.parametrize('text', ['```json\n{"implementation_notes":"use decoder"}\n```', 'Progress: ' + 'detail ' * 4000])
def test_continuation_format_cannot_invent_receipts_or_force_schema_retry(tmp_path, monkeypatch, text):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    current = a._repair_state = state()
    observed = receipt(current)
    calls=[]
    async def summarize(*args, **kwargs):
        calls.append(kwargs)
        return text
    monkeypatch.setattr(a, '_side_query_once', summarize)
    result=json.loads(asyncio.run(a._run_compaction_query([], 'Summarize')))
    assert len(calls) == 1
    assert len(result['unverified_model_proposals']['implementation_summary'].encode('utf-8')) <= 20000
    assert result['selected_runtime_receipts'][0]['evidence_id'] == observed.id
    assert not current.completion_decision().complete


def test_full_pin_capacity_does_not_append_extra_receipts(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    current = a._repair_state = state()
    pins = [receipt(current).id for _ in range(12)]
    current.upsert_hypothesis('cause', 'Shared parser issue', 'Fails in parser', 'Parser succeeds', evidence_ids=pins)
    extra = receipt(current)
    async def summarize(*args, **kwargs):
        return 'Pending: inspect the parser.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    result=json.loads(asyncio.run(a._run_compaction_query([], 'Summarize')))
    identifiers={r['evidence_id'] for r in result['selected_runtime_receipts']}
    assert identifiers == set(pins) and extra.id not in identifiers


def test_native_fold_installs_plain_continuation_and_keeps_latest_tool_pair(tmp_path, monkeypatch):
    import agents.agent as agent_module
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    current = a._repair_state = state()
    observed = receipt(current)
    history(a)
    async def summarize(*args, **kwargs):
        return 'Progress: decoder validates values. Pending: wire the new decoder into the call site.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    assert asyncio.run(a._compact_conversation(trigger='periodic'))
    summaries=[folded_memory_summary(m) for m in a._openai_messages if folded_memory_summary(m)]
    result=json.loads(summaries[0])
    assert result['schema_version']=='evidence-continuation-v1'
    assert observed.id in result['selected_runtime_receipts'][0]['evidence_id']
    assert a._openai_messages[-1]['tool_call_id']=='last-read'
    assert a._openai_messages[-2]['tool_calls'][0]['id']=='last-read'
    assert not current.completion_decision().complete


def test_interval_due_does_not_spend_when_next_solve_fits(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop,
              auto_compact=True, fold_every_decisions=3, max_decisions=18)
    a.decision_clock['completed'] = 3
    a._openai_messages.append({'role': 'user', 'content': 'Continue the pending edit.'})
    async def unexpected(*args, **kwargs):
        pytest.fail('A fitting conversation must not pay for an interval summary')
    monkeypatch.setattr(a, '_side_query_once', unexpected)
    assert not asyncio.run(a._check_and_compact())


@pytest.mark.parametrize('pressure', ['budget', 'context'])
def test_pressure_fold_keeps_error_and_allows_next_solve_without_refolding(tmp_path, monkeypatch, pressure):
    import agents.agent as agent_module
    monkeypatch.setattr(agent_module, 'COMPACTION_KEEP_RECENT_TOKENS', 600)
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop,
              auto_compact=True, fold_every_decisions=3, max_decisions=18)
    a._repair_state = state()
    observed = receipt(a._repair_state)
    history(a)
    a._openai_messages[-1]['content'] = 'test_value.py:198 AssertionError: rounded output differs from raw float'
    a.decision_clock['completed'] = 12
    if pressure == 'budget':
        a.token_budget['limit'] = a._next_request_reservation() - 1
    else:
        a.last_input_token_count = a.context_window
    calls = []
    async def summarize(*args, **kwargs):
        calls.append(kwargs)
        return 'Pending: correct the precision mismatch in test_value.py:198 and recheck.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    assert asyncio.run(a._check_and_compact())
    assert a._openai_messages[-1]['tool_call_id'] == 'last-read'
    assert 'test_value.py:198' in a._openai_messages[-1]['content']
    assert a._openai_messages[-2]['tool_calls'][0]['id'] == 'last-read'
    assert observed.id in json.loads(folded_memory_summary(a._openai_messages[2]))['selected_runtime_receipts'][0]['evidence_id']
    a.decision_clock['completed'] += 1
    a._openai_messages.append({'role': 'assistant', 'content': 'Apply the pending correction.'})
    assert not asyncio.run(a._check_and_compact())
    assert len(calls) == 1
    assert not a._repair_state.completion_decision().complete


def test_pressure_with_no_reclaimable_history_does_not_pay(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, auto_compact=True)
    a._repair_state = state()
    a._openai_messages.append({'role': 'user', 'content': 'fix value'})
    a.last_input_token_count = a.context_window
    before = json.dumps(a._openai_messages)
    async def unexpected(*args, **kwargs):
        pytest.fail('No history can be reclaimed')
    monkeypatch.setattr(a, '_side_query_once', unexpected)
    assert not asyncio.run(a._check_and_compact())
    assert json.dumps(a._openai_messages) == before


def test_insufficient_remaining_budget_cannot_buy_an_unusable_checkpoint(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, auto_compact=True)
    a._repair_state = state()
    receipt(a._repair_state)
    history(a)
    a.token_budget['limit'] = 500
    before = json.dumps(a._openai_messages)
    async def unexpected(*args, **kwargs):
        pytest.fail('Even the minimum continuation cannot enter the remaining budget')
    monkeypatch.setattr(a, '_side_query_once', unexpected)
    assert not asyncio.run(a._check_and_compact())
    assert json.dumps(a._openai_messages) == before


def test_small_window_archives_old_large_exchange_without_losing_latest_error(tmp_path):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, auto_compact=True)
    a._repair_state = state()
    a._openai_messages += [
        {'role':'user','content':'fix value'},
        {'role':'assistant','content':'earlier exploration ' * 1500},
        {'role':'assistant','tool_calls':[{'id':'old-read','type':'function',
            'function':{'name':'read_file','arguments':'{"file_path":"large.py"}'}}]},
        {'role':'tool','tool_call_id':'old-read','content':'old recoverable source ' * 1000},
        {'role':'assistant','tool_calls':[{'id':'last-test','type':'function',
            'function':{'name':'run_shell','arguments':'{"command":"pytest test_value.py"}'}}]},
        {'role':'tool','tool_call_id':'last-test','content':'test_value.py:198 AssertionError'},
    ]
    archive,tail,_ = a._compaction_history_parts(a._openai_messages,start=2,trigger='budget')
    assert any(m.get('tool_call_id')=='old-read' for m in archive)
    assert not any(m.get('tool_call_id')=='old-read' for m in tail)
    assert tail[-1]['tool_call_id']=='last-test'
    assert tail[-2]['tool_calls'][0]['id']=='last-test'


def test_small_window_preview_keeps_failure_and_recovers_middle_from_original_blob(tmp_path):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    a._repair_state = state()
    red = receipt(a._repair_state)
    header = '<tool_observation>' + json.dumps({'evidence_id':red.id,'exit_code':1}) + '</tool_observation>'
    raw = header + '\n' + 'line of source\n' * 700 + 'UNIQUE_MIDDLE_FAILURE\n' + 'more source\n' * 700 + 'FINAL: 1 failed'
    ref = a.trace.store_payload({'output':raw})
    a._observations[red.id] = ref
    before = a._repair_state.snapshot()
    preview = a._persist_large_result('run_shell',raw)
    assert len(preview) < 3000 and red.id in preview and 'FINAL: 1 failed' in preview
    assert 'UNIQUE_MIDDLE_FAILURE' not in preview
    recovered = a._read_observation({'evidence_id':red.id,'offset':700,'limit':4})
    assert 'UNIQUE_MIDDLE_FAILURE' in recovered
    assert a._persist_large_result('read_observation',recovered) == recovered
    assert a._repair_state.snapshot() == before
    assert not a._repair_state.completion_decision().complete
    (a.trace.blob_dir/ref['sha256']).write_text('corrupted',encoding='utf-8')
    assert 'integrity check' in a._read_observation({'evidence_id':red.id})


@pytest.mark.parametrize('context_pressure', [False, True])
def test_budget_can_archive_complete_oversized_tail_to_leave_a_solve(tmp_path, monkeypatch, context_pressure):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, auto_compact=True)
    a._repair_state = state()
    red = receipt(a._repair_state)
    history(a)
    a._openai_messages[-1]['content'] = 'large recently observed file\n' * 1300
    a.token_budget['limit'] = 60000
    if context_pressure:
        a.last_input_token_count = a.context_window
    original = list(a._openai_messages)
    async def summarize(*args, **kwargs):
        return 'Pending: run the newly corrected test. Full source remains in recorded observations.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    assert asyncio.run(a._check_and_compact())
    assert a._next_request_reservation() < a.token_budget['limit']
    assert not any(m.get('tool_call_id')=='last-read' for m in a._openai_messages)
    assert original[-1]['tool_call_id']=='last-read'
    summary = next(folded_memory_summary(m) for m in a._openai_messages if folded_memory_summary(m))
    assert red.id in summary and not a._repair_state.completion_decision().complete


def test_rejected_summary_does_not_pay_again_without_progress(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, auto_compact=True)
    current = a._repair_state = state()
    red = receipt(current)
    history(a)
    before = json.dumps(a._openai_messages)
    before_state = current.snapshot()
    calls = []

    async def summarize(*args, **kwargs):
        calls.append(args)
        return 'Pending: inspect the failing value. Full source is preserved in observations.'

    monkeypatch.setattr(a, '_side_query_once', summarize)
    # Real failure shape: minimum checkpoint fits, but the generated candidate
    # does not reclaim enough. Two scheduler entry points attempt the same fold.
    monkeypatch.setattr(a, '_checkpoint_cost_decreases',
                        lambda candidate, *, trigger, scope: scope == 'minimum_checkpoint_not_smaller')
    assert not asyncio.run(a._compact_conversation(trigger='auto'))
    assert not asyncio.run(a._compact_conversation(trigger='auto'))
    assert len(calls) == 1
    assert json.dumps(a._openai_messages) == before
    assert current.snapshot() == before_state
    assert red.test_failure_observed and not current.completion_decision().complete


@pytest.mark.parametrize('change', ['history', 'task', 'repository', 'receipt', 'hypothesis', 'prompt', 'output_limit', 'model', 'thinking'])
def test_progress_invalidates_summary_reuse(tmp_path, monkeypatch, change):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    a._repair_state = state()
    receipt(a._repair_state)
    messages = [{'role': 'assistant', 'content': 'decoder is pending'}]
    prompt = 'Summarize'
    calls = []

    async def summarize(*args, **kwargs):
        calls.append(args)
        return 'Pending: decoder still needs work.'

    monkeypatch.setattr(a, '_side_query_once', summarize)
    asyncio.run(a._run_compaction_query(messages, prompt))
    asyncio.run(a._run_compaction_query(messages, prompt))
    assert len(calls) == 1
    if change == 'history':
        messages.append({'role': 'user', 'content': 'New constraint: preserve negative values.'})
    elif change == 'task':
        a._repair_state = state()
        receipt(a._repair_state)
    elif change == 'repository':
        a._repair_state.observe_repo_state('new-code-generation')
    elif change == 'receipt':
        receipt(a._repair_state, passed=True)
    elif change == 'hypothesis':
        a._repair_state.upsert_hypothesis('h1', 'Parser', 'Fails in parser', 'Parser succeeds')
    elif change == 'prompt':
        prompt = 'Focus on the remaining integration work.'
    elif change == 'output_limit':
        a.checkpoint_summary_tokens += 1
    elif change == 'model':
        a.model = 'different-model'
    elif change == 'thinking':
        a._thinking_mode = 'enabled' if a._thinking_mode == 'disabled' else 'disabled'
    result = json.loads(asyncio.run(a._run_compaction_query(messages, prompt)))
    assert len(calls) == 2
    assert result['execution_id'] == a._repair_state.execution_id
    assert not a._repair_state.completion_decision().complete


@pytest.mark.parametrize('failure', ['empty', 'exception', 'state_change'])
def test_failed_or_stale_generation_is_not_reused(tmp_path, monkeypatch, failure):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    a._repair_state = state()
    receipt(a._repair_state)
    calls = []

    async def summarize(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            if failure == 'empty':
                return '   '
            if failure == 'exception':
                raise RuntimeError('provider failed')
            a._repair_state = state()
        return 'Pending: inspect parser.'

    monkeypatch.setattr(a, '_side_query_once', summarize)
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(a._run_compaction_query([], 'Summarize'))
    result = json.loads(asyncio.run(a._run_compaction_query([], 'Summarize')))
    assert len(calls) == 2
    assert result['execution_id'] == a._repair_state.execution_id


def test_reading_stall_survives_folding_without_changing_completion(tmp_path, monkeypatch):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, auto_compact=False)
    current = a._repair_state = state()
    red = receipt(current)
    asyncio.run(a._check_and_compact())
    before = current.snapshot()
    a.decision_clock['completed'] = 8
    messages = [{'role': 'user', 'content': 'Implement the remaining export path.'}]
    projected = a._messages_with_runtime_state(messages)
    frame = projected[0]['content'].split('<runtime-continuation>\n')[1].split('\n</runtime-continuation>')[0]
    assert json.loads(frame)['main_responses_without_repository_change'] == 8
    assert '<runtime-continuation>' not in messages[0]['content']
    # Summary calls and repeated projection must not consume solve responses.
    async def summarize(*args, **kwargs):
        request = json.loads(args[1])
        assert request['runtime_continuation']['main_responses_without_repository_change'] == 8
        assert not request['recent_context_retained']
        return 'Next action: connect the export handler to dispatch; then verify its output.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    summary = asyncio.run(a._run_compaction_query([], 'Summarize'))
    assert red.id in summary
    assert a._messages_with_runtime_state(messages) == projected
    assert current.snapshot() == before and not current.completion_decision().complete


@pytest.mark.parametrize('change', ['revision', 'execution'])
def test_actual_state_change_resets_stall_with_cumulative_clock(tmp_path, change):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop, auto_compact=False)
    a._repair_state = state()
    a.decision_clock['completed'] = 100
    asyncio.run(a._check_and_compact())
    a.decision_clock['completed'] = 109
    asyncio.run(a._check_and_compact())
    assert a._continuation_stall()['main_responses_without_repository_change'] == 9
    if change == 'revision':
        a._repair_state.observe_repo_state('new-source')
    else:
        a._repair_state = state()
    assert a._continuation_stall() is None
    asyncio.run(a._check_and_compact())
    a.decision_clock['completed'] = 116
    assert a._continuation_stall() is None
    a.decision_clock['completed'] = 117
    assert a._continuation_stall()['main_responses_without_repository_change'] == 8


@pytest.mark.parametrize('retained', [False, True])
def test_summary_receives_true_tail_boundary_without_upgrading_green_checks(tmp_path, monkeypatch, retained):
    a = agent(tmp_path, loop_class=ContinuationFoldingLoop)
    current = a._repair_state = state()
    observed = receipt(current, passed=True)
    before = current.snapshot()
    tail = [{'role':'assistant', 'content':'Public decoder assertions pass; export path is untested.'}]
    async def summarize(_system, content, **kwargs):
        request = json.loads(content)
        assert request['recent_context_retained'] is retained
        assert 'export path is untested' in request['conversation_untrusted']
        return 'Progress: all work is complete.'
    monkeypatch.setattr(a, '_side_query_once', summarize)
    result = json.loads(asyncio.run(a._run_compaction_query(
        [] if retained else tail, 'Summarize', recent_context=tail if retained else [])))
    assert result['selected_runtime_receipts'][0]['evidence_id'] == observed.id
    assert result['unverified_model_proposals']['implementation_summary'] == 'Progress: all work is complete.'
    assert current.snapshot() == before and not current.completion_decision().complete
