"""A failed generation is not a replay answer, and judge labels must be booleans."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agents.online_skill_eval as evaluation


@pytest.mark.parametrize('failure', ['timeout', 'empty'])
def test_generation_failure_cannot_earn_rule_scores(failure):
    snapshot = {'name': 'concise', 'instructions': 'Answer first. Keep answers under 3 paragraphs.'}
    rules = evaluation._compile_eval_rules(snapshot)
    async def query(system, user):
        if failure == 'timeout':
            raise TimeoutError('constructed transport failure')
        return ''
    outputs, outcomes, summary = asyncio.run(evaluation._evaluate_generated_variant_async(
        variant={'variant_id': 'candidate', 'snapshot': snapshot},
        samples=[{'sample_id': 'heldout', 'split': 'promotion_test',
                  'messages': [{'role': 'user', 'content': 'Explain validation.'}]}],
        rules=rules, side_query=query))
    assert summary['total_score'] == 0
    assert outcomes and all(not outcome['passed'] for outcome in outcomes)
    assert outputs[0]['generation_error'] and outputs[0]['response_text'] == ''


def test_failed_generation_does_not_spend_a_judge_call():
    judge_calls = []
    async def query(system, user):
        if 'strict binary evaluator' in system:
            judge_calls.append(user)
            return '{"pass":true,"reason":"constructed permissive judge"}'
        raise TimeoutError('constructed timeout')
    snapshot = {'name': 'accurate', 'instructions': 'Do not hallucinate. State uncertainty.'}
    rules = evaluation._compile_eval_rules(snapshot, include_llm_rules=True)
    _, _, summary = asyncio.run(evaluation._evaluate_generated_variant_async(
        variant={'variant_id': 'candidate', 'snapshot': snapshot},
        samples=[{'sample_id': 'heldout', 'split': 'promotion_test',
                  'messages': [{'role': 'user', 'content': 'Answer the question.'}]}],
        rules=rules, side_query=query))
    assert judge_calls == [] and summary['total_score'] == 0


@pytest.mark.parametrize('value', [False, 'false', 'true', 1, [], {'yes': True}, None, True])
def test_binary_judge_accepts_only_a_literal_boolean_true(value):
    async def judge(system, payload):
        return json.dumps({'pass': value, 'reason': 'constructed schema response'})
    result = asyncio.run(evaluation._evaluate_rule_async(
        {'rule_id': 'alignment', 'kind': 'llm_binary', 'hard': True,
         'params': {'requirement_text': 'Preserve input order.'}},
        'An answer.', sample={'latest_user': 'Preserve input order.'},
        skill_name='ordering', side_query=judge))
    assert result['passed'] is (value is True)


def test_literal_failure_marker_in_a_real_response_is_not_a_transport_failure():
    async def query(system, user):
        return '[candidate_response_failed: example from the requested documentation]'
    snapshot = {'name': 'documentation', 'instructions': 'Write a concise explanation.'}
    outputs, outcomes, _ = asyncio.run(evaluation._evaluate_generated_variant_async(
        variant={'variant_id': 'candidate', 'snapshot': snapshot},
        samples=[{'sample_id': 'documentation', 'messages': [{'role': 'user', 'content': 'Explain the literal marker.'}]}],
        rules=evaluation._compile_eval_rules(snapshot), side_query=query))
    assert outputs[0]['response_text'].startswith('[candidate_response_failed:')
    assert all(outcome['passed'] for outcome in outcomes)


@pytest.mark.parametrize('failed_variant', ['candidate', 'champion'])
def test_incomplete_fresh_replay_cannot_promote(failed_variant):
    candidate = {'evaluation_id': 'same-heldout', 'average_score': 1, 'hard_failures': 0}
    baseline = {'evaluation_id': 'same-heldout', 'average_score': 0, 'hard_failures': 0}
    (candidate if failed_variant == 'candidate' else baseline)['generation_failures'] = 1
    decision = evaluation._promotion_decision(status='healthy', candidate=candidate,
                                              champion={'summary': baseline})
    assert not decision['promoted'] and decision['status'] == 'unobserved'


def test_transport_failure_stops_mutation_and_comparison_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, '_load_champion', lambda lineage: {})
    calls = []
    async def query(system, user):
        calls.append((system, user))
        raise TimeoutError('constructed missing replay')
    snapshot = {'name': 'concise', 'instructions': 'Keep answers under 3 paragraphs.'}
    result = asyncio.run(evaluation._build_candidate_eval_bundle_async(
        lineage_id='trial', snapshot=snapshot,
        replay_pool=[{'sample_id': split, 'split': split,
                      'messages': [{'role': 'user', 'content': 'Explain validation.'}]}
                     for split in ('mutate_dev', 'promotion_test')],
        rules=evaluation._compile_eval_rules(snapshot), rule_summary={}, side_query=query))
    assert len(calls) == 1
    assert result['current_dev_summary']['generation_failures'] == 1
    assert not result.get('best_test_summary')


def test_aggregate_improvement_cannot_hide_a_new_hard_failure():
    baseline = {'evaluation_id': 'cohort', 'average_score': 1, 'hard_failures': 2,
                'hard_failure_keys': [('task-a', 'api-contract'), ('task-b', 'api-contract')]}
    candidate = {'evaluation_id': 'cohort', 'average_score': 2, 'hard_failures': 1,
                 'hard_failure_keys': [('previously-passing-task', 'api-contract')]}
    decision = evaluation._promotion_decision(status='healthy', candidate=candidate, champion={'summary': baseline})
    assert not decision['promoted']
    assert decision['introduced_hard_failures'] == [('previously-passing-task', 'api-contract')]
    candidate['hard_failure_keys'] = [('task-a', 'api-contract')]
    improved = evaluation._promotion_decision(status='healthy', candidate=candidate, champion={'summary': baseline})
    assert improved['promoted']


def test_display_failure_limit_cannot_hide_a_new_hard_failure():
    samples = [{'sample_id': f'task-{index:02}', 'split': 'promotion_test'} for index in range(30)]
    rule = {'rule_id': 'api-contract', 'hard': True}
    def summary(failed):
        outcomes = [{**sample, **rule, 'passed': index not in failed, 'score': 0 if index in failed else 2}
                    for index, sample in enumerate(samples)]
        raw = evaluation._summarize_outcomes_from_rows(rules=[rule], samples=samples, outcomes=outcomes)
        assert len(raw['failures']) == 20  # This is a report display limit.
        return {**evaluation._variant_summary_from_eval({}, raw, len(samples)), 'evaluation_id': 'same-cohort'}
    baseline = summary(set(range(25)))
    candidate = summary(set(range(20)) | {29})
    decision = evaluation._promotion_decision(status='healthy', candidate=candidate, champion={'summary': baseline})
    assert not decision['promoted']
    assert decision['introduced_hard_failures'] == [('task-29', 'api-contract')]
