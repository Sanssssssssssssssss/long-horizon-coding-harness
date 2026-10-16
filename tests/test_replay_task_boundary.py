"""Feedback may train a Skill, but must not replace its fresh replay task."""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.agent import Agent
import agents.online_skill_eval as evaluation
import agents.skill_evolution as evolution
import agents.skills as skills


def window_with_feedback():
    agent = Agent(api_base='http://unused.invalid', api_key='test', custom_system_prompt='fixed',
                  custom_tools=[], is_sub_agent=True)
    task = 'Implement a function that preserves input order.'
    target = 'HISTORICAL_COMPLETE_TASK_ANSWER'
    agent._openai_messages.extend([agent._user_message(task), {'role': 'assistant', 'content': target}])
    agent._set_pending_skill_extraction_window(original_user_message=task,
                                               assistant_text=target, retrieved_reference=None)
    return task, target, agent._pop_pending_skill_extraction_window('Great, keep this workflow.')


def test_extraction_window_has_a_separate_task_input_boundary():
    task, target, window = window_with_feedback()
    assert window['messages'][-1]['content'] == 'Great, keep this workflow.'
    assert window['replay_messages'] == [{'role': 'user', 'content': task}]
    assert target not in json.dumps(window['replay_messages'])


def test_feedback_and_historical_answer_are_not_fresh_replay_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(evolution, 'get_evolution_dir', lambda: tmp_path)
    task, target, window = window_with_feedback()
    replay_messages = [{'role': 'user', 'content': task}]
    skills.record_online_provenance(action='discard', skill_name='ordering',
        result={'ok': True, 'replay_messages': replay_messages}, messages=window['messages'])
    rows = [json.loads(line) for line in (tmp_path / evolution.ONLINE_PROVENANCE_LOG).read_text(encoding='utf-8').splitlines()]
    index = json.loads((tmp_path / evolution.ONLINE_PROVENANCE_INDEX).read_text(encoding='utf-8'))
    pool = evaluation._build_replay_pool('ordering', rows, index['ordering'], freeze=False)
    assert len(pool) == 1
    assert pool[0]['latest_user'] == task and pool[0]['latest_assistant'] == target
    async def query(system, prompt):
        assert task in prompt and target not in prompt and 'Great, keep this workflow.' not in prompt
        return 'Fresh generated implementation.'
    answer = asyncio.run(evaluation._generate_variant_response_async(
        variant={'snapshot': {'instructions': 'Preserve input order.'}}, sample=pool[0], side_query=query))
    assert answer == 'Fresh generated implementation.'


def test_ingest_preserves_explicit_replay_inputs_even_for_rejected_candidates(tmp_path, monkeypatch):
    from agents.online_skill_evolution import online_ingest
    monkeypatch.setattr(evolution, 'get_evolution_dir', lambda: tmp_path)
    monkeypatch.setattr(skills, 'discover_skills', lambda: [])
    monkeypatch.setattr(skills, 'retrieve_relevant_skills', lambda *args, **kwargs: [])
    task, target, window = window_with_feedback()
    async def query(system, payload):
        if 'Skill Extractor' in system:
            return '{"skills":[{"name":"ordering","description":"Preserve input order","instructions":"Preserve input order.","source":"user"}]}'
        return '{"action":"discard","reason":"Constructed rejected duplicate"}'
    result = asyncio.run(online_ingest(messages=window['messages'], side_query=query,
                                       replay_messages=[{'role': 'user', 'content': task}]))
    assert result['action'] == 'discard'
    row = json.loads((tmp_path / evolution.ONLINE_PROVENANCE_LOG).read_text(encoding='utf-8').splitlines()[0])
    assert row['result']['replay_messages'] == [{'role': 'user', 'content': task}]
