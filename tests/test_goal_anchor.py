"""Long extraction windows must keep the original goal through feedback dequeue."""
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.agent import Agent

@pytest.mark.parametrize('feedback', ['', 'The task is still incomplete; those were read-only probes.'])
def test_long_feedback_window_keeps_original_goal(feedback):
    agent = Agent(api_base='http://unused.invalid', api_key='test', custom_system_prompt='fixed',
                  custom_tools=[], skills_enabled=False, is_sub_agent=True)
    task = 'Recover the lost changes, merge them into the requested branch, and keep the result.'
    agent._openai_messages += [{'role': 'user', 'content': task}]
    agent._openai_messages += [{'role': 'tool', 'tool_call_id': f'read-{i}', 'content': f'Observed unchanged state in probe {i}.'} for i in range(16)]
    agent._set_pending_skill_extraction_window(original_user_message=task, assistant_text='The repository was investigated but not repaired.', retrieved_reference=None)
    window = agent._pop_pending_skill_extraction_window(feedback)
    assert window['messages'][0] == {'role': 'user', 'content': task}
    assert sum(row.get('content') == task for row in window['messages']) == 1
    assert len(window['messages']) <= 10
    if feedback:
        assert window['messages'][-1]['content'] == feedback
    assert window['replay_messages'] == [{'role': 'user', 'content': task}]
