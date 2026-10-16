"""Unavailable or malformed judgments cannot retire an otherwise valid Skill."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agents.skill_evolution as evolution
from agents.online_skill_evolution import judge_retrieved_skill_usage


@pytest.mark.parametrize('mode', ['no_judge', 'missing', 'invalid_boolean'])
def test_unobserved_usage_does_not_prune_active_memory(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(evolution, 'get_evolution_dir', lambda: tmp_path / 'history')
    monkeypatch.setenv('SANS_SKILL_USAGE_PRUNE_MIN_RETRIEVED', '1')
    path = tmp_path / 'validation'
    path.mkdir()
    (path / 'SKILL.md').write_text('Preserve known API behavior.\n', encoding='utf-8')
    hit = {'name': 'validation', 'source': 'user', 'skill_dir': str(path), 'score': 1.0}
    async def query(system, payload):
        if mode == 'missing':
            return '{"judgments":[]}'
        return '{"judgments":[{"name":"validation","relevant":"false","used":"false"}]}'
    rows = asyncio.run(judge_retrieved_skill_usage(hits=[hit], user_message='Fix the bug.',
        assistant_text='Added the requested function.', side_query=None if mode == 'no_judge' else query))
    result = evolution.record_skill_usage_judgments(rows)
    assert path.is_dir() and not result['pruned']
    assert rows == []


def test_confirmed_negative_usage_is_still_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(evolution, 'get_evolution_dir', lambda: tmp_path / 'history')
    hit = {'name': 'validation', 'source': 'project', 'skill_dir': str(tmp_path / 'validation'), 'score': 1.0}
    async def query(system, payload):
        return '{"judgments":[{"name":"validation","relevant":false,"used":false,"reason":"Wrong task family"}]}'
    rows = asyncio.run(judge_retrieved_skill_usage(hits=[hit], user_message='Write documentation.',
        assistant_text='Documentation added.', side_query=query))
    assert len(rows) == 1 and rows[0]['used'] is False
    evolution.record_skill_usage_judgments(rows)
    stats = json.loads((tmp_path / 'history' / evolution.SKILL_USAGE_STATS).read_text(encoding='utf-8'))
    assert stats['validation']['retrieved'] == 1 and stats['validation']['used'] == 0
