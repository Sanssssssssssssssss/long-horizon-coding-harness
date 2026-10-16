"""Exercise real Skill writes with deterministic manager decisions, without paid calls."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import agents.skill_evolution as evolution
import agents.skills as skills
from agents.online_skill_evolution import OnlineSkillCandidate, maintain_online_skill_candidate


@pytest.fixture
def active_skill(tmp_path, monkeypatch):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\nname: validation\ndescription: input validation\nversion: 0.1.0\n---\n\nValidate options before consuming input, including empty input.\nPreserve input order.\n', encoding='utf-8')
    definition = skills._parse_skill_file(path, 'project', str(tmp_path))
    monkeypatch.setattr(skills, 'discover_skills', lambda: [definition])
    monkeypatch.setattr(skills, 'retrieve_relevant_skills', lambda *args, **kwargs: [])
    monkeypatch.setattr(evolution, 'resolve_skill_file', lambda *args, **kwargs: path)
    monkeypatch.setattr(evolution, 'get_evolution_dir', lambda: tmp_path / 'history')
    return path


@pytest.mark.parametrize('decision', ['discard', 'invalid-action'])
@pytest.mark.parametrize('required_target', ['', 'validation'])
def test_identity_match_does_not_override_rejection(active_skill, decision, required_target):
    before = active_skill.read_bytes()
    candidate = OnlineSkillCandidate(name='validation', description='input validation',
                                    instructions='Validate options before consuming input, including empty input.',
                                    evidence='Repeats the existing rule.')
    async def manager(system, payload):
        return json.dumps({'action': decision, 'reason': 'Duplicate lesson; adds no durable improvement.'})
    result = asyncio.run(maintain_online_skill_candidate(candidate=candidate, side_query=manager,
                                                        required_target_skill=required_target))
    assert result['action'] == 'discard'
    assert active_skill.read_bytes() == before
    assert not (active_skill.parent / 'history').exists()


def test_valid_identity_merge_still_preserves_unrelated_rules(active_skill):
    candidate = OnlineSkillCandidate(name='validation', description='input validation',
                                    instructions='Also reject boolean control values.', evidence='Explicit user policy.')
    async def manager(system, payload):
        return json.dumps({'action': 'add', 'reason': 'Additional validation boundary.',
                           'instruction_updates': [{'old': '', 'new': candidate.instructions}]})
    result = asyncio.run(maintain_online_skill_candidate(candidate=candidate, side_query=manager))
    assert result['action'] == 'merge' and result['ok']
    after = active_skill.read_text(encoding='utf-8')
    assert 'Preserve input order.' in after and candidate.instructions in after
    assert Path(result['history']).is_file()
