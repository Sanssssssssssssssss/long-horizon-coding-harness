"""Local skill edits must preserve unrelated knowledge and reject ambiguous edits."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import agents.skill_evolution as evolution
from agents.frontmatter import parse_frontmatter


@pytest.fixture
def skill(tmp_path,monkeypatch):
    path=tmp_path/'SKILL.md'
    path.write_text('---\nname: examples\ndescription: workflow\nversion: 0.1.0\n---\n\nKeep input order.\nReject boolean sizes.\nPreserve Unicode.\n',encoding='utf-8')
    monkeypatch.setattr(evolution,'resolve_skill_file',lambda *args,**kwargs:path)
    monkeypatch.setattr(evolution,'get_evolution_dir',lambda:tmp_path/'history')
    return path


def test_local_skill_update_preserves_other_rules_and_snapshots(skill):
    old=skill.read_text(encoding='utf-8')
    result=evolution.evolve_skill_file(skill_name='examples',lesson='Clarify ordering',
           instruction_updates=[{'old':'Keep input order.','new':'Keep input order including duplicate values.'}])
    assert result['ok']
    body=parse_frontmatter(skill.read_text(encoding='utf-8')).body
    assert 'Keep input order including duplicate values.' in body
    assert 'Reject boolean sizes.' in body and 'Preserve Unicode.' in body
    assert old in Path(result['history']).read_text(encoding='utf-8').replace('\\n','\n')


def test_missing_skill_edit_leaves_active_version_unchanged(skill):
    before=skill.read_bytes()
    with pytest.raises(ValueError,match='unique'):
        evolution.evolve_skill_file(skill_name='examples',lesson='Bad target',
             instruction_updates=[{'old':'A guessed rule not in this skill.','new':'Overwrite everything.'}])
    assert skill.read_bytes()==before


def test_repeated_delta_append_does_not_grow_skill(skill):
    update=[{'old':'','new':'Check empty input before reporting coverage.'}]
    evolution.evolve_skill_file(skill_name='examples',lesson='Add a boundary',instruction_updates=update)
    after=skill.read_bytes()
    result=evolution.evolve_skill_file(skill_name='examples',lesson='Add a boundary',instruction_updates=update)
    assert result['ok'] and skill.read_bytes()==after
