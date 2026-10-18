"""Evaluation receipts must not overwrite another run in the same second."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import agents.online_skill_eval as evaluation


def test_two_same_second_evaluations_preserve_the_first_output(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'get_evolution_dir', lambda: tmp_path)
    monkeypatch.setattr(evaluation.time, 'strftime', lambda *args, **kwargs: '20261008T000000Z')
    common = dict(skill_name='validation', snapshot={'name': 'validation', 'instructions': 'Validate policies.'},
                  rules=[], rule_summary={}, status='unobserved', reasons=['No fresh comparison'], candidate_bundle={})
    first = evaluation._persist_eval_artifacts(replay_pool=[
        {'sample_id': 'task', 'split': 'mutate_dev', 'latest_assistant': 'First preserved historical response.'}], **common)
    path = Path(first['run_dir']) / 'outputs.jsonl'
    before = path.read_bytes()
    second = evaluation._persist_eval_artifacts(replay_pool=[
        {'sample_id': 'task', 'split': 'mutate_dev', 'latest_assistant': 'Second historical response.'}], **common)
    assert path.read_bytes() == before
    assert first['run_id'] != second['run_id']
    assert json.loads((Path(second['run_dir']) / 'outputs.jsonl').read_text(encoding='utf-8'))['response_text'] == 'Second historical response.'
