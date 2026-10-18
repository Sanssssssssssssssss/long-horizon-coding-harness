"""A held-out sample must not become mutation data when the pool grows."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import agents.online_skill_eval as evaluation


def test_frozen_skill_split_survives_older_sample_arrival(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'_lineage_id_for_skill',lambda name:'lineage')
    monkeypatch.setattr(evaluation,'_lineage_dataset_dir',lambda lineage:tmp_path)
    original=[{'sample_id':'00000001','time':'2026-10-07'}, {'sample_id':'00000002','time':'2026-10-06'}]
    first=evaluation._freeze_replay_pool(skill_name='test',samples=original)
    before={row['sample_id']:row['split'] for row in first}
    assert set(before.values())=={'mutate_dev','promotion_test'}
    expanded=evaluation._freeze_replay_pool(skill_name='test',samples=[{'sample_id':'00000003','time':'2026-10-05'}])
    after={row['sample_id']:row['split'] for row in expanded}
    assert all(after[sample_id]==split for sample_id,split in before.items())
