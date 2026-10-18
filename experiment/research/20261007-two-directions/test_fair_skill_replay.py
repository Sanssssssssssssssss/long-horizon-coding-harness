import asyncio
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import agents.online_skill_eval as evaluation


def test_mutation_uses_dev_and_promotion_replays_champion_on_same_test(monkeypatch):
    calls=[]
    current={'name':'workflow','instructions':'Current active rules'}
    candidate={'variant_id':'candidate','label':'candidate','snapshot':{'instructions':'Improved rules'}}
    champion={'snapshot':{'instructions':'Previous champion'},'summary':{'average_score':999}}
    pool=[{'sample_id':'dev','split':'mutate_dev'}, {'sample_id':'heldout','split':'promotion_test'}]
    monkeypatch.setattr(evaluation,'_load_champion',lambda lineage:champion)
    monkeypatch.setattr(evaluation,'_build_heuristic_candidate_variants',lambda **kwargs:[candidate])
    async def mutation(**kwargs):
        assert kwargs['rule_summary']['failures']==['dev failure']
        return None
    monkeypatch.setattr(evaluation,'_build_llm_candidate_variant_async',mutation)
    async def evaluate(**kwargs):
        variant=kwargs['variant']; sample_ids=[s['sample_id'] for s in kwargs['samples']]
        calls.append((variant['variant_id'],sample_ids))
        score=0 if variant['variant_id']=='current_active' else 1
        return [],[],{'average_score':score,'hard_failures':0,'failures':['dev failure']}
    monkeypatch.setattr(evaluation,'_evaluate_generated_variant_async',evaluate)
    result=asyncio.run(evaluation._build_candidate_eval_bundle_async(lineage_id='l',snapshot=current,
             replay_pool=pool,rules=[],rule_summary={'failures':['HOLDOUT DATA MUST NOT TRAIN']},side_query=lambda:None))
    assert ('current_active',['dev']) in calls
    assert ('candidate',['dev']) in calls
    assert ('candidate',['heldout']) in calls and ('champion_baseline',['heldout']) in calls
    assert result['best_test_summary']['evaluation_id']==result['baseline_test_summary']['evaluation_id']
    assert result['baseline_test_summary']['average_score']==1


def test_unmatched_historical_champion_score_cannot_release_candidate():
    decision=evaluation._promotion_decision(status='healthy',
          candidate={'average_score':10,'hard_failures':0,'evaluation_id':'new-tasks'},
          champion={'summary':{'average_score':0,'hard_failures':0,'evaluation_id':'old-tasks'}})
    assert not decision['promoted']


def test_fresh_replay_does_not_receive_historical_target_answer():
    async def query(system,prompt):
        assert 'HISTORICAL TARGET ANSWER' not in prompt
        assert 'Earlier conversation context' in prompt
        assert 'Write the requested new function' in prompt
        return 'Fresh model answer'
    sample={'messages':[{'role':'user','content':'Earlier request'},
                        {'role':'assistant','content':'Earlier conversation context'},
                        {'role':'user','content':'Write the requested new function'},
                        {'role':'assistant','content':'HISTORICAL TARGET ANSWER'}]}
    response=asyncio.run(evaluation._generate_variant_response_async(
                variant={'snapshot':{'instructions':'Coding procedure'}},sample=sample,side_query=query))
    assert response=='Fresh model answer'
