import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from agents.agent import Agent

@pytest.mark.parametrize('provider',['openai','anthropic'])
def test_compaction_keeps_user_update_with_its_original_authority(provider):
    agent=Agent(api_base='http://unused.invalid' if provider=='openai' else None,api_key='test',
                custom_system_prompt='Coding assistant',custom_tools=[],is_sub_agent=True)
    system={'role':'system','content':'Coding assistant'}
    canonical='Version one uses eager conversion.'
    update={'role':'user','content':'Version two must return a lazy iterator and replaces version one.'}
    tail=[{'role':'assistant','content':'Ready to implement.'},{'role':'user','content':'Implement the latest version.'}]
    archived=[update,{'role':'user','content':'<stop_hook>Internal verification followup</stop_hook>'}]
    messages=[{'role':'user','content':canonical},*archived,*tail]
    if provider=='openai':agent._openai_messages=[system,*messages]
    else:agent._anthropic_messages=messages
    agent._compaction_history_parts=lambda *args,**kwargs:(archived,tail,None)
    async def summarize(*args,**kwargs):return 'Latest version requires lazy conversion.'
    agent._generate_folded_session_memory=summarize
    method=agent._compact_openai if provider=='openai' else agent._compact_anthropic
    assert asyncio.run(method(trigger='manual'))
    folded=agent._openai_messages if provider=='openai' else agent._anthropic_messages
    assert update in folded and folded.index(update)>next(i for i,m in enumerate(folded) if m['content']==canonical)
    assert not any('Internal verification followup' in str(m) for m in folded)

def test_pinned_current_task_follows_older_archived_requests():
    old={'role':'user','content':'Previous episode: use eager conversion.'}
    current='Current task: replace it with a lazy API.'
    folded=Agent._folded_history(current,[old],'Current task is lazy.',[],canonical_is_first=False)
    assert folded.index(old)<next(i for i,m in enumerate(folded) if m.get('content')==current)

def test_archived_user_revisions_retain_authority_in_order():
    history=[{'role':'user','content':'Old update '+str(i)} for i in range(20)]
    folded=Agent._folded_history('Task',history,'Prior updates summarized.',[],canonical_is_first=True)
    preserved=[m for m in folded if m['role']=='user' and m['content']!='Task']
    assert preserved==history
