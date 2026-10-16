"""Current file working set yields to verified current completion evidence."""
from copy import deepcopy
import hashlib,json,re
from pathlib import Path
TAG='<runtime-working-set>';END='</runtime-working-set>'
ORDER=('migration_tool.py','test_conformance_public.py','test_public.py')
def project(agent,messages,views,*,checked_path,digest,cost,estimate,finish_ready=False,completion=None):
    messages=deepcopy(messages);known=getattr(agent,'_working_set_hashes',set())
    for row in messages:
        if isinstance(row.get('content'),str):
            row['content']=re.sub(re.escape(TAG)+r'\n[^\n]*\n'+re.escape(END),lambda m:'' if hashlib.sha256(m.group().encode()).hexdigest() in known else m.group(),row['content'])
    rows=[];r=agent._repair_state
    for view in views:
        sha=None
        try:sha=digest(checked_path(Path(agent.cell['workspace']),view['file']))
        except (OSError,ValueError):pass
        valid=r is not None and r.execution_id==view['execution_id'] and sha==view['source_sha256']
        text=view['actual_output'].partition('</tool_observation>')[2].lstrip('\n') or view['actual_output']
        present=valid and bool(text) and any(text in str(row.get('content','')) for row in messages)
        rows.append({k:v for k,v in view.items() if k!='actual_output'}|{'current_file_content':valid,'current_file_sha256':sha,
            'authority':'Actual file text only; tests still need current native execution','body_omitted':True,'body_in_recent_messages':present,
            'reader':{'evidence_id':view['evidence_id'],'part':'output','offset':0,'limit':200}})
    def render(entries):
        result=deepcopy(messages);info={'files':entries,'finish_ready':finish_ready}
        if finish_ready:info.update(current_completion=completion,runtime_next_action='Current native completion conditions are satisfied. Report the changes and verification results now.')
        frame=TAG+'\n'+json.dumps(info)+'\n'+END
        for row in reversed(result):
            if row.get('role') in {'user','tool'} and isinstance(row.get('content'),str):row['content']+='\n'+frame;break
        return result,frame
    candidate,frame=render(rows);reference=cost(agent,candidate);remaining=agent.token_budget['limit']-agent.token_budget['charged']-4096
    selected=[]
    for name in (() if finish_ready else ORDER):
        for i,view in enumerate(views):
            if view['file']!=name or not rows[i]['current_file_content'] or rows[i]['body_in_recent_messages']:continue
            trial=deepcopy(rows);trial[i].update(body_omitted=False,actual_output=view['actual_output']);proposed,new_frame=render(trial)
            if cost(agent,proposed)<=remaining and estimate(agent,proposed)<=agent.max_input_tokens_per_call:
                rows=trial;candidate=proposed;frame=new_frame;selected.append(name)
    agent._working_set_hashes=known|{hashlib.sha256(frame.encode()).hexdigest()}
    agent._working_set_projection={'reference_reservation':reference,'whole_request_reservation':cost(agent,candidate),'remaining_after_margin':remaining,
        'selected_bodies':selected,'finish_ready':finish_ready,'authority':'Current typed completion evidence controls the reporting hint; file text has no verification authority.'}
    return candidate
