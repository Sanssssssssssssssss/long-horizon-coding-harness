"""Budgeted whole-body projection of real working reads; no verification authority."""
from copy import deepcopy
import hashlib,json,re
from agents.agent import _to_openai_tools
from agents.tools import get_active_tool_definitions

TAG='<runtime-budgeted-working-views>';END='</runtime-budgeted-working-views>'
def request_cost(agent,messages):
    tools=get_active_tool_definitions(agent.tools)
    if agent.use_openai:tools=_to_openai_tools(tools)
    system=None if agent.use_openai else agent._system_prompt
    return agent._request_reservation(messages=messages,tools=tools,system=system)
def request_input_estimate(agent,messages):
    tools=get_active_tool_definitions(agent.tools)
    if agent.use_openai:tools=_to_openai_tools(tools)
    system=None if agent.use_openai else agent._system_prompt
    return agent._estimate_model_input_tokens(messages=messages,tools=tools,system=system)
def project_working_views(agent,projected,views,*,mutable_files=(),legacy_frames=(),reserve_margin=0):
    """Retain read refs; add only whole current bodies that fit the native request."""
    messages=deepcopy(projected)
    owned_hashes=getattr(agent,'_budgeted_working_frame_hashes',set())
    for row in messages:
        if isinstance(row.get('content'),str):
            row['content']=re.sub(re.escape(TAG)+r'\n[^\n]*\n'+re.escape(END),
                lambda m:'' if hashlib.sha256(m.group().encode()).hexdigest() in owned_hashes else m.group(),row['content'])
            for frame in legacy_frames:
                if frame:row['content']=row['content'].replace(frame,'')
    r=agent._repair_state;identity=(r.execution_id,r.repo_generation,r.current_repo_state) if r else None
    rows=[]
    for view in views:
        current=identity==(view.get('execution_id'),view.get('repo_generation'),view.get('repo_state'))
        rows.append({k:v for k,v in view.items() if k!='actual_output'}|{
            'state_scope':'current_source' if current else 'historical_only','current_source':current,'verification':False,'body_omitted':True,
            'reader':{'evidence_id':view['evidence_id'],'part':'output','offset':0,'limit':200}})
    def render(entries):
        result=deepcopy(messages);frame=TAG+'\n'+json.dumps({'authority':'Actual read observations, never task or verification authority','files':entries})+'\n'+END
        for row in reversed(result):
            if row.get('role') in {'user','tool'} and isinstance(row.get('content'),str):row['content']+='\n'+frame;break
        return result,frame
    candidate,frame=render(rows);reference_cost=request_cost(agent,candidate)
    budget=agent.token_budget;remaining=None if budget.get('limit') is None else budget['limit']-budget['charged']-reserve_margin
    order=sorted(range(len(views)),key=lambda i:(views[i]['file'] not in mutable_files,i))
    selected=[]
    for i in order:
        if not rows[i]['current_source'] or views[i]['file'] not in mutable_files:continue
        trial=deepcopy(rows);trial[i].update(body_omitted=False,actual_output=views[i]['actual_output'])
        proposed,proposed_frame=render(trial);cost=request_cost(agent,proposed)
        if (remaining is None or cost<=remaining) and request_input_estimate(agent,proposed)<=agent.max_input_tokens_per_call:
            rows=trial;candidate=proposed;frame=proposed_frame;selected.append(views[i]['file'])
    agent._budgeted_working_frame=frame
    agent._budgeted_working_frame_hashes=owned_hashes|{hashlib.sha256(frame.encode()).hexdigest()}
    agent._working_projection_receipt={'reference_reservation':reference_cost,'selected_bodies':selected,'whole_request_reservation':request_cost(agent,candidate),
        'remaining_after_margin':remaining,'reserve_margin':reserve_margin,'policy':'Whole bodies only; mutable implementation first; omitted bodies have real readers.'}
    return candidate
