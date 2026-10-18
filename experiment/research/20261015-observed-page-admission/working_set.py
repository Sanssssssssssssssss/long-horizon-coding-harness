"""Project observed current files and scoped verification facts into a request."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


TAG = '<runtime-working-set>'
END = '</runtime-working-set>'
ORDER = ()



def observed_chunks(view, messages):
    """Yield bounded portions of actual read receipts, without rereading disk."""
    if not view.get('assembled_from_native_pages'):
        return
    lines = {}
    for line in view['actual_output'].split('\n'):
        match = re.match(r'^ *(\d+) \| ', line)
        if match:
            lines[int(match[1]) - 1] = line
    covered = set()
    for receipt in view.get('source_receipts', []):
        start = receipt['offset']
        stop = start + receipt['line_count']
        for begin in range(start, stop, 200):
            end = min(begin + 200, stop)
            indices = range(begin, end)
            if any(i not in lines or i in covered for i in indices):
                continue
            covered.update(indices)
            body = '\n'.join(lines[i] for i in indices)
            if not any(body in str(m.get('content', '')) for m in messages):
                yield begin, end, body, receipt['evidence_id']


def packed_body(chunks):
    ordered = sorted(chunks)
    parts, ranges = [], []
    for begin, end, body, _ in ordered:
        if ranges and begin != ranges[-1][1]:
            parts.append('[unobserved or omitted lines]')
        parts.append(body)
        if ranges and begin == ranges[-1][1]:
            ranges[-1][1] = end
        else:
            ranges.append([begin, end])
    return '\n'.join(parts), ranges, list(dict.fromkeys(c[3] for c in ordered))


def project(agent, messages, views, *, checked_path, digest, cost, estimate,
            finish_ready=False, completion=None):
    messages = deepcopy(messages)
    known = getattr(agent, '_working_set_hashes', set())
    pattern = re.escape(TAG) + r'\n[^\n]*\n' + re.escape(END)
    for message in messages:
        if isinstance(message.get('content'), str):
            message['content'] = re.sub(pattern, lambda match: '' if
                hashlib.sha256(match.group().encode()).hexdigest() in known else match.group(),
                message['content'])
    state = agent._repair_state
    rows = []
    for view in views:
        sha = None
        try:
            sha = digest(checked_path(Path(agent.cell['workspace']), view['file']))
        except (OSError, ValueError, AssertionError):
            pass
        valid = bool(state and state.execution_id == view['execution_id'] and sha == view['source_sha256'])
        body = view['actual_output'].partition('</tool_observation>')[2].lstrip('\n') or view['actual_output']
        present = valid and bool(body) and any(body in str(m.get('content', '')) for m in messages)
        rows.append({k: v for k, v in view.items() if k not in {'actual_output', 'recent_output'}} | {
            'current_file_content': valid, 'current_file_sha256': sha,
            'authority': 'Observed file text; no test verification authority.',
            'body_omitted': True, 'body_in_recent_messages': present,
            'reader': {'evidence_id': view['evidence_id'], 'part': 'output', 'offset': 0, 'limit': 200}})

    def render(entries):
        result = deepcopy(messages)
        info = {'files': entries, 'public_verification': {
            'registered_checks_ready': bool(finish_ready),
            'evidence': completion if finish_ready else None,
            'scope': 'Registered public tests and current diff only.',
            'limitation': 'Passing these checks does not establish that every requirement is implemented.'}}
        frame = TAG + '\n' + json.dumps(info) + '\n' + END
        for message in reversed(result):
            if message.get('role') in {'user', 'tool'} and isinstance(message.get('content'), str):
                message['content'] += '\n' + frame
                break
        return result, frame

    candidate, frame = render(rows)
    reference = cost(agent, candidate)
    remaining = agent.token_budget['limit'] - agent.token_budget['charged'] - 4096
    selected = []
    # Optional source must not itself trigger the next automatic fold.
    pressure_limit = max(1, agent.context_window - agent._compaction_reserve_tokens())
    input_limit = min(agent.max_input_tokens_per_call, pressure_limit)
    # Public test readiness must not withdraw code needed to finish uncovered work.
    for name in (() if getattr(agent, "_compaction_required_only", False) else ORDER):
        for index, view in enumerate(views):
            if view['file'] != name or not rows[index]['current_file_content'] or rows[index]['body_in_recent_messages']:
                continue
            options = [('observed_file', view['actual_output'], view.get('observed_ranges'))]
            if view.get('recent_output') and view.get('recent_range'):
                options.append(('recent_read', view['recent_output'], [view['recent_range']]))
            for scope, body, ranges in options:
                if any(body in str(message.get('content', '')) for message in messages):
                    continue
                trial = deepcopy(rows)
                trial[index].update(body_omitted=False, actual_output=body, body_scope=scope,
                    body_ranges=ranges, body_complete=scope == 'observed_file' and bool(view.get('complete')),
                    body_evidence_ids=([view['evidence_id']] if scope == 'recent_read' else
                        [row['evidence_id'] for row in view.get('source_receipts', [])]))
                proposed, new_frame = render(trial)
                if cost(agent, proposed) <= remaining and estimate(agent, proposed) <= input_limit:
                    rows, candidate, frame = trial, proposed, new_frame
                    selected.append(name)
                    break
            else:
                chunks = []
                for chunk in observed_chunks(view, messages):
                    body, ranges, identifiers = packed_body(chunks + [chunk])
                    trial = deepcopy(rows)
                    trial[index].update(body_omitted=False, actual_output=body,
                        body_scope='observed_windows', body_ranges=ranges,
                        body_complete=False, body_evidence_ids=identifiers)
                    proposed, new_frame = render(trial)
                    if cost(agent, proposed) <= remaining and estimate(agent, proposed) <= input_limit:
                        chunks.append(chunk)
                        rows, candidate, frame = trial, proposed, new_frame
                if chunks:
                    selected.append(name)
    agent._working_set_hashes = known | {hashlib.sha256(frame.encode()).hexdigest()}
    agent._working_set_projection = {
        'reference_reservation': reference, 'whole_request_reservation': cost(agent, candidate),
        'remaining_after_margin': remaining, 'selected_bodies': selected,
        'public_checks_ready': bool(finish_ready),
        'optional_body_input_limit': input_limit,
        'native_auto_fold_threshold': pressure_limit,
        'authority': 'Public verification facts; task completeness still depends on its requirements.'}
    return candidate
