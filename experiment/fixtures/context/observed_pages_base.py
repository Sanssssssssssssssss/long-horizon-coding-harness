"""Cache complete file views assembled only from successful native read pages."""
from collections import OrderedDict
import hashlib
import json

from agents.tools import format_file_window


class ObservedPages:
    """Bound each cache to 8 files, 20k lines and 512 KiB per file revision."""

    def __init__(self):
        self.files = OrderedDict()

    def observe(self, state, evidence, result, path, relative):
        if evidence.tool != 'read_file' or evidence.status != 'ok':
            return None
        if not evidence.id.startswith(state.execution_id + ':'):
            return None
        try:
            inp = json.loads(evidence.input_summary)
            raw = path.read_bytes()
            if len(raw) > 512 * 1024:
                self.files.pop(relative, None)
                return None
            text = raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
            total = len(text.split('\n'))
            if total > 20000 or not result or str(result) != format_file_window(text, inp):
                return None
        except (OSError, ValueError, TypeError, KeyError):
            return None
        sha = hashlib.sha256(raw).hexdigest()
        identity = (state.execution_id, sha)
        entry = self.files.pop(relative, None)
        if not entry or entry['identity'] != identity:
            entry = {'identity': identity, 'lines': {}, 'receipts': []}
        offset = inp.get('offset', 0)
        page = str(result).split('\n')
        added = any(i not in entry['lines'] for i in range(offset, offset + len(page)))
        if added:
            # Store the actual returned lines, never the unread host file contents.
            entry['lines'].update(enumerate(page, offset))
            entry['receipts'].append({'evidence_id': evidence.id, 'offset': offset, 'line_count': len(page)})
        self.files[relative] = entry
        while len(self.files) > 8:
            self.files.popitem(last=False)
        if len(entry['lines']) != total or any(i not in entry['lines'] for i in range(total)):
            return None
        return dict(file=relative, source_sha256=sha, execution_id=state.execution_id,
            evidence_id=evidence.id, repo_generation=evidence.repo_generation, repo_state=evidence.repo_after,
            verification=False, source_receipts=list(entry['receipts']), assembled_from_native_pages=True,
            actual_output='\n'.join(entry['lines'][i] for i in range(total)))
