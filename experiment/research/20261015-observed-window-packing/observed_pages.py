"""Bounded current source windows, assembled exclusively from observed read results."""
from collections import OrderedDict
import hashlib
import json

from agents.tools import format_file_window


class ObservedPages:
    """Keep partial reads useful; invalidate all pages on execution or content changes."""

    def __init__(self):
        self.files = OrderedDict()

    def observe(self, state, evidence, result, path, relative):
        if (evidence.tool != 'read_file' or evidence.status != 'ok'
                or not evidence.id.startswith(state.execution_id + ':')):
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
            entry = {'identity': identity, 'pages': OrderedDict()}
        offset = inp.get('offset', 0)
        lines = str(result).split('\n')
        # Redundant reads must not evict wider observations of this same revision.
        covered = {index for begin, (_, page) in entry['pages'].items()
                   for index in range(begin, begin + len(page))}
        if any(index not in covered for index in range(offset, offset + len(lines))):
            previous = entry['pages'].pop(offset, None)
            entry['pages'][offset] = (previous if previous and len(previous[1]) > len(lines)
                                      else (evidence.id, lines))
        while len(entry['pages']) > 8:
            entry['pages'].popitem(last=False)
        self.files[relative] = entry
        while len(self.files) > 8:
            self.files.popitem(last=False)
        # Keep observed coverage within the file/page bounds above. The request
        # projector admits complete bodies against the actual remaining token budget.
        selected, receipts = {}, []
        for begin, (identifier, page) in reversed(entry['pages'].items()):
            kept = []
            for index, line in enumerate(page, begin):
                if index in selected:
                    continue
                selected[index] = line
                kept.append(index)
            if kept:
                receipts.append({'evidence_id': identifier, 'offset': begin, 'line_count': len(page)})
        parts, ranges, last = [], [], None
        for index in sorted(selected):
            if last is None or index != last + 1:
                if last is not None:
                    parts.append('[unobserved or omitted lines]')
                ranges.append([index, index + 1])
            else:
                ranges[-1][1] = index + 1
            parts.append(selected[index])
            last = index
        return dict(file=relative, source_sha256=sha, execution_id=state.execution_id,
            evidence_id=evidence.id, repo_generation=evidence.repo_generation, repo_state=evidence.repo_after,
            verification=False, source_receipts=receipts, assembled_from_native_pages=True,
            observed_ranges=ranges, total_lines=total, complete=len(selected) == total,
            recent_output=str(result), recent_range=[offset, offset + len(lines)],
            actual_output='\n'.join(parts))
