"""Keep protected public checks independent of mutable actor assertions."""
import hashlib
import json


ACTOR_COMMAND = 'python -B -m pytest test_actor.py -q -c pytest.ini'
EMPTY_ACTOR = b'# Validated public-contract assertions may be added here.\n'


def install(caller, parent, *, empty_actor_sha256):
    legacy = caller.legacy
    original_command, original_scope = legacy.public_command, legacy.scoped

    def public_command(checkpoint):
        return original_command(checkpoint).replace(' test_actor.py ', ' ')

    def scoped(name, inp, work, checkpoint, protected):
        if name == 'run_shell' and inp.get('command') == ACTOR_COMMAND:
            assert not inp.get('acceptance_ids') and not inp.get('diagnostic')
            original_scope('read_file', {'file_path': 'test_actor.py'}, work, checkpoint, protected)
            return dict(inp)
        return original_scope(name, inp, work, checkpoint, protected)

    legacy.public_command, legacy.scoped = public_command, scoped
    caller.SHELL += (' Validated actor assertions run separately with exactly "' + ACTOR_COMMAND +
                     '", without acceptance_ids or diagnostic=true.')

    class Candidate(parent):
        async def _verify_current(self):
            await super()._verify_current()
            state = self._repair_state
            key = (state.execution_id, state.repo_generation, state.current_repo_state)
            if key == getattr(self, '_actor_checked_key', None):
                return
            # An unchanged fixture contains no assertions; pytest exit 5 is not a pass.
            path = caller.checked_path(self.workspace, 'test_actor.py')
            if hashlib.sha256(path.read_bytes()).hexdigest() != empty_actor_sha256:
                first_evidence = len(state.evidence)
                output = await super()._execute_tool_call('run_shell', {
                    'command': ACTOR_COMMAND, 'acceptance_ids': [],
                    'scope': 'focused', 'oracle': 'existing_test',
                }, runtime_owned=True)
                evidence = next(item for item in reversed(state.evidence[first_evidence:])
                                if item.tool == 'run_shell' and item.input_summary == ACTOR_COMMAND)
                check = next((item for item in reversed(state.verifications)
                              if item.evidence_id == evidence.id), None)
                note = dict(evidence_id=evidence.id, execution_id=state.execution_id,
                    repo_generation=evidence.repo_generation, repo_state=evidence.repo_after,
                    success=check.success if check else None,
                    runner_recognized=check.runner_recognized if check else False,
                    test_cases=evidence.test_cases, actual_output=str(output),
                    verification_id=check.id if check else None,
                    classification='verification' if check else 'inconclusive; no validated assertions',
                    scope='Actor assertions; no public acceptance authority.')
                self._pending_public_notes.append({'role': 'user', 'content':
                    '<runtime-actor-test>\n' + json.dumps(note, ensure_ascii=False) +
                    '\n</runtime-actor-test>', '_source_user_text': None})
            self._actor_checked_key = (state.execution_id, state.repo_generation, state.current_repo_state)
            # A successful actor recheck may resolve an earlier actor failure.
            await super()._verify_current()

    return Candidate
