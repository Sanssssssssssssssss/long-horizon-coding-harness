"""Draft shared SCB caller. Original runtime and frozen experiments stay unchanged."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R2 = HERE.parent / '20261012-skill-context-results-round2'
sys.path.insert(0, str(ROOT))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

legacy = load('full_pair_original_caller', R2 / 'context_run.py')
cache = load('full_pair_ready_cache', R2 / 'skill-ready-to-report-023/working_set.py')
costs = load('full_pair_working_cost', R2 / 'working_context.py')
pages = load('full_pair_observed_pages', HERE.parent / '20261014-context-observation-pages/observed_pages.py')
from agents.agent import Agent
from benchmarks.context_folding import _FoldingLoop
from benchmarks.folding_checkpoint import EvidenceFoldingLoop, _required_reverification

PARAMS = {**legacy.PARAMS, 'allocation': 2000000, 'phase_solve_responses': 80,
          'max_model_calls': 320, 'max_tool_calls': 1500, 'summary_calls': 160,
          'max_output_tokens': 32768, 'keep_recent_tokens': 8192, 'auto_compact': True}
legacy.PARAMS = PARAMS
SHELL = ('Run one Windows cmd.exe command; no chains, pipeline, redirection, ls or git -C. '
         'Use read_file/list_files for inspection. Allowed: the supplied cumulative public pytest; '
         'diagnostic_NAME.py pytest with diagnostic=true/no acceptance_ids; del a new untracked '
         'diagnostic draft; python -B circopt.py check/eval/help/version/json; git status --short; '
         'git diff with only --check/--stat/--name-only/--. Existing scope validates arguments. ')
SYSTEM = legacy.SYSTEM.replace(
    'Register the supplied acceptance cases together with update_work_state before editing or Shell.',
    'Current acceptance cases are registered by the host; do not register or replace those IDs.') + (
    '\nCurrent full public pytest and actual current diff are required before reporting completion. '
    'The runtime checks the cumulative public suite initially and after actual changes. '
    'Preserve passing behaviors and use the actual failed public results; do not infer missing '
    'implementation merely from a mixed failing suite. When native current readiness is true, report now.')

def prompt(cp):
    prior = ', '.join(f'prior_task_{i}.md' for i in range(1, cp)) or 'none'
    header = (f'Current checkpoint: {cp}. Host acceptance IDs are already registered; do not re-register.\n'
              f'Public command: {legacy.public_command(cp)}\n'
              f'Protected historical requirement files: {prior}. Preserve cumulative public behavior.\n'
              'Use current observations. Temporary diagnostics stay in diagnostic_NAME.py.\n')
    return header + '\n' + (legacy.data.FROZEN / f'checkpoint_{cp}.md').read_text(encoding='utf-8')

def checked_path(work, value):
    name = legacy.scoped('read_file', {'file_path': value}, work, 1, {})['file_path']
    return work / name

class NativeTail(_FoldingLoop):
    async def _run_compaction_query(self, *args, **kwargs):
        kwargs['retry_with_full_output'] = False
        return await super()._run_compaction_query(*args, **kwargs)

    def _compaction_history_parts(self, messages, *, start, trigger):
        if trigger == 'budget':
            return Agent._compaction_history_parts(self, messages, start=start, trigger=trigger)
        return super()._compaction_history_parts(messages, start=start, trigger=trigger)

class SharedCaller(legacy.CommonScope):
    active = None

    def __init__(self, **kwargs):
        kwargs['auto_compact'] = True
        super().__init__(**kwargs)
        SharedCaller.active = self
        self.cell = {'workspace': str(self.workspace)}
        self._working_sources = []
        self._observed_pages = pages.ObservedPages()
        self._pending_public_notes = []
        self._closing_anchor = None
        self._online_evolution_enabled = lambda: False
        for tool in self.tools:
            if tool['name'] == 'run_shell':
                tool['description'] = SHELL

    def _user_message(self, content):
        message = super()._user_message(content)
        if content == prompt(self.checkpoint):
            spec = self._repair_task.canonical_issue
            message['_source_user_text'] = content.replace(spec, '', 1).strip()
        return message

    def _flush_public(self):
        self._openai_messages.extend(self._pending_public_notes)
        self._pending_public_notes = []

    async def _check_and_compact(self):
        self._flush_public()
        return await super()._check_and_compact()

    def _format_tool_observation(self, name, result, **kwargs):
        returned = super()._format_tool_observation(name, result, **kwargs)
        r, identifier = self._repair_state, kwargs.get('evidence_id')
        if name == 'read_file' and r and identifier:
            e = next(e for e in r.evidence if e.id == identifier)
            if e.status != 'ok':
                return returned
            try:
                inp = json.loads(e.input_summary)
                path = checked_path(self.workspace, inp['file_path'])
            except (OSError, ValueError, TypeError, KeyError, UnicodeError):
                return returned
            view = self._observed_pages.observe(r, e, result, path, path.relative_to(self.workspace).as_posix())
            if view is not None:
                self._working_sources = [v for v in self._working_sources if v['file'] != view['file']] + [view]
        return returned

    async def _verify_current(self):
        r = self._repair_state
        for identifier, behavior, quote in legacy.CASES[self.checkpoint]:
            if identifier not in r.acceptance_cases:
                r.register_acceptance_case(identifier, behavior, quote)
        key = (r.execution_id, r.repo_generation, r.current_repo_state)
        if key != getattr(self, '_checked_key', None):
            output = await super()._execute_tool_call('run_shell', {
                'command': legacy.public_command(self.checkpoint),
                'acceptance_ids': [c[0] for c in legacy.CASES[self.checkpoint]],
                'scope': 'full', 'oracle': 'existing_test'}, runtime_owned=True)
            check = r.verifications[-1]
            e = next(e for e in r.evidence if e.id == check.evidence_id)
            self._checked_key = (r.execution_id, check.repo_generation, check.repo_state)
            note = dict(evidence_id=e.id, execution_id=r.execution_id, repo_generation=check.repo_generation,
                        repo_state=check.repo_state, success=check.success, runner_recognized=check.runner_recognized,
                        test_cases=e.test_cases, actual_output=str(output), verification_id=check.id)
            self._pending_public_notes.append({'role': 'user', 'content': '<runtime-public-test>\n' +
                json.dumps(note, ensure_ascii=False) + '\n</runtime-public-test>', '_source_user_text': None})
        self._closing_anchor = None
        formal = next((v for v in reversed(r.verifications) if v.success and v.runner_recognized and
            v.scope == 'full' and v.oracle == 'existing_test' and v.repo_generation == r.repo_generation and
            v.repo_state == r.current_repo_state and set(r.acceptance_cases) <= set(v.acceptance_ids)), None)
        if formal:
            diffs = [v for v in r.verifications if v.scope == 'diff' and
                     v.repo_generation == r.repo_generation and v.repo_state == r.current_repo_state]
            if not diffs or not diffs[-1].success:
                await super()._execute_tool_call('run_shell', {'command': self._diff_check_command}, runtime_owned=True)
                diffs = [v for v in r.verifications if v.scope == 'diff' and
                         v.repo_generation == r.repo_generation and v.repo_state == r.current_repo_state]
            if diffs and diffs[-1].success and r.completion_decision().complete:
                self._closing_anchor = {'identity': (r.execution_id, r.repo_generation, r.current_repo_state),
                    'formal_id': formal.id, 'diff_id': diffs[-1].id}

    def finish_ready(self):
        r, anchor = self._repair_state, self._closing_anchor
        return bool(r and anchor and tuple(anchor['identity']) ==
                    (r.execution_id, r.repo_generation, r.current_repo_state) and r.completion_decision().complete)

    async def _execute_tool_call(self, name, inp, **kwargs):
        before = self._repair_state.repo_generation if self._repair_state else None
        output = await super()._execute_tool_call(name, inp, **kwargs)
        r = self._repair_state
        if r and before is not None and r.repo_generation != before and not kwargs.get('runtime_owned'):
            await super()._execute_tool_call('read_file', {'file_path': 'circopt.py', 'limit': 1000}, runtime_owned=True)
            await self._verify_current()
        return output

    async def _call_openai_stream(self):
        r = self._repair_state
        if r.execution_id != getattr(self, '_primed_execution', None):
            for name in ('circopt.py', f'test_checkpoint_{self.checkpoint}.py'):
                await super()._execute_tool_call('read_file', {'file_path': name, 'limit': 1000}, runtime_owned=True)
            self._primed_execution = r.execution_id
        await self._verify_current()
        self._flush_public()
        return await super()._call_openai_stream()

    def _messages_with_runtime_state(self, messages, *, include_counters=True):
        projected = super()._messages_with_runtime_state(messages, include_counters=include_counters)
        r = self._repair_state
        obligation = _required_reverification(r) if r else None
        if obligation:
            note = '<runtime-required-reverification>\n' + json.dumps(obligation, ensure_ascii=False) + '\n</runtime-required-reverification>'
            if not any(note in str(m.get('content', '')) for m in projected):
                next(m for m in reversed(projected) if m.get('role') in {'user', 'tool'})['content'] += '\n' + note
        # Same caller policy in both arms; only actually read current files are eligible.
        names = [v['file'] for v in reversed(self._working_sources)
                 if v['file'] != 'TASK.md' and not v['file'].startswith('prior_task_')]
        cache.ORDER = tuple(dict.fromkeys(['circopt.py', *names]))
        return cache.project(self, projected, self._working_sources, checked_path=checked_path,
            digest=legacy.data.digest, cost=costs.request_cost, estimate=costs.request_input_estimate,
            finish_ready=self.finish_ready(), completion=self._closing_anchor)

class TextCaller(SharedCaller, NativeTail):
    pass

class EvidenceCaller(SharedCaller, EvidenceFoldingLoop, NativeTail):
    pass
