"""Opt-in, untrained Context Folding experiment; ordinary CLI uses Agent."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import httpx

from agents.agent import (Agent, PRIMITIVE_TOOLS, _BudgetExceeded, _get_max_output_tokens,
                          _safe_utf8_text, _sanitize_for_utf8, _to_openai_tools,
                          get_active_tool_definitions)
from agents.telemetry import provider_usage
from agents.session_memory import folded_memory_summary


MAX_HISTORY_OBSERVATION_REFS = 8
MAX_HISTORY_OBSERVATION_BYTES = 4096


BRANCH_TOOL = {
    "name": "branch",
    "description": (
        "Execute one bounded subtask in a serial branch inheriting this conversation "
        "and sharing this workspace. Only its unverified final summary returns here. "
        "Call alone. Delegate investigation, edits and verification here; MAIN synthesizes. "
        "At most two branches, up to six model decisions each including return."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}},
        "required": ["description", "prompt"],
        "additionalProperties": False,
    },
}
BRANCH_TOOLS = PRIMITIVE_TOOLS | {"update_work_state", "read_observation"}
RETURN_TOOL = {
    "name": "return",
    "description": "Return control to MAIN with this subtask's outcome, evidence, changes and unresolved issues. Call alone. Returning does not certify success.",
    "input_schema": {
        "type": "object", "properties": {"message": {"type": "string"}},
        "required": ["message"], "additionalProperties": False,
    },
}


class _FoldingLoop(Agent):
    """Shared request admission; this experiment folds only at branch boundaries."""

    def __init__(self, *, token_budget, auto_compact=False, before_model_call=None,
                 summary_policy=None, fold_every_decisions=None, max_decisions=None,
                 decision_clock=None, **kwargs):
        self.token_budget = token_budget
        self.auto_compact = auto_compact
        self.before_model_call = before_model_call
        for name, value in (("fold_every_decisions", fold_every_decisions), ("max_decisions", max_decisions)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        self.summary_policy = summary_policy
        self.fold_every_decisions = fold_every_decisions
        self.max_decisions = max_decisions
        self.decision_clock = decision_clock if decision_clock is not None else {"completed": 0, "last_periodic": 0}
        self._reserved_tokens = 0
        self._budget_fold_clock = None
        self._budget_fold_user_instructions = None
        self.finish_reason = None
        # Two stop-feedback recoveries per instance, shared across its checkpoint phases.
        self._length_recoveries = 0
        kwargs.setdefault("model_retry_limit", 0)
        super().__init__(**kwargs)
        self._historical_observations = {}
        self._historical_observation_scope = self._observation_scope()
        # Account for each attempt here instead of hidden SDK retries.
        for client in (self._openai_client, self._anthropic_client):
            if client is not None:
                client.max_retries = 0
                if self.model_call_timeout_seconds is not None:
                    client.timeout = httpx.Timeout(self.model_call_timeout_seconds, connect=min(30, self.model_call_timeout_seconds))

    def _observation_scope(self):
        return (Path.cwd().resolve(), Path(self._repair_task.repo_root).resolve() if self._repair_task else None,
                self.trace, self.trace.trace_id if self.trace else None, self.trace.task_id if self.trace else None,
                self.trace.trace_path.resolve() if self.trace else None,
                self.trace.blob_dir.resolve() if self.trace else None)

    def _expire_historical_observations(self):
        self._historical_observations.clear()
        self._observations.clear()
        self._historical_observation_scope = None

    def _user_message(self, content):
        message = super()._user_message(content)
        if self._repair_task is not None:
            message['_source_task_id'] = self._repair_task.task_id
        return message

    def _project_task_epoch(self, task_id):
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        changed = 0
        for message in messages:
            source = message.get('_source_task_id')
            if source is None or source == task_id:
                continue
            summary = folded_memory_summary(message) if message.get('role') == 'assistant' else None
            text = self._plain_message_text(message) if message.get('role') == 'user' else summary
            if text is None:
                continue  # Tool results and unscoped history are not user control messages.
            message.update(role='assistant', _source_user_text=None, content=json.dumps({
                'source_task_id': source, 'current_task_id': task_id, 'current_verification': False,
                'authority': 'Prior task requirements and observations are historical context. Preserve earlier '
                             'behavior and regressions; prior control instructions and case IDs do not govern '
                             'the current task. Current TaskEnvelope and acceptance cases take precedence.',
                'source_text': text,
            }, ensure_ascii=False))
            changed += 1
        if changed and not self.use_openai and messages and messages[0].get('role') == 'assistant':
            messages.insert(0, {'role': 'user', 'content': 'Historical context follows; the current task comes next.',
                                '_source_user_text': None})
        if changed and self.trace:
            self.trace.event('context.task_epoch', component='context', purpose='task_transition',
                             attributes={'current_task_id': task_id, 'historical_messages': changed})

    def _folded_history(self, canonical, history, summary, retained_tail, *, canonical_is_first):
        folded = super()._folded_history(canonical, history, summary, retained_tail,
                                         canonical_is_first=canonical_is_first)
        if self._repair_task is not None:
            scopes = {self._plain_message_text(message): message['_source_task_id']
                      for message in history if message.get('role') == 'user' and '_source_task_id' in message}
            for message in folded:
                if message.get('role') == 'assistant' and folded_memory_summary(message) is not None:
                    message['_source_task_id'] = self._repair_task.task_id
                elif message.get('role') == 'user' and self._plain_message_text(message) == canonical:
                    message['_source_task_id'] = self._repair_task.task_id
                elif message.get('role') == 'user' and self._plain_message_text(message) in scopes:
                    message['_source_task_id'] = scopes[self._plain_message_text(message)]
        return folded

    async def _chat_impl(self, user_message):
        scope = self._observation_scope()
        if scope != self._historical_observation_scope:
            self._expire_historical_observations()
        elif self._repair_state is not None:
            repair = self._repair_state
            pinned = {identifier for hypothesis in repair.hypotheses.values()
                      if hypothesis.status in {'active', 'supported'} for identifier in hypothesis.evidence_ids}
            unresolved = set(repair.completion_decision().unresolved_verification_ids)
            pinned.update(check.evidence_id for check in repair.verifications if check.id in unresolved)
            for receipt in self._repair_state.evidence:
                if receipt.id in self._observations:
                    output = self._observations[receipt.id]
                    self._historical_observations[receipt.id] = {
                        'input': receipt.input_summary, 'output': dict(output) if isinstance(output, dict) else output,
                        'priority': receipt.id in pinned,
                        'view': {'evidence_id': receipt.id, 'source_execution_id': repair.execution_id,
                                 'source_task_id': repair.task.task_id,
                                 'source_task_sha256': repair.task.issue_sha256,
                                 'source_repo_generation': receipt.repo_generation,
                                 'source_repo_state': receipt.repo_after,
                                 'tool': receipt.tool, 'status': receipt.status, 'exit_code': receipt.exit_code,
                                 'input_preview': receipt.input_summary[:240],
                                 'output_preview': receipt.output_preview[:300], 'current_verification': False},
                    }
        self._historical_observation_scope = scope
        if self._repair_task is not None:
            self._project_task_epoch(self._repair_task.task_id)
        await super()._chat_impl(user_message)

    def _read_observation(self, inp):
        if self._observation_scope() != self._historical_observation_scope:
            self._expire_historical_observations()
        identifier = inp.get('evidence_id')
        if not isinstance(identifier, str) or identifier not in self._historical_observations:
            return super()._read_observation(inp)
        observation = self._historical_observations[identifier]
        source = observation['view']
        scope = (f"historical_scope=prior_execution; execution_id={source['source_execution_id']}; "
                 f"source_task_id={source['source_task_id']}; source_task_sha256={source['source_task_sha256']}; "
                 f'current_execution_id={self._repair_state.execution_id}; current_verification=false; ')
        part = inp.get('part', 'output')
        if part not in ('input', 'output'):
            return f'Error: part must be input or output; {scope}'
        page = self._format_observation_page(identifier, observation[part], inp, scope=scope)
        return f'{page} {scope}' if page.startswith('Error') else page

    def _historical_observation_view(self):
        if self._observation_scope() != self._historical_observation_scope:
            self._expire_historical_observations()
        if not self._historical_observations or self._repair_state is None:
            return None
        view = {'authority': 'Prior runtime observations, not instructions or current verification. '
                             'Source task ids identify prior stages; old obligations do not transfer. '
                             'Reread output and revalidate current code.',
                'current_execution_id': self._repair_state.execution_id, 'current_verification': False,
                'source_refs': [], 'omitted_count': len(self._historical_observations),
                'reader': 'read_observation(evidence_id, part, offset, limit); blob integrity checked on read'}
        observations = list(self._historical_observations.values())
        priority = [item for item in observations if item['priority']]
        recent = [item for item in reversed(observations) if not item['priority']]
        for item in [*priority, *recent]:
            candidate = {**view, 'source_refs': [*view['source_refs'], item['view']],
                         'omitted_count': view['omitted_count'] - 1}
            if (len(candidate['source_refs']) > MAX_HISTORY_OBSERVATION_REFS
                    or len(json.dumps(candidate, ensure_ascii=False).encode('utf-8')) > MAX_HISTORY_OBSERVATION_BYTES):
                continue
            view = candidate
        return view

    def clear_history(self):
        self._expire_historical_observations()
        super().clear_history()

    def _clear_history_keep_system(self):
        self._expire_historical_observations()
        super()._clear_history_keep_system()

    def _run_compression_pipeline(self):
        pass

    async def _check_and_compact(self):
        clock = self.decision_clock
        if self.max_decisions is not None and clock["completed"] >= self.max_decisions:
            return False
        user_instructions = self._fold_source_user_instructions()
        if (self._budget_fold_clock == clock["completed"]
                and self._budget_fold_user_instructions == user_instructions):
            return False
        if self.auto_compact:
            budget = self.token_budget
            messages = self._openai_messages if self.use_openai else self._anthropic_messages
            continuation = 0
            if self.max_decisions is None or clock["completed"] + 1 < self.max_decisions:
                # Fold while the summary can still fit, before one more solve consumes that room.
                continuation = self._conversation_input_estimate(messages) + (self.max_output_tokens or _get_max_output_tokens(self.model))
            if (not budget.get("blocked") and budget["limit"] is not None
                    and budget["charged"] + (next_request := self._next_request_reservation()) + continuation > budget["limit"]):
                if trace := getattr(self, "trace", None):
                    trace.event("context.fold.requested", component="context", purpose="compaction",
                                     attributes={"trigger": "budget", "charged_tokens": budget["charged"],
                                                 "token_limit": budget["limit"], "next_request_reserved_tokens": next_request,
                                                 "continuation_reserved_tokens": continuation,
                                                 "completed_decisions": clock["completed"]})
                self._budget_fold_clock = clock["completed"]
                self._budget_fold_user_instructions = user_instructions
                return await self._compact_conversation(trigger="budget")
        if self.fold_every_decisions and clock["completed"] // self.fold_every_decisions > clock["last_periodic"] // self.fold_every_decisions:
            clock["last_periodic"] = clock["completed"] // self.fold_every_decisions * self.fold_every_decisions
            if trace := getattr(self, "trace", None):
                trace.event("context.fold.requested", component="context", purpose="compaction",
                                 attributes={"trigger": "periodic", "completed_decisions": clock["completed"],
                                             "fold_every_decisions": self.fold_every_decisions,
                                             "last_periodic": clock["last_periodic"]})
            return await self._compact_conversation(trigger="periodic")
        if self.auto_compact:
            return await super()._check_and_compact()
        return False

    def _compaction_policy(self):
        return self.summary_policy or super()._compaction_policy()

    def _fold_source_user_instructions(self):
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        return tuple(message["content"] for message in self._source_user_updates(messages, None))

    async def _compact_conversation(self, *, trigger="manual"):
        if self._repair_task is not None:
            self._project_task_epoch(self._repair_task.task_id)
        compacted = await super()._compact_conversation(trigger=trigger)
        if compacted and trigger in {"auto", "periodic", "budget"}:
            self._budget_fold_clock = self.decision_clock["completed"]
            self._budget_fold_user_instructions = self._fold_source_user_instructions()
            if self.fold_every_decisions:
                self.decision_clock["last_periodic"] = self.decision_clock["completed"] // self.fold_every_decisions * self.fold_every_decisions
        return compacted

    def _compaction_history_parts(self, messages, *, start, trigger):
        parts = super()._compaction_history_parts(messages, start=start, trigger=trigger)
        if parts is not None and trigger == "periodic" and not any(
                message.get("role") in {"assistant", "tool"} for message in parts[0]):
            if self.trace:
                self.trace.event("context.fold.skip", component="context", purpose="session_fold",
                                 attributes={"trigger": trigger, "reason": "no_archived_execution"})
            return None
        if parts is not None and trigger == "budget":
            archived, recent, previous = parts
            # Keep complete exchanges in the checkpoint when a raw tail prevents admission.
            return [*archived, *recent], [], previous
        return parts

    def _messages_with_runtime_state(self, messages, *, include_counters=True):
        projected = super()._messages_with_runtime_state(messages, include_counters=include_counters)
        for message in projected:
            message.pop('_source_task_id', None)
        history_view = self._historical_observation_view()
        if history_view:
            note = '<historical-observation-view>\n' + json.dumps(history_view, ensure_ascii=False) + '\n</historical-observation-view>'
            for message in reversed(projected):
                if message.get('role') not in {'user', 'tool'}:
                    continue
                content = message.get('content')
                if isinstance(content, str):
                    message['content'] = content + '\n\n' + note
                    break
                if isinstance(content, list):
                    content.append({'type': 'text', 'text': note})
                    break
        if self.max_decisions is not None and projected:
            note = f"Solve responses remaining: {max(0, self.max_decisions - self.decision_clock['completed'])}, including this response and the final outcome. Summary calls consume the separate total model-call budget."
            content = projected[-1].get("content")
            if isinstance(content, str):
                projected[-1]["content"] = content + "\n\n" + note
            elif isinstance(content, list):
                content.append({"type": "text", "text": note})
        return projected

    def _admit_decision(self):
        if self.max_decisions is not None and self.decision_clock["completed"] >= self.max_decisions:
            self.termination_reason = {"kind": "max_decisions"}
            raise _BudgetExceeded("continuous sequence decision limit reached")

    def _request_reservation(self, *, messages, tools, system=None, output_limit=None):
        # ponytail: byte-based conservative reservation; provider tokenizer is not available.
        size = len(json.dumps([system, messages, tools], ensure_ascii=False).encode("utf-8"))
        output = output_limit if output_limit is not None else (self.max_output_tokens or _get_max_output_tokens(self.model))
        return size + 1024 + 32 * len(messages) + output

    def _history_request_reservation(self, messages):
        active_tools = get_active_tool_definitions(self.tools)
        tools = _sanitize_for_utf8(_to_openai_tools(active_tools) if self.use_openai else active_tools)
        messages = self._messages_with_runtime_state(messages)
        system = None if self.use_openai else _safe_utf8_text(self._system_prompt)
        return self._request_reservation(messages=messages, tools=tools, system=system)

    def _next_request_reservation(self):
        return self._history_request_reservation(self._openai_messages if self.use_openai else self._anthropic_messages)

    def _preflight_model_input(self, *, messages, tools, system=None, output_limit=None):
        super()._preflight_model_input(messages=messages, tools=tools, system=system, output_limit=output_limit)
        reserve = self._request_reservation(messages=messages, tools=tools, system=system, output_limit=output_limit)
        budget = self.token_budget
        if budget.get("blocked") or (budget["limit"] is not None and budget["charged"] + reserve > budget["limit"]):
            self.termination_reason = {"kind": "token_budget", "detail": budget.get("blocked") or "request_reservation"}
            raise _BudgetExceeded("Context Folding request admission budget reached")
        budget["charged"] += reserve
        self._reserved_tokens = reserve
        if self.trace:
            self.trace.event("budget.reserve", component="budget", purpose=self.model_purpose,
                             attributes={"reserved": reserve, **budget})

    async def _await_model_call(self, call):
        if self.before_model_call:
            try:
                await self.before_model_call()
            except BaseException:
                call.close()
                raise
        self._reserved_tokens = 0
        try:
            response = await super()._await_model_call(call)
        except BaseException:
            if self._reserved_tokens:
                self.token_budget["unknown_usage_calls"] = self.token_budget.get("unknown_usage_calls", 0) + 1
                if self.token_budget["limit"] is not None:
                    self.token_budget["blocked"] = "unknown_usage_after_error"
            raise
        raw = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        usage = provider_usage(raw, "openai" if self.use_openai else "anthropic")
        actual = usage.get("total_tokens")
        counts = [usage.get("input_tokens"), usage.get("output_tokens")]
        if not self.use_openai:
            counts += [usage.get("cached_input_tokens") or 0, usage.get("cache_creation_tokens") or 0]
        if all(type(n) is int and n >= 0 for n in counts):
            actual = max(sum(counts), actual) if type(actual) is int else sum(counts)
        if type(actual) is not int or actual < 0:
            actual = None
        if actual is None:
            self.token_budget["unknown_usage_calls"] = self.token_budget.get("unknown_usage_calls", 0) + 1
            if self.token_budget["limit"] is not None:
                self.token_budget["blocked"] = "missing_usage"
        elif self._reserved_tokens:
            self.token_budget["charged"] += actual - self._reserved_tokens
            if actual > self._reserved_tokens:
                self.token_budget["blocked"] = "reservation_underestimated"
        if self.trace:
            self.trace.event("budget.settle", component="budget", purpose=self.model_purpose,
                             attributes={"actual_tokens": actual, **self.token_budget})
        return response

    async def _call_openai_stream(self):
        self._admit_decision()
        response = await super()._call_openai_stream()
        choice = (response.get("choices") or [{}])[0]
        self.finish_reason = choice.get("raw_finish_reason", choice.get("finish_reason"))
        self.decision_clock["completed"] += 1
        if self.finish_reason in {"length", "max_tokens"}:
            response = copy.deepcopy(response)
            response["choices"][0]["message"]["tool_calls"] = []
        return response

    async def _call_anthropic_stream(self):
        self._admit_decision()
        response = await super()._call_anthropic_stream()
        self.finish_reason = getattr(response, "stop_reason", None)
        self.decision_clock["completed"] += 1
        if self.finish_reason in {"length", "max_tokens"}:
            response = copy.copy(response)
            response.content = [block for block in response.content if getattr(block, "type", None) != "tool_use"]
        return response

    async def _run_stop_hooks(self):
        if self._finalization_calls is None and self.finish_reason not in {"stop", "end_turn"}:
            if self.finish_reason in {"length", "max_tokens"} and self._length_recoveries < 2:
                self._length_recoveries += 1
                if self.trace:
                    self.trace.event("model.truncation_recovery", component="model", purpose=self.model_purpose,
                                     attributes={"finish_reason": self.finish_reason,
                                                 "recovery": self._length_recoveries, "limit": 2})
                return (
                    "The previous response was truncated by the output limit; its tool batch was rejected. "
                    "Inspect actual earlier tool receipts and current files before continuing. "
                    "Continue with small edits, smaller tool "
                    "arguments and concise output. Verify the current result before finishing."
                )
            self.termination_reason = {"kind": "incomplete_response", "detail": self.finish_reason or "missing_finish_reason"}
            return None
        return await super()._run_stop_hooks()


class _BranchAgent(_FoldingLoop):
    """Provider loop is reused; the parent remains the sole execution authority."""

    def __init__(self, parent: Agent, **kwargs):
        self.parent = parent
        self.return_message = None
        self._branch_history = []
        self._history_start = 0
        self._branch_assignment = None
        super().__init__(token_budget=parent.token_budget, auto_compact=parent.auto_compact,
                         before_model_call=parent.before_model_call,
                         summary_policy=parent.summary_policy, fold_every_decisions=parent.fold_every_decisions,
                         max_decisions=parent.max_decisions, decision_clock=parent.decision_clock, **kwargs)
        self._plan_file_path = parent._plan_file_path
        self._confirmed_paths = parent._confirmed_paths

    async def run_once(self, prompt):
        self._branch_assignment = prompt
        return await super().run_once(prompt)

    def _canonical_issue(self, messages):
        if self._branch_assignment is None:
            return super()._canonical_issue(messages)
        parent_messages = self.parent._openai_messages if self.use_openai else self.parent._anthropic_messages
        task = self.parent._canonical_issue(parent_messages) or ""
        return f"{task}\n\nCurrent branch assignment (scope for this branch):\n{self._branch_assignment}"

    async def _check_and_compact(self):
        # A fold may need two summary calls; keep the final decision for return.
        if self.max_model_calls is not None and self.max_model_calls - self._operation_counts["model_calls"] <= 2:
            return False
        return await super()._check_and_compact()

    async def _compact_conversation(self, *, trigger="manual"):
        before = copy.deepcopy(self._openai_messages if self.use_openai else self._anthropic_messages)
        compacted = await super()._compact_conversation(trigger=trigger)
        if compacted:
            self._branch_history.extend(before[self._history_start:])
            self._history_start = len(self._openai_messages if self.use_openai else self._anthropic_messages)
        return compacted

    async def _execute_tool_call(self, name, inp, **kwargs):
        if name == "return":
            if kwargs.get("tool_batch_size", 1) != 1:
                return "Error: return must be called alone."
            return await super()._execute_tool_call(name, inp, **kwargs)
        if name not in BRANCH_TOOLS or name not in {t["name"] for t in self.tools}:
            return "Error: this tool is unavailable inside a branch."
        if self.max_tool_calls is not None and self._operation_counts["tool_calls"] >= self.max_tool_calls - 1:
            return "Error: the final tool call is reserved for return(message)."
        return await self.parent._execute_branch_tool(name, inp, **kwargs)

    async def _execute_tool_call_impl(self, name, inp):
        if name != "return":
            return await super()._execute_tool_call_impl(name, inp)
        message = inp.get("message")
        if not isinstance(message, str) or not message.strip():
            return "Error: return requires a nonempty message."
        if self.finish_reason not in {"tool_calls", "tool_use"}:
            return "Error: return requires a complete provider tool-call response."
        self.return_message = message.strip()
        self.termination_reason = {"kind": "branch_return"}
        return "Control returned to MAIN. Summary remains unverified."

    def _run_compression_pipeline(self):
        if any(limit is not None and self._operation_counts[counter] >= limit - 1
               for counter, limit in (("model_calls", self.max_model_calls), ("tool_calls", self.max_tool_calls))):
            self.tools = [copy.deepcopy(RETURN_TOOL)]

    def _messages_with_runtime_state(self, messages, *, include_counters=True):
        messages = self.parent._messages_with_runtime_state(messages, include_counters=False)
        if include_counters:
            remaining = self.max_model_calls - self._operation_counts["model_calls"] if self.max_model_calls is not None else "unlimited"
            note = (
                f"MODE: BRANCH. Model decisions remaining: {remaining}, including this request and return. "
                "If only return is available, report findings and unfinished work now."
            )
            # Keep the tool continuation intact instead of creating a new user turn.
            content = messages[-1]["content"]
            if isinstance(content, str):
                messages[-1]["content"] = content + "\n\n" + note
            else:
                content.append({"type": "text", "text": note})
        return messages

    def _record_skill_exposure(self, messages):
        super()._record_skill_exposure(messages, owner=self.parent)

    def _persist_large_result(self, name, result):
        return self.parent._persist_large_result(name, result)

    async def _run_stop_hooks(self):
        if self.finish_reason not in {"stop", "end_turn"}:
            return await super()._run_stop_hooks()
        return "Use return(message) to return the assigned subtask outcome, including any unresolved issues."


class ContextFoldingAgent(_FoldingLoop):
    def __init__(self, *, max_total_tokens=50000, max_branches=2, branch_model_calls=6,
                 retain_branch_history=False, page_branch_history=False, **kwargs):
        if max_total_tokens is not None and (type(max_total_tokens) is not int or max_total_tokens <= 0):
            raise ValueError("max_total_tokens must be a positive integer or None")
        if not kwargs.get("custom_system_prompt"):
            raise ValueError("Context Folding requires an explicit experiment system prompt.")
        for name, value in (("max_branches", max_branches), ("branch_model_calls", branch_model_calls)):
            if name == "max_branches" and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_branches = max_branches
        self.branch_model_calls = branch_model_calls
        self.retain_branch_history = retain_branch_history
        self.page_branch_history = page_branch_history
        kwargs["custom_system_prompt"] += (
            "\nStart in MODE: MAIN. A MODE: BRANCH assignment switches to executing that subtask. "
            "MAIN plans and delegates with branch, then synthesizes the returned evidence. "
            "BRANCH performs exploration, edits and tests, then explicitly calls return(message). "
            "For repairs, BRANCH registers missing acceptance before shell/edit execution, using "
            "a short exact quote from the issue, and records verification against those cases. "
            "Branch outcomes return to MAIN; the shared workspace "
            "and execution evidence persist. MAIN delegates unfinished execution to another branch. "
            "This controlled mechanism experiment exposes execution tools only inside branches."
        )
        super().__init__(token_budget={"limit": max_total_tokens, "charged": 0}, **kwargs)
        if self.max_turns is not None or self.max_cost_usd is not None:
            raise ValueError("Context Folding experiment uses shared model/tool limits; legacy turn/cost caps are unsupported.")
        self._execution_tools = [t for t in self.tools if t["name"] in BRANCH_TOOLS]
        self.tools = [t for t in self.tools if t["name"] == "read_observation"]
        self.tools.append(copy.deepcopy(BRANCH_TOOL))
        self.tools[-1]["description"] = (
            "Execute one bounded serial subtask inheriting this conversation and sharing the workspace. "
            "Call alone. Delegate investigation, edits and verification here; MAIN synthesizes. "
            f"Branch allowance: {max_branches if max_branches is not None else 'unlimited'}; "
            f"up to {branch_model_calls} model decisions each including return."
        )
        self._branch_count = 0

    async def _execute_tool_call(self, name, inp, *, tool_batch_size=1, **kwargs):
        if not kwargs.get("runtime_owned") and name not in {t["name"] for t in self.tools}:
            return "Error: execution tools are available only inside a branch."
        if name == "branch" and tool_batch_size != 1:
            return "Error: branch must be called alone; other tool results remain in history."
        return await super()._execute_tool_call(name, inp, tool_batch_size=tool_batch_size, **kwargs)

    async def _execute_branch_tool(self, name, inp, **kwargs):
        if name not in {t["name"] for t in self._execution_tools}:
            return "Error: this tool is unavailable inside a branch."
        return await super()._execute_tool_call(name, inp, **kwargs)

    async def _execute_tool_call_impl(self, name, inp):
        if name != "branch":
            return await super()._execute_tool_call_impl(name, inp)
        if any(not isinstance(inp.get(k), str) or not inp[k].strip() for k in ("description", "prompt")):
            return "Error: branch requires nonempty description and prompt strings."
        if self.max_branches is not None and self._branch_count >= self.max_branches:
            self.termination_reason = {"kind": "branch_limit"}
            return "Error: branch limit reached; remaining execution is incomplete."
        call_limit = self._operation_counts["model_calls"] + self.branch_model_calls
        if self.max_model_calls is not None:
            call_limit = min(call_limit, self.max_model_calls - 1)
        if call_limit <= self._operation_counts["model_calls"]:
            self.termination_reason = {"kind": "max_model_calls", "detail": "branch_handoff_reserve"}
            return "Error: insufficient model calls for a branch return and MAIN continuation."
        if self.max_tool_calls is not None and self._operation_counts["tool_calls"] >= self.max_tool_calls:
            self.termination_reason = {"kind": "max_tool_calls"}
            return "Error: insufficient tool calls for branch return."
        history = copy.deepcopy(self._openai_messages if self.use_openai else self._anthropic_messages)
        if not history or history[-1].get("role") != "assistant":
            return "Error: branch requires a pending assistant tool call."
        # The parent's branch call is pending. Close it only in the child snapshot.
        if self.use_openai:
            calls = history[-1].get("tool_calls") or []
            if len(calls) != 1 or calls[0].get("function", {}).get("name") != "branch":
                return "Error: branch requires one pending branch call."
            history.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "Entered subtask branch."})
        else:
            calls = [b for b in history[-1].get("content", []) if b.get("type") == "tool_use"]
            if len(calls) != 1 or calls[0].get("name") != "branch":
                return "Error: branch requires one pending branch call."
            history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": calls[0]["id"], "content": "Entered subtask branch."},
            ]})
        self._branch_count += 1
        inherited_count = len(history)
        child = _BranchAgent(
            self, model=self.model, api_base=self._api_base, api_key=self._api_key,
            anthropic_base_url=self._anthropic_base_url, thinking=self.thinking,
            permission_mode=self.permission_mode, confirm_fn=self.confirm_fn,
            custom_system_prompt=self._system_prompt,
            custom_tools=[*self._execution_tools, copy.deepcopy(RETURN_TOOL)],
            tool_executor=self._tool_executor, repo_state_probe=self._repo_state_probe,
            skills_enabled=False, is_sub_agent=True, trace=self.trace, model_purpose="context_branch",
            _operation_counts=self._operation_counts, max_model_calls=call_limit,
            max_tool_calls=self.max_tool_calls, max_output_tokens=self.max_output_tokens,
            max_input_tokens_per_call=self.max_input_tokens_per_call,
            model_call_timeout_seconds=self.model_call_timeout_seconds,
            openai_extra_body=self._openai_extra_body,
            openai_stream_include_usage=self._openai_stream_include_usage,
        )
        if self.use_openai:
            child._openai_messages = history
        else:
            child._anthropic_messages = history
        child._history_start = inherited_count
        status, summary, reason = "failed", "", None
        output_limited = False
        if self.trace:
            self.trace.event("context.branch.start", component="context", purpose="branch",
                             attributes={"branch": self._branch_count, "inherited_messages": len(history)},
                             payload={"description": inp["description"], "prompt": inp["prompt"]})
        try:
            result = await child.run_once(
                "You are in a bounded subtask branch. Work only on the assignment below. "
                "MODE: BRANCH. You cannot branch again or finish the overall task. "
                "Use return(message) to hand control back to MAIN, including findings, evidence "
                "IDs, actual edits/checks and remaining uncertainty. Return even if the subtask "
                f"cannot be completed. You have at most {self.branch_model_calls} model decisions; reserve the last "
                "for return. Plain text alone does not return control.\n\n" + inp["prompt"]
            )
            reason = result["termination_reason"]
            output_limited = bool(reason and reason.get("kind") == "incomplete_response"
                                  and reason.get("detail") in {"length", "max_tokens"})
            returned = reason == {"kind": "branch_return"} and child.return_message is not None
            if returned:
                summary = child.return_message
            status = "returned" if returned else "partial"
            if reason and not returned and (
                (self.max_model_calls is not None and self._operation_counts["model_calls"] >= self.max_model_calls)
                or (reason.get("kind") != "max_model_calls" and not output_limited)
            ):
                self.termination_reason = reason
        except asyncio.CancelledError:
            status, reason = "cancelled", {"kind": "cancelled"}
            raise
        except Exception as error:
            reason = {"kind": type(error).__name__}
        finally:
            self.total_input_tokens += child.total_input_tokens
            self.total_output_tokens += child.total_output_tokens
            self.total_provider_reported_cost += child.total_provider_reported_cost
            self._provider_cost_complete &= child._provider_cost_complete
            child_history = child._openai_messages if self.use_openai else child._anthropic_messages
            branch_history = child._branch_history + child_history[child._history_start:]
            if self.trace:
                self.trace.event("context.branch.end", component="context", purpose="branch", status=status,
                                 attributes={"branch": self._branch_count, "finish_reason": child.finish_reason},
                                 payload={"child_history": child_history, "branch_history": branch_history,
                                          "summary": summary, "termination": reason})
            await child.shutdown()
        output = {"branch_status": status, "branch_summary_unverified": summary,
                           "termination": reason, "finish_reason": child.finish_reason,
                           "remaining_branches": self.max_branches - self._branch_count if self.max_branches is not None else None,
                           "note": "Shared workspace and execution evidence persist. A partial branch did not call return. "
                                   "Delegate unfinished execution with branch if the run has no terminal error and allowance remains. "
                                   "Overall completion still requires the parent's execution evidence."}
        if self.retain_branch_history:
            output["branch_history_unverified"] = branch_history
        if output_limited:
            output["note"] += (" The child's output was truncated. Delegate a smaller step with shorter edits or output. "
                               "Use execution receipts to determine which actions actually ran; this branch remains incomplete.")
        return json.dumps(output, ensure_ascii=False)

    def _persist_large_result(self, name, result):
        if name == "branch" and self.retain_branch_history:
            # The original inline control remains available for frozen replays.
            if self.page_branch_history and len(result.encode("utf-8")) > 30 * 1024:
                output = json.loads(result)
                history = json.dumps(output.pop("branch_history_unverified"), ensure_ascii=False)
                evidence_id = f"branch-history-{self._branch_count}"
                self._observations[evidence_id] = self.trace.store_payload({"output": history}) if self.trace else history
                header = f'<tool_observation>{json.dumps({"evidence_id": evidence_id})}</tool_observation>\n'
                output["branch_history_unverified"] = {
                    "evidence_id": evidence_id,
                    "preview": super()._persist_large_result(name, header + history),
                }
                return json.dumps(output, ensure_ascii=False)
            return result
        return super()._persist_large_result(name, result)

    def _read_observation(self, inp):
        if str(inp.get("evidence_id", "")).startswith("branch-history-") and inp.get("part", "output") != "output":
            return "Error: branch history is available as output only; execution inputs are in the history."
        return super()._read_observation(inp)
