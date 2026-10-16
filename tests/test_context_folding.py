"""Protocol and execution-evidence checks for the opt-in branch experiment."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from agents.agent import Agent, _BudgetExceeded
from agents.repair_state import RepairState, TaskEnvelope
from agents.telemetry import TraceRecorder, read_trace
from agents.tool_result import ShellResult
from agents.tools import get_solver_tool_definitions
from benchmarks.context_folding import ContextFoldingAgent, _BranchAgent, _FoldingLoop


@pytest.mark.parametrize("number", [1, 2, 3])
def test_periodic_admission_uses_archived_history_from_actual_trace(tmp_path, monkeypatch, number):
    root = Path(__file__).resolve().parents[1]
    folder = root / "experiment/fixtures/folding" / f"{number:03}"
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    before = json.loads((folder / "before.json").read_text(encoding="utf-8"))
    parts = receipt["parts"]
    selected = (parts["archived"], parts["retained_tail"], parts["previous_checkpoint"])
    a = parent(tmp_path)
    a._openai_messages = before["messages"]
    frozen = copy.deepcopy(a._openai_messages)
    assert any(message.get("role") in {"assistant", "tool"} for message in frozen)
    monkeypatch.setattr(Agent, "_compaction_history_parts", lambda *args, **kwargs: copy.deepcopy(selected))
    calls = []

    async def summary(*args, **kwargs):
        calls.append(args)
        return "The existing recorded facts remain available for task continuation."

    monkeypatch.setattr(a, "_generate_folded_session_memory", summary)

    async def run():
        try:
            if number < 3:
                assert await a._compact_conversation(trigger="periodic") is False
                assert not calls and a._openai_messages == frozen
                assert a._operation_counts["model_calls"] == 0 and a._fold_count == 0
                assert a._failed_compaction_signature is None
            else:
                actual = a._compaction_history_parts(frozen, start=2, trigger="periodic")
                assert actual == selected
                assert receipt["archived_assistant"] == 5 and receipt["archived_tool"] == 13
        finally:
            await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("trigger", ["periodic", "auto", "budget", "manual", "tool"])
def test_user_only_archive_blocks_only_periodic_fold(tmp_path, monkeypatch, provider, trigger):
    a = parent(tmp_path, provider)
    archived = [{"role": "user", "content": "Keep public compatibility and preserve the current fix."}]
    tail = [{"role": "assistant", "content": "Recent observed execution is retained."}]
    previous = "An earlier checkpoint with verified state."
    parts = (archived, tail, previous)
    monkeypatch.setattr(Agent, "_compaction_history_parts", lambda *args, **kwargs: copy.deepcopy(parts))
    actual = a._compaction_history_parts(archived + tail, start=0, trigger=trigger)
    if trigger == "periodic":
        assert actual is None
    elif trigger == "budget":
        assert actual == (archived + tail, [], previous)
    else:
        assert actual == parts
    asyncio.run(a.shutdown())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("prior_failure", [None, "a previous genuine failure"])
def test_skipped_periodic_fold_does_not_block_budget_fold_on_same_messages(tmp_path, monkeypatch, provider, prior_failure):
    a = parent(tmp_path, provider)
    a._failed_compaction_signature = prior_failure
    archived = [{"role": "user", "content": "Preserve the current behavioral contract."}]
    tail = [{"role": "assistant", "content": "Latest execution evidence with an intact exchange."}]
    monkeypatch.setattr(Agent, "_compaction_history_parts",
                        lambda *args, **kwargs: (copy.deepcopy(archived), copy.deepcopy(tail), None))
    calls = []

    async def summary(history, *args):
        calls.append(history)
        return "The latest verified execution is preserved together with the public behavioral contract."

    monkeypatch.setattr(a, "_generate_folded_session_memory", summary)

    async def run():
        try:
            assert await a._compact_conversation(trigger="periodic") is False
            assert not calls
            assert a._failed_compaction_signature == prior_failure
            assert await a._compact_conversation(trigger="budget") is True
            assert calls == [archived + tail]
            assert a._fold_count == 1
        finally:
            await a.shutdown()

    asyncio.run(run())


def parent(tmp_path, provider="openai", **options):
    a = ContextFoldingAgent(
        api_base="https://example.invalid/v1" if provider == "openai" else None,
        api_key="test", model="test", custom_system_prompt="test instructions",
        custom_tools=get_solver_tool_definitions("B1"), skills_enabled=False, is_sub_agent=True,
        permission_mode="bypassPermissions", repair_task=TaskEnvelope("case", "fix value", "/repo"),
        repo_state_probe=lambda: "base", max_model_calls=12,
        trace=TraceRecorder(tmp_path, run_id="r", task_id="t", attempt_id="a"),
        **options,
    )
    if provider == "openai":
        a._openai_messages += [{"role": "user", "content": "fix value"},
                              {"role": "assistant", "content": "", "tool_calls": [
                                  {"id": "branch-1", "type": "function", "function": {
                                      "name": "branch", "arguments": "{}"}}]}]
    else:
        a._anthropic_messages = [{"role": "user", "content": "fix value"},
                                {"role": "assistant", "content": [
                                    {"type": "tool_use", "id": "branch-1", "name": "branch", "input": {}}]}]
    return a


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_budget_pressure_folds_before_request_admission(provider):
    budget = {"limit": 50000, "charged": 0}
    agent = _FoldingLoop(token_budget=budget, auto_compact=True,
        api_base="https://example.invalid/v1" if provider == "openai" else None,
        api_key="test", model="test", custom_system_prompt="fixed", custom_tools=[],
        skills_enabled=False, is_sub_agent=True, max_output_tokens=8192,
        max_input_tokens_per_call=32768)
    messages = [{"role": "user", "content": "Implement parser."},
        {"role": "assistant", "content": "Archived inspection. " * 1100},
        {"role": "user", "content": "Keep the public API unchanged.", "_source_user_text": "Keep the public API unchanged."},
        {"role": "assistant", "content": "Latest work is complete; execute it next."}]
    if provider == "openai":
        agent._openai_messages = [{"role": "system", "content": "fixed"}, *messages]
    else:
        agent._anthropic_messages = messages
    budget["limit"] = agent._next_request_reservation() + 1
    assert agent._next_request_reservation() < budget["limit"]
    assert agent._conversation_input_estimate(messages) < agent.context_window - agent._compaction_reserve_tokens()
    calls = []

    async def summarize(*args):
        calls.append(args)
        budget["charged"] += 700
        return "Parser implementation is complete; execute it and check outputs."

    agent._generate_folded_session_memory = summarize

    async def run():
        assert await agent._check_and_compact()
        assert len(calls) == 1
        projected = agent._openai_request_messages() if provider == "openai" else agent._anthropic_request_messages()
        assert "Keep the public API unchanged." in json.dumps(projected)
        reserve = agent._next_request_reservation()
        agent._preflight_model_input(messages=projected, tools=[], system=None if provider == "openai" else agent._system_prompt)
        assert agent._reserved_tokens == reserve
        assert budget["charged"] <= budget["limit"]
        await agent.shutdown()

    asyncio.run(run())


def test_budget_fold_does_not_repeat_failed_unchanged_history(monkeypatch):
    agent = _FoldingLoop(token_budget={"limit": 1, "charged": 0}, auto_compact=True,
        api_base="https://example.invalid/v1", api_key="test", model="test",
        custom_system_prompt="fixed", custom_tools=[], skills_enabled=False, is_sub_agent=True)
    agent._openai_messages.append({"role": "user", "content": "Task"})
    attempts = []

    async def fail(self, *, trigger):
        attempts.append(trigger)
        return False

    monkeypatch.setattr(Agent, "_compact_openai", fail)

    async def run():
        assert not await agent._check_and_compact()
        assert not await agent._check_and_compact()
        assert attempts == ["budget"]
        agent.token_budget["blocked"] = "missing_usage"
        assert not await agent._check_and_compact()
        assert attempts == ["budget"]
        await agent.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("auto_compact", [False, True])
def test_exhausted_solve_budget_does_not_pay_for_unusable_summary(monkeypatch, provider, auto_compact):
    agent = _FoldingLoop(token_budget={"limit": 1, "charged": 0}, auto_compact=auto_compact,
        fold_every_decisions=2, max_decisions=8, decision_clock={"completed": 8, "last_periodic": 6},
        api_base="https://example.invalid/v1" if provider == "openai" else None,
        api_key="test", model="test", custom_system_prompt="fixed", custom_tools=[],
        skills_enabled=False, is_sub_agent=True)
    attempts = []

    async def summarize(self, *, trigger="manual"):
        attempts.append(trigger)
        return True

    monkeypatch.setattr(_FoldingLoop, "_compact_conversation", summarize)

    async def run():
        assert not await agent._check_and_compact()
        assert attempts == []
        with pytest.raises(_BudgetExceeded):
            agent._admit_decision()
        assert agent.termination_reason == {"kind": "max_decisions"}
        await agent.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("trigger", ["budget", "periodic", "auto"])
def test_successful_fold_preserves_next_solve_before_another_budget_fold(monkeypatch, trigger):
    agent = _FoldingLoop(token_budget={"limit": 1, "charged": 0}, auto_compact=True,
        api_base="https://example.invalid/v1", api_key="test", model="test",
        custom_system_prompt="fixed", custom_tools=[], skills_enabled=False, is_sub_agent=True)
    agent._openai_messages.append({"role": "user", "content": "Task"})
    attempts = []

    async def succeed(self, *, trigger):
        attempts.append(trigger)
        return True

    monkeypatch.setattr(Agent, "_compact_openai", succeed)

    async def run():
        assert await agent._compact_conversation(trigger=trigger)
        await agent._check_and_compact()
        assert attempts == [trigger]
        agent.decision_clock["completed"] += 1
        assert await agent._check_and_compact()
        assert attempts == [trigger, "budget"]
        await agent.shutdown()

    asyncio.run(run())


def test_budget_pressure_takes_priority_over_due_periodic_fold(monkeypatch):
    agent = _FoldingLoop(token_budget={"limit": 1, "charged": 0}, auto_compact=True,
        fold_every_decisions=6, decision_clock={"completed": 6, "last_periodic": 0},
        api_base="https://example.invalid/v1", api_key="test", model="test",
        custom_system_prompt="fixed", custom_tools=[], skills_enabled=False, is_sub_agent=True)
    attempts = []

    async def succeed(self, *, trigger):
        attempts.append(trigger)
        return True

    monkeypatch.setattr(Agent, "_compact_openai", succeed)

    async def run():
        assert await agent._check_and_compact()
        assert attempts == ["budget"]
        assert agent.decision_clock["last_periodic"] == 6
        assert not await agent._check_and_compact()
        assert attempts == ["budget"]
        await agent.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_budget_fold_archives_latest_completed_tool_pair_together(provider):
    agent = _FoldingLoop(token_budget={"limit": 50000, "charged": 0},
        api_base="https://example.invalid/v1" if provider == "openai" else None,
        api_key="test", model="test", custom_system_prompt="fixed", custom_tools=[],
        skills_enabled=False, is_sub_agent=True, max_output_tokens=8192,
        max_input_tokens_per_call=32768)
    old = {"role": "assistant", "content": "Earlier investigation. " * 300}
    if provider == "openai":
        pair = [{"role": "assistant", "content": "", "tool_calls": [{"id": "write", "type": "function",
            "function": {"name": "write_file", "arguments": json.dumps({"content": "code " * 600})}}]},
            {"role": "tool", "tool_call_id": "write", "content": "Wrote solution.py; e0007."}]
        messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": "Task"}, old, *pair]
        agent._openai_messages = messages
        start = 2
    else:
        pair = [{"role": "assistant", "content": [{"type": "tool_use", "id": "write", "name": "write_file",
            "input": {"content": "code " * 600}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "write", "content": "Wrote solution.py; e0007."}]}]
        messages = [{"role": "user", "content": "Task"}, old, *pair]
        agent._anthropic_messages = messages
        start = 1
    ordinary = agent._compaction_history_parts(messages, start=start, trigger="auto")
    assert ordinary is not None and ordinary[1] == pair
    archived, tail, previous = agent._compaction_history_parts(messages, start=start, trigger="budget")
    assert archived == [old, *pair] and tail == [] and previous is None
    asyncio.run(agent.shutdown())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("retain", [False, True])
def test_real_child_loop_folds_history_but_keeps_parent_evidence(tmp_path, monkeypatch, provider, retain):
    repo = {"state": "base"}

    async def execute(name, inp):
        assert name == "run_shell"
        repo["state"] = "changed"
        return ShellResult(stdout="command failed after changing a file", exit_code=1)

    a = parent(tmp_path, provider, tool_executor=execute, retain_branch_history=retain)
    a._repo_state_probe = lambda: repo["state"]
    original = copy.deepcopy(a._openai_messages if a.use_openai else a._anthropic_messages)
    seen = []

    def response(self):
        self._reserve_call("model_calls", self.max_model_calls)
        seen.append(self)
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        if len(seen) == 1:
            assert messages[:len(original)] == original
            closed = messages[len(original)]
            assert (closed.get("tool_call_id") == "branch-1" if self.use_openai else
                    closed["content"][0]["tool_use_id"] == "branch-1")
            assert "branch" not in {t["name"] for t in self.tools}
            assert self.permission_mode == a.permission_mode
            return "update_work_state", {"acceptance_cases": [
                {"id": "value", "behavior": "fix value", "source_quote": "fix value"}], "hypotheses": []}
        if len(seen) == 2:
            self._emit_text("INTERMEDIATE MUST NOT RETURN")
            return "run_shell", {"command": "change-and-fail"}
        return "return", {"message": "Final summary references e0001, failure remains."}

    async def openai(self):
        name, args = response(self)
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 2}, "choices": [{
            "finish_reason": "tool_calls" if name else "stop",
            "message": {"role": "assistant", "content": "" if name else "Final summary references e0001, failure remains.",
                        "tool_calls": [{"id": f"c{len(seen)}", "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}] if name else []}}]}

    async def anthropic(self):
        name, args = response(self)
        return NS(usage=NS(input_tokens=10, output_tokens=2),
                  stop_reason="tool_use" if name else "end_turn",
                  content=[NS(type="tool_use", id=f"c{len(seen)}", name=name, input=args)] if name else
                          [NS(type="text", text="Final summary references e0001, failure remains.")])

    monkeypatch.setattr(Agent, "_call_openai_stream", openai)
    monkeypatch.setattr(Agent, "_call_anthropic_stream", anthropic)

    async def run():
        assert "only inside a branch" in await a._execute_tool_call("run_shell", {"command": "change-and-fail"})
        assert repo["state"] == "base"
        value = await a._execute_tool_call("branch", {"description": "inspect", "prompt": "inspect and report"})
        assert (a._openai_messages if a.use_openai else a._anthropic_messages) == original
        assert "INTERMEDIATE" not in value
        assert json.loads(value)["branch_status"] == "returned"
        if retain:
            history = json.loads(value)["branch_history_unverified"]
            assert "command failed after changing a file" in json.dumps(history)
            assert "branch-1" not in json.dumps(history)  # inherited context is not duplicated
        else:
            assert "branch_history_unverified" not in json.loads(value)
        assert "Final summary" in json.loads(value)["branch_summary_unverified"]
        assert a._operation_counts == {"model_calls": 3, "tool_calls": 4}
        assert a.total_input_tokens == 30 and a.total_output_tokens == 6
        assert "value" in a._repair_state.acceptance_cases
        evidence = a._repair_state.evidence[-1]
        assert evidence.exit_code == 1 and evidence.repo_after == "changed"
        assert not a._repair_state.completion_decision().complete
        assert await seen[-1]._execute_tool_call("branch", {}) == "Error: this tool is unavailable inside a branch."
        events = read_trace(a.trace.trace_path)
        assert events[-2]["event_type"] == "context.branch.end"
        assert not any(e["event_type"] == "hook.stop" for e in events)
        await a.shutdown()

    asyncio.run(run())


def test_full_history_control_survives_large_output_and_child_compaction(tmp_path, monkeypatch):
    a = parent(tmp_path, retain_branch_history=True, auto_compact=True, max_branches=8, branch_model_calls=10)
    text = "actual execution result " * 3000

    async def compact(self, *, trigger="manual"):
        self._openai_messages = [{"role": "user", "content": "compacted"}]
        return True

    async def run():
        child = _BranchAgent(a, api_base="https://example.invalid/v1", api_key="test", model="test",
                             custom_system_prompt="test", skills_enabled=False, is_sub_agent=True)
        child._openai_messages = [{"role": "user", "content": "inherited"},
                                  {"role": "assistant", "content": text}]
        child._history_start = 1
        monkeypatch.setattr(Agent, "_compact_conversation", compact)
        assert await child._compact_conversation()
        child._openai_messages.append({"role": "assistant", "content": "after compaction"})
        retained = child._branch_history + child._openai_messages[child._history_start:]
        payload = json.dumps({"branch_history_unverified": retained})
        assert a._persist_large_result("branch", payload) == payload
        assert text in payload and "after compaction" in payload and "inherited" not in payload
        assert child.auto_compact is True
        await child.shutdown()
        await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_paged_branch_history_is_complete_retrievable_and_fits_next_request(tmp_path, provider):
    a = parent(tmp_path, provider, retain_branch_history=True, page_branch_history=True,
               max_input_tokens_per_call=65536, max_total_tokens=None)
    history = [{"role": "assistant", "content": "execution evidence\n" * 18000}]
    payload = {"branch_status": "partial", "branch_summary_unverified": "Tests still fail.",
               "termination": {"kind": "incomplete_response"}, "branch_history_unverified": history}

    async def run():
        try:
            raw = json.dumps(payload)
            with pytest.raises(RuntimeError, match="Input budget exceeded"):
                a._preflight_model_input(messages=[{"role": "user", "content": raw}], tools=[])
            value = a._persist_large_result("branch", raw)
            result = json.loads(value)
            assert result["branch_summary_unverified"] == "Tests still fail."
            assert result["termination"] == payload["termination"]
            assert len(value) < 12000
            evidence_id = result["branch_history_unverified"]["evidence_id"]
            received, offset = "", 0
            while True:
                page = await a._execute_tool_call("read_observation", {
                    "evidence_id": evidence_id, "char_offset": offset, "limit": 200})
                header, body = page.split("\n", 1)
                received += body
                if "next_char_offset=None" in header:
                    break
                offset += len(body)
            assert json.loads(received.split(" | ", 1)[1]) == history
            assert a._read_observation({"evidence_id": evidence_id, "part": "input"}).startswith("Error:")
            if provider == "openai":
                a._openai_messages.append({"role": "tool", "tool_call_id": "branch-1", "content": value})
                messages = a._openai_messages
            else:
                a._anthropic_messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "branch-1", "content": value}]})
                messages = a._anthropic_messages
            a._preflight_model_input(messages=a._messages_with_runtime_state(messages), tools=a.tools)
            assert not a._repair_state.completion_decision().complete
            # Repeated rendering and another branch must not overwrite earlier observations.
            a._branch_count += 1
            a._persist_large_result("branch", json.dumps({**payload, "branch_history_unverified": ["other"] * 40000}))
            assert "execution evidence" in a._read_observation({"evidence_id": evidence_id})
        finally:
            await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_active_branch_assignment_survives_repeated_real_compaction(tmp_path, monkeypatch, provider):
    a = parent(tmp_path, provider, auto_compact=True)
    assignment = "Read only source.py and return its public API. Do not implement the overall repair."
    observed = []

    async def summary(self, *args):
        return "Observed source interfaces. No edits were made and no behavioral tests have been run."

    async def work(self, prompt):
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        messages.append({"role": "user", "content": prompt})
        for cycle in range(2):
            for i in range(5):
                call_id = f"read-{cycle}-{i}"
                if self.use_openai:
                    messages.extend([
                        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": "read_file", "arguments": '{"file_path":"source.py"}'}}]},
                        {"role": "tool", "tool_call_id": call_id, "content": "source contents\n" * 1500},
                    ])
                else:
                    messages.extend([
                        {"role": "assistant", "content": [{"type": "tool_use", "id": call_id,
                         "name": "read_file", "input": {"file_path": "source.py"}}]},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                         "content": "source contents\n" * 1500}]},
                    ])
            assert await self._compact_conversation(trigger="auto")
            messages = self._openai_messages if self.use_openai else self._anthropic_messages
            pinned = [m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)]
            observed.append((any(assignment in text for text in pinned), any("fix value" in text for text in pinned)))
        self.return_message = "Read-only investigation complete; no edits made."
        self.finish_reason = "tool_calls" if self.use_openai else "tool_use"
        return {"termination_reason": {"kind": "branch_return"}}

    monkeypatch.setattr(Agent, "_generate_folded_session_memory", summary)
    monkeypatch.setattr(Agent, "run_once", work)

    async def run():
        try:
            result = json.loads(await a._execute_tool_call("branch", {"description": "inspect", "prompt": assignment}))
            assert result["branch_status"] == "returned"
            assert observed == [(True, True), (True, True)]
        finally:
            await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("finish,termination,expected", [
    ("length", None, "partial"), (None, None, "partial"),
    ("stop", {"kind": "max_model_calls"}, "partial"),
    ("stop", None, "partial"),
])
def test_incomplete_branch_never_becomes_verified_success(tmp_path, monkeypatch, finish, termination, expected):
    a = parent(tmp_path)

    async def fake_run(self, prompt):
        self.finish_reason = finish
        self._openai_messages.append({"role": "assistant", "content": "unverified branch claim"})
        return {"text": "earlier narration", "termination_reason": termination}

    monkeypatch.setattr(_BranchAgent, "run_once", fake_run)

    async def run():
        result = json.loads(await a._execute_tool_call("branch", {"description": "subtask", "prompt": "inspect"}))
        assert result["branch_status"] == expected
        assert result["branch_summary_unverified"] == ""  # Plain text is not an explicit handoff.
        assert not a._repair_state.completion_decision().complete
        assert a.termination_reason is None  # local branch allowance does not spend the parent's remaining calls
        await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("used,tool_limit,expected_steps", [(5, None, 6), (10, None, 1), (5, 3, 2)])
@pytest.mark.parametrize("auto_compact", [False, True])
def test_last_branch_decision_returns_and_preserves_main_allowance(tmp_path, monkeypatch, provider, used, tool_limit, expected_steps, auto_compact):
    executed, requests = [], []

    async def execute(name, inp):
        executed.append(name)
        return "source contents"

    a = parent(tmp_path, provider, tool_executor=execute, max_tool_calls=tool_limit, auto_compact=auto_compact)
    a._operation_counts["model_calls"] = used

    def response(self):
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        view = self._messages_with_runtime_state(messages)
        assert [m["role"] for m in view] == [m["role"] for m in messages]
        assert "Model decisions remaining:" not in json.dumps(messages)
        names = {t["name"] for t in self.tools}
        self._reserve_call("model_calls", self.max_model_calls)
        if isinstance(self, _BranchAgent):
            requests.append((names, json.dumps(view, ensure_ascii=False), self.max_model_calls))
            name = "return" if names == {"return"} else "read_file"
            return name, {"message": "Investigation incomplete; continue repair in another branch."} if name == "return" else {"file_path": "source.py"}
        return None, {}

    async def openai(self):
        name, args = response(self)
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 2}, "choices": [{
            "finish_reason": "tool_calls" if name else "stop", "message": {
                "role": "assistant", "content": "", "tool_calls": [] if name is None else [{
                    "id": f"c{self._operation_counts['model_calls']}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]}}]}

    async def anthropic(self):
        name, args = response(self)
        return NS(usage=NS(input_tokens=10, output_tokens=2), stop_reason="tool_use" if name else "end_turn",
                  content=[NS(type="tool_use", id=f"c{self._operation_counts['model_calls']}", name=name, input=args)] if name else [])

    monkeypatch.setattr(Agent, "_call_openai_stream", openai)
    monkeypatch.setattr(Agent, "_call_anthropic_stream", anthropic)

    async def fold_at_handoff(self):
        # Reproduce a fold admitted after the penultimate decision, before return.
        if self._operation_counts["model_calls"] == self.max_model_calls - 1:
            self._reserve_call("model_calls", self.max_model_calls)

    # Cover the experiment's budget fold as well as the base loop; this replay is offline.
    monkeypatch.setattr(_FoldingLoop, "_check_and_compact", fold_at_handoff)

    async def run():
        result = json.loads(await a._execute_tool_call("branch", {"description": "inspect", "prompt": "read source"}))
        assert result["branch_status"] == "returned"
        assert requests[0][2] == 11  # Shared maximum is 12; MAIN keeps the final call.
        assert f"Model decisions remaining: {11 - used}" in requests[0][1]
        assert "model call 6/12" not in requests[0][1]
        assert requests[-1][0] == {"return"}
        assert len(requests) == expected_steps and len(executed) == expected_steps - 1
        assert a._operation_counts["model_calls"] == used + len(requests)
        before = a._operation_counts["model_calls"]
        await (a._call_openai_stream() if a.use_openai else a._call_anthropic_stream())
        assert a._operation_counts["model_calls"] == before + 1 <= 12
        assert not a._repair_state.completion_decision().complete
        await a.shutdown()

    asyncio.run(run())


def test_batch_rejection_and_cancellation_preserve_parent(tmp_path, monkeypatch):
    a = parent(tmp_path)
    original = copy.deepcopy(a._openai_messages)

    async def cancelled(self, prompt):
        self.total_input_tokens = 17
        raise asyncio.CancelledError()

    monkeypatch.setattr(_BranchAgent, "run_once", cancelled)

    async def run():
        assert "called alone" in await a._execute_tool_call("branch", {}, tool_batch_size=2)
        with pytest.raises(asyncio.CancelledError):
            await a._execute_tool_call("branch", {"description": "subtask", "prompt": "inspect"})
        assert a._openai_messages == original
        assert a.total_input_tokens == 17
        assert any(e["event_type"] == "context.branch.end" and e["status"] == "cancelled"
                   for e in read_trace(a.trace.trace_path))
        await a.shutdown()

    asyncio.run(run())


def test_real_stream_keeps_missing_finish_reason_unknown(tmp_path):
    a = parent(tmp_path)
    child = _BranchAgent(a, api_base="https://example.invalid/v1", api_key="test",
                         custom_system_prompt="test", custom_tools=[], is_sub_agent=True)

    async def stream():
        yield NS(choices=[NS(delta=NS(content="possibly truncated", tool_calls=None),
                             finish_reason=None)], usage=None)
        yield NS(choices=[], usage=NS(prompt_tokens=3, completion_tokens=1, total_tokens=4))

    async def create(**kwargs):
        return stream()

    async def run():
        await child._openai_client.close()
        child._openai_client = NS(chat=NS(completions=NS(create=create)))
        response = await child._call_openai_stream()
        assert response["choices"][0]["finish_reason"] == "stop"  # existing provider contract
        assert response["choices"][0]["raw_finish_reason"] is None
        assert child.finish_reason is None
        assert a.token_budget["charged"] == 4
        await child.shutdown()
        await a.shutdown()

    asyncio.run(run())


def test_return_requires_single_complete_call_and_does_not_certify_success(tmp_path):
    a = parent(tmp_path)
    child = _BranchAgent(a, api_base="https://example.invalid/v1", api_key="test",
                         custom_system_prompt="test", custom_tools=[], is_sub_agent=True)

    async def run():
        child.finish_reason = "tool_calls"
        assert "alone" in await child._execute_tool_call("return", {"message": "done"}, tool_batch_size=2)
        assert "nonempty" in await child._execute_tool_call("return", {"message": " "})
        child.finish_reason = "length"
        assert "complete" in await child._execute_tool_call("return", {"message": "done"})
        assert child.return_message is None and child.termination_reason is None
        for _ in range(2):
            assert "truncated" in await child._run_stop_hooks()
            assert child.return_message is None and child.termination_reason is None
        assert await child._run_stop_hooks() is None
        assert child.termination_reason == {"kind": "incomplete_response", "detail": "length"}
        child.termination_reason = None
        child.finish_reason = "stop"
        assert "return(message)" in await child._run_stop_hooks()
        child.finish_reason = "tool_calls"
        await child._execute_tool_call("return", {"message": "Unable to complete; test is still failing."})
        assert child.termination_reason == {"kind": "branch_return"}
        assert not a._repair_state.completion_decision().complete
        assert {t["name"] for t in a.tools}.isdisjoint({"run_shell", "write_file", "return"})
        assert "run_shell" in {t["name"] for t in a._execution_tools}
        await child.shutdown()
        await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
@pytest.mark.parametrize('current_verified', [False, True])
def test_budget_finalization_after_tool_response_runs_gate_and_preserves_exit(tmp_path, monkeypatch, provider, current_verified):
    a = parent(tmp_path, provider)
    async def probe():
        return 'base'
    a._repo_state_probe = probe
    a._repair_state = RepairState(TaskEnvelope('case', 'fix value', '/repo', requires_patch=False,
                                               requires_repair_witness=False), 'base')
    repair = a._repair_state
    repair.register_acceptance_case('value', 'fix value', 'fix value')
    if current_verified:
        observed = repair.record_evidence(tool='run_shell', status='ok', exit_code=0,
            input_summary='python -m pytest test_value.py -q', output_preview='one passed',
            repo_before='base', repo_after='base', test_pass_observed=True)
        repair.record_verification(observed.id, ['value'], scope='focused', oracle='existing_test', runner_recognized=True)
    checked = []

    async def recheck():
        checked.append(a._finalization_calls)

    monkeypatch.setattr(a, '_recheck_completion_failures', recheck)

    async def run():
        try:
            a.finish_reason = 'tool_calls' if provider == 'openai' else 'tool_use'
            a.termination_reason = {'kind': 'max_decisions'}
            await a._finalize_budget_exit()
            assert checked == [0]
            assert a._completion_evidence['status'] == ('complete' if current_verified else 'incomplete')
            assert a.termination_reason == {'kind': 'max_decisions'}
            assert a.finish_reason in {'tool_calls', 'tool_use'}
            assert a._operation_counts['model_calls'] == 0 and a._finalization_calls is None
            stop_events = [row for row in read_trace(a.trace.trace_path) if row['event_type'] == 'hook.stop']
            assert len(stop_events) == 1
            assert stop_events[0]['status'] == ('ok' if current_verified else 'blocked')
        finally:
            await a.shutdown()

    asyncio.run(run())


def test_budget_finalization_does_not_enter_unbounded_synchronous_probe(tmp_path):
    a = parent(tmp_path)
    called = []
    a._repo_state_probe = lambda: called.append('entered') or 'base'

    async def run():
        try:
            a.finish_reason = 'tool_calls'
            a.termination_reason = {'kind': 'max_decisions'}
            await a._finalize_budget_exit()
            assert called == []
            assert a._completion_evidence['status'] == 'unknown'
            assert a.termination_reason == {'kind': 'max_decisions'}
            assert a._operation_counts['model_calls'] == 0
        finally:
            await a.shutdown()

    asyncio.run(run())


def test_branch_request_exposure_uses_parent_skill_receipt_and_child_purpose(tmp_path, monkeypatch):
    a = parent(tmp_path, max_total_tokens=None, max_output_tokens=128)
    a._loaded_skill_context = '<skill-knowledge>Frozen procedure, not a user instruction.</skill-knowledge>'
    a._loaded_skill_receipt = {'name': 'value', 'sha256': 'frozen-body-digest', 'bytes': 87}
    child = _BranchAgent(a, api_base='https://example.invalid/v1', api_key='test', model='test',
                         custom_system_prompt='fixed', custom_tools=[], is_sub_agent=True,
                         skills_enabled=False, trace=a.trace, model_purpose='context_branch',
                         max_output_tokens=128, _operation_counts=a._operation_counts)
    child._openai_messages.append({'role': 'user', 'content': 'Inspect the current source.'})
    requests = []

    async def create(**kwargs):
        requests.append(copy.deepcopy(kwargs['messages']))
        async def chunks():
            yield NS(usage=NS(prompt_tokens=10, completion_tokens=2, total_tokens=12), choices=[
                NS(delta=NS(content='Observed source.', tool_calls=None), finish_reason='stop')])
        return chunks()

    monkeypatch.setattr(child._openai_client.chat.completions, 'create', create)

    async def run():
        try:
            await child._call_openai_stream()
            assert child._loaded_skill_receipt is None and child._loaded_skill_context == ''
            assert len(requests) == 1 and a._loaded_skill_context in json.dumps(requests[0])
            exposures = [row for row in read_trace(a.trace.trace_path) if row['event_type'] == 'skill.body_exposure']
            assert len(exposures) == 1 and exposures[0]['purpose'] == 'context_branch'
            assert exposures[0]['attributes']['model_call'] == 1
            payload = json.loads((a.trace.blob_dir / exposures[0]['payload_ref']['sha256']).read_text(encoding='utf-8'))
            assert payload == a._loaded_skill_receipt
        finally:
            await child.shutdown()
            await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("failure,limit", [
    ("budget", 1), ("missing_usage", 50000), ("transport", 50000),
])
def test_request_admission_blocks_before_http_and_after_unknown_usage(tmp_path, failure, limit):
    import httpx
    import openai

    a = parent(tmp_path, max_total_tokens=limit)
    calls = []

    async def respond(request):
        calls.append(request)
        if failure == "transport":
            return httpx.Response(503, json={"error": {"message": "temporary error"}})
        data = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "test",
                "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n")

    async def run():
        await a._openai_client.close()
        a._openai_client = openai.AsyncOpenAI(api_key="test", base_url="https://example.invalid/v1",
                                             max_retries=0,
                                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        expected_error = (_BudgetExceeded if failure == "budget" else
                          openai.InternalServerError if failure == "transport" else RuntimeError)
        with pytest.raises(expected_error):
            await a._call_openai_stream()
        if failure != "budget":
            assert a.token_budget["blocked"] == "unknown_usage_after_error"
            assert a.token_budget["charged"] > 0
        child = _BranchAgent(a, api_base="https://example.invalid/v1", api_key="test",
                             custom_system_prompt="test", custom_tools=[], is_sub_agent=True)
        # Shared admission also blocks a new child; no second HTTP request is possible.
        with pytest.raises(_BudgetExceeded):
            await child._call_openai_stream()
        assert len(calls) == (0 if failure == "budget" else 1)
        if failure == "budget":
            assert a.termination_reason["kind"] == "token_budget"
        else:
            assert child.termination_reason["kind"] == "token_budget"
        await child.shutdown()
        await a.shutdown()

    asyncio.run(run())


def test_partial_branch_can_be_followed_by_another_branch_in_the_real_loop(tmp_path, monkeypatch):
    async def execute(name, inp):
        return "RAW_CHILD_ONLY_SOURCE"

    a = parent(tmp_path, tool_executor=execute)
    a._repair_task = None  # Read-only protocol replay; repair gating is checked above.
    a._openai_messages = a._openai_messages[:1]
    main_results = []

    async def model(self):
        self._reserve_call("model_calls", self.max_model_calls)
        if isinstance(self, _BranchAgent):
            # First child ignores the return-only tool set. It must remain partial.
            name = "read_file" if self.parent._branch_count == 1 else "return"
            args = {"file_path": "source.py"} if name == "read_file" else {"message": "Source inspected."}
        else:
            if self._branch_count:
                result = json.loads(self._openai_messages[-1]["content"])
                main_results.append(result)
                assert "RAW_CHILD_ONLY_SOURCE" not in json.dumps(self._openai_messages)
            name = "branch" if self._branch_count < 2 else None
            args = {"description": "inspect", "prompt": "Inspect and return."}
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 2}, "choices": [{
            "raw_finish_reason": "tool_calls" if name else "stop", "finish_reason": "tool_calls" if name else "stop",
            "message": {"role": "assistant", "content": "" if name else "Inspection complete.", "tool_calls": [] if name is None else [{
                "id": f"call-{self._operation_counts['model_calls']}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}}]}

    monkeypatch.setattr(Agent, "_call_openai_stream", model)

    async def run():
        result = await a.run_once("Inspect source in a branch.")
        assert [r["branch_status"] for r in main_results] == ["partial", "returned"]
        assert main_results[0]["branch_summary_unverified"] == ""
        assert main_results[0]["remaining_branches"] == 1
        assert result["termination_reason"] is None and result["model_calls"] == 10
        assert a._operation_counts["tool_calls"] == 8  # Hidden read was rejected without executing it.
        await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("truncated", [True, False])
def test_child_output_limit_returns_control_but_missing_finish_still_stops(tmp_path, monkeypatch, provider, truncated):
    async def execute(name, inp):
        assert name == "read_file"
        return "OBSERVATION BEFORE INTERRUPTED RESPONSE"

    a = parent(tmp_path, provider, tool_executor=execute)
    a._repair_task = None
    a._openai_messages = a._openai_messages[:1]
    a._anthropic_messages = []

    def decision(self):
        self._reserve_call("model_calls", self.max_model_calls)
        if isinstance(self, _BranchAgent):
            self._test_decisions = getattr(self, "_test_decisions", 0) + 1
            if self.parent._branch_count == 1:
                if self._test_decisions == 1:
                    return "read_file", {"file_path": "source.py"}, "tool"
                return None, {}, "truncated" if truncated else "missing"
            assert any(e["event_type"] == "tool.call.end" and e["attributes"].get("tool_name") == "read_file"
                       for e in read_trace(a.trace.trace_path))
            return "return", {"message": "Read-only investigation completed after a partial branch."}, "tool"
        if self._branch_count < 2:
            return "branch", {"description": "inspect", "prompt": "Inspect source and return."}, "tool"
        return None, {}, "stop"

    async def openai(self):
        name, args, state = decision(self)
        finish = {"tool": "tool_calls", "truncated": "length", "missing": None, "stop": "stop"}[state]
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 2}, "choices": [{
            "raw_finish_reason": finish, "finish_reason": finish or "stop", "message": {
                "role": "assistant", "content": "" if name else "response text",
                "tool_calls": [] if not name else [{"id": f"c{self._operation_counts['model_calls']}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]}}]}

    async def anthropic(self):
        name, args, state = decision(self)
        return NS(usage=NS(input_tokens=10, output_tokens=2),
                  stop_reason={"tool": "tool_use", "truncated": "max_tokens", "missing": None, "stop": "end_turn"}[state],
                  content=[NS(type="tool_use", id=f"c{self._operation_counts['model_calls']}", name=name, input=args)] if name else
                          [NS(type="text", text="response text")])

    monkeypatch.setattr(Agent, "_call_openai_stream", openai)
    monkeypatch.setattr(Agent, "_call_anthropic_stream", anthropic)

    async def run():
        try:
            result = await a.run_once("Inspect source.")
            branches = [e for e in read_trace(a.trace.trace_path) if e["event_type"] == "context.branch.end"]
            assert branches[0]["status"] == "partial"
            assert a._branch_count == (2 if truncated else 1)
            if truncated:
                assert result["termination_reason"] is None
                assert branches[-1]["status"] == "returned"
            else:
                assert result["termination_reason"] == {"kind": "incomplete_response", "detail": "missing_finish_reason"}
        finally:
            await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("finish", ["length", None])
def test_incomplete_main_response_exhausts_only_bounded_length_recovery(tmp_path, monkeypatch, finish):
    a = parent(tmp_path)
    a._repair_task = None
    a._openai_messages = a._openai_messages[:1]

    async def model(self):
        self._reserve_call("model_calls", self.max_model_calls)
        return {"usage": {"prompt_tokens": 3, "completion_tokens": 1}, "choices": [{
            "raw_finish_reason": finish, "finish_reason": finish or "stop",
            "message": {"role": "assistant", "content": ""}}]}

    monkeypatch.setattr(Agent, "_call_openai_stream", model)

    async def run():
        result = await a.run_once("Inspect source.")
        assert result["model_calls"] == (3 if finish == "length" else 1)
        assert result["termination_reason"]["kind"] == "incomplete_response"
        assert not any(e["event_type"] == "hook.stop" and e["status"] == "ok" for e in read_trace(a.trace.trace_path))
        await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("remaining_calls,tool_limit", [(1, None), (12, 1)])
def test_branch_is_not_started_without_handoff_allowance(tmp_path, remaining_calls, tool_limit):
    a = parent(tmp_path, max_tool_calls=tool_limit)
    a._operation_counts["model_calls"] = 12 - remaining_calls

    async def run():
        result = await a._execute_tool_call("branch", {"description": "inspect", "prompt": "Inspect source."})
        assert result.startswith("Error: insufficient")
        assert a._branch_count == 0
        assert a._operation_counts["model_calls"] == 12 - remaining_calls
        assert not any(e["event_type"] == "context.branch.start" for e in read_trace(a.trace.trace_path))
        await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_first_child_request_compacts_inherited_history_before_http(tmp_path, monkeypatch, provider):
    import httpx
    import openai
    import anthropic
    if provider == "anthropic":
        import httpx2 as httpx

    a = parent(tmp_path, provider, auto_compact=True, max_total_tokens=None,
               max_input_tokens_per_call=32768)
    child = _BranchAgent(a, api_base="https://example.invalid/v1" if provider == "openai" else None, api_key="test",
                         anthropic_base_url="https://example.invalid" if provider == "anthropic" else None,
                         custom_system_prompt="solver", custom_tools=[], model="test",
                         is_sub_agent=True, max_input_tokens_per_call=32768)
    child._branch_assignment = "Read only; investigate the current failure."
    messages = [{"role": "user", "content": "Repair the public issue."},
                {"role": "assistant", "content": "Prior evidence.",
                 **({"reasoning_content": "r" * 120000} if provider == "openai" else
                    {"content": [{"type": "thinking", "thinking": "r" * 120000, "signature": "test"},
                                 {"type": "text", "text": "Prior evidence."}]})},
                {"role": "user", "content": child._branch_assignment}]
    if provider == "openai":
        child._openai_messages = [{"role": "system", "content": "solver"}, *messages]
    else:
        child._anthropic_messages = messages
    summarized = []

    async def summary(history, previous_summary=None, recent_context=None):
        assert "Read only" in child._canonical_issue(messages)
        summarized.append(history)
        return "The public issue remains active. This branch must only investigate and report the failure."

    monkeypatch.setattr(child, "_generate_folded_session_memory", summary)
    requests = []

    async def respond(request):
        payload = json.loads(request.content)
        assert summarized  # No HTTP request is issued for the oversized inherited conversation.
        assert child._estimate_model_input_tokens(messages=payload["messages"], tools=payload.get("tools", []),
                    system=payload.get("system")) <= 32768
        assert "Read only" in json.dumps(payload["messages"])
        requests.append(request)
        if provider == "openai":
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "test",
                     "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
            data = "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
        else:
            events = [
                {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
                    "content": [], "model": "test", "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                 "usage": {"output_tokens": 2}},
                {"type": "message_stop"},
            ]
            data = ''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n' for e in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)

    async def run():
        try:
            if provider == "openai":
                await child._openai_client.close()
                child._openai_client = openai.AsyncOpenAI(api_key="test", base_url="https://example.invalid/v1",
                    max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
                await child._call_openai_stream()
            else:
                await child._anthropic_client.close()
                child._anthropic_client = anthropic.AsyncAnthropic(api_key="test", base_url="https://example.invalid",
                    max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
                await child._call_anthropic_stream()
            assert len(requests) == 1
        finally:
            await child.shutdown()
            await a.shutdown()

    asyncio.run(run())
