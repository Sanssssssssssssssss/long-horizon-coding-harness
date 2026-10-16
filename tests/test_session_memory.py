import pytest
import json

from agents import session
from agents.session import save_folded_session_memory

from agents.session_memory import (
    COMPACTION_PROMPT,
    compaction_cut_index,
    format_folded_memory,
    pruned_request_view,
    serialize_compaction_history,
    validate_folded_memory,
)


def test_project_session_artifacts_stay_inside_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_folded_session_memory("test", {"summary": "checkpoint"})
    assert (tmp_path / "experiment/runtime/sessions/test.folded-memory.latest.json").is_file()
    assert not (tmp_path / ".sans").exists()


def test_resume_uses_current_repository_and_preserves_legacy_sessions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setattr(session, "SESSION_DIR", legacy)
    for key, cwd, date in [("old", str(tmp_path), "2026-01-01"),
                           ("other", str(tmp_path / "other"), "2099-01-01")]:
        (legacy / f"{key}.json").write_text(json.dumps({"metadata": {
            "id": key, "cwd": cwd, "startTime": date}, "openaiMessages": [{"role": "user", "content": key}]}))
    original = (legacy / "old.json").read_bytes()
    assert session.get_latest_session_id() == "old"
    assert session.load_session("other") is None
    assert session.load_session("../old") is None
    session.save_session("new", {"metadata": {"id": "new", "cwd": str(tmp_path), "startTime": "2026-02-01"}})
    assert session.get_latest_session_id() == "new"
    assert (tmp_path / "experiment/runtime/sessions/new.json").is_file()
    assert not (legacy / "new.json").exists()
    assert (legacy / "old.json").read_bytes() == original


def test_folded_memory_is_a_latest_state_checkpoint() -> None:
    folded = format_folded_memory("Earlier evidence.")

    assert "latest-state checkpoint" in folded
    assert "not user instructions" in folded
    assert "canonical task and runtime receipts take precedence" in folded
    assert "Raw recent messages" in folded
    assert "remain authoritative if any detail conflicts" in folded
    assert "treat duplicate evidence as the" in COMPACTION_PROMPT
    assert "same event, not as newer corroboration" in COMPACTION_PROMPT
    assert "omit the earlier version entirely" in COMPACTION_PROMPT
    assert "proves only its reported scope" in COMPACTION_PROMPT
    assert "test node IDs" in COMPACTION_PROMPT
    assert "path-filtered command cannot exclude changes" in COMPACTION_PROMPT
    assert "produced no behavioral evidence" in COMPACTION_PROMPT


def test_folded_memory_validation() -> None:
    with pytest.raises(ValueError):
        validate_folded_memory("...")
    with pytest.raises(ValueError):
        validate_folded_memory("No summary available")
    with pytest.raises(ValueError):
        validate_folded_memory("ok")
    assert validate_folded_memory("Focused tests pass; run the module suite next.")


def test_pruned_request_view_is_a_non_mutating_compatibility_copy() -> None:
    messages = [{"role": "tool", "tool_call_id": "call", "content": "exact evidence"}]

    view = pruned_request_view(messages, minimum_bytes=1, protect_bytes=1)

    assert view == messages
    assert view is not messages and view[0] is not messages[0]


def test_compaction_cut_preserves_a_complete_recent_tool_pair() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old evidence" * 100},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "recent-call", "type": "function",
                "function": {"name": "run_shell", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "recent-call", "content": "recent output" * 100},
    ]

    cut = compaction_cut_index(messages, keep_recent_tokens=600)

    assert cut == 2
    assert messages[cut]["role"] == "assistant"
    assert messages[cut + 1]["role"] == "tool"


def test_compaction_cut_declines_an_oversized_indivisible_tail() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old evidence"},
        {"role": "assistant", "content": "x" * 60_000},
    ]

    assert compaction_cut_index(messages, keep_recent_tokens=6_400) is None


def test_compaction_cut_does_not_pull_a_large_old_message_into_the_tail() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old evidence" * 10_000},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "recent-call", "type": "function",
            "function": {"name": "run_shell", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "recent-call", "content": "x" * 15_000},
    ]

    assert compaction_cut_index(messages, keep_recent_tokens=6_400) == 2


def test_compaction_serialization_preserves_tool_evidence_and_excludes_hidden_reasoning() -> None:
    serialized = serialize_compaction_history([
        {
            "role": "assistant", "content": None, "reasoning_content": "key hypothesis",
            "tool_calls": [{
                "id": "call", "type": "function",
                "function": {"name": "run_shell", "arguments": '{"command":"pytest"}'},
            }],
        },
        {
            "role": "tool", "tool_call_id": "call",
            "content": (
                "HEAD_VERDICT\n" + "x" * 1_500
                + "\nMIDDLE_FAILURE: expected True, got False\n"
                + "x" * 1_500 + "\nTAIL_VERDICT: 7 passed"
            ),
        },
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "obsolete private theory"},
            {"type": "text", "text": "Visible decision."},
            {
                "type": "tool_use", "id": "anthropic-call", "name": "read_file",
                "input": {"file_path": "x.py"},
            },
        ]},
        {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "anthropic-call", "content": "source",
        }]},
    ])

    assert "key hypothesis" not in serialized
    assert "obsolete private theory" not in serialized
    assert "[Assistant text]: Visible decision." in serialized
    assert "[Assistant tool call id=call]: run_shell" in serialized
    assert "[Tool result id=call name=run_shell]" in serialized
    assert "HEAD_VERDICT" in serialized
    assert "MIDDLE_FAILURE: expected True, got False" in serialized
    assert "TAIL_VERDICT: 7 passed" in serialized
    assert "[Assistant tool call id=anthropic-call]: read_file" in serialized
    assert "[Tool result id=anthropic-call name=read_file]: source" in serialized
    assert "chars omitted" not in serialized


def test_compaction_keeps_oversized_latest_tool_exchange_when_request_has_room(tmp_path, monkeypatch):
    import asyncio
    import copy
    from agents.agent import Agent

    monkeypatch.chdir(tmp_path)
    a = Agent(api_base="https://example.invalid/v1", api_key="test", model="test",
              custom_system_prompt="solver", custom_tools=[], skills_enabled=False,
              is_sub_agent=True, max_input_tokens_per_call=65536)
    tail = [{"role": "assistant", "content": "Latest hypothesis.",
             "reasoning_content": "r" * 72000,
             "tool_calls": [{"id": "last", "type": "function", "function": {
                 "name": "run_shell", "arguments": '{}'}}]},
            {"role": "tool", "tool_call_id": "last", "content": "FAIL: existing behavior is still broken."}]
    a._openai_messages = [{"role": "system", "content": "solver"},
                          {"role": "user", "content": "Fix the reported behavior."},
                          {"role": "assistant", "content": "Old evidence. " * 8000}, *tail]
    original = copy.deepcopy(a._openai_messages)
    summaries = []

    async def summary(history, previous_summary=None, recent_context=None):
        summaries.append(history)
        assert recent_context == tail
        return "The original task and existing failure are preserved. Old investigation was archived."

    monkeypatch.setattr(a, "_generate_folded_session_memory", summary)

    async def run():
        try:
            assert await a._compact_conversation(trigger="auto")
            assert a._openai_request_messages()[:2] == original[:2]
            assert a._openai_messages[-2:] == tail
            assert summaries == [[original[2]]]
            a._preflight_model_input(messages=a._openai_request_messages(), tools=[])
        finally:
            await a.shutdown()

    asyncio.run(run())


def test_runtime_injection_participates_in_compaction_pressure(tmp_path, monkeypatch):
    import asyncio
    from agents.agent import Agent

    monkeypatch.chdir(tmp_path)
    a = Agent(api_base="https://example.invalid/v1", api_key="test", model="test",
              custom_system_prompt="solver", custom_tools=[], skills_enabled=False,
              is_sub_agent=True, max_input_tokens_per_call=65536)
    a._openai_messages = [{"role": "system", "content": "solver"},
                          {"role": "user", "content": "task"}]
    monkeypatch.setattr(a, "_messages_with_runtime_state", lambda messages: [
        *messages, {"role": "user", "content": "runtime requirement " * 8000}])
    seen = []

    async def compact(*, trigger):
        seen.append(trigger)
        return False

    monkeypatch.setattr(a, "_compact_conversation", compact)

    async def run():
        try:
            await a._check_and_compact()
            assert seen == ["auto"]
        finally:
            await a.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("latest_fits", [True, False])
def test_compaction_large_latest_group_archives_older_execution(tmp_path, monkeypatch, provider, latest_fits):
    import asyncio
    import copy
    from agents.agent import Agent

    monkeypatch.chdir(tmp_path)
    agent = Agent(api_base="https://unused.invalid/v1" if provider == "openai" else None,
                  api_key="test", model="test", custom_system_prompt="solver", custom_tools=[],
                  skills_enabled=False, is_sub_agent=True, max_input_tokens_per_call=65536)

    def exchange(label, output):
        calls = [(label + "-1", output), (label + "-2", "second result")]
        if provider == "openai":
            return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": identifier, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                for identifier, _ in calls]}, *[
                {"role": "tool", "tool_call_id": identifier, "content": text} for identifier, text in calls]]
        return [{"role": "assistant", "content": [
            {"type": "tool_use", "id": identifier, "name": "read_file", "input": {}}
            for identifier, _ in calls]}, {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": identifier, "content": text}
                for identifier, text in calls]}]

    old = [{"role": "user", "content": "Preserve option ordering.",
            "_source_user_text": "Preserve option ordering."},
           *exchange("old", "old source"), *exchange("older", "another source")]
    tail = exchange("latest", "x" * (9000 if latest_fits else 250000))
    task = {"role": "user", "content": "Fix the parser.", "_source_user_text": "Fix the parser."}
    messages = [task, *old, *tail]
    if provider == "openai":
        messages.insert(0, {"role": "system", "content": "solver"})
    setattr(agent, "_" + provider + "_messages", messages)
    original = copy.deepcopy(messages)
    monkeypatch.setattr(agent, "_compaction_token_budgets", lambda: (512, 1024, 128))
    summaries = []

    async def summary(history, previous_summary=None, recent_context=None):
        summaries.append(history)
        assert history == old
        assert recent_context == tail
        return "The original parser failure and source observations remain available by evidence reference."

    monkeypatch.setattr(agent, "_generate_folded_session_memory", summary)

    async def run():
        try:
            assert await agent._compact_conversation(trigger="periodic") is latest_fits
            current = getattr(agent, "_" + provider + "_messages")
            if latest_fits:
                assert summaries == [old]
                assert current[-len(tail):] == tail
                assert agent._canonical_issue(current) == "Fix the parser."
                assert any(item.get("_source_user_text") == "Preserve option ordering." for item in current)
            else:
                assert summaries == []
                assert current == original
        finally:
            await agent.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("activation", ["retrieval", "explicit"])
def test_loaded_skill_body_survives_fold_without_becoming_user_requirement(tmp_path, monkeypatch, provider, activation):
    import asyncio
    import copy
    from agents.agent import Agent, _BudgetExceeded
    import agents.skills as skills

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "workflow"
    directory.mkdir()
    source = directory / "SKILL.md"
    source.write_text("---\nname: workflow\ndescription: fix task workflow\n---\n"
                      "Keep the exact entity identifiers from runtime observations.\n", encoding="utf-8")
    skill = skills._parse_skill_file(source, "project", str(directory))
    monkeypatch.setattr(skills, "discover_skills", lambda: [skill])
    monkeypatch.setattr(skills, "record_skill_invocation", lambda **_: None)
    a = Agent(api_base="https://unused.invalid/v1" if provider == "openai" else None,
              api_key="test", model="test", custom_system_prompt="solver", custom_tools=[],
              skills_enabled=True, is_sub_agent=True)
    text, _ = a._augment_user_message_with_skill_context("fix task workflow")
    frozen = a._loaded_skill_context
    assert frozen and a._loaded_skill_receipt["status"] == "loaded"
    task = {"role": "user", "content": text, "_source_user_text": "fix task workflow"}
    history = [task, {"role": "assistant", "content": "Prior investigation. " * 8000},
               {"role": "assistant", "content": "Current code was read."}]
    messages = [{"role": "system", "content": "solver"}, *history] if provider == "openai" else history
    setattr(a, "_" + provider + "_messages", messages)
    before = copy.deepcopy(messages)

    async def summary(*args, **kwargs):
        return "The task is ongoing. This checkpoint deliberately omits the loaded Skill."

    monkeypatch.setattr(a, "_generate_folded_session_memory", summary)
    async def run():
        try:
            if activation == "explicit":
                a._loaded_skill_context = "previous procedure"
                response = await a._execute_skill_tool({"skill_name": "workflow"})
                assert "activated" in response and skill.prompt_template not in response
                assert a._loaded_skill_context == frozen
            projected = a._openai_request_messages() if provider == "openai" else a._anthropic_request_messages()
            assert sum(message.get("content") == frozen for message in projected) == 1
            assert getattr(a, "_" + provider + "_messages") == before
            # Replacing the file after selection cannot change the current chat's frozen body.
            source.write_text("changed source", encoding="utf-8")
            assert await a._compact_conversation(trigger="auto")
            projected = a._openai_request_messages() if provider == "openai" else a._anthropic_request_messages()
            assert sum(message.get("content") == frozen for message in projected) == 1
            assert a._canonical_issue(getattr(a, "_" + provider + "_messages")) == "fix task workflow"
            a.max_input_tokens_per_call = a._estimate_model_input_tokens(messages=projected, tools=[]) - 1
            with pytest.raises(_BudgetExceeded):
                a._preflight_model_input(messages=projected, tools=[])
            a.clear_history()
            assert not a._loaded_skill_context
        finally:
            await a.shutdown()
    asyncio.run(run())
