from __future__ import annotations

import asyncio
import itertools
import json
import subprocess
from types import SimpleNamespace

import pytest

import agents.agent as agent_module
import agents.prompt as prompt_module
import agents.skills as skills_module
from agents.agent import Agent
from agents.repair_state import TaskEnvelope
from agents.tool_result import PytestReport, ShellResult
from agents.session_memory import (
    COMPACTION_PROMPT,
    COMPACTION_RETRY_PROMPT,
    COMPACTION_SUMMARY_MAX_TOKENS,
    SUMMARIZATION_SYSTEM_PROMPT,
)
from agents.telemetry import TraceRecorder
from agents.tools import tool_definitions


async def _empty_stream():
    yield SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=1,
            total_tokens=4,
            prompt_tokens_details=None,
            completion_tokens_details=None,
            reasoning_tokens=2,
            cost=0.0,
        ),
        choices=[],
    )


def test_transient_provider_errors_are_retryable() -> None:
    assert agent_module._is_retryable(RuntimeError("Request timed out."))
    assert agent_module._is_retryable(TimeoutError())
    assert agent_module._is_retryable(RuntimeError("Upstream stream ended before terminal chunk"))
    assert agent_module._is_retryable(RuntimeError("decryption failed or bad record mac"))
    assert agent_module._is_context_window_error(RuntimeError(
        "Input budget exceeded before provider call: tokenizer-free estimate=40000"
    ))


@pytest.mark.parametrize("disconnect", [False, True])
def test_incomplete_provider_stream_retries_without_spending_another_decision(
    monkeypatch: pytest.MonkeyPatch, disconnect: bool,
) -> None:
    agent = Agent(
        api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
        custom_tools=[], is_sub_agent=True, max_model_calls=1,
    )
    agent._openai_messages.append({"role": "user", "content": "retry empty stream"})
    calls = 0

    async def no_usage_stream():
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(
                content=None, reasoning="unfinished", reasoning_content=None, tool_calls=None,
            ),
            finish_reason="stop",
        )])
        if disconnect:
            try:
                from httpx2 import RemoteProtocolError
            except ImportError:
                from httpx import RemoteProtocolError
            raise RemoteProtocolError("incomplete chunked read")

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        return no_usage_stream() if calls == 1 else _empty_stream()

    async def no_sleep(_delay: float) -> None:
        return None

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(agent_module.asyncio, "sleep", no_sleep)

    response = asyncio.run(agent._call_openai_stream())

    assert calls == 2
    assert response["usage"]["total_tokens"] == 4
    assert agent._operation_counts["model_calls"] == 1


def test_stop_hook_followup_omits_null_tool_calls_from_openai_history() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        repo_state_probe=lambda: "base",
    )
    responses = iter((
        {"choices": [{"message": {
            "role": "assistant", "content": "premature", "tool_calls": None,
        }}]},
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    ))
    stop_reasons = iter(("verification missing", None))
    histories: list[list[dict]] = []

    async def fake_stream() -> dict:
        histories.append(list(agent._openai_messages))
        return next(responses)

    async def fake_stop_hook() -> str | None:
        return next(stop_reasons)

    agent._call_openai_stream = fake_stream
    agent._run_stop_hooks = fake_stop_hook
    asyncio.run(agent.run_once("fix it"))

    assert "tool_calls" not in histories[1][-2]
    assert histories[1][-1]["content"].startswith("<stop_hook>")


def test_openai_output_limit_and_extra_body_reach_provider() -> None:
    agent = Agent(
        model="DeepSeek-V4-Flash",
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        custom_tools=[],
        is_sub_agent=True,
        max_output_tokens=2048,
        openai_extra_body={"thinking": {"type": "disabled"}},
        openai_stream_include_usage=False,
    )
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _empty_stream()

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response = asyncio.run(agent._call_openai_stream())

    assert calls[0]["max_tokens"] == 2048
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "stream_options" not in calls[0]
    assert calls[0]["tools"] == []
    assert response["usage"]["completion_tokens_details"] == {"reasoning_tokens": 2}
    assert response["usage"]["cost"] == 0.0


def test_openai_runtime_counters_are_an_ephemeral_request_suffix() -> None:
    agent = Agent(
        model="DeepSeek-V4-Flash",
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        max_model_calls=40,
        max_tool_calls=40,
    )
    agent._openai_messages.append({"role": "user", "content": "fix it"})
    agent._operation_counts.update(model_calls=11, tool_calls=19)
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _empty_stream()

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    asyncio.run(agent._call_openai_stream())

    assert calls[0]["messages"][-1]["content"] == (
        "fix it\n\n<runtime_state>model call 12/40; tool calls completed 19/40</runtime_state>"
    )
    assert agent._openai_messages[-1]["content"] == "fix it"
    assert agent._openai_messages[0]["content"] == "fixed"


def test_model_call_timeout_is_enforced() -> None:
    agent = Agent(
        model="DeepSeek-V4-Flash",
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        model_call_timeout_seconds=0.01,
    )

    async def create(**_kwargs):
        await asyncio.sleep(3600)

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(agent._call_openai_stream())

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(agent._run_side_query_call(
            provider="openai",
            purpose="test",
            system="system",
            user_message="user",
            invoke=lambda: create(),
            extract_text=lambda _response: "",
        ))


def test_deepseek_thinking_is_enabled_and_preserved_for_tool_followup() -> None:
    agent = Agent(
        model="DeepSeek-V4-Flash",
        api_base="https://developer.amd.com.cn/radeon/api/v1",
        api_key="test",
        thinking=True,
        custom_system_prompt="test",
        custom_tools=[{
            "name": "noop",
            "description": "No operation",
            "input_schema": {"type": "object", "properties": {}},
        }],
        is_sub_agent=True,
    )
    calls: list[dict] = []

    async def first_stream():
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id="call-1",
                        function=SimpleNamespace(name="noop", arguments="{}"),
                    )],
                ),
                finish_reason="tool_calls",
            )],
        )

    async def second_stream():
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content="done",
                    reasoning_content="The tool completed.",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )],
        )

    streams = iter((first_stream(), second_stream()))

    async def create(**kwargs):
        calls.append(kwargs)
        return next(streams)

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    first = asyncio.run(agent._call_openai_stream())
    assistant = first["choices"][0]["message"]
    assert assistant["reasoning_content"] == "(reasoning omitted)"
    agent._openai_messages.extend((
        assistant,
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ))
    second = asyncio.run(agent._call_openai_stream())

    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"thinking": True}}
    assert calls[0]["parallel_tool_calls"] is True
    assert calls[1]["messages"][-2]["reasoning_content"] == "(reasoning omitted)"
    assert second["choices"][0]["message"]["reasoning_content"] == "The tool completed."


def test_commandcode_uses_gateway_defaults_and_replays_reasoning_alias() -> None:
    agent = Agent(
        model="deepseek/deepseek-v4-flash",
        api_base="https://api.commandcode.ai/provider/v1",
        api_key="test",
        thinking=True,
        custom_system_prompt="test",
        custom_tools=[],
        is_sub_agent=True,
    )

    async def stream():
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content="done",
                    reasoning="A continuous plan.",
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )],
        )

    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return stream()

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    response = asyncio.run(agent._call_openai_stream())

    assert "extra_body" not in calls[0]
    assert calls[0]["parallel_tool_calls"] is True
    assert response["choices"][0]["message"]["reasoning_content"] == "A continuous plan."


def test_official_deepseek_uses_standard_thinking_parameter() -> None:
    agent = Agent(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        api_key="test",
        thinking=True,
        custom_system_prompt="test",
        custom_tools=[],
        is_sub_agent=True,
    )

    assert agent._openai_request_extra_body() == {"thinking": {"type": "enabled"}}


def test_input_preflight_blocks_oversized_request_before_network() -> None:
    agent = Agent(
        model="unknown-provider-model",
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        custom_tools=[],
        is_sub_agent=True,
        max_input_tokens_per_call=8,
    )
    calls = 0

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        return _empty_stream()

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(RuntimeError, match="tokenizer-free estimate=.*max_input_tokens_per_call=8"):
        asyncio.run(agent._call_openai_stream())
    assert calls == 0


def test_explicit_input_budget_replaces_unknown_model_window_for_price_control() -> None:
    unbudgeted = Agent(
        model="unknown-provider-model",
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
    )
    budgeted = Agent(
        model="unknown-provider-model",
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
        max_input_tokens_per_call=30_000,
    )

    assert unbudgeted.effective_window == 200_000
    assert budgeted.effective_window == 30_000


def test_openai_compaction_fails_closed_without_losing_history() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    agent._openai_messages = [
        {"role": "system", "content": "canonical context"},
        {"role": "user", "content": "original task"},
        *(
            {"role": "assistant" if index % 2 == 0 else "user", "content": f"message {index}"}
            for index in range(12)
        ),
    ]
    original = json.loads(json.dumps(agent._openai_messages))

    async def no_memory(
        _history: list[dict], _previous_summary: str | None = None,
        _recent_context: list[dict] | None = None,
    ):
        return None

    agent._generate_folded_session_memory = no_memory

    assert asyncio.run(agent._compact_openai(trigger="auto")) is False
    assert agent._openai_messages == original
    assert agent._fold_count == 0


def test_compaction_uses_a_dedicated_bounded_summary_request() -> None:
    canonical = "Print subtraction; no exact term order is specified."
    agent = Agent(
        model="deepseek/deepseek-v4-flash",
        api_base="https://api.commandcode.ai/provider/v1",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
        repair_task=TaskEnvelope("fold", canonical, "/repo"),
    )
    history = [
        {"role": "system", "content": "canonical context"},
        {"role": "assistant", "content": "current evidence"},
    ]
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Tests pass; inspect the final diff next."),
                finish_reason="stop",
            )],
            usage=None,
        )

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    recent = [{"role": "assistant", "content": "the edit is now applied"}]
    summary = asyncio.run(agent._run_compaction_query(
        history, COMPACTION_PROMPT, recent_context=recent,
    ))

    assert summary == "Tests pass; inspect the final diff next."
    assert calls[0]["messages"][0] == {
        "role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT,
    }
    assert "<new-archived-evidence>" in calls[0]["messages"][1]["content"]
    assert f"<canonical-task>\n{canonical}\n</canonical-task>" in calls[0]["messages"][1]["content"]
    assert "[Assistant]: current evidence" in calls[0]["messages"][1]["content"]
    assert "<retained-recent-context>" in calls[0]["messages"][1]["content"]
    assert "[Assistant]: the edit is now applied" in calls[0]["messages"][1]["content"]
    assert calls[0]["messages"][1]["content"].endswith(COMPACTION_PROMPT)
    assert "tools" not in calls[0]
    assert calls[0]["max_tokens"] == COMPACTION_SUMMARY_MAX_TOKENS


def test_compaction_side_query_counts_tokens_and_cost() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )

    async def create(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="latest checkpoint"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=25,
                total_tokens=125,
                prompt_tokens_details=None,
                completion_tokens_details=None,
                cost=0.01,
            ),
        )

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    asyncio.run(agent._run_compaction_query(
        [{"role": "assistant", "content": "evidence"}], COMPACTION_PROMPT,
    ))

    assert agent.total_input_tokens == 100
    assert agent.total_output_tokens == 25
    assert agent.total_provider_reported_cost == pytest.approx(0.01)
    agent.max_cost_usd = 0.005
    assert agent._check_budget()["kind"] == "max_cost"


def test_cost_exhausted_by_side_query_stops_before_the_next_model_call() -> None:
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="canonical context", custom_tools=[], is_sub_agent=True,
        max_cost_usd=0.005,
    )
    agent.total_provider_reported_cost = 0.01
    called = False

    async def stream() -> dict:
        nonlocal called
        called = True
        return {}

    agent._call_openai_stream = stream
    asyncio.run(agent._chat_openai("continue"))

    assert called is False
    assert agent.termination_reason == {"kind": "max_cost"}


def test_manual_compaction_respects_an_exhausted_cost_budget() -> None:
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="canonical context", custom_tools=[], is_sub_agent=True,
        max_cost_usd=0.005,
    )
    agent.total_provider_reported_cost = 0.01
    called = False

    async def create(**_kwargs):
        nonlocal called
        called = True

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(agent_module._BudgetExceeded, match="max_cost"):
        asyncio.run(agent._run_compaction_query(
            [{"role": "assistant", "content": "evidence"}], COMPACTION_PROMPT,
        ))
    assert called is False


def test_compaction_transport_retry_counts_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="canonical context", custom_tools=[], is_sub_agent=True,
        max_model_calls=2,
    )
    calls = 0

    class RetryableError(RuntimeError):
        status_code = 503

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableError("retry")
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="checkpoint"), finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=2, total_tokens=12,
                prompt_tokens_details=None, completion_tokens_details=None, cost=0.0,
            ),
        )

    async def no_sleep(_delay: float) -> None:
        return None

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(agent_module.asyncio, "sleep", no_sleep)

    asyncio.run(agent._run_compaction_query(
        [{"role": "assistant", "content": "evidence"}], COMPACTION_PROMPT,
    ))

    assert calls == 2
    assert agent._operation_counts["model_calls"] == 2
    assert (agent.total_input_tokens, agent.total_output_tokens) == (10, 2)


def test_large_tool_preview_preserves_head_and_tail(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module.Path, "home", classmethod(lambda _cls: tmp_path))
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="canonical context", custom_tools=[], is_sub_agent=True,
    )

    raw = "HEAD\n" + "x\n" * 22_500 + "FINAL_VERDICT: 7 failed"
    receipt = agent._persist_large_result("run_shell", raw)

    assert "head+tail, capped at 8000 chars" in receipt
    assert "HEAD" in receipt
    assert "FINAL_VERDICT: 7 failed" in receipt
    assert len(receipt) < 9_000
    artifact = next((tmp_path / ".sans" / "tool-results").iterdir())
    assert artifact.read_text(encoding="utf-8") == raw


def test_parallel_large_tool_results_use_distinct_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module.Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr(agent_module.time, "time", lambda: 1.0)
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="canonical context", custom_tools=[], is_sub_agent=True,
    )

    first = "A" * (31 * 1024)
    second = "B" * (31 * 1024)
    agent._persist_large_result("run_shell", first)
    agent._persist_large_result("run_shell", second)

    artifacts = list((tmp_path / ".sans" / "tool-results").iterdir())
    assert len(artifacts) == 2
    assert {path.read_text(encoding="utf-8") for path in artifacts} == {first, second}


def test_compaction_budget_fits_a_32k_context() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        is_sub_agent=True,
        max_input_tokens_per_call=32_000,
    )

    summary_tokens, recent_tokens, overhead = agent._compaction_token_budgets()

    summary_footprint = (summary_tokens * 4 + 2) // 3
    assert summary_footprint + recent_tokens + overhead <= (
        agent.context_window - agent_module.COMPACTION_RESERVE_TOKENS
    )
    assert recent_tokens < agent_module.COMPACTION_KEEP_RECENT_TOKENS


@pytest.mark.parametrize("window", [30_000, 32_000])
@pytest.mark.parametrize("unit", ["x", "证"])
def test_small_context_compaction_reduces_the_next_request_below_threshold(
    window: int, unit: str,
) -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        is_sub_agent=True,
        max_input_tokens_per_call=window,
        max_model_calls=60,
        max_tool_calls=60,
    )
    agent._openai_messages.append({"role": "user", "content": "Repair the requested behavior."})
    agent._openai_messages.extend(
        {"role": "assistant" if index % 2 == 0 else "user",
         "content": f"evidence-{index} " + unit * 500, "_source_user_text": None}
        for index in range(160)
    )

    summary_tokens, _, _ = agent._compaction_token_budgets()

    async def folded(*_args, **_kwargs):
        return "y" * (summary_tokens * 4)

    agent._generate_folded_session_memory = folded
    assert asyncio.run(agent._compact_openai(trigger="auto")) is True

    pressure = agent._conversation_input_estimate(agent._openai_request_messages())
    assert pressure <= agent.context_window - agent_module.COMPACTION_RESERVE_TOKENS


def test_openai_mixed_compaction_preserves_sibling_tool_result() -> None:
    executed: list[str] = []

    async def executor(name: str, _inp: dict) -> str:
        executed.append(name)
        return "READ_EVIDENCE"

    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
        tool_executor=executor,
        repo_state_probe=lambda: "unchanged",
    )
    responses = iter([
        {
            "choices": [{"message": {
                "role": "assistant", "content": None, "tool_calls": [
                    {"id": "read", "type": "function", "function": {
                        "name": "read_file", "arguments": '{"file_path":"x.py"}',
                    }},
                    {"id": "fold", "type": "function", "function": {
                        "name": "compact_context", "arguments": "{}",
                    }},
                ],
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
        },
        {
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
        },
    ])

    async def stream() -> dict:
        return next(responses)

    agent._call_openai_stream = stream
    asyncio.run(agent._chat_openai("fix the bug"))

    results = {message.get("tool_call_id"): str(message.get("content"))
               for message in agent._openai_messages if message.get("role") == "tool"}
    assert executed == ["read_file"]
    assert "READ_EVIDENCE" in results["read"]
    assert "must be called alone" in results["fold"]
    assert agent._fold_count == 0


def test_anthropic_mixed_compaction_preserves_sibling_tool_result() -> None:
    executed: list[str] = []

    async def executor(name: str, _inp: dict) -> str:
        executed.append(name)
        return "READ_EVIDENCE"

    agent = Agent(
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
        tool_executor=executor,
        repo_state_probe=lambda: "unchanged",
    )
    responses = iter([
        SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cost=0.0),
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="tool_use", id="read", name="read_file",
                                input={"file_path": "x.py"}),
                SimpleNamespace(type="tool_use", id="fold", name="compact_context", input={}),
            ],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cost=0.0),
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="done")],
        ),
    ])

    async def stream(*_args, **_kwargs):
        return next(responses)

    agent._call_anthropic_stream = stream
    asyncio.run(agent._chat_anthropic("fix the bug"))

    tool_results = next(
        message["content"] for message in agent._anthropic_messages
        if isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_result" for block in message["content"])
    )
    results = {block["tool_use_id"]: block["content"] for block in tool_results}
    assert executed == ["read_file"]
    assert "READ_EVIDENCE" in results["read"]
    assert "must be called alone" in results["fold"]
    assert agent._fold_count == 0


@pytest.mark.parametrize("stream_fails", [False, True])
def test_anthropic_read_waits_for_prior_write_and_complete_response(tmp_path, stream_fails):
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    executed = []

    async def executor(name, inp):
        executed.append(name)
        if name == "write_file":
            target.write_text(inp["content"], encoding="utf-8")
            return "written"
        return target.read_text(encoding="utf-8")

    agent = Agent(api_key="test", custom_system_prompt="offline", custom_tools=[],
                  is_sub_agent=True, permission_mode="bypassPermissions", skills_enabled=False,
                  tool_executor=executor, repo_state_probe=lambda: "stable")
    blocks = [
        SimpleNamespace(type="tool_use", id="write", name="write_file",
                        input={"file_path": str(target), "content": "new"}),
        SimpleNamespace(type="tool_use", id="read", name="read_file", input={"file_path": str(target)}),
    ]
    calls = 0

    async def stream(on_tool_block_complete=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            if on_tool_block_complete:
                for block in blocks:
                    on_tool_block_complete(vars(block))
            await asyncio.sleep(0)  # The provider may keep streaming after tool blocks.
            if stream_fails:
                raise RuntimeError("incomplete response")
            content = blocks
        else:
            content = [SimpleNamespace(type="text", text="done")]
        return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=0, output_tokens=0, cost=0))

    agent._call_anthropic_stream = stream

    async def exercise():
        try:
            if stream_fails:
                with pytest.raises(RuntimeError, match="incomplete response"):
                    await agent._chat_anthropic("write then read")
                assert executed == []
            else:
                await agent._chat_anthropic("write then read")
                results = [block for message in agent._anthropic_messages
                           if isinstance(message.get("content"), list) for block in message["content"]
                           if block.get("type") == "tool_result"]
                assert executed == ["write_file", "read_file"]
                assert results[-1]["content"].endswith("new")
        finally:
            await agent.shutdown()

    asyncio.run(exercise())


def test_compaction_orders_previous_checkpoint_before_new_and_recent_evidence() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="latest checkpoint"),
                finish_reason="stop",
            )],
            usage=None,
        )

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    asyncio.run(agent._run_compaction_query(
        [{"role": "assistant", "content": "NEW_ARCHIVED_EVIDENCE"}],
        COMPACTION_PROMPT,
        previous_summary="OLDER_PREVIOUS_CHECKPOINT",
        recent_context=[{"role": "assistant", "content": "NEWEST_RAW_TAIL"}],
    ))

    prompt = calls[0]["messages"][-1]["content"]
    assert prompt.index("OLDER_PREVIOUS_CHECKPOINT") < prompt.index("NEW_ARCHIVED_EVIDENCE")
    assert prompt.index("NEW_ARCHIVED_EVIDENCE") < prompt.index("NEWEST_RAW_TAIL")


def test_compaction_retries_one_placeholder_with_conservative_prompt() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    prompts: list[str] = []

    async def query(
        _history: list[dict], prompt: str, _previous_summary: str | None = None,
        _recent_context: list[dict] | None = None,
    ) -> str:
        prompts.append(prompt)
        return "N/A" if len(prompts) == 1 else "Focused tests pass; run the module suite."

    agent._run_compaction_query = query
    result = asyncio.run(agent._generate_folded_session_memory([
        {"role": "system", "content": "canonical context"},
        {"role": "user", "content": "task"},
    ]))

    assert result == "Focused tests pass; run the module suite."
    assert prompts == [COMPACTION_PROMPT, COMPACTION_RETRY_PROMPT]


def test_compaction_rejects_incomplete_provider_response() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )

    async def create(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="partial handoff"),
                finish_reason="length",
            )],
            usage=None,
        )

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(agent._run_compaction_query(
            [{"role": "system", "content": "canonical context"}],
            COMPACTION_PROMPT,
        ))


def test_auto_compaction_waits_until_the_reserved_tail_of_the_context_window() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    agent._openai_messages = [{"role": "system", "content": "canonical context"}]
    calls = 0

    async def compact(*, trigger: str) -> bool:
        nonlocal calls
        assert trigger == "auto"
        calls += 1
        return True

    agent._compact_conversation = compact
    agent.last_input_token_count = 180_000
    asyncio.run(agent._check_and_compact())
    assert calls == 0

    agent.last_input_token_count = 184_000
    asyncio.run(agent._check_and_compact())
    assert calls == 1


def test_auto_compaction_pressure_uses_canonical_history_before_request_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    agent.context_window = 40_000
    agent.last_input_token_count = 0
    agent._openai_messages = [
        {"role": "system", "content": "canonical context"},
        {"role": "user", "content": "original task"},
        {"role": "tool", "tool_call_id": "old", "content": "CANONICAL_EVIDENCE"},
    ]
    compacted: list[str] = []

    agent._conversation_input_estimate = lambda messages: (
        30_000 if "CANONICAL_EVIDENCE" in str(messages) else 1
    )

    async def compact(*, trigger: str) -> bool:
        compacted.append(trigger)
        return True

    agent._compact_conversation = compact

    asyncio.run(agent._check_and_compact())

    assert compacted == ["auto"]


def test_context_overflow_preserves_every_raw_compaction_message() -> None:
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="system", custom_tools=[], is_sub_agent=True,
    )
    history = [
        {"role": "assistant", "content": "old root-cause evidence"},
        {"role": "tool", "tool_call_id": "call-1", "content": "failure in the middle"},
    ]
    original = json.loads(json.dumps(history))
    calls = 0

    async def overflow(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("maximum context length exceeded")

    agent._run_compaction_query = overflow

    assert asyncio.run(agent._generate_folded_session_memory(history)) is None
    assert history == original
    assert calls == 1


def test_openai_compaction_keeps_a_real_recent_tool_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "COMPACTION_KEEP_RECENT_TOKENS", 600)
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    history = [
        {"role": "system", "content": "canonical context"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "old evidence " * 100},
    ]
    recent_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "recent-call",
            "type": "function",
            "function": {"name": "run_shell", "arguments": "{}"},
        }],
    }
    recent_result = {
        "role": "tool", "tool_call_id": "recent-call", "content": "recent output " * 100,
    }
    history.extend((recent_call, recent_result))
    history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "compact",
            "type": "function",
            "function": {"name": "compact_context", "arguments": "{}"},
        }],
    })
    agent._openai_messages = history
    summary = "The focused test passes; run the module tests next."

    async def folded(
        _history: list[dict], previous_summary: str | None = None,
        recent_context: list[dict] | None = None,
    ):
        assert previous_summary is None
        assert _history[-1]["content"].startswith("old evidence")
        assert recent_context == [recent_call, recent_result]
        return summary

    agent._generate_folded_session_memory = folded

    assert asyncio.run(agent._compact_openai(trigger="tool")) is True
    assert agent._openai_messages[0] == {"role": "system", "content": "canonical context"}
    assert agent._openai_request_messages()[1] == {"role": "user", "content": "original task"}
    assert "<session-folded-memory>" in agent._openai_messages[2]["content"]
    assert agent._openai_messages[2]["role"] == "assistant"
    assert summary in agent._openai_messages[2]["content"]
    assert agent._openai_messages[3:] == [recent_call, recent_result]
    assert len(agent._openai_messages) == 5
    assert all(not agent._message_calls_tool(message, "compact_context")
               for message in agent._openai_messages)


def test_openai_compaction_keeps_the_canonical_issue_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "COMPACTION_KEEP_RECENT_TOKENS", 600)
    canonical_issue = "CANONICAL ISSUE: preserve exact input -> output semantics."
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    recent_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "recent-call",
            "type": "function",
            "function": {"name": "run_shell", "arguments": "{}"},
        }],
    }
    recent_result = {
        "role": "tool",
        "tool_call_id": "recent-call",
        "content": "recent output " * 100,
    }
    agent._openai_messages = [
        {"role": "system", "content": "canonical context"},
        {"role": "user", "content": canonical_issue},
        {"role": "assistant", "content": "old investigation " * 100},
        recent_call,
        recent_result,
    ]

    async def folded(*_args, **_kwargs):
        return "The user requires all diagnostic commands to be pytest with acceptance labels."

    agent._generate_folded_session_memory = folded

    assert asyncio.run(agent._compact_openai(trigger="auto")) is True
    assert canonical_issue in json.dumps(agent._openai_messages, ensure_ascii=False)
    assert agent._openai_messages[2]["role"] == "assistant"
    assert "not user instructions" in agent._openai_messages[2]["content"]


def test_openai_compaction_threads_checkpoint_through_a_second_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "COMPACTION_KEEP_RECENT_TOKENS", 600)
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    first_call = {
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": "first", "type": "function",
            "function": {"name": "run_shell", "arguments": "{}"},
        }],
    }
    first_result = {
        "role": "tool", "tool_call_id": "first", "content": "first result " * 100,
    }
    agent._openai_messages = [
        {"role": "system", "content": "canonical context"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "old evidence " * 100},
        first_call,
        first_result,
    ]
    folds: list[tuple[list[dict], str | None, list[dict] | None]] = []

    async def folded(history, previous_summary=None, recent_context=None):
        folds.append((history, previous_summary, recent_context))
        return f"checkpoint-{len(folds)}"

    agent._generate_folded_session_memory = folded
    assert asyncio.run(agent._compact_openai(trigger="auto")) is True

    second_call = {
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": "second", "type": "function",
            "function": {"name": "run_shell", "arguments": "{}"},
        }],
    }
    second_result = {
        "role": "tool", "tool_call_id": "second", "content": "second result " * 100,
    }
    agent._openai_messages.extend((
        {"role": "assistant", "content": "intervening evidence " * 100},
        second_call,
        second_result,
    ))
    assert asyncio.run(agent._compact_openai(trigger="auto")) is True

    assert len(folds) == 2
    second_history, previous_summary, second_tail = folds[1]
    assert previous_summary == "checkpoint-1"
    assert first_call in second_history and first_result in second_history
    assert any("intervening evidence" in str(message.get("content"))
               for message in second_history)
    assert second_tail == [second_call, second_result]
    assert agent._openai_messages[1]["content"] == "original task"
    assert agent._openai_messages[3:] == [second_call, second_result]


def test_anthropic_compaction_threads_checkpoint_through_a_second_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "COMPACTION_KEEP_RECENT_TOKENS", 600)
    agent = Agent(
        api_key="test",
        custom_system_prompt="canonical context",
        custom_tools=[],
        is_sub_agent=True,
    )
    first_call = {"role": "assistant", "content": [{
        "type": "tool_use", "id": "first", "name": "read_file", "input": {},
    }]}
    first_result = {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "first", "content": "first result " * 100,
    }]}
    agent._anthropic_messages = [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": [{"type": "text", "text": "old evidence " * 100}]},
        first_call,
        first_result,
    ]
    folds: list[tuple[list[dict], str | None, list[dict] | None]] = []

    async def folded(history, previous_summary=None, recent_context=None):
        folds.append((history, previous_summary, recent_context))
        return f"checkpoint-{len(folds)}"

    agent._generate_folded_session_memory = folded
    assert asyncio.run(agent._compact_anthropic(trigger="auto")) is True

    second_call = {"role": "assistant", "content": [{
        "type": "tool_use", "id": "second", "name": "run_shell", "input": {},
    }]}
    second_result = {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "second", "content": "second result " * 100,
    }]}
    agent._anthropic_messages.extend((
        {"role": "assistant", "content": [{
            "type": "text", "text": "intervening evidence " * 100,
        }]},
        second_call,
        second_result,
    ))
    assert asyncio.run(agent._compact_anthropic(trigger="auto")) is True

    second_history, previous_summary, second_tail = folds[1]
    assert previous_summary == "checkpoint-1"
    assert first_call in second_history and first_result in second_history
    assert second_tail == [second_call, second_result]
    assert agent._anthropic_messages[0]["content"] == "original task"
    assert agent._anthropic_messages[1]["role"] == "assistant"
    assert agent._anthropic_messages[2:] == [second_call, second_result]


def test_default_system_prompt_has_no_frozen_fold_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SANS_DISABLE_MCP", "SANS_DISABLE_MEMORY", "SANS_DISABLE_SESSION_PERSISTENCE"):
        monkeypatch.setenv(name, "1")
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_tools=[],
        skills_enabled=False,
        is_sub_agent=True,
    )

    assert "# Runtime Fold Guidance" not in agent._system_prompt
    assert "Current context utilization" not in agent._system_prompt


def test_run_shell_observation_exposes_real_failure_status() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        tool_executor=lambda _name, _inp: ShellResult(stderr="broken", exit_code=2),
        repo_state_probe=lambda: "unchanged-state",
    )

    result = asyncio.run(agent._execute_tool_call("run_shell", {"command": "broken"}))
    header, output = result.split("\n", 1)
    observation = json.loads(
        header.removeprefix("<tool_observation>").removesuffix("</tool_observation>")
    )

    assert observation.pop("duration_ms") >= 0
    assert observation == {
        "tool": "run_shell",
        "status": "error",
        "exit_code": 2,
        "state_changed": False,
        "repo_state": "unchanged-st",
        "timed_out": False,
        "execution_error": None,
        "pytest_report": None,
    }
    assert output == "Command failed (exit code 2)\nStderr: broken"
    assert agent._tool_error_streak == 1


@pytest.mark.parametrize(
    "name",
    ["max_model_calls", "max_tool_calls", "max_output_tokens", "max_input_tokens_per_call"],
)
def test_token_limits_must_be_positive_integers(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Agent(
            api_base="http://unused.invalid",
            api_key="test",
            custom_system_prompt="test",
            is_sub_agent=True,
            **{name: 0},
        )


def test_skill_fork_inherits_provider_credentials_and_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_executor = lambda _name, _inp: "unused"
    repo_state_probe = lambda: "state"
    parent = Agent(
        model="DeepSeek-V4-Flash",
        api_base="http://unused.invalid",
        api_key="fork-secret",
        custom_system_prompt="parent",
        custom_tools=[],
        is_sub_agent=True,
        max_output_tokens=1024,
        max_input_tokens_per_call=30_000,
        model_call_timeout_seconds=180,
        max_model_calls=40,
        max_tool_calls=40,
        openai_stream_include_usage=False,
        skills_enabled=False,
        tool_executor=tool_executor,
        repo_state_probe=repo_state_probe,
    )
    captured: dict = {}

    class FakeSubAgent:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def run_once(self, _prompt: str) -> dict:
            return {"text": "done", "tokens": {"input": 1, "output": 1}}

    monkeypatch.setattr(agent_module, "Agent", FakeSubAgent)
    monkeypatch.setattr(
        skills_module,
        "execute_skill",
        lambda *_args: {"context": "fork", "prompt": "child", "allowed_tools": []},
    )
    monkeypatch.setattr(agent_module, "print_sub_agent_start", lambda *_args: None)
    monkeypatch.setattr(agent_module, "print_sub_agent_end", lambda *_args: None)

    assert asyncio.run(parent._execute_skill_tool({"skill_name": "test"})) == "done"
    assert captured["api_key"] == "fork-secret"
    assert captured["max_output_tokens"] == 1024
    assert captured["max_input_tokens_per_call"] == 30_000
    assert captured["model_call_timeout_seconds"] == 180
    assert captured["max_model_calls"] == 40
    assert captured["max_tool_calls"] == 40
    assert captured["_operation_counts"] is parent._operation_counts
    assert captured["openai_stream_include_usage"] is False
    assert captured["skills_enabled"] is False
    assert captured["thinking"] is False
    assert captured["tool_executor"] is tool_executor
    assert captured["repo_state_probe"] is repo_state_probe
    assert "fork-secret" not in repr(parent)


def test_skills_disabled_has_no_injection_or_background_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SANS_DISABLE_MCP", "SANS_DISABLE_MEMORY", "SANS_DISABLE_SESSION_PERSISTENCE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        prompt_module,
        "build_skill_descriptions",
        lambda: pytest.fail("disabled skills must not enter the system prompt"),
    )
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_tools=[],
        skills_enabled=False,
    )
    seen_messages: list[str] = []

    async def fake_chat(user_message: str) -> None:
        seen_messages.append(user_message)
        agent._emit_text("done")

    monkeypatch.setattr(agent, "_chat_openai", fake_chat)
    monkeypatch.setattr(
        agent,
        "_augment_user_message_with_skill_context",
        lambda _message: pytest.fail("disabled skills must not inject retrieved context"),
    )
    monkeypatch.setattr(
        agent,
        "_schedule_background_skill_task",
        lambda _coro: pytest.fail("disabled skills must not schedule background model work"),
    )

    assert asyncio.run(agent.run_once("plain task"))["text"] == "done"
    frozen_prompt = agent._system_prompt
    agent._tool_error_streak = 3
    agent._refresh_runtime_system_prompt()
    assert agent._system_prompt == frozen_prompt
    assert seen_messages == ["plain task"]
    assert agent._prompt_sections["skill_descriptions"] == ""
    assert not agent._background_skill_tasks


def test_custom_system_prompt_keeps_legacy_skills_default() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=False,
    )
    assert agent._skills_enabled is True


def test_product_prompt_accepts_remote_execution_context(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SANS_DISABLE_MCP", "SANS_DISABLE_MEMORY", "SANS_DISABLE_SESSION_PERSISTENCE"):
        monkeypatch.setenv(name, "1")
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_tools=[],
        skills_enabled=False,
        system_prompt_context={
            "cwd": "/testbed",
            "platform": "Linux container",
            "shell": "/bin/bash",
            "git_context": "## main",
            "project_rules": "Use the repository test runner.",
        },
    )
    assert "Working directory: /testbed" in agent._system_prompt
    assert "Platform: Linux container" in agent._system_prompt
    assert "Shell: /bin/bash" in agent._system_prompt
    assert "## main" in agent._system_prompt
    assert "Use the repository test runner." in agent._system_prompt
    assert "every independently reported input, output, exception, and behavior" in agent._system_prompt
    assert "add or update a regression test" in agent._system_prompt
    assert "update_plan" in {tool["name"] for tool in tool_definitions}


def test_update_plan_is_small_and_rejects_multiple_active_steps() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
    )

    result = agent._execute_update_plan_tool({"plan": [
        {"step": "reproduce", "status": "completed"},
        {"step": "fix and verify", "status": "in_progress"},
    ]})
    assert json.loads(result)["plan"] == agent._task_plan
    rejected = agent._execute_update_plan_tool({"plan": [
        {"step": "one", "status": "in_progress"},
        {"step": "two", "status": "in_progress"},
    ]})
    assert rejected.startswith("Error:")
    assert agent._task_plan[-1]["step"] == "fix and verify"


def test_stop_hook_requires_successful_verification_after_latest_edit() -> None:
    state = {"repo": "base", "test_result": ShellResult(exit_code=1, pytest_report=PytestReport(1, failure=1))}

    async def executor(name: str, _inp: dict) -> str:
        if name == "edit_file":
            state["repo"] += "+edit"
            return "Updated file."
        if name == "run_shell":
            return state["test_result"]
        return "ok"

    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[{"name": "edit_file"}, {"name": "run_shell"}],
        is_sub_agent=True,
        tool_executor=executor,
        repo_state_probe=lambda: state["repo"],
    )

    asyncio.run(agent._execute_tool_call("edit_file", {"file_path": "a.py"}))
    assert "no relevant test" in asyncio.run(agent._run_stop_hooks())

    asyncio.run(agent._execute_tool_call("run_shell", {"command": "pytest -q"}))
    assert "latest verification failed" in asyncio.run(agent._run_stop_hooks())

    state["test_result"] = ShellResult(exit_code=0, pytest_report=PytestReport(1))
    asyncio.run(agent._execute_tool_call("run_shell", {"command": "pytest -q"}))
    assert asyncio.run(agent._run_stop_hooks()) is None

    asyncio.run(agent._execute_tool_call("edit_file", {"file_path": "a.py"}))
    assert "changed after" in asyncio.run(agent._run_stop_hooks())


@pytest.mark.parametrize("command", [
    "grep -R pytest .", "pytest --collect-only -q", "pytest --collectonly -q",
    "pytest --co -q", "pytest -q || true",
    "set +e; pytest -q; true", "pytest -q | cat", "echo $(pytest -q)",
    "pytest --version", "pytest --help", "pytest --fixtures", "cargo test --no-run",
])
def test_non_execution_commands_are_not_verification(command: str) -> None:
    assert Agent._is_verification_command(command) is False


@pytest.mark.parametrize("command", [
    "python -m pytest -q", "cd src && cargo test",
    r".venv\Scripts\python.exe -m pytest -q", "python.exe -m pytest -q",
    "py -m pytest -q", r".\gradlew.bat test",
])
def test_real_runner_commands_are_verification(command: str) -> None:
    assert Agent._is_verification_command(command) is True


def test_stop_hook_does_not_let_unrelated_green_hide_current_state_red() -> None:
    state = {"repo": "base"}

    async def executor(name: str, inp: dict) -> str:
        if name == "edit_file":
            state["repo"] = "patched"
            return "Updated file."
        command = str(inp.get("command") or "")
        if "test_target" in command:
            return ShellResult("target behavior is still wrong", exit_code=1, pytest_report=PytestReport(1, failure=1))
        if "test_unrelated" in command:
            return ShellResult(exit_code=0, pytest_report=PytestReport(1))
        return ShellResult(exit_code=0)

    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[{"name": "edit_file"}, {"name": "run_shell"}],
        is_sub_agent=True,
        tool_executor=executor,
        repo_state_probe=lambda: state["repo"],
    )

    asyncio.run(agent._execute_tool_call("edit_file", {"file_path": "a.py"}))
    asyncio.run(agent._execute_tool_call("run_shell", {"command": "pytest -q test_target.py"}))
    asyncio.run(agent._execute_tool_call("run_shell", {"command": "pytest -q test_unrelated.py"}))

    assert asyncio.run(agent._run_stop_hooks()) is not None


def test_repair_runtime_binds_state_real_failures_and_diff_gate() -> None:
    runtime = {"repo": "base", "target_green": False}

    async def executor(name: str, inp: dict) -> str:
        if name == "edit_file":
            runtime["repo"] = "patched"
            return "Updated file."
        if name == "read_file":
            return "   1 | assert target() is False"
        command = str(inp.get("command") or "")
        if command.startswith("python -B -c "):
            return ShellResult(stdout="a" * 64, exit_code=0)
        if "test_target" in command:
            failed = not runtime["target_green"]
            return ShellResult(exit_code=int(failed), pytest_report=PytestReport(
                1, failure=int(failed), cases=(("test_target::test_target", "failure" if failed else "passed", ""),),
            ))
        return ShellResult(exit_code=0, pytest_report=PytestReport(1))

    task = TaskEnvelope(
        task_id="task-1", canonical_issue="target returns False",
        repo_root="/workspace",
    )
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True,
        tool_executor=executor, repo_state_probe=lambda: runtime["repo"], repair_task=task,
    )
    state_result = asyncio.run(agent._execute_tool_call("update_work_state", {
        "acceptance_cases": [{
            "id": "target", "behavior": "target returns False",
            "source_quote": "target returns False",
        }],
        "hypotheses": [],
    }))
    assert "Unknown tool" not in state_result and "Error" not in state_result

    asyncio.run(agent._execute_tool_call("run_shell", {"command": "pytest -q test_target.py"}))
    asyncio.run(agent._execute_tool_call("edit_file", {"file_path": "target.py"}))
    asyncio.run(agent._execute_tool_call("run_shell", {
        "command": "pytest -q test_unrelated.py",
        "acceptance_ids": ["target"], "scope": "full", "oracle": "existing_test",
    }))
    blocked = asyncio.run(agent._run_stop_hooks())
    assert blocked is not None and "unscoped verification pending recheck" in blocked

    runtime["target_green"] = True
    asyncio.run(agent._execute_tool_call("run_shell", {
        "command": "pytest -q test_target.py",
        "acceptance_ids": ["target"], "scope": "focused", "oracle": "issue_reproduction",
    }))
    assert asyncio.run(agent._run_stop_hooks()) is None
    agent._openai_messages.append({"role": "user", "content": "continue"})
    request = agent._openai_request_messages()
    assert "<repair_state>" in request[-1]["content"]
    assert '"id":"target"' in request[-1]["content"]


def test_repair_runtime_requires_a_problem_contract_before_editing() -> None:
    runtime = {"repo": "base", "executed": False}

    def executor(name: str, _inp: dict) -> str:
        runtime["executed"] = True
        runtime["repo"] = "changed"
        return "Updated file."

    agent = Agent(
        api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
        custom_tools=[], is_sub_agent=True, tool_executor=executor,
        repo_state_probe=lambda: runtime["repo"],
        repair_task=TaskEnvelope("task-1", "target returns False", "/workspace"),
    )

    result = asyncio.run(agent._execute_tool_call("edit_file", {"file_path": "target.py"}))
    shell = asyncio.run(agent._execute_tool_call("run_shell", {"command": "printf changed > target.py"}))

    assert "register at least one" in result
    assert "register at least one" in shell
    assert runtime == {"repo": "base", "executed": False}


def test_editor_only_profile_does_not_require_an_impossible_shell_check() -> None:
    state = {"repo": "base"}

    def executor(_name: str, _inp: dict) -> str:
        state["repo"] = "patched"
        return "Updated file."

    agent = Agent(
        custom_system_prompt="fixed", custom_tools=[{"name": "edit_file"}],
        is_sub_agent=True, tool_executor=executor,
        repo_state_probe=lambda: state["repo"],
    )
    asyncio.run(agent._execute_tool_call("edit_file", {"file_path": "answer.txt"}))

    assert asyncio.run(agent._run_stop_hooks()) is None


def test_default_diff_check_catches_untracked_whitespace(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "untracked.txt").write_text("trailing space \n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    agent = Agent(custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True)

    result = asyncio.run(agent._execute_tool_call(
        "run_shell", {"command": agent._diff_check_command}, runtime_owned=True,
    ))

    assert '"status":"error"' in result
    assert "trailing whitespace" in result


def test_repair_evidence_hashes_written_content_instead_of_copying_it() -> None:
    runtime = {"repo": "base"}

    def executor(name: str, _inp: dict) -> str:
        if name == "write_file":
            runtime["repo"] = "patched"
        return "Updated file."

    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True,
        tool_executor=executor, repo_state_probe=lambda: runtime["repo"],
        repair_task=TaskEnvelope("task-1", "create target", "/workspace"),
    )
    asyncio.run(agent._execute_tool_call("update_work_state", {
        "acceptance_cases": [{
            "id": "target", "behavior": "create target", "source_quote": "create target",
        }],
        "hypotheses": [],
    }))
    content = "unique-body-" * 1_000

    asyncio.run(agent._execute_tool_call("write_file", {
        "file_path": "target.py", "content": content,
    }))

    summary = agent._repair_state.evidence[-1].input_summary
    assert content not in summary
    assert '"content_chars": 12000' in summary
    assert '"content_sha256"' in summary


def test_model_cannot_promote_arbitrary_shell_success_to_acceptance_or_diff() -> None:
    runtime = {"repo": "base"}

    def executor(name: str, _inp: dict) -> str:
        if name == "edit_file":
            runtime["repo"] = "patched"
            return "Updated file."
        return "ok"

    agent = Agent(
        api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
        custom_tools=[], is_sub_agent=True, tool_executor=executor,
        repo_state_probe=lambda: runtime["repo"],
        repair_task=TaskEnvelope("task-1", "target returns False", "/workspace"),
    )
    asyncio.run(agent._execute_tool_call("update_work_state", {
        "acceptance_cases": [{
            "id": "target", "behavior": "target returns False",
            "source_quote": "target returns False",
        }],
        "hypotheses": [],
    }))
    asyncio.run(agent._execute_tool_call("edit_file", {"file_path": "target.py"}))
    arbitrary = asyncio.run(agent._execute_tool_call("run_shell", {
        "command": "true", "acceptance_ids": ["target"],
        "scope": "full", "oracle": "existing_test",
    }))
    fake_diff = asyncio.run(agent._execute_tool_call("run_shell", {
        "command": "git diff --check --cached", "acceptance_ids": [],
        "scope": "diff", "oracle": "diff_check",
    }))

    assert "retained as diagnostic evidence" in arbitrary
    assert "diff verification is runtime-owned" in fake_diff
    assert not agent._repair_state.completion_decision().complete
    assert not agent._repair_state.verifications

    asyncio.run(agent._execute_tool_call("run_shell", {"command": "pytest -q unrelated.py"}))
    assert not agent._repair_state.completion_decision().complete


def test_invalid_verification_labels_cannot_erase_a_real_test_failure() -> None:
    def executor(_name: str, inp: dict) -> str:
        if "fail.py" in str(inp.get("command") or ""):
            return ShellResult(exit_code=1, pytest_report=PytestReport(1, failure=1))
        return ShellResult(exit_code=0, pytest_report=PytestReport(1))

    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True,
        tool_executor=executor, repo_state_probe=lambda: "base",
        repair_task=TaskEnvelope(
            "task-1", "target returns False", "/workspace", requires_patch=False,
        ),
    )
    asyncio.run(agent._execute_tool_call("update_work_state", {
        "acceptance_cases": [{
            "id": "target", "behavior": "target returns False",
            "source_quote": "target returns False",
        }],
        "hypotheses": [],
    }))

    failed = asyncio.run(agent._execute_tool_call("run_shell", {
        "command": "pytest -q fail.py", "acceptance_ids": ["target"],
    }))
    asyncio.run(agent._execute_tool_call("run_shell", {
        "command": "pytest -q pass.py", "acceptance_ids": ["target"],
        "scope": "focused", "oracle": "issue_reproduction",
    }))

    decision = agent._repair_state.completion_decision()
    assert "verification metadata missing" in failed
    assert not decision.complete
    assert "unscoped verification pending recheck" in decision.reasons[-1]


def test_verification_that_changes_repo_reports_state_error_without_crashing() -> None:
    runtime = {"repo": "base"}

    def executor(_name: str, _inp: dict) -> str:
        runtime["repo"] = "changed"
        return ShellResult(exit_code=0, pytest_report=PytestReport(1))

    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True,
        tool_executor=executor, repo_state_probe=lambda: runtime["repo"],
        repair_task=TaskEnvelope(
            "task-1", "target returns False", "/workspace", requires_patch=False,
        ),
    )
    asyncio.run(agent._execute_tool_call("update_work_state", {
        "acceptance_cases": [{
            "id": "target", "behavior": "target returns False",
            "source_quote": "target returns False",
        }],
        "hypotheses": [],
    }))

    result = asyncio.run(agent._execute_tool_call(
        "run_shell", {"command": "pytest -q mutating_test.py"},
    ))

    assert "verification command must not change repository state" in result


def test_stop_hook_requires_model_plan_completion() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        repo_state_probe=lambda: "base",
    )
    agent._execute_update_plan_tool({
        "plan": [{"step": "verify second behavior", "status": "pending"}],
    })
    assert "incomplete steps" in asyncio.run(agent._run_stop_hooks())


@pytest.mark.parametrize("use_openai", [False, True], ids=["anthropic", "openai"])
def test_skill_extraction_window_preserves_dialogue_without_retrieval_context(use_openai) -> None:
    agent = Agent(
        api_base="http://unused.invalid" if use_openai else None,
        api_key="test", custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True,
    )
    request = "Preserve invalid rows and report their errors."
    injected = request + "\n\n<retrieved_skills>Prior skill metadata.</retrieved_skills>"
    answer = "Added per-row error reporting."
    history = agent._openai_messages if use_openai else agent._anthropic_messages
    history.extend([
        {"role": "user", "content": injected},
        {"role": "assistant", "content": answer if use_openai else [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": answer},
        ]},
    ])
    agent._set_pending_skill_extraction_window(
        original_user_message=request, assistant_text=answer, retrieved_reference=None,
    )
    window = agent._pop_pending_skill_extraction_window("Keep this behavior.")
    assert window["messages"] == [
        {"role": "user", "content": request},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Keep this behavior."},
    ]
    # The canonical task retains the user's request independently of retrieved hints.
    assert agent._canonical_issue(history) == request


def test_skill_extraction_window_keeps_original_training_feedback() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
    )
    agent._openai_messages.extend(
        {"role": "assistant", "content": f"step {index}"} for index in range(20)
    )
    agent._set_pending_skill_extraction_window(
        original_user_message="durable training feedback",
        assistant_text="done",
        retrieved_reference=None,
    )
    messages = agent._pending_skill_extraction_window["messages"]
    assert messages[0] == {"role": "user", "content": "durable training feedback"}
    assert len(messages) == 12

    captured = {}

    async def fake_evolution(window, **_kwargs):
        captured.update(window)
        return {"ok": True, "action": "add"}

    agent._run_online_skill_evolution = fake_evolution
    asyncio.run(agent.extract_now("confirmed reusable workflow"))
    assert captured["messages"][-1] == {
        "role": "user", "content": "confirmed reusable workflow"
    }


def test_openai_tttool_waits_for_complete_fragmented_arguments(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = TraceRecorder(
        tmp_path / "attempt",
        run_id="run",
        task_id="task",
        attempt_id="attempt",
    )
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        trace=trace,
    )
    stage = {"value": "before"}

    async def stream():
        stage["value"] = "partial"
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id="call-1",
                        function=SimpleNamespace(name="read_file", arguments='{"file_'),
                    )],
                ),
                finish_reason=None,
            )],
        )
        stage["value"] = "complete"
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(name=None, arguments='path":"a.py"}'),
                    )],
                ),
                finish_reason="tool_calls",
            )],
        )
        stage["value"] = "done"
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            choices=[],
        )

    async def create(**_kwargs):
        return stream()

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    tick_stages: dict[int, str] = {}
    ticks = itertools.count(1)

    def monotonic_ns() -> int:
        tick = next(ticks)
        tick_stages[tick] = stage["value"]
        return tick

    captured_timing: dict = {}
    real_timing_metrics = agent_module.timing_metrics

    def capture_timing(started_ns: int, ended_ns: int, **points):
        captured_timing.update(points)
        return real_timing_metrics(started_ns, ended_ns, **points)

    monkeypatch.setattr(agent_module.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(agent_module, "timing_metrics", capture_timing)

    response = asyncio.run(agent._call_openai_stream())

    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"file_path":"a.py"}'
    assert tick_stages[captured_timing["first_tool_ns"]] == "complete"


def test_openai_tttool_records_empty_arguments_at_tool_finish_with_late_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = TraceRecorder(
        tmp_path / "attempt",
        run_id="run",
        task_id="task",
        attempt_id="attempt",
    )
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        trace=trace,
    )

    stage = {"value": "before"}

    async def stream():
        stage["value"] = "initial"
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(name=None, arguments=""),
                    )],
                ),
                finish_reason=None,
            )],
        )
        stage["value"] = "finish"
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id="call-late",
                        function=SimpleNamespace(name="no_args", arguments=""),
                    )],
                ),
                finish_reason="tool_calls",
            )],
        )
        stage["value"] = "usage"
        yield SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=1,
            total_tokens=6,
            prompt_tokens_details=None,
            completion_tokens_details=None,
            reasoning_tokens=None,
            cost=0.0,
        ), choices=[])

    async def create(**_kwargs):
        return stream()

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    tick_stages: dict[int, str] = {}
    ticks = itertools.count(1)

    def monotonic_ns() -> int:
        tick = next(ticks)
        tick_stages[tick] = stage["value"]
        return tick

    captured_timing: dict = {}
    real_timing_metrics = agent_module.timing_metrics

    def capture_timing(started_ns: int, ended_ns: int, **points):
        captured_timing.update(points)
        return real_timing_metrics(started_ns, ended_ns, **points)

    monkeypatch.setattr(agent_module.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(agent_module, "timing_metrics", capture_timing)

    response = asyncio.run(agent._call_openai_stream())
    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"] == "call-late"
    assert tool_call["function"]["name"] == "no_args"
    assert tick_stages[captured_timing["first_tool_ns"]] == "finish"


def test_openai_tttool_ignores_incomplete_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = TraceRecorder(
        tmp_path / "attempt",
        run_id="run",
        task_id="task",
        attempt_id="attempt",
    )
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        trace=trace,
    )

    async def stream():
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id="call-1",
                        function=SimpleNamespace(name="broken", arguments='{"path":'),
                    )],
                ),
                finish_reason="tool_calls",
            )],
        )

    async def create(**_kwargs):
        return stream()

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    captured_timing: dict = {}
    real_timing_metrics = agent_module.timing_metrics

    def capture_timing(started_ns: int, ended_ns: int, **points):
        captured_timing.update(real_timing_metrics(started_ns, ended_ns, **points))
        return captured_timing

    monkeypatch.setattr(agent_module, "timing_metrics", capture_timing)

    asyncio.run(agent._call_openai_stream())
    assert captured_timing["tttool_ms"] is None


def test_openai_max_turns_termination_is_returned_and_reset() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        max_turns=1,
    )
    responses = iter([
        {
            "choices": [{"message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ])

    async def fake_stream() -> dict:
        return next(responses)

    agent._call_openai_stream = fake_stream

    stopped = asyncio.run(agent.run_once("use a tool"))
    completed = asyncio.run(agent.run_once("finish normally"))

    assert stopped["termination_reason"] == {"kind": "max_turns"}
    assert completed["termination_reason"] is None


def test_model_call_budget_counts_one_decision_across_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        max_model_calls=1,
    )
    agent._openai_messages.append({"role": "user", "content": "retry once"})
    calls = 0
    messages_seen: list[list[dict]] = []

    class RetryableError(RuntimeError):
        status_code = 503

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        messages_seen.append(kwargs["messages"])
        if calls == 1:
            raise RetryableError("retry")
        return _empty_stream()

    async def no_sleep(_delay: float) -> None:
        return None

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(agent_module.asyncio, "sleep", no_sleep)

    asyncio.run(agent._call_openai_stream())

    assert calls == 2
    assert messages_seen[0] == messages_seen[1]
    assert messages_seen[0][-1]["content"].endswith("model call 1/1</runtime_state>")
    assert agent._operation_counts["model_calls"] == 1
    with pytest.raises(RuntimeError, match="max_model_calls limit reached"):
        asyncio.run(agent._call_openai_stream())


def test_tool_call_budget_limits_one_parallel_batch_without_overrun() -> None:
    executed: list[str] = []

    async def executor(_name: str, inp: dict) -> str:
        executed.append(inp["file_path"])
        return "read"

    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        max_tool_calls=3,
        tool_executor=executor,
    )
    provider_responses = 0

    async def fake_stream() -> dict:
        nonlocal provider_responses
        provider_responses += 1
        return {
            "choices": [{"message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"file_path": f"{index}.py"}),
                    },
                } for index in range(5)],
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    agent._call_openai_stream = fake_stream

    result = asyncio.run(agent.run_once("read a batch"))

    assert provider_responses == 1
    assert sorted(executed) == ["0.py", "1.py", "2.py"]
    assert result["tool_calls"] == 3
    assert result["termination_reason"] == {"kind": "max_tool_calls"}


def test_skill_invocation_counts_toward_the_same_tool_budget() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        max_tool_calls=1,
    )
    executed: list[str] = []

    async def execute(name: str, _inp: dict) -> str:
        executed.append(name)
        return "ok"

    agent._execute_tool_call_impl = execute

    assert asyncio.run(agent._execute_tool_call("skill", {"skill_name": "workflow"})) == "ok"
    blocked = asyncio.run(agent._execute_tool_call("read_file", {"file_path": "README.md"}))

    assert executed == ["skill"]
    assert "max_tool_calls" in blocked
    assert agent._operation_counts["tool_calls"] == 1


def test_anthropic_max_cost_termination_is_returned() -> None:
    agent = Agent(
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
        max_cost_usd=0.0,
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        content=[SimpleNamespace(
            type="tool_use",
            id="tool-1",
            name="read_file",
            input={},
        )],
    )

    async def fake_stream(*_args, **_kwargs):
        return response

    agent._call_anthropic_stream = fake_stream

    result = asyncio.run(agent.run_once("use a tool"))
    assert result["termination_reason"] == {"kind": "max_cost"}


def test_provider_cost_is_accumulated_and_clients_are_closed() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="fixed",
        custom_tools=[],
        is_sub_agent=True,
    )
    closed: list[str] = []

    class Client:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            closed.append(self.name)

    agent._openai_client = Client("openai")
    agent._anthropic_client = Client("anthropic")
    agent._record_provider_cost({"cost": 0.125})
    agent._record_provider_cost({"cost": 0.25})

    assert agent._get_current_cost_usd() == pytest.approx(0.375)
    asyncio.run(agent.shutdown())
    assert closed == ["openai", "anthropic"]


def test_shutdown_drains_owned_tasks_before_closing_model_transports() -> None:
    async def run():
        agent = Agent(api_base="http://unused.invalid", api_key="test",
                      custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True)
        order = []

        class Client:
            def __init__(self, name):
                self.name = name

            async def close(self):
                order.append(self.name)

        class MCP:
            async def disconnect_all(self):
                order.append("mcp")

        async def background():
            await asyncio.sleep(0)
            order.append("background")

        agent._openai_client = Client("openai")
        agent._anthropic_client = Client("anthropic")
        agent._mcp_manager = MCP()
        agent._schedule_background_skill_task(background())
        try:
            await agent.shutdown()
            observed = list(order)
            drained_at_return = not agent._background_skill_tasks
        finally:
            await agent.drain_background_skill_tasks()
        assert observed == ["background", "mcp", "openai", "anthropic"]
        assert drained_at_return

    asyncio.run(run())
