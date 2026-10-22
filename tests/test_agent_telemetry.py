from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import agents.agent as agent_module
from agents.agent import Agent
from agents.telemetry import TraceRecorder, read_trace, summarize_trace


def _recorder(tmp_path: Path) -> TraceRecorder:
    return TraceRecorder(
        tmp_path / "attempt",
        blob_dir=tmp_path / "blobs",
        run_id="run-1",
        task_id="task-1",
        attempt_id="attempt-1",
        trace_id="trace-1",
    )


def test_b0_environment_disables_persistent_product_state(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SANS_DISABLE_MCP", "SANS_DISABLE_MEMORY", "SANS_DISABLE_SESSION_PERSISTENCE"):
        monkeypatch.setenv(name, "1")
    agent = Agent(
        api_base="http://unused.invalid", api_key="test",
        custom_system_prompt="test", is_sub_agent=False,
    )
    assert not agent._mcp_enabled
    assert not agent._memory_enabled
    assert not agent._session_persistence_enabled


def test_openai_side_query_records_exact_usage_and_purpose(tmp_path: Path) -> None:
    trace = _recorder(tmp_path)
    agent = Agent(
        model="deepseek/deepseek-v4-flash",
        api_base="https://api.commandcode.ai/provider/v1",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
        trace=trace,
    )
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="folded answer",
                        reasoning_content="provisional planning",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=4,
                total_tokens=15,
            ),
        )

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent._anthropic_client = None

    async def exercise() -> str:
        side_query = agent._build_side_query(max_tokens=99, purpose="session_fold")
        assert side_query is not None
        return await side_query("fold system", "fold this conversation")

    assert asyncio.run(exercise()) == "folded answer"
    assert calls[0]["max_tokens"] == 99
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}

    events = read_trace(trace.trace_path)
    started = [event for event in events if event["event_type"] == "model.call.start"]
    terminal = [event for event in events if event["event_type"] == "model.call.end"]
    assert len(started) == 1
    assert len(terminal) == 1
    event = terminal[0]
    assert started[0]["span_id"] == event["span_id"]
    assert event["purpose"] == "session_fold"
    assert started[0]["attributes"]["provider"] == "openai"
    assert started[0]["attributes"]["model"] == agent.model
    assert started[0]["attributes"]["prefix_scope"] == "session_fold"
    assert started[0]["attributes"]["prefix_changed"] is False
    assert started[0]["attributes"]["reasoning_mode"] == "disabled"
    assert event["metrics"]["provider_usage"] == {
        "input_tokens": 11,
        "output_tokens": 4,
        "cached_input_tokens": None,
        "cache_miss_tokens": None,
        "cache_creation_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": 15,
        "provider_reported_cost": None,
    }
    assert event["payload_ref"] is not None
    assert (tmp_path / "blobs" / event["payload_ref"]["sha256"]).is_file()


@pytest.mark.parametrize("finish_reason", ["tool_calls", "length", None])
def test_openai_stream_records_text_tool_timing_and_exact_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str | None
) -> None:
    trace = _recorder(tmp_path)
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
        trace=trace,
        max_model_calls=1,
    )
    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hello", tool_calls=None),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call-1",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"file_path":"base.txt"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=finish_reason,
                )
            ],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
            choices=[],
        ),
    ]

    async def stream():
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    attempts = 0

    async def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = RuntimeError("retry once")
            error.status_code = 429  # type: ignore[attr-defined]
            raise error
        return stream()

    async def no_delay(_seconds: float) -> None:
        return None

    agent._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(agent, "_emit_text", lambda _text: None)
    monkeypatch.setattr(agent_module, "stop_spinner", lambda: None)
    monkeypatch.setattr(agent_module, "print_retry", lambda *_args: None)
    monkeypatch.setattr(agent_module.asyncio, "sleep", no_delay)

    response = asyncio.run(agent._call_openai_stream())
    assert response["choices"][0]["message"]["content"] == "hello"
    calls = response["choices"][0]["message"]["tool_calls"]
    if finish_reason == "tool_calls":
        assert calls[0]["id"] == "call-1"
    else:
        assert calls == []  # Incomplete requests remain in the trace but cannot execute.

    events = read_trace(trace.trace_path)
    terminal = [
        event
        for event in events
        if event["event_type"] == "model.call.end"
    ]
    assert attempts == 2
    assert agent._operation_counts["model_calls"] == 1
    assert len([event for event in events if event["event_type"] == "model.call.error"]) == 1
    assert len(terminal) == 1
    assert terminal[0]["attributes"]["raw_finish_reason"] == finish_reason
    raw = json.loads((trace.blob_dir / terminal[0]["payload_ref"]["sha256"]).read_bytes())
    assert raw["tool_calls"][0]["id"] == "call-1"
    assert terminal[0]["metrics"]["provider_usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cached_input_tokens": 2,
        "cache_miss_tokens": 5,
        "cache_creation_tokens": None,
        "reasoning_tokens": 1,
        "total_tokens": 10,
        "provider_reported_cost": None,
    }
    assert terminal[0]["attributes"]["prefix_changed"] is False
    assert len(terminal[0]["attributes"]["prefix_sha256"]) == 64
    assert len(terminal[0]["attributes"]["system_sha256"]) == 64
    assert len(terminal[0]["attributes"]["tools_sha256"]) == 64
    timing = terminal[0]["metrics"]["timing"]
    assert all(
        timing[name] is not None
        for name in (
            "ttfe_ms",
            "ttft_ms",
            "tttool_ms",
            "service_ms",
            "stream_generation_ms",
        )
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_tool_span_records_one_primitive_state_change_and_external_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "base.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Telemetry Test",
            "-c",
            "user.email=telemetry@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=repository,
        check=True,
    )
    monkeypatch.chdir(repository)

    trace = _recorder(tmp_path / "artifacts")
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
        trace=trace,
    )
    output = "tool-output:" + ("x" * 500)

    async def fake_impl(name: str, inp: dict) -> str:
        assert name == "write_file"
        (repository / inp["file_path"]).write_text(inp["content"], encoding="utf-8")
        return output

    monkeypatch.setattr(agent, "_execute_tool_call_impl", fake_impl)
    result = asyncio.run(
        agent._execute_tool_call(
            "write_file", {"file_path": "created.txt", "content": "created\n"}
        )
    )
    header, body = result.split("\n", 1)
    observation = json.loads(
        header.removeprefix("<tool_observation>").removesuffix("</tool_observation>")
    )
    assert body == output
    assert observation["tool"] == "write_file"
    assert observation["status"] == "ok"
    assert observation["state_changed"] is True
    asyncio.run(
        agent._execute_tool_call(
            "write_file", {"file_path": "created.txt", "content": "created\n"}
        )
    )

    events = read_trace(trace.trace_path)
    terminal = [event for event in events if event["event_type"] == "tool.call.end"]
    assert len(terminal) == 2
    event = terminal[0]
    assert event["status"] == "ok"
    assert event["attributes"]["primitive_operation"] is True
    assert event["attributes"]["state_changed"] is True
    assert event["attributes"]["repo_state_before"] != event["attributes"]["repo_state_after"]
    assert event["payload_ref"] is not None
    blob = trace.blob_dir / event["payload_ref"]["sha256"]
    assert json.loads(blob.read_text(encoding="utf-8")) == {"output": result}
    assert output not in trace.trace_path.read_text(encoding="utf-8")
    assert terminal[1]["attributes"]["state_changed"] is False
    assert terminal[1]["attributes"]["no_progress"] is True

    summary = summarize_trace(events)
    assert summary["tool_calls"] == 2
    assert summary["primitive_operations"] == 2
    assert summary["no_progress_tool_calls"] == 1


def test_subagent_model_span_inherits_parent_tool_span_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _recorder(tmp_path)
    tool_executor = lambda _name, _inp: "unused"
    repo_state_probe = lambda: "state"
    parent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="parent",
        is_sub_agent=True,
        trace=trace,
        skills_enabled=False,
        tool_executor=tool_executor,
        repo_state_probe=repo_state_probe,
    )
    captured: dict = {}

    class FakeSubAgent:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.trace = kwargs["trace"]
            self.model_purpose = kwargs["model_purpose"]

        async def run_once(self, prompt: str) -> dict:
            assert prompt == "inspect child"
            with self.trace.span(
                "model.call", component="model", purpose=self.model_purpose
            ):
                await asyncio.sleep(0)
            return {"text": "child answer", "tokens": {"input": 2, "output": 1}}

    monkeypatch.setattr(agent_module, "Agent", FakeSubAgent)
    monkeypatch.setattr(
        agent_module,
        "get_sub_agent_config",
        lambda _kind: {"system_prompt": "child", "tools": []},
    )
    monkeypatch.setattr(agent_module, "print_sub_agent_start", lambda *_args: None)
    monkeypatch.setattr(agent_module, "print_sub_agent_end", lambda *_args: None)

    result = asyncio.run(
        parent._execute_tool_call(
            "agent",
            {"type": "general", "description": "child", "prompt": "inspect child"},
        )
    )
    assert result == "child answer"
    assert captured["trace"] is trace
    assert captured["model_purpose"] == "subagent"
    assert captured["is_sub_agent"] is True
    assert captured["skills_enabled"] is False
    assert captured["tool_executor"] is tool_executor
    assert captured["repo_state_probe"] is repo_state_probe
    assert captured["_operation_counts"] is parent._operation_counts

    events = read_trace(trace.trace_path)
    tool_start = next(
        event
        for event in events
        if event["event_type"] == "tool.call.start" and event["purpose"] == "agent"
    )
    model_terminal = [
        event for event in events if event["event_type"] == "model.call.end"
    ]
    assert len(model_terminal) == 1
    assert model_terminal[0]["trace_id"] == trace.trace_id
    assert model_terminal[0]["parent_span_id"] == tool_start["span_id"]
    assert model_terminal[0]["purpose"] == "subagent"


@pytest.mark.parametrize("name", sorted(agent_module.PRIMITIVE_TOOLS))
def test_injected_executor_handles_every_primitive_tool(name: str) -> None:
    calls: list[tuple[str, dict]] = []

    async def executor(tool_name: str, inp: dict) -> str:
        calls.append((tool_name, inp))
        return "container-result"

    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
        tool_executor=executor,
    )

    assert asyncio.run(agent._execute_tool_call_impl(name, {"value": 1})) == "container-result"
    assert calls == [(name, {"value": 1})]


def test_injected_probe_and_sync_executor_drive_tool_trace(tmp_path: Path) -> None:
    trace = _recorder(tmp_path)
    states = iter(("container-before", "container-after"))

    async def probe() -> str:
        return next(states)

    def executor(name: str, inp: dict) -> str:
        assert (name, inp) == ("write_file", {"file_path": "x", "content": "y"})
        return "written"

    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
        trace=trace,
        tool_executor=executor,
        repo_state_probe=probe,
    )

    result = asyncio.run(agent._execute_tool_call(
        "write_file", {"file_path": "x", "content": "y"}
    ))
    assert result.endswith("\nwritten")
    event = next(
        item for item in read_trace(trace.trace_path)
        if item["event_type"] == "tool.call.end"
    )
    assert event["attributes"]["repo_state_before"] == "container-before"
    assert event["attributes"]["repo_state_after"] == "container-after"
    assert event["attributes"]["state_changed"] is True


def test_external_timeout_records_cancelled_then_reraises(tmp_path: Path) -> None:
    trace = _recorder(tmp_path)
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        custom_tools=[],
        is_sub_agent=True,
        trace=trace,
    )

    async def never_finishes(_message: str) -> None:
        await asyncio.Future()

    agent._chat_openai = never_finishes

    async def time_out() -> dict:
        return await asyncio.wait_for(agent.run_once("wait"), timeout=0.01)

    with pytest.raises(TimeoutError):
        asyncio.run(time_out())
    assert agent.termination_reason == {"kind": "cancelled"}
    terminal = next(
        event for event in read_trace(trace.trace_path)
        if event["event_type"] == "agent.run_once.error"
    )
    assert terminal["status"] == "cancelled"


def test_explicit_abort_returns_cancelled_result(tmp_path: Path) -> None:
    trace = _recorder(tmp_path)
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        custom_tools=[],
        is_sub_agent=True,
        trace=trace,
    )
    started = asyncio.Event()

    async def never_finishes(_message: str) -> None:
        started.set()
        await asyncio.Future()

    agent._chat_openai = never_finishes

    async def abort_from_owner() -> dict:
        task = asyncio.create_task(agent.run_once("wait"))
        await started.wait()
        agent.abort()
        return await task

    result = asyncio.run(abort_from_owner())
    assert result["termination_reason"] == {"kind": "cancelled"}
    terminal = next(
        event for event in read_trace(trace.trace_path)
        if event["event_type"] == "agent.run_once.error"
    )
    assert terminal["status"] == "cancelled"


def test_tool_failure_detection_uses_result_prefix_not_source_text() -> None:
    agent = Agent(
        api_base="http://unused.invalid",
        api_key="test",
        custom_system_prompt="test",
        is_sub_agent=True,
    )

    assert not agent._looks_like_tool_failure(
        "read_file", "class ValueError(Exception):\n    pass", "class ValueError(Exception):\n    pass"
    )
    assert agent._looks_like_tool_failure("run_shell", "0 errors found", "0 errors found")
    assert agent._looks_like_tool_failure(
        "read_file", "Error reading file: missing", "Error reading file: missing"
    )
    assert agent._looks_like_tool_failure(
        "run_shell", "Command failed (exit code 1)", "Command failed (exit code 1)"
    )
