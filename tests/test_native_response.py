"""Native Agent response completion using real SDKs and offline HTTP streams."""
import asyncio
import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from agents.agent import Agent
from agents.telemetry import TraceRecorder, read_trace
from agents.tools import tool_definitions


def stream_body(provider, reason, with_tool):
    args = {"file_path": "must-not-write.txt", "content": "unverified"}
    if provider == "openai":
        delta = {"content": "done" if reason == "stop" else "partial answer"}
        if with_tool:
            delta["tool_calls"] = [{"index": 0, "id": "call-1", "type": "function",
                                    "function": {"name": "write_file", "arguments": json.dumps(args)}}]
        body = {"id": "mock", "object": "chat.completion.chunk", "created": 0, "model": "test",
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        return "data: " + json.dumps(body) + "\n\ndata: [DONE]\n\n"
    events = [{"type": "message_start", "message": {
        "id": "mock", "type": "message", "role": "assistant", "model": "test", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 0}}}]
    block = ({"type": "tool_use", "id": "call-1", "name": "write_file", "input": {}} if with_tool
             else {"type": "text", "text": ""})
    delta = ({"type": "input_json_delta", "partial_json": json.dumps(args)} if with_tool else
             {"type": "text_delta", "text": "done" if reason == "end_turn" else "partial answer"})
    events += [{"type": "content_block_start", "index": 0, "content_block": block},
               {"type": "content_block_delta", "index": 0, "delta": delta},
               {"type": "content_block_stop", "index": 0},
               {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                "usage": {"output_tokens": 2}}, {"type": "message_stop"}]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("case,with_tool", [(case, with_tool)
    for case in ("recover", "exhaust", "unknown") for with_tool in (False, True)]
    + [("normal", True), ("budget", True)])
def test_native_response_lifecycle(tmp_path, provider, case, with_tool):
    end, clipped, tool_end = (("stop", "length", "tool_calls") if provider == "openai"
                              else ("end_turn", "max_tokens", "tool_use"))
    replies = {"recover": [(clipped, with_tool), (end, False)],
               "exhaust": [(clipped, with_tool)] * 3,
               "unknown": [(None, with_tool)], "normal": [(tool_end, True), (end, False)],
               "budget": [(clipped, True)]}[case]
    requests, executed = [], []
    transport = httpx if provider == "openai" else httpx2

    async def respond(request):
        index = len(requests)
        requests.append(json.loads(request.content))
        assert index < len(replies), "Unexpected provider request"
        return transport.Response(200, headers={"content-type": "text/event-stream"},
                              text=stream_body(provider, *replies[index]))

    async def execute(name, args):
        executed.append((name, args))
        return "ok"

    async def run():
        agent = Agent(api_base="https://example.invalid/v1" if provider == "openai" else None,
                      api_key="test", model="test", custom_system_prompt="Offline fixture",
                      custom_tools=[tool for tool in tool_definitions if tool["name"] == "write_file"],
                      is_sub_agent=True, skills_enabled=False, permission_mode="bypassPermissions",
                      tool_executor=execute, repo_state_probe=lambda: "same", model_retry_limit=0,
                      max_model_calls=1 if case == "budget" else 8, max_output_tokens=1024,
                      trace=TraceRecorder(tmp_path / "trace", run_id="native", task_id=case, attempt_id=provider))
        client = agent._openai_client if provider == "openai" else agent._anthropic_client
        await client.close()
        http = transport.AsyncClient(transport=transport.MockTransport(respond))
        if provider == "openai":
            agent._openai_client = openai.AsyncOpenAI(api_key="test", http_client=http, max_retries=0)
        else:
            agent._anthropic_client = anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)
        try:
            result = await agent.run_once("Complete this task.")
            assert len(requests) == len(replies)
            assert len(executed) == (1 if case == "normal" else 0)
            assert result["tokens"] == {"input": 10 * len(replies), "output": 2 * len(replies)}
            if case in {"normal", "recover"}:
                assert result["termination_reason"] is None
                assert "done" in result["text"]
                if case == "recover":
                    assert "partial answer" not in result["text"]
            elif case == "budget":
                assert result["termination_reason"]["kind"] == "max_model_calls"
            else:
                assert result["termination_reason"] == {"kind": "incomplete_response",
                    "detail": clipped if case == "exhaust" else "missing_finish_reason"}
                assert "partial answer" not in result["text"]
            events = read_trace(agent.trace.trace_path)
            assert sum(e["event_type"] == "model.call.end" for e in events) == len(replies)
            assert sum(e["event_type"] == "model.truncation_recovery" for e in events) == {
                "recover": 1, "exhaust": 2, "unknown": 0, "normal": 0, "budget": 1}[case]
        finally:
            await agent.shutdown()

    asyncio.run(run())
