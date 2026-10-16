import asyncio
from collections import Counter

import agents.agent as agent_module
from agents.agent import Agent


def test_openai_parallel_tool_calls_execute_once_and_preserve_reply_order(monkeypatch) -> None:
    async def exercise() -> None:
        agent = Agent(
            api_base="http://unused.invalid",
            api_key="test",
            custom_system_prompt="test",
            is_sub_agent=True,
        )

        responses = iter([
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {
                                    "name": "read_file" if index % 2 == 0 else "list_files",
                                    "arguments": f'{{"path": "{index}"}}',
                                },
                            }
                            for index in range(10)
                        ],
                    }
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ])
        executions: list[str] = []
        completions: list[str] = []

        async def fake_stream() -> dict:
            return next(responses)

        async def fake_execute(name: str, inp: dict) -> str:
            executions.append(name)
            await asyncio.sleep(0.02 if name == "read_file" else 0)
            completions.append(name)
            return f"result:{inp['path']}"

        async def no_compaction() -> None:
            return None

        agent._call_openai_stream = fake_stream
        agent._execute_tool_call = fake_execute
        agent._check_and_compact = no_compaction
        agent._run_compression_pipeline = lambda: None
        agent._check_budget = lambda: {"exceeded": False}
        agent._persist_large_result = lambda _name, result: result
        agent._record_tool_outcome = lambda _name, _success: None
        agent._looks_like_tool_failure = lambda _name, _raw, _result: False
        agent._refresh_runtime_system_prompt = lambda: None

        monkeypatch.setattr(
            agent_module,
            "check_permission",
            lambda *_args, **_kwargs: {"action": "allow"},
        )

        await agent._chat_openai("run both")

        assert Counter(executions) == Counter({"read_file": 4, "list_files": 4})
        assert len(completions) == 8
        tool_replies = [message for message in agent._openai_messages if message["role"] == "tool"]
        assert [message["tool_call_id"] for message in tool_replies] == [
            f"call-{index}" for index in range(8)
        ]

    asyncio.run(exercise())
