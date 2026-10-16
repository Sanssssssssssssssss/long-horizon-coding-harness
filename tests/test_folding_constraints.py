"""Independent user constraints survive a lossy checkpoint and repeated folding."""
import asyncio

import pytest

from agents.agent import Agent


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_repeated_folding_keeps_all_user_revisions(provider):
    agent = Agent(api_base="http://unused.invalid" if provider == "openai" else None,
                  api_key="test", custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True)
    task = "Add a streaming decoder."
    updates = ["Keep the scalar API unchanged.", "Validate options before consuming input.",
               "Do not modify existing tests."]
    history = [{"role": "user", "content": text, "_source_user_text": text}
               for text in updates]
    tail = [{"role": "assistant", "content": "Ready to implement."}]
    messages = [{"role": "user", "content": task}, *history, *tail]
    if provider == "openai":
        agent._openai_messages = [{"role": "system", "content": "fixed"}, *messages]
    else:
        agent._anthropic_messages = messages

    async def summarize(*args, **kwargs):
        return "Implement a streaming decoder; no implementation changes have been made yet."

    agent._generate_folded_session_memory = summarize
    method = agent._compact_openai if provider == "openai" else agent._compact_anthropic
    for _ in range(2):
        current = agent._openai_messages[1:] if provider == "openai" else agent._anthropic_messages
        agent._compaction_history_parts = lambda *args, **kwargs: (current[1:-1], tail, None)
        assert asyncio.run(method(trigger="manual"))
        folded = agent._openai_messages if provider == "openai" else agent._anthropic_messages
        assert [row["content"] for row in folded if row["role"] == "user"] == [task, *updates]


def test_fold_budget_reserves_combined_user_revisions():
    agent = Agent(api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
                  custom_tools=[], is_sub_agent=True)
    agent._openai_messages = [{"role": "system", "content": "fixed"},
        {"role": "user", "content": "Implement decoder."},
        {"role": "user", "content": "Keep scalar API. " * 1200},
        {"role": "user", "content": "Reject invalid options. " * 1200}]
    task = agent._canonical_issue(agent._openai_messages[1:])
    from agents.session_memory import format_folded_memory
    shell = [{"role": "system", "content": "fixed"}, {"role": "user", "content": task},
             {"role": "assistant", "content": format_folded_memory("")}]
    expected = agent._conversation_input_estimate(
        [*shell, *agent._source_user_updates(agent._openai_messages[1:], task)])
    assert agent._compaction_token_budgets()[2] == expected


def test_oversized_authoritative_requests_do_not_trigger_lossy_folding():
    agent = Agent(api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
                  custom_tools=[], is_sub_agent=True, max_input_tokens_per_call=30000)
    messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": "Task"},
                *[{"role": "user", "content": "Preserve literal constraint " + str(i) + "x" * 2000}
                  for i in range(80)]]
    agent._openai_messages = messages
    assert not asyncio.run(agent._compact_openai(trigger="auto"))
    assert agent._openai_messages == messages
    assert agent._operation_counts["model_calls"] == 0
