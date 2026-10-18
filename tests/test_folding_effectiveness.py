"""Failures from real SkillsBench prefixes must not spend repeated empty fold calls."""
import asyncio
from copy import deepcopy

import pytest

from agents.agent import Agent, _BudgetExceeded
from benchmarks.context_folding import _FoldingLoop


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_failed_automatic_fold_waits_for_new_source_and_manual_retry_is_available(provider, monkeypatch):
    agent = Agent(api_base="http://unused.invalid" if provider == "openai" else None,
        api_key="test", custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True)
    calls = []

    async def fail(**kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(agent, "_compact_openai" if provider == "openai" else "_compact_anthropic", fail)
    async def exercise():
        assert not await agent._compact_conversation(trigger="auto")
        assert not await agent._compact_conversation(trigger="periodic")
        assert len(calls) == 1
        messages = agent._openai_messages if provider == "openai" else agent._anthropic_messages
        messages.append({"role": "user", "content": "New requirement."})
        assert not await agent._compact_conversation(trigger="auto")
        assert len(calls) == 2
        assert not await agent._compact_conversation(trigger="manual")
        assert len(calls) == 3
    asyncio.run(exercise())


def test_fold_query_allocates_output_to_checkpoint_text(monkeypatch):
    agent = Agent(api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
                  custom_tools=[], is_sub_agent=True)
    seen = []
    async def query(*args, **kwargs):
        seen.append(kwargs)
        return "The source contract is known. No code has been written or validated yet."
    monkeypatch.setattr(agent, "_side_query_once", query)
    asyncio.run(agent._run_compaction_query([{"role": "user", "content": "Implement parser."}], "Summarize."))
    assert seen[0]["disable_thinking"] is True


def test_summary_reservation_uses_actual_output_ceiling_without_shrinking_solve_budget():
    def model():
        return _FoldingLoop(token_budget={"limit": 100000, "charged": 0}, api_base="http://unused.invalid",
            api_key="test", custom_system_prompt="fixed", custom_tools=[], is_sub_agent=True,
            max_output_tokens=8192)
    summary, solve = model(), model()
    args = {"messages": [{"role": "user", "content": "fixed"}], "tools": []}
    summary._preflight_model_input(**args, output_limit=2048)
    solve._preflight_model_input(**args)
    assert solve.token_budget["charged"] - summary.token_budget["charged"] == 6144


def test_explicit_output_bound_scales_headroom_without_changing_default():
    from agents.session_memory import COMPACTION_RESERVE_TOKENS
    common = dict(api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
                  custom_tools=[], is_sub_agent=True, max_input_tokens_per_call=32768)
    configured = Agent(**common, max_output_tokens=8192)
    default = Agent(**common)
    assert configured._compaction_reserve_tokens() == 10240
    assert default._compaction_reserve_tokens() == COMPACTION_RESERVE_TOKENS


def test_gateway_deepseek_side_query_uses_native_thinking_toggle(monkeypatch):
    from types import SimpleNamespace
    agent = Agent(api_base="https://api.commandcode.ai/provider/v1", api_key="test",
        model="deepseek/deepseek-v4.1-flash", custom_system_prompt="fixed", custom_tools=[],
        is_sub_agent=True, thinking=True, openai_extra_body={"reasoning_effort": "low"})
    captured = []
    async def respond(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")],
                               usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    monkeypatch.setattr(agent._openai_client.chat.completions, "create", respond)
    assert asyncio.run(agent._side_query_once("fixed", "fixed", max_tokens=2048,
                     purpose="session_fold", disable_thinking=True)) == "OK"
    assert captured[0]["extra_body"] == {"reasoning_effort": "low", "thinking": {"type": "disabled"}}
    assert captured[0]["max_tokens"] == 2048
    assert agent._openai_request_extra_body()["reasoning_effort"] == "low"


@pytest.mark.parametrize("previous", [None, {"kind": "max_cost", "reason": "prior cost stop"}])
def test_recoverable_summary_input_rejection_preserves_history_and_previous_termination(monkeypatch, previous):
    agent = Agent(api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
        custom_tools=[], skills_enabled=False, is_sub_agent=True, max_input_tokens_per_call=32768,
        max_output_tokens=8192)
    agent._openai_messages += [{"role": "user", "content": "Implement parser."},
                              {"role": "assistant", "content": "Old observations " * 1500},
                              {"role": "assistant", "content": "Latest observation."}]
    original = deepcopy(agent._openai_messages)
    agent.termination_reason = previous
    async def oversized(*args, **kwargs):
        agent._preflight_model_input(messages=[{"role": "user", "content": "x" * 110000}],
                                     tools=[], output_limit=2048)
        pytest.fail("Summary input must be rejected before any provider call")
    monkeypatch.setattr(agent, "_run_compaction_query", oversized)
    async def exercise():
        assert not await agent._compact_conversation(trigger="auto")
        assert agent._openai_messages == original
        assert agent.termination_reason == previous
        agent._preflight_model_input(messages=[{"role": "user", "content": "continue"}], tools=[])
        assert agent.termination_reason == previous
        await agent.shutdown()
    asyncio.run(exercise())


@pytest.mark.parametrize("fatal", [{"kind": "token_budget", "detail": "request_reservation"},
                                   {"kind": "token_budget", "detail": "unknown_usage_after_error"},
                                   {"kind": "max_cost"}, {"kind": "max_model_calls"}])
def test_fold_failure_does_not_clear_fatal_budget_termination(monkeypatch, fatal):
    agent = Agent(api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
                  custom_tools=[], skills_enabled=False, is_sub_agent=True)
    async def rejected(*args, **kwargs):
        agent.termination_reason = dict(fatal)
        raise _BudgetExceeded("Global budget or call allowance is exhausted")
    monkeypatch.setattr(agent, "_run_compaction_query", rejected)
    async def exercise():
        assert await agent._generate_folded_session_memory([{"role": "assistant", "content": "old evidence"}]) is None
        assert agent.termination_reason == fatal
        await agent.shutdown()
    asyncio.run(exercise())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_same_clock_budget_guard_reopens_for_real_user_revision_only(monkeypatch, provider):
    agent = _FoldingLoop(token_budget={"limit": 10000, "charged": 0}, auto_compact=True,
        api_base="http://unused.invalid" if provider == "openai" else None, api_key="test", model="test",
        custom_system_prompt="fixed", custom_tools=[], skills_enabled=False, is_sub_agent=True,
        max_output_tokens=8192, max_input_tokens_per_call=32768, max_decisions=12)
    messages = agent._openai_messages if provider == "openai" else agent._anthropic_messages
    messages.append({"role": "user", "content": "Original requirement.", "_source_user_text": "Original requirement."})
    calls = []
    async def failed(**kwargs):
        calls.append(kwargs["trigger"])
        return False
    monkeypatch.setattr(agent, "_compact_conversation", failed)
    async def exercise():
        assert not await agent._check_and_compact()
        assert calls == ["budget"]
        messages.append({"role": "assistant", "content": "Another model proposal."})
        messages.append({"role": "user", "content": "Runtime notice", "_source_user_text": None})
        assert not await agent._check_and_compact()
        assert calls == ["budget"]
        messages.append({"role": "user", "content": "New explicit constraint.", "_source_user_text": "New explicit constraint."})
        assert not await agent._check_and_compact()
        assert calls == ["budget", "budget"]
        assert not await agent._check_and_compact()
        assert calls == ["budget", "budget"]
        await agent.shutdown()
    asyncio.run(exercise())


def test_successful_fold_records_post_fold_user_authority_before_guarding(monkeypatch):
    agent = _FoldingLoop(token_budget={"limit": 10000, "charged": 0}, auto_compact=True,
        api_base="http://unused.invalid", api_key="test", model="test", custom_system_prompt="fixed",
        custom_tools=[], skills_enabled=False, is_sub_agent=True, max_output_tokens=8192, max_decisions=12)
    original = {"role": "user", "content": "Continue pinned task.", "_source_user_text": "Continue pinned task."}
    agent._openai_messages.append(original)
    calls = []
    async def folded(**kwargs):
        calls.append(kwargs["trigger"])
        # The real folding path may add the canonical task at user authority.
        agent._openai_messages.append({"role": "user", "content": "Pinned task.", "_source_user_text": "Pinned task."})
        return True
    monkeypatch.setattr(agent, "_compact_openai", folded)
    async def exercise():
        assert await agent._check_and_compact()
        assert calls == ["budget"]
        assert not await agent._check_and_compact()
        assert calls == ["budget"]
        agent._openai_messages.append({"role": "user", "content": "Change output.", "_source_user_text": "Change output."})
        assert await agent._check_and_compact()
        assert calls == ["budget", "budget"]
        await agent.shutdown()
    asyncio.run(exercise())
