"""Paid-run accounting must remain conservative across errors and repeated settlement."""
import asyncio
import json

import pytest

from agents.telemetry import TraceRecorder
from benchmarks import research_budget
from benchmarks.research_budget import ResearchBudget, usage_from_trace
from benchmarks.context_folding import _FoldingLoop
from agents.agent import _BudgetExceeded


def event(kind, seq, total=None, purpose="solve", **attributes):
    return {"event_type": kind, "seq": seq, "purpose": purpose, "attributes": attributes,
            "metrics": {"provider_usage": {"total_tokens": total}}}


def test_unknown_usage_uses_its_own_reservation_not_all_calls():
    events = [event("budget.reserve", 1, reserved=100), event("model.call.end", 2, total=70),
              event("budget.reserve", 3, purpose="fold", reserved=40),
              event("model.call.error", 4, purpose="fold", error_type="TimeoutError"),
              event("model.call.error", 5, error_type="_BudgetExceeded")]
    assert usage_from_trace(events) == {"known_tokens": 70, "physical_requests": 2,
        "unknown_usage_requests": 1, "unknown_reserved_tokens": 80}


def test_unreserved_unknown_usage_cannot_silently_charge_zero():
    with pytest.raises(ValueError, match="reservation"):
        usage_from_trace([event("model.call.error", 1)])


def test_summary_reservation_uses_span_even_when_purpose_differs():
    reservation = event("budget.reserve", 1, purpose="main", reserved=40)
    failure = event("model.call.error", 2, purpose="session_fold_evidence")
    reservation["span_id"] = failure["span_id"] = "summary-call"
    assert usage_from_trace([reservation, failure])["unknown_reserved_tokens"] == 80


def test_research_side_query_does_not_retry_network_failure(monkeypatch):
    import httpx
    import openai
    agent = _FoldingLoop(token_budget={"limit": 10000, "charged": 0},
        api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
        custom_tools=[], skills_enabled=False, is_sub_agent=True)
    calls = []

    async def fail(**kwargs):
        calls.append(kwargs)
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://unused.invalid"))

    monkeypatch.setattr(agent, "_run_side_query_call", fail)
    with pytest.raises(openai.APIConnectionError):
        asyncio.run(agent._side_query_once("system", "task", max_tokens=100, purpose="test"))
    assert agent.model_retry_limit == 0 and len(calls) == 1


def test_unknown_usage_rejects_next_call_without_an_extra_physical_request(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"known_api_tokens": 0, "physical_requests": 0,
        "unknown_usage_requests": 0, "additional_api_token_limit": 1000}), encoding="utf-8")
    trace = TraceRecorder(tmp_path / "run", run_id="r", task_id="t", attempt_id="1")
    with ResearchBudget(tmp_path / "run", trace, 100, state_path=state_path, lock_path=tmp_path / "lock") as budget:
        trace.event("budget.reserve", attributes={"reserved": 40})
        trace.event("model.call.error", attributes={"error_type": "TimeoutError"})
        with pytest.raises(_BudgetExceeded):
            asyncio.run(budget.before_call())
        trace.event("model.call.error", attributes={"error_type": "_BudgetExceeded"})
        assert budget.settle()["physical_requests"] == 1


def test_input_limit_refusal_is_typed_and_not_a_physical_request(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"known_api_tokens": 0, "physical_requests": 0,
        "unknown_usage_requests": 0, "additional_api_token_limit": 100000}), encoding="utf-8")
    trace = TraceRecorder(tmp_path / "run", run_id="r", task_id="t", attempt_id="1")
    with ResearchBudget(tmp_path / "run", trace, 10000, state_path=state_path, lock_path=tmp_path / "lock") as budget:
        agent = _FoldingLoop(token_budget=budget.token_budget, before_model_call=budget.before_call,
            api_base="http://unused.invalid", api_key="test", custom_system_prompt="fixed",
            custom_tools=[], skills_enabled=False, is_sub_agent=True, trace=trace,
            max_input_tokens_per_call=1)
        with pytest.raises(_BudgetExceeded, match="before provider"):
            asyncio.run(agent._call_openai_stream())
        assert agent.termination_reason["kind"] == "input_budget"
        from agents.telemetry import read_trace
        assert any(row["event_type"] == "budget.input_rejected" for row in read_trace(trace.trace_path))
        assert budget.settle()["physical_requests"] == 0
        assert budget.usage["unknown_usage_requests"] == 0


def test_serialized_ledger_is_idempotent_and_survives_exception(tmp_path):
    state_path, lock = tmp_path / "state.json", tmp_path / "paid.lock"
    state_path.write_text(json.dumps({"known_api_tokens": 10, "physical_requests": 1,
        "unknown_usage_requests": 0, "additional_api_token_limit": 1000}), encoding="utf-8")
    trace = TraceRecorder(tmp_path / "run", run_id="r", task_id="t", attempt_id="1")
    with pytest.raises(RuntimeError, match="solver"):
        with ResearchBudget(tmp_path / "run", trace, 100, state_path=state_path, lock_path=lock) as budget:
            with pytest.raises(FileExistsError):
                ResearchBudget(tmp_path / "other", trace, 100, state_path=state_path, lock_path=lock).__enter__()
            trace.event("model.call.end", purpose="solve", metrics={"provider_usage": {"total_tokens": 7}})
            asyncio.run(budget.before_call())
            budget.settle()
            assert json.loads(state_path.read_text())["known_api_tokens"] == 17
            raise RuntimeError("solver failed")
    assert not lock.exists()
    assert json.loads(state_path.read_text())["physical_requests"] == 2


def windows_access_error():
    error = PermissionError("destination temporarily locked")
    error.winerror = 5
    return error


def metered_budget(tmp_path):
    state, lock = tmp_path / "state.json", tmp_path / "paid.lock"
    state.write_text(json.dumps({"known_api_tokens": 10, "physical_requests": 1,
        "unknown_usage_requests": 1, "additional_api_token_limit": 1000,
        "unknown_usage_reservations": [{"run": "prior", "requests": 1, "reserved_tokens": 20}]}), encoding="utf-8")
    trace = TraceRecorder(tmp_path / "run", run_id="r", task_id="t", attempt_id="1")
    budget = ResearchBudget(tmp_path / "run", trace, 100, state_path=state, lock_path=lock)
    budget.__enter__()
    trace.event("model.call.end", metrics={"provider_usage": {"total_tokens": 7}})
    trace.event("budget.reserve", attributes={"reserved": 40})
    trace.event("model.call.error", attributes={"error_type": "TimeoutError"})
    return budget, state, lock


@pytest.mark.parametrize("failures", [1, 3])
def test_transient_windows_lock_persists_complete_charge(tmp_path, monkeypatch, failures):
    budget, state, lock = metered_budget(tmp_path)
    replace = research_budget.os.replace
    blocked, sleeps = [], []

    def transient(source, destination):
        if destination == state and len(blocked) < failures:
            assert json.loads(state.read_text())["known_api_tokens"] == 10
            assert json.loads(source.read_text())["known_api_tokens"] == 17
            blocked.append(destination)
            raise windows_access_error()
        replace(source, destination)

    monkeypatch.setattr(research_budget.os, "replace", transient)
    monkeypatch.setattr(research_budget.time, "sleep", sleeps.append)
    budget.settle()
    saved = json.loads(state.read_text())
    assert saved["known_api_tokens"] == 17 and saved["physical_requests"] == 3
    assert saved["unknown_usage_requests"] == 2
    assert [row["reserved_tokens"] for row in saved["unknown_usage_reservations"]] == [20, 80]
    budget.__exit__(None, None, None)
    assert json.loads(state.read_text()) == saved
    assert len(blocked) == failures and sum(sleeps) <= 0.15 + 1e-9
    assert not lock.exists() and not state.with_suffix(".json.pending").exists()
    assert json.loads((tmp_path / "run/usage.json").read_text())["known_tokens"] == 7


def test_permanent_windows_lock_does_not_claim_settlement(tmp_path, monkeypatch):
    budget, state, lock = metered_budget(tmp_path)
    original = json.loads(state.read_text())
    replace = research_budget.os.replace
    calls, sleeps = [], []
    failure = windows_access_error()

    def denied(source, destination):
        if destination == state:
            calls.append(destination)
            raise failure
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(research_budget.os, "replace", denied)
        patch.setattr(research_budget.time, "sleep", sleeps.append)
        with pytest.raises(PermissionError) as raised:
            budget.__exit__(None, None, None)
    assert raised.value is failure and 1 < len(calls) <= 4
    assert len(sleeps) == len(calls) - 1 and sum(sleeps) <= 0.15 + 1e-9
    assert json.loads(state.read_text()) == original and budget.usage == {}
    assert lock.exists() and budget._owned
    assert json.loads(state.with_suffix(".json.pending").read_text())["known_api_tokens"] == 17
    assert not (tmp_path / "run/usage.json").exists()
    budget.__exit__(None, None, None)
    assert json.loads(state.read_text())["known_api_tokens"] == 17 and not lock.exists()


@pytest.mark.parametrize("failed_file", ["state", "usage"])
def test_failed_finalization_reconciles_already_charged_trace_once(tmp_path, monkeypatch, failed_file):
    budget, state, lock = metered_budget(tmp_path)
    budget.settle()
    charged = json.loads(state.read_text())
    replace = research_budget.os.replace
    failed_path = state if failed_file == "state" else tmp_path / "run/usage.json"

    def denied(source, destination):
        if destination == failed_path:
            raise windows_access_error()
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(research_budget.os, "replace", denied)
        patch.setattr(research_budget.time, "sleep", lambda duration: None)
        with pytest.raises(PermissionError):
            budget.__exit__(None, None, None)
    assert lock.exists() and json.loads(state.read_text()) == charged
    assert failed_path.with_suffix(".json.pending").exists()
    budget.__exit__(None, None, None)
    assert not lock.exists() and not state.with_suffix(".json.pending").exists()
    assert not failed_path.with_suffix(".json.pending").exists()
    assert json.loads(state.read_text()) == charged
    assert charged["known_api_tokens"] == 17 and charged["physical_requests"] == 3
    assert charged["unknown_usage_requests"] == 2
    assert [r["reserved_tokens"] for r in charged["unknown_usage_reservations"]] == [20, 80]


@pytest.mark.parametrize("failure", [PermissionError("ordinary permission denial"), OSError("replace failed")])
def test_other_replace_errors_are_immediate(tmp_path, monkeypatch, failure):
    target = tmp_path / "state.json"
    target.write_text('{"before":true}', encoding="utf-8")
    calls, sleeps = [], []

    def denied(source, destination):
        calls.append(destination)
        raise failure

    monkeypatch.setattr(research_budget.os, "replace", denied)
    monkeypatch.setattr(research_budget.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as raised:
        research_budget.save_json(target, {"after": True})
    assert raised.value is failure and len(calls) == 1 and not sleeps
    assert json.loads(target.read_text()) == {"before": True}
    assert json.loads(target.with_suffix(".json.pending").read_text()) == {"after": True}
