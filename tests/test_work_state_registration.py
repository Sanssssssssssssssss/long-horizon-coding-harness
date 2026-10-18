import asyncio
from types import SimpleNamespace

import pytest

from agents.agent import Agent
from agents.repair_state import RepairState, TaskEnvelope
from agents.tool_result import ShellResult


ISSUE = "Return manifest metadata and deliver generated reports."


def runtime(*, requires_patch=True):
    return SimpleNamespace(_repair_state=RepairState(
        TaskEnvelope("registration", ISSUE, "/repo", requires_patch=requires_patch), "base"), trace=None)


def case(identifier="A1", quote="Return manifest metadata", **kwargs):
    return {"id": identifier, "behavior": quote, "source_quote": quote, **kwargs}


def hypothesis(claim):
    return {"id": "H1", "claim": claim, "prediction": "metadata matches", "falsifier": "metadata differs"}


def update(agent, cases, hypotheses=()):
    return Agent._execute_update_work_state_tool(agent, {"acceptance_cases": cases, "hypotheses": list(hypotheses)})


@pytest.mark.parametrize("requires_patch", [True, False])
def test_rejected_first_batch_preserves_cases_and_hypotheses(requires_patch):
    agent = runtime(requires_patch=requires_patch)
    update(agent, [], [hypothesis("existing plan")])
    original = agent._repair_state
    before = original.snapshot()
    result = update(agent, [case(source_kind="existing_test", source_ref="README.md; tests/test_api.py")],
                    [hypothesis("replacement plan")])
    assert "issue-derived" in result and "No acceptance IDs or hypotheses were saved" in result
    assert "retry the same IDs" in result
    assert agent._repair_state is original and original.snapshot() == before


def test_rejected_ids_can_be_corrected_and_committed_ids_stay_immutable():
    agent = runtime()
    assert update(agent, [case(source_kind="existing_test")]).startswith("Error")
    assert not update(agent, [case()], [hypothesis("build the manifest")]).startswith("Error")
    assert agent._repair_state.acceptance_cases["A1"].source_kind == "issue"
    before = agent._repair_state.snapshot()
    result = update(agent, [case("A2", "deliver generated reports"), case(behavior="changed")],
                    [hypothesis("changed")])
    assert "immutable" in result and agent._repair_state.snapshot() == before


def test_issue_and_supplemental_cases_can_be_mixed_then_extended():
    agent = runtime()
    result = update(agent, [case(), case("T1", "test_manifest", source_kind="existing_test", source_ref="tests/test_api.py")])
    assert not result.startswith("Error")
    result = update(agent, [case("P1", "UTF-8 reports", source_kind="project_rule", source_ref="RULES.md")])
    assert not result.startswith("Error")
    assert set(agent._repair_state.acceptance_cases) == {"A1", "T1", "P1"}


def test_bad_issue_quote_rejects_the_whole_batch():
    agent = runtime()
    original = agent._repair_state
    before = original.snapshot()
    result = update(agent, [case(), case("A2", "undeclared requirement")], [hypothesis("new")])
    assert "canonical_issue" in result
    assert agent._repair_state is original and original.snapshot() == before


@pytest.mark.parametrize("requires_patch", [True, False])
def test_execution_gate_still_requires_issue_cases(requires_patch):
    async def exercise():
        executed = []
        async def executor(name, inp):
            executed.append((name, inp))
            return ShellResult(stdout="executed", exit_code=0)
        agent = Agent(api_key="unused", api_base="https://example.invalid/v1", custom_system_prompt="offline",
                      custom_tools=[], skills_enabled=False, is_sub_agent=True, tool_executor=executor,
                      repo_state_probe=lambda: "base", repair_task=TaskEnvelope("registration", ISSUE, "/repo",
                                                                             requires_patch=requires_patch))
        try:
            agent._repair_state.register_acceptance_case("T1", "test_manifest", "test_manifest", source_kind="existing_test")
            for name, inp in [("run_shell", {"command": "pwd"}), ("write_file", {"file_path": "/repo/code.py", "content": "pass\n"}),
                              ("edit_file", {"file_path": "/repo/code.py", "old_string": "pass", "new_string": "done"})]:
                assert "issue-derived" in await agent._execute_tool_call(name, inp)
            assert executed == []
            assert not update(agent, [case()]).startswith("Error")
            assert "executed" in await agent._execute_tool_call("run_shell", {"command": "pwd"})
            assert len(executed) == 1
        finally:
            await agent.shutdown()
    asyncio.run(exercise())
