"""Offline counterexamples and an actual Git/pytest repair through the shared agent loop."""

import asyncio
import json
import subprocess
import sys

import pytest

from agents.agent import Agent, _BudgetExceeded
from agents.repair_state import RepairState, TaskEnvelope
from agents.telemetry import TraceRecorder
from agents.tools import get_solver_tool_definitions
from agents.tool_result import PytestReport, ShellResult


def test_learning_keeps_verification_facts_after_repeated_reads():
    from agents.online_skill_evolution import execution_relations
    agent = Agent.__new__(Agent)
    agent.trace, agent.session_id = None, "session"
    agent._recent_dialog_messages = lambda **kwargs: []
    state = RepairState(TaskEnvelope("learning", "Preserve input", "/repo"), "base")
    agent._repair_state = state
    for success, repo in ((False, "base"), (True, "fixed")):
        state.observe_repo_state(repo)
        state.record_evidence(tool="run_shell", status="ok" if success else "error",
            exit_code=0 if success else 1, input_summary="python -m pytest test_case.py -q",
            repo_before=repo, repo_after=repo, check_fingerprint="same-conditions",
            test_failure_observed=not success, test_pass_observed=success,
            test_cases=(("test_case::test_contract", "passed" if success else "failure", ""),))
    for _ in range(15):
        state.record_evidence(tool="read_file", status="ok", input_summary="TASK.md",
            repo_before="fixed", repo_after="fixed")
    agent._set_pending_skill_extraction_window(original_user_message="Preserve input", assistant_text="",
        retrieved_reference=None)
    rows = agent._pending_skill_extraction_window["execution_evidence"]
    assert len(rows) == 2 and all(row["execution_id"] == state.execution_id for row in rows)
    assert execution_relations(rows)[0]["kind"] == "same_conditions_failure_to_pass"


@pytest.mark.parametrize("invalid_quote", [
    "Preserve complete utterance and option text, including arrows and bracketed tags within that text.",
    "utterance/option text",
])
def test_quote_error_identifies_case_and_preserves_atomic_registration(invalid_quote):
    canonical_quote = "Preserve complete\nutterance and option text, including arrows and bracketed tags within that text."
    task = TaskEnvelope("quote-regression", "Keep the contract.\n" + canonical_quote, "/repo")
    agent = object.__new__(Agent)
    agent._repair_state = RepairState(task, "base")
    agent.trace = None
    original = agent._repair_state.snapshot()
    first = {"id": "AC1", "behavior": "Keep the contract", "source_quote": "Keep the contract."}
    second = {"id": "AC2", "behavior": "Keep complete utterance and option text", "source_quote": invalid_quote}
    response = agent._execute_update_work_state_tool({"acceptance_cases": [first, second]})
    assert "Error updating work state" in response and "AC2" in response and "source_quote" in response
    assert repr(invalid_quote) in response
    if "Preserve complete" in invalid_quote:
        assert repr(canonical_quote) in response and r"JSON \n" in response
    assert agent._repair_state.snapshot() == original
    assert not agent._repair_state.acceptance_cases
    corrected = {**second, "source_quote": canonical_quote}
    wire = json.dumps({"acceptance_cases": [first, corrected]})
    assert r"\n" in wire
    response = agent._execute_update_work_state_tool(json.loads(wire))
    assert not response.startswith("Error")
    assert set(agent._repair_state.acceptance_cases) == {"AC1", "AC2"}
    assert agent._repair_state.acceptance_cases["AC2"].source_quote == canonical_quote
    assert agent._repair_state.task.issue_sha256 == task.issue_sha256


def test_prior_execution_reference_cannot_resolve_to_new_task_observation(tmp_path):
    old = RepairState(TaskEnvelope("old", "Keep old behavior", str(tmp_path)), "base")
    new = RepairState(TaskEnvelope("new", "Keep new behavior", str(tmp_path)), "base")
    old_receipt = old.record_evidence(tool="read_file", status="ok", input_summary="old.py")
    new_receipt = new.record_evidence(tool="read_file", status="ok", input_summary="new.py")
    reader = object.__new__(Agent)
    reader._repair_state = old
    reader._observations = {old_receipt.id: "OLD_TASK_OUTPUT"}
    assert "OLD_TASK_OUTPUT" in reader._read_observation({"evidence_id": old_receipt.id})

    # chat resets both the task state and observation store while retaining conversation history.
    reader._repair_state = new
    reader._observations.clear()
    reader._observations[new_receipt.id] = "NEW_TASK_OUTPUT"
    for part in ("output", "input"):
        assert "unknown evidence_id" in reader._read_observation({"evidence_id": old_receipt.id, "part": part})
    assert "NEW_TASK_OUTPUT" in reader._read_observation({"evidence_id": new_receipt.id})
    assert "new.py" in reader._read_observation({"evidence_id": new_receipt.id, "part": "input"})
    with pytest.raises(ValueError, match="unknown evidence"):
        new.upsert_hypothesis("h", "Keep behavior", "Supported", "Rejected", evidence_ids=[old_receipt.id])
    new.upsert_hypothesis("h", "Keep behavior", "Supported", "Rejected", evidence_ids=[new_receipt.id])


@pytest.mark.parametrize("oracle", ["new_regression", "existing_test", "issue_reproduction"])
def test_green_only_relabelled_check_cannot_prove_repair(oracle):
    state = RepairState(TaskEnvelope("task", "preserve groups", "/repo"), "base")
    state.register_acceptance_case("groups", "preserve groups", "preserve groups")
    state.observe_repo_state("patch")
    for scope, label, ids in [("focused", oracle, ["groups"]), ("diff", "diff_check", [])]:
        evidence = state.record_evidence(
            tool="run_shell", status="ok", exit_code=0, input_summary=scope,
            repo_before="patch", repo_after="patch", check_fingerprint="weak-test",
            test_pass_observed=True,
        )
        state.record_verification(evidence.id, ids, scope=scope, oracle=label, runner_recognized=True)
    assert not state.completion_decision().complete
    assert "no observed failing-to-passing" in state.completion_decision().reasons[0]


def test_hypothesis_revisions_are_claims_and_stale_evidence_is_visible():
    state = RepairState(TaskEnvelope("task", "preserve groups", "/repo"), "base")
    observed = state.record_evidence(
        tool="run_shell", status="error", exit_code=1, input_summary="probe",
        repo_before="base", repo_after="base", output_preview="prefix" * 200 + "AssertionError: lost groups",
    )
    state.upsert_hypothesis("h1", "keep groups", "retained", "lost", status="supported", evidence_ids=[observed.id])
    state.observe_repo_state("patch")
    with pytest.raises(ValueError, match="new evidence or active"):
        state.upsert_hypothesis("h1", "delete groups", "lost", "retained", status="supported", evidence_ids=[observed.id])
    state.upsert_hypothesis("h1", "delete groups", "lost", "retained", evidence_ids=[observed.id])
    context = state.render_runtime_context()
    assert '"claimed_status":"active"' in context and '"evidence_stale":true' in context
    assert '"claim":"keep groups"' in context and "AssertionError: lost groups" in context
    assert [item["revision"] for item in state.snapshot()["hypothesis_history"]] == [1, 2]


@pytest.mark.parametrize("with_trace", [False, True])
def test_real_reproducer_first_repair_rejects_weakened_check_and_recovers(tmp_path, monkeypatch, with_trace):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setattr("agents.agent.Path.home", lambda: tmp_path)
    (repo / "target.py").write_text("def groups():\n    return []\n", encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    for args in [
        ["init", "-q"], ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline"],
    ]:
        subprocess.run(["git", *args], check=True, capture_output=True)
    trace = TraceRecorder(tmp_path / "trace", run_id="offline", task_id="groups", attempt_id="1") if with_trace else None
    agent = Agent(
        api_base="http://unused.invalid", api_key="test", is_sub_agent=True, skills_enabled=False,
        custom_system_prompt="offline tools only", custom_tools=get_solver_tool_definitions("B0"),
        repair_task=TaskEnvelope("groups", "preserve groups", str(repo)), trace=trace,
    )

    async def forbid_provider(*args, **kwargs):
        raise AssertionError("offline regression must not call a model")

    monkeypatch.setattr(agent._openai_client.chat.completions, "create", forbid_provider)
    python = sys.executable.replace("\\", "/")
    test_command = f'cd "{repo.as_posix()}" && "{python}" -B -m pytest -q -p no:cacheprovider test_groups.py'
    test_source = "# witness source\n" * 1100 + "from target import groups\n\ndef test_groups():\n    assert groups() == [1, 2]\n"
    claim = {"acceptance_ids": ["groups"], "scope": "focused", "oracle": "existing_test"}

    async def exercise():
        await agent._execute_tool_call("update_work_state", {
            "acceptance_cases": [{"id": "groups", "behavior": "preserve groups", "source_quote": "preserve groups"}],
            "hypotheses": [],
        })
        for command in ["git branch --list", "git rev-parse --verify refs/heads/missing-diagnostic-ref"]:
            diagnostic = await agent._execute_tool_call("run_shell", {"command": command, **claim})
            assert "retained as diagnostic evidence" in diagnostic
            assert not agent._repair_state.verifications
        invalid = await agent._execute_tool_call("run_shell", {
            "command": test_command.replace("test_groups.py", "missing.py"), **claim,
        })
        assert "not a failing test obligation" in invalid
        assert not agent._repair_state.verifications
        await agent._execute_tool_call("write_file", {"file_path": "test_groups.py", "content": test_source})
        # An implementation already edited before its baseline must remain recoverable.
        await agent._execute_tool_call("read_file", {"file_path": "target.py"})
        await agent._execute_tool_call("edit_file", {"file_path": "target.py", "old_string": "return []", "new_string": "return [1, 2]"})
        await agent._execute_tool_call("run_shell", {"command": test_command, **claim})
        assert "no observed failing-to-passing" in await agent._run_stop_hooks()
        await agent._execute_tool_call("edit_file", {"file_path": "target.py", "old_string": "return [1, 2]", "new_string": "return []"})
        red = await agent._execute_tool_call("run_shell", {"command": test_command})
        assert "1 failed" in red
        red_receipt = agent._repair_state.evidence[-1]
        assert red_receipt.repo_generation > 0 and red_receipt.check_fingerprint
        assert red_receipt.test_failure_observed
        assert '"repair_witness_source_bound":true' in red

        await agent._execute_tool_call("read_file", {"file_path": "target.py"})
        await agent._execute_tool_call("edit_file", {"file_path": "target.py", "old_string": "return []", "new_string": "return [1, 2]"})
        await agent._execute_tool_call("write_file", {"file_path": "test_other.py", "content": "def test_other():\n    assert True\n"})
        await agent._execute_tool_call("run_shell", {"command": test_command.replace("test_groups.py", "test_other.py"), **claim})
        assert "pending recheck" in await agent._run_stop_hooks()
        assert '"state":"pending_recheck"' in agent._repair_state.render_runtime_context()

        await agent._execute_tool_call("write_file", {"file_path": "test_groups.py", "content": "def test_groups():\n    assert True\n"})
        weakened = await agent._execute_tool_call("run_shell", {"command": test_command, **claim})
        assert "1 passed" in weakened
        assert "no observed failing-to-passing" in await agent._run_stop_hooks()

        await agent._execute_tool_call("write_file", {"file_path": "test_groups.py", "content": test_source})
        green = await agent._execute_tool_call("run_shell", {"command": test_command, **claim})
        assert "1 passed" in green
        assert await agent._run_stop_hooks() is None
        assert agent._repair_state.completion_decision().complete
        assert agent._repair_state.evidence[-1].input_summary == agent._diff_check_command
        assert f'"failed":"{red_receipt.id}"' in agent._repair_state.render_runtime_context()
        agent.termination_reason = {"kind": "max_model_calls"}
        await agent._finalize_budget_exit()
        assert agent._completion_evidence["status"] == "complete"
        assert agent.termination_reason == {"kind": "max_model_calls"}
        assert agent._operation_counts["model_calls"] == 0

        # Exercise the real local tool's old 50k truncation boundary, then simulate folded history.
        output_command = f'''"{python}" -c "print('A'*60000 + 'MIDDLE_RECEIPT' + 'Z'*60000)"'''
        output = await agent._execute_tool_call("run_shell", {"command": output_command})
        evidence_id = agent._repair_state.evidence[-1].id
        assert "MIDDLE_RECEIPT" in output
        preview = agent._persist_large_result("run_shell", output)
        assert "MIDDLE_RECEIPT" not in preview
        agent._openai_messages = [{"role": "user", "content": "folded checkpoint"}]
        agent._anthropic_messages = [{"role": "user", "content": "folded checkpoint"}]
        recovered = await agent._execute_tool_call("read_observation", {
            "evidence_id": evidence_id, "offset": 0, "limit": 1, "char_offset": 59000,
        })
        assert "MIDDLE_RECEIPT" in recovered and len(recovered) < 17000
        original_command = await agent._execute_tool_call("read_observation", {"evidence_id": evidence_id, "part": "input"})
        assert output_command in original_command
        assert "unknown evidence_id" in await agent._execute_tool_call("read_observation", {"evidence_id": "../target.py"})
        assert "char_offset" in await agent._execute_tool_call("read_observation", {"evidence_id": evidence_id, "char_offset": -1})
        if trace:
            ref = agent._observations[evidence_id]
            (trace.blob_dir / ref["sha256"]).write_text("corrupted", encoding="utf-8")
            assert "integrity check" in await agent._execute_tool_call("read_observation", {"evidence_id": evidence_id})
        assert agent._operation_counts["model_calls"] == 0
        (tmp_path / "round1-receipt.json").write_text(json.dumps({
            "model_calls": 0, "with_trace": with_trace, "state": agent._repair_state.snapshot(),
            "completion": agent._repair_state.completion_decision().complete,
        }, indent=2), encoding="utf-8")

    asyncio.run(exercise())


@pytest.mark.parametrize("command", ["python reproduce.py", "pytest test_groups.py | tail -10", "pytest --collect-only test_groups.py"])
def test_unrecognized_check_explains_how_to_recover(command):
    output = Agent._format_tool_observation(
        "run_shell", "ok", failed=False, duration_ms=0, before="base", after="base", command=command,
    )
    assert '"verification_runner_recognized":false' in output
    assert "cannot satisfy test acceptance" in output and "read_observation" in output


def test_fingerprint_rejects_partially_bound_directory_selection():
    executions = []

    def execute(name, inp):
        executions.append((name, inp))
        return ShellResult(stdout="a" * 64, exit_code=0)

    agent = Agent(
        api_base="http://unused.invalid", api_key="test", is_sub_agent=True, skills_enabled=False,
        custom_system_prompt="offline", custom_tools=[],
        tool_executor=execute,
        repair_task=TaskEnvelope("task", "preserve groups", "/repo"),
    )
    command = "pytest -q -p no:cacheprovider test_a.py"
    fingerprint = asyncio.run(agent._check_fingerprint(command))
    assert fingerprint
    assert asyncio.run(agent._check_fingerprint("cd /repo && " + command)) == fingerprint
    assert asyncio.run(agent._check_fingerprint("cd . && " + command)) == fingerprint
    for prefix in ["cd /other && ", "cd /repo && echo changed && ", "echo changed && "]:
        wrapped = prefix + command
        assert asyncio.run(agent._check_fingerprint(wrapped)) is None
        output = Agent._format_tool_observation(
            "run_shell", "1 passed", failed=False, duration_ms=0,
            before="base", after="base", command=wrapped,
        )
        assert '"repair_witness_source_bound":false' in output
        assert "cannot form a repair witness" in output
    assert asyncio.run(agent._check_fingerprint("pytest test_a.py tests/")) is None
    assert asyncio.run(agent._check_fingerprint("pytest ../outside.py")) is None
    assert Agent._normalized_verification_command('pytest -k "a  b"') != Agent._normalized_verification_command('pytest -k "a b"')
    assert executions and all(name == "run_shell" for name, _ in executions)
    for result in ["a" * 64, ShellResult(stdout="a" * 64, exit_code=1)]:
        agent._tool_executor = lambda name, inp: result
        assert asyncio.run(agent._check_fingerprint(command)) is None


def test_refined_reproducer_does_not_create_conflicting_rerun_obligations():
    state = RepairState(TaskEnvelope("task", "preserve groups", "/repo"), "base")
    state.register_acceptance_case("groups", "preserve groups", "preserve groups")
    for generation, fingerprint, success in [("base", "v1", False), ("refined-test", "v2", False), ("fixed", "v2", True)]:
        state.observe_repo_state(generation)
        receipt = state.record_evidence(
            tool="run_shell", status="ok" if success else "error", exit_code=0 if success else 1,
            input_summary="pytest test_groups.py", repo_before=generation, repo_after=generation,
            check_fingerprint=fingerprint, test_failure_observed=not success, test_pass_observed=success,
            test_cases=(("test_groups::test_groups", "passed" if success else "failure", "" if success else "AssertionError"),),
        )
        state.record_verification(receipt.id, ["groups"], scope="focused", oracle="issue_reproduction", runner_recognized=True)
    diff = state.record_evidence(tool="run_shell", status="ok", exit_code=0,
                                 input_summary="git diff --check", repo_before="fixed", repo_after="fixed")
    state.record_verification(diff.id, [], scope="diff", oracle="diff_check")
    assert state.completion_decision().complete


@pytest.mark.parametrize("exit_code,word", [(1, "failed"), (0, "passed")])
def test_colored_pytest_outcomes_match_receipts_and_visible_feedback(exit_code, word):
    agent = Agent.__new__(Agent)
    agent.trace = None
    agent._observations = {}
    agent._repair_state = RepairState(TaskEnvelope("color", "preserve groups", "/repo"), "base")
    raw = ShellResult(f"\x1b[32m\x1b[1m2 {word}\x1b[0m in 0.01s", exit_code=exit_code,
                      pytest_report=PytestReport(2, failure=2 if exit_code else 0))
    evidence_id, error = agent._record_repair_evidence(
        "run_shell", {"command": "pytest test_groups.py"}, raw, failed=bool(exit_code),
        duration_ms=0, before="base", after="base", check_fingerprint="unchanged-test",
    )
    assert error is None
    receipt = agent._repair_state.evidence[-1]
    assert receipt.test_failure_observed is bool(exit_code)
    assert receipt.test_pass_observed is (exit_code == 0)
    output = Agent._format_tool_observation(
        "run_shell", raw, failed=bool(exit_code), duration_ms=0, before="base", after="base",
        command="pytest test_groups.py", check_fingerprint="unchanged-test",
    )
    metadata = json.loads(output.split("<tool_observation>")[1].split("</tool_observation>")[0])
    assert metadata["test_failure_observed"] is receipt.test_failure_observed
    assert metadata["test_pass_observed"] is receipt.test_pass_observed
    assert str(raw) in output and agent._observations[evidence_id] == str(raw)


def test_one_repair_cannot_clear_another_weakened_test():
    state = RepairState(TaskEnvelope("two-cases", "preserve a and b", "/repo"), "base")
    for case in ("a", "b"):
        state.register_acceptance_case(case, f"preserve {case}", "preserve a and b")

    def check(case, fingerprint, ok):
        receipt = state.record_evidence(
            tool="run_shell", status="ok" if ok else "error", exit_code=0 if ok else 1,
            input_summary=f"pytest test_{case}.py", repo_before=state.current_repo_state,
            repo_after=state.current_repo_state, check_fingerprint=fingerprint,
            test_failure_observed=not ok, test_pass_observed=ok,
            test_cases=((f"test_{case}::test_{case}", "passed" if ok else "failure", "" if ok else "AssertionError"),),
        )
        return state.record_verification(receipt.id, [case], scope="focused", oracle="issue_reproduction", runner_recognized=True)

    check("a", "original-a", False)
    failed_b = check("b", "original-b", False)
    state.observe_repo_state("a-fixed-b-test-weakened")
    check("a", "original-a", True)
    revised_b = check("b", "weakened-b", True)
    assert failed_b.id in state.completion_decision().unresolved_verification_ids
    payload = json.loads(state.render_runtime_context().removeprefix('<repair_state>').removesuffix('</repair_state>'))
    changed = payload['changed_check_conditions']
    assert changed['pairs'] == [{
        'failed_verification_id': failed_b.id,
        'current_passed_verification_id': revised_b.id,
        'original_check_fingerprint': 'original-b',
        'current_check_fingerprint': 'weakened-b',
    }]
    assert 'controlled wrong' in changed['instruction']
    # A revised test needs its own newly observed baseline, even without coverage labels.
    state.observe_repo_state("revised-b-baseline")
    baseline = state.record_evidence(
        tool="run_shell", status="error", exit_code=1, input_summary="pytest test_b.py",
        repo_before=state.current_repo_state, repo_after=state.current_repo_state,
        check_fingerprint="revised-b", test_failure_observed=True,
        test_cases=(("test_b::test_b", "failure", "AssertionError"),),
    )
    state.record_verification(baseline.id, [], scope="focused", oracle="existing_test", runner_recognized=True)
    state.observe_repo_state("both-fixed")
    check("a", "original-a", True)
    check("b", "revised-b", True)
    assert not state.completion_decision().unresolved_verification_ids
    assert 'changed_check_conditions' not in state.render_runtime_context()


@pytest.mark.parametrize("boundary", ["stale", "unknown", "dropped_case", "other_command", "diagnostic", "failed_again"])
def test_changed_check_guidance_requires_current_comparable_evidence(boundary):
    state = RepairState(TaskEnvelope("changed-check", "preserve behavior", "/repo"), "base")
    state.register_acceptance_case("a", "preserve behavior", "preserve behavior")
    for success in (False, True):
        if success:
            state.observe_repo_state("candidate")
        receipt = state.record_evidence(
            tool="run_shell", status="ok" if success else "error", exit_code=0 if success else 1,
            input_summary="pytest other.py" if success and boundary == "other_command" else "pytest test_a.py",
            repo_before=state.current_repo_state, repo_after=state.current_repo_state,
            check_fingerprint=("" if boundary == "unknown" else "revised") if success else "original",
            test_failure_observed=not success, test_pass_observed=success,
            test_cases=(("test_a::other" if success and boundary == "dropped_case" else "test_a::test_a",
                         "passed" if success else "failure", ""),),
            diagnostic=success and boundary == "diagnostic",
        )
        state.record_verification(receipt.id, ["a"], scope="focused", oracle="existing_test", runner_recognized=True)
    if boundary == "stale":
        state.observe_repo_state("later-edit")
    if boundary == "failed_again":
        receipt = state.record_evidence(
            tool="run_shell", status="error", exit_code=1, input_summary="pytest test_a.py",
            repo_before=state.current_repo_state, repo_after=state.current_repo_state,
            check_fingerprint="revised", test_failure_observed=True,
            test_cases=(("test_a::test_a", "failure", "AssertionError"),),
        )
        state.record_verification(receipt.id, ["a"], scope="focused", oracle="existing_test", runner_recognized=True)
    before = state.snapshot()
    assert state.completion_decision().unresolved_verification_ids
    assert "changed_check_conditions" not in state.render_runtime_context()
    assert state.snapshot() == before


@pytest.mark.parametrize("change", ["product_fix", "dropped_case", "fixture", "pytest_config"])
def test_real_repair_requires_original_failures_under_unchanged_test_conditions(tmp_path, monkeypatch, change):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setattr("agents.agent.Path.home", lambda: tmp_path)
    files = {
        ".gitignore": "experiment/\n__pycache__/\n.pytest_cache/\n",
        "core.py": "SUPPORTED = [0, 1]\ndef answer(value):\n    return 0\n",
        "test_case.py": (
            "import pytest\nfrom core import SUPPORTED, answer\n"
            "@pytest.mark.parametrize('value', SUPPORTED)\n"
            "def test_answer(value):\n    assert answer(value) == value\n"
        ),
    }
    edited_file = "core.py"
    edited_content = files[edited_file].replace("return 0", "return value")
    if change == "dropped_case":
        edited_content = files[edited_file].replace("[0, 1]", "[0]")
    elif change == "fixture":
        files["test_case.py"] = "def test_answer(actual):\n    assert actual == 1\n"
        files["conftest.py"] = (
            "import pytest\nfrom core import answer\n@pytest.fixture\n"
            "def actual():\n    return answer(1)\n"
        )
        edited_file = "conftest.py"
        edited_content = files[edited_file].replace("return answer(1)", "return 1")
    elif change == "pytest_config":
        files["pytest.ini"] = "[pytest]\n"
        files["test_case.py"] += "\ndef test_smoke():\n    assert True\n"
        edited_file, edited_content = "pytest.ini", "[pytest]\naddopts = -k test_smoke\n"
    for filename, content in files.items():
        (repo / filename).write_text(content, encoding="utf-8")
    for args in [
        ["init", "-q"], ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "baseline"],
    ]:
        subprocess.run(["git", *args], check=True, capture_output=True)
    agent = Agent(
        api_base="http://unused.invalid", api_key="test", is_sub_agent=True, skills_enabled=False,
        custom_system_prompt="offline tools only", custom_tools=get_solver_tool_definitions("B0"),
        repair_task=TaskEnvelope(change, "return the expected value", str(repo)),
    )

    async def forbid_provider(*args, **kwargs):
        raise AssertionError("offline regression must not call a model")

    monkeypatch.setattr(agent._openai_client.chat.completions, "create", forbid_provider)
    command = subprocess.list2cmdline([sys.executable, "-B", "-m", "pytest", "test_case.py", "-q", "-p", "no:cacheprovider"])
    labels = {"acceptance_ids": ["answer"], "scope": "focused", "oracle": "issue_reproduction"}

    async def exercise():
        try:
            await agent._execute_tool_call("update_work_state", {
                "acceptance_cases": [{"id": "answer", "behavior": "return the expected value", "source_quote": "return the expected value"}],
                "hypotheses": [],
            })
            await agent._execute_tool_call("run_shell", {"command": command, **labels})
            red = agent._repair_state.evidence[-1]
            assert red.test_failure_observed and red.check_fingerprint
            await agent._execute_tool_call("read_file", {"file_path": edited_file})
            await agent._execute_tool_call("write_file", {"file_path": edited_file, "content": edited_content})
            if change == "product_fix":
                agent.termination_reason = {"kind": "max_model_calls"}
                await agent._finalize_budget_exit()
                assert agent._completion_evidence["status"] == "complete", agent._completion_evidence
                assert 0 < agent._completion_evidence["tool_calls"] <= 8
            else:
                await agent._execute_tool_call("run_shell", {"command": command, **labels})
            green = agent._repair_state.evidence[-1]
            assert green.test_pass_observed
            blocker = await agent._run_stop_hooks()
            actual = subprocess.check_output([sys.executable, "-B", "-c", "from core import answer; print(answer(1))"], text=True).strip()
            (tmp_path / "evidence-receipt.json").write_text(json.dumps({
                "change": change, "stop_reason": blocker, "actual": actual,
                "same_fingerprint": red.check_fingerprint == green.check_fingerprint,
                "state": agent._repair_state.snapshot(), "model_calls": agent._operation_counts["model_calls"],
            }, indent=2), encoding="utf-8")
            assert agent._operation_counts["model_calls"] == 0
            assert actual == ("1" if change == "product_fix" else "0")
            assert (blocker is None) is (change == "product_fix")
            assert agent._repair_state.completion_decision().complete is (change == "product_fix")
            assert (red.check_fingerprint == green.check_fingerprint) is (change in {"product_fix", "dropped_case"})
            agent.termination_reason = {"kind": "max_model_calls"}
            await agent._finalize_budget_exit()
            assert agent._completion_evidence["status"] == ("complete" if change == "product_fix" else "incomplete")
            count = len(agent._repair_state.evidence)
            await agent._finalize_budget_exit()
            assert len(agent._repair_state.evidence) == count
            assert agent.termination_reason == {"kind": "max_model_calls"}
        finally:
            await agent.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("repo_state", ["known", None])
def test_budget_exit_runs_shared_evidence_check_without_calling_model(tmp_path, monkeypatch, provider, repo_state):
    async def probe():
        return repo_state
    agent = Agent(api_base="https://unused.invalid/v1" if provider == "openai" else None,
                  api_key="test", model="test", custom_system_prompt="solver", custom_tools=[],
                  skills_enabled=False, is_sub_agent=True,
                  repair_task=TaskEnvelope("task", "fix task", str(tmp_path)),
                  repo_state_probe=probe,
                  tool_executor=lambda *_: ShellResult(stdout="no clean baseline", exit_code=0))
    async def exhausted():
        agent.termination_reason = {"kind": "max_model_calls"}
        raise _BudgetExceeded("frozen provider refusal")
    monkeypatch.setattr(agent, "_call_" + provider + "_stream", exhausted)
    async def run():
        try:
            result = await agent.run_once("fix task")
            assert result["completion_evidence"]["status"] == ("incomplete" if repo_state else "unknown")
            assert result["termination_reason"] == {"kind": "max_model_calls"}
            assert result["model_calls"] == 0
        finally:
            await agent.shutdown()
    asyncio.run(run())


def test_finalization_shell_calls_have_their_own_hard_limit(tmp_path, monkeypatch):
    calls = []
    agent = Agent(api_base="https://unused.invalid/v1", api_key="test", model="test",
                  custom_system_prompt="solver", custom_tools=[], skills_enabled=False, is_sub_agent=True,
                  repair_task=TaskEnvelope("task", "fix task", str(tmp_path)),
                  repo_state_probe=lambda: "known",
                  tool_executor=lambda name, inp: calls.append((name, inp)) or ShellResult(exit_code=0))
    async def checks():
        for _ in range(9):
            await agent._execute_tool_call_impl("run_shell", {"command": "test-check", "timeout": 90_000})
    monkeypatch.setattr(agent, "_run_stop_hooks", checks)
    async def run():
        try:
            agent.termination_reason = {"kind": "token_budget"}
            await agent._finalize_budget_exit()
            assert len(calls) == 8
            assert all(0 < inp["timeout"] <= 30_000 for _, inp in calls)
            assert agent._completion_evidence["status"] == "unknown"
            assert agent._completion_evidence["tool_calls"] == 8
            assert agent.termination_reason == {"kind": "token_budget"}
        finally:
            await agent.shutdown()
    asyncio.run(run())
