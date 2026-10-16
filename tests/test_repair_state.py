from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
import subprocess
import sys

import pytest

from agents.repair_state import RepairState, TaskEnvelope
from agents.tool_result import PytestReport


ISSUE = "Return true for equivalent sets and preserve unknown comparisons."


def state(*, requires_patch: bool = True, requires_repair_witness: bool = True) -> RepairState:
    repair = RepairState(
        TaskEnvelope("task-1", ISSUE, "/repo", requires_patch=requires_patch,
                     requires_repair_witness=requires_repair_witness),
        "base",
    )
    repair.register_acceptance_case("equivalent", "Equivalent sets are true", "equivalent sets")
    repair.register_acceptance_case(
        "unknown", "Unknown comparisons stay unknown", "unknown comparisons"
    )
    return repair


def evidence(
    repair: RepairState,
    command: str,
    *,
    ok: bool,
    before: str | None = None,
    after: str | None = None,
    cases: tuple[tuple[str, str, str], ...] | None = None,
):
    return repair.record_evidence(
        tool="run_shell",
        status="ok" if ok else "error",
        exit_code=0 if ok else 1,
        input_summary=command,
        repo_before=repair.current_repo_state if before is None else before,
        repo_after=repair.current_repo_state if after is None else after,
        output_preview="passed" if ok else "failed",
        check_fingerprint="test-source-sha256",
        test_failure_observed=not ok,
        test_pass_observed=ok,
        test_cases=cases if cases is not None else ((command, "passed" if ok else "failure", ""),),
    )


def verify(
    repair: RepairState,
    receipt_id: str,
    case_ids: tuple[str, ...],
    *,
    scope: str = "focused",
    oracle: str = "existing_test",
):
    return repair.record_verification(
        receipt_id,
        case_ids,
        scope=scope,
        oracle=oracle,
        runner_recognized=scope != "diff",
    )


def test_task_and_acceptance_contract_are_immutable() -> None:
    repair = state()
    with pytest.raises(FrozenInstanceError):
        repair.task.canonical_issue = "rewritten"  # type: ignore[misc]
    with pytest.raises(TypeError):
        repair.acceptance_cases["invented"] = repair.acceptance_cases["equivalent"]  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical_issue"):
        repair.register_acceptance_case("invented", "Made up", "not in the issue")
    assert not hasattr(repair.acceptance_cases["equivalent"], "status")


def test_non_issue_cases_cannot_replace_the_task_contract() -> None:
    repair = RepairState(
        TaskEnvelope("task-1", ISSUE, "/repo", requires_patch=False),
        "base",
    )
    repair.register_acceptance_case(
        "incidental", "An existing test passes", "test_existing",
        source_kind="existing_test", source_ref="tests/test_existing.py",
    )
    passed = evidence(repair, "pytest tests/test_existing.py", ok=True)
    verify(repair, passed.id, ("incidental",))

    decision = repair.completion_decision()
    assert not decision.complete
    assert "no issue-derived acceptance case is registered" in decision.reasons


def test_hypothesis_transitions_require_real_evidence() -> None:
    repair = state()
    with pytest.raises(ValueError, match="unknown evidence"):
        verify(repair, "e9999", ("equivalent",))
    repair.upsert_hypothesis("h1", "Wrong owner", "Probe fails", "Probe passes")
    with pytest.raises(ValueError, match="unknown evidence"):
        repair.upsert_hypothesis(
            "h1", "Wrong owner", "Probe fails", "Probe passes",
            status="supported", evidence_ids=("e9999",),
        )
    observed = evidence(repair, "python probe.py", ok=False)
    supported = repair.upsert_hypothesis(
        "h1", "Wrong owner", "Probe fails", "Probe passes",
        status="supported", evidence_ids=(observed.id,),
    )
    assert supported.evidence_ids == (observed.id,)


def test_non_shell_evidence_cannot_verify() -> None:
    repair = state()
    observed = repair.record_evidence(
        tool="read_file", status="ok", input_summary="target.py",
        repo_before="base", repo_after="base", output_preview="looks plausible",
    )

    with pytest.raises(ValueError, match="must come from run_shell"):
        verify(repair, observed.id, ("equivalent",))


def test_verification_requires_explicit_exit_and_stable_known_repo() -> None:
    repair = state(requires_patch=False)
    missing_exit = repair.record_evidence(
        tool="run_shell", status="ok", input_summary="pytest -q",
        repo_before="base", repo_after="base",
    )
    with pytest.raises(ValueError, match="explicit exit_code"):
        verify(repair, missing_exit.id, ("equivalent",))

    unknown = RepairState(TaskEnvelope("unknown", ISSUE, "/repo", requires_patch=False), None)
    unknown.register_acceptance_case("equivalent", "Equivalent sets are true", "equivalent sets")
    no_state = unknown.record_evidence(
        tool="run_shell", status="ok", exit_code=0, input_summary="pytest -q",
    )
    with pytest.raises(ValueError, match="known repo_before and repo_after"):
        verify(unknown, no_state.id, ("equivalent",))

    changed = repair.record_evidence(
        tool="run_shell", status="ok", exit_code=0, input_summary="pytest --update",
        repo_before="base", repo_after="changed",
    )
    with pytest.raises(ValueError, match="must not change repository"):
        verify(repair, changed.id, ("equivalent",))


@pytest.mark.parametrize(
    ("red_case", "green_case", "complete", "unresolved"),
    [
        ("equivalent", "unknown", False, ("v0001",)),
        ("equivalent", "equivalent", False, ("v0001",)),
    ],
)
def test_unrelated_green_never_erases_red(
    red_case: str, green_case: str, complete: bool, unresolved: tuple[str, ...]
) -> None:
    repair = state(requires_patch=False)
    red = evidence(repair, "pytest failing", ok=False)
    verify(repair, red.id, (red_case,))
    green = evidence(repair, "pytest passing", ok=True)
    verify(repair, green.id, (green_case,), scope="module")
    decision = repair.completion_decision()
    assert decision.complete is complete
    assert decision.unresolved_verification_ids == unresolved


def test_same_check_green_resolves_red_but_all_cases_still_need_evidence() -> None:
    repair = state(requires_patch=False)
    failed = evidence(repair, "pytest test_sets.py", ok=False)
    verify(repair, failed.id, ("equivalent",))
    passed = evidence(repair, "pytest test_sets.py", ok=True)
    verify(repair, passed.id, ("equivalent",))
    other = evidence(repair, "pytest test_unknown.py", ok=True)
    verify(repair, other.id, ("unknown",))
    decision = repair.completion_decision()
    assert decision.complete
    assert decision.unresolved_verification_ids == ()


@pytest.mark.parametrize("before,after,complete", [
    ((("case", "failure", ""),), (("case", "passed", ""),), True),
    ((("case", "failure", ""),), (("other", "passed", ""),), False),
    ((("case", "failure", ""),), (("case", "skipped", ""),), False),
    ((("case", "failure", ""),), (), False),
    ((), (("case", "passed", ""),), False),
    ((("::", "failure", ""),), (("::", "passed", ""),), False),
    ((("case", "failure", ""),) * 2, (("case", "passed", ""),), False),
    ((("case", "failure", ""),), (("case", "passed", ""),) * 2, False),
])
def test_witness_and_failure_resolution_require_original_failed_cases(before, after, complete):
    repair = state()
    failed = evidence(repair, "pytest test_subject.py", ok=False, cases=before)
    red = verify(repair, failed.id, ("equivalent", "unknown"))
    repair.observe_repo_state("patched")
    passed = evidence(repair, "pytest test_subject.py", ok=True, cases=after)
    verify(repair, passed.id, ("equivalent", "unknown"))
    diff = evidence(repair, "git diff --check", ok=True)
    verify(repair, diff.id, (), scope="diff", oracle="diff_check")
    decision = repair.completion_decision()
    assert decision.complete is complete
    assert (red.id not in decision.unresolved_verification_ids) is complete


def test_real_pytest_dropped_parameter_cannot_prove_repair(tmp_path):
    (tmp_path / "test_subject.py").write_text(
        "import pytest\nfrom target import answer, selected\n"
        "@pytest.mark.parametrize('item', selected)\n"
        "def test_answer(item):\n    assert answer(item) == item\n", encoding="utf-8",
    )
    repair = state()
    for revision, selected, implementation, complete in [
        ("base", "[0, 1]", "0", False),
        ("dropped-failure", "[0]", "0", False),
        ("fixed-with-new-case", "[0, 1, 2]", "item", True),
    ]:
        (tmp_path / "target.py").write_text(
            f"selected = {selected}\ndef answer(item):\n    return {implementation}\n", encoding="utf-8",
        )
        repair.observe_repo_state(revision)
        result = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "test_subject.py", f"--junitxml={revision}.xml"], cwd=tmp_path,
            capture_output=True, text=True, check=False,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        report = PytestReport.read(tmp_path / f"{revision}.xml")
        assert report is not None, result.stdout + result.stderr
        observed = evidence(repair, "pytest test_subject.py", ok=result.returncode == 0, cases=report.cases)
        verify(repair, observed.id, ("equivalent", "unknown"))
        diff = evidence(repair, "git diff --check", ok=True)
        verify(repair, diff.id, (), scope="diff", oracle="diff_check")
        assert repair.completion_decision().complete is complete
        if revision == "base":
            assert result.returncode == 1 and report.collected == 2 and report.failure == 1
        if revision == "dropped-failure":
            assert result.returncode == 0 and report.collected == 1
            assert repair.completion_decision().unresolved_verification_ids
        if complete:
            assert report.collected == 3  # Extra passing cases are allowed.


def test_new_baseline_cannot_erase_an_original_case_that_disappeared():
    repair = state()
    original = evidence(repair, "pytest test_subject.py", ok=False, cases=(("original", "failure", ""),))
    red = verify(repair, original.id, ("equivalent", "unknown"))
    repair.observe_repo_state("new-baseline")
    changed = evidence(repair, "pytest test_subject.py", ok=False, cases=(("replacement", "failure", ""),))
    newer_red = verify(repair, changed.id, ("equivalent", "unknown"))
    repair.observe_repo_state("fixed-replacement")
    passed = evidence(repair, "pytest test_subject.py", ok=True, cases=(("replacement", "passed", ""),))
    verify(repair, passed.id, ("equivalent", "unknown"))
    decision = repair.completion_decision()
    assert red.id in decision.unresolved_verification_ids
    assert newer_red.id not in decision.unresolved_verification_ids


def test_repo_change_invalidates_old_verification_and_success_is_derived() -> None:
    repair = state(requires_patch=False)
    for case_id in repair.acceptance_cases:
        receipt = evidence(repair, f"pytest {case_id}", ok=True)
        recorded = verify(repair, receipt.id, (case_id,))
        assert recorded.success is receipt.succeeded
    assert repair.completion_decision().complete

    edit = repair.record_evidence(
        tool="edit_file",
        status="ok",
        input_summary="edit owner.py",
        repo_before="base",
        repo_after="patched",
    )
    assert edit.repo_generation == 1
    decision = repair.completion_decision()
    assert not decision.complete
    with pytest.raises(ValueError, match="stale evidence"):
        verify(repair, repair.evidence[0].id, ("equivalent",))


def test_unknown_post_edit_state_invalidates_old_verification() -> None:
    repair = state(requires_patch=False)
    for case_id in repair.acceptance_cases:
        receipt = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, receipt.id, (case_id,))
    assert repair.completion_decision().complete

    edit = repair.record_evidence(
        tool="edit_file", status="ok", input_summary="edit owner.py",
        repo_before="base", repo_after=None,
    )

    assert edit.repo_after is None
    assert edit.repo_generation == 1
    assert repair.current_repo_state is None
    assert not repair.completion_decision().complete
    repair.record_evidence(
        tool="read_file", status="ok", input_summary="read owner.py",
        repo_before="patched-again", repo_after="patched-again",
    )
    assert repair.snapshot()["repo"]["initial_state"] == "base"


def test_new_regression_is_kept_as_a_claimed_check() -> None:
    repair = state(requires_patch=False)
    new_test = evidence(repair, "pytest new_test.py", ok=True)
    verify(
        repair, new_test.id, ("equivalent", "unknown"),
        oracle="new_regression",
    )
    assert repair.completion_decision().complete


def test_patch_requirement_is_runtime_derived() -> None:
    repair = state()
    for case_id in repair.acceptance_cases:
        receipt = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, receipt.id, (case_id,))
    assert "repository has no net change" in repair.completion_decision().reasons[0]
    red = evidence(repair, "pytest equivalent", ok=False)
    verify(repair, red.id, ("equivalent",))
    repair.record_evidence(
        tool="edit_file", status="ok", input_summary="edit fix.py",
        repo_before="base", repo_after="patched",
    )
    for case_id in repair.acceptance_cases:
        receipt = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, receipt.id, (case_id,))
    assert "no successful diff verification" in repair.completion_decision().reasons[-1]
    diff = evidence(repair, "git diff --check", ok=True)
    verify(repair, diff.id, (), scope="diff", oracle="diff_check")
    assert repair.completion_decision().complete


def test_ready_gate_allows_boundary_checks_and_keeps_new_failures_blocking() -> None:
    repair = state()
    red = evidence(repair, "pytest equivalent", ok=False)
    verify(repair, red.id, ("equivalent",))
    repair.upsert_hypothesis("h1", "Wrong comparison", "Check fails", "Check passes", evidence_ids=(red.id,))
    repair.record_evidence(
        tool="edit_file", status="ok", input_summary="edit fix.py",
        repo_before="base", repo_after="patched",
    )
    for case_id in repair.acceptance_cases:
        receipt = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, receipt.id, (case_id,))

    assert not repair.completion_decision().complete
    rendered = repair.render_runtime_context()
    payload = json.loads(rendered.removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert payload["finish_gate"]["ready"] is True
    assert payload["finish_gate"]["blockers"] == []
    assert payload["finish_gate"]["action"] == "finish_when_verified"
    assert "next_experiment" not in payload
    assert "sibling or boundary check" in payload["finish_gate"]["instruction"]
    assert "Stop calling tools" not in rendered

    boundary = evidence(repair, "pytest boundary", ok=False)
    verify(repair, boundary.id, ("unknown",))
    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert payload["finish_gate"]["ready"] is False
    assert "action" not in payload["finish_gate"]
    assert repair.completion_decision().unresolved_verification_ids

    boundary = evidence(repair, "pytest boundary", ok=True)
    verify(repair, boundary.id, ("unknown",))
    assert not repair.completion_decision().unresolved_verification_ids
    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert payload["finish_gate"]["ready"] is True

    diff = evidence(repair, "git diff --check", ok=False)
    verify(repair, diff.id, (), scope="diff", oracle="diff_check")
    rendered = repair.render_runtime_context()
    payload = json.loads(rendered.removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert payload["finish_gate"]["ready"] is False
    assert "action" not in payload["finish_gate"]
    assert "final diff has unresolved failure" in payload["finish_gate"]["blockers"][0]


def test_edit_then_revert_cannot_satisfy_patch_requirement() -> None:
    repair = state()
    repair.record_evidence(
        tool="edit_file", status="ok", input_summary="edit fix.py",
        repo_before="base", repo_after="patched",
    )
    repair.record_evidence(
        tool="edit_file", status="ok", input_summary="revert fix.py",
        repo_before="patched", repo_after="base",
    )
    for case_id in repair.acceptance_cases:
        passed = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, passed.id, (case_id,))
    diff = evidence(repair, "git diff --check", ok=True)
    verify(repair, diff.id, (), scope="diff", oracle="diff_check")

    decision = repair.completion_decision()
    assert not decision.complete
    assert "repository has no net change from initial state" in decision.reasons


def test_unrelated_diff_green_does_not_clear_failed_diff_check() -> None:
    repair = state()
    red = evidence(repair, "pytest equivalent", ok=False)
    verify(repair, red.id, ("equivalent",))
    repair.record_evidence(
        tool="edit_file", status="ok", input_summary="edit fix.py",
        repo_before="base", repo_after="patched",
    )
    for case_id in repair.acceptance_cases:
        passed = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, passed.id, (case_id,))
    failed = evidence(repair, "git diff --check", ok=False)
    verify(repair, failed.id, (), scope="diff", oracle="diff_check")
    passed = evidence(repair, "git diff --stat", ok=True)
    verify(repair, passed.id, (), scope="diff", oracle="diff_check")
    decision = repair.completion_decision()
    assert not decision.complete
    assert decision.unresolved_verification_ids == ("v0004",)
    rerun = evidence(repair, "git diff --check", ok=True)
    verify(repair, rerun.id, (), scope="diff", oracle="diff_check")
    assert repair.completion_decision().complete


def test_runtime_context_is_bounded_factual_and_excludes_issue_body() -> None:
    repair = state()
    repair.upsert_hypothesis(
        "h1", "Equality dispatch is too narrow", "Both cases fail", "Either case passes",
    )
    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert payload["next_experiment"]["hypothesis_ids"] == ["h1"]
    assert payload["hypotheses"][0]["evidence_ids"] == []
    assert payload["hypotheses"][0]["claimed_status"] == "active"
    observed = evidence(repair, "python probe.py", ok=False)
    repair.upsert_hypothesis(
        "h1", "Equality dispatch is too narrow", "Both cases fail", "Either case passes",
        status="supported", evidence_ids=(observed.id,),
    )
    rendered = repair.render_runtime_context(latest=1)
    payload = json.loads(rendered.removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert ISSUE not in rendered
    assert payload["canonical_task_sha256"] == repair.task.issue_sha256
    assert [item["id"] for item in payload["acceptance"]] == ["equivalent", "unknown"]
    assert payload["hypotheses"][0]["evidence_ids"] == [observed.id]
    assert payload["latest_evidence"][0]["id"] == observed.id
    assert payload["finish_gate"]["ready"] is False
    assert payload["finish_gate"]["blockers"]
    assert payload["hypotheses"][0]["claimed_status"] == "supported"
    assert "Model claims" in payload["hypothesis_authority"]
    assert "next_experiment" not in payload

    repair.upsert_hypothesis(
        "h1", "Equality dispatch is too narrow", "Both cases fail", "Either case passes",
        status="rejected", evidence_ids=(observed.id,),
    )
    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert "next_experiment" not in payload


def test_next_experiment_selects_active_without_removing_other_claims():
    repair = state()
    observed = evidence(repair, "python probe.py", ok=True)
    before = repair.completion_decision()
    for status in ("active", "supported", "rejected"):
        repair.upsert_hypothesis(
            status, "Model explanation", "Probe distinguishes behavior", "Probe contradicts explanation",
            status=status, evidence_ids=(observed.id,),
        )

    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert payload["next_experiment"]["hypothesis_ids"] == ["active"]
    assert {item["id"]: item["claimed_status"] for item in payload["hypotheses"]} == {
        "active": "active", "supported": "supported", "rejected": "rejected",
    }
    assert all(item["evidence_ids"] == [observed.id] for item in payload["hypotheses"])
    assert "Model claims" in payload["hypothesis_authority"]
    assert repair.completion_decision() == before
    assert not payload["finish_gate"]["ready"]
    assert [item["status"] for item in repair.snapshot()["hypotheses"]] == ["active", "supported", "rejected"]


def test_supported_hypothesis_cannot_clear_unresolved_verification_failure():
    repair = state()
    failed = evidence(repair, "pytest test_equivalent.py", ok=False)
    verification = verify(repair, failed.id, ("equivalent",))
    before = repair.completion_decision()
    repair.upsert_hypothesis(
        "h1", "Model says the cause is supported", "Same case passes", "Same case fails",
        status="supported", evidence_ids=(failed.id,),
    )

    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert "next_experiment" not in payload
    assert payload["hypotheses"][0]["claimed_status"] == "supported"
    assert payload["hypotheses"][0]["evidence_ids"] == [failed.id]
    assert "Model claims" in payload["hypothesis_authority"]
    assert repair.completion_decision() == before
    assert not payload["finish_gate"]["ready"]
    assert verification.id in repair.completion_decision().unresolved_verification_ids
    assert payload["unresolved_reds"][0]["verification_ids"] == [verification.id]
    assert payload["unresolved_reds"][0]["evidence_id"] == failed.id
    assert payload["unresolved_reds"][0]["state"] == "failed"


def test_green_checks_without_a_baseline_request_verification_recovery():
    repair = state()
    repair.upsert_hypothesis("h1", "Wrong owner", "Probe fails", "Probe passes")
    repair.observe_repo_state("patched")
    for case_id in repair.acceptance_cases:
        passed = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, passed.id, (case_id,))
    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert not payload["finish_gate"]["ready"]
    assert "next_experiment" not in payload
    assert "verification_recovery" in payload
    assert "same command" in payload["verification_recovery"]
    assert payload["latest_verifications"][-1]["test_pass_observed"] is True
    # A real current failure must still guide an implementation experiment.
    failed = evidence(repair, "pytest equivalent", ok=False)
    verify(repair, failed.id, ("equivalent",))
    payload = json.loads(repair.render_runtime_context().removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert "next_experiment" in payload and "verification_recovery" not in payload


def test_broader_green_check_keeps_original_baseline_for_current_code_recheck():
    repair = state()
    focused = "python -m pytest test_equivalent.py -q"
    original = evidence(repair, focused, ok=False)
    verify(repair, original.id, ("equivalent",))
    repair.observe_repo_state("patched")
    broad = evidence(repair, "python -m pytest test_equivalent.py test_unknown.py -q", ok=True,
                     cases=((focused, "passed", ""), ("unknown", "passed", "")))
    verify(repair, broad.id, ("equivalent", "unknown"), scope="related")
    before = repair.snapshot()
    decision = repair.completion_decision()
    assert not decision.complete and decision.unresolved_verification_ids
    payload = json.loads(repair.render_runtime_context()[14:-15])
    assert payload["verification_rechecks"] == [{
        "evidence_id": original.id, "command": focused,
        "acceptance_ids": ["equivalent"], "scope": "focused", "oracle": "existing_test",
        "check_fingerprint": original.check_fingerprint,
    }]
    assert "current candidate" in payload["verification_recovery"]
    assert repair.snapshot() == before and repair.completion_decision() == decision
    passed = evidence(repair, focused, ok=True)
    verify(repair, passed.id, ("equivalent",))
    payload = json.loads(repair.render_runtime_context()[14:-15])
    assert "verification_rechecks" not in payload
    assert payload["finish_gate"]["ready"]


@pytest.mark.parametrize("diagnostic,recognized", [(True, True), (False, False)])
def test_recheck_guidance_excludes_diagnostic_or_unrecognized_failures(diagnostic, recognized):
    repair = state()
    failed = repair.record_evidence(
        tool="run_shell", status="error", exit_code=1, input_summary="python -m pytest test_probe.py -q",
        repo_before="base", repo_after="base", test_failure_observed=True, diagnostic=diagnostic,
        check_fingerprint="probe", test_cases=(("probe", "failure", ""),))
    repair.record_verification(failed.id, ["equivalent"], scope="focused", oracle="existing_test",
                               runner_recognized=recognized)
    repair.observe_repo_state("patched")
    passed = evidence(repair, "python -m pytest test_real.py -q", ok=True)
    verify(repair, passed.id, ("equivalent", "unknown"))
    payload = json.loads(repair.render_runtime_context()[14:-15])
    assert not payload.get("verification_rechecks")


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_repair_witness_requirement_is_a_strict_caller_boolean(value):
    with pytest.raises(ValueError, match="requires_repair_witness must be a boolean"):
        TaskEnvelope("task", ISSUE, "/repo", requires_repair_witness=value)


def test_task_snapshot_preserves_build_requirement_and_old_tasks_default_to_repair():
    repair = state(requires_repair_witness=False)
    task = repair.snapshot()["task"]
    assert task["requires_patch"] is True and task["requires_repair_witness"] is False
    fields = {key: value for key, value in task.items() if key != "canonical_issue_sha256"}
    assert TaskEnvelope(**json.loads(json.dumps(fields))).requires_repair_witness is False
    fields.pop("requires_repair_witness")
    assert TaskEnvelope(**fields).requires_repair_witness is True
    with pytest.raises(FrozenInstanceError):
        repair.task.requires_repair_witness = True


def test_model_work_state_update_cannot_relax_the_task_requirement():
    from types import SimpleNamespace
    from agents.agent import Agent
    repair = state()
    runtime = SimpleNamespace(_repair_state=repair, trace=None)
    Agent._execute_update_work_state_tool(runtime, {
        "requires_repair_witness": False,
        "task": {"requires_repair_witness": False},
    })
    assert runtime._repair_state.task.requires_repair_witness is True
    assert not runtime._repair_state.completion_decision().complete


@pytest.mark.parametrize("require_witness", [True, False])
def test_build_only_skips_the_global_repair_pair(require_witness):
    repair = state(requires_repair_witness=require_witness)
    for case_id in repair.acceptance_cases:
        passed = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, passed.id, (case_id,))
    assert "repository has no net change" in repair.completion_decision().reasons[0]
    repair.observe_repo_state("implemented")
    assert not repair.completion_decision().complete
    for case_id in repair.acceptance_cases:
        passed = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, passed.id, (case_id,))
    payload = json.loads(repair.render_runtime_context()[14:-15])
    assert payload["task_mode"] == ("repair" if require_witness else "artifact_build")
    assert payload["check_requirements"]["requires_patch"] is True
    assert payload["check_requirements"]["requires_repair_witness"] is require_witness
    assert payload["finish_gate"]["ready"] is (not require_witness)
    diff = evidence(repair, "git diff --check", ok=True)
    verify(repair, diff.id, (), scope="diff", oracle="diff_check")
    decision = repair.completion_decision()
    assert decision.complete is (not require_witness)
    if require_witness:
        assert any("no observed failing-to-passing" in reason for reason in decision.reasons)


@pytest.mark.parametrize("case_ids", [("equivalent",), ()])
def test_build_keeps_scoped_and_unscoped_failures_pending(case_ids):
    repair = state(requires_repair_witness=False)
    repair.observe_repo_state("implemented")
    failed = evidence(repair, "pytest test_previous.py", ok=False)
    red = verify(repair, failed.id, case_ids)
    for case_id in repair.acceptance_cases:
        passed = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, passed.id, (case_id,))
    diff = evidence(repair, "git diff --check", ok=True)
    verify(repair, diff.id, (), scope="diff", oracle="diff_check")
    assert red.id in repair.completion_decision().unresolved_verification_ids
    assert not repair.completion_decision().complete
    rechecked = evidence(repair, "pytest test_previous.py", ok=True)
    verify(repair, rechecked.id, case_ids)
    assert repair.completion_decision().complete


def test_build_does_not_accept_a_mutating_validation_or_stale_checks():
    repair = state(requires_repair_witness=False)
    repair.observe_repo_state("implemented")
    for case_id in repair.acceptance_cases:
        passed = evidence(repair, f"pytest {case_id}", ok=True)
        verify(repair, passed.id, (case_id,))
    generated = evidence(repair, "python producer.py", ok=True, before="implemented", after="artifacts")
    with pytest.raises(ValueError, match="must not change repository state"):
        verify(repair, generated.id, ("equivalent", "unknown"))
    assert not repair.completion_decision().complete
    assert set(repair.completion_decision().unverified_acceptance_ids) == {"equivalent", "unknown"}


def test_build_guidance_prioritizes_implementation_execution_and_readonly_checks():
    repair = state(requires_repair_witness=False)
    repair.upsert_hypothesis("h1", "Public contract", "Valid output", "Invalid output")
    payload = json.loads(repair.render_runtime_context()[14:-15])
    instruction = payload["next_experiment"]["instruction"]
    assert "smallest runnable component" in instruction and "entry point" in instruction
    assert "read-only pytest" in instruction and "unresolved failures" in instruction
    assert "verification_recovery" not in payload
    assert "Model claims" in payload["hypothesis_authority"]


def test_artifact_prompt_keeps_public_task_and_verification_boundaries():
    from agents.prompt import ARTIFACT_SOLVER_SYSTEM_PROMPT, SOLVER_SYSTEM_PROMPT
    assert "before and\nafter the fix" in SOLVER_SYSTEM_PROMPT
    assert "A green-only test cannot establish repair" in SOLVER_SYSTEM_PROMPT
    for phrase in ("quoting the task exactly", "caller owns", "smallest runnable", "generate the requested artifacts",
                   "read-only pytest", "Never weaken a valid test", "Old failures remain pending", "no network"):
        assert phrase in ARTIFACT_SOLVER_SYSTEM_PROMPT


def test_runtime_context_keeps_all_unresolved_reds_before_recent_evidence() -> None:
    repair = state(requires_patch=False)
    failed = repair.record_evidence(
        tool="run_shell", status="error", exit_code=1,
        input_summary="pytest failing_case", repo_before="base", repo_after="base",
        output_preview="AssertionError: expected True",
    )
    verify(repair, failed.id, ("equivalent",))
    passed = evidence(repair, "pytest unknown_case", ok=True)
    verify(repair, passed.id, ("unknown",))
    recent = repair.record_evidence(
        tool="read_file", status="ok", input_summary="owner.py:10-30",
        repo_before="base", repo_after="base", output_preview="current implementation",
    )

    rendered = repair.render_runtime_context(latest=1)
    payload = json.loads(rendered.removeprefix("<repair_state>").removesuffix("</repair_state>"))
    assert payload["unresolved_reds"] == [{
        "verification_ids": ["v0001"],
        "evidence_id": failed.id,
        "operation_key": failed.operation_key,
        "input": "pytest failing_case",
        "exit_code": 1,
        "repo_generation": 0,
        "state": "failed",
        "output": "AssertionError: expected True",
    }]
    assert [item["id"] for item in payload["latest_evidence"]] == [recent.id]
    assert failed.id not in {item["id"] for item in payload["latest_evidence"]}
    assert [item["id"] for item in payload["latest_verifications"]] == ["v0002"]
