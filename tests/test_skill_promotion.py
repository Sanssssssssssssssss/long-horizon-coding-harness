from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from agents import skills
from agents.online_skill_evolution import OnlineSkillCandidate, maintain_online_skill_candidate, promote_staged_skill


def observed_pair(*, fingerprint="same", current_repo="fixed", execution="episode", cases=None):
    before = {"id": "failed", "execution_id": "episode", "command": "python -m pytest test_actor.py -q",
              "tool": "run_shell", "status": "error", "check_fingerprint": "same",
              "repo_before": "base", "repo_after": "base", "test_failure_observed": True,
              "test_cases": [["test_actor::test_contract", "failure", "AssertionError"]]}
    after = {**before, "id": "passed", "status": "ok", "execution_id": execution,
             "check_fingerprint": fingerprint, "repo_before": current_repo, "repo_after": current_repo,
             "test_failure_observed": False, "test_pass_observed": True,
             "test_cases": cases if cases is not None else [["test_actor::test_contract", "passed", ""]]}
    return [before, after]


def test_changed_test_conditions_cannot_be_extracted_as_check_repair(bank):
    from agents.online_skill_evolution import online_ingest
    active = active_skill(bank)
    before = active.read_bytes()
    calls = []

    async def model(system, payload):
        calls.append(system)
        assert json.loads(payload)["execution_relations"][0]["kind"] == "test_conditions_changed"
        return json.dumps({"skills": [{"name": "check-first", "description": "Validate batch options",
            "instructions": "Always fix implementation when a test fails.", "source": "execution",
            "claim_kind": "check_repair", "evidence_ids": ["failed", "passed"]}]})

    result = asyncio.run(online_ingest(messages=[{"role": "user", "content": "Correct the invalid assertion."}],
        execution_evidence=observed_pair(fingerprint="revised"), side_query=model))
    assert result["action"] == "failed" and "trusted transition" in result["error"]
    assert len(calls) == 1 and active.read_bytes() == before


@pytest.mark.parametrize("cases", [[['other', 'passed', '']],
    [['test_actor::test_contract', 'skipped', '']],
    [['test_actor::test_contract', 'passed', ''], ['test_actor::test_contract', 'passed', '']]])
def test_missing_skipped_or_duplicate_failed_identity_cannot_certify_transition(cases):
    from agents.online_skill_evolution import execution_relations
    assert execution_relations(observed_pair(cases=cases)) == []


def test_unchanged_repository_or_other_episode_does_not_prove_repair():
    from agents.online_skill_evolution import execution_relations
    assert execution_relations(observed_pair(current_repo="base"))[0]["kind"] == "outcome_changed_without_recorded_repository_change"
    assert execution_relations(observed_pair(execution="other")) == []


def test_manager_receives_the_same_cited_facts_and_condition_relations(bank):
    from agents.online_skill_evolution import online_ingest
    active = active_skill(bank)
    original = active.read_bytes()
    rows = observed_pair()
    requests = []

    async def model(system, payload):
        requests.append(json.loads(payload))
        if "Extractor" in system:
            return json.dumps({"skills": [{"name": "check-first", "description": "Validate batch options",
                "instructions": "Preserve test conditions while validating the fix.", "source": "execution",
                "claim_kind": "check_repair", "evidence_ids": ["failed", "passed"]}]})
        assert requests[0]["execution_relations"] == requests[1]["execution_relations"]
        assert [row["id"] for row in requests[1]["execution_evidence"]] == ["failed", "passed"]
        assert requests[1]["execution_evidence"][0]["test_cases"] == rows[0]["test_cases"]
        return json.dumps({"action": "merge", "target_skill": "check-first",
            "instruction_updates": [{"old": "", "new": "Check expected values against the public contract first."}]})

    result = asyncio.run(online_ingest(messages=[{"role": "user", "content": "Repair the batch implementation."}],
        execution_evidence=rows, side_query=model))
    assert result["action"] == "staged" and not result["promotion"]["promoted"]
    assert active.read_bytes() == original


def test_direct_manager_cannot_bypass_transition_validation(bank):
    async def forbidden(*args):
        pytest.fail("unsupported claim must be rejected before a model call")
    result = asyncio.run(maintain_online_skill_candidate(candidate=OnlineSkillCandidate(
        "check-first", "Validate batch options", source="execution", claim_kind="check_repair",
        instructions="Always fix code.", evidence_ids=["failed", "passed"]),
        execution_evidence=observed_pair(fingerprint="revised"), side_query=forbidden))
    assert not result["ok"] and result["action"] == "discard"


def test_non_runner_mutation_cannot_certify_a_test_transition():
    from agents.online_skill_evolution import execution_relations
    rows = observed_pair()
    rows[1]["tool"] = "write_file"
    assert execution_relations(rows) == []


def test_model_source_label_cannot_bypass_strong_execution_claim(bank):
    from agents.online_skill_evolution import online_ingest
    async def model(system, payload):
        assert "Extractor" in system
        return json.dumps({"skills": [{"name": "check-first", "description": "Validate batch options",
            "instructions": "I assert a repair without execution.", "source": "user",
            "claim_kind": "check_repair", "evidence_ids": ["unknown"]}]})
    result = asyncio.run(online_ingest(messages=[{"role": "user", "content": "Preserve input."}],
        execution_evidence=[], side_query=model))
    assert result["action"] == "none" and not list(bank.glob(".sans/skill-evolution/candidates/*/SKILL.md"))


@pytest.mark.parametrize("source,kind", [("user", "check_repair"), ("other", "procedure"), ("execution", "unrecognised")])
def test_direct_manager_rejects_untrusted_source_or_claim_kind(bank, source, kind):
    async def forbidden(*args):
        pytest.fail("unsupported source/kind must fail before a model call")
    result = asyncio.run(maintain_online_skill_candidate(candidate=OnlineSkillCandidate(
        "check-first", "Validate batch options", source=source, claim_kind=kind,
        instructions="Unsupported claim.", evidence_ids=["failed", "passed"]),
        execution_evidence=observed_pair(), side_query=forbidden))
    assert not result["ok"] and result["action"] == "discard"


def test_direct_manager_rejects_an_extra_unregistered_reference(bank):
    async def forbidden(*args):
        pytest.fail("unregistered reference must be rejected before a model call")
    result = asyncio.run(maintain_online_skill_candidate(candidate=OnlineSkillCandidate(
        "check-first", "Validate batch options", source="execution", instructions="Verify the fix.",
        evidence_ids=["failed", "passed", "unknown"]), execution_evidence=observed_pair(), side_query=forbidden))
    assert not result["ok"] and "unregistered" in result["error"]


@pytest.fixture
def bank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANS_SKILL_EVOLUTION_DIR", str(tmp_path / ".sans/skill-evolution"))
    skills.reset_skill_cache()
    yield tmp_path
    skills.reset_skill_cache()


def maintain(source="execution", action="add"):
    async def manager(_system, _payload):
        return json.dumps({"action": action, "target_skill": "check-first", "reason": "Observed failed check",
                           "instruction_updates": [{"old": "", "new": "Validate empty input before returning."}]})

    return asyncio.run(maintain_online_skill_candidate(
        candidate=OnlineSkillCandidate(name="check-first", description="Validate batch options", source=source,
                                       instructions="Validate empty input before returning.", evidence="An empty input check failed.",
                                       evidence_ids=["run:attempt:task:check"]),
        side_query=manager,
    ))


def active_skill(bank):
    path = bank / ".sans/skills/check-first/SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\nname: check-first\ndescription: Validate batch options\nversion: 0.1.0\n---\n\nPreserve input.\n", encoding="utf-8")
    skills.reset_skill_cache()
    return path


def receipt(bank, staged):
    # Synthetic local receipts exercise the activation boundary, not a task-performance claim.
    oracle = bank / "oracle.py"
    oracle.write_text("def verify(value):\n    return value == 42\n", encoding="utf-8")
    protocol = bank / "protocol.json"
    protocol.write_text(json.dumps({"task_ids": ["heldout-a", "heldout-b"], "training_task_ids": ["train-a"],
                                    "split": "promotion_test", "baseline_sha256": staged["baseline_sha256"],
                                    "candidate_sha256": staged["candidate_sha256"]}), encoding="utf-8")
    common = {"completed": True, "task_ids": ["heldout-a", "heldout-b"], "evaluation_id": "matched-run",
              "generation_failures": 0}
    return {"completed": True, "task_level": True, "evaluation_id": "matched-run",
            "oracle": {"kind": "local_task_verifier", "path": str(oracle), "sha256": digest(oracle)},
            "protocol": {"path": str(protocol), "sha256": digest(protocol)},
            "baseline": {**common, "average_score": 0.5, "hard_failures": 1, "hard_failure_keys": [["heldout-a", "empty-input"]]},
            "candidate": {**common, "average_score": 1.0, "hard_failures": 0, "hard_failure_keys": []}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("action", ["add", "merge"])
def test_execution_candidate_is_staged_and_not_discoverable(bank, action):
    active = active_skill(bank) if action == "merge" else bank / ".sans/skills/check-first/SKILL.md"
    before = active.read_bytes() if active.is_file() else None
    result = maintain(action=action)
    assert result["action"] == "staged" and not result["promotion"]["promoted"]
    assert Path(result["candidate_file"]).is_file()
    candidate_body = skills.parse_frontmatter(Path(result["candidate_file"]).read_text()).body
    assert "An empty input check failed." not in candidate_body
    assert result["evidence"] == "An empty input check failed."
    assert (active.read_bytes() if active.is_file() else None) == before
    found = skills.discover_skills()
    assert len(found) == (1 if before else 0)
    if before:
        assert "Validate empty input" not in found[0].prompt_template
        assert "Preserve input." in Path(result["candidate_file"]).read_text(encoding="utf-8")
    assert promote_staged_skill(result)["action"] == "staged"


@pytest.mark.parametrize("action", ["add", "merge"])
def test_user_preference_keeps_existing_maintenance(bank, action):
    if action == "merge":
        active_skill(bank)
    result = maintain(source="user", action=action)
    assert result["action"] == action
    assert "Validate empty input" in skills.get_skill_by_name("check-first").prompt_template


def test_failed_local_edit_does_not_stage_baseline_as_candidate(bank, monkeypatch):
    from agents import skill_evolution

    active = active_skill(bank)
    before = active.read_bytes()
    monkeypatch.setattr(skill_evolution, "evolve_skill_file", lambda **kwargs: {"ok": False, "error": "edit refused"})
    result = maintain(action="merge")
    assert not result["ok"] and result["action"] == "staging_failed" and result["error"] == "edit refused"
    assert active.read_bytes() == before
    assert not list((bank / ".sans/skill-evolution/candidates").glob("*/stage.json"))


@pytest.mark.parametrize("tamper", ["candidate", "receipt"])
def test_repeated_candidate_keeps_original_stage_hash(bank, tamper):
    active = active_skill(bank)
    before = active.read_bytes()
    staged = maintain(action="merge")
    assert maintain(action="merge")["candidate_sha256"] == staged["candidate_sha256"]
    path = Path(staged["candidate_file"])
    (path if tamper == "candidate" else path.with_name("stage.json")).write_text("tampered", encoding="utf-8")
    result = maintain(action="merge")
    assert not result["ok"] and result["action"] == "staging_failed"
    assert active.read_bytes() == before


@pytest.mark.parametrize("field,value", [("action", "merge"), ("proposed_action", "add"),
                                         ("candidate_file", "other/SKILL.md"), ("evidence_ids", []), ("source", "user")])
def test_cached_stage_metadata_cannot_claim_a_different_action_or_source(bank, field, value):
    active = active_skill(bank)
    before = active.read_bytes()
    staged = maintain(action="merge")
    path = Path(staged["candidate_file"]).with_name("stage.json")
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored[field] = value
    path.write_text(json.dumps(stored), encoding="utf-8")
    result = maintain(action="merge")
    assert not result["ok"] and result["action"] == "staging_failed"
    assert active.read_bytes() == before


@pytest.mark.parametrize("action", ["add", "merge"])
def test_matched_task_oracle_promotes_exact_staged_file(bank, action):
    if action == "merge":
        active_skill(bank)
    staged = maintain(action=action)
    result = promote_staged_skill(staged, receipt(bank, staged))
    assert result["action"] == action and result["promotion"]["promoted"]
    assert Path(result["active_file"]).read_bytes() == Path(result["candidate_file"]).read_bytes()
    assert "Validate empty input" in skills.get_skill_by_name("check-first").prompt_template


@pytest.mark.parametrize("complete", [True, False])
@pytest.mark.parametrize("entrypoint", ["ingest", "agent"])
def test_online_ingest_uses_only_explicit_task_evaluator(bank, complete, entrypoint, monkeypatch):
    from agents.online_skill_evolution import online_ingest

    calls = []

    async def model(system, _payload):
        if "Extractor" in system:
            return json.dumps({"skills": [{"name": "check-first", "description": "Validate batch options",
                               "instructions": "Validate empty input before returning.", "source": "execution",
                               "evidence_ids": ["trusted-check"]}]})
        return json.dumps({"action": "add"})

    async def evaluate(staged):
        calls.append(staged["candidate_file"])
        evaluation = receipt(bank, staged)
        evaluation["completed"] = complete
        return evaluation

    window = {"messages": [{"role": "user", "content": "Fix batch validation."}],
              "execution_evidence": [{"id": "trusted-check"}]}
    if entrypoint == "agent":
        from agents.agent import Agent
        monkeypatch.setenv("SANS_AUTO_SKILL_EVOLUTION", "1")
        agent = Agent(api_key="test", custom_system_prompt="test", custom_tools=[],
                      permission_mode="bypassPermissions", is_sub_agent=True,
                      skill_task_evaluator=evaluate)
        monkeypatch.setattr(agent, "_build_side_query", lambda **kwargs: model)
        async def run():
            try:
                return await agent._run_online_skill_evolution(window)
            finally:
                await agent.shutdown()
        result = asyncio.run(run())
    else:
        result = asyncio.run(online_ingest(**window, side_query=model, task_evaluator=evaluate))
    assert len(calls) == 1
    assert result["action"] == ("add" if complete else "staged")
    assert bool(skills.get_skill_by_name("check-first")) is complete


def test_candidate_status_uses_persisted_hashes_and_reports_damaged_receipt(bank):
    from agents.online_skill_evolution import list_staged_skills
    staged = maintain()
    assert list_staged_skills()[0]["status"] == "pending_evaluation"
    promote_staged_skill(staged, receipt(bank, staged))
    assert list_staged_skills()[0]["status"] == "active"
    Path(staged["active_file"]).write_text("changed", encoding="utf-8")
    assert list_staged_skills()[0]["status"] == "baseline_changed"
    Path(staged["candidate_file"]).write_text("changed candidate", encoding="utf-8")
    assert list_staged_skills()[0]["status"] == "candidate_changed"
    Path(staged["candidate_file"]).with_name("stage.json").write_text("{", encoding="utf-8")
    assert list_staged_skills()[0]["status"] == "invalid_receipt"


@pytest.mark.parametrize("source", ["user", "execution", None])
@pytest.mark.parametrize("action", ["add", "merge"])
def test_automatic_candidate_cannot_authorize_activation_with_its_source_label(bank, source, action):
    from agents.online_skill_evolution import online_ingest

    active = active_skill(bank) if action == "merge" else bank / ".sans/skills/check-first/SKILL.md"
    before = active.read_bytes() if active.is_file() else None
    calls = []

    async def model(system, _payload):
        calls.append(system)
        if "Extractor" in system:
            candidate = {"name": "check-first", "description": "Validate batch options",
                         "instructions": "Apply the unverified guessed fix on every task.",
                         "evidence": "User said preserve the input, so I claim this fix is a user preference.",
                         "evidence_ids": ["trusted-check"]}
            if source is not None:
                candidate["source"] = source
            return json.dumps({"skills": [candidate]})
        return json.dumps({"action": action, "target_skill": "check-first"})

    result = asyncio.run(online_ingest(messages=[{"role": "user", "content": "Preserve input while fixing this task."}],
                                      side_query=model, execution_evidence=[{"id": "trusted-check"}]))
    assert (active.read_bytes() if active.is_file() else None) == before
    if source is None:
        assert result["action"] == "none" and len(calls) == 1
    else:
        assert result["action"] == "staged" and not result["promotion"]["promoted"]
        assert result["source"] == source and source in result["promotion"]["reason"]
        assert len(calls) == 2


@pytest.mark.parametrize("failure", ["text_judge", "incomplete", "partial_tasks", "different_evaluation", "unknown_generation",
                                     "oracle_changed", "protocol_changed", "training_overlap", "candidate_mismatch", "equal_score",
                                     "new_hard_failure", "candidate_changed", "baseline_changed", "boolean_score", "boolean_count",
                                     "negative_count", "malformed_protocol", "malformed_summary"])
def test_unverified_or_regressing_receipt_cannot_activate(bank, failure):
    active = active_skill(bank)
    staged = maintain(action="merge")
    evaluation = copy.deepcopy(receipt(bank, staged))
    if failure == "text_judge":
        evaluation["oracle"]["kind"] = "llm_text_judge"
    elif failure == "incomplete":
        evaluation["completed"] = False
    elif failure == "partial_tasks":
        evaluation["candidate"]["task_ids"] = ["heldout-a"]
    elif failure == "different_evaluation":
        evaluation["candidate"]["evaluation_id"] = "other-run"
    elif failure == "unknown_generation":
        evaluation["candidate"]["generation_failures"] = 1
    elif failure in {"oracle_changed", "protocol_changed"}:
        Path(evaluation[failure.split("_")[0]]["path"]).write_text("changed", encoding="utf-8")
    elif failure in {"training_overlap", "candidate_mismatch"}:
        path = Path(evaluation["protocol"]["path"])
        protocol = json.loads(path.read_text(encoding="utf-8"))
        protocol["training_task_ids" if failure == "training_overlap" else "candidate_sha256"] = ["heldout-a"] if failure == "training_overlap" else "other-snapshot"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        evaluation["protocol"]["sha256"] = digest(path)
    elif failure == "equal_score":
        evaluation["candidate"]["average_score"] = 0.5
    elif failure == "new_hard_failure":
        evaluation["candidate"].update(average_score=0.9, hard_failures=1, hard_failure_keys=[["heldout-b", "preserve-input"]])
    elif failure == "candidate_changed":
        Path(staged["candidate_file"]).write_text("Changed draft", encoding="utf-8")
    elif failure == "baseline_changed":
        active.write_text("User changed active Skill", encoding="utf-8")
    elif failure == "boolean_score":
        evaluation["candidate"]["average_score"] = True
    elif failure == "boolean_count":
        evaluation["candidate"]["generation_failures"] = False
    elif failure == "negative_count":
        evaluation["candidate"]["hard_failures"] = -1
    elif failure == "malformed_protocol":
        path = Path(evaluation["protocol"]["path"])
        path.write_text("[invalid", encoding="utf-8")
        evaluation["protocol"]["sha256"] = digest(path)
    elif failure == "malformed_summary":
        evaluation["candidate"] = ["invalid"]
    before = active.read_bytes()
    result = promote_staged_skill(staged, evaluation)
    assert result["action"] == "staged" and not result["promotion"]["promoted"]
    assert active.read_bytes() == before
