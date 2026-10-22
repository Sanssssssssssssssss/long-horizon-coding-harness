"""Evaluated Skill versions can be restored without overwriting newer work."""
import asyncio
import json
from pathlib import Path

import pytest

from test_skill_promotion import bank, active_skill, maintain, receipt
from agents import online_skill_evolution as evolution


@pytest.mark.parametrize("action", ["add", "merge"])
def test_activation_retains_evaluation_and_exact_reversible_snapshot(bank, action):
    active = active_skill(bank) if action == "merge" else bank / ".sans/skills/check-first/SKILL.md"
    before = active.read_bytes() if active.exists() else None
    staged = maintain(action=action)
    evaluated = receipt(bank, staged)
    result = evolution.promote_staged_skill(staged, evaluated)
    assert result.get("activation_id"), "Successful activation must retain a recoverable receipt"
    from agents.skill_activation import rollback_skill_activation, list_skill_activations
    record = json.loads(Path(result["activation_receipt"]).read_text(encoding="utf-8"))
    assert record["evaluation"] == evaluated
    assert list_skill_activations()[0]["status"] == "active"
    restored = rollback_skill_activation(result["activation_id"])
    assert restored["ok"] and restored["rollback_of"] == result["activation_id"]
    assert (active.read_bytes() if active.exists() else None) == before
    assert not rollback_skill_activation(result["activation_id"])["ok"]


def test_rollback_refuses_a_newer_manual_edit(bank):
    active = active_skill(bank)
    staged = maintain(action="merge")
    result = evolution.promote_staged_skill(staged, receipt(bank, staged))
    assert result.get("activation_id"), "Activation identity missing"
    from agents.skill_activation import rollback_skill_activation
    active.write_text("A newer manual procedure", encoding="utf-8")
    assert not rollback_skill_activation(result["activation_id"])["ok"]
    assert active.read_text(encoding="utf-8") == "A newer manual procedure"


def test_rollback_rejects_corrupt_snapshot_and_foreign_workspace(bank, monkeypatch):
    active = active_skill(bank)
    staged = maintain(action="merge")
    result = evolution.promote_staged_skill(staged, receipt(bank, staged))
    assert result.get("activation_id"), "Activation identity missing"
    from agents.skill_activation import rollback_skill_activation
    current = active.read_bytes()
    path = Path(result["activation_receipt"])
    record = json.loads(path.read_text(encoding="utf-8"))
    other = bank / "other-project"
    other.mkdir()
    monkeypatch.chdir(other)
    assert not rollback_skill_activation(result["activation_id"])["ok"]
    monkeypatch.chdir(bank)
    record["before_base64"] = "Y29ycnVwdA=="
    path.write_text(json.dumps(record), encoding="utf-8")
    assert not rollback_skill_activation(result["activation_id"])["ok"]
    assert active.read_bytes() == current


def test_promotion_and_manual_updates_respect_the_same_lock(bank):
    from agents.skill_activation import skill_file_lock
    from agents.skill_evolution import evolve_skill_file
    active = active_skill(bank)
    before = active.read_bytes()
    staged = maintain(action="merge")
    evaluated = receipt(bank, staged)
    with skill_file_lock(active):
        result = evolution.promote_staged_skill(staged, evaluated)
        assert result["action"] == "staged" and "another operation" in result["promotion"]["reason"]
        with pytest.raises(RuntimeError, match="another operation"):
            evolve_skill_file(skill_name="check-first", lesson="Keep newer input.", instructions="Keep newer input.")
    assert active.read_bytes() == before
    assert evolution.promote_staged_skill(staged, evaluated)["promotion"]["promoted"]


def test_recovery_record_failure_prevents_activation(bank, monkeypatch):
    import agents.skill_activation as activation
    active = active_skill(bank)
    before = active.read_bytes()
    staged = maintain(action="merge")
    evaluated = receipt(bank, staged)
    def fail(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(activation, "atomic_write_json", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        evolution.promote_staged_skill(staged, evaluated)
    assert active.read_bytes() == before


def test_manual_update_rejects_a_snapshot_changed_during_merge(bank, monkeypatch):
    from agents import skill_evolution
    active = active_skill(bank)
    original_update = skill_evolution._apply_instruction_updates
    def concurrent_update(body, updates):
        active.write_text("A simultaneous edit", encoding="utf-8")
        return original_update(body, updates)
    monkeypatch.setattr(skill_evolution, "_apply_instruction_updates", concurrent_update)
    with pytest.raises(ValueError, match="changed since"):
        skill_evolution.evolve_skill_file(skill_name="check-first", lesson="Updated", instruction_updates=[{"old": "", "new": "Validate output."}])
    assert active.read_text(encoding="utf-8") == "A simultaneous edit"


@pytest.mark.parametrize("foreign", [True, False])
def test_activation_requires_original_workspace_even_with_shared_store(bank, monkeypatch, foreign):
    active = active_skill(bank)
    before = active.read_bytes()
    staged = maintain(action="merge")
    evaluated = receipt(bank, staged)
    if foreign:
        other = bank / "other-project"
        other.mkdir()
        monkeypatch.chdir(other)
    else:
        staged.pop("workspace")
    result = evolution.promote_staged_skill(staged, evaluated)
    assert result["action"] == "staged" and "workspace" in result["promotion"]["reason"]
    assert active.read_bytes() == before


@pytest.mark.parametrize("planning", [False, True])
def test_cli_applies_trusted_evaluation_and_can_restore_without_a_model_call(bank, monkeypatch, planning):
    from agents import main as cli
    from agents.skill_activation import list_skill_activations
    active = active_skill(bank)
    before = active.read_bytes()
    staged = maintain(action="merge")
    evaluation_file = bank / "trusted evaluation.json"
    evaluation_file.write_text(json.dumps(receipt(bank, staged)), encoding="utf-8")
    candidate_id = Path(staged["candidate_file"]).parent.name
    agent = cli.Agent(api_base="http://unused.invalid/v1", api_key="test", is_sub_agent=True)
    agent.permission_mode = "plan" if planning else "acceptEdits"
    commands = iter(["promote", "history", "rollback", "learning", "exit"])
    def read_input(*_args):
        command = next(commands)
        if command == "promote":
            return f'/skill-promote {candidate_id} "{evaluation_file}"'
        if command == "history":
            assert (active.read_bytes() != before) is (not planning)
            return "/skill-history"
        if command == "rollback":
            rows = list_skill_activations()
            return f"/skill-rollback {rows[0]['id']}" if rows else "/skill-history"
        return "/skill-learning" if command == "learning" else "/exit"
    async def forbidden(*_args, **_kwargs):
        pytest.fail("Offline Skill commands must not call a model")
    monkeypatch.setattr("builtins.input", read_input)
    monkeypatch.setattr(agent, "chat", forbidden)
    monkeypatch.setattr(agent, "_side_query_once", forbidden)
    asyncio.run(cli.run_cli(agent, None))
    assert active.read_bytes() == before
