import json

from agents import skills
from agents.frontmatter import parse_frontmatter
from agents.skill_evolution import create_skill_file, evolve_skill_file


def test_procedure_updates_keep_evidence_in_history_and_survive_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANS_SKILL_EVOLUTION_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(skills.Path, "home", lambda: tmp_path / "home")
    evidence = "Creation incident " + "x" * 1300 + " source-tail"
    lesson = "Observed failed then passed in episode-123 " + "x" * 1300 + " lesson-tail"
    rationale = "Observed matching checks " + "x" * 1300 + " reason-tail"
    create_skill_file(name="verify", description="Verify repository changes",
        instructions="Run checks.\nPreserve supplied fixtures.", evidence=evidence)
    path = tmp_path / ".sans/skills/verify/SKILL.md"
    original = path.read_text(encoding="utf-8")
    assert "Creation incident" not in original
    result = evolve_skill_file(skill_name="verify", active_dir=str(path.parent),
        lesson=lesson, rationale=rationale,
        instruction_updates=[{"old": "Run checks.", "new": "After editing, run the original suite to verify the change."}])
    assert result["ok"] and result["version"] == "0.1.1"
    skills.reset_skill_cache()
    try:
        reloaded = skills.get_skill_by_name("verify")
        assert "After editing" in reloaded.prompt_template
        assert "Preserve supplied fixtures." in reloaded.prompt_template
        assert "episode-123" not in reloaded.prompt_template
        assert "Evolution Notes" not in reloaded.prompt_template
        history = [json.loads(row) for row in (tmp_path / "history/history/verify.jsonl").read_text().splitlines()]
        assert history[0]["content"] == original
        assert history[0]["lesson"] == lesson and history[0]["rationale"] == rationale
        usage = [json.loads(row) for row in (tmp_path / "history/usage.jsonl").read_text().splitlines()]
        assert usage[0]["evidence"] == evidence
    finally:
        skills.reset_skill_cache()


def test_manual_rule_is_idempotent_and_full_replacement_omits_incident(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANS_SKILL_EVOLUTION_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(skills.Path, "home", lambda: tmp_path / "home")
    create_skill_file(name="verify", description="Verify changes", instructions="Preserve input.")
    path = tmp_path / ".sans/skills/verify/SKILL.md"
    kwargs = dict(skill_name="verify", active_dir=str(path.parent), lesson="Check original constraints after editing.")
    evolve_skill_file(**kwargs)
    first = path.read_bytes()
    assert evolve_skill_file(**kwargs)["event"] == "unchanged"
    assert path.read_bytes() == first
    evolve_skill_file(**kwargs, instructions="# Verify\n\nRun the original suite.", rationale="incident context")
    body = parse_frontmatter(path.read_text()).body
    assert "Run the original suite." in body
    assert "incident context" not in body and kwargs["lesson"] not in body


def test_unchanged_procedure_still_accepts_metadata_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANS_SKILL_EVOLUTION_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(skills.Path, "home", lambda: tmp_path / "home")
    create_skill_file(name="verify", description="Verify changes", instructions="# Verify\nRun checks.")
    path = tmp_path / ".sans/skills/verify/SKILL.md"
    before = parse_frontmatter(path.read_text())
    result = evolve_skill_file(skill_name="verify", active_dir=str(path.parent), lesson="Trigger clarified",
        instruction_updates=[], when_to_use="After repository edits")
    after = parse_frontmatter(path.read_text())
    assert result["event"] == "evolve" and after.body == before.body
    assert after.meta["when-to-use"] == "After repository edits"
    frozen = path.read_bytes()
    repeated = evolve_skill_file(skill_name="verify", active_dir=str(path.parent), lesson="Trigger clarified",
        instruction_updates=[], when_to_use="After  repository edits", description="Verify changes")
    assert repeated["event"] == "unchanged" and path.read_bytes() == frozen
    replaced = evolve_skill_file(skill_name="verify", active_dir=str(path.parent), lesson="Trigger clarified",
        instructions=after.body, when_to_use="After repository edits", description="Verify changes")
    assert replaced["event"] == "unchanged" and path.read_bytes() == frozen
