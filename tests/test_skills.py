from __future__ import annotations

import hashlib

import pytest

import agents.skills as skills_module


def test_retrieval_discloses_metadata_before_skill_body(monkeypatch) -> None:
    body = "Run the focused test, inspect the diff, then stop."
    skill = skills_module.SkillDefinition(
        name="repo-workflow",
        description="Repository execution workflow",
        when_to_use="When fixing a repository task",
        prompt_template=body,
    )
    monkeypatch.setattr(skills_module, "discover_skills", lambda: [skill])
    monkeypatch.setattr(skills_module, "retrieve_relevant_skills", lambda _query, limit=3: [{
        "score": 1.0,
        "name": "repo-workflow",
        "description": "Repository execution workflow",
        "when_to_use": "When fixing a repository task",
        "source": "project",
    }])

    context, reference = skills_module.format_retrieved_skill_context("fix the repository")

    assert "repo-workflow" in context
    assert "Repository execution workflow" in context
    assert body not in context
    assert reference["name"] == "repo-workflow"


def _discovered_skill(tmp_path, monkeypatch, *, body="Replay the focused test.", context="inline"):
    directory = tmp_path / "repo-workflow"
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text(
        f"---\nname: repo-workflow\ndescription: Repository execution workflow\n"
        f"context: {context}\nallowed-tools: shell,edit\n---\n{body}\n",
        encoding="utf-8",
    )
    skill = skills_module._parse_skill_file(path, "project", str(directory))
    monkeypatch.setattr(skills_module, "discover_skills", lambda: [skill])
    reference = skills_module.retrieve_relevant_skills("repo-workflow", limit=1)[0]
    return path, skill, reference


def test_selected_body_is_bound_to_disk_version_without_executing_skill(tmp_path, monkeypatch):
    body = "Replay the focused test at ${CLAUDE_SKILL_DIR}; ignore $ARGUMENTS."
    path, skill, reference = _discovered_skill(tmp_path, monkeypatch, body=body)
    monkeypatch.setattr(skills_module, "record_skill_invocation", lambda **_: pytest.fail("executed"))

    context, receipt = skills_module.load_retrieved_skill_body(reference)

    assert receipt["status"] == "loaded"
    assert receipt["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    resolved = skills_module.resolve_skill_prompt(skill, "")
    assert resolved in context
    assert receipt["body_sha256"] == hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    assert receipt["context_bytes"] == len(context.encode("utf-8"))
    assert "User requirements and runtime evidence take precedence" in context
    assert "allowed_tools" not in receipt


@pytest.mark.parametrize("replacement,status", [(None, "source_unavailable"), ("changed", "source_changed")])
def test_body_loader_rejects_missing_or_replaced_source_without_cached_fallback(
    tmp_path, monkeypatch, replacement, status,
):
    path, _, reference = _discovered_skill(tmp_path, monkeypatch)
    if replacement is None:
        path.unlink()
    else:
        path.write_text(replacement, encoding="utf-8")

    context, receipt = skills_module.load_retrieved_skill_body(reference)

    assert context == ""
    assert receipt["status"] == status


def test_body_budget_counts_entire_utf8_context_and_refuses_partial_procedure(tmp_path, monkeypatch):
    _, _, reference = _discovered_skill(tmp_path, monkeypatch, body="验证约束和失败证据。" * 20)
    context, receipt = skills_module.load_retrieved_skill_body(reference)
    assert receipt["status"] == "loaded"

    exact, exact_receipt = skills_module.load_retrieved_skill_body(reference, max_bytes=receipt["context_bytes"])
    assert exact == context
    assert exact_receipt["status"] == "loaded"
    rejected, rejected_receipt = skills_module.load_retrieved_skill_body(reference, max_bytes=len(context))
    assert rejected == ""
    assert rejected_receipt["status"] == "context_too_large"


def test_fork_skill_requires_explicit_activation(tmp_path, monkeypatch):
    _, _, reference = _discovered_skill(tmp_path, monkeypatch, context="fork")

    context, receipt = skills_module.load_retrieved_skill_body(reference)

    assert context == ""
    assert receipt["status"] == "requires_explicit_activation"


@pytest.mark.parametrize("changed_field,status", [("source_sha256", "unversioned_source"), ("skill_dir", "source_changed")])
def test_loader_requires_discovery_binding_instead_of_trusting_reference_paths(
    tmp_path, monkeypatch, changed_field, status,
):
    _, _, reference = _discovered_skill(tmp_path, monkeypatch)
    reference.pop(changed_field)

    context, receipt = skills_module.load_retrieved_skill_body(reference)

    assert context == ""
    assert receipt["status"] == status


@pytest.mark.parametrize("replacement", [None, "replaced source"])
def test_explicit_activation_refuses_stale_source_before_recording_invocation(tmp_path, monkeypatch, replacement):
    path, skill, _ = _discovered_skill(tmp_path, monkeypatch)
    monkeypatch.setattr(skills_module, "record_skill_invocation", lambda **_: pytest.fail("invoked stale source"))
    if replacement is None:
        path.unlink()
    else:
        path.write_text(replacement, encoding="utf-8")
    result = skills_module.execute_skill(skill.name, "")
    assert "error" in result and "prompt" not in result


@pytest.mark.parametrize("context", ["inline", "fork"])
def test_explicit_activation_binds_literal_arguments_and_full_body_budget(tmp_path, monkeypatch, context):
    _, skill, _ = _discovered_skill(tmp_path, monkeypatch, body="Inspect $ARGUMENTS", context=context)
    invocations = []
    monkeypatch.setattr(skills_module, "record_skill_invocation", lambda **kw: invocations.append(kw))
    args = r"E:\new\task\1.txt"
    result = skills_module.execute_skill(skill.name, args)
    assert args in result["prompt"] and result["prompt"] in result["loaded_context"]
    assert result["receipt"]["body_sha256"] == hashlib.sha256(result["prompt"].encode()).hexdigest()
    refused = skills_module.execute_skill(skill.name, "x" * 12000)
    assert refused["receipt"]["status"] == "context_too_large"
    assert len(invocations) == 1
