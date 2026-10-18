from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from agents.telemetry import TraceRecorder
from benchmarks.live import LiveAgentAdapter, TraceAcceptanceError, validate_trace
from benchmarks.full_eval import run_full_eval
from benchmarks.smoke import _parser, main as smoke_main, run_smoke


EXACT_USAGE = {
    "input_tokens": 3,
    "output_tokens": 2,
    "cached_input_tokens": None,
    "cache_miss_tokens": None,
    "cache_creation_tokens": None,
    "reasoning_tokens": None,
    "total_tokens": 5,
    "provider_reported_cost": None,
}


def recorder(root: Path, attempt: str = "attempt-0001") -> TraceRecorder:
    return TraceRecorder(
        root / attempt,
        blob_dir=root / "blobs",
        run_id="run-1",
        task_id="task-1",
        attempt_id=attempt,
    )


def test_live_adapter_builds_a_fresh_agent_and_keeps_key_out_of_tool_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "test-secret-that-must-not-leak"
    monkeypatch.setenv("ACCIDENTAL_SECRET_ALIAS", secret)
    monkeypatch.setenv("ACCIDENTAL_AUTH_HEADER", f"Bearer {secret}")
    made: list[object] = []

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["api_key"] == secret
            assert kwargs["api_base"] == "https://example.invalid/v1"
            assert kwargs["model"] == "test-model"
            assert kwargs["permission_mode"] == "bypassPermissions"
            assert kwargs["model_purpose"] == "main"
            assert kwargs["max_output_tokens"] == 2048
            assert kwargs["max_input_tokens_per_call"] == 30_000
            assert kwargs["openai_extra_body"] == {"chat_template_kwargs": {"thinking": False}}
            assert kwargs["openai_stream_include_usage"] is True
            assert kwargs["custom_tools"] == [{"name": "read_file"}]
            assert kwargs["skills_enabled"] is False
            assert kwargs["is_sub_agent"] is True
            assert kwargs["repair_task"] is None
            self.trace = kwargs["trace"]
            made.append(self)

        async def run_once(self, prompt: str) -> dict[str, object]:
            assert all(secret not in value for value in os.environ.values())
            assert "Fix the fixture" in prompt
            with self.trace.span(
                "model.call",
                component="model",
                purpose="main",
                attributes={"provider": "openai", "model": "test-model"},
                payload={"messages": [{"role": "user", "content": prompt}]},
            ) as span:
                span.finish(
                    metrics={"provider_usage": EXACT_USAGE},
                    payload={"text": "done"},
                )
            return {"text": "done", "tokens": {"input": 3, "output": 2}}

        async def shutdown(self) -> None:
            assert all(secret not in value for value in os.environ.values())

    adapter = LiveAgentAdapter(
        api_base="https://example.invalid/v1",
        api_key=secret,
        model="test-model",
        max_output_tokens=2048,
        openai_extra_body={"chat_template_kwargs": {"thinking": False}},
        custom_tools=({"name": "read_file"},),
        agent_factory=FakeAgent,
    )
    task = {
        "task_id": "task-1",
        "problem_statement": "Fix the fixture.",
        "raw_problem_statement": "Raw issue exactly.",
        "platform": "local",
        "timeout_seconds": 30,
        "network_policy": "enabled",
    }

    for index in range(2):
        result = asyncio.run(adapter(
            task=task,
            workdir=tmp_path,
            trace=recorder(tmp_path, f"attempt-{index}"),
            env=dict(os.environ),
        ))
        assert result["text"] == "done"

    assert len(made) == 2
    assert made[0] is not made[1]
    assert os.environ["ACCIDENTAL_SECRET_ALIAS"] == secret
    assert os.environ["ACCIDENTAL_AUTH_HEADER"] == f"Bearer {secret}"
    assert secret not in repr(adapter)


def test_live_adapter_chat_does_not_load_interactive_mcp_or_persistence(
    tmp_path: Path,
) -> None:
    from agents.agent import Agent

    mcp_loads = 0

    class TrapMcp:
        async def load_and_connect(self) -> None:
            nonlocal mcp_loads
            mcp_loads += 1

        def get_tool_definitions(self) -> list[dict[str, object]]:
            return [{"name": "unexpected_mcp"}]

        async def disconnect_all(self) -> None:
            return None

    class ChatAgent(Agent):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self._mcp_manager = TrapMcp()

        async def _probe_repo_state(self) -> str:
            return "clean"

        async def _chat_openai(self, _: str) -> None:
            with self.trace.span(
                "model.call", component="model", purpose="main"
            ) as span:
                span.finish(metrics={"provider_usage": EXACT_USAGE}, payload={"text": "done"})
            self._emit_text("done")

        def _auto_save(self) -> None:
            raise AssertionError("benchmark solver must not persist an interactive session")

    adapter = LiveAgentAdapter(
        api_base="https://example.invalid/v1",
        api_key="secret",
        model="test-model",
        agent_factory=ChatAgent,
    )

    result = asyncio.run(adapter(
        task={"task_id": "task-1", "problem_statement": "Fix it."},
        workdir=tmp_path,
        trace=recorder(tmp_path),
        env={},
    ))

    assert result["text"] == "done"
    assert mcp_loads == 0


def test_live_adapter_rejects_a_run_without_a_main_model_terminal_span(tmp_path: Path) -> None:
    class SilentAgent:
        def __init__(self, **_: object) -> None:
            pass

        async def run_once(self, _: str) -> dict[str, str]:
            return {"text": "untraced"}

    adapter = LiveAgentAdapter(
        api_base="https://example.invalid/v1",
        api_key="secret",
        model="test-model",
        custom_tools=(),
        agent_factory=SilentAgent,
    )

    with pytest.raises(TraceAcceptanceError, match="trace is empty|main model terminal"):
        asyncio.run(adapter(
            task={"task_id": "task-1", "problem_statement": "Do it."},
            workdir=tmp_path,
            trace=recorder(tmp_path),
            env={},
        ))


@pytest.mark.parametrize(
    ("status", "usage", "message"),
    [
        ("error", EXACT_USAGE, "successful main model"),
        ("ok", None, "exact provider usage"),
        ("ok", {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}, "all fields"),
        ("ok", {**EXACT_USAGE, "output_tokens": None, "total_tokens": None}, "exact provider usage"),
        ("ok", {**EXACT_USAGE, "total_tokens": 6}, "not self-consistent"),
        ("ok", {**EXACT_USAGE, "reasoning_tokens": "unknown"}, "numbers or null"),
    ],
)
def test_validate_trace_requires_successful_main_model_with_exact_usage(
    tmp_path: Path, status: str, usage: dict[str, object] | None, message: str
) -> None:
    trace = recorder(tmp_path)
    with trace.span("model.call", component="model", purpose="main") as span:
        span.finish(
            status=status,
            metrics={} if usage is None else {"provider_usage": usage},
            payload={"text": "result"},
        )

    with pytest.raises(TraceAcceptanceError, match=message):
        validate_trace(trace)


def test_validate_trace_accepts_provider_total_that_adds_hidden_reasoning(tmp_path: Path) -> None:
    trace = recorder(tmp_path)
    usage = {
        **EXACT_USAGE,
        "reasoning_tokens": 8,
        "total_tokens": 13,
    }
    with trace.span("model.call", component="model", purpose="main") as span:
        span.finish(metrics={"provider_usage": usage}, payload={"text": "result"})

    validate_trace(trace)


def test_validate_trace_rejects_missing_or_corrupt_payload_blobs(tmp_path: Path) -> None:
    trace = recorder(tmp_path)
    with trace.span("model.call", component="model", purpose="main", payload={"input": "x"}) as span:
        span.finish(metrics={"provider_usage": {
            **EXACT_USAGE, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
        }})
    events = validate_trace(trace)
    assert any(event["event_type"] == "model.call.end" for event in events)

    payload_event = next(event for event in events if event["payload_ref"])
    blob = trace.blob_dir / payload_event["payload_ref"]["sha256"]
    original = blob.read_bytes()
    blob.unlink()
    with pytest.raises(TraceAcceptanceError, match="payload blob is missing"):
        validate_trace(trace)

    blob.write_bytes(original)
    blob.write_bytes(b"x" * payload_event["payload_ref"]["bytes"])
    with pytest.raises(TraceAcceptanceError, match="payload blob hash"):
        validate_trace(trace)

    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    assert digest != payload_event["payload_ref"]["sha256"]


def test_offline_smoke_exercises_runner_without_network(tmp_path: Path) -> None:
    summary = asyncio.run(run_smoke(runs_root=tmp_path / "runs"))

    assert summary["counts"]["tasks"] == 1
    assert summary["counts"]["completed"] == 1
    assert summary["counts"]["resolved"] == 1
    assert summary["counts"]["unknown_resolution"] == 0
    assert summary["counts"]["infra_failures"] == 0
    assert summary["counts"].get("task_failures", 0) == 0
    assert summary["totals"]["model_calls"] == 1
    assert summary["totals"]["tokens"] == 2
    attempt = (
        tmp_path / "runs" / "infra-smoke" / "smoke-1" /
        "attempts" / "replace-answer" / "attempt-0001"
    )
    assert (attempt / "final.patch").read_bytes()
    assert (attempt / "result.json").is_file()


def test_live_smoke_requires_explicit_persistent_runs_root() -> None:
    with pytest.raises(SystemExit, match="--runs-root"):
        smoke_main([
            "--live",
            "--api-base", "https://example.invalid/v1",
            "--model", "test-model",
            "--provider-revision", "test-rev",
        ])


def test_stream_usage_defaults_on_and_has_explicit_off_switch() -> None:
    assert _parser().parse_args([]).stream_usage is True
    assert _parser().parse_args(["--no-stream-usage"]).stream_usage is False


@pytest.mark.parametrize(
    "extra_body",
    [
        {"Authorization": "value"},
        {"nested": {"api_token": "value"}},
        {"nested": [{"clientSecret": "value"}]},
        {"password_hint": "value"},
        {"credentialMode": "value"},
        {"custom_headers": {"x": "value"}},
    ],
)
def test_extra_body_rejects_secret_bearing_key_names(extra_body: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="secret-bearing key"):
        LiveAgentAdapter(
            api_base="https://example.invalid/v1",
            api_key="secret",
            model="test-model",
            custom_tools=(),
            openai_extra_body=extra_body,
        )


def test_b0_uses_the_narrow_solver_defaults() -> None:
    adapter = LiveAgentAdapter(
        api_base="https://example.invalid/v1",
        api_key="secret",
        model="test-model",
    )

    assert {tool["name"] for tool in adapter._resolved_tools()} == {
        "read_file", "write_file", "edit_file", "list_files", "grep_search",
        "run_shell", "update_work_state", "read_observation",
    }
    assert "repository-task solver" in adapter._resolved_system_prompt()


def test_live_config_is_non_secret_complete_and_immutable_for_a_run_id(tmp_path: Path) -> None:
    secret = "manifest-secret"
    tools = ({
        "name": "read_file",
        "description": "read",
        "input_schema": {"type": "object", "properties": {}},
    },)
    adapter = LiveAgentAdapter(
        api_base="HTTPS://Example.Invalid:443/v1/",
        api_key=secret,
        model="test-model",
        provider_revision="provider-2026-08-23",
        max_turns=4,
        max_model_calls=40,
        max_tool_calls=40,
        model_call_timeout_seconds=180,
        max_output_tokens=2048,
        max_input_tokens=30_000,
        openai_extra_body={"thinking": {"type": "disabled"}},
        openai_stream_include_usage=False,
        custom_tools=tools,
        custom_system_prompt="fixed B0 prompt",
        agent_factory=lambda **_: None,
    )
    manifest = adapter.benchmark_manifest()
    config = adapter.benchmark_config()
    serialized = json.dumps({"manifest": manifest, "config": config}, sort_keys=True)

    assert secret not in serialized
    assert manifest["api_base_identifier"] == "https://example.invalid"
    assert manifest["model_identifier"] == "test-model"
    assert manifest["provider_revision"] == "provider-2026-08-23"
    assert manifest["max_model_calls"] == 40
    assert manifest["max_tool_calls"] == 40
    assert manifest["model_call_timeout_seconds"] == 180
    assert len(manifest["api_base_sha256"]) == 64
    assert len(manifest["tool_schema_hash"]) == 64
    assert len(manifest["system_prompt_hash"]) == 64
    assert len(manifest["openai_extra_body_sha256"]) == 64
    expected_extra_hash = hashlib.sha256(
        b'{"thinking":{"type":"disabled"}}'
    ).hexdigest()
    assert manifest["openai_extra_body_sha256"] == expected_extra_hash
    live = config["live_agent"]
    assert live == {
        "api_base_identifier": "https://example.invalid",
        "api_base_sha256": manifest["api_base_sha256"],
        "custom_tools": True,
        "max_cost_usd": None,
        "max_input_tokens": 30_000,
        "max_model_calls": 40,
        "max_output_tokens": 2048,
        "max_tool_calls": 40,
        "max_turns": 4,
        "model_call_timeout_seconds": 180,
        "model": "test-model",
        "openai_extra_body_keys": ["thinking"],
        "openai_extra_body_sha256": manifest["openai_extra_body_sha256"],
        "openai_stream_include_usage": False,
        "online_evolution": False,
        "permission_mode": "bypassPermissions",
        "profile": "B0",
        "provider_revision": "provider-2026-08-23",
        "skills_enabled": False,
        "skills_seed_sha256": None,
        "system_prompt_hash": manifest["system_prompt_hash"],
        "system_prompt_mode": "fixed",
        "thinking": False,
        "tool_schema_hash": manifest["tool_schema_hash"],
    }
    assert "disabled" not in serialized


def test_live_smoke_fixture_is_stable_and_reusable(tmp_path: Path) -> None:
    calls = 0

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            self.trace = kwargs["trace"]

        async def run_once(self, _: str) -> dict[str, object]:
            with self.trace.span("model.call", component="model", purpose="main") as span:
                span.finish(metrics={"provider_usage": EXACT_USAGE}, payload={"text": "fixed"})
            Path("answer.txt").write_text("answer=fixed\n", encoding="utf-8")
            return {"text": "fixed", "tokens": {"input": 3, "output": 2}}

    runs_root = tmp_path / "persistent-runs"
    adapter = LiveAgentAdapter(
        api_base="https://example.invalid/private-path-value/v1",
        api_key="secret",
        model="test-model",
        provider_revision="test-rev",
        custom_tools=({"name": "write_file"},),
        agent_factory=FakeAgent,
    )

    first = asyncio.run(run_smoke(agent=adapter, runs_root=runs_root, run_id="same-run"))
    tasks_path = runs_root / "infra-smoke" / "same-run" / "tasks.jsonl"
    manifest_path = runs_root / "infra-smoke" / "same-run" / "manifest.json"
    config_path = runs_root / "infra-smoke" / "same-run" / "config.lock.json"
    first_tasks = tasks_path.read_bytes()
    second = asyncio.run(run_smoke(agent=adapter, runs_root=runs_root, run_id="same-run"))

    fixture = runs_root / ".fixtures" / "infra-smoke-v1"
    assert fixture.is_dir()
    assert (fixture / ".git").is_dir()
    assert (fixture / "answer.txt").read_text(encoding="utf-8") == "answer=broken\n"
    assert tasks_path.read_bytes() == first_tasks
    persisted_identity = (
        manifest_path.read_text(encoding="utf-8")
        + config_path.read_text(encoding="utf-8")
    )
    assert "private-path-value" not in persisted_identity
    assert first["counts"]["resolved"] == second["counts"]["resolved"] == 1
    assert calls == 1


def test_product_profile_runs_full_fixture_without_limits(tmp_path: Path) -> None:
    implementation = '''"""Small URL slug helper."""
import re
import unicodedata


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"
'''

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["max_turns"] is None
            assert kwargs["max_output_tokens"] is None
            assert kwargs["max_input_tokens_per_call"] is None
            assert {tool["name"] for tool in kwargs["custom_tools"]} == {
                "read_file", "write_file", "edit_file", "list_files", "grep_search",
                "run_shell", "skill", "update_work_state", "read_observation",
            }
            assert "repository-task solver" in kwargs["custom_system_prompt"]
            assert kwargs["skills_enabled"] is True
            assert kwargs["repair_task"].canonical_issue
            self.trace = kwargs["trace"]

        async def run_once(self, _: str) -> dict[str, object]:
            with self.trace.span("model.call", component="model", purpose="main") as span:
                span.finish(metrics={"provider_usage": EXACT_USAGE}, payload={"text": "fixed"})
            Path("slugger.py").write_text(implementation, encoding="utf-8", newline="\n")
            return {"text": "fixed", "tokens": {"input": 3, "output": 2}}

        async def drain_background_skill_tasks(self) -> None:
            return None

        async def extract_now(self, hint: str = "") -> dict[str, object]:
            assert "durable repository-execution lessons" in hint
            skill = Path.home() / ".sans" / "skills" / "slug-quality" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("---\nname: slug-quality\n---\nKeep slug patches focused.\n", encoding="utf-8")
            from agents.skill_evolution import get_evolution_dir
            provenance = get_evolution_dir() / "online_provenance.jsonl"
            provenance.parent.mkdir(parents=True)
            provenance.write_text("{}\n", encoding="utf-8")
            return {"ok": True, "action": "add", "skill": "slug-quality"}

        async def shutdown(self) -> None:
            return None

    adapter = LiveAgentAdapter(
        api_base="https://example.invalid/v1",
        api_key="secret",
        model="test-model",
        profile="B1",
        online_evolution=True,
        max_turns=None,
        max_output_tokens=None,
        max_input_tokens=None,
        custom_tools=None,
        custom_system_prompt=None,
        agent_factory=FakeAgent,
    )
    summary = asyncio.run(run_full_eval(
        adapter=adapter, runs_root=tmp_path / "runs", run_id="product"
    ))

    assert summary["counts"]["resolved"] == 1
    config = json.loads(
        (tmp_path / "runs" / "harness-full-eval" / "product" / "config.lock.json")
        .read_text(encoding="utf-8")
    )
    assert config["profile"] == "B1"
    assert config["live_agent"]["max_turns"] is None
    assert config["live_agent"]["custom_tools"] is False
    attempt = tmp_path / "runs" / "harness-full-eval" / "product" / "attempts" / "slugify-unicode" / "attempt-0001"
    snapshot = json.loads((attempt / "learning" / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["files"] == 1
    assert (attempt / "learning" / "skills" / "slug-quality" / "SKILL.md").is_file()
    assert (attempt / "learning" / "evolution" / "online_provenance.jsonl").is_file()
    artifacts = json.loads(
        (tmp_path / "runs" / "harness-full-eval" / "product" / "artifacts.json").read_text(encoding="utf-8")
    )["artifacts"]
    assert any(item["path"].endswith("learning/skills/slug-quality/SKILL.md") for item in artifacts)


def test_execution_training_fixtures_and_target_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from agents import skills
    from agents.online_skill_evolution import OnlineSkillCandidate, maintain_online_skill_candidate
    from benchmarks.execution_training import FIXTURES, TARGET_SKILL, _fixture, _verify_single_skill

    seed = tmp_path / "seed" / TARGET_SKILL
    seed.mkdir(parents=True)
    (seed / "SKILL.md").write_text(f"---\nname: {TARGET_SKILL}\n---\nKeep checks focused.\n", encoding="utf-8")
    assert len(_verify_single_skill(seed.parent)) == 64
    for spec in FIXTURES:
        repository, commit = _fixture(tmp_path / "fixtures" / spec.task_id, spec)
        assert _fixture(repository, spec)[1] == commit

    captured: dict[str, object] = {}
    existing = SimpleNamespace(
        name=TARGET_SKILL, description="existing", when_to_use="repository tasks",
        source="user", context="inline", prompt_template="Keep checks focused.",
    )
    monkeypatch.setattr(skills, "discover_skills", lambda: [existing])
    monkeypatch.setattr(skills, "retrieve_relevant_skills", lambda *args, **kwargs: [])
    monkeypatch.setattr(skills, "create_skill", lambda **kwargs: pytest.fail("must not add"))

    def evolve_skill(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True, "skill": kwargs["skill_name"]}

    monkeypatch.setattr(skills, "evolve_skill", evolve_skill)

    async def side_query(_system: str, _payload: str) -> str:
        return json.dumps({"action": "add", "target_skill": "other", "merged_instructions": "merged"})

    result = asyncio.run(maintain_online_skill_candidate(
        candidate=OnlineSkillCandidate("other", "other", instructions="other"),
        side_query=side_query, required_target_skill=TARGET_SKILL,
    ))
    assert result["action"] == "merge"
    assert captured["skill_name"] == TARGET_SKILL


def test_training_reports_independently_verified_patches() -> None:
    from benchmarks.execution_training import _verified_training_patch

    assert _verified_training_patch({"resolved": True}, {})
    assert _verified_training_patch(
        {"resolved": False}, {"evaluation": {"diagnostic": {"resolved": True}}}
    )
    assert not _verified_training_patch(
        {"resolved": False}, {"evaluation": {"diagnostic": {"resolved": False}}}
    )
