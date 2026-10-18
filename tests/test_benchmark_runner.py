from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import benchmarks.runner as runner_module
from benchmarks import BenchmarkRunner, TaskSpec
from benchmarks.live import LiveAgentAdapter, TraceAcceptanceError


def test_cli_output_paths_stay_inside_experiment(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    monkeypatch.setattr(runner_module, "EXPERIMENT_ROOT", root)
    assert runner_module.experiment_path(root / "runs") == (root / "runs").resolve()
    with pytest.raises(ValueError, match="output must stay under"):
        runner_module.experiment_path(root / ".." / "outside")


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def make_repository(root: Path) -> tuple[Path, str]:
    repository = root / "target"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.name", "Benchmark Test")
    git(repository, "config", "user.email", "benchmark@example.invalid")
    git(repository, "config", "core.autocrlf", "false")
    (repository / "answer.txt").write_bytes(b"answer=old\n")
    git(repository, "add", "--", "answer.txt")
    git(repository, "commit", "-qm", "base")
    base = git(repository, "rev-parse", "HEAD")

    # A future commit deliberately contains the answer. Isolated checkouts must hide it.
    (repository / "answer.txt").write_bytes(b"answer=hidden-future-solution\n")
    git(repository, "commit", "-qam", "future solution")
    return repository, base


def task_for(repository: Path, base: str) -> TaskSpec:
    return TaskSpec(
        task_id="sample-1",
        repository=repository,
        base_commit=base,
        problem_statement="Change answer from old to fixed and add evidence.txt.",
        platform="local",
        network_policy="disabled",
        evaluator_metadata={
            "expected": "answer=fixed\n", "hidden_test": "never expose",
            "nested": {"checks": ["answer", "evidence"]},
        },
    )


def test_task_is_frozen_and_agent_view_hides_runner_data(tmp_path: Path) -> None:
    repository, base = make_repository(tmp_path)
    task = task_for(repository, base)

    with pytest.raises(FrozenInstanceError):
        task.task_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        task.evaluator_metadata["expected"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        task.evaluator_metadata["nested"]["checks"] += ("changed",)  # type: ignore[index,operator]
    assert set(task.agent_view()) == {
        "task_id", "problem_statement", "platform", "timeout_seconds", "network_policy"
    }
    assert "evaluator_metadata" not in task.agent_view()
    assert "repository" not in task.agent_view()
    assert "base_commit" not in task.agent_view()


def test_guards_sequential_execution_and_external_runs_root(tmp_path: Path) -> None:
    repository, base = make_repository(tmp_path)
    task = task_for(repository, base)

    async def agent(**_: object) -> None:
        return None

    def evaluator(**_: object) -> bool:
        return False

    with pytest.raises(ValueError, match="worker_count=1"):
        BenchmarkRunner(
            runs_root=tmp_path / "runs", suite_name="probe", run_id="r1",
            tasks=[task], agent=agent, evaluator=evaluator, worker_count=2,
        )
    with pytest.raises(ValueError, match="outside"):
        BenchmarkRunner(
            runs_root=repository / "runs", suite_name="probe", run_id="r1",
            tasks=[task], agent=agent, evaluator=evaluator,
        )
    with pytest.raises(ValueError, match="outside"):
        BenchmarkRunner(
            runs_root=repository.parent, suite_name=repository.name, run_id="r1",
            tasks=[task], agent=agent, evaluator=evaluator,
        )


def test_product_profile_enables_product_state_in_isolated_home(tmp_path: Path) -> None:
    repository, base = make_repository(tmp_path)
    task = task_for(repository, base)
    env = runner_module._environment(tmp_path / "home", task, "B1")

    assert env["SANS_AUTO_SKILL_EVOLUTION"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "SANS_DISABLE_MCP" not in env
    assert "SANS_DISABLE_MEMORY" not in env
    assert "SANS_DISABLE_SESSION_PERSISTENCE" not in env


def test_patch_evaluation_artifacts_and_resume_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, base = make_repository(tmp_path)
    task = task_for(repository, base)
    calls = {"agent": 0, "evaluator": 0}
    scratch_paths: list[Path] = []
    monkeypatch.setenv("BENCH_ENV_SENTINEL", "original")

    async def agent(*, task, workdir: Path, trace, env) -> dict[str, str]:
        calls["agent"] += 1
        assert "evaluator_metadata" not in task
        assert "repository" not in task
        assert Path.cwd() == workdir
        assert Path(os.environ["HOME"]) == Path(env["HOME"])
        assert env["SANS_AUTO_SKILL_EVOLUTION"] == "0"
        assert env["SANS_DISABLE_MCP"] == "1"
        assert env["SANS_DISABLE_MEMORY"] == "1"
        assert env["SANS_DISABLE_SESSION_PERSISTENCE"] == "1"
        assert not runner_module._within(workdir, runner.run_dir)
        assert not runner_module._within(Path(env["HOME"]), runner.run_dir)
        assert git(workdir, "remote") == ""
        assert git(workdir, "for-each-ref", "--format=%(refname)") == ""
        assert git(workdir, "fsck", "--unreachable", "--no-reflogs") == ""
        assert (workdir / "answer.txt").read_bytes() == b"answer=old\n"
        scratch_paths.extend((workdir, Path(env["HOME"])))
        (Path(env["HOME"]) / "agent-state").write_text("isolated", encoding="utf-8")
        os.environ["BENCH_ENV_SENTINEL"] = "mutated"
        os.environ["BENCH_ENV_LEAK"] = "must disappear"
        (workdir / "answer.txt").write_bytes(b"answer=fixed\n")
        (workdir / "evidence.txt").write_bytes(b"created by agent\n")
        with trace.span("model.call", component="model", purpose="main") as span:
            span.finish(metrics={"provider_usage": {
                "input_tokens": 5, "output_tokens": 2, "total_tokens": 7,
            }})
        with trace.span("tool.call", component="tool", purpose="write_file"):
            pass
        print("agent stdout")
        return {"text": "fixed", "tokens": 7}

    def evaluator(*, task: TaskSpec, workdir: Path, env) -> dict[str, object]:
        calls["evaluator"] += 1
        assert task.evaluator_metadata["hidden_test"] == "never expose"
        assert os.environ["BENCH_ENV_SENTINEL"] == "original"
        assert "BENCH_ENV_LEAK" not in os.environ
        assert scratch_paths[-2].exists() is False  # Agent scratch was already removed.
        assert not runner_module._within(workdir, runner.run_dir)
        assert not runner_module._within(Path(env["HOME"]), runner.run_dir)
        scratch_paths.extend((workdir, Path(env["HOME"])))
        assert (workdir / "answer.txt").read_text(encoding="utf-8") == task.evaluator_metadata["expected"]
        assert (workdir / "evidence.txt").read_bytes() == b"created by agent\n"
        print("evaluator stdout")
        return {"resolved": True, "score": 1}

    runner = BenchmarkRunner(
        runs_root=tmp_path / "runs",
        suite_name="probe",
        run_id="run-1",
        tasks=[task],
        agent=agent,
        evaluator=evaluator,
        manifest={"harness_commit": "test", "harness_dirty_state": False},
    )
    summary = asyncio.run(runner.run())

    run_dir = tmp_path / "runs" / "probe" / "run-1"
    attempt = run_dir / "attempts" / "sample-1" / "attempt-0001"
    assert not runner_module._within(run_dir, repository)
    assert all(not path.exists() for path in scratch_paths)
    assert set(path.name for path in attempt.iterdir()) == {
        "evaluator", "final.patch", "result.json", "stderr.log", "stdout.log", "trace.jsonl"
    }
    assert set(path.name for path in (attempt / "evaluator").iterdir()) == {
        "eval.log", "report.json", "test-output.log"
    }
    for name in ("manifest.json", "config.lock.json", "tasks.jsonl", "summary.json", "predictions.jsonl"):
        assert (run_dir / name).is_file()
    config = json.loads((run_dir / "config.lock.json").read_text())
    assert config == {
        "attempts_per_task": 1, "mcp": False, "memory": False,
        "infrastructure_retries_on_resume": 1,
        "online_evolution": False, "profile": "B0", "schema_version": 1,
        "session_persistence": False, "worker_count": 1,
    }
    assert (run_dir / "blobs").is_dir()
    result = json.loads((attempt / "result.json").read_text())
    assert result["evaluation"]["resolved"] is True
    assert result["final_output"] == "fixed"
    assert json.loads((attempt / "evaluator" / "report.json").read_text())["score"] == 1
    assert "agent stdout" in (attempt / "stdout.log").read_text(encoding="utf-8")
    assert "evaluator stdout" in (attempt / "evaluator" / "test-output.log").read_text(encoding="utf-8")
    assert summary["counts"] == {
        "tasks": 1, "completed": 1, "resolved": 1,
        "unknown_resolution": 0, "infra_failures": 0,
    }
    assert summary["totals"]["tokens"] == 7
    assert summary["totals"]["model_calls"] == 1
    assert summary["totals"]["tool_calls"] == 1
    assert summary["totals"]["no_progress_tool_calls"] == 0
    assert os.environ["BENCH_ENV_SENTINEL"] == "original"
    assert "BENCH_ENV_LEAK" not in os.environ

    # The exported patch is sufficient in a third checkout, without the Agent.
    replay = tmp_path / "independent-replay"
    git(tmp_path, "clone", "-q", str(repository), str(replay))
    git(replay, "checkout", "-q", "--detach", base)
    git(replay, "apply", str(attempt / "final.patch"))
    assert (replay / "answer.txt").read_text(encoding="utf-8") == "answer=fixed\n"
    assert (replay / "evidence.txt").read_text(encoding="utf-8") == "created by agent\n"
    assert (repository / "answer.txt").read_bytes() == b"answer=hidden-future-solution\n"
    assert git(repository, "status", "--porcelain") == ""

    # Existing logs are not completion markers; only a valid, artifact-backed result is.
    asyncio.run(runner.run())
    assert calls == {"agent": 1, "evaluator": 1}
    malformed = json.loads((attempt / "result.json").read_text())
    malformed["patch"]["sha256"] = "0" * 64
    (attempt / "result.json").write_text(json.dumps(malformed), encoding="utf-8")
    (attempt / "stale-artifact").write_text("partial", encoding="utf-8")
    asyncio.run(runner.run())
    assert calls == {"agent": 2, "evaluator": 2}
    assert not (attempt / "stale-artifact").exists()


def test_environment_failure_still_writes_terminal_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, base = make_repository(tmp_path)

    async def agent(**_: object) -> None:
        raise AssertionError("must not run")

    monkeypatch.setattr(runner_module, "_checkout", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    runner = BenchmarkRunner(
        runs_root=tmp_path / "runs", suite_name="probe", run_id="infra-failure",
        tasks=[task_for(repository, base)], agent=agent, evaluator=lambda **_: True,
        manifest={"harness_commit": "test", "harness_dirty_state": False},
    )
    asyncio.run(runner.run())

    attempt = runner.run_dir / "attempts" / "sample-1" / "attempt-0001"
    assert (attempt / "final.patch").read_bytes() == b""
    report = json.loads((attempt / "evaluator" / "report.json").read_text())
    result = json.loads((attempt / "result.json").read_text())
    assert report == {
        "resolved": False,
        "status": "infrastructure_failure",
        "infrastructure_failure": {
            "phase": "environment_prepare", "kind": "environment_prepare_error",
        },
    }
    assert result["status"] == "infrastructure_failure"
    assert result["infrastructure_failure"] == report["infrastructure_failure"]
    assert result["termination_reason"] is None
    assert result["patch"]["bytes"] == 0


@pytest.mark.parametrize(
    ("failure", "expected_kind"),
    [
        (RuntimeError("estimated input exceeds max_input_tokens_per_call=8"), "input_budget"),
        (RuntimeError("Turn limit reached (4 >= 4)"), "turn_budget"),
        (RuntimeError("Cost limit reached ($1 >= $1)"), "cost_budget"),
        (RuntimeError("provider exploded"), "agent_crash"),
        ("timeout", "timeout"),
    ],
)
def test_agent_task_failure_is_not_infrastructure_or_retried(
    tmp_path: Path, failure: Exception | str, expected_kind: str
) -> None:
    repository, base = make_repository(tmp_path)
    original = task_for(repository, base)
    task = TaskSpec(
        task_id=original.task_id,
        repository=original.repository,
        base_commit=original.base_commit,
        problem_statement=original.problem_statement,
        timeout_seconds=0.01 if failure == "timeout" else None,
        evaluator_metadata=original.evaluator_metadata,
    )

    calls = 0

    async def agent(*, workdir: Path, **_: object) -> None:
        nonlocal calls
        calls += 1
        (workdir / "answer.txt").write_text("answer=fixed\n", encoding="utf-8")
        (workdir / "evidence.txt").write_text("created by agent\n", encoding="utf-8")
        if failure == "timeout":
            await asyncio.sleep(0.2)
        else:
            raise failure  # type: ignore[misc]

    def evaluator(**_: object) -> dict[str, object]:
        return {"resolved": True, "score": 1, "status": "completed"}

    runner = BenchmarkRunner(
        runs_root=tmp_path / "runs", suite_name="infra", run_id="failure",
        tasks=[task], agent=agent, evaluator=evaluator,
        manifest={"harness_commit": "test", "harness_dirty_state": False},
    )
    summary = asyncio.run(runner.run())
    attempt = runner.run_dir / "attempts" / task.task_id / "attempt-0001"
    result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    prediction = json.loads((runner.run_dir / "predictions.jsonl").read_text(encoding="utf-8"))

    assert result["status"] == "task_failure"
    assert result["infrastructure_failure"] is None
    assert result["termination_reason"] == {"kind": expected_kind, "phase": "agent"}
    assert result["evaluation"]["resolved"] is False
    assert result["evaluation"]["status"] == "task_failure"
    assert result["evaluation"]["diagnostic"]["resolved"] is True
    assert prediction["resolved"] is False
    assert prediction["termination_reason"] == result["termination_reason"]
    assert summary["counts"]["resolved"] == 0
    assert summary["counts"]["infra_failures"] == 0
    assert summary["tasks"][0]["status"] == "task_failure"

    asyncio.run(runner.run())
    assert calls == 1


@pytest.mark.parametrize(
    ("reported", "expected_kind"),
    [
        ("turn_budget_exhausted", "turn_budget"),
        ({"kind": "cost_budget_exhausted"}, "cost_budget"),
    ],
)
def test_agent_reported_termination_reason_is_a_task_failure(
    tmp_path: Path, reported: object, expected_kind: str
) -> None:
    repository, base = make_repository(tmp_path)
    calls = 0

    async def agent(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"text": "stopped", "termination_reason": reported}

    runner = BenchmarkRunner(
        runs_root=tmp_path / "runs", suite_name="reported", run_id="failure",
        tasks=[task_for(repository, base)], agent=agent,
        evaluator=lambda **_: {"resolved": True},
        manifest={"harness_commit": "test", "harness_dirty_state": False},
    )
    summary = asyncio.run(runner.run())
    result = json.loads((runner.run_dir / "attempts" / "sample-1" / "attempt-0001" / "result.json").read_text())

    assert result["status"] == "task_failure"
    assert result["termination_reason"] == {"phase": "agent", "kind": expected_kind}
    assert result["infrastructure_failure"] is None
    assert summary["counts"]["infra_failures"] == 0
    assert summary["tasks"][0]["status"] == "task_failure"
    asyncio.run(runner.run())
    assert calls == 1


def test_invalid_patch_is_a_task_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository, base = make_repository(tmp_path)
    evaluator_calls = 0

    async def agent(**_: object) -> dict[str, str]:
        return {"text": "invalid patch"}

    def evaluator(**_: object) -> dict[str, bool]:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return {"resolved": True}

    monkeypatch.setattr(runner_module, "_apply_patch", lambda *args, **kwargs: False)
    runner = BenchmarkRunner(
        runs_root=tmp_path / "runs", suite_name="invalid", run_id="patch",
        tasks=[task_for(repository, base)], agent=agent, evaluator=evaluator,
        manifest={"harness_commit": "test", "harness_dirty_state": False},
    )
    summary = asyncio.run(runner.run())
    result = json.loads((runner.run_dir / "attempts" / "sample-1" / "attempt-0001" / "result.json").read_text())

    assert result["status"] == "task_failure"
    assert result["termination_reason"] == {"phase": "patch_replay", "kind": "invalid_patch"}
    assert result["infrastructure_failure"] is None
    assert result["evaluation"]["resolved"] is False
    assert summary["counts"]["infra_failures"] == 0
    assert summary["tasks"][0]["status"] == "task_failure"
    assert evaluator_calls == 0


def test_infrastructure_failure_resumes_once_without_discarding_facts(tmp_path: Path) -> None:
    repository, base = make_repository(tmp_path)
    calls = 0

    async def agent(*, workdir: Path, trace, **_: object) -> dict[str, str]:
        nonlocal calls
        calls += 1
        with trace.span("model.call", component="model", purpose="main") as span:
            tokens = 3 if calls == 1 else 7
            span.finish(metrics={"provider_usage": {
                "input_tokens": tokens - 1, "output_tokens": 1, "total_tokens": tokens,
            }})
        (workdir / "answer.txt").write_text("answer=fixed\n", encoding="utf-8")
        (workdir / "evidence.txt").write_text("created by agent\n", encoding="utf-8")
        if calls == 1:
            raise TraceAcceptanceError("bad trace")
        return {"text": "fixed"}

    runner = BenchmarkRunner(
        runs_root=tmp_path / "runs", suite_name="resume", run_id="infra",
        tasks=[task_for(repository, base)], agent=agent,
        evaluator=lambda **_: {"resolved": True},
        manifest={"harness_commit": "test", "harness_dirty_state": False},
    )
    first_summary = asyncio.run(runner.run())
    attempts = runner.run_dir / "attempts" / "sample-1"
    first_result = (attempts / "attempt-0001" / "result.json").read_bytes()
    first_trace = (attempts / "attempt-0001" / "trace.jsonl").read_bytes()

    assert first_summary["counts"]["infra_failures"] == 1
    assert first_summary["totals"]["tokens"] == 3
    assert not (attempts / "attempt-0002").exists()

    resumed = asyncio.run(runner.run())
    assert calls == 2
    assert (attempts / "attempt-0001" / "result.json").read_bytes() == first_result
    assert (attempts / "attempt-0001" / "trace.jsonl").read_bytes() == first_trace
    assert (attempts / "attempt-0002" / "result.json").is_file()
    assert resumed["tasks"][0]["attempt_ids"] == ["attempt-0001", "attempt-0002"]
    assert resumed["counts"]["resolved"] == 1
    assert resumed["counts"]["infra_failures"] == 1
    assert resumed["totals"]["tokens"] == 10
    assert resumed["totals"]["model_calls"] == 2

    asyncio.run(runner.run())
    assert calls == 2


def test_live_agent_configuration_is_automatically_locked(tmp_path: Path) -> None:
    repository, base = make_repository(tmp_path)
    common = {
        "api_base": "https://example.invalid/v1",
        "api_key": "must-not-be-recorded",
        "model": "test-model",
        "provider_revision": "test-rev",
        "custom_system_prompt": "fixed prompt",
        "custom_tools": (),
        "agent_factory": lambda **_: None,
    }
    first = BenchmarkRunner(
        runs_root=tmp_path / "runs", suite_name="locked", run_id="same-id",
        tasks=[task_for(repository, base)],
        agent=LiveAgentAdapter(max_turns=4, **common), evaluator=lambda **_: {"resolved": False},
    )
    first._prepare_run()

    manifest = (first.run_dir / "manifest.json").read_text(encoding="utf-8")
    config = (first.run_dir / "config.lock.json").read_text(encoding="utf-8")
    assert "must-not-be-recorded" not in manifest + config
    assert json.loads(config)["live_agent"]["max_turns"] == 4

    changed = BenchmarkRunner(
        runs_root=tmp_path / "runs", suite_name="locked", run_id="same-id",
        tasks=[task_for(repository, base)],
        agent=LiveAgentAdapter(max_turns=5, **common), evaluator=lambda **_: {"resolved": False},
    )
    with pytest.raises(ValueError, match="immutable run artifact"):
        changed._prepare_run()
