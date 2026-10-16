from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.panel_run import (
    LIVE_AGENT,
    MIN_FREE_BYTES,
    PANEL,
    REPO_ROOT,
    SECRET_ENV,
    _disk_guard_root,
    harbor_command,
    harbor_target,
    load_wave,
    run_panel,
    task_key,
)


def _space(free: int = MIN_FREE_BYTES + 1) -> SimpleNamespace:
    return SimpleNamespace(free=free)


def _job_result(path: Path, *, errored: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1, "n_errored_trials": errored,
            "n_input_tokens": 10, "n_cache_tokens": 6, "n_output_tokens": 2,
            "cost_usd": 0.01,
            "evals": {"eval": {"metrics": [{"mean": 1.0}], "pass_at_k": {"1": 1.0}}},
        },
    }), encoding="utf-8")


def _one_task_panel(path: Path) -> Path:
    panel, tasks = load_wave(PANEL, 1)
    target = path / "panel.json"
    target.write_text(json.dumps({
        "schema_version": 1, "panel_id": "one-task", "trial_budget": panel["trial_budget"],
        "waves": [{"wave": 1, "tasks": tasks[:1]}],
    }), encoding="utf-8")
    return target


def _two_task_panel(path: Path) -> Path:
    panel, tasks = load_wave(PANEL, 1)
    target = path / "panel.json"
    target.write_text(json.dumps({
        "schema_version": 1, "panel_id": "two-task", "trial_budget": panel["trial_budget"],
        "waves": [{"wave": 1, "tasks": tasks[:2]}],
    }), encoding="utf-8")
    return target


def test_frozen_waves_are_balanced_and_map_to_harbor() -> None:
    legacy = json.loads(PANEL.with_name("freeze40-v1.json").read_text(encoding="utf-8"))
    assert legacy["panel_id"] == "freeze40-v1"
    assert "trial_budget" not in legacy
    for wave in (1, 2):
        panel, tasks = load_wave(PANEL, wave)
        assert panel["panel_id"] == "freeze40-cap40-v1"
        assert panel["trial_budget"] == {"max_model_calls": 40, "max_tool_calls": 40}
        assert [sum(task["suite"] == suite for task in tasks) for suite in (
            "swebench_verified", "swebench_pro", "terminal_bench",
        )] == [10, 6, 4]

    assert harbor_target({
        "suite": "swebench_verified", "instance_id": "django__django-15022",
    }) == (
        "swe-bench/swe-bench-verified@sha256:b934b0cc3dc800fe945eaf9f1623329db97ee3133c706d20644524c7759fb341",
        "swe-bench/django__django-15022",
    )
    assert harbor_target({
        "suite": "swebench_pro", "instance_id": "instance_flipt-io__flipt-abc",
    }) == (
        "scale-ai/swe-bench-pro@sha256:88411d32ff27e53a4c1a7e29f0c2aeba180c8e5d60f221cab5ed56325f33549d",
        "scale-ai/instance_flipt-io__flipt-abc",
    )
    assert harbor_target({
        "suite": "terminal_bench", "instance_id": "terminal-bench/fix-git",
    }) == ("terminal-bench/terminal-bench-2-1@6", "terminal-bench/fix-git")
    assert harbor_target({
        "suite": "swebench_pro",
        "instance_id": "instance_NodeBB__NodeBB-a917210c5b2c20637094545401f85783905c074c-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
    })[1] == (
        "scale-ai/instance_nodebb__nodebb-a917210c5b2c20637094545401f85783905c074c-"
        "vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
    )


def test_cap60_panel_reuses_the_frozen_40_tasks() -> None:
    cap60 = PANEL.with_name("freeze40-cap60-v1.json")
    for wave in (1, 2):
        cap40_panel, cap40_tasks = load_wave(PANEL, wave)
        cap60_panel, cap60_tasks = load_wave(cap60, wave)
        assert cap60_tasks == cap40_tasks
        assert cap60_panel["sources"] == cap40_panel["sources"]
        assert cap60_panel["trial_budget"] == {
            "max_model_calls": 40,
            "max_tool_calls": 60,
            "model_call_timeout_seconds": 180,
        }


def test_legacy_panel_is_unlimited_without_inheriting_cap40(tmp_path: Path) -> None:
    legacy_path = PANEL.with_name("freeze40-v1.json")
    panel, tasks = load_wave(legacy_path, 1)
    assert "trial_budget" not in panel
    command = harbor_command(
        tasks[0], jobs_dir=tmp_path, job_name="legacy", mode="live",
        skills_seed=tmp_path,
    )
    assert not any(item.startswith(("max_model_calls=", "max_tool_calls=")) for item in command)

    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json")
        return SimpleNamespace(returncode=0, stdout="")

    summary = run_panel(
        panel_path=legacy_path, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=fake_run, disk_usage=lambda _: _space() if calls == 0 else _space(0),
    )
    record = next((tmp_path / "runs" / "results").rglob("attempt-*.json"))
    assert calls == 1
    assert summary["trial_budget"] is None
    assert json.loads(record.read_text(encoding="utf-8"))["trial_budget"] is None


def test_present_trial_budget_must_be_valid(tmp_path: Path) -> None:
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    panel["trial_budget"]["max_model_calls"] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="trial_budget"):
        load_wave(path, 1)

    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    panel["trial_budget"]["agent_timeout_multiplier"] = 0
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="agent_timeout_multiplier"):
        load_wave(path, 1)

    panel["trial_budget"]["agent_timeout_multiplier"] = 8
    panel["trial_budget"]["environment_build_timeout_multiplier"] = 0
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="environment_build_timeout_multiplier"):
        load_wave(path, 1)

    panel["thinking"] = "false"
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="thinking"):
        load_wave(path, 1)

    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    panel["trial_budget"]["model_call_timeout_seconds"] = 0
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="trial_budget"):
        load_wave(path, 1)


def test_live_command_is_single_attempt_frozen_and_contains_no_secret(tmp_path: Path) -> None:
    task = {"suite": "swebench_verified", "instance_id": "django__django-15022"}
    command = harbor_command(
        task, jobs_dir=tmp_path / "jobs", job_name="job", mode="live",
        skills_seed=tmp_path / "frozen", extra_docker_compose=tmp_path / "proxy.yaml",
        max_model_calls=40, max_tool_calls=40,
        model_call_timeout_seconds=180,
        agent_timeout_multiplier=8,
        environment_build_timeout_multiplier=2,
        thinking=False,
        executable="harbor",
    )
    secret = "never-in-command"

    assert command[:2] == ["harbor", "run"]
    assert [command[command.index(flag) + 1] for flag in ("-n", "-k", "--max-retries")] == ["1", "1", "0"]
    assert "--delete" in command and "--yes" in command
    assert command[command.index("--extra-docker-compose") + 1] == str(tmp_path / "proxy.yaml")
    assert "HTTP_PROXY=http://host.docker.internal:18080" in command
    assert "HTTPS_PROXY=http://host.docker.internal:18080" in command
    assert "NO_PROXY=localhost,127.0.0.1,::1" in command
    assert command[command.index("-a") + 1] == LIVE_AGENT
    assert "thinking=false" in command
    assert "online_evolution=false" in command
    assert "max_model_calls=40" in command
    assert "max_tool_calls=40" in command
    assert "model_call_timeout_seconds=180" in command
    assert command[command.index("--agent-timeout-multiplier") + 1] == "8"
    assert command[command.index("--environment-build-timeout-multiplier") + 1] == "2"
    assert any(item.startswith("skills_seed=") for item in command)
    assert all(secret not in item for item in command)
    assert "thinking=true" in harbor_command(
        task, jobs_dir=tmp_path / "jobs", job_name="default", mode="live",
        skills_seed=tmp_path / "frozen",
    )
    commandcode = harbor_command(
        task, jobs_dir=tmp_path / "jobs", job_name="commandcode", mode="live",
        skills_seed=tmp_path / "frozen", provider="commandcode",
    )
    assert "deepseek/deepseek-v4-flash" in commandcode
    assert "api_base=https://api.commandcode.ai/provider/v1" in commandcode
    assert not any("_price=" in item for item in commandcode)


def test_existing_result_is_skipped_without_starting_harbor(tmp_path: Path) -> None:
    panel, tasks = load_wave(PANEL, 1)
    name = task_key(panel["panel_id"], 1, "nop", tasks[0])
    assert name == "freeze40-cap40-v1-w1-nop-verified-c9bdfb107741"
    harbor_result = tmp_path / "jobs" / f"{name}-attempt-0001" / "result.json"
    _job_result(harbor_result)
    result = tmp_path / "results" / name / "attempt-0001.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({
        "status": "completed", "instance_id": tasks[0]["instance_id"],
        "harbor_result": str(harbor_result.relative_to(tmp_path)),
    }), encoding="utf-8")
    calls = 0

    def fake_run(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("completed result must be skipped")

    # Stop before the second task; the assertion here is only about task-level resume.
    summary = run_panel(
        panel_path=PANEL, wave=1, runs_root=tmp_path, mode="nop", run=fake_run,
        disk_usage=lambda _: _space(0),
    )
    assert calls == 0
    assert summary["counts"]["skipped"] == 1
    assert summary["stop_reason"] == "disk_guard"
    assert (tmp_path / "aggregate-freeze40-cap40-v1-w1-nop.json").is_file()


def test_live_secret_only_enters_child_env_and_is_redacted_from_files(tmp_path: Path) -> None:
    secret = "rc-test-super-secret"
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        assert all(secret not in item for item in command)
        assert kwargs["env"][SECRET_ENV] == secret  # type: ignore[index]
        job_name = command[command.index("--job-name") + 1]
        jobs_dir = Path(command[command.index("--jobs-dir") + 1])
        result = jobs_dir / job_name / "result.json"
        _job_result(result)
        return SimpleNamespace(returncode=0, stdout=f"provider accidentally echoed {secret}\n")

    summary = run_panel(
        panel_path=PANEL, wave=1, runs_root=tmp_path, mode="live",
        skills_seed=frozen, api_key=secret, run=fake_run,
        disk_usage=lambda _: _space(),
    )
    assert calls == 20
    assert summary["counts"]["completed"] == 20
    assert summary["stop_reason"] is None
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")


def test_429_stops_at_complete_task_boundary(tmp_path: Path) -> None:
    calls = 0

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stdout="HTTP 429 quota exhausted")

    summary = run_panel(
        panel_path=PANEL, wave=1, runs_root=tmp_path, mode="nop", run=fake_run,
        disk_usage=lambda _: _space(),
    )
    assert calls == 1
    assert summary["stop_reason"] == "quota"
    assert summary["counts"] == {
        "selected": 20, "completed": 0, "failed": 1, "skipped": 0, "pending": 19,
    }
    result = next((tmp_path / "results").rglob("attempt-*.json"))
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "quota_failure"


def test_zero_exit_with_errored_harbor_result_is_failure(tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json", errored=1)
        return SimpleNamespace(returncode=0, stdout="")

    summary = run_panel(
        panel_path=_one_task_panel(tmp_path), wave=1, runs_root=tmp_path / "runs",
        mode="nop", run=fake_run, disk_usage=lambda _: _space(),
    )
    assert summary["stop_reason"] == "harbor_failure"
    record = next((tmp_path / "runs" / "results").rglob("attempt-*.json"))
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["harbor_summary"]["n_errored_trials"] == 1


def test_continue_policy_moves_to_next_task_after_recorded_trial_error(tmp_path: Path) -> None:
    names: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        name = command[command.index("--job-name") + 1]
        names.append(name)
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json", errored=1 if len(names) == 1 else 0)
        return SimpleNamespace(returncode=0, stdout="")

    summary = run_panel(
        panel_path=_two_task_panel(tmp_path), wave=1, runs_root=tmp_path / "runs",
        mode="nop", run=fake_run, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert len(names) == 2
    assert all(name.endswith("attempt-0001") for name in names)
    assert summary["stop_reason"] is None
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["completed"] == 1


def test_continue_policy_still_stops_on_429_with_zero_harbor_exit(tmp_path: Path) -> None:
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json", errored=1)
        return SimpleNamespace(returncode=0, stdout="AMD HTTP 429 quota exhausted")

    summary = run_panel(
        panel_path=_two_task_panel(tmp_path), wave=1, runs_root=tmp_path / "runs",
        mode="nop", run=fake_run, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert calls == 1
    assert summary["stop_reason"] == "quota"
    assert summary["results"][0]["status"] == "quota_failure"


def test_concurrency_rate_limit_does_not_stop_the_panel(tmp_path: Path) -> None:
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json", errored=1 if calls == 1 else 0)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "process_concurrency_rate_limit_exceeded; retry later"
                if calls == 1 else ""
            ),
        )

    summary = run_panel(
        panel_path=_two_task_panel(tmp_path), wave=1, runs_root=tmp_path / "runs",
        mode="nop", run=fake_run, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert calls == 2
    assert summary["stop_reason"] is None
    assert [item["status"] for item in summary["results"]] == ["failed", "completed"]


def test_completed_trial_with_recovered_429_stays_completed_and_resumes_without_run(
    tmp_path: Path,
) -> None:
    panel = _one_task_panel(tmp_path)

    def recovered(command: list[str], **kwargs: object) -> SimpleNamespace:
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json")
        return SimpleNamespace(returncode=0, stdout="recovered Retry HTTP 429")

    first = run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=recovered, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert first["stop_reason"] is None
    assert first["results"][0]["status"] == "completed"
    calls = 0

    def must_not_run(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        raise AssertionError("completed result must resume without another Harbor run")

    resumed = run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=must_not_run, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert calls == 0
    assert resumed["counts"]["completed"] == 1


def test_child_pythonpath_prepends_repo_and_preserves_existing_value(tmp_path: Path) -> None:
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = "existing-pythonpath"

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        env = kwargs["env"]  # type: ignore[assignment]
        assert env["PYTHONPATH"].split(os.pathsep) == [str(REPO_ROOT), "existing-pythonpath"]
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json")
        return SimpleNamespace(returncode=0, stdout="")

    try:
        summary = run_panel(
            panel_path=_one_task_panel(tmp_path), wave=1, runs_root=tmp_path / "runs",
            mode="nop", run=fake_run, disk_usage=lambda _: _space(),
        )
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous
    assert summary["counts"]["completed"] == 1


def test_continue_policy_stops_on_rc1_without_job_result(tmp_path: Path) -> None:
    calls = 0

    def fail_import(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stdout="No module named benchmarks")

    summary = run_panel(
        panel_path=_two_task_panel(tmp_path), wave=1, runs_root=tmp_path / "runs",
        mode="nop", run=fail_import, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert calls == 1
    assert summary["stop_reason"] == "harness_failure"


def test_orphan_job_without_result_advances_attempt(tmp_path: Path) -> None:
    panel = _one_task_panel(tmp_path)
    manifest, tasks = load_wave(panel, 1)
    key = task_key(manifest["panel_id"], 1, "nop", tasks[0])
    (tmp_path / "runs" / "jobs" / f"{key}-attempt-0001").mkdir(parents=True)

    def succeed(command: list[str], **kwargs: object) -> SimpleNamespace:
        name = command[command.index("--job-name") + 1]
        assert name.endswith("attempt-0002")
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json")
        return SimpleNamespace(returncode=0, stdout="")

    summary = run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=succeed, disk_usage=lambda _: _space(),
    )
    assert summary["counts"]["completed"] == 1


def test_continue_policy_skips_prior_failed_attempt_on_resume(tmp_path: Path) -> None:
    panel = _two_task_panel(tmp_path)

    def fail_first(command: list[str], **kwargs: object) -> SimpleNamespace:
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json", errored=1)
        return SimpleNamespace(returncode=0, stdout="API timeout")

    run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=fail_first, disk_usage=lambda _: _space(),
    )
    calls = 0

    def finish_second(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        task_name = command[command.index("-i") + 1]
        assert task_name.endswith("django__django-15128")
        name = command[command.index("--job-name") + 1]
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json")
        return SimpleNamespace(returncode=0, stdout="")

    summary = run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=finish_second, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert calls == 1
    assert summary["counts"] == {
        "selected": 2, "completed": 1, "failed": 1, "skipped": 1, "pending": 0,
    }


def test_resume_recomputes_stale_trial_failure_summary_from_harbor_result(tmp_path: Path) -> None:
    panel = _one_task_panel(tmp_path)
    manifest, tasks = load_wave(panel, 1)
    key = task_key(manifest["panel_id"], 1, "nop", tasks[0])
    harbor_result = tmp_path / "runs" / "jobs" / f"{key}-attempt-0001" / "result.json"
    _job_result(harbor_result, errored=1)
    wrapper = tmp_path / "runs" / "results" / key / "attempt-0001.json"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(json.dumps({
        "status": "failed",
        "instance_id": tasks[0]["instance_id"],
        "harbor_result": str(harbor_result.relative_to(tmp_path / "runs")),
        "harbor_summary": {"integrity": "incomplete"},
    }), encoding="utf-8")
    calls = 0

    def must_not_run(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        raise AssertionError("stale wrapper must be refreshed and skipped")

    summary = run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=must_not_run, disk_usage=lambda _: _space(),
        continue_on_task_failure=True,
    )
    assert calls == 0
    assert summary["counts"] == {
        "selected": 1, "completed": 0, "failed": 1, "skipped": 1, "pending": 0,
    }
    assert summary["results"][0]["harbor_summary"]["integrity"] == "trial_error"


def test_failed_attempt_resumes_with_incremented_job_name(tmp_path: Path) -> None:
    panel = _one_task_panel(tmp_path)

    run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="HTTP 429 quota"),
        disk_usage=lambda _: _space(),
    )

    def succeed(command: list[str], **kwargs: object) -> SimpleNamespace:
        name = command[command.index("--job-name") + 1]
        assert name.endswith("attempt-0002")
        jobs = Path(command[command.index("--jobs-dir") + 1])
        _job_result(jobs / name / "result.json")
        return SimpleNamespace(returncode=0, stdout="")

    summary = run_panel(
        panel_path=panel, wave=1, runs_root=tmp_path / "runs", mode="nop",
        run=succeed, disk_usage=lambda _: _space(),
    )
    records = list((tmp_path / "runs" / "results").rglob("attempt-*.json"))
    assert len(records) == 2
    assert summary["counts"]["completed"] == 1
    assert summary["counts"]["skipped"] == 0


def test_wsl_disk_guard_never_falls_back_to_virtual_root(tmp_path: Path) -> None:
    assert _disk_guard_root(tmp_path, is_wsl=True) == Path("/mnt/d")
