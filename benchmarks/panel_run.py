"""Run a frozen benchmark panel one Harbor task at a time."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from .runner import EXPERIMENT_ROOT, experiment_path


PANEL = Path(__file__).with_name("panels") / "freeze40-cap40-v1.json"
MIN_FREE_BYTES = 20 * 1024**3
SECRET_ENV = "SANS_BENCHMARK_API_KEY"
LIVE_AGENT = "benchmarks.harbor_agent:SansAgent"
AMD_BASE = "https://developer.amd.com.cn/radeon/api/v1"
AMD_MODEL = "DeepSeek-V4-Flash"
AMD_REVISION = "amd-radeon-DeepSeek-V4-Flash-0731"
COMMANDCODE_BASE = "https://api.commandcode.ai/provider/v1"
COMMANDCODE_MODEL = "deepseek/deepseek-v4-flash"
COMMANDCODE_REVISION = "commandcode-deepseek-v4-flash"
REPO_ROOT = Path(__file__).resolve().parent.parent

_DATASETS = {
    "swebench_verified": (
        "swe-bench/swe-bench-verified@sha256:b934b0cc3dc800fe945eaf9f1623329db97ee3133c706d20644524c7759fb341",
        "swe-bench/",
    ),
    "swebench_pro": (
        "scale-ai/swe-bench-pro@sha256:88411d32ff27e53a4c1a7e29f0c2aeba180c8e5d60f221cab5ed56325f33549d",
        "scale-ai/",
    ),
    "terminal_bench": ("terminal-bench/terminal-bench-2-1@6", ""),
}
_TASK_NAME_OVERRIDES = {
    "instance_NodeBB__NodeBB-a917210c5b2c20637094545401f85783905c074c-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e":
        "instance_nodebb__nodebb-a917210c5b2c20637094545401f85783905c074c-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e",
}


def load_wave(path: str | Path, wave: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    panel = json.loads(Path(path).read_text(encoding="utf-8"))
    if "thinking" in panel and type(panel["thinking"]) is not bool:
        raise ValueError("thinking must be true or false")
    budget = panel.get("trial_budget")
    if budget is not None and (
        not isinstance(budget, dict) or any(
            type(budget.get(name)) is not int or budget[name] <= 0
            for name in ("max_model_calls", "max_tool_calls")
        )
    ):
        raise ValueError("trial_budget call limits must be positive integers")
    for name in (
        "model_call_timeout_seconds", "agent_timeout_multiplier",
        "environment_build_timeout_multiplier",
    ):
        value = budget.get(name) if isinstance(budget, dict) else None
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        ):
            raise ValueError(f"trial_budget {name} must be positive")
    selected = next((item for item in panel["waves"] if item["wave"] == wave), None)
    if selected is None:
        raise ValueError(f"wave {wave} is not present in {path}")
    tasks = selected["tasks"]
    for task in tasks:
        if task.get("suite") not in _DATASETS or not task.get("instance_id"):
            raise ValueError(f"invalid panel task: {task!r}")
    return panel, tasks


def harbor_target(task: dict[str, str]) -> tuple[str, str]:
    dataset, prefix = _DATASETS[task["suite"]]
    instance = _TASK_NAME_OVERRIDES.get(task["instance_id"], task["instance_id"])
    return dataset, instance if instance.startswith(prefix) else prefix + instance


def task_key(panel_id: str, wave: int, mode: str, task: dict[str, str]) -> str:
    digest = hashlib.sha256(task["instance_id"].encode()).hexdigest()[:12]
    suite = {"swebench_verified": "verified", "swebench_pro": "pro", "terminal_bench": "terminal"}[
        task["suite"]
    ]
    return f"{panel_id}-w{wave}-{mode}-{suite}-{digest}"


def harbor_command(
    task: dict[str, str], *, jobs_dir: Path, job_name: str, mode: str,
    skills_seed: Path | None = None, extra_docker_compose: Path | None = None,
    max_model_calls: int | None = None, max_tool_calls: int | None = None,
    model_call_timeout_seconds: float | None = None,
    agent_timeout_multiplier: float | None = None,
    environment_build_timeout_multiplier: float | None = None,
    thinking: bool = True,
    provider: str = "amd",
    executable: str = "harbor",
) -> list[str]:
    dataset, name = harbor_target(task)
    command = [
        executable, "run", "-d", dataset, "-i", name, "-e", "docker",
        "-n", "1", "-k", "1", "--max-retries", "0", "--delete", "--yes",
        "--job-name", job_name, "--jobs-dir", str(jobs_dir),
    ]
    if agent_timeout_multiplier is not None:
        command += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]
    if environment_build_timeout_multiplier is not None:
        command += [
            "--environment-build-timeout-multiplier",
            str(environment_build_timeout_multiplier),
        ]
    if extra_docker_compose is not None:
        command += [
            "--extra-docker-compose", str(extra_docker_compose),
            "--ve", "HTTP_PROXY=http://host.docker.internal:18080",
            "--ve", "HTTPS_PROXY=http://host.docker.internal:18080",
            "--ve", "NO_PROXY=localhost,127.0.0.1,::1",
        ]
    if mode in {"oracle", "nop"}:
        return command + ["-a", mode]
    if mode != "live" or skills_seed is None:
        raise ValueError("live mode requires a frozen skills seed")
    if provider == "amd":
        api_base, model, revision = AMD_BASE, AMD_MODEL, AMD_REVISION
    elif provider == "commandcode":
        api_base, model, revision = COMMANDCODE_BASE, COMMANDCODE_MODEL, COMMANDCODE_REVISION
    else:
        raise ValueError(f"unsupported provider: {provider}")
    command += [
        "-a", LIVE_AGENT, "-m", model,
        "--ak", f"api_base={api_base}",
        "--ak", f"provider_revision={revision}",
        "--ak", f"thinking={str(thinking).lower()}",
        "--ak", f"skills_seed={skills_seed}",
        "--ak", "online_evolution=false",
    ]
    if provider == "amd":
        command += [
            "--ak", "cache_hit_price=0.0028",
            "--ak", "cache_miss_price=0.14",
            "--ak", "output_price=0.28",
        ]
    for name, value in (
        ("max_model_calls", max_model_calls), ("max_tool_calls", max_tool_calls),
        ("model_call_timeout_seconds", model_call_timeout_seconds),
    ):
        if value is not None:
            command += ["--ak", f"{name}={value}"]
    return command


def run_panel(
    *, panel_path: str | Path, wave: int, runs_root: str | Path, mode: str,
    skills_seed: Path | None = None, api_key: str | None = None,
    extra_docker_compose: Path | None = None,
    executable: str = "harbor", run: Callable[..., Any] = subprocess.run,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    continue_on_task_failure: bool = False,
    provider: str = "amd",
) -> dict[str, Any]:
    panel, tasks = load_wave(panel_path, wave)
    trial_budget = panel.get("trial_budget")
    root = Path(runs_root).expanduser().resolve()
    jobs, results, logs = root / "jobs", root / "results", root / "logs"
    for directory in (jobs, results, logs):
        directory.mkdir(parents=True, exist_ok=True)
    if mode == "live":
        if not api_key:
            raise ValueError("live mode requires an API key")
        if skills_seed is None or not skills_seed.expanduser().resolve().is_dir():
            raise ValueError("live mode requires an existing frozen skills directory")
        skills_seed = skills_seed.expanduser().resolve()

    aggregate_path = root / f"aggregate-{panel['panel_id']}-w{wave}-{mode}.json"
    seen: list[dict[str, Any]] = []
    skipped = 0
    stop_reason: str | None = None
    exit_code = 0

    for task in tasks:
        key = task_key(panel["panel_id"], wave, mode, task)
        task_results = results / key
        task_results.mkdir(exist_ok=True)
        prior = _completed_attempt(task_results, root)
        if prior is None and continue_on_task_failure:
            prior = _latest_trial_failure(task_results, root)
        if prior is not None:
            seen.append(prior)
            skipped += 1
            _write_aggregate(aggregate_path, panel, wave, mode, tasks, seen, skipped, None)
            print(f"[{len(seen)}/{len(tasks)}] skip {task['instance_id']}", flush=True)
            continue

        guard_root = _disk_guard_root(root)
        try:
            free = disk_usage(guard_root).free
        except OSError:
            free = 0
        if free < MIN_FREE_BYTES:
            stop_reason, exit_code = "disk_guard", 73
            _write_aggregate(aggregate_path, panel, wave, mode, tasks, seen, skipped, stop_reason)
            print(f"stop: {guard_root} has less than 20 GiB free", flush=True)
            break

        attempt = _next_attempt(task_results, jobs, key)
        name = f"{key}-attempt-{attempt:04d}"
        result_path = task_results / f"attempt-{attempt:04d}.json"
        command = harbor_command(
            task, jobs_dir=jobs, job_name=name, mode=mode,
            skills_seed=skills_seed, extra_docker_compose=extra_docker_compose,
            max_model_calls=trial_budget["max_model_calls"] if trial_budget else None,
            max_tool_calls=trial_budget["max_tool_calls"] if trial_budget else None,
            model_call_timeout_seconds=(
                trial_budget.get("model_call_timeout_seconds") if trial_budget else None
            ),
            agent_timeout_multiplier=(
                trial_budget.get("agent_timeout_multiplier") if trial_budget else None
            ),
            environment_build_timeout_multiplier=(
                trial_budget.get("environment_build_timeout_multiplier")
                if trial_budget else None
            ),
            thinking=panel.get("thinking", True),
            provider=provider,
            executable=executable,
        )
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(REPO_ROOT) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        if api_key is not None:
            env[SECRET_ENV] = api_key
        print(f"[{len(seen) + 1}/{len(tasks)}] run {task['instance_id']}", flush=True)
        try:
            completed = run(
                command, env=env, text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            output = completed.stdout or ""
            returncode = int(completed.returncode)
        except OSError as exc:
            output, returncode = f"{type(exc).__name__}: {exc}\n", 127

        safe_output = output.replace(api_key, "[REDACTED]") if api_key else output
        (logs / f"{name}.log").write_text(safe_output, encoding="utf-8", newline="\n")
        harbor_result = jobs / name / "result.json"
        harbor_text = harbor_result.read_text(encoding="utf-8", errors="replace") if harbor_result.is_file() else ""
        quota = _is_quota_failure(output) or _is_quota_failure(harbor_text)
        complete, harbor_summary = _harbor_summary(harbor_result)
        status = "completed" if returncode == 0 and complete else "failed"
        if quota and status != "completed":
            status = "quota_failure"
        record = {
            "schema_version": 1,
            "panel_id": panel["panel_id"],
            "trial_budget": trial_budget,
            "wave": wave,
            "mode": mode,
            "suite": task["suite"],
            "instance_id": task["instance_id"],
            "attempt": attempt,
            "status": status,
            "returncode": returncode,
            "harbor_result": str(harbor_result.relative_to(root)) if harbor_result.is_file() else None,
            "harbor_summary": harbor_summary,
        }
        _write_attempt(result_path, record)
        seen.append(record)
        if quota:
            exit_code = returncode or 1
            stop_reason = "quota"
        elif status != "completed":
            exit_code = returncode or 1
            if returncode != 0 or harbor_summary["integrity"] in {
                "missing_or_invalid", "incomplete",
            }:
                stop_reason = "harness_failure"
            elif not continue_on_task_failure:
                stop_reason = "harbor_failure"
        _write_aggregate(aggregate_path, panel, wave, mode, tasks, seen, skipped, stop_reason)
        print(f"[{len(seen)}/{len(tasks)}] {status} {task['instance_id']}", flush=True)
        if stop_reason:
            break

    return json.loads(aggregate_path.read_text(encoding="utf-8")) | {"exit_code": exit_code}


def _is_quota_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in ("quota", "daily limit", "额度"))


def _harbor_summary(path: Path) -> tuple[bool, dict[str, Any]]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        stats = result["stats"]
    except (OSError, ValueError, KeyError, TypeError):
        return False, {"integrity": "missing_or_invalid"}
    counts = {
        "n_total_trials": result.get("n_total_trials"),
        "n_completed_trials": stats.get("n_completed_trials"),
        "n_errored_trials": stats.get("n_errored_trials"),
    }
    complete = counts == {
        "n_total_trials": 1, "n_completed_trials": 1, "n_errored_trials": 0,
    }
    trial_error = counts == {
        "n_total_trials": 1, "n_completed_trials": 1, "n_errored_trials": 1,
    }
    evals = stats.get("evals") or {}
    first_eval = next(iter(evals.values()), {}) if isinstance(evals, dict) else {}
    return complete, {
        "integrity": "complete" if complete else "trial_error" if trial_error else "incomplete",
        **counts,
        "input_tokens": stats.get("n_input_tokens"),
        "cache_tokens": stats.get("n_cache_tokens"),
        "output_tokens": stats.get("n_output_tokens"),
        "cost_usd": stats.get("cost_usd"),
        "reward_metrics": first_eval.get("metrics") if isinstance(first_eval, dict) else None,
        "pass_at_k": first_eval.get("pass_at_k") if isinstance(first_eval, dict) else None,
    }


def _completed_attempt(
    task_results: Path, root: Path,
) -> dict[str, Any] | None:
    for record_path in sorted(task_results.glob("attempt-*.json"), reverse=True):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        harbor = record.get("harbor_result")
        if record.get("status") == "completed" and harbor:
            complete, summary = _harbor_summary(root / harbor)
            if complete:
                record["harbor_summary"] = summary
                return record
    return None


def _latest_trial_failure(task_results: Path, root: Path) -> dict[str, Any] | None:
    for record_path in sorted(task_results.glob("attempt-*.json"), reverse=True):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        harbor = record.get("harbor_result")
        if record.get("status") == "failed" and harbor:
            _, summary = _harbor_summary(root / harbor)
            if summary.get("integrity") == "trial_error":
                record["harbor_summary"] = summary
                return record
    return None


def _next_attempt(task_results: Path, jobs: Path, key: str) -> int:
    numbers = [int(path.stem.rsplit("-", 1)[1]) for path in task_results.glob("attempt-*.json")]
    numbers += [int(path.name.rsplit("-", 1)[1]) for path in jobs.glob(f"{key}-attempt-*")]
    return max(numbers, default=0) + 1


def _disk_guard_root(root: Path, *, is_wsl: bool | None = None) -> Path:
    if is_wsl is None:
        try:
            is_wsl = "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
        except OSError:
            is_wsl = False
    if is_wsl:
        return Path("/mnt/d")
    return Path("D:/") if Path("D:/").exists() else Path(root.anchor)


def _write_aggregate(
    path: Path, panel: dict[str, Any], wave: int, mode: str,
    tasks: list[dict[str, str]], seen: list[dict[str, Any]], skipped: int,
    stop_reason: str | None,
) -> None:
    completed = sum(item["status"] == "completed" for item in seen)
    failed = sum(item["status"] != "completed" for item in seen)
    _atomic_json(path, {
        "schema_version": 1,
        "panel_id": panel["panel_id"],
        "trial_budget": panel.get("trial_budget"),
        "wave": wave,
        "mode": mode,
        "updated_at": datetime.now(UTC).isoformat(),
        "counts": {
            "selected": len(tasks), "completed": completed, "failed": failed,
            "skipped": skipped, "pending": len(tasks) - len(seen),
        },
        "stop_reason": stop_reason,
        "results": seen,
    })


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_attempt(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite attempt record: {path}")
    _atomic_json(path, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--wave", type=int, choices=(1, 2), required=True)
    parser.add_argument("--runs-root", type=experiment_path, default=EXPERIMENT_ROOT / "runs")
    parser.add_argument("--skills-seed", type=Path)
    parser.add_argument("--harbor", default="harbor")
    parser.add_argument("--extra-docker-compose", type=Path)
    parser.add_argument("--continue-on-task-failure", action="store_true")
    parser.add_argument("--provider", choices=("amd", "commandcode"), default="amd")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", choices=("oracle", "nop"))
    mode.add_argument("--live", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "live" if args.live else args.preflight
    if args.live and (args.skills_seed is None or not args.skills_seed.expanduser().is_dir()):
        _parser().error("--live requires an existing --skills-seed directory")
    key = getpass.getpass(f"{args.provider} API key (not stored): ") if args.live else None
    summary = run_panel(
        panel_path=args.panel, wave=args.wave, runs_root=args.runs_root, mode=mode,
        skills_seed=args.skills_seed, api_key=key, executable=args.harbor,
        extra_docker_compose=args.extra_docker_compose,
        continue_on_task_failure=args.continue_on_task_failure,
        provider=args.provider,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
