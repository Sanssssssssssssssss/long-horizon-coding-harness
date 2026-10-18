"""Sequential, file-backed benchmark execution."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = 1
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent / "experiment"
_MAX_INFRASTRUCTURE_RETRIES = 1
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IDENTITY_FIELDS = (
    "harness_commit", "harness_dirty_state", "model_identifier", "provider",
    "reasoning_setting", "temperature", "tool_schema_hash", "system_prompt_hash",
    "api_base_identifier", "api_base_sha256", "provider_revision",
    "suite_revision", "evaluator_revision", "docker_image_digest", "network_policy",
    "cpu_limit", "memory_limit", "timeouts", "random_seed", "price_table_revision",
)


def experiment_path(value: str | Path) -> Path:
    """Keep benchmark CLI outputs in this checkout's artifact directory."""
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError(f"output must stay under {EXPERIMENT_ROOT}")
    return path


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Frozen task definition whose private evaluator data never reaches the Agent."""

    task_id: str
    repository: str | Path
    base_commit: str
    problem_statement: str
    container_image: str | None = None
    platform: str | None = None
    timeout_seconds: float | None = None
    network_policy: str | None = None
    evaluator_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _safe_id("task_id", self.task_id)
        if not self.base_commit.strip() or not self.problem_statement.strip():
            raise ValueError("base_commit and problem_statement must not be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        metadata = _freeze(self.evaluator_metadata)
        json.dumps(_thaw(metadata))
        object.__setattr__(self, "repository", str(self.repository))
        object.__setattr__(self, "evaluator_metadata", MappingProxyType(metadata))

    def agent_view(self) -> Mapping[str, Any]:
        # Source paths, revisions, and evaluator data stay outside the Agent context.
        return MappingProxyType({
            "task_id": self.task_id,
            "problem_statement": self.problem_statement,
            "platform": self.platform,
            "timeout_seconds": self.timeout_seconds,
            "network_policy": self.network_policy,
        })

    def record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "repository": self.repository,
            "base_commit": self.base_commit, "problem_statement": self.problem_statement,
            "container_image": self.container_image, "platform": self.platform,
            "timeout_seconds": self.timeout_seconds, "network_policy": self.network_policy,
            "evaluator_metadata": _thaw(self.evaluator_metadata),
        }


class BenchmarkRunner:
    """Run one deterministic attempt per task; phase one is deliberately sequential.

    ``agent`` is async and receives ``task``, ``workdir``, ``trace``, and ``env``.
    It must construct a fresh Agent per call with product persistence/injection off.
    The evaluator receives the full ``TaskSpec``, ``workdir``, and ``env``.
    """

    def __init__(
        self, *, runs_root: str | Path, suite_name: str, run_id: str,
        tasks: Sequence[TaskSpec], agent: Callable[..., Awaitable[Any]],
        evaluator: Callable[..., Any], worker_count: int = 1,
        profile: str = "B0",
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        _safe_id("suite_name", suite_name)
        _safe_id("run_id", run_id)
        if worker_count != 1:
            raise ValueError("phase-one BenchmarkRunner requires worker_count=1")
        if profile not in {"B0", "B1"}:
            raise ValueError("profile must be B0 or B1")
        if not tasks or len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("tasks must be non-empty and have unique task_id values")
        if not callable(agent):
            raise TypeError("agent must be callable")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")

        self.runs_root = Path(runs_root).expanduser().resolve()
        self.suite_name, self.run_id = suite_name, run_id
        self.profile = profile
        self.tasks = tuple(tasks)
        self.agent, self.evaluator = agent, evaluator
        self.manifest = _merge_records(
            dict(manifest or {}), _callback_record(agent, "benchmark_manifest"), "manifest"
        )
        self.agent_config = _callback_record(agent, "benchmark_config")
        reserved = {"schema_version", "suite_name", "run_id", "task_list_hash", "worker_count"}
        if reserved & self.manifest.keys():
            raise ValueError(f"manifest cannot override {sorted(reserved)}")
        config_reserved = {
            "schema_version", "profile", "worker_count", "attempts_per_task",
            "online_evolution", "mcp", "memory", "session_persistence",
        }
        if config_reserved & self.agent_config.keys():
            raise ValueError(f"agent config cannot override {sorted(config_reserved)}")
        json.dumps({"manifest": self.manifest, "agent_config": self.agent_config})
        self.run_dir = self.runs_root / suite_name / run_id
        self.repositories = {
            task.task_id: _repository(task.repository, task.base_commit) for task in self.tasks
        }
        for repository in self.repositories.values():
            if _within(self.run_dir, repository) or _within(repository, self.run_dir):
                raise ValueError("run_dir must stay outside and not contain the target repository/worktree")

    async def run(self) -> dict[str, Any]:
        self._prepare_run()
        for task in self.tasks:
            await self._run_task(task)
        summary = self._aggregate()
        _write_artifact_index(self.run_dir)
        return summary

    def _prepare_run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.runs_root / ".scratch").mkdir(exist_ok=True)
        (self.run_dir / "attempts").mkdir(exist_ok=True)
        (self.run_dir / "blobs").mkdir(exist_ok=True)
        tasks_text = "".join(_json(task.record()) + "\n" for task in self.tasks)
        commit, dirty = _harness_identity()
        identity = dict.fromkeys(_IDENTITY_FIELDS)
        identity.update(self.manifest)
        identity.update({
            "schema_version": SCHEMA_VERSION, "suite_name": self.suite_name,
            "run_id": self.run_id,
            "harness_commit": identity["harness_commit"] or commit,
            "harness_dirty_state": dirty if identity["harness_dirty_state"] is None else identity["harness_dirty_state"],
            "task_list_hash": hashlib.sha256(tasks_text.encode()).hexdigest(),
            "worker_count": 1,
        })
        _immutable(self.run_dir / "manifest.json", (_json(identity) + "\n").encode())
        live_config = self.agent_config.get("live_agent", {})
        online_evolution = (
            live_config.get("online_evolution", self.profile == "B1")
            if isinstance(live_config, Mapping) else self.profile == "B1"
        )
        config = {
            "schema_version": SCHEMA_VERSION, "profile": self.profile, "worker_count": 1,
            "attempts_per_task": 1,
            "online_evolution": online_evolution,
            "mcp": self.profile == "B1",
            "memory": self.profile == "B1",
            "session_persistence": self.profile == "B1",
            "infrastructure_retries_on_resume": _MAX_INFRASTRUCTURE_RETRIES,
        }
        config.update(self.agent_config)
        _immutable(self.run_dir / "config.lock.json", (_json(config) + "\n").encode())
        _immutable(self.run_dir / "tasks.jsonl", tasks_text.encode())

    async def _run_task(self, task: TaskSpec) -> dict[str, Any]:
        attempts_root = self.run_dir / "attempts"
        for number in range(1, _MAX_INFRASTRUCTURE_RETRIES + 2):
            attempt_id = f"attempt-{number:04d}"
            attempt = attempts_root / task.task_id / attempt_id
            completed = _complete(attempt / "result.json", task.task_id, attempt_id)
            if completed is None:
                break
            if completed["status"] != "infrastructure_failure" or number > _MAX_INFRASTRUCTURE_RETRIES:
                return completed
        _reset(attempt, attempts_root)

        evaluator_dir = attempt / "evaluator"
        for directory in (attempt, evaluator_dir):
            directory.mkdir(parents=True, exist_ok=True)
        stdout, stderr = attempt / "stdout.log", attempt / "stderr.log"
        eval_stdout, eval_stderr = evaluator_dir / "test-output.log", evaluator_dir / "eval.log"
        patch_path, report_path = attempt / "final.patch", evaluator_dir / "report.json"
        for path in (stdout, stderr, eval_stdout, eval_stderr, patch_path):
            path.touch()

        from agents.telemetry import TraceRecorder

        trace = TraceRecorder(
            attempt, blob_dir=self.run_dir / "blobs", run_id=self.run_id,
            task_id=task.task_id, attempt_id=attempt_id,
        )
        source = self.repositories[task.task_id]
        started_ns, started_utc = time.monotonic_ns(), _utc()
        timing: dict[str, float | None] = {
            "environment_prepare_ms": None, "attempt_wall_ms": None, "evaluator_ms": None,
        }
        status, agent_status = "completed", "not_started"
        infrastructure_failure: dict[str, str] | None = None
        termination_reason: dict[str, str] | None = None
        environment_ready = False
        agent_value: Any = None
        evaluation: dict[str, Any] = {"resolved": None, "status": "not_run"}
        patch = b""

        prepare_started = time.monotonic_ns()
        try:
            with tempfile.TemporaryDirectory(prefix=f"benchmark-agent-{task.task_id}-", dir=self.runs_root / ".scratch") as raw:
                scratch = Path(raw).resolve()
                if _within(scratch, self.run_dir) or _within(scratch, source):
                    raise RuntimeError("Agent scratch directory overlaps a run or target repository")
                workspace, home = scratch / "worktree", scratch / "home"
                home.mkdir()
                with trace.span("environment.prepare", component="benchmark_runner", purpose="main"):
                    _checkout(source, task.base_commit, workspace, _environment(home, task, self.profile))
                environment_ready = True
                timing["environment_prepare_ms"] = _elapsed(prepare_started)

                agent_started = time.monotonic_ns()
                try:
                    with (
                        _context(home, workspace, task, self.profile) as env,
                        stdout.open("a", encoding="utf-8") as out,
                        stderr.open("a", encoding="utf-8") as err,
                        contextlib.redirect_stdout(out), contextlib.redirect_stderr(err),
                        trace.span("agent.run", component="benchmark_runner", purpose="main"),
                    ):
                        result = self.agent(task=task.agent_view(), workdir=workspace, trace=trace, env=env)
                        if not inspect.isawaitable(result):
                            raise TypeError("agent must return an awaitable")
                        agent_value = await (result if task.timeout_seconds is None else asyncio.wait_for(result, task.timeout_seconds))
                    termination_reason = _reported_termination(agent_value)
                    if termination_reason is None:
                        agent_status = "completed"
                    else:
                        agent_status, status = "terminated", "task_failure"
                except TimeoutError:
                    agent_status = "timeout"
                    status = "task_failure"
                    termination_reason = _termination("timeout")
                    _append(stderr, "Agent timed out.\n")
                except Exception as exc:
                    agent_status = "error"
                    if type(exc).__name__ == "TraceAcceptanceError":
                        status = "infrastructure_failure"
                        infrastructure_failure = _infra_failure("agent", kind="trace_acceptance")
                    else:
                        status = "task_failure"
                        termination_reason = _agent_termination(exc)
                    _exception(stderr)
                timing["attempt_wall_ms"] = _elapsed(agent_started)

                try:
                    patch = _export_patch(workspace, _environment(home, task, self.profile))
                    _atomic(patch_path, patch)
                except Exception:
                    if agent_status == "completed":
                        agent_status = "patch_export_error"
                    status = "infrastructure_failure"
                    infrastructure_failure = infrastructure_failure or _infra_failure(
                        "patch_export", kind="patch_export"
                    )
                    _exception(stderr)
        except Exception as exc:
            status = "infrastructure_failure"
            agent_status = "infrastructure_failure"
            infrastructure_failure = _infra_failure("environment_prepare", exc)
            _exception(stderr)

        if environment_ready:
            eval_started = time.monotonic_ns()
            try:
                with tempfile.TemporaryDirectory(prefix=f"benchmark-evaluator-{task.task_id}-", dir=self.runs_root / ".scratch") as raw:
                    scratch = Path(raw).resolve()
                    if _within(scratch, self.run_dir) or _within(scratch, source):
                        raise RuntimeError("Evaluator scratch directory overlaps a run or target repository")
                    eval_workspace, evaluator_home = scratch / "worktree", scratch / "home"
                    evaluator_home.mkdir()
                    eval_env = _environment(evaluator_home, task, self.profile)
                    with trace.span("evaluator.prepare", component="benchmark_runner", purpose="main"):
                        _checkout(source, task.base_commit, eval_workspace, eval_env)
                        applied = _apply_patch(eval_workspace, patch_path, eval_env, eval_stderr)
                    if applied:
                        with trace.span("evaluator.run", component="evaluator", purpose="evaluator_assist"):
                            evaluation = await self._evaluate(
                                task, eval_workspace, evaluator_home, eval_stdout, eval_stderr
                            )
                    else:
                        evaluation = {"resolved": False, "status": "invalid_patch"}
                        if infrastructure_failure is None:
                            status = "task_failure"
                            termination_reason = termination_reason or _termination("invalid_patch", "patch_replay")
            except Exception as exc:
                status = "infrastructure_failure"
                evaluation = {"resolved": None, "status": "evaluator_error"}
                infrastructure_failure = infrastructure_failure or _infra_failure("evaluator", exc)
                _exception(eval_stderr)
            timing["evaluator_ms"] = _elapsed(eval_started)

        if infrastructure_failure is not None:
            diagnostic = evaluation
            evaluation = {
                "resolved": False,
                "status": "infrastructure_failure",
                "infrastructure_failure": infrastructure_failure,
            }
            termination_reason = None
            if diagnostic.get("status") not in {"not_run", "infrastructure_failure"}:
                evaluation["diagnostic"] = diagnostic
        elif termination_reason is not None:
            diagnostic = evaluation
            evaluation = {
                "resolved": False,
                "status": "task_failure",
                "termination_reason": termination_reason,
            }
            if diagnostic.get("status") not in {"not_run", "task_failure"}:
                evaluation["diagnostic"] = diagnostic

        _atomic(report_path, (_json(evaluation) + "\n").encode())
        result = {
            "schema_version": SCHEMA_VERSION, "suite_name": self.suite_name,
            "run_id": self.run_id, "task_id": task.task_id, "attempt_id": attempt_id,
            "status": status, "agent_status": agent_status,
            "infrastructure_failure": infrastructure_failure,
            "termination_reason": termination_reason,
            "started_utc": started_utc, "finished_utc": _utc(),
            "final_output": _agent_output(agent_value),
            "learning": (
                agent_value.get("learning")
                if isinstance(agent_value, Mapping) else None
            ),
            "patch": {"path": "final.patch", "sha256": hashlib.sha256(patch).hexdigest(), "bytes": len(patch)},
            "evaluation": evaluation,
            "timing": {**timing, "end_to_end_ms": _elapsed(started_ns)},
        }
        # The final atomic write is the sole completion marker.
        _atomic(attempt / "result.json", (_json(result) + "\n").encode())
        return result

    async def _evaluate(
        self, task: TaskSpec, workdir: Path, home: Path, stdout: Path, stderr: Path
    ) -> dict[str, Any]:
        with (
            _context(home, workdir, task, self.profile) as env,
            stdout.open("a", encoding="utf-8") as out,
            stderr.open("a", encoding="utf-8") as err,
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err),
        ):
            value = self.evaluator(task=task, workdir=workdir, env=env)
            value = await value if inspect.isawaitable(value) else value
            return _evaluation(value)

    def _aggregate(self) -> dict[str, Any]:
        from agents.telemetry import read_trace, summarize_trace

        rows, predictions = [], []
        for task in self.tasks:
            attempts = []
            for number in range(1, _MAX_INFRASTRUCTURE_RETRIES + 2):
                attempt_id = f"attempt-{number:04d}"
                attempt = self.run_dir / "attempts" / task.task_id / attempt_id
                result = _complete(attempt / "result.json", task.task_id, attempt_id)
                if result is None:
                    break
                trace = summarize_trace(read_trace(attempt / "trace.jsonl"))
                usage = trace["provider_usage"]
                tokens = usage["total_tokens"]
                if tokens is None and all(isinstance(usage[key], (int, float)) for key in ("input_tokens", "output_tokens")):
                    tokens = usage["input_tokens"] + usage["output_tokens"]
                timing = result["timing"]
                attempts.append((attempt_id, result, {
                    "tokens": tokens, "model_calls": trace["model_calls"],
                    "tool_calls": trace["tool_calls"],
                    "failed_tool_calls": trace["failed_tool_calls"],
                    "primitive_operations": trace["primitive_operations"],
                    "no_progress_tool_calls": trace["no_progress_tool_calls"],
                    "prefix_changes": trace["prefix_changes"],
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cached_input_tokens": usage["cached_input_tokens"],
                    "cache_miss_tokens": usage["cache_miss_tokens"],
                    "cache_creation_tokens": usage["cache_creation_tokens"],
                    "reasoning_tokens": usage["reasoning_tokens"],
                    "provider_reported_cost": usage["provider_reported_cost"],
                    "wall_time_ms": timing.get("attempt_wall_ms"),
                    "end_to_end_ms": timing.get("end_to_end_ms"),
                }))
                if result["status"] != "infrastructure_failure":
                    break
            if not attempts:
                continue
            attempt_id, result, _ = attempts[-1]
            metrics = [item[2] for item in attempts]
            evaluation = result["evaluation"]
            infrastructure_failure = result["infrastructure_failure"]
            termination_reason = result["termination_reason"]
            resolved = False if infrastructure_failure is not None or termination_reason is not None else evaluation.get("resolved")
            row = {
                "task_id": task.task_id, "attempt_id": attempt_id,
                "attempt_ids": [item[0] for item in attempts], "resolved": resolved,
                "status": result["status"],
                "infrastructure_failure": infrastructure_failure,
                "infrastructure_failures": sum(item[1]["status"] == "infrastructure_failure" for item in attempts),
                "termination_reason": termination_reason,
                **{key: _strict_sum(metrics, key) for key in metrics[0]},
            }
            rows.append(row)
            predictions.append({
                "task_id": task.task_id, "attempt_id": attempt_id,
                "status": result["status"], "resolved": resolved,
                "infrastructure_failure": infrastructure_failure,
                "termination_reason": termination_reason,
                "final_output": result["final_output"], "patch": result["patch"],
            })
        summary = {
            "schema_version": SCHEMA_VERSION, "suite_name": self.suite_name,
            "run_id": self.run_id,
            "counts": {
                "tasks": len(self.tasks), "completed": len(rows),
                "resolved": sum(row["resolved"] is True for row in rows),
                "unknown_resolution": sum(row["resolved"] is None for row in rows),
                "infra_failures": sum(row["infrastructure_failures"] for row in rows),
            },
            "totals": {key: _strict_sum(rows, key) for key in (
                "tokens", "model_calls", "tool_calls", "failed_tool_calls", "primitive_operations",
                "no_progress_tool_calls", "prefix_changes", "input_tokens",
                "output_tokens", "cached_input_tokens", "cache_miss_tokens",
                "cache_creation_tokens", "reasoning_tokens", "provider_reported_cost",
                "wall_time_ms", "end_to_end_ms",
            )},
            "tasks": rows,
        }
        hit = summary["totals"]["cached_input_tokens"]
        miss = summary["totals"]["cache_miss_tokens"]
        summary["totals"]["cache_hit_rate"] = (
            hit / (hit + miss)
            if isinstance(hit, (int, float)) and isinstance(miss, (int, float)) and hit + miss > 0
            else None
        )
        _atomic(self.run_dir / "predictions.jsonl", "".join(_json(row) + "\n" for row in predictions).encode())
        _atomic(self.run_dir / "summary.json", (_json(summary) + "\n").encode())
        return summary


def _safe_id(name: str, value: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{name} must match {_ID.pattern}")


def _callback_record(callback: Callable[..., Any], method_name: str) -> dict[str, Any]:
    method = getattr(callback, method_name, None)
    if method is None:
        return {}
    if not callable(method):
        raise TypeError(f"agent {method_name} must be callable")
    value = method()
    if not isinstance(value, Mapping):
        raise TypeError(f"agent {method_name} must return a mapping")
    record = dict(value)
    json.dumps(record)
    return record


def _merge_records(
    explicit: dict[str, Any], generated: Mapping[str, Any], label: str
) -> dict[str, Any]:
    conflicts = sorted(
        key for key in explicit.keys() & generated.keys() if explicit[key] != generated[key]
    )
    if conflicts:
        raise ValueError(f"agent {label} conflicts with explicit values: {conflicts}")
    return {**explicit, **generated}


def _infra_failure(
    phase: str, exc: BaseException | None = None, *, kind: str | None = None
) -> dict[str, str]:
    if kind is None:
        kind = f"{phase}_error"
    return {"phase": phase, "kind": kind}


def _termination(kind: str, phase: str = "agent") -> dict[str, str]:
    return {"phase": phase, "kind": kind}


def _termination_kind(value: Any) -> str:
    if value is None:
        return "agent_reported_failure"
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if "input" in text and ("budget" in text or "token" in text):
        return "input_budget"
    if "turn" in text:
        return "turn_budget"
    if "cost" in text:
        return "cost_budget"
    if "timeout" in text or "timed_out" in text:
        return "timeout"
    return text or "agent_reported_failure"


def _agent_termination(exc: BaseException) -> dict[str, str]:
    detail = f"{type(exc).__name__}: {exc}"
    kind = _termination_kind(detail)
    if kind not in {"input_budget", "turn_budget", "cost_budget", "timeout"}:
        kind = "agent_crash"
    return _termination(kind)


def _reported_termination(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or value.get("termination_reason") is None:
        return None
    reason = value["termination_reason"]
    kind = reason.get("kind", reason.get("reason")) if isinstance(reason, Mapping) else reason
    return _termination(_termination_kind(kind))


def _repository(repository: str | Path, commit: str) -> Path:
    path = Path(repository).expanduser()
    if not path.is_dir():
        raise ValueError("phase one supports existing local repositories only")
    root = Path(_git(path, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
    return root


def _checkout(source: Path, commit: str, destination: Path, env: Mapping[str, str]) -> None:
    _git(destination.parent, "clone", "--quiet", "--no-local", "--no-hardlinks", str(source), str(destination), env=env)
    _git(destination, "checkout", "--quiet", "--detach", commit, env=env)
    for remote in _git(destination, "remote", env=env).stdout.splitlines():
        _git(destination, "remote", "remove", remote, env=env)
    for ref in _git(destination, "for-each-ref", "--format=%(refname)", env=env).stdout.splitlines():
        _git(destination, "update-ref", "-d", ref, env=env)
    _git(destination, "reflog", "expire", "--expire=now", "--all", env=env)
    _git(destination, "gc", "--quiet", "--prune=now", env=env)


def _export_patch(workdir: Path, env: Mapping[str, str]) -> bytes:
    _git(workdir, "add", "-N", "--", ".", env=env)
    result = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."],
        cwd=workdir, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def _apply_patch(workdir: Path, patch: Path, env: Mapping[str, str], error_log: Path) -> bool:
    if not patch.stat().st_size:
        return True
    check = subprocess.run(
        ["git", "apply", "--check", str(patch)], cwd=workdir, env=dict(env),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if check.returncode:
        _append(error_log, check.stderr.decode(errors="replace"))
        return False
    _git(workdir, "apply", str(patch), env=env)
    return True


def _git(cwd: Path, *args: str, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=dict(env) if env else None, text=True,
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _environment(home: Path, task: TaskSpec, profile: str = "B0") -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        HOME=str(home), USERPROFILE=str(home), XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"), XDG_DATA_HOME=str(home / ".local/share"),
        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
        GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1",
        PYTHONDONTWRITEBYTECODE="1",
        BENCH_TASK_ID=task.task_id,
    )
    if os.name == "nt":
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="core.longpaths", GIT_CONFIG_VALUE_0="true")
    if profile == "B0":
        env.update(
            SANS_AUTO_SKILL_EVOLUTION="0", SANS_DISABLE_MCP="1",
            SANS_DISABLE_MEMORY="1", SANS_DISABLE_SESSION_PERSISTENCE="1",
        )
    else:
        env["SANS_AUTO_SKILL_EVOLUTION"] = "1"
        for name in ("SANS_DISABLE_MCP", "SANS_DISABLE_MEMORY", "SANS_DISABLE_SESSION_PERSISTENCE"):
            env.pop(name, None)
    return env


@contextlib.contextmanager
def _context(home: Path, workdir: Path, task: TaskSpec, profile: str = "B0"):
    replacement = _environment(home, task, profile)
    previous = dict(os.environ)
    previous_cwd = Path.cwd()
    try:
        os.environ.update(replacement)
        os.chdir(workdir)
        yield dict(os.environ)
    finally:
        os.chdir(previous_cwd)
        os.environ.clear()
        os.environ.update(previous)


def _evaluation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("evaluator must return a mapping")
    result = dict(value)
    if "resolved" not in result:
        raise TypeError("evaluator result must include resolved")
    result.setdefault("status", "completed")
    if result["resolved"] is not None and not isinstance(result["resolved"], bool):
        raise TypeError("resolved must be bool or None")
    json.dumps(result)
    return result


def _agent_output(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        value = value.get("final_output", value.get("output", value.get("text")))
        return None if value is None else str(value)
    return str(value)


def _reset(attempt: Path, root: Path) -> None:
    if not attempt.exists():
        return
    if attempt == root or not _within(attempt, root) or attempt.is_symlink() or (
        hasattr(attempt, "is_junction") and attempt.is_junction()
    ):
        raise ValueError(f"refusing to reset unmanaged attempt path: {attempt}")
    # No result means no durable attempt; discard partial facts before deterministic retry.
    shutil.rmtree(attempt)


def _complete(path: Path, task_id: str, attempt_id: str) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    if result.get("schema_version") != SCHEMA_VERSION or result.get("task_id") != task_id or result.get("attempt_id") != attempt_id:
        return None
    required = {
        "suite_name", "run_id", "status", "agent_status", "started_utc",
        "finished_utc", "final_output", "patch", "evaluation", "timing",
        "infrastructure_failure", "termination_reason",
    }
    if not required <= result.keys() or not isinstance(result["patch"], dict):
        return None
    patch_record, attempt = result["patch"], path.parent
    patch_path, report_path = attempt / "final.patch", attempt / "evaluator" / "report.json"
    required_files = (
        attempt / "trace.jsonl", attempt / "stdout.log", attempt / "stderr.log",
        attempt / "evaluator" / "test-output.log", attempt / "evaluator" / "eval.log",
    )
    if (
        patch_record.get("path") != "final.patch"
        or not patch_path.is_file() or patch_path.is_symlink()
        or not report_path.is_file() or report_path.is_symlink()
        or any(not item.is_file() or item.is_symlink() for item in required_files)
    ):
        return None
    try:
        patch = patch_path.read_bytes()
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    infrastructure_failure = result["infrastructure_failure"]
    termination_reason = result["termination_reason"]
    if (
        type(patch_record.get("bytes")) is not int
        or patch_record["bytes"] != len(patch)
        or patch_record.get("sha256") != hashlib.sha256(patch).hexdigest()
        or not isinstance(result["evaluation"], dict)
        or (infrastructure_failure is not None and not isinstance(infrastructure_failure, dict))
        or (termination_reason is not None and not isinstance(termination_reason, dict))
        or (infrastructure_failure is not None and termination_reason is not None)
        or report != result["evaluation"]
        or not isinstance(result["timing"], dict)
    ):
        return None
    if infrastructure_failure is not None and (
        set(infrastructure_failure) != {"phase", "kind"}
        or not all(isinstance(value, str) and value for value in infrastructure_failure.values())
        or result["status"] != "infrastructure_failure"
        or result["evaluation"].get("resolved") is not False
        or result["evaluation"].get("status") != "infrastructure_failure"
        or result["evaluation"].get("infrastructure_failure") != infrastructure_failure
    ):
        return None
    if termination_reason is not None and (
        set(termination_reason) != {"phase", "kind"}
        or not all(isinstance(value, str) and value for value in termination_reason.values())
        or result["status"] != "task_failure"
        or result["evaluation"].get("resolved") is not False
        or result["evaluation"].get("status") != "task_failure"
        or result["evaluation"].get("termination_reason") != termination_reason
    ):
        return None
    if infrastructure_failure is None and termination_reason is None and result["status"] != "completed":
        return None
    return result


def _harness_identity() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[1]
    commit: str | None = None
    dirty: bool | None = None
    safe = f"safe.directory={root.as_posix()}"
    try:
        commit = _git(root, "-c", safe, "rev-parse", "HEAD").stdout.strip()
    except (OSError, RuntimeError):
        pass
    try:
        dirty = bool(
            _git(root, "-c", safe, "status", "--porcelain", "--untracked-files=all").stdout
        )
    except (OSError, RuntimeError):
        pass
    return commit, dirty


def _strict_sum(rows: Sequence[Mapping[str, Any]], key: str) -> int | float | None:
    values = [row.get(key) for row in rows]
    if not values or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    return sum(values)


def _write_artifact_index(run_dir: Path) -> None:
    artifacts = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.name == "artifacts.json":
            continue
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise RuntimeError(f"run artifacts must not contain links: {path}")
        if not path.is_file():
            continue
        data = path.read_bytes()
        artifacts.append({
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    _atomic(run_dir / "artifacts.json", (_json({"artifacts": artifacts}) + "\n").encode())


def _within(path: Path, parent: Path) -> bool:
    path, parent = path.resolve(), parent.resolve()
    try:
        return os.path.commonpath((os.path.normcase(str(path)), os.path.normcase(str(parent)))) == os.path.normcase(str(parent))
    except ValueError:
        return False


def _immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable run artifact does not match: {path.name}")
    else:
        _atomic(path, data)


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _exception(path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        traceback.print_exc(file=handle)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _elapsed(started_ns: int) -> float:
    return round((time.monotonic_ns() - started_ns) / 1_000_000, 3)


def _utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
