"""Harbor external-agent bridge for the instrumented Sans runtime."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import math
import os
import posixpath
import shlex
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from agents.tool_result import PytestReport, ShellResult
from .runner import EXPERIMENT_ROOT

try:  # Keep the repository testable before the pinned Harbor runtime is installed.
    from harbor.agents.base import BaseAgent as _BaseAgent
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by this workspace
    if exc.name != "harbor":
        raise

    class _BaseAgent:  # type: ignore[no-redef]
        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            *,
            extra_env: dict[str, str] | None = None,
            **_: Any,
        ) -> None:
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self._extra_env = dict(extra_env or {})
            self.session_id = None
            self.context_id = None


from agents.telemetry import TraceRecorder, summarize_trace
from agents.tools import format_file_window
from benchmarks.live import LiveAgentAdapter, validate_trace


API_KEY_ENV = "SANS_BENCHMARK_API_KEY"
PRIMITIVE_NAMES = (
    "read_file", "write_file", "edit_file", "list_files", "grep_search", "run_shell",
)
HARBOR_TASK_PROMPT = """You are working in an isolated Linux benchmark task.
All provided file and shell tools operate inside the task container, and the current working
directory is already the repository root. Paths or versions mentioned in the issue may be
historical and are not evidence about this container. Prefer the repository-declared runner
and interpreter; do not pip install without evidence of a missing dependency. Start with the
smallest focused reproduction of the reported behavior, reuse what earlier reads and failures
already established, and avoid repeated reads or one-off probes that do not change the next
decision. Once you have a concrete cause and patch location, edit promptly, make the smallest
correct change, and use focused checks to refine it. Stop once the required checks are clean;
the stop hook owns the final diff. Never issue more than 8 tool calls in one response; dependent calls must wait for prior
observations. Do not access network services or hidden evaluator data.
"""
_TESTBED_BIN = "/opt/miniconda3/envs/testbed/bin"
_FULL_DIFF_CHECK = """set -eu
index=$(mktemp)
trap 'rm -f "$index"' EXIT
GIT_INDEX_FILE="$index" git read-tree HEAD
GIT_INDEX_FILE="$index" git add -A -- .
GIT_INDEX_FILE="$index" git diff --cached --check HEAD --"""


def _bool(value: bool | str, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _optional_number(value: Any, name: str, kind: type[int] | type[float]) -> Any:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none"}):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive")
    try:
        parsed = kind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _price(value: Any, name: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if parsed < 0 or not math.isfinite(parsed):
        raise ValueError(f"{name} must be a non-negative number")
    return parsed


class EnvironmentTools:
    """The six primitive tools, backed only by Harbor's BaseEnvironment API."""

    def __init__(
        self, environment: Any, workdir: str, shell_path_prefix: str | None = None
    ) -> None:
        self.environment = environment
        self.workdir = workdir
        self.shell_path_prefix = shell_path_prefix
        self._read_state: dict[str, str] = {}
        self.scratch_dir = EXPERIMENT_ROOT / "cache" / "harbor"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, raw: object) -> str:
        value = str(raw or ".")
        if "\0" in value:
            raise ValueError("path must not contain NUL")
        return posixpath.normpath(value if value.startswith("/") else posixpath.join(self.workdir, value))

    async def __call__(self, name: str, inp: dict[str, Any]) -> str | ShellResult:
        handler = getattr(self, f"_{name}", None)
        if name not in PRIMITIVE_NAMES or not callable(handler):
            return f"Unknown tool: {name}"
        try:
            return await handler(inp)
        except Exception as error:
            if name == "run_shell":
                # Harbor wraps TimeoutError in RuntimeError; retain the typed cause.
                timed_out = any(isinstance(item, TimeoutError) for item in (error, error.__cause__, error.__context__))
                prefix = "Command timed out" if timed_out else "Error executing run_shell"
                return ShellResult(timed_out=timed_out, error=f"{prefix}: {error}")
            return f"Error executing {name}: {error}"

    async def _raw_file(self, path: str) -> tuple[str | None, str | None]:
        result = await self.environment.exec(
            f"cat -- {shlex.quote(path)}", cwd=self.workdir, timeout_sec=30
        )
        if result.return_code:
            return None, result.stderr or result.stdout or "file is not readable"
        text = result.stdout or ""
        return text, None

    async def _read_file(self, inp: dict[str, Any]) -> str:
        path = self._path(inp.get("file_path"))
        content, error = await self._raw_file(path)
        if error is not None or content is None:
            return f"Error reading file: {error}"
        self._read_state[path] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return format_file_window(content, inp)

    async def _unchanged_since_read(self, path: str) -> str | None:
        if path not in self._read_state:
            return "Error: You must read this file before writing or editing it."
        content, error = await self._raw_file(path)
        if error is not None or content is None:
            return f"Error reading file: {error}"
        current = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if current != self._read_state[path]:
            return "Warning: The file changed since your last read. Read it again before editing."
        return None

    async def _upload_text(self, path: str, content: str) -> str | None:
        parent = posixpath.dirname(path) or self.workdir
        mkdir = await self.environment.exec(
            f"mkdir -p -- {shlex.quote(parent)}", cwd=self.workdir, timeout_sec=30
        )
        if mkdir.return_code:
            return mkdir.stderr or mkdir.stdout or "could not create parent directory"
        with tempfile.TemporaryDirectory(prefix="sans-harbor-upload-", dir=self.scratch_dir) as temporary:
            source = Path(temporary) / "content"
            source.write_text(content, encoding="utf-8", newline="")
            await self.environment.upload_file(source, path)
        self._read_state[path] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return None

    async def _write_file(self, inp: dict[str, Any]) -> str:
        path = self._path(inp.get("file_path"))
        content = str(inp.get("content") or "")
        if await self.environment.is_file(path):
            error = await self._unchanged_since_read(path)
            if error:
                return error
        error = await self._upload_text(path, content)
        if error:
            return f"Error writing file: {error}"
        preview = "\n".join(
            f"{index:4d} | {line}" for index, line in enumerate(content.split("\n")[:30], 1)
        )
        return f"Successfully wrote to {path} ({len(content.splitlines())} lines)\n\n{preview}"

    async def _edit_file(self, inp: dict[str, Any]) -> str:
        path = self._path(inp.get("file_path"))
        error = await self._unchanged_since_read(path)
        if error:
            return error
        with tempfile.TemporaryDirectory(prefix="sans-harbor-edit-", dir=self.scratch_dir) as temporary:
            source = Path(temporary) / "source"
            await self.environment.download_file(path, source)
            content = source.read_text(encoding="utf-8", errors="replace")
        old = str(inp.get("old_string") or "")
        new = str(inp.get("new_string") or "")
        occurrences = content.count(old) if old else 0
        if occurrences != 1:
            detail = "not found" if occurrences == 0 else f"found {occurrences} times"
            return f"Error: old_string {detail} in {path}. It must match exactly once."
        updated = content.replace(old, new, 1)
        diff = "".join(difflib.unified_diff(
            content.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile=path, tofile=path,
        ))
        error = await self._upload_text(path, updated)
        if error:
            return f"Error editing file: {error}"
        return f"Successfully edited {path}\n\n{diff}"

    async def _list_files(self, inp: dict[str, Any]) -> str:
        base = self._path(inp.get("path") or ".")
        pattern = str(inp.get("pattern") or "*")
        match = (
            f"-path {shlex.quote(posixpath.join(base, pattern))}"
            if "/" in pattern else f"-name {shlex.quote(pattern)}"
        )
        while pattern.startswith("**/"):
            pattern = pattern[3:]
            alternative = f"-path {shlex.quote(posixpath.join(base, pattern))}" if "/" in pattern else f"-name {shlex.quote(pattern)}"
            match = f"\\( {match} -o {alternative} \\)"
        result = await self.environment.exec(
            f"find {shlex.quote(base)} -type f {match} "
            "-not -path '*/.git/*' -not -path '*/node_modules/*'",
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code:
            detail = result.stderr or result.stdout or "find failed"
            return f"Error listing files (exit code {result.return_code}): {detail.strip()}"
        output = result.stdout or ""
        if not output.strip():
            return "No files found matching the pattern."
        prefix = base.rstrip("/") + "/"
        paths = [
            line[len(prefix):] if line.startswith(prefix) else line
            for line in output.splitlines()
        ]
        suffix = f"\n[... {len(paths) - 200} more files omitted ...]" if len(paths) > 200 else ""
        return "\n".join(paths[:200]) + suffix

    async def _grep_search(self, inp: dict[str, Any]) -> str:
        path = self._path(inp.get("path") or ".")
        include = inp.get("include")
        include_flag = f" --include={shlex.quote(str(include))}" if include else ""
        result = await self.environment.exec(
            f"grep -R -n -E{include_flag} -- {shlex.quote(str(inp.get('pattern') or ''))} "
            f"{shlex.quote(path)}",
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code == 1:
            return "No matches found."
        if result.return_code:
            detail = result.stderr or result.stdout or "grep failed"
            return f"Error searching files (exit code {result.return_code}): {detail.strip()}"
        return (result.stdout or result.stderr or "No matches found.").strip()

    async def _run_shell(self, inp: dict[str, Any]) -> ShellResult:
        timeout_ms = inp.get("timeout", 30_000)
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, (int, float)) or timeout_ms <= 0:
            return ShellResult(error="Error: timeout must be a positive number of milliseconds")
        command = str(inp.get("command") or "")
        if inp.get("_pytest_report"):
            report_path = f"/tmp/sans-pytest-{uuid.uuid4().hex}.xml"
            result = await self("run_shell", {
                **inp, "_pytest_report": False,
                "command": f"{command} --junitxml={shlex.quote(report_path)}",
            })
            try:
                # Read through exec: Docker archive downloads cannot read tmpfs mounts.
                report = await self.environment.exec(f"cat -- {shlex.quote(report_path)}", cwd=self.workdir, timeout_sec=30)
                parsed = PytestReport.parse(report.stdout or "") if report.return_code == 0 else None
                return replace(result, pytest_report=parsed)
            except Exception:
                return result  # Missing reports cannot establish a pytest outcome.
            finally:
                with contextlib.suppress(Exception):
                    await self.environment.exec(f"rm -f -- {shlex.quote(report_path)}", cwd=self.workdir, timeout_sec=30)
        if self.shell_path_prefix:
            command = f"export PATH={shlex.quote(self.shell_path_prefix)}:$PATH\n{command}"
        result = await self.environment.exec(
            f"bash -e -o pipefail -c {shlex.quote(command)}",
            cwd=self.workdir,
            timeout_sec=max(1, math.ceil(timeout_ms / 1000)),
        )
        return ShellResult(result.stdout or "", result.stderr or "", result.return_code)

    async def repo_state(self) -> str | None:
        command = (
            "git rev-parse --is-inside-work-tree >/dev/null 2>&1 && "
            "{ git rev-parse --verify HEAD 2>/dev/null || printf '<unborn>'; "
            "git diff --binary HEAD -- 2>/dev/null || git diff --binary --; "
            "git ls-files --others --exclude-standard -z | sort -z | "
            "xargs -0 -r sha256sum; } | sha256sum"
        )
        result = await self.environment.exec(command, cwd=self.workdir, timeout_sec=30)
        if result.return_code or not (result.stdout or "").strip():
            return None
        return (result.stdout or "").split()[0]


class _HarborLiveAdapter(LiveAgentAdapter):
    def _save_learning_snapshot(self, trace: TraceRecorder, evolution_result: Any) -> dict[str, Any] | None:
        if not self.online_evolution:
            return None
        return super()._save_learning_snapshot(trace, evolution_result)


@contextlib.contextmanager
def _isolated_home(root: Path):
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    previous_env = dict(os.environ)
    previous_cwd = Path.cwd()
    try:
        os.environ.update(
            HOME=str(home), USERPROFILE=str(home),
            XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"),
            XDG_DATA_HOME=str(home / ".local" / "share"),
            GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
            GIT_TERMINAL_PROMPT="0", SANS_DISABLE_MCP="1", SANS_DISABLE_MEMORY="1",
            SANS_DISABLE_SESSION_PERSISTENCE="1",
        )
        os.chdir(home)
        yield home
    finally:
        os.chdir(previous_cwd)
        os.environ.clear()
        os.environ.update(previous_env)


def _calculated_cost(
    usage: Mapping[str, Any], *, hit_price: float | None,
    miss_price: float | None, output_price: float | None,
) -> float | None:
    reported = usage.get("provider_reported_cost")
    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
        return float(reported)
    if hit_price is None or miss_price is None or output_price is None:
        return None
    input_tokens = usage.get("input_tokens")
    cached = usage.get("cached_input_tokens")
    missed = usage.get("cache_miss_tokens")
    output = usage.get("output_tokens")
    if not all(type(value) is int and value >= 0 for value in (input_tokens, cached, output)):
        return None
    if missed is None:
        missed = max(0, input_tokens - cached)
    if type(missed) is not int or missed < 0:
        return None
    return (cached * hit_price + missed * miss_price + output * output_price) / 1_000_000


class SansAgent(_BaseAgent):
    """Harbor 0.22 External Agent that keeps model I/O on the host and tools in-container."""

    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        api_base: str,
        provider_revision: str = "openai-compatible-v1",
        thinking: bool | str = False,
        skills_seed: str | Path | None = None,
        online_evolution: bool | str = False,
        context_folding: bool | str = False,
        max_turns: int | str | None = None,
        max_model_calls: int | str | None = None,
        max_tool_calls: int | str | None = None,
        max_input_tokens: int | str | None = None,
        max_output_tokens: int | str | None = None,
        model_call_timeout_seconds: float | str | None = None,
        max_cost_usd: float | str | None = None,
        cache_hit_price: float | str | None = None,
        cache_miss_price: float | str | None = None,
        output_price: float | str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        if "api_key" in kwargs or API_KEY_ENV in kwargs:
            raise ValueError(f"{API_KEY_ENV} must not be supplied through agent config")
        if extra_env and API_KEY_ENV in extra_env:
            raise ValueError(f"{API_KEY_ENV} must not be supplied through agent config")
        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must not be empty")
        if not api_base or not str(api_base).strip():
            raise ValueError("api_base must not be empty")
        super().__init__(
            logs_dir=Path(logs_dir), model_name=str(model_name), extra_env=extra_env, **kwargs
        )
        self.api_base = str(api_base)
        self.provider_revision = str(provider_revision)
        self.thinking = _bool(thinking, "thinking")
        self.online_evolution = _bool(online_evolution, "online_evolution")
        self.context_folding = _bool(context_folding, "context_folding")
        self.skills_seed = None if skills_seed is None else Path(skills_seed).expanduser().resolve()
        self.max_turns = _optional_number(max_turns, "max_turns", int)
        self.max_model_calls = _optional_number(max_model_calls, "max_model_calls", int)
        self.max_tool_calls = _optional_number(max_tool_calls, "max_tool_calls", int)
        self.max_input_tokens = _optional_number(max_input_tokens, "max_input_tokens", int)
        self.max_output_tokens = _optional_number(max_output_tokens, "max_output_tokens", int)
        self.model_call_timeout_seconds = _optional_number(
            model_call_timeout_seconds, "model_call_timeout_seconds", float
        )
        self.max_cost_usd = _optional_number(max_cost_usd, "max_cost_usd", float)
        self.cache_hit_price = _price(cache_hit_price, "cache_hit_price")
        self.cache_miss_price = _price(cache_miss_price, "cache_miss_price")
        self.output_price = _price(output_price, "output_price")
        self._workdir: str | None = None
        self._shell_path_prefix: str | None = None

    @staticmethod
    def name() -> str:
        return "sans"

    def version(self) -> str:
        return "0.1.3"

    async def setup(self, environment: Any) -> None:
        configured = getattr(getattr(environment, "task_env_config", None), "workdir", None)
        if configured:
            workdir = str(configured)
        else:
            result = await environment.exec("pwd", timeout_sec=10)
            if result.return_code:
                raise RuntimeError(result.stderr or result.stdout or "could not determine task workdir")
            workdir = (result.stdout or "/").strip()
        shell_path_prefix = _TESTBED_BIN if await environment.is_file(f"{_TESTBED_BIN}/python") else None
        python = f"{shell_path_prefix}/python" if shell_path_prefix else "python"
        result = await environment.exec(
            f"{shlex.quote(python)} -m pytest --version", cwd=workdir, timeout_sec=30,
        )
        if result.return_code:
            raise RuntimeError(
                "Solver requires pytest for repair verification; provision it in the task "
                f"interpreter before starting model calls. {result.stderr or result.stdout or ''}"
            )
        self._workdir = workdir
        self._shell_path_prefix = shell_path_prefix

    async def _save_final_patch(self, environment: Any) -> str | None:
        destination = self.logs_dir / "final.patch"
        remote = "/tmp/sans-final.patch"
        command = f"""set -eu
index=/tmp/sans-final.index
rm -f -- \"$index\" {remote}
trap 'rm -f -- \"$index\"' EXIT
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    GIT_INDEX_FILE=\"$index\" git read-tree HEAD
    GIT_INDEX_FILE=\"$index\" git add -A -- .
    GIT_INDEX_FILE=\"$index\" git diff --cached --binary --full-index HEAD -- > {remote}
else
    : > {remote}
fi"""
        try:
            result = await environment.exec(command, cwd=self._workdir or "/", timeout_sec=120)
            if result.return_code:
                raise RuntimeError(result.stderr or result.stdout or "could not export final patch")
            await environment.download_file(remote, destination)
        except Exception:
            destination.write_bytes(b"")
            raise
        finally:
            with contextlib.suppress(Exception):
                await environment.exec(f"rm -f -- {remote}", timeout_sec=10)
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        api_key = os.environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(f"set {API_KEY_ENV} in the host environment")
        if self._workdir is None:
            await self.setup(environment)
        tools = EnvironmentTools(
            environment, self._workdir or "/", shell_path_prefix=self._shell_path_prefix
        )
        git = await environment.exec(
            "git status --short --branch\ngit log --oneline -5",
            cwd=self._workdir or "/", timeout_sec=15,
        )
        rules = await environment.exec(
            "if [ -f CLAUDE.md ]; then cat -- CLAUDE.md; fi",
            cwd=self._workdir or "/", timeout_sec=15,
        )
        prompt_context = {
            "cwd": self._workdir or "/",
            "platform": "Linux container",
            "shell": "/bin/bash",
            "git_context": (git.stdout or git.stderr or "").strip(),
            "project_rules": (rules.stdout or "").strip(),
        }

        def agent_factory(**agent_kwargs: Any) -> Any:
            from agents.agent import Agent
            if self.context_folding:
                from benchmarks.context_folding import ContextFoldingAgent as Agent

            return Agent(
                **agent_kwargs,
                tool_executor=tools,
                repo_state_probe=tools.repo_state,
                diff_check_command=_FULL_DIFF_CHECK,
            )

        limits = []
        if self.max_model_calls is not None:
            limits.append(f"{self.max_model_calls} model calls")
        if self.max_tool_calls is not None:
            limits.append(f"{self.max_tool_calls} tool calls")
        task_prompt = HARBOR_TASK_PROMPT
        if limits:
            task_prompt += f"\nThis trial has a hard limit of {' and '.join(limits)}."
            if self.max_tool_calls is not None:
                task_prompt += (
                    " Tool requests beyond the tool-call limit are rejected, including edits."
                )
            task_prompt += "\n"
        task_prompt += f"\nBenchmark issue:\n{instruction}"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trace = TraceRecorder(
            self.logs_dir,
            run_id=str(self.context_id or getattr(environment, "context_id", None) or self.session_id or "harbor"),
            task_id=str(getattr(environment, "environment_name", None) or "task"),
            attempt_id=str(self.session_id or "attempt-0001"),
        )
        adapter_kwargs = dict(
            api_base=self.api_base,
            api_key=api_key,
            model=str(self.model_name),
            provider_revision=self.provider_revision,
            profile="B1",
            permission_mode="bypassPermissions",
            thinking=self.thinking,
            max_turns=self.max_turns,
            max_model_calls=self.max_model_calls,
            max_tool_calls=self.max_tool_calls,
            max_cost_usd=self.max_cost_usd,
            max_output_tokens=self.max_output_tokens,
            max_input_tokens=self.max_input_tokens,
            model_call_timeout_seconds=self.model_call_timeout_seconds,
            online_evolution=self.online_evolution,
            skills_seed=self.skills_seed,
            custom_tools=None,
            custom_system_prompt=None,
            agent_factory=agent_factory,
        )
        adapter = _HarborLiveAdapter(**adapter_kwargs)
        with tempfile.TemporaryDirectory(prefix="sans-harbor-home-", dir=self.logs_dir) as temporary:
            with _isolated_home(Path(temporary)):
                try:
                    result = await adapter(
                        task={
                            "task_id": str(getattr(environment, "environment_name", None) or "task"),
                            "problem_statement": task_prompt,
                            "raw_problem_statement": instruction,
                            "repo_root": self._workdir or "/",
                            "environment_facts": json.dumps(prompt_context, ensure_ascii=False),
                        },
                        workdir=Path.cwd(),
                        trace=trace,
                        env={},
                    )
                finally:
                    patch_error = await self._save_final_patch(environment)
        events = validate_trace(trace)
        summary = summarize_trace(events)
        usage = summary["provider_usage"]
        context.n_input_tokens = usage.get("input_tokens")
        context.n_cache_tokens = usage.get("cached_input_tokens")
        context.n_output_tokens = usage.get("output_tokens")
        observed_cost = _calculated_cost(
            usage,
            hit_price=self.cache_hit_price,
            miss_price=self.cache_miss_price,
            output_price=self.output_price,
        )
        cost_complete = usage.get("usage_complete") is True
        context.cost_usd = observed_cost if cost_complete else None
        adapter_config = adapter.benchmark_config()["live_agent"]
        metadata = dict(getattr(context, "metadata", None) or {})
        metadata.update({
            "provider_revision": self.provider_revision,
            "provider_usage": usage,
            "observed_cost_usd": observed_cost,
            "cost_complete": cost_complete,
            "reasoning_setting": self.thinking,
            "online_evolution": self.online_evolution,
            "skills_seed_sha256": adapter_config["skills_seed_sha256"],
            "system_prompt_hash": adapter_config["system_prompt_hash"],
            "tool_schema_hash": adapter_config["tool_schema_hash"],
            "pricing_usd_per_million": {
                "cache_hit": self.cache_hit_price,
                "cache_miss": self.cache_miss_price,
                "output": self.output_price,
            },
            "model_calls": summary["model_calls"],
            "decision_model_calls": result.get("model_calls"),
            "max_model_calls": self.max_model_calls,
            "tool_calls": summary["tool_calls"],
            "max_tool_calls": self.max_tool_calls,
            "model_call_timeout_seconds": self.model_call_timeout_seconds,
            "failed_tool_calls": summary["failed_tool_calls"],
            "prefix_changes": summary["prefix_changes"],
            "termination_reason": result.get("termination_reason"),
            "trace_file": "trace.jsonl",
            "trace_id": trace.trace_id,
            "final_patch": "final.patch",
            "final_patch_error": patch_error,
        })
        if result.get("phase_metrics") is not None:
            metadata["phase_metrics"] = result["phase_metrics"]
        context.metadata = metadata


__all__ = ["SansAgent", "EnvironmentTools"]
