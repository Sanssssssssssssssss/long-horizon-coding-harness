"""Small OpenAI-compatible bridge for live benchmark attempts."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agents.prompt import SOLVER_SYSTEM_PROMPT
from agents.repair_state import TaskEnvelope
from agents.telemetry import PROVIDER_USAGE_FIELDS, TraceRecorder, read_trace
from agents.tools import get_solver_tool_definitions


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
B0_SYSTEM_PROMPT = SOLVER_SYSTEM_PROMPT


class TraceAcceptanceError(RuntimeError):
    """The attempt trace cannot support a valid live benchmark result."""


def validate_trace(trace: TraceRecorder) -> list[dict[str, Any]]:
    """Require intact payloads and successful main calls with exact provider usage."""
    try:
        raw = trace.trace_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise TraceAcceptanceError("trace has a truncated final record")
        events = read_trace(trace.trace_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise TraceAcceptanceError(f"trace is not parseable: {exc}") from exc

    if not events:
        raise TraceAcceptanceError("trace is empty")

    previous_seq = 0
    starts: set[str] = set()
    successful_main_calls = 0
    for event in events:
        sequence = event.get("seq")
        if type(sequence) is not int or sequence <= previous_seq:
            raise TraceAcceptanceError("trace sequence is not strictly increasing")
        previous_seq = sequence

        for field_name, expected in (
            ("run_id", trace.run_id),
            ("task_id", trace.task_id),
            ("attempt_id", trace.attempt_id),
            ("trace_id", trace.trace_id),
        ):
            if event.get(field_name) != expected:
                raise TraceAcceptanceError(f"trace {field_name} does not match its recorder")

        event_type = str(event.get("event_type") or "")
        span_id = event.get("span_id")
        component = str(event.get("component") or "").lower()
        purpose = event.get("purpose")
        is_main_model = (
            (component in {"model", "llm", "provider"} or component.startswith("model."))
            and purpose == "main"
        )
        if event_type.endswith(".start") and isinstance(span_id, str) and is_main_model:
            starts.add(span_id)
        if (
            event_type.endswith(".end")
            and str(event.get("status") or "").lower() in {"ok", "success"}
            and isinstance(span_id, str)
            and span_id in starts
            and is_main_model
        ):
            _validate_provider_usage(event)
            successful_main_calls += 1

        payload_ref = event.get("payload_ref")
        if payload_ref is not None:
            _validate_payload_ref(trace.blob_dir, payload_ref)

    if not successful_main_calls:
        raise TraceAcceptanceError("trace has no successful main model terminal span")
    return events


def _validate_provider_usage(event: Mapping[str, Any]) -> None:
    metrics = event.get("metrics")
    usage = metrics.get("provider_usage") if isinstance(metrics, Mapping) else None
    if not isinstance(usage, Mapping):
        raise TraceAcceptanceError("successful main model call lacks exact provider usage")
    if any(name not in usage for name in PROVIDER_USAGE_FIELDS):
        raise TraceAcceptanceError("provider usage must explicitly contain all fields")
    if any(
        value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
        for value in (usage[name] for name in PROVIDER_USAGE_FIELDS)
    ):
        raise TraceAcceptanceError("provider usage fields must be numbers or null")
    fields = ("input_tokens", "output_tokens", "total_tokens")
    if any(type(usage[name]) is not int or usage[name] < 0 for name in fields):
        raise TraceAcceptanceError("successful main model call lacks exact provider usage")
    expected_totals = {usage["input_tokens"] + usage["output_tokens"]}
    reasoning = usage["reasoning_tokens"]
    if type(reasoning) is int and reasoning >= 0:
        expected_totals.add(usage["input_tokens"] + usage["output_tokens"] + reasoning)
    if usage["total_tokens"] not in expected_totals:
        raise TraceAcceptanceError("provider usage is not self-consistent")


def _validate_payload_ref(blob_dir: Path, payload_ref: Any) -> None:
    if not isinstance(payload_ref, Mapping):
        raise TraceAcceptanceError("trace payload_ref is not an object")
    digest = payload_ref.get("sha256")
    expected_bytes = payload_ref.get("bytes")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise TraceAcceptanceError("trace payload_ref has an invalid sha256")
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise TraceAcceptanceError("trace payload_ref has an invalid byte count")

    blob = blob_dir / digest
    if not blob.is_file() or blob.is_symlink():
        raise TraceAcceptanceError(f"trace payload blob is missing: {digest}")
    try:
        data = blob.read_bytes()
    except OSError as exc:
        raise TraceAcceptanceError(f"trace payload blob is unreadable: {digest}") from exc
    if len(data) != expected_bytes:
        raise TraceAcceptanceError(f"trace payload blob size does not match: {digest}")
    if hashlib.sha256(data).hexdigest() != digest:
        raise TraceAcceptanceError(f"trace payload blob hash does not match: {digest}")


@contextlib.contextmanager
def _without_secret_in_environment(secret: str):
    """Hide a constructor-supplied key from tools that inherit ``os.environ``."""
    removed = {key: value for key, value in os.environ.items() if secret in value}
    try:
        for key in removed:
            os.environ.pop(key, None)
        yield
    finally:
        os.environ.update(removed)


@contextlib.contextmanager
def _skill_environment(online_evolution: bool, target_skill: str | None):
    previous = {
        name: os.environ.get(name)
        for name in (
            "SANS_AUTO_SKILL_EVOLUTION", "SANS_AUTO_SKILL_TARGET",
            "SANS_SKILL_EVOLUTION_DIR", "SANS_EVOLUTION_TARGET_SKILL",
        )
    }
    try:
        os.environ["SANS_AUTO_SKILL_EVOLUTION"] = "1" if online_evolution else "0"
        os.environ["SANS_AUTO_SKILL_TARGET"] = "user"
        os.environ["SANS_SKILL_EVOLUTION_DIR"] = str(
            Path.home() / ".sans" / "skill-evolution"
        )
        if target_skill:
            os.environ["SANS_EVOLUTION_TARGET_SKILL"] = target_skill
        else:
            os.environ.pop("SANS_EVOLUTION_TARGET_SKILL", None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        _thaw_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_hash(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    if root.is_dir():
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError(f"learning state must not contain links: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().encode("utf-8")
            data = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
            count += 1
    return digest.hexdigest(), count


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"learning state must not contain links: {path}")
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _reject_secret_keys(value: Any) -> None:
    forbidden = ("auth", "key", "token", "secret", "password", "credential", "header")
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in forbidden):
                raise ValueError(f"openai_extra_body contains secret-bearing key: {key}")
            _reject_secret_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_keys(item)


def _normalize_api_base(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("api_base must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("api_base must not contain credentials, query, or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("api_base has an invalid port") from exc
    host = parsed.hostname.lower()
    host = f"[{host}]" if ":" in host else host
    default_port = (parsed.scheme.lower(), port) in {("http", 80), ("https", 443)}
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def _api_base_origin(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


@dataclass(frozen=True, slots=True)
class LiveAgentAdapter:
    """Construct one fresh instrumented Agent for each BenchmarkRunner task."""

    api_base: str = field(repr=False)
    api_key: str = field(repr=False)
    model: str
    provider_revision: str = "openai-compatible-v1"
    profile: str = "B0"
    permission_mode: str = "bypassPermissions"
    thinking: bool = False
    max_turns: int | None = 6
    max_cost_usd: float | None = None
    max_output_tokens: int | None = 4096
    max_input_tokens: int | None = 30_000
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    model_call_timeout_seconds: float | None = None
    openai_extra_body: Mapping[str, Any] | None = field(default=None, repr=False)
    openai_stream_include_usage: bool = True
    online_evolution: bool = False
    skills_seed: str | Path | None = field(default=None, repr=False)
    evolution_target_skill: str | None = None
    evolution_hint: str = ""
    custom_tools: tuple[Mapping[str, Any], ...] | None = field(default=None, repr=False)
    custom_system_prompt: str | None = field(default=None, repr=False)
    agent_factory: Callable[..., Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("api_base", "api_key", "model", "provider_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.profile not in {"B0", "B1"}:
            raise ValueError("profile must be B0 or B1")
        if self.custom_system_prompt is not None and not self.custom_system_prompt.strip():
            raise ValueError("custom_system_prompt must be non-empty or None")
        object.__setattr__(self, "api_base", _normalize_api_base(self.api_base))
        for name in (
            "max_turns", "max_output_tokens", "max_input_tokens",
            "max_model_calls", "max_tool_calls",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool) or self.max_cost_usd <= 0
        ):
            raise ValueError("max_cost_usd must be positive")
        if self.model_call_timeout_seconds is not None and (
            isinstance(self.model_call_timeout_seconds, bool)
            or not isinstance(self.model_call_timeout_seconds, (int, float))
            or self.model_call_timeout_seconds <= 0
        ):
            raise ValueError("model_call_timeout_seconds must be positive")
        if type(self.openai_stream_include_usage) is not bool:
            raise ValueError("openai_stream_include_usage must be bool")
        if type(self.online_evolution) is not bool:
            raise ValueError("online_evolution must be bool")
        if self.online_evolution and self.profile != "B1":
            raise ValueError("online_evolution requires profile B1")
        if self.evolution_target_skill is not None and (
            not isinstance(self.evolution_target_skill, str)
            or not self.evolution_target_skill.strip()
        ):
            raise ValueError("evolution_target_skill must be non-empty or None")
        if not isinstance(self.evolution_hint, str):
            raise ValueError("evolution_hint must be a string")
        if self.evolution_target_skill and not self.online_evolution:
            raise ValueError("evolution_target_skill requires online_evolution")
        if self.skills_seed is not None:
            seed = Path(self.skills_seed).expanduser().resolve()
            if self.profile != "B1" or not seed.is_dir():
                raise ValueError("skills_seed requires profile B1 and an existing directory")
            _tree_hash(seed)
            object.__setattr__(self, "skills_seed", seed)
        if self.openai_extra_body is not None:
            _reject_secret_keys(self.openai_extra_body)
            frozen_body = _freeze_json(self.openai_extra_body)
            json.dumps(_thaw_json(frozen_body))
            object.__setattr__(self, "openai_extra_body", frozen_body)
        if self.custom_tools is not None:
            frozen_tools = tuple(_freeze_json(tool) for tool in self.custom_tools)
            json.dumps(_thaw_json(frozen_tools))
            object.__setattr__(self, "custom_tools", frozen_tools)

    def _resolved_tools(self) -> tuple[Mapping[str, Any], ...]:
        if self.custom_tools is not None:
            return self.custom_tools
        return tuple(get_solver_tool_definitions(self.profile))

    def _resolved_system_prompt(self) -> str:
        return self.custom_system_prompt or SOLVER_SYSTEM_PROMPT

    def benchmark_manifest(self) -> dict[str, Any]:
        config = self.benchmark_config()["live_agent"]
        return {
            "api_base_identifier": config["api_base_identifier"],
            "api_base_sha256": config["api_base_sha256"],
            "model_identifier": self.model,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "model_call_timeout_seconds": self.model_call_timeout_seconds,
            "openai_extra_body_sha256": config["openai_extra_body_sha256"],
            "provider": "openai-compatible",
            "provider_revision": self.provider_revision,
            "reasoning_setting": self.thinking,
            "online_evolution": self.online_evolution,
            "evolution_target_skill": self.evolution_target_skill,
            "evolution_hint_sha256": hashlib.sha256(self.evolution_hint.encode("utf-8")).hexdigest(),
            "skills_seed_sha256": config["skills_seed_sha256"],
            "system_prompt_hash": config["system_prompt_hash"],
            "tool_schema_hash": config["tool_schema_hash"],
        }

    def benchmark_config(self) -> dict[str, Any]:
        tools = self._resolved_tools()
        extra_body = self.openai_extra_body or {}
        prompt_identity = self._resolved_system_prompt()
        seed_hash = _tree_hash(self.skills_seed)[0] if isinstance(self.skills_seed, Path) else None
        return {"live_agent": {
            "api_base_identifier": _api_base_origin(self.api_base),
            "api_base_sha256": hashlib.sha256(self.api_base.encode("utf-8")).hexdigest(),
            "custom_tools": self.custom_tools is not None,
            "max_cost_usd": self.max_cost_usd,
            "max_input_tokens": self.max_input_tokens,
            "max_model_calls": self.max_model_calls,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_turns": self.max_turns,
            "model_call_timeout_seconds": self.model_call_timeout_seconds,
            "model": self.model,
            "openai_extra_body_keys": sorted(extra_body.keys()),
            "openai_extra_body_sha256": _json_hash(extra_body),
            "openai_stream_include_usage": self.openai_stream_include_usage,
            "online_evolution": self.online_evolution,
            "permission_mode": self.permission_mode,
            "profile": self.profile,
            "provider_revision": self.provider_revision,
            "skills_enabled": self.profile == "B1",
            "skills_seed_sha256": seed_hash,
            "system_prompt_hash": hashlib.sha256(
                prompt_identity.encode("utf-8")
            ).hexdigest(),
            "system_prompt_mode": "fixed" if self.custom_system_prompt is not None else "solver",
            "thinking": self.thinking,
            "tool_schema_hash": _json_hash(tools),
        }}

    async def __call__(
        self,
        *,
        task: Mapping[str, Any],
        workdir: Path,
        trace: TraceRecorder,
        env: Mapping[str, str],
    ) -> dict[str, Any]:
        del env  # BenchmarkRunner already scopes the process environment.
        statement = str(task.get("problem_statement") or "").strip()
        if not statement:
            raise ValueError("task problem_statement must not be empty")
        canonical_issue = str(task.get("raw_problem_statement") or statement).strip()
        environment_facts = str(task.get("environment_facts") or "").strip()
        repair_task = TaskEnvelope(
            task_id=str(task.get("task_id") or trace.task_id),
            canonical_issue=canonical_issue,
            repo_root=str(task.get("repo_root") or workdir),
        )
        prompt = (
            "Complete this benchmark task in the current repository. "
            "Inspect the repository, make only the necessary changes, and verify the result.\n\n"
            + (f"<repository_environment>\n{environment_facts}\n</repository_environment>\n\n"
               if environment_facts else "")
            + statement
        )
        resolved_tools = self._resolved_tools()
        repair_enabled = any(tool.get("name") == "update_work_state" for tool in resolved_tools)

        factory = self.agent_factory
        if factory is None:
            from agents.agent import Agent

            factory = Agent

        from agents.skills import reset_skill_cache

        home_skills = Path.home() / ".sans" / "skills"
        if isinstance(self.skills_seed, Path):
            _copy_tree(self.skills_seed, home_skills)
        reset_skill_cache()
        learning: dict[str, Any] | None = None
        with _skill_environment(
            self.online_evolution, self.evolution_target_skill
        ), _without_secret_in_environment(self.api_key):
            agent = factory(
                permission_mode=self.permission_mode,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                thinking=self.thinking,
                max_cost_usd=self.max_cost_usd,
                max_turns=self.max_turns,
                max_model_calls=self.max_model_calls,
                max_tool_calls=self.max_tool_calls,
                max_output_tokens=self.max_output_tokens,
                max_input_tokens_per_call=self.max_input_tokens,
                model_call_timeout_seconds=self.model_call_timeout_seconds,
                openai_extra_body=_thaw_json(self.openai_extra_body or {}),
                openai_stream_include_usage=self.openai_stream_include_usage,
                custom_tools=_thaw_json(resolved_tools),
                custom_system_prompt=self._resolved_system_prompt(),
                repair_task=repair_task if repair_enabled else None,
                skills_enabled=self.profile == "B1",
                is_sub_agent=True,
                trace=trace,
                model_purpose="main",
            )
            try:
                result = await agent.run_once(prompt)
                evolution_result = None
                if self.online_evolution:
                    extract = getattr(agent, "extract_now", None)
                    if callable(extract):
                        evolution_result = extract(
                            self.evolution_hint or (
                                "Extract only durable repository-execution lessons supported by this "
                                "run. Never retain task-specific code, paths, or facts."
                            )
                        )
                        if inspect.isawaitable(evolution_result):
                            evolution_result = await evolution_result
                drain = getattr(agent, "drain_background_skill_tasks", None)
                if callable(drain):
                    pending = drain()
                    if inspect.isawaitable(pending):
                        await pending
                if self.profile == "B1":
                    learning = self._save_learning_snapshot(trace, evolution_result)
            finally:
                drain = getattr(agent, "drain_background_skill_tasks", None)
                if callable(drain):
                    pending = drain()
                    if inspect.isawaitable(pending):
                        await pending
                shutdown = getattr(agent, "shutdown", None)
                if callable(shutdown):
                    closed = shutdown()
                    if inspect.isawaitable(closed):
                        await closed
                reset_skill_cache()
            validate_trace(trace)
        if not isinstance(result, Mapping):
            raise TypeError("Agent.run_once must return a mapping")
        output = dict(result)
        output["learning"] = learning
        return output

    def _save_learning_snapshot(
        self, trace: TraceRecorder, evolution_result: Any
    ) -> dict[str, Any]:
        root = trace.attempt_dir / "learning"
        skills = root / "skills"
        _copy_tree(Path.home() / ".sans" / "skills", skills)
        _copy_tree(Path.home() / ".sans" / "skill-evolution", root / "evolution")
        state_sha256, files = _tree_hash(skills)
        record = {
            "files": files,
            "online_evolution": self.online_evolution,
            "skills_path": "learning/skills",
            "skills_sha256": state_sha256,
            "evolution_result": evolution_result,
        }
        (root / "snapshot.json").write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return record


__all__ = [
    "B0_SYSTEM_PROMPT", "SOLVER_SYSTEM_PROMPT", "LiveAgentAdapter",
    "TraceAcceptanceError", "validate_trace",
]
