from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1"
CORE_FIELDS = (
    "schema_version",
    "seq",
    "ts_utc",
    "monotonic_ns",
    "event_type",
    "run_id",
    "task_id",
    "attempt_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "component",
    "purpose",
    "status",
    "duration_ms",
    "metrics",
    "attributes",
    "payload_ref",
)
PROVIDER_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_miss_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "total_tokens",
    "provider_reported_cost",
)
MODEL_TIMING_FIELDS = (
    "ttfe_ms",
    "ttft_ms",
    "tttool_ms",
    "service_ms",
    "stream_generation_ms",
)
_FAILURE_STATUSES = {"error", "failed", "failure", "timeout", "cancelled", "denied"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _json_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _discard_truncated_tail(path: Path) -> None:
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    head, separator, tail = raw.rpartition(b"\n")
    with path.open("rb+") as handle:
        try:
            json.loads(tail)
        except (UnicodeDecodeError, json.JSONDecodeError):
            handle.truncate(len(head) + len(separator))
        else:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")


class TraceRecorder:
    """Small append-only trace writer. Payload bodies live in a SHA256 blob store."""

    def __init__(
        self,
        attempt_dir: str | os.PathLike[str],
        *,
        blob_dir: str | os.PathLike[str] | None = None,
        run_id: str,
        task_id: str,
        attempt_id: str,
        trace_id: str | None = None,
    ) -> None:
        self.attempt_dir = Path(attempt_dir)
        self.trace_path = self.attempt_dir / "trace.jsonl"
        self.blob_dir = Path(blob_dir) if blob_dir is not None else self.attempt_dir / "blobs"
        self.run_id = run_id
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.trace_id = trace_id or uuid.uuid4().hex
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        _discard_truncated_tail(self.trace_path)
        self.trace_path.touch(exist_ok=True)
        existing = read_trace(self.trace_path)
        self._seq = max((int(event.get("seq", 0)) for event in existing), default=0)
        self._write_lock = threading.Lock()
        self._span_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
            f"trace_span_stack_{id(self)}", default=()
        )

    @property
    def current_span_id(self) -> str | None:
        stack = self._span_stack.get()
        return stack[-1] if stack else None

    def store_payload(self, payload: Any, *, preview_chars: int = 240) -> dict[str, Any]:
        data = _json_bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        destination = self.blob_dir / digest
        if not destination.exists():
            fd, temporary = tempfile.mkstemp(prefix=f".{digest}.", dir=self.blob_dir)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        preview = data.decode("utf-8", errors="replace")[: max(0, preview_chars)]
        return {"sha256": digest, "bytes": len(data), "preview": preview}

    def event(
        self,
        event_type: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        component: str | None = None,
        purpose: str | None = None,
        status: str | None = None,
        duration_ms: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
        payload: Any = None,
        payload_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if payload is not None and payload_ref is not None:
            raise ValueError("pass payload or payload_ref, not both")
        if payload is not None:
            payload_ref = self.store_payload(payload)
        record = {
            "schema_version": SCHEMA_VERSION,
            "seq": None,
            "ts_utc": None,
            "monotonic_ns": None,
            "event_type": event_type,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "trace_id": self.trace_id,
            "span_id": span_id or self.current_span_id,
            "parent_span_id": parent_span_id,
            "component": component,
            "purpose": purpose,
            "status": status,
            "duration_ms": duration_ms,
            "metrics": dict(metrics or {}),
            "attributes": dict(attributes or {}),
            "payload_ref": dict(payload_ref) if payload_ref is not None else None,
        }
        with self._write_lock:
            self._seq += 1
            record["seq"] = self._seq
            record["ts_utc"] = _utc_now()
            record["monotonic_ns"] = time.monotonic_ns()
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
            with self.trace_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(line + "\n")
                handle.flush()
        return record

    def span(
        self,
        name: str,
        *,
        component: str | None = None,
        purpose: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> _TraceSpan:
        return _TraceSpan(
            self,
            name,
            component or name,
            purpose,
            metrics,
            attributes,
            payload,
        )


class _TraceSpan:
    def __init__(
        self,
        recorder: TraceRecorder,
        name: str,
        component: str,
        purpose: str | None,
        metrics: Mapping[str, Any] | None,
        attributes: Mapping[str, Any] | None,
        payload: Any,
    ) -> None:
        self.recorder = recorder
        self.name = name
        self.component = component
        self.purpose = purpose
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_span_id: str | None = None
        self._start_metrics = metrics
        self._start_attributes = attributes
        self._start_payload = payload
        self._started_ns: int | None = None
        self._token: contextvars.Token[tuple[str, ...]] | None = None
        self._finished = False

    def __enter__(self) -> _TraceSpan:
        self.parent_span_id = self.recorder.current_span_id
        self._started_ns = time.monotonic_ns()
        self.recorder.event(
            f"{self.name}.start",
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            component=self.component,
            purpose=self.purpose,
            status="running",
            metrics=self._start_metrics,
            attributes=self._start_attributes,
            payload=self._start_payload,
        )
        stack = self.recorder._span_stack.get()
        self._token = self.recorder._span_stack.set(stack + (self.span_id,))
        return self

    def event(
        self,
        event_type: str,
        *,
        status: str | None = None,
        duration_ms: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
        payload: Any = None,
        payload_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.recorder.event(
            event_type,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            component=self.component,
            purpose=self.purpose,
            status=status,
            duration_ms=duration_ms,
            metrics=metrics,
            attributes=attributes,
            payload=payload,
            payload_ref=payload_ref,
        )

    def finish(
        self,
        *,
        status: str = "ok",
        metrics: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
        payload: Any = None,
        payload_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self._finished:
            return None
        self._finished = True
        duration_ms = None
        if self._started_ns is not None:
            duration_ms = (time.monotonic_ns() - self._started_ns) / 1_000_000
        suffix = "error" if status.lower() in _FAILURE_STATUSES else "end"
        record = self.event(
            f"{self.name}.{suffix}",
            status=status,
            duration_ms=duration_ms,
            metrics=metrics,
            attributes=attributes,
            payload=payload,
            payload_ref=payload_ref,
        )
        if self._token is not None:
            self.recorder._span_stack.reset(self._token)
            self._token = None
        return record

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if exc is None:
            self.finish()
        else:
            self.finish(
                status="error",
                attributes={"error_type": exc_type.__name__, "error_message": str(exc)},
            )
        return False


def read_trace(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    raw = trace_path.read_bytes()
    lines = raw.splitlines()
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index == len(lines) - 1 and not raw.endswith((b"\n", b"\r")):
                break
            raise
        if not isinstance(event, dict):
            raise ValueError(f"trace line {index + 1} is not a JSON object")
        events.append(event)
    return events


def _terminal(event: Mapping[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "").lower()
    status = str(event.get("status") or "").lower()
    return event_type.endswith((".end", ".error")) or (
        not event_type.endswith(".start") and status in ({"ok", "success"} | _FAILURE_STATUSES)
    )


def summarize_trace(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    calls: dict[str, list[Mapping[str, Any]]] = {"model": [], "tool": [], "primitive": []}
    failures = 0
    for event in events:
        status = str(event.get("status") or "").lower()
        if status in _FAILURE_STATUSES:
            failures += 1
        if not _terminal(event):
            continue
        component = str(event.get("component") or "").lower()
        attributes = event.get("attributes") or {}
        if component in {"model", "llm", "provider"} or component.startswith("model."):
            calls["model"].append(event)
        if component == "tool" or component.startswith("tool."):
            calls["tool"].append(event)
        if component == "primitive" or component.startswith("primitive.") or (
            isinstance(attributes, Mapping) and attributes.get("primitive_operation") is True
        ):
            calls["primitive"].append(event)

    def sum_nested(
        fields: Iterable[str], key: str,
        source_events: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, int | float | None]:
        totals: dict[str, int | float | None] = {}
        selected = calls["model"] if source_events is None else list(source_events)
        for field in fields:
            values: list[int | float] = []
            for event in selected:
                metrics = event.get("metrics") or {}
                nested = metrics.get(key) if isinstance(metrics, Mapping) else None
                source = nested if isinstance(nested, Mapping) else metrics
                value = source.get(field) if isinstance(source, Mapping) else None
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    values = []
                    break
                values.append(value)
            totals[field] = (
                sum(values) if values and len(values) == len(selected) else None
            )
        return totals

    usage_events = []
    for event in calls["model"]:
        metrics = event.get("metrics") or {}
        nested = metrics.get("provider_usage") if isinstance(metrics, Mapping) else None
        if isinstance(nested, Mapping) and all(
            isinstance(nested.get(field), (int, float))
            and not isinstance(nested.get(field), bool)
            for field in ("input_tokens", "output_tokens")
        ):
            usage_events.append(event)

    usage = sum_nested(PROVIDER_USAGE_FIELDS, "provider_usage", usage_events)
    if usage["total_tokens"] is None and all(
        isinstance(usage[field], (int, float)) and not isinstance(usage[field], bool)
        for field in ("input_tokens", "output_tokens")
    ):
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["usage_observed_model_calls"] = len(usage_events)
    usage["usage_complete"] = len(usage_events) == len(calls["model"])
    timing = sum_nested(MODEL_TIMING_FIELDS, "timing")

    def total_duration(kind: str) -> float | None:
        values = [event.get("duration_ms") for event in calls[kind]]
        if not values or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return None
        return float(sum(values))

    return {
        "model_calls": len(calls["model"]),
        "tool_calls": len(calls["tool"]),
        "failed_tool_calls": sum(
            str(event.get("status") or "").lower() in _FAILURE_STATUSES
            for event in calls["tool"]
        ),
        "primitive_operations": len(calls["primitive"]),
        "no_progress_tool_calls": sum(
            (event.get("attributes") or {}).get("no_progress") is True
            for event in calls["tool"]
        ),
        "prefix_changes": sum(
            (event.get("attributes") or {}).get("prefix_changed") is True
            for event in calls["model"]
        ),
        "failures": failures,
        "provider_usage": usage,
        "model_timing": timing,
        "model_duration_ms": total_duration("model"),
        "tool_duration_ms": total_duration("tool"),
    }


def measure_text_sections(sections: Mapping[str, Any]) -> dict[str, dict[str, int | bool]]:
    measured: dict[str, dict[str, int | bool]] = {}
    for name, value in sections.items():
        text = "" if value is None else str(value)
        byte_count = len(text.encode("utf-8"))
        measured[name] = {
            "chars": len(text),
            "utf8_bytes": byte_count,
            "estimated_tokens": (byte_count + 3) // 4,
            "estimated": True,
        }
    return measured


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def provider_usage(usage: Any, provider: str) -> dict[str, int | float | None]:
    """Normalize exact provider counters; unavailable counters remain None."""
    if provider == "openai":
        input_tokens = _field(usage, "prompt_tokens")
        output_tokens = _field(usage, "completion_tokens")
        cached = _field(usage, "prompt_cache_hit_tokens")
        if cached is None:
            cached = _field(_field(usage, "prompt_tokens_details"), "cached_tokens")
        cache_miss = _field(usage, "prompt_cache_miss_tokens")
        if (
            cache_miss is None
            and isinstance(input_tokens, int)
            and isinstance(cached, int)
        ):
            cache_miss = max(0, input_tokens - cached)
        cache_creation = None
        reasoning = _field(_field(usage, "completion_tokens_details"), "reasoning_tokens")
        if reasoning is None:
            reasoning = _field(usage, "reasoning_tokens")
    else:
        input_tokens = _field(usage, "input_tokens")
        output_tokens = _field(usage, "output_tokens")
        cached = _field(usage, "cache_read_input_tokens")
        cache_miss = _field(usage, "cache_miss_input_tokens")
        cache_creation = _field(usage, "cache_creation_input_tokens")
        reasoning = _field(usage, "reasoning_tokens")
    total = _field(usage, "total_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached,
        "cache_miss_tokens": cache_miss,
        "cache_creation_tokens": cache_creation,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "provider_reported_cost": _field(usage, "cost"),
    }


def timing_metrics(
    started_ns: int,
    ended_ns: int,
    *,
    first_event_ns: int | None = None,
    first_text_ns: int | None = None,
    first_tool_ns: int | None = None,
) -> dict[str, float | None]:
    def elapsed(point: int | None) -> float | None:
        return None if point is None else (point - started_ns) / 1_000_000

    return {
        "ttfe_ms": elapsed(first_event_ns),
        "ttft_ms": elapsed(first_text_ns),
        "tttool_ms": elapsed(first_tool_ns),
        "service_ms": (ended_ns - started_ns) / 1_000_000,
        "stream_generation_ms": None if first_event_ns is None else (ended_ns - first_event_ns) / 1_000_000,
    }


def measure_model_input_sections(
    prompt_sections: Mapping[str, Any],
    runtime_guidance: str,
    messages: list[Any],
    tools: list[Any],
) -> dict[str, dict[str, int | bool]]:
    """Partition a provider request into stable, estimated prompt sections."""
    sections = dict.fromkeys((
        "static_system", "environment_and_git", "project_rules", "memory_index",
        "retrieved_memory", "skill_descriptions", "retrieved_skills",
        "subagent_descriptions", "task_prompt", "history", "tool_results",
        "folded_memory",
    ), "")
    sections.update(prompt_sections)
    sections["runtime_guidance"] = runtime_guidance
    sections["tool_schemas"] = json.dumps(tools, ensure_ascii=False, default=str)
    history, tool_results, folded = [], [], []
    for index, message in enumerate(messages):
        serialized = json.dumps(message, ensure_ascii=False, default=str)
        content = message.get("content") if isinstance(message, Mapping) else None
        blocks = content if isinstance(content, list) else []
        if (isinstance(message, Mapping) and message.get("role") == "tool") or any(
            isinstance(block, Mapping) and block.get("type") == "tool_result" for block in blocks
        ):
            tool_results.append(message)
        elif "<session-folded-memory>" in serialized:
            folded.append(message)
        elif index == len(messages) - 1 and isinstance(message, Mapping) and message.get("role") == "user":
            task_text = content if isinstance(content, str) else serialized
            skills = re.findall(r"<retrieved_skills>.*?</retrieved_skills>", task_text, re.DOTALL)
            memories = re.findall(r"<system-reminder>.*?</system-reminder>", task_text, re.DOTALL)
            sections["retrieved_skills"] = "\n".join(skills)
            sections["retrieved_memory"] = "\n".join(memories)
            task_text = re.sub(r"<(retrieved_skills|system-reminder)>.*?</\1>", "", task_text, flags=re.DOTALL)
            sections["task_prompt"] = task_text
        else:
            history.append(message)
    sections["history"] = json.dumps(history, ensure_ascii=False, default=str)
    sections["tool_results"] = json.dumps(tool_results, ensure_ascii=False, default=str)
    sections["folded_memory"] = json.dumps(folded, ensure_ascii=False, default=str)
    return measure_text_sections(sections)


def repo_state_hash(cwd: str | os.PathLike[str], *, timeout_seconds: float = 10.0) -> str | None:
    deadline = time.monotonic() + timeout_seconds
    def remaining():
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise TimeoutError("repository probe deadline reached")
        return duration
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=remaining(),
        )
        if probe.returncode != 0 or probe.stdout.strip() != b"true":
            return None
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=remaining(),
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=remaining(),
        )
        if diff.returncode != 0:
            diff = subprocess.run(
                ["git", "diff", "--binary", "--"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=remaining(),
            )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=remaining(),
        )
        if diff.returncode != 0 or untracked.returncode != 0:
            return None
    except (OSError, ValueError, subprocess.TimeoutExpired, TimeoutError):
        return None
    digest = hashlib.sha256()
    digest.update(b"head\0")
    digest.update(head.stdout.strip() if head.returncode == 0 else b"<unborn>")
    digest.update(b"\0diff\0")
    digest.update(diff.stdout)
    digest.update(b"\0untracked\0")
    root = Path(cwd).resolve()
    for name in sorted(filter(None, untracked.stdout.split(b"\0"))):
        path = root / os.fsdecode(name)
        content_digest = hashlib.sha256()
        try:
            remaining()
            if path.is_symlink():
                content_digest.update(os.fsencode(os.readlink(path)))
            else:
                path.resolve().relative_to(root)
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        remaining()
                        content_digest.update(chunk)
        except (OSError, ValueError, TimeoutError):
            return None
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(content_digest.digest())
    return digest.hexdigest()


__all__ = [
    "CORE_FIELDS",
    "MODEL_TIMING_FIELDS",
    "PROVIDER_USAGE_FIELDS",
    "SCHEMA_VERSION",
    "TraceRecorder",
    "measure_model_input_sections",
    "measure_text_sections",
    "provider_usage",
    "read_trace",
    "repo_state_hash",
    "summarize_trace",
    "timing_metrics",
]
