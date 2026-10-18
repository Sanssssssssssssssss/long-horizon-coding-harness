from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.telemetry import (
    CORE_FIELDS,
    TraceRecorder,
    measure_model_input_sections,
    measure_text_sections,
    provider_usage,
    read_trace,
    repo_state_hash,
    summarize_trace,
)


def recorder(tmp_path: Path) -> TraceRecorder:
    return TraceRecorder(
        tmp_path / "attempt",
        blob_dir=tmp_path / "blobs",
        run_id="run-1",
        task_id="task-1",
        attempt_id="attempt-1",
        trace_id="trace-1",
    )


def test_event_has_stable_fields_and_external_payload(tmp_path: Path) -> None:
    trace = recorder(tmp_path)
    payload = "large prompt\n" * 100

    first = trace.event("run.start", component="run", status="ok", payload=payload)
    second = trace.event("run.note", component="run", payload=payload)

    assert tuple(first) == CORE_FIELDS
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert first["payload_ref"] == second["payload_ref"]
    assert first["payload_ref"]["bytes"] == len(payload.encode("utf-8"))
    assert len(first["payload_ref"]["preview"]) <= 240
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert first["payload_ref"]["sha256"] == digest
    assert (tmp_path / "blobs" / digest).read_text(encoding="utf-8") == payload
    assert payload not in trace.trace_path.read_text(encoding="utf-8")
    assert len(list((tmp_path / "blobs").iterdir())) == 1


def test_nested_spans_capture_parent_finish_data_and_exception(tmp_path: Path) -> None:
    trace = recorder(tmp_path)

    with trace.span("run", component="run") as outer:
        with trace.span("model.call", component="model", purpose="main") as inner:
            inner.event("model.first_text", metrics={"ttft_ms": 3})
            inner.finish(
                metrics={"provider_usage": {"input_tokens": 4}},
                attributes={"model": "test-model"},
                payload="answer",
            )
        assert trace.current_span_id == outer.span_id
        with pytest.raises(RuntimeError, match="boom"):
            with trace.span("tool.call", component="tool", purpose="read_file"):
                raise RuntimeError("boom")

    events = read_trace(trace.trace_path)
    model_end = next(event for event in events if event["event_type"] == "model.call.end")
    tool_error = next(event for event in events if event["event_type"] == "tool.call.error")
    assert model_end["parent_span_id"] == outer.span_id
    assert model_end["metrics"]["provider_usage"]["input_tokens"] == 4
    assert model_end["attributes"]["model"] == "test-model"
    assert model_end["payload_ref"]["preview"] == "answer"
    assert tool_error["status"] == "error"
    assert tool_error["attributes"] == {
        "error_type": "RuntimeError",
        "error_message": "boom",
    }
    assert trace.current_span_id is None


def test_thread_and_async_concurrency_have_unique_ordered_sequences(tmp_path: Path) -> None:
    trace = recorder(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda number: trace.event("thread", attributes={"n": number}), range(80)))

    async def worker(number: int) -> tuple[str, str]:
        with trace.span("outer", component="run") as outer:
            await asyncio.sleep(0)
            with trace.span("inner", component="primitive") as inner:
                await asyncio.sleep(0)
            return outer.span_id, inner.span_id

    async def run_workers() -> list[tuple[str, str]]:
        return list(await asyncio.gather(*(worker(number) for number in range(10))))

    span_pairs = asyncio.run(run_workers())
    events = read_trace(trace.trace_path)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    inner_starts = {
        event["span_id"]: event
        for event in events
        if event["event_type"] == "inner.start"
    }
    for outer_id, inner_id in span_pairs:
        assert inner_starts[inner_id]["parent_span_id"] == outer_id


def test_read_trace_ignores_only_truncated_tail(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"seq":1}\n{"seq":2', encoding="utf-8")
    assert read_trace(path) == [{"seq": 1}]

    path.write_text('{bad}\n{"seq":2}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_trace(path)

    path.write_text('{"seq":1}\n{"seq":2\n{"seq":3}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_trace(path)


def test_existing_trace_sequence_is_resumed(tmp_path: Path) -> None:
    first = recorder(tmp_path)
    first.event("one")
    second = recorder(tmp_path)
    assert second.event("two")["seq"] == 2


def test_recorder_discards_truncated_tail_before_resuming(tmp_path: Path) -> None:
    first = recorder(tmp_path)
    first.event("complete")
    with first.trace_path.open("ab") as handle:
        handle.write(b'{"schema_version":"1","seq":2')

    resumed = recorder(tmp_path)
    resumed.event("after-restart")

    events = read_trace(resumed.trace_path)
    assert [event["event_type"] for event in events] == ["complete", "after-restart"]
    assert [event["seq"] for event in events] == [1, 2]

    valid = recorder(tmp_path / "valid-tail")
    valid.event("complete-without-newline")
    valid.trace_path.write_bytes(valid.trace_path.read_bytes().rstrip(b"\n"))
    resumed_valid = recorder(tmp_path / "valid-tail")
    resumed_valid.event("after-valid-tail")
    assert [event["seq"] for event in read_trace(resumed_valid.trace_path)] == [1, 2]


def test_summary_counts_layers_usage_durations_and_failures() -> None:
    def terminal(
        seq: int,
        component: str,
        *,
        status: str = "ok",
        duration_ms: float = 1.0,
        metrics: dict | None = None,
        attributes: dict | None = None,
    ) -> dict:
        return {
            "seq": seq,
            "event_type": f"{component}.end" if status == "ok" else f"{component}.error",
            "span_id": f"span-{seq}",
            "component": component,
            "status": status,
            "duration_ms": duration_ms,
            "metrics": metrics or {},
            "attributes": attributes or {},
        }

    events = [
        {**terminal(0, "model"), "event_type": "model.start", "status": "running"},
        terminal(
            1,
            "model",
            duration_ms=12.5,
            metrics={
                "provider_usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                },
                "timing": {
                    "ttfe_ms": 1.0,
                    "ttft_ms": 2.0,
                    "tttool_ms": 3.0,
                    "service_ms": 12.5,
                    "stream_generation_ms": 11.5,
                },
            },
        ),
        terminal(
            2,
            "model",
            status="error",
            duration_ms=2.5,
            metrics={
                "provider_usage": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                },
                "timing": {
                    "ttfe_ms": 0.5,
                    "ttft_ms": 1.0,
                    "service_ms": 2.5,
                    "stream_generation_ms": 2.0,
                },
            },
        ),
        terminal(3, "tool", duration_ms=4.0),
        terminal(4, "tool", status="denied", duration_ms=1.0),
        terminal(5, "primitive", duration_ms=0.5),
        terminal(
            6,
            "tool",
            attributes={"primitive_operation": True, "no_progress": True},
        ),
    ]

    summary = summarize_trace(events)

    assert summary["model_calls"] == 2
    assert summary["tool_calls"] == 3
    assert summary["failed_tool_calls"] == 1
    assert summary["primitive_operations"] == 2
    assert summary["no_progress_tool_calls"] == 1
    assert summary["failures"] == 2
    assert summary["provider_usage"]["input_tokens"] == 12
    assert summary["provider_usage"]["output_tokens"] == 4
    assert summary["provider_usage"]["total_tokens"] == 16
    assert summary["provider_usage"]["cached_input_tokens"] is None
    assert summary["provider_usage"]["provider_reported_cost"] is None
    assert summary["provider_usage"]["usage_observed_model_calls"] == 2
    assert summary["provider_usage"]["usage_complete"] is True
    assert summary["model_timing"]["ttfe_ms"] == 1.5
    assert summary["model_timing"]["ttft_ms"] == 3.0
    assert summary["model_timing"]["tttool_ms"] is None
    assert summary["model_timing"]["service_ms"] == 15.0
    assert summary["model_timing"]["stream_generation_ms"] == 13.5
    assert summary["model_duration_ms"] == 15.0
    assert summary["tool_duration_ms"] == 6.0


def test_summary_keeps_recovered_exact_usage_without_claiming_completeness() -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 3,
        "cached_input_tokens": 8,
        "cache_miss_tokens": 2,
        "cache_creation_tokens": None,
        "reasoning_tokens": 1,
        "total_tokens": 13,
        "provider_reported_cost": None,
    }
    events = [
        {
            "event_type": "model.call.error", "component": "model", "status": "error",
            "metrics": {}, "attributes": {"error_message": "429"},
        },
        {
            "event_type": "model.call.end", "component": "model", "status": "ok",
            "metrics": {"provider_usage": usage}, "attributes": {},
        },
    ]

    summary = summarize_trace(events)

    assert summary["model_calls"] == 2
    assert summary["provider_usage"]["input_tokens"] == 10
    assert summary["provider_usage"]["output_tokens"] == 3
    assert summary["provider_usage"]["cached_input_tokens"] == 8
    assert summary["provider_usage"]["usage_observed_model_calls"] == 1
    assert summary["provider_usage"]["usage_complete"] is False


def test_provider_usage_preserves_missing_exact_total_and_cache_partitions() -> None:
    assert provider_usage({
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
        "cost": 0.25,
    }, "openai") == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cached_input_tokens": 2,
        "cache_miss_tokens": 5,
        "cache_creation_tokens": None,
        "reasoning_tokens": 1,
        "total_tokens": None,
        "provider_reported_cost": 0.25,
    }
    assert provider_usage(SimpleNamespace(
        input_tokens=5,
        output_tokens=2,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=4,
        reasoning_tokens=2,
    ), "anthropic") == {
        "input_tokens": 5,
        "output_tokens": 2,
        "cached_input_tokens": 1,
        "cache_miss_tokens": None,
        "cache_creation_tokens": 4,
        "reasoning_tokens": 2,
        "total_tokens": None,
        "provider_reported_cost": None,
    }
    deepseek = provider_usage({
        "prompt_tokens": 20,
        "completion_tokens": 4,
        "prompt_cache_hit_tokens": 12,
        "prompt_cache_miss_tokens": 8,
    }, "openai")
    assert deepseek["cached_input_tokens"] == 12
    assert deepseek["cache_miss_tokens"] == 8


def test_measure_text_sections_uses_one_documented_estimate() -> None:
    measured = measure_text_sections({"ascii": "abcd", "unicode": "熊", "empty": None})
    assert measured["ascii"] == {
        "chars": 4,
        "utf8_bytes": 4,
        "estimated_tokens": 1,
        "estimated": True,
    }
    assert measured["unicode"] == {
        "chars": 1,
        "utf8_bytes": 3,
        "estimated_tokens": 1,
        "estimated": True,
    }
    assert measured["empty"]["estimated_tokens"] == 0


def test_model_input_sections_separate_runtime_injections_and_tool_results() -> None:
    measured = measure_model_input_sections(
        {"static_system": "system"},
        "runtime",
        [
            {"role": "tool", "content": "tool output"},
            {"role": "user", "content": (
                "task<retrieved_skills>skill</retrieved_skills>"
                "<system-reminder>memory</system-reminder>"
            )},
        ],
        [{"name": "read_file"}],
    )
    assert measured["task_prompt"]["chars"] == len("task")
    assert measured["retrieved_skills"]["chars"] > 0
    assert measured["retrieved_memory"]["chars"] > 0
    assert measured["tool_results"]["chars"] > 0


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_repo_state_hash_tracks_diff_and_untracked_name_and_body(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Trace Test",
            "-c",
            "user.email=trace@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=tmp_path,
        check=True,
    )

    clean = repo_state_hash(tmp_path)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Trace Test",
            "-c",
            "user.email=trace@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "same tree, new head",
        ],
        cwd=tmp_path,
        check=True,
    )
    new_head = repo_state_hash(tmp_path)
    tracked.write_text("changed\n", encoding="utf-8")
    changed = repo_state_hash(tmp_path)
    untracked = tmp_path / "new.txt"
    untracked.write_text("one", encoding="utf-8")
    named = repo_state_hash(tmp_path)
    untracked.write_text("different contents", encoding="utf-8")
    edited = repo_state_hash(tmp_path)

    assert clean is not None
    assert new_head != clean
    assert changed != new_head
    assert named != changed
    assert edited != named
    assert repo_state_hash(tmp_path) == edited
    assert repo_state_hash(tmp_path / "missing") is None


def test_repo_state_probe_returns_unknown_when_git_times_out(tmp_path, monkeypatch):
    calls = []
    def blocked(args, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])
    monkeypatch.setattr("agents.telemetry.subprocess.run", blocked)
    assert repo_state_hash(tmp_path, timeout_seconds=0.1) is None
    assert len(calls) == 1 and 0 < calls[0] <= 0.100001
