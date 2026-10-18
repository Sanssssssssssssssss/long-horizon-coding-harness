"""Tiny end-to-end BenchmarkRunner smoke test; offline unless ``--live`` is set."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from .live import LiveAgentAdapter, validate_trace
from .runner import BenchmarkRunner, TaskSpec, EXPERIMENT_ROOT, experiment_path


AgentCallback = Callable[..., Awaitable[Any]]


async def run_smoke(
    *,
    agent: AgentCallback | None = None,
    runs_root: str | Path | None = None,
    run_id: str = "smoke-1",
) -> dict[str, Any]:
    """Exercise checkout, trace, patch export/replay, and evaluation on one tiny repo."""
    callback = agent or _offline_agent
    if isinstance(callback, LiveAgentAdapter) and runs_root is None:
        raise ValueError("live smoke requires an explicit persistent runs_root")

    if isinstance(callback, LiveAgentAdapter):
        output_root = Path(runs_root).expanduser().resolve()  # type: ignore[arg-type]
        fixture_path = (output_root / ".fixtures" / "infra-smoke-v1").resolve()
        try:
            fixture_path.relative_to(output_root)
        except ValueError as exc:
            raise RuntimeError("smoke fixture must stay inside runs_root") from exc
        repository, base_commit = _fixture(fixture_path)
        return await _run_fixture(callback, output_root, run_id, repository, base_commit)

    scratch_root = EXPERIMENT_ROOT / "cache" / "smoke"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-smoke-", dir=scratch_root) as raw:
        scratch = Path(raw)
        repository, base_commit = _fixture(scratch / "fixture")
        output_root = Path(runs_root).resolve() if runs_root is not None else scratch / "runs"
        return await _run_fixture(callback, output_root, run_id, repository, base_commit)


async def _run_fixture(
    callback: AgentCallback,
    output_root: Path,
    run_id: str,
    repository: Path,
    base_commit: str,
) -> dict[str, Any]:
    live = isinstance(callback, LiveAgentAdapter)
    task = TaskSpec(
        task_id="replace-answer",
        repository=repository,
        base_commit=base_commit,
        problem_statement=(
            "Change answer.txt from `answer=broken` to exactly `answer=fixed` followed by "
            "one trailing newline. Do not change any other file."
        ),
        platform="local",
        timeout_seconds=120,
        network_policy="enabled" if live else "disabled",
    )

    def evaluator(*, workdir: Path, **_: Any) -> dict[str, Any]:
        resolved = (workdir / "answer.txt").read_text(encoding="utf-8") == "answer=fixed\n"
        return {"resolved": resolved, "status": "completed"}

    runner = BenchmarkRunner(
        runs_root=output_root,
        suite_name="infra-smoke",
        run_id=run_id,
        tasks=[task],
        agent=callback,
        evaluator=evaluator,
        manifest={} if live else {
            "model_identifier": "offline-fixture",
            "provider": "offline",
            "reasoning_setting": False,
        },
    )
    return await runner.run()


def _fixture(repository: Path) -> tuple[Path, str]:
    repository = repository.resolve()
    if repository.exists():
        item_is_junction = hasattr(repository, "is_junction") and repository.is_junction()
        if not repository.is_dir() or repository.is_symlink() or item_is_junction:
            raise RuntimeError(f"unsafe smoke fixture path: {repository}")
        root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
        if root != repository:
            raise RuntimeError("smoke fixture git root does not match its managed path")
        if _git(repository, "status", "--porcelain"):
            raise RuntimeError("smoke fixture must remain clean")
        if _git(repository, "ls-files") != "answer.txt":
            raise RuntimeError("smoke fixture tracked files do not match infra-smoke-v1")
        if (repository / "answer.txt").read_text(encoding="utf-8") != "answer=broken\n":
            raise RuntimeError("smoke fixture content does not match infra-smoke-v1")
        return repository, _git(repository, "rev-parse", "HEAD")

    repository.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "core.autocrlf", "false")
    (repository / "answer.txt").write_text("answer=broken\n", encoding="utf-8")
    _git(repository, "add", "--", "answer.txt")
    commit_env = dict(os.environ)
    commit_env.update({
        "GIT_AUTHOR_NAME": "Benchmark Smoke",
        "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "Benchmark Smoke",
        "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    })
    _git(repository, "-c", "commit.gpgsign=false", "commit", "-qm", "infra smoke fixture v1", env=commit_env)
    return repository, _git(repository, "rev-parse", "HEAD")


async def _offline_agent(*, workdir: Path, trace, **_: Any) -> dict[str, Any]:
    with trace.span(
        "model.call",
        component="model",
        purpose="main",
        attributes={"provider": "offline", "model": "fixture"},
        payload={"messages": [{"role": "user", "content": "replace the answer"}]},
    ) as span:
        span.finish(
            metrics={"provider_usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_input_tokens": None,
                "cache_miss_tokens": None,
                "cache_creation_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": 2,
                "provider_reported_cost": None,
            }},
            payload={"text": "fixed"},
        )
    (workdir / "answer.txt").write_text("answer=fixed\n", encoding="utf-8")
    validate_trace(trace)
    return {"text": "fixed", "tokens": {"input": 1, "output": 1}}


def _git(cwd: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=experiment_path, help="Directory under experiment/; required for live runs.")
    parser.add_argument("--run-id", default="smoke-1")
    parser.add_argument("--live", action="store_true", help="Explicitly allow one live API-backed smoke run.")
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--provider-revision")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=30_000)
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument(
        "--stream-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    callback = None
    if args.live:
        if args.runs_root is None:
            raise SystemExit("--live requires an explicit persistent --runs-root")
        if not args.api_base or not args.model or not args.provider_revision:
            raise SystemExit("--live requires --api-base, --model, and --provider-revision")
        try:
            extra_body = json.loads(args.extra_body_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--extra-body-json is invalid: {exc}") from exc
        if not isinstance(extra_body, dict):
            raise SystemExit("--extra-body-json must decode to an object")
        from agents.tools import tool_definitions

        smoke_tool_names = {"read_file", "write_file", "edit_file"}
        callback = LiveAgentAdapter(
            api_base=args.api_base,
            api_key=getpass.getpass("API key (not stored): "),
            model=args.model,
            provider_revision=args.provider_revision,
            max_turns=args.max_turns,
            max_output_tokens=args.max_output_tokens,
            max_input_tokens=args.max_input_tokens,
            openai_extra_body=extra_body,
            openai_stream_include_usage=args.stream_usage,
            custom_tools=tuple(tool for tool in tool_definitions if tool["name"] in smoke_tool_names),
            custom_system_prompt=(
                "You are running a tiny repository infrastructure smoke test. "
                "Use only the supplied file tools to make exactly the requested edit, then stop."
            ),
        )
    elif args.api_base or args.model or args.provider_revision:
        raise SystemExit("provider options require explicit --live")

    summary = asyncio.run(run_smoke(
        agent=callback,
        runs_root=args.runs_root,
        run_id=args.run_id,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["counts"]["resolved"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
