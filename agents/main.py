"""CLI entry point and interactive REPL — mirrors cli.ts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import shlex
import sys
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv

from .agent import Agent
from . import __version__
from .cli_display import cli_session
from .ui import (
    print_welcome,
    print_user_prompt,
    print_error,
    print_info,
    print_plan_for_approval,
    print_plan_approval_options,
    print_goodbye,
    print_interrupted,
    print_memory_entries,
    print_skill_entries,
    print_warning,
)
from .session import load_session, get_latest_session_id
from .telemetry import TraceRecorder, read_trace, summarize_trace
from .memory import list_memories
from .skills import (
    create_skill,
    discover_skills,
    evolve_skill,
    execute_skill,
    get_skill_by_name,
    record_feedback,
    skill_stats,
)
from .online_skill_eval import format_online_skill_eval_async


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sans",
        description="Sans — Long-Horizon Coding Harness",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    parser.add_argument("--yolo", "-y", action="store_true", help="Skip all confirmation prompts")
    parser.add_argument("--plan", action="store_true", help="Plan mode: read-only")
    parser.add_argument("--accept-edits", action="store_true", help="Auto-approve file edits")
    parser.add_argument("--dont-ask", action="store_true", help="Auto-deny confirmations (for CI)")
    parser.add_argument("--thinking", action="store_true", help="Enable extended thinking")
    parser.add_argument("--model", "-m", default=None, help="Model to use")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--resume", action="store_true", help="Resume last session")
    parser.add_argument("--no-trace", action="store_true", help="Disable local execution trace")
    parser.add_argument("--max-cost", type=float, default=None, help="Max USD spend")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agentic turns")
    parser.add_argument("--max-model-calls", type=int, default=None, help="Max solve model calls")
    parser.add_argument("--max-tool-calls", type=int, default=None, help="Max tool calls")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Max output tokens per call")
    parser.add_argument("--request-timeout", type=float, default=None, help="Model request timeout in seconds")
    parser.add_argument("--version", action="version", version=f"sans {__version__}")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")
    return parser.parse_args()


def _resolve_permission_mode(args: argparse.Namespace) -> str:
    if args.yolo:
        return "bypassPermissions"
    if args.plan:
        return "plan"
    if args.accept_edits:
        return "acceptEdits"
    if args.dont_ask:
        return "dontAsk"
    return "default"


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _load_env_file() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def _is_anthropic_compatible_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    path = (parsed.path or "").lower().rstrip("/")
    return path.endswith("/anthropic") or "/anthropic/" in path


def _resolve_api_config(cli_api_base: str | None) -> tuple[str | None, str | None, bool]:
    generic_api_key = _clean_env(os.environ.get("APIKEY")) or _clean_env(os.environ.get("MINI_CLAUDE_API_KEY"))
    openai_api_key = _clean_env(os.environ.get("OPENAI_API_KEY"))
    anthropic_api_key = _clean_env(os.environ.get("ANTHROPIC_API_KEY"))

    generic_api_base = _clean_env(os.environ.get("API")) or _clean_env(os.environ.get("MINI_CLAUDE_API_BASE"))
    openai_api_base = _clean_env(os.environ.get("OPENAI_BASE_URL"))
    anthropic_api_base = _clean_env(os.environ.get("ANTHROPIC_BASE_URL"))

    resolved_api_base = _clean_env(cli_api_base) or generic_api_base or openai_api_base or anthropic_api_base

    if resolved_api_base:
        if _is_anthropic_compatible_base_url(resolved_api_base):
            return resolved_api_base, generic_api_key or anthropic_api_key or openai_api_key, False
        return resolved_api_base, generic_api_key or openai_api_key or anthropic_api_key, True

    if anthropic_api_key or anthropic_api_base:
        return anthropic_api_base, generic_api_key or anthropic_api_key or openai_api_key, False

    if openai_api_key or openai_api_base:
        return openai_api_base, generic_api_key or openai_api_key or anthropic_api_key, True

    if generic_api_key:
        return None, generic_api_key, False

    return None, None, False


async def run_repl(agent: Agent) -> None:
    """Interactive REPL loop."""

    async def confirm_fn(message: str) -> bool:
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False

    agent.set_confirm_fn(confirm_fn)

    async def plan_approval_fn(plan_content: str) -> dict:
        print_plan_for_approval(plan_content)
        print_plan_approval_options()
        while True:
            try:
                choice = input("  Enter choice (1-4): ").strip()
            except EOFError:
                return {"choice": "manual-execute"}
            if choice == "1":
                return {"choice": "clear-and-execute"}
            elif choice == "2":
                return {"choice": "execute"}
            elif choice == "3":
                return {"choice": "manual-execute"}
            elif choice == "4":
                try:
                    feedback = input("  Feedback (what to change): ").strip()
                except EOFError:
                    feedback = ""
                return {"choice": "keep-planning", "feedback": feedback or None}
            else:
                print_warning("Invalid choice. Enter 1, 2, 3, or 4.")

    agent.set_plan_approval_fn(plan_approval_fn)

    sigint_count = 0

    def handle_sigint(sig, frame):
        nonlocal sigint_count
        if agent._current_task is not None and not agent._current_task.done():
            # Agent is processing
            agent.abort()
            print_interrupted()
            sigint_count = 0
        else:
            sigint_count += 1
            if sigint_count >= 2:
                print_goodbye()
                raise KeyboardInterrupt
            print_warning("Press Ctrl+C again to exit.")
            print_user_prompt()

    signal.signal(signal.SIGINT, handle_sigint)
    print_welcome()

    while True:
        print_user_prompt()
        try:
            line = input()
        except EOFError:
            print_goodbye()
            break

        inp = line.strip()
        sigint_count = 0

        if not inp:
            continue
        if inp in ("exit", "quit", "/exit", "/quit"):
            print_goodbye()
            break

        # REPL commands
        if inp == "/clear":
            agent.clear_history()
            continue
        if inp == "/plan":
            agent.toggle_plan_mode()
            continue
        if inp == "/cost":
            agent.show_cost()
            continue
        if inp == "/trace":
            print_info(str(agent.trace.trace_path) if agent.trace else "Tracing is disabled.")
            continue
        if inp == "/compact":
            try:
                await agent.compact()
            except Exception as e:
                print_error(str(e))
            continue
        if inp == "/memory":
            memories = list_memories()
            if not memories:
                print_info("No memories saved yet.")
            else:
                print_memory_entries(memories)
            continue
        if inp == "/skills":
            skills = discover_skills()
            if not skills:
                print_info("No skills found. Add skills to .sans/skills/<name>/SKILL.md")
            else:
                print_skill_entries(skills)
            continue
        if inp == "/skill-stats":
            print_info(skill_stats())
            continue
        if inp == "/skill-candidates":
            from .online_skill_evolution import list_staged_skills
            candidates = list_staged_skills()
            print_info("\n".join(f"{row['skill']}: {row['status']}\n  {row['receipt']}" for row in candidates)
                       or "No staged Skill candidates.")
            continue
        if inp == "/skill-learning":
            rows = agent.skill_learning_status()
            print_info("\n".join(f"{row['id']}: {row['status']} (attempts={row['attempts']})" for row in rows)
                       or "No pending Skill learning.")
            continue
        if inp == "/skill-history":
            from .skill_activation import list_skill_activations
            rows = list_skill_activations()
            print_info("\n".join(f"{row['id']}  {row['skill']}: {row['action']} / {row['status']}" for row in rows)
                       or "No Skill activations.")
            continue
        if inp.startswith(("/skill-promote ", "/skill-rollback ")):
            try:
                if agent.permission_mode == "plan":
                    raise ValueError("Skill activation is unavailable in plan mode")
                arguments = [part.strip('\"') for part in shlex.split(inp, posix=False)]
                if arguments[0] == "/skill-promote" and len(arguments) == 3:
                    import re
                    from .skill_evolution import get_evolution_dir
                    from .online_skill_evolution import promote_staged_skill
                    if not re.fullmatch(r"[0-9a-f]{64}", arguments[1]):
                        raise ValueError("Use the candidate directory ID shown by /skill-candidates")
                    staged = json.loads((get_evolution_dir() / "candidates" / arguments[1] / "stage.json").read_text(encoding="utf-8"))
                    evaluation = json.loads(Path(arguments[2]).read_text(encoding="utf-8"))
                    result = promote_staged_skill(staged, evaluation)
                elif arguments[0] == "/skill-rollback" and len(arguments) == 2:
                    from .skill_activation import rollback_skill_activation
                    result = rollback_skill_activation(arguments[1])
                else:
                    raise ValueError("Use /skill-promote <candidate-id> <evaluation.json> or /skill-rollback <activation-id>")
                agent._refresh_runtime_system_prompt()
                print_info(json.dumps(result, ensure_ascii=False, indent=2))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                print_error(str(error))
            continue
        if inp == "/skill-eval":
            side_query = agent._build_side_query(max_tokens=2400, purpose="evaluator_assist")
            print_info(await format_online_skill_eval_async(side_query=side_query))
            continue
        if inp.startswith("/extract_now"):
            hint = inp[len("/extract_now") :].strip()
            result = await agent.extract_now(hint)
            if result.get("ok"):
                print_info("Ran online skill extraction for the current pending window.")
            else:
                print_error(str(result.get("error") or result))
            continue
        if inp.startswith("/skill-feedback "):
            _, rest = inp.split(" ", 1)
            parts = rest.strip().split(" ", 2)
            if len(parts) < 2:
                print_error("Usage: /skill-feedback <skill-name> <rating> [note]")
                continue
            note = parts[2] if len(parts) > 2 else ""
            record_feedback(parts[0], parts[1], note)
            print_info(f"Recorded feedback for skill: {parts[0]}")
            continue
        if inp.startswith("/skill-evolve "):
            _, rest = inp.split(" ", 1)
            parts = rest.strip().split(" ", 1)
            if len(parts) < 2:
                print_error("Usage: /skill-evolve <skill-name> <durable lesson>")
                continue
            result = evolve_skill(parts[0], parts[1], rationale="Manual REPL evolution", target="active")
            if result.get("ok"):
                print_info(f"Evolved skill {result.get('skill')} to version {result.get('version')}")
            else:
                print_error(str(result.get("error") or result))
            continue
        if inp.startswith("/skill-create "):
            _, rest = inp.split(" ", 1)
            parts = [part.strip() for part in rest.split("|", 3)]
            if len(parts) < 4 or not all(parts[:4]):
                print_error("Usage: /skill-create <name> | <description> | <when-to-use> | <instructions>")
                continue
            result = create_skill(
                name=parts[0],
                description=parts[1],
                when_to_use=parts[2],
                instructions=parts[3],
                target="project",
                context="inline",
                user_invocable=False,
                evidence="Manual REPL skill creation",
            )
            if result.get("ok"):
                print_info(f"Created skill {result.get('skill')} at {result.get('file')}")
            else:
                print_error(str(result.get("error") or result))
            continue

        # Skill invocation: /<skill-name> [args]
        if inp.startswith("/"):
            space_idx = inp.find(" ")
            cmd_name = inp[1:space_idx] if space_idx > 0 else inp[1:]
            cmd_args = inp[space_idx + 1:] if space_idx > 0 else ""
            skill = get_skill_by_name(cmd_name)
            if skill and skill.user_invocable:
                print_info(f"Invoking skill: {skill.name}")
                try:
                    if skill.context == "fork":
                        await agent.chat(f'Use the skill tool to invoke "{skill.name}" with args: {cmd_args or "(none)"}')
                    else:
                        result = execute_skill(skill.name, cmd_args)
                        if not result:
                            print_error(f"Unknown skill: {skill.name}")
                            continue
                        if result.get("error"):
                            print_error(result["error"])
                            continue
                        await agent.chat(result["prompt"])
                except Exception as e:
                    if "abort" not in str(e).lower():
                        print_error(str(e))
                continue

        # Normal chat
        try:
            await agent.chat(inp)
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))


@cli_session
async def run_cli(agent: Agent, prompt: str | None) -> None:
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        if prompt:
            await agent.chat(prompt)
        else:
            await run_repl(agent)
    finally:
        try:
            await agent.drain_background_skill_tasks()
        finally:
            try:
                await agent.shutdown()
            finally:
                signal.signal(signal.SIGINT, previous_handler)


def save_run_summary(trace: TraceRecorder, exit_code: int) -> None:
    events = read_trace(trace.trace_path)
    summary = summarize_trace(events)
    calls = sum(event["event_type"] == "model.call.start" for event in events)
    usage = summary["provider_usage"]
    missing = calls - usage["usage_observed_model_calls"]
    usage["usage_complete"] = missing == 0
    if calls == 0:
        usage.update(input_tokens=0, output_tokens=0, total_tokens=0)
    summary.update(run_id=trace.run_id, trace_id=trace.trace_id, exit_code=exit_code,
                   model_calls_started=calls, calls_without_usage=missing)
    (trace.attempt_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    total = usage["total_tokens"]
    label = "Recorded tokens" if missing else "Tokens"
    print_info(
        f"{label}: {total if total is not None else 'unknown'} "
        f"(input {usage['input_tokens'] if usage['input_tokens'] is not None else 'unknown'}, "
        f"output {usage['output_tokens'] if usage['output_tokens'] is not None else 'unknown'}); "
        f"model calls: {calls}; calls without usage: {missing}\n"
        f"Trace: {trace.trace_path}"
    )


def main() -> None:
    """CLI 程序入口：准备运行配置，创建 Agent，并按参数选择一次性执行或交互模式。"""
    # 解析命令行参数，例如 --plan、--resume、--model，以及可选的一次性 prompt。
    args = parse_args()
    _load_env_file()

    if args.help:
        # 自定义帮助文本，展示 Sans 支持的启动参数和 REPL 内置命令。
        print("""
Sans
Usage: sans [options] [prompt]

Options:
  --yolo, -y          Skip all confirmation prompts (bypassPermissions mode)
  --plan              Plan mode: read-only, describe changes without executing
  --accept-edits      Auto-approve file edits, still confirm dangerous shell
  --dont-ask          Auto-deny anything needing confirmation (for CI)
  --thinking          Enable reasoning where supported by the provider
  --model, -m         Model to use (default: deepseek-chat, or MODEL env)
  --api-base URL      Override API base URL from CLI or .env
  --resume            Resume the last session in this repository
  --no-trace          Disable local execution trace
  --max-cost USD      Stop when estimated cost exceeds this amount
  --max-turns N       Stop after N agentic turns
  --max-model-calls N Limit solve model calls, including truncation recovery
  --max-tool-calls N  Limit tool calls
  --max-output-tokens N  Limit output tokens per model call
  --request-timeout S Set model request timeout in seconds
  --version          Show installed version
  --help, -h          Show this help

REPL commands:
  /exit, /quit        Quit the session (exit and quit also work)
  /clear              Clear conversation history
  /plan               Toggle plan mode (read-only <-> normal)
  /cost               Show token usage and cost
  /trace              Show the current trace path
  /compact            Manually compact conversation
  /memory             List saved memories
  /skills             List available skills
  /skill-stats        Show skill usage and evolution stats
  /skill-candidates   Inspect pending and active Skill candidates (offline)
  /skill-learning     Inspect queued, failed and completed learning windows
  /skill-history      Inspect Skill activation and rollback receipts
  /skill-promote      Apply a trusted task evaluation: <candidate-id> <evaluation.json>
  /skill-rollback     Restore a previous Skill snapshot: <activation-id>
  /skill-eval         Evaluate response replay quality (uses the model API)
  /extract_now        Extract the current pending online skill window: /extract_now [hint]
  /skill-feedback     Record feedback: /skill-feedback <skill> <rating> [note]
  /skill-evolve       Evolve a skill: /skill-evolve <skill> <durable lesson>
  /skill-create       Create a skill: /skill-create <name> | <description> | <when-to-use> | <instructions>
  /<skill-name>       Invoke a skill (e.g. /commit "fix types")

Examples:
  python -m agents.main "fix the bug in src/app.ts"
  python -m agents.main --plan "how would you refactor this?"
  python -m agents.main --max-cost 0.50 --max-turns 20 "implement feature X"
  python -m agents.main --resume
  python -m agents.main  # starts interactive REPL
""")
        sys.exit(0)

    # 将命令行布尔开关统一转换成 Agent 内部使用的权限模式。
    permission_mode = _resolve_permission_mode(args)
    # 模型优先使用命令行参数，其次读取 .env 中的 MODEL，最后回落到默认模型。
    model = args.model or os.environ.get("MODEL") or "deepseek-chat"
    resolved_api_base, resolved_api_key, resolved_use_openai = _resolve_api_config(args.api_base)

    # 没有可用 API key 时无法调用模型，直接提示配置方式并退出。
    if not resolved_api_key:
        print_error(
            "API key is required.\n"
            "  Set APIKEY (+ optional API) in .env for generic config,\n"
            "  or use ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL,\n"
            "  or use OPENAI_API_KEY / OPENAI_BASE_URL."
        )
        sys.exit(1)

    session = None
    resumed_from = None
    if args.resume:
        resumed_from = get_latest_session_id()
        session = load_session(resumed_from) if resumed_from else None
        if session is None:
            print_error("No session found for this repository.")
            sys.exit(1)

    prompt = " ".join(args.prompt) if args.prompt else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    trace = None if args.no_trace else TraceRecorder(
        Path.cwd() / "experiment" / "runs" / "cli" / run_id,
        run_id=run_id, task_id=resumed_from or run_id, attempt_id=run_id,
    )
    if trace:
        print_info(f"Trace: {trace.trace_path}")
    exit_code = 0
    try:
        with (trace.span("cli.run", component="cli", attributes={
            "cwd": str(Path.cwd()), "model": model,
            "mode": "one-shot" if prompt else "interactive",
            "resumed_from_session_id": resumed_from,
        }) if trace else nullcontext()) as span:
            try:
                agent = Agent(
                    permission_mode=permission_mode, model=model, thinking=args.thinking,
                    max_cost_usd=args.max_cost, max_turns=args.max_turns,
                    max_model_calls=args.max_model_calls, max_tool_calls=args.max_tool_calls,
                    max_output_tokens=args.max_output_tokens, model_call_timeout_seconds=args.request_timeout,
                    api_base=resolved_api_base if resolved_use_openai else None,
                    anthropic_base_url=resolved_api_base if not resolved_use_openai else None,
                    api_key=resolved_api_key, trace=trace,
                )
                if session is not None:
                    agent.restore_session(session)
                asyncio.run(run_cli(agent, prompt))
                if agent._aborted:
                    exit_code = 130
                elif agent.termination_reason:
                    exit_code = 2
                if span:
                    span.finish(
                        status="cancelled" if exit_code == 130 else "incomplete" if exit_code else "ok",
                        attributes={"session_id": agent.session_id,
                                    "termination_reason": agent.termination_reason},
                    )
                else:
                    agent.show_cost()
            except (KeyboardInterrupt, asyncio.CancelledError):
                if span:
                    span.finish(status="cancelled")
                raise
    except (KeyboardInterrupt, asyncio.CancelledError):
        exit_code = 130
        print_interrupted()
    except Exception as error:
        exit_code = 1
        print_error(str(error))
    finally:
        if trace:
            save_run_summary(trace, exit_code)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
