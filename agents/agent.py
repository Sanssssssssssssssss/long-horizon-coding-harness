#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import shlex
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Awaitable, Any

import anthropic
import openai

from agents.mcp_client import McpManager
from agents.memory import MemoryPrefetch, start_memory_prefetch, format_memories_for_injection
from agents.prompt import build_system_prompt_with_sections
from agents.repair_state import RepairState, TaskEnvelope, preview_output
from agents.tool_result import ShellResult
from agents.verification import CHECK_FINGERPRINT_SCRIPT, DIFF_CHECK_SCRIPT, compare_baseline, pytest_selection, python_command
from agents.session_memory import (
    COMPACTION_KEEP_RECENT_TOKENS,
    COMPACTION_PROMPT,
    COMPACTION_RESERVE_TOKENS,
    COMPACTION_RETRY_PROMPT,
    COMPACTION_SUMMARY_MAX_TOKENS,
    SUMMARIZATION_SYSTEM_PROMPT,
    compaction_cut_index,
    folded_memory_summary,
    format_folded_memory,
    serialize_compaction_history,
    validate_folded_memory,
)
from agents.session import save_folded_session_memory, save_session
from agents.subagent import get_sub_agent_config
from agents.telemetry import (
    TraceRecorder,
    measure_model_input_sections,
    measure_text_sections,
    provider_usage,
    repo_state_hash,
    timing_metrics,
)
from agents.tools import ToolDef, tool_definitions, execute_tool, CONCURRENCY_SAFE_TOOLS, check_permission, \
    get_active_tool_definitions, format_file_window, _truncate_result
from agents.ui import print_info, print_divider, print_assistant_text, print_sub_agent_start, print_sub_agent_end, \
    start_spinner, stop_spinner, print_cost, print_tool_call, print_tool_result, print_confirmation, print_retry, \
    print_error
from agents.cli_display import activity, stream_phase, tool_execution_result


class _IncompleteSideResult(ValueError):
    def __init__(self, finish_reason: str):
        self.finish_reason = finish_reason
        super().__init__(f"side query response was incomplete: {finish_reason}")


# 指数退避重试


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if type(error).__name__ == "APIConnectionError" and type(error).__module__.startswith(("openai", "anthropic")):
        return True
    # Provider SDKs may use either httpx or its httpx2 packaging.
    if any(cls.__name__ == "TransportError" and cls.__module__ in {"httpx", "httpx2"}
           for cls in type(error).__mro__):
        return True
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error).lower()
    return any(marker in msg for marker in (
        "overloaded", "econnreset", "etimedout", "timed out", "empty response",
        "upstream stream ended before terminal chunk", "bad record mac",
    ))


def _is_context_window_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in (
        "context_length_exceeded", "context window", "maximum context length",
        "maximum prompt length", "prompt is too long", "too long for this model",
        "input budget exceeded before provider call",
    ))


def _safe_utf8_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _env_disabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_for_utf8(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_utf8_text(value)
    if isinstance(value, list):
        return [_sanitize_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_utf8(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_for_utf8(key): _sanitize_for_utf8(item)
            for key, item in value.items()
        }
    return value


async def _with_retry(fn, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason, delay)
            await asyncio.sleep(delay)

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "deepseek-chat":200000
}

def _get_context_windows(model: str) -> int:
    return MODEL_CONTEXT.get(model, 200000)


MAX_TOOL_CALLS_PER_RESPONSE = 8
REASONING_REPLAY_PLACEHOLDER = "(reasoning omitted)"

PRIMITIVE_TOOLS = {"read_file", "write_file", "edit_file", "list_files", "grep_search", "run_shell"}
STATE_CHANGING_TOOLS = {
    "write_file", "edit_file", "run_shell", "skill_create", "skill_evolve",
    "compact_context", "enter_plan_mode", "exit_plan_mode",
}

# ponytail: common runner markers cover current benchmark repos; add typed receipts only if custom runners become common.
VERIFICATION_COMMAND_MARKERS = (
    "pytest", "unittest", "tox", "nox", "cargo test", "cargo check", "go test",
    "npm test", "npm run test", "npm run check", "npm run lint", "npm run build",
    "pnpm test", "pnpm run test", "pnpm run check", "pnpm run lint", "pnpm run build",
    "yarn test", "yarn run test", "yarn run check", "yarn run lint", "yarn run build",
    "bun test", "bin/test", "ctest", "dotnet test", "mvn test", "gradle test", "gradlew test",
    "make test", "make check",
)

_DEFAULT_DIFF_CHECK_COMMAND = (
    python_command(DIFF_CHECK_SCRIPT, [], str(Path.cwd()), sys.executable)
    if os.name == "nt"
    else """set -eu
index=$(mktemp)
trap 'rm -f "$index"' EXIT
GIT_INDEX_FILE="$index" git read-tree HEAD
GIT_INDEX_FILE="$index" git add -A -- .
GIT_INDEX_FILE="$index" git diff --cached --check HEAD --"""
)

class _BudgetExceeded(RuntimeError):
    pass



def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384

#转换tool的形式到openai
def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class Agent:
    def __init__(self,
                 *,
                 permission_mode:str="default",
                 model:str="deepseek-chat",
                 api_base: str | None=None,
                 anthropic_base_url: str | None=None,
                 api_key: str | None=None,
                 thinking: bool=False,
                 max_cost_usd: float | None=None,
                 max_turns: int | None=None,
                 max_model_calls: int | None=None,
                 max_tool_calls: int | None=None,
                 max_output_tokens: int | None=None,
                 max_input_tokens_per_call: int | None=None,
                 model_call_timeout_seconds: float | None=None,
                 model_retry_limit: int=3,
                 openai_extra_body: dict[str, Any] | None=None,
                 openai_stream_include_usage: bool=True,
                 confirm_fn:Callable[[str], Awaitable[bool]] | None=None,
                 custom_system_prompt: str | None=None,
                 system_prompt_context: dict[str, str] | None=None,
                 custom_tools: list[ToolDef] | None=None,
                 tool_executor: Callable[[str, dict], Any] | None=None,
                 repo_state_probe: Callable[[], Any] | None=None,
                 diff_check_command: str=_DEFAULT_DIFF_CHECK_COMMAND,
                 repair_task: TaskEnvelope | None=None,
                 skills_enabled: bool | None=None,
                 skill_task_evaluator: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None=None,
                 is_sub_agent: bool=False,
                 trace: TraceRecorder | None=None,
                 model_purpose: str="main",
                 _operation_counts: dict[str, int] | None=None,):
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.model = model
        self.use_openai = bool(api_base)
        self.is_sub_agent = is_sub_agent
        self.trace = trace
        self.model_purpose = model_purpose
        self._skill_task_evaluator = skill_task_evaluator
        self._mcp_enabled = not is_sub_agent and not _env_disabled("SANS_DISABLE_MCP")
        self._memory_enabled = not is_sub_agent and not _env_disabled("SANS_DISABLE_MEMORY")
        if skills_enabled is not None and not isinstance(skills_enabled, bool):
            raise ValueError("skills_enabled must be a boolean or None")
        self._skills_enabled_explicit = skills_enabled
        self._skills_enabled = (
            skills_enabled
            if skills_enabled is not None
            else not is_sub_agent
        )
        self._session_persistence_enabled = (
            not is_sub_agent and not _env_disabled("SANS_DISABLE_SESSION_PERSISTENCE")
        )
        self.tools = tool_definitions if custom_tools is None else custom_tools
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        for name, value in (
            ("max_model_calls", max_model_calls),
            ("max_tool_calls", max_tool_calls),
            ("max_output_tokens", max_output_tokens),
            ("max_input_tokens_per_call", max_input_tokens_per_call),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens_per_call = max_input_tokens_per_call
        if type(model_retry_limit) is not int or model_retry_limit < 0:
            raise ValueError("model_retry_limit must be a non-negative integer")
        self.model_retry_limit = model_retry_limit
        if model_call_timeout_seconds is not None and (
            isinstance(model_call_timeout_seconds, bool)
            or not isinstance(model_call_timeout_seconds, (int, float))
            or model_call_timeout_seconds <= 0
        ):
            raise ValueError("model_call_timeout_seconds must be positive")
        self.model_call_timeout_seconds = model_call_timeout_seconds
        self._openai_extra_body = dict(openai_extra_body or {})
        self._openai_stream_include_usage = bool(openai_stream_include_usage)
        self._api_base = api_base
        self._anthropic_base_url = anthropic_base_url
        self._api_key = api_key
        self.confirm_fn = confirm_fn
        self._custom_system_prompt = custom_system_prompt
        self._system_prompt_context = dict(system_prompt_context or {})
        self._tool_executor = tool_executor
        self._repo_state_probe = repo_state_probe
        self._diff_check_command = str(diff_check_command).strip()
        if not self._diff_check_command:
            raise ValueError("diff_check_command must not be empty")
        if repair_task is not None and not isinstance(repair_task, TaskEnvelope):
            raise TypeError("repair_task must be a TaskEnvelope or None")
        self._repair_task = repair_task
        self._repair_state = RepairState(repair_task, None) if repair_task is not None else None
        self._observations: dict[str, str | dict] = {}
        self._baseline_ref: str | None = None
        self._baseline_attempts: set[tuple[str, str]] = set()
        self._diagnostic_checks: dict[str, list[str]] = {}
        self.context_window = max_input_tokens_per_call or _get_context_windows(model)
        self.effective_window = self.context_window
        self.session_id = uuid.uuid4().hex[:8]
        self._current_user_source: str | None = None
        self.session_start_time= time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_provider_reported_cost = 0.0
        self._provider_cost_complete = True
        self.last_input_token_count = 0
        self.current_turns = 0
        self.last_api_call_time = 0
        self.termination_reason: dict[str, str] | None = None
        self.finish_reason: str | None = None
        self._response_observed = False
        # Shared by native sessions and benchmark checkpoint phases.
        self._length_recoveries = 0
        self._operation_counts = _operation_counts or {"model_calls": 0, "tool_calls": 0}
        self._task_plan: list[dict[str, str]] = []
        self._stop_initial_repo_state: str | None = None
        self._stop_repo_state_seen = False
        self._last_verification: dict[str, Any] | None = None
        self._verification_history: list[dict[str, Any]] = []
        self._stop_hook_blocks = 0
        self._completion_evidence: dict[str, Any] | None = None
        self._finalization_calls: int | None = None
        self._finalization_deadline = 0.0


        self._aborted = False
        #存储异步任务
        self._current_task:asyncio.Task | None = None
        #权限白名单
        self._confirmed_paths: set[str] = set()


        # 计划模式”（Plan Mode）状态的变量
        self._pre_plan_mode: str | None=None
        self._plan_file_path: str | None=None
        self._plan_approval_fn : Callable[[str], Awaitable[bool]] | None=None
        self._context_cleared : bool=False

        #思考模式
        self._thinking_mode = self._resolve_thinking_mode()

        #子agent的输出缓存
        self._output_buffer: list[str] | None=None
        self._turn_output_buffer: list[str] | None = None

        # 编辑前读取
        self._read_file_state: dict[str, float] ={}

        #MCP集成
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        #记忆回溯
        #记忆agent已经回答过的信息
        self._already_surfaced_memories: set[str] = set()
        #当前会话占用的字节数
        self._session_memory_bytes = 0

        #区分message的历史消息
        self._anthropic_messages: list[str] = []
        self._openai_messages: list[str] = []
        self._last_retrieved_skill_reference: dict[str, Any] | None = None
        self._last_retrieved_skill_hits: list[dict[str, Any]] = []
        self._loaded_skill_context = ""
        self._loaded_skill_receipt: dict[str, Any] | None = None
        self._pending_skill_extraction_window: dict[str, Any] | None = None
        self._background_skill_tasks: set[asyncio.Task] = set()
        self._folded_session_memories: list[dict[str, Any]] = []
        self._fold_last_time: float = 0.0
        self._fold_count: int = 0
        self._failed_compaction_signature: str | None = None
        self._tool_error_streak: int = 0
        self._same_tool_repeat_count: int = 0
        self._last_tool_name: str = ""
        self._last_progress_signature: str = ""
        self._last_prefix_sha256: dict[str, str] = {}

        #构建系统提示词
        if custom_system_prompt is None:
            self._base_system_prompt, self._prompt_sections = build_system_prompt_with_sections(
                include_skills=self._skills_enabled, context=self._system_prompt_context,
            )
        else:
            self._base_system_prompt = custom_system_prompt
            self._prompt_sections = {"static_system": custom_system_prompt}
        self._runtime_guidance = ""
        self._system_prompt_frozen = custom_system_prompt is not None

        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt

        #初始化大模型客户端
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._anthropic_client = None
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            kwargs : dict[str,Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

        self._refresh_runtime_system_prompt()

    async def _await_model_call(self, call: Awaitable[Any]) -> Any:
        if self.model_call_timeout_seconds is None:
            return await call
        return await asyncio.wait_for(call, timeout=self.model_call_timeout_seconds)

    #判断返回模型的思考模式
    def _resolve_thinking_mode(self) -> str:
        if not self.thinking:
            return "disabled"
        if not self._model_supports_thinking():
            return "disabled"

        if self._model_supports_adaptive_thinking():
            return "adaptive"
        return "enabled"

    @staticmethod
    def _estimate_model_input_tokens(
        *, messages: list[Any], tools: list[Any], system: str | None = None
    ) -> int:
        payload: dict[str, Any] = {"messages": messages, "tools": tools}
        if system is not None:
            payload["system"] = system
        # ponytail: tokenizer-free estimate; replace only if a provider tokenizer is required.
        payload_bytes = len(
            json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")
        )
        return 256 + (payload_bytes + 2) // 3

    def _preflight_model_input(self, *, messages: list[Any], tools: list[Any], system: str | None = None,
                               output_limit: int | None = None) -> None:
        limit = self.max_input_tokens_per_call
        if limit is None:
            return
        estimated_tokens = self._estimate_model_input_tokens(
            messages=messages, tools=tools, system=system,
        )
        if estimated_tokens > limit:
            self.termination_reason = {"kind": "input_budget", "estimated_tokens": estimated_tokens, "limit": limit}
            if self.trace:
                self.trace.event("budget.input_rejected", component="budget", purpose=self.model_purpose,
                                 attributes=self.termination_reason)
            raise _BudgetExceeded(
                "Input budget exceeded before provider call: "
                f"tokenizer-free estimate={estimated_tokens}, "
                f"max_input_tokens_per_call={limit}"
            )

    def _model_supports_thinking(self) -> bool:
        m = self.model.lower()
        if "deepseek" in m:
            return True
        if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
            return False
        if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
            return True
        return False

    def _openai_request_extra_body(self) -> dict[str, Any]:
        body = dict(self._openai_extra_body)
        if self._thinking_mode in ("adaptive", "enabled"):
            api_base = (self._api_base or "").lower()
            if "developer.amd.com.cn" in api_base:
                template = dict(body.get("chat_template_kwargs") or {})
                template.setdefault("thinking", True)
                body["chat_template_kwargs"] = template
            elif "api.commandcode.ai" not in api_base:
                body.setdefault("thinking", {"type": "enabled"})
        return body

    def _model_supports_adaptive_thinking(self) -> bool:
        m = self.model.lower()
        return "opus-4-6" in m or "sonnet-4-6" in m

    #生成一个用于保存 AI 计划（Plan）的 Markdown 文件的绝对路径。
    def _generate_plan_file_path(self) -> str:
        d = Path.home() / ".sans" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

    # Plan Mode Active

    Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

    ## Plan File: {self._plan_file_path}
    Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

    ## Workflow
    1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
    2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
    3. **Write Plan**: Write a structured plan to the plan file including:
       - **Context**: Why this change is needed
       - **Steps**: Implementation steps with critical file paths
       - **Verification**: How to test the changes
    4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

    IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    #判断当前的任务所有的任务是否完成
    @property
    def is_processing(self)->bool:
        return self._current_task is not None and not self._current_task.done()

    #大模型调用的工厂方法,构建一个用于记忆召回（memory recall）的 sideQuery 可调用对象，兼容anthropic, openai。
    @staticmethod
    def _side_query_purpose(default: str, system: str) -> str:
        lowered = system.lower()
        if "extractor" in lowered:
            return "skill_extract"
        if "maintainer" in lowered or "skill set manager" in lowered:
            return "skill_maintain"
        if "judge whether retrieved skills" in lowered:
            return "skill_usage_judge"
        return default

    async def _run_side_query_call(
        self,
        *,
        provider: str,
        purpose: str,
        system: str,
        user_message: str,
        invoke: Callable[[], Awaitable[Any]],
        extract_text: Callable[[Any], str],
        reasoning_mode: str | None = None,
        request_messages: list[Any] | None = None,
    ) -> tuple[Any, str]:
        if self._stop_before_model_if_cost_exhausted():
            raise _BudgetExceeded("max_cost limit reached before side query")
        if self.trace is None:
            response = await self._await_model_call(invoke())
            text = extract_text(response)
            self._record_normalized_usage(
                provider_usage(getattr(response, "usage", None), provider)
            )
            return response, text
        started_ns = time.monotonic_ns()
        model_attributes = self._model_trace_attributes(
            provider, [], system=system, purpose=purpose
        )
        if reasoning_mode is not None:
            model_attributes["reasoning_mode"] = reasoning_mode
        with self.trace.span(
            "model.call",
            component="model",
            purpose=purpose,
            attributes=model_attributes,
            payload={
                "system": system,
                "messages": request_messages,
                "user_message": user_message,
            },
        ) as span:
            response = await self._await_model_call(invoke())
            ended_ns = time.monotonic_ns()
            text = extract_text(response)
            usage = provider_usage(getattr(response, "usage", None), provider)
            self._record_normalized_usage(usage)
            reason = (getattr(response.choices[0], "finish_reason", None) if provider == "openai" and response.choices
                      else getattr(response, "stop_reason", None))
            span.finish(
                attributes={**model_attributes, "finish_reason": reason},
                metrics={
                    "provider_usage": usage,
                    "timing": timing_metrics(
                        started_ns,
                        ended_ns,
                        first_event_ns=ended_ns,
                        first_text_ns=ended_ns if text else None,
                    ),
                    "input_sections": measure_text_sections({
                        "static_system": system,
                        "history": request_messages or [],
                        "task_prompt": user_message,
                    }),
                    "output_sections": measure_text_sections({"model_text": text}),
                },
                payload={"text": text},
            )
            return response, text

    async def _side_query_once(self, system: str, user_message: str, *, max_tokens: int, purpose: str,
                               disable_thinking: bool = True) -> str:
        provider = "openai" if self.use_openai else "anthropic"
        messages = [{"role": "user", "content": user_message}]
        if self.use_openai:
            messages.insert(0, {"role": "system", "content": system})
        output_limit = min(max(1, int(max_tokens)), self.max_output_tokens or _get_max_output_tokens(self.model))
        async def invoke():
            self._preflight_model_input(messages=messages, tools=[], system=None if self.use_openai else system,
                                        output_limit=output_limit)
            self._reserve_call("model_calls", self.max_model_calls)
            if not self.use_openai:
                return await self._anthropic_client.messages.create(
                    model=self.model, max_tokens=output_limit, system=system, messages=messages)
            extra_body = dict(self._openai_extra_body) if disable_thinking else self._openai_request_extra_body()
            if disable_thinking and self._model_supports_thinking():
                if "api.commandcode.ai" in (self._api_base or "").lower():
                    extra_body["reasoning_effort"] = "off"
                elif "developer.amd.com.cn" in (self._api_base or "").lower():
                    template = dict(extra_body.get("chat_template_kwargs") or {})
                    template["thinking"] = False
                    extra_body["chat_template_kwargs"] = template
                else:
                    extra_body["thinking"] = {"type": "disabled"}
            params = dict(model=self.model, messages=messages, max_tokens=output_limit)
            if extra_body: params["extra_body"] = extra_body
            return await self._openai_client.chat.completions.create(**params)

        def extract(response):
            if self.use_openai:
                return (getattr(response.choices[0].message, "content", None) or "") if response.choices else ""
            return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

        response, text = await _with_retry(lambda: self._run_side_query_call(
            provider=provider, purpose=purpose, system=system, user_message=user_message,
            invoke=invoke, extract_text=extract, request_messages=messages,
            reasoning_mode="disabled" if disable_thinking else self._thinking_mode,
        ), max_retries=self.model_retry_limit)
        if self.use_openai and not response.choices:
            raise ValueError("side query provider returned no choices")
        reason = (getattr(response.choices[0], "finish_reason", None) if self.use_openai
                  else getattr(response, "stop_reason", None))
        if reason in {"length", "max_tokens", "model_context_window_exceeded"}:
            raise _IncompleteSideResult(str(reason))
        return text

    def _build_side_query(self, *, max_tokens: int = 256, purpose: str = "memory_select"):
        if not (self._openai_client or self._anthropic_client):
            return None
        async def query(system: str, user_message: str) -> str:
            call_purpose = self._side_query_purpose(purpose, system)
            limits = (max_tokens, self.max_output_tokens or _get_max_output_tokens(self.model))
            for attempt, limit in enumerate(limits):
                try:
                    return await self._side_query_once(system, user_message, max_tokens=limit, purpose=call_purpose)
                except _IncompleteSideResult as error:
                    if attempt or error.finish_reason not in {"length", "max_tokens"}:
                        raise
                    logging.warning("side query truncated; retrying once with configured output ceiling")
            return ""
        return query

    #异步任务取消（Abort）
    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn:Callable[[str], Awaitable[bool]]) -> None:
        self._plan_approval_fn = fn


    #计划模式开关（“状态切换与现场保护”机制）
    def toggle_plan_mode(self) -> str:
        """
               1. 退出计划模式（从 plan 切回原模式）
               当当前模式已经是 plan 时，执行 if 分支：
               恢复之前的状态：self.permission_mode = self._pre_plan_mode or "default"。
                   在进入计划模式时，程序会把原本的模式保存在 _pre_plan_mode 里。退出时，就把它重新拿出来赋值回去，恢复到切换前的状态。
               清理计划模式的痕迹：把 _pre_plan_mode 和 _plan_file_path（计划文件路径）清空，并将系统提示词 _system_prompt 恢复为最基础的 _base_system_prompt。
               同步 OpenAI 消息：如果底层使用的是 OpenAI 接口，它还会同步更新消息列表里的第一条系统提示词，确保 AI 的上下文也跟着切换回来。
               反馈返回：打印退出提示，并返回恢复后的模式名称。

               2. 进入计划模式（从其他模式切入 plan）
       当当前模式不是 plan 时，执行 else 分支：
       保护当前现场：self._pre_plan_mode = self.permission_mode。先把当前正在使用的模式（比如正常模式或自动接受模式）暂存起来，方便以后能原路返回。
       切换并初始化：将当前模式设为 "plan"，生成一个专属的计划文件路径，并扩展系统提示词。通过拼接 _build_plan_mode_prompt()，给 AI 注入“只动脑不动手、输出结构化计划”的专属指令。
       同步 OpenAI 消息：同样地，如果使用 OpenAI，也会实时更新上下文里的系统提示词。
       反馈与返回：打印进入提示（包含计划文件的路径），并返回 "plan"。
        """
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] =self._system_prompt
            print_info(f"Exited plan mode -> {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> dict:
        return {"input":self.total_input_tokens, "output":self.total_output_tokens}

    async def shutdown(self) -> None:
        """Release model transports and optional MCP subprocesses."""
        await self.drain_background_skill_tasks()
        await self._mcp_manager.disconnect_all()
        for client in (self._openai_client, self._anthropic_client):
            close = getattr(client, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    #主入口

    async def  chat(self, user_message:str)->None:
        if self.trace is None:
            return await self._chat_impl(user_message)
        with self.trace.span(
            "agent.chat",
            component="lifecycle",
            purpose=self.model_purpose,
            payload={"user_message": _safe_utf8_text(user_message)},
        ) as span:
            try:
                await self._chat_impl(user_message)
            except asyncio.CancelledError:
                span.finish(status="cancelled", attributes={"termination_reason": "cancelled"})
                raise
            span.finish(status="cancelled" if self._aborted else "ok")

    @activity("turn")
    async def _chat_impl(self, user_message: str) -> None:
        #懒加载MCP服务在第一次chat的时候
        if not self._mcp_initialized and self._mcp_enabled:
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print_error(f"MCP init failed: {e}")

        original_user_message = _safe_utf8_text(user_message)
        self._current_user_source = original_user_message
        self._task_plan = []
        initial_repo_state = await self._probe_repo_state() if self._repair_task is not None else None
        self._stop_initial_repo_state = initial_repo_state
        self._stop_repo_state_seen = initial_repo_state is not None
        self._last_verification = None
        self._verification_history = []
        self._stop_hook_blocks = 0
        self._completion_evidence = None
        self._repair_state = (
            RepairState(self._repair_task, initial_repo_state)
            if self._repair_task is not None else None
        )
        self._observations.clear()
        self._baseline_ref = None
        self._baseline_attempts.clear()
        self._diagnostic_checks.clear()
        if self._repair_state is not None:
            script = "import subprocess,sys; r=subprocess.run(['git','status','--porcelain'],stdout=subprocess.PIPE); sys.exit(1) if r.returncode or r.stdout else None; subprocess.run(['git','rev-parse','HEAD'],check=True)"
            base = await self._execute_tool_call_impl("run_shell", {"command": python_command(script, [], self._repair_state.task.repo_root)})
            if isinstance(base, ShellResult) and not base.failed and re.fullmatch(r"[0-9a-f]{40,64}", base.stdout.strip()):
                self._baseline_ref = base.stdout.strip()
            if self.trace:
                self.trace.event("verification.baseline_captured", component="state", purpose="initialization",
                                 attributes={"baseline_ref": self._baseline_ref, "repo_state": initial_repo_state})
        ready_skill_extraction_window: dict[str, Any] | None = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        self._loaded_skill_context = ""
        self._loaded_skill_receipt = None
        if self._skills_enabled:
            ready_skill_extraction_window = self._pop_pending_skill_extraction_window(original_user_message)
            user_message, self._last_retrieved_skill_reference = self._augment_user_message_with_skill_context(
                original_user_message
            )

        self._aborted = False
        self._turn_output_buffer = []
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        self._current_task = asyncio.create_task(coro)
        try:
            await self._current_task
        except _BudgetExceeded as error:
            print_info(f"Budget exceeded: {error}")
        except asyncio.CancelledError:
            requested_abort = self._aborted
            self._aborted = True
            self.termination_reason = {"kind": "cancelled"}
            if not requested_abort:
                raise

        finally:
            self._current_task = None
            self._current_user_source = None
        if not self._aborted and self.termination_reason and self.termination_reason.get("kind") in {
            "max_cost", "max_turns", "max_model_calls", "max_tool_calls",
            "input_budget", "token_budget", "max_decisions",
        }:
            await self._finalize_budget_exit()
        assistant_text = "".join(self._turn_output_buffer or []).strip()
        self._turn_output_buffer = None
        if self._skills_enabled and not self._aborted:
            self._schedule_background_skill_task(self._run_skill_usage_tracking(original_user_message, assistant_text))
            if ready_skill_extraction_window:
                self._schedule_background_skill_task(self._run_online_skill_evolution(ready_skill_extraction_window))
            self._set_pending_skill_extraction_window(
                original_user_message=original_user_message,
                assistant_text=assistant_text,
                retrieved_reference=self._last_retrieved_skill_reference,
            )
        if self._session_persistence_enabled:
            print_divider()
            self._auto_save()



   #执行一次对话，收集本轮模型输出文本，并返回本轮消耗的 token 数
    async def run_once(self, prompt: str) -> dict[str, Any]:
        if self.trace is None:
            return await self._run_once_impl(prompt)
        with self.trace.span(
            "agent.run_once",
            component="lifecycle",
            purpose=self.model_purpose,
        ) as span:
            try:
                result = await self._run_once_impl(prompt)
            except asyncio.CancelledError:
                span.finish(status="cancelled", attributes={"termination_reason": "cancelled"})
                raise
            cancelled = result.get("termination_reason") == {"kind": "cancelled"}
            span.finish(
                status="cancelled" if cancelled else "ok",
                metrics={"tokens": result["tokens"]},
                payload={"text": result["text"]},
            )
            return result

    async def _run_once_impl(self, prompt: str) -> dict[str, Any]:
        self.termination_reason = None
        self._output_buffer = []
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        prev_model_calls = self._operation_counts["model_calls"]
        prev_tool_calls = self._operation_counts["tool_calls"]
        await self.chat(prompt)
        text = "".join(self._output_buffer)
        self._output_buffer = None
        return {
            "text": text,
            "tokens":{
                "input":self.total_input_tokens-prev_in,
                "output":self.total_output_tokens-prev_out
            },
            "model_calls": self._operation_counts["model_calls"] - prev_model_calls,
            "tool_calls": self._operation_counts["tool_calls"] - prev_tool_calls,
            "termination_reason": self.termination_reason,
            "completion_evidence": self._completion_evidence,
        }

    #输出工具：统一处理模型输出文本。根据当前是否处于“收集输出”的模式
    # 决定是把文本存进缓冲区，还是直接打印到终端。
    def _emit_text(self, text:str)->None:
        text = _safe_utf8_text(text)
        if self._turn_output_buffer is not None:
            self._turn_output_buffer.append(text)
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    def _refresh_runtime_system_prompt(self) -> None:
        if self._system_prompt_frozen:
            return
        self._base_system_prompt, self._prompt_sections = build_system_prompt_with_sections(
            include_skills=self._skills_enabled, context=self._system_prompt_context,
        )
        if self.permission_mode == "plan":
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt
        self._runtime_guidance = ""
        if self.use_openai and self._openai_messages:
            self._openai_messages[0]["content"] = self._system_prompt
        self._system_prompt_frozen = True

    def _model_trace_attributes(
        self, provider: str, tools: list[Any], *, system: str | None = None, purpose: str | None = None
    ) -> dict[str, Any]:
        system_text = self._system_prompt if system is None else system
        scope = purpose or self.model_purpose
        system_sha = hashlib.sha256(system_text.encode("utf-8")).hexdigest()
        tools_bytes = json.dumps(
            tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        tools_sha = hashlib.sha256(tools_bytes).hexdigest()
        prefix_sha = hashlib.sha256(
            system_text.encode("utf-8") + b"\0" + tools_bytes
        ).hexdigest()
        previous = self._last_prefix_sha256.get(scope)
        changed = bool(previous and previous != prefix_sha)
        self._last_prefix_sha256[scope] = prefix_sha
        return {
            "provider": provider,
            "model": self.model,
            "reasoning_mode": self._thinking_mode,
            "system_sha256": system_sha,
            "tools_sha256": tools_sha,
            "prefix_sha256": prefix_sha,
            "prefix_changed": changed,
            "prefix_scope": scope,
        }

    def _record_tool_outcome(self, tool_name: str, success: bool) -> None:
        if tool_name == self._last_tool_name:
            self._same_tool_repeat_count += 1
        else:
            self._same_tool_repeat_count = 1
        self._last_tool_name = tool_name
        if success:
            self._tool_error_streak = 0
        else:
            self._tool_error_streak += 1

    @staticmethod
    def _is_verification_command(command: str) -> bool:
        command = str(command or "")
        if "&" not in command.replace("&&", "") and pytest_selection(command, str(Path.cwd())):
            return True
        command = command.lower()
        if any(marker in command for marker in ("||", ";", "\n", "|", "$(", "`")):
            return False
        if "&" in command.replace("&&", ""):
            return False
        for segment in command.split("&&"):
            try:
                tokens = shlex.split(segment.replace("\\", "/"))
            except ValueError:
                continue
            if any(token in {
                "--collect-only", "--collectonly", "--co", "--help", "-h", "--version", "--fixtures",
                "--fixtures-per-test", "--markers", "--no-run",
            } for token in tokens):
                continue
            while tokens and ("=" in tokens[0] or tokens[0] in {"env", "sudo"}):
                tokens.pop(0)
            if len(tokens) >= 2 and tokens[:2] in (["uv", "run"], ["poetry", "run"], ["pdm", "run"]):
                tokens = tokens[2:]
            if not tokens:
                continue
            executable = tokens[0].replace("\\", "/").rsplit("/", 1)[-1]
            for suffix in (".exe", ".bat", ".cmd"):
                if executable.endswith(suffix):
                    executable = executable[:-len(suffix)]
                    break
            normalized = " ".join([executable, *tokens[1:]])
            if re.match(r"(?:python(?:\d+(?:\.\d+)?)?|py)$", executable):
                runner_tokens = tokens[1:]
                while runner_tokens and runner_tokens[0] in {"-b", "-u", "-i", "-e", "-s"}:
                    runner_tokens = runner_tokens[1:]
                if len(runner_tokens) >= 2 and runner_tokens[0] == "-m":
                    normalized = " ".join(runner_tokens[1:])
            if any(normalized == marker or normalized.startswith(marker + " ") for marker in VERIFICATION_COMMAND_MARKERS):
                return True
        return False

    @staticmethod
    def _pytest_tokens(command: str) -> list[str]:
        try:
            tokens = shlex.split(command.replace("\\", "/"))
        except ValueError:
            return []
        return tokens if any(token.rsplit("/", 1)[-1].lower() in {"pytest", "pytest.exe"} for token in tokens) else []

    def _execute_update_plan_tool(self, inp: dict) -> str:
        plan = inp.get("plan")
        if not isinstance(plan, list):
            return "Error: plan must be a list."
        cleaned: list[dict[str, str]] = []
        for item in plan:
            if not isinstance(item, dict) or not str(item.get("step") or "").strip():
                return "Error: every plan item needs a non-empty step."
            status = str(item.get("status") or "")
            if status not in {"pending", "in_progress", "completed"}:
                return f"Error: invalid plan status {status!r}."
            cleaned.append({"step": str(item["step"]).strip(), "status": status})
        if sum(item["status"] == "in_progress" for item in cleaned) > 1:
            return "Error: at most one plan item can be in progress."
        self._task_plan = cleaned
        explanation = str(inp.get("explanation") or "").strip()
        result = {"plan": cleaned}
        if explanation:
            result["explanation"] = explanation
        return json.dumps(result, ensure_ascii=False)

    def _execute_update_work_state_tool(self, inp: dict) -> str:
        if self._repair_state is None:
            return "Error: update_work_state is available only for repository repair tasks."
        try:
            candidate = copy.deepcopy(self._repair_state)
            for item in inp.get("acceptance_cases") or []:
                candidate.register_acceptance_case(
                    item.get("id"), item.get("behavior"), item.get("source_quote"),
                    source_kind=item.get("source_kind", "issue"),
                    source_ref=item.get("source_ref"),
                )
            if (inp.get("acceptance_cases") and not self._repair_state.acceptance_cases
                    and not any(case.source_kind == "issue" for case in candidate.acceptance_cases.values())):
                raise ValueError(
                    "Initial acceptance registration requires at least one issue-derived case. "
                    "No acceptance IDs or hypotheses were saved. Use source_kind='issue' "
                    "for an exact canonical task quote and retry the same IDs."
                )
            for item in inp.get("hypotheses") or []:
                candidate.upsert_hypothesis(
                    item.get("id"), item.get("claim"), item.get("prediction"),
                    item.get("falsifier"), status=item.get("status", "active"),
                    evidence_ids=item.get("evidence_ids") or (),
                )
        except (AttributeError, TypeError, ValueError) as error:
            return f"Error updating work state: {error}"
        self._repair_state = candidate
        if self.trace is not None:
            self.trace.event(
                "state.change", component="state", purpose="repair_state", status="ok",
                payload=candidate.snapshot(),
            )
        return candidate.render_runtime_context()

    @staticmethod
    def _normalized_verification_command(command: str) -> str:
        # Whitespace inside shell quotes is significant to check identity.
        return str(command or "").strip()

    def _observe_stop_gate_tool(
        self, name: str, inp: dict, result: str | ShellResult, before: str | None, after: str | None,
    ) -> None:
        if not self._stop_repo_state_seen and before is not None:
            self._stop_initial_repo_state = before
            self._stop_repo_state_seen = True
        if self._repair_state is not None:
            return
        if name == "run_shell" and self._is_verification_command(str(inp.get("command") or "")):
            success = isinstance(result, ShellResult) and not result.failed
            if self._pytest_tokens(str(inp.get("command") or "")):
                success = success and result.test_outcome == "passed"
            self._last_verification = {
                "command": str(inp.get("command") or ""),
                "operation_key": self._normalized_verification_command(inp.get("command") or ""),
                "success": success,
                "repo_state": after,
                "sequence": len(self._verification_history) + 1,
            }
            self._verification_history.append(dict(self._last_verification))

    def _guard_openai_response(self, response: dict) -> dict:
        choice = (response.get("choices") or [{}])[0]
        self.finish_reason = choice.get("raw_finish_reason", choice.get("finish_reason"))
        self._response_observed = True
        if self.finish_reason not in {"stop", "tool_calls"}:
            response = copy.deepcopy(response)
            if response.get("choices"):
                response["choices"][0].setdefault("message", {})["tool_calls"] = []
        return response

    def _guard_anthropic_response(self, response):
        self.finish_reason = getattr(response, "stop_reason", None)
        self._response_observed = True
        if self.finish_reason not in {"end_turn", "tool_use"}:
            response = copy.copy(response)
            response.content = [block for block in response.content if getattr(block, "type", None) != "tool_use"]
        return response

    def _incomplete_response_feedback(self) -> str | None:
        if self.finish_reason in {"length", "max_tokens"} and self._length_recoveries < 2:
            self._length_recoveries += 1
            if self.trace:
                self.trace.event("model.truncation_recovery", component="model", purpose=self.model_purpose,
                                 attributes={"finish_reason": self.finish_reason,
                                             "recovery": self._length_recoveries, "limit": 2})
            return (
                "The previous response was truncated by the output limit; its tool batch was rejected. "
                "Inspect actual earlier tool receipts and current files before continuing. "
                "Continue with small edits, smaller tool arguments and concise output. "
                "Verify the current result before finishing."
            )
        self.termination_reason = {"kind": "incomplete_response", "detail": self.finish_reason or "missing_finish_reason"}
        return None

    @activity("completion")
    async def _run_stop_hooks(self) -> str | None:
        if (self._finalization_calls is None and self._response_observed
                and self.finish_reason not in {"stop", "end_turn"}):
            return self._incomplete_response_feedback()
        current = await self._probe_repo_state()
        if not self._stop_repo_state_seen and current is not None:
            self._stop_initial_repo_state = current
            self._stop_repo_state_seen = True
        incomplete = [item["step"] for item in self._task_plan if item["status"] != "completed"]
        changed = (
            self._stop_repo_state_seen
            and current is not None
            and self._stop_initial_repo_state != current
        )
        reason: str | None = None
        if self._repair_state is not None:
            if current is not None:
                self._repair_state.observe_repo_state(current)
            current_diff_receipts = [
                receipt for receipt in self._repair_state.verifications
                if receipt.scope == "diff"
                and receipt.repo_generation == self._repair_state.repo_generation
                and receipt.repo_state == self._repair_state.current_repo_state
            ]
            diff_observation = None
            diff_failed = False
            if (
                self._repair_state.task.requires_patch
                and self._repair_state.repo_generation > 0
                and not any(receipt.success for receipt in current_diff_receipts)
            ):
                diff_observation = await self._execute_tool_call(
                    "run_shell", {"command": self._diff_check_command}, runtime_owned=True,
                )
                diff_failed = self._repair_state.evidence[-1].status == "error"
            await self._recheck_completion_failures()
            decision = self._repair_state.completion_decision()
            if not decision.complete:
                reason = "; ".join(decision.reasons)
                if diff_observation and diff_failed:
                    reason += f"; runtime diff check: {diff_observation}"
        elif incomplete:
            reason = f"task plan still has incomplete steps: {', '.join(incomplete)}"
        elif changed and any(tool.get("name") == "run_shell" for tool in self.tools):
            current_receipts = [
                receipt for receipt in self._verification_history
                if receipt["repo_state"] == current
            ]
            unresolved = [
                failed for failed in current_receipts if not failed["success"] and not any(
                    passed["success"]
                    and passed["sequence"] > failed["sequence"]
                    and passed["operation_key"] == failed["operation_key"]
                    for passed in current_receipts
                )
            ]
            if unresolved:
                reason = "latest verification failed: " + ", ".join(
                    receipt["command"] for receipt in unresolved
                )
            elif not any(receipt["success"] for receipt in current_receipts):
                reason = (
                    "repository changed after the latest successful verification"
                    if any(receipt["success"] for receipt in self._verification_history)
                    else "repository changed but no relevant test or check has passed on the current state"
                )
        if reason:
            self._stop_hook_blocks += 1
        if self.trace is not None:
            self.trace.event(
                "hook.stop",
                component="hook",
                purpose="completion",
                status="blocked" if reason else "ok",
                attributes={
                    "state_changed": changed,
                    "plan_items": len(self._task_plan),
                    "block_count": self._stop_hook_blocks,
                },
                payload={
                    "reason": reason,
                    "verification": self._last_verification,
                    "repair_state": self._repair_state.snapshot() if self._repair_state else None,
                },
            )
        return reason

    async def _finalize_budget_exit(self) -> None:
        """Check current evidence once, without another model request or replacing the exit reason."""
        if self._completion_evidence is not None:
            return
        original_reason = dict(self.termination_reason or {})
        started = time.monotonic()
        self._finalization_calls = 0
        self._finalization_deadline = started + 30.0
        result = {"status": "unknown", "reason": "completion check did not finish"}
        try:
            async def check():
                reason = await self._run_stop_hooks()
                current = await self._probe_repo_state()
                repair = self._repair_state
                if current is None or repair is None or current != repair.current_repo_state:
                    return {"status": "unknown", "reason": "current repository evidence unavailable"}
                decision = repair.completion_decision()
                return {"status": "complete" if not reason and decision.complete else "incomplete",
                        "reason": reason, "decision": asdict(decision), "repo_state": current}
            result = await asyncio.wait_for(check(), timeout=30.0)
        except Exception as error:
            result = {"status": "unknown", "reason": type(error).__name__}
        finally:
            result.update(tool_calls=self._finalization_calls, elapsed_ms=round((time.monotonic() - started) * 1000))
            self._finalization_calls = None
            self._finalization_deadline = 0.0
            self.termination_reason = original_reason
            self._completion_evidence = result
            if self.trace:
                self.trace.event("hook.finalization", component="hook", purpose="completion",
                                 status="ok" if result["status"] == "complete" else "blocked",
                                 payload={"termination_reason": original_reason, **result})

    def _output_marks(self) -> tuple[int | None, int | None]:
        return (
            len(self._turn_output_buffer) if self._turn_output_buffer is not None else None,
            len(self._output_buffer) if self._output_buffer is not None else None,
        )

    def _discard_output_since(self, marks: tuple[int | None, int | None]) -> None:
        turn_mark, output_mark = marks
        if turn_mark is not None and self._turn_output_buffer is not None:
            del self._turn_output_buffer[turn_mark:]
        if output_mark is not None and self._output_buffer is not None:
            del self._output_buffer[output_mark:]

    @staticmethod
    def _stop_hook_feedback(reason: str) -> str:
        return (
            f"<stop_hook>Completion blocked: {reason}. Continue from the observed tool results, "
            "finish the remaining acceptance work, and verify the final repository state.</stop_hook>"
        )

    def _messages_with_runtime_state(
        self, messages: list[dict[str, Any]], *, include_counters: bool = True,
    ) -> list[dict[str, Any]]:
        position = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                         if messages[i].get("role") == "user"
                         and isinstance(messages[i].get("_source_user_text"), str)),
                        next((i + 1 for i, msg in enumerate(messages) if msg.get("role") == "user"), len(messages)))
        messages = _sanitize_for_utf8(messages)
        for message in messages:
            message.pop("_source_user_text", None)
        if self._loaded_skill_context:
            # Pin one frozen body in the request copy; folded history cannot drop or rewrite it.
            messages.insert(position, {"role": "user", "content": self._loaded_skill_context})
        counters = []
        if include_counters and self.max_model_calls is not None:
            counters.append(
                f"model call {self._operation_counts['model_calls'] + 1}/{self.max_model_calls}"
            )
        if include_counters and self.max_tool_calls is not None:
            counters.append(
                f"tool calls completed {self._operation_counts['tool_calls']}/{self.max_tool_calls}"
            )
        notes = [f"<runtime_state>{'; '.join(counters)}</runtime_state>"] if counters else []
        if self._repair_state is not None:
            notes.append(self._repair_state.render_runtime_context())
        if not notes:
            return messages
        note = "\n".join(notes)
        for message in reversed(messages):
            if message.get("role") not in {"user", "tool"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content + f"\n\n{note}"
                break
            if isinstance(content, list):
                content.append({"type": "text", "text": note})
                break
        return messages

    def _openai_request_messages(self) -> list[dict[str, Any]]:
        return self._messages_with_runtime_state(self._openai_messages)

    def _anthropic_request_messages(self) -> list[dict[str, Any]]:
        return self._messages_with_runtime_state(self._anthropic_messages)

    def _record_skill_exposure(self, messages: list[dict[str, Any]], *, owner=None) -> None:
        owner = owner or self
        if self.trace and owner._loaded_skill_context and owner._loaded_skill_receipt:
            if any(owner._loaded_skill_context in str(message.get("content", "")) for message in messages):
                self.trace.event("skill.body_exposure", component="skill", purpose=self.model_purpose,
                                 attributes={"fold_count": self._fold_count,
                                             "model_call": self._operation_counts["model_calls"]},
                                 payload=owner._loaded_skill_receipt)
    def _trace_permission(self, tool_name: str, inp: dict, decision: dict) -> None:
        if self.trace is None:
            return
        action = str(decision.get("action") or "unknown")
        self.trace.event(
            "tool.permission",
            component="permission",
            purpose=tool_name,
            status={"allow": "ok", "deny": "denied"}.get(action, action),
            attributes={"action": action, "permission_mode": self.permission_mode},
            payload={"input": inp, "message": decision.get("message")},
        )

    def _record_fold_event(self) -> None:
        self._fold_last_time = time.time()
        self._fold_count += 1
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""
        self._last_progress_signature = ""
        if self.trace is not None:
            self.trace.event(
                "state.change",
                component="state",
                purpose="session_fold",
                status="ok",
                attributes={"fold_count": self._fold_count},
            )

    def _looks_like_tool_failure(self, tool_name: str, raw: str | ShellResult, result: str | ShellResult) -> bool:
        if isinstance(raw, ShellResult):
            return raw.failed
        if isinstance(result, ShellResult):
            return result.failed
        if tool_name == "run_shell":
            return True  # Text alone carries no execution status.
        texts = (str(raw or "").lstrip().lower(), str(result or "").lstrip().lower())
        failure_prefixes = (
            "action denied", "command failed", "command timed out", "error ", "error:", "permission denied",
            "timed out", "timeout:", "user denied", "warning: the file changed",
        )
        if any(text.startswith(failure_prefixes) for text in texts):
            return True
        if tool_name == "compact_context" and any(
            "no context compaction" in text for text in texts
        ):
            return True
        return False

    def _augment_user_message_with_skill_context(self, user_message: str) -> tuple[str, dict[str, Any] | None]:
        try:
            from .skills import format_retrieved_skill_context, load_retrieved_skill_body

            context, top_ref = format_retrieved_skill_context(user_message, limit=3)
            if top_ref:
                self._loaded_skill_context, self._loaded_skill_receipt = load_retrieved_skill_body(top_ref)
                if self.trace:
                    self.trace.event("skill.body_load", component="skill", purpose="retrieval",
                                     status="ok" if self._loaded_skill_context else "skipped",
                                     payload=self._loaded_skill_receipt)
        except Exception:
            return user_message, None
        if top_ref and isinstance(top_ref.get("all_hits"), list):
            self._last_retrieved_skill_hits = list(top_ref.get("all_hits") or [])
        if not context.strip():
            return user_message, top_ref
        return f"{user_message}\n\n{context}", top_ref

    @staticmethod
    def _strip_runtime_injections(text: str) -> str:
        raw = str(text or "")
        request, marker, metadata = raw.rpartition("\n\n<retrieved_skills>")
        if marker and metadata.rstrip().endswith("</retrieved_skills>"):
            return request.strip()
        return raw.strip()

    def _dialog_message_text(self, msg: dict[str, Any]) -> str:
        if msg.get("role") == "user" and isinstance(msg.get("_source_user_text"), str):
            return msg["_source_user_text"].strip()
        content = msg.get("content")
        if isinstance(content, str):
            return self._strip_runtime_injections(content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif block.get("type") == "tool_result":
                        parts.append(f"[tool result]\n{block.get('content') or ''}")
                    elif "content" in block and block.get("type") not in {"tool_result", "tool_use"}:
                        parts.append(str(block.get("content") or ""))
            return self._strip_runtime_injections("\n".join(parts))
        return ""

    def _recent_dialog_messages(self, *, max_messages: int = 8) -> list[dict[str, str]]:
        raw_messages = self._openai_messages if self.use_openai else self._anthropic_messages
        out: list[dict[str, str]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role == "user" and "_source_user_text" in msg and msg["_source_user_text"] is None:
                role = "tool"
            if role not in {"user", "assistant", "tool"}:
                continue
            if role == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        out.append({"role": "tool", "content": str(block.get("content") or "")[:4000]})
                    elif block.get("type") == "text":
                        text = self._strip_runtime_injections(str(block.get("text") or ""))
                        if text:
                            out.append({"role": "user", "content": text})
                continue
            text = self._dialog_message_text(msg)
            if text:
                if role == "tool":
                    text = text[:4000]
                out.append({"role": role, "content": text})
        return out[-max(2, int(max_messages)) :]

    async def _confirm_online_skill_write(self, summary: str) -> bool:
        if self.permission_mode in {"bypassPermissions", "acceptEdits"}:
            return True
        if self.permission_mode in {"plan", "dontAsk"}:
            return False
        if self.confirm_fn is None:
            return False
        print_confirmation(summary)
        try:
            return bool(await self.confirm_fn(summary))
        except Exception:
            return False

    async def _confirm_background_online_skill_write(self, summary: str) -> bool:
        return self.permission_mode in {"bypassPermissions", "acceptEdits"}

    def _online_evolution_enabled(self) -> bool:
        raw = os.environ.get("SANS_AUTO_SKILL_EVOLUTION", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _schedule_background_skill_task(self, coro) -> None:
        if self.permission_mode == "plan":
            try:
                coro.close()
            except Exception:
                pass
            return
        task = asyncio.create_task(coro)
        self._background_skill_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._background_skill_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception:
                pass

        task.add_done_callback(_done)

    async def drain_background_skill_tasks(self) -> None:
        tasks = [task for task in self._background_skill_tasks if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _pop_pending_skill_extraction_window(self, next_user_feedback: str) -> dict[str, Any] | None:
        pending = self._pending_skill_extraction_window
        self._pending_skill_extraction_window = None
        if not pending:
            return None
        messages = list(pending.get("messages") or [])
        feedback = _safe_utf8_text(next_user_feedback).strip()
        if feedback:
            messages.append({"role": "user", "content": feedback})
        pending["messages"] = ([messages[0], *messages[1:][-9:]]
                               if messages and messages[0].get("role") == "user" else messages[-10:])
        pending["next_user_feedback"] = feedback
        return pending

    def _set_pending_skill_extraction_window(
        self,
        *,
        original_user_message: str,
        assistant_text: str,
        retrieved_reference: dict[str, Any] | None,
    ) -> None:
        evidence_scope = f"{self.trace.run_id}:{self.trace.attempt_id}" if self.trace else self.session_id
        execution_evidence = [
            {"id": f"{evidence_scope}:{self._repair_state.execution_id}:{self._repair_state.task.task_id}:{item.id}", "tool": item.tool,
             "original_evidence_id": item.id, "execution_id": self._repair_state.execution_id,
             "command": item.input_summary, "operation_key": item.operation_key,
             "repo_before": item.repo_before, "repo_after": item.repo_after,
             "status": item.status, "test_failure_observed": item.test_failure_observed,
             "test_pass_observed": item.test_pass_observed, "test_cases": item.test_cases,
             "check_fingerprint": item.check_fingerprint, "repo_generation": item.repo_generation}
            for item in [e for e in self._repair_state.evidence
                         if e.check_fingerprint and (e.test_failure_observed or e.test_pass_observed) and not e.diagnostic][-12:]
        ] if self._repair_state is not None else []
        if not original_user_message.strip() or (not assistant_text.strip() and not execution_evidence):
            return
        dialog = self._recent_dialog_messages(max_messages=12)
        task_index = max((index for index, message in enumerate(dialog)
                          if message.get("role") == "user" and message.get("content") == original_user_message), default=-1)
        replay_messages = dialog[:task_index + 1] if task_index >= 0 else [{"role": "user", "content": original_user_message}]
        recent = [
            message for message in dialog
            if not (
                message.get("role") == "user"
                and message.get("content") == original_user_message
            )
        ]
        self._pending_skill_extraction_window = {
            "messages": [
                {"role": "user", "content": original_user_message},
                *recent[-11:],
            ],
            "latest_user": original_user_message,
            "latest_assistant": assistant_text,
            "retrieved_reference": self._compact_retrieved_reference(retrieved_reference),
            "session_id": self.session_id,
            "execution_evidence": execution_evidence,
            "replay_messages": replay_messages,
        }

    def _compact_retrieved_reference(self, ref: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ref:
            return None
        return {k: v for k, v in ref.items() if k != "all_hits"}

    async def _run_online_skill_evolution(self, window: dict[str, Any], *, interactive_confirm: bool = False) -> dict[str, Any]:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return {"ok": False, "action": "disabled"}
        messages = list(window.get("messages") or [])
        if not messages:
            return {"ok": False, "action": "empty"}

        side_query = self._build_side_query(max_tokens=2200, purpose="skill_extract")
        if side_query is None:
            return {"ok": False, "action": "unavailable"}

        try:
            from .online_skill_evolution import online_ingest
        except Exception:
            return {"ok": False, "action": "unavailable"}

        result = await online_ingest(
            messages=messages,
            side_query=side_query,
            retrieved_reference=window.get("retrieved_reference") or None,
            hint=str(window.get("hint") or ""),
            confirm_write=self._confirm_online_skill_write if interactive_confirm else self._confirm_background_online_skill_write,
            target=os.environ.get("SANS_AUTO_SKILL_TARGET", "project"),
            required_target_skill=os.environ.get("SANS_EVOLUTION_TARGET_SKILL", ""),
            execution_evidence=window.get("execution_evidence") or [],
            replay_messages=window.get("replay_messages"),
            task_evaluator=self._skill_task_evaluator,
        )
        if self.trace:
            self.trace.event("skill.candidate_decision", component="skill", purpose="learning",
                             status="ok" if result.get("ok") else "error",
                             attributes={"action": result.get("action"), "skill": result.get("skill"),
                                         "promotion_status": (result.get("promotion") or {}).get("status")})
        if result.get("ok"):
            if result.get("action") in {"add", "merge"}:
                self._refresh_runtime_system_prompt()
                print_info(f"Online skill {result.get('action')}: {result.get('skill')}")
            elif result.get("action") == "staged":
                print_info(f"Skill candidate saved: {result.get('skill')}; awaiting task evaluation. Use /skill-candidates.")
        elif result.get("action") not in {"add_denied", "merge_denied"}:
            print_error(f"Online skill evolution failed: {result.get('error') or result}")
        return result

    async def _run_skill_usage_tracking(self, original_user_message: str, assistant_text: str) -> None:
        if not self._online_evolution_enabled() or self.permission_mode == "plan":
            return
        hits = list(self._last_retrieved_skill_hits or [])
        if not hits or not assistant_text.strip():
            return
        side_query = self._build_side_query(max_tokens=700, purpose="skill_usage_judge")
        try:
            from .online_skill_evolution import judge_retrieved_skill_usage
            from .skills import record_usage_judgments

            judgments = await judge_retrieved_skill_usage(
                hits=hits,
                user_message=original_user_message,
                assistant_text=assistant_text,
                side_query=side_query,
            )
            result = record_usage_judgments(judgments)
            if result.get("pruned"):
                self._refresh_runtime_system_prompt()
        except Exception:
            return

    async def extract_now(self, hint: str = "") -> dict[str, Any]:
        pending = self._pending_skill_extraction_window
        if not pending:
            return {"ok": False, "error": "no pending online skill extraction window"}
        window = dict(pending)
        window["hint"] = hint
        if hint.strip():
            window["messages"] = [
                *list(window.get("messages") or []),
                {"role": "user", "content": hint.strip()},
            ]
        result = await self._run_online_skill_evolution(window, interactive_confirm=True)
        self._pending_skill_extraction_window = None
        return result


    def clear_history(self)->None:
        self._anthropic_messages = []
        self._openai_messages = []
        self._pending_skill_extraction_window = None
        self._last_retrieved_skill_reference = None
        self._last_retrieved_skill_hits = []
        self._loaded_skill_context = ""
        self._loaded_skill_receipt = None
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._failed_compaction_signature = None
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""
        self._last_progress_signature = ""
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content":self._system_prompt})
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_provider_reported_cost = 0.0
        self._provider_cost_complete = True
        self.last_input_token_count = 0
        print_info("Conversation cleared.")

    def show_cost(self):
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        label = "Provider-reported cost" if self._provider_cost_complete else "Legacy estimate"
        print_info(
            f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out\n  {label}: ${total:.8f}{budget_info}{turn_info}")

    #获取当前的花费，
    def _get_current_cost_usd(self) -> float:
        if self._provider_cost_complete:
            return self.total_provider_reported_cost
        return (self.total_input_tokens / 1_000_000) * 3 + (self.total_output_tokens / 1_000_000) * 15

    def _record_provider_cost(self, usage: Any) -> None:
        cost = usage.get("cost") if isinstance(usage, dict) else getattr(usage, "cost", None)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            if self._provider_cost_complete:
                self.total_provider_reported_cost += float(cost)
        else:
            self._provider_cost_complete = False

    def _record_normalized_usage(self, usage: dict[str, int | float | None]) -> None:
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
            self.total_input_tokens += input_tokens
        if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
            self.total_output_tokens += output_tokens
        self._record_provider_cost({"cost": usage.get("provider_reported_cost")})

    #检查预算
    def _raise_if_call_exhausted(self, counter: str, limit: int | None) -> None:
        if limit is None or self._operation_counts[counter] < limit:
            return
        kind = f"max_{counter}"
        self.termination_reason = {"kind": kind}
        raise _BudgetExceeded(f"{kind} limit reached ({limit})")

    def _reserve_call(self, counter: str, limit: int | None) -> None:
        self._raise_if_call_exhausted(counter, limit)
        self._operation_counts[counter] += 1

    def _check_budget(self) -> dict:
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "kind": "max_cost", "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "kind": "max_turns", "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    def _stop_before_model_if_cost_exhausted(self) -> bool:
        if self.max_cost_usd is None or self._get_current_cost_usd() < self.max_cost_usd:
            return False
        self.termination_reason = {"kind": "max_cost"}
        print_info(
            f"Budget exceeded: Cost limit reached "
            f"(${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"
        )
        return True

    #压缩会话
    async def compact(self)->None:
        compacted = await self._compact_conversation(trigger="manual")
        if not compacted:
            print_info("Nothing to compact yet.")


    #恢复会话信息
    def restore_session(self, data:dict)->None:
        if data.get("anthropicMessages"):
            self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(data["anthropicMessages"]))
        if data.get("openaiMessages"):
            self._openai_messages = _sanitize_for_utf8(data["openaiMessages"])
        if isinstance(data.get("foldedSessionMemories"), list):
            self._folded_session_memories = _sanitize_for_utf8(data["foldedSessionMemories"])
        print_info(f"Session restored ({self._get_message_count()} messages).")



#整理 Anthropic 的历史消息，修正部分角色错误，并丢弃不合法的工具调用消息。
    def _normalize_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        role_normalized = []
        for msg in messages:
            copied = dict(msg)
            content = copied.get("content")
            if copied.get("role") == "user" and isinstance(content, list):
                if any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content):
                    copied["role"] = "assistant"
            role_normalized.append(copied)

        normalized = []
        i = 0
        while i < len(role_normalized):
            msg = role_normalized[i]
            tool_use_ids = self._anthropic_tool_use_ids(msg)
            if not tool_use_ids:
                normalized.append(msg)
                i += 1
                continue

            next_msg = role_normalized[i + 1] if i + 1 < len(role_normalized) else None
            result_ids = self._anthropic_tool_result_ids(next_msg) if next_msg else set()
            if tool_use_ids.issubset(result_ids):
                normalized.append(msg)
                normalized.append(next_msg)
                i += 2
                continue

            i += 1
        return normalized

    @staticmethod
    def _anthropic_tool_use_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        }

    @staticmethod
    def _anthropic_tool_result_ids(msg: dict | None) -> set[str]:
        if not msg or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            return set()
        return {
            block.get("tool_use_id")
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
        }

    def _get_message_count(self) -> int:
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self) -> None:
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self._get_message_count(),
                },
                "anthropicMessages": _sanitize_for_utf8(self._anthropic_messages) if not self.use_openai else None,
                "openaiMessages": _sanitize_for_utf8(self._openai_messages) if self.use_openai else None,
                "foldedSessionMemories": _sanitize_for_utf8(self._folded_session_memories),
            })
        except Exception:
            pass

    def _conversation_input_estimate(self, messages: list[dict[str, Any]]) -> int:
        active_tools = get_active_tool_definitions(self.tools)
        tools = _to_openai_tools(active_tools) if self.use_openai else active_tools
        system = None if self.use_openai else self._system_prompt
        return self._estimate_model_input_tokens(
            messages=self._messages_with_runtime_state(messages) if messages else messages,
            tools=tools, system=system,
        )

    def _compaction_token_budgets(self) -> tuple[int, int, int]:
        folded_shell = [{"role": "assistant", "content": format_folded_memory("")}]
        source_messages = self._openai_messages[1:] if self.use_openai else self._anthropic_messages
        canonical = self._canonical_issue(source_messages)
        if canonical:
            folded_shell.insert(0, {"role": "user", "content": canonical})
        if self.use_openai:
            folded_shell.insert(0, {"role": "system", "content": self._system_prompt})
        overhead = self._conversation_input_estimate(folded_shell)
        updates = self._source_user_updates(source_messages, canonical)
        if updates:
            overhead = self._conversation_input_estimate([*folded_shell, *updates])
        available = max(
            2,
            self.context_window - self._compaction_reserve_tokens() - overhead - 256,
        )
        summary_limit = COMPACTION_SUMMARY_MAX_TOKENS
        if self.max_output_tokens is not None:
            summary_limit = min(summary_limit, self.max_output_tokens)
        summary_footprint = max(1, available // 3)
        summary = min(summary_limit, max(1, summary_footprint * 3 // 4))
        summary_footprint = min(summary_footprint, (summary * 4 + 2) // 3)
        recent = min(
            COMPACTION_KEEP_RECENT_TOKENS,
            max(1, available - summary_footprint),
        )
        return summary, recent, overhead

    def _compaction_reserve_tokens(self) -> int:
        if self.max_output_tokens is None:
            return COMPACTION_RESERVE_TOKENS
        return min(COMPACTION_RESERVE_TOKENS, self.max_output_tokens + 2048)

    #自动压缩
    async def _check_and_compact(self)->None:
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        last_input = self.last_input_token_count
        conversation_estimate = self._conversation_input_estimate(messages)
        pressure = max(last_input, conversation_estimate)
        soft_limit = max(1, self.context_window - self._compaction_reserve_tokens())
        if pressure > soft_limit:
            if trace := getattr(self, "trace", None):
                trace.event("context.fold.requested", component="context", purpose="compaction",
                                 attributes={"trigger": "auto", "last_input_tokens": last_input,
                                             "conversation_input_estimate": conversation_estimate,
                                             "soft_limit": soft_limit, "pressure": pressure})
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation(trigger="auto")

    @activity("compaction")
    async def _compact_conversation(self, *, trigger: str = "manual")->bool:
        messages = self._openai_messages if self.use_openai else self._anthropic_messages
        signature = hashlib.sha256(json.dumps([messages, self._canonical_issue(messages), self.context_window,
            self.max_output_tokens, self._compaction_policy()], ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
        automatic = trigger in {"auto", "periodic", "budget"}
        if automatic and signature == self._failed_compaction_signature:
            if self.trace:
                self.trace.event("context.fold_skipped", component="context", purpose="compaction",
                                 attributes={"reason": "unchanged_source_after_failure", "source_sha256": signature})
            return False
        if self.use_openai:
            compacted = await self._compact_openai(trigger=trigger)
        else:
            compacted = await self._compact_anthropic(trigger=trigger)
        if compacted:
            self._failed_compaction_signature = None
            print_info("Conversation compacted.")
        elif compacted is False and automatic:
            self._failed_compaction_signature = signature
        return bool(compacted)

    @staticmethod
    def _message_calls_tool(message: dict[str, Any], tool_name: str) -> bool:
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and (call.get("function") or {}).get("name") == tool_name:
                return True
        content = message.get("content")
        return isinstance(content, list) and any(
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == tool_name
            for block in content
        )

    def _compaction_history_parts(
        self, messages: list[dict[str, Any]], *, start: int, trigger: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None] | None:
        end = len(messages)
        if trigger == "tool":
            for index in range(end - 1, start - 1, -1):
                if self._message_calls_tool(messages[index], "compact_context"):
                    end = index
                    break
        previous_summary = None
        if start < end:
            previous_summary = folded_memory_summary(messages[start])
            if previous_summary is not None:
                start += 1
        active = messages[:end]
        summary_tokens, keep_recent_tokens, overhead = self._compaction_token_budgets()
        cut = compaction_cut_index(
            active, start=start, keep_recent_tokens=keep_recent_tokens,
        )
        if cut is None:
            # Keep the latest exchange intact when it exceeds the usual tail
            # allowance but still fits alongside the summary and runtime state.
            room = self.context_window - self._compaction_reserve_tokens() - overhead - (summary_tokens * 4 + 2) // 3 - 256
            if room > keep_recent_tokens:
                cut = compaction_cut_index(active, start=start, keep_recent_tokens=room,
                                           latest_exchange_only=True)
        if cut is None:
            return None
        return active[start:cut], active[cut:end], previous_summary

    def _compaction_policy(self):
        return SUMMARIZATION_SYSTEM_PROMPT, (COMPACTION_PROMPT, COMPACTION_RETRY_PROMPT)

    async def _run_compaction_query(
        self, history: list[dict[str, Any]], prompt: str,
        previous_summary: str | None = None,
        recent_context: list[dict[str, Any]] | None = None,
        *, retry_with_full_output: bool = False,
    ) -> str:
        messages = self._openai_messages[1:] if self.use_openai else self._anthropic_messages
        canonical = self._canonical_issue(messages)
        summary_prompt = f"<canonical-task>\n{canonical}\n</canonical-task>\n\n" if canonical else ""
        if previous_summary is not None:
            summary_prompt += (
                f"<previous-checkpoint>\n{previous_summary}"
                "\n</previous-checkpoint>\n\n"
            )
        summary_prompt += (
            f"<new-archived-evidence>\n{serialize_compaction_history(history)}"
            "\n</new-archived-evidence>\n\n"
        )
        if recent_context:
            summary_prompt += (
                "<retained-recent-context>\n"
                f"{serialize_compaction_history(recent_context)}\n"
                "</retained-recent-context>\n\n"
            )
        summary_prompt += prompt
        system, _ = self._compaction_policy()
        summary_tokens, _, _ = self._compaction_token_budgets()
        if retry_with_full_output:
            summary_tokens = self.max_output_tokens or _get_max_output_tokens(self.model)
        return await self._side_query_once(system, summary_prompt, max_tokens=summary_tokens,
                                           purpose="session_fold", disable_thinking=True)

    async def _generate_folded_session_memory(
        self, history: list[dict[str, Any]], previous_summary: str | None = None,
        recent_context: list[dict[str, Any]] | None = None,
    ) -> str | None:
        request_history = _sanitize_for_utf8(history)
        _, prompts = self._compaction_policy()
        retry_options = {}
        for attempt, prompt in enumerate(prompts):
            while True:
                if self.max_model_calls is not None and self._operation_counts["model_calls"] >= self.max_model_calls - 1:
                    return None  # Preserve the final model call for continuation or an outcome report.
                previous_termination = self.termination_reason
                try:
                    summary = await self._run_compaction_query(
                        request_history, prompt, previous_summary, recent_context,
                        **retry_options,
                    )
                    return validate_folded_memory(summary)
                except Exception as error:
                    if _is_context_window_error(error):
                        if (isinstance(error, _BudgetExceeded) and self.termination_reason
                                and self.termination_reason.get("kind") == "input_budget"):
                            self.termination_reason = previous_termination
                        logging.warning("session fold exceeded context; preserving raw history")
                        return None
                    if isinstance(error, ValueError) and attempt == 0:
                        if isinstance(error, _IncompleteSideResult) and error.finish_reason in {"length", "max_tokens"}:
                            retry_options = {"retry_with_full_output": True}
                        logging.warning("compaction summary unusable; retrying once: %s", error)
                        break
                    logging.warning("session fold failed; preserving raw history: %s", error)
                    return None
        return None

    def _compaction_candidate_fits(self, candidate, *, trigger):
        if self._conversation_input_estimate(candidate) > self.context_window:
            logging.warning("folded request exceeds input limit; preserving raw history")
            return False
        return True

    async def _compact_anthropic(self, *, trigger: str)->bool | None:
        messages = _sanitize_for_utf8(self._anthropic_messages)
        canonical = self._canonical_issue(messages)
        start = 1 if canonical and self._plain_message_text(messages[0] if messages else {}) == canonical else 0
        parts = self._compaction_history_parts(messages, start=start, trigger=trigger)
        if parts is None:
            return None  # Admission skipped; no summary attempt failed.
        history, retained_tail, previous_summary = parts
        summary = await self._generate_folded_session_memory(
            history, previous_summary, retained_tail,
        )
        if summary is None:
            return False
        candidate = self._folded_history(canonical, history, summary, retained_tail,
                                         canonical_is_first=start == 1)
        admission = self._compaction_candidate_fits(candidate, trigger=trigger)
        if admission is not True:
            return admission
        self._record_folded_session_memory(trigger, summary)
        self._record_fold_event()
        self._anthropic_messages = candidate
        self.last_input_token_count = 0
        self._refresh_runtime_system_prompt()
        return True

    async def _compact_openai(self, *, trigger: str)->bool | None:
        messages = _sanitize_for_utf8(self._openai_messages)
        if not messages:
            return False
        system_msg = messages[0]
        canonical = self._canonical_issue(messages[1:])
        start = 2 if canonical and len(messages) > 1 and self._plain_message_text(messages[1]) == canonical else 1
        parts = self._compaction_history_parts(messages, start=start, trigger=trigger)
        if parts is None:
            return None  # Admission skipped; no summary attempt failed.
        history, retained_tail, previous_summary = parts
        summary = await self._generate_folded_session_memory(
            history, previous_summary, retained_tail,
        )
        if summary is None:
            return False
        candidate = [system_msg, *self._folded_history(canonical, history, summary, retained_tail,
                                                       canonical_is_first=start == 2)]
        admission = self._compaction_candidate_fits(candidate, trigger=trigger)
        if admission is not True:
            return admission
        self._record_folded_session_memory(trigger, summary)
        self._record_fold_event()
        self._openai_messages = candidate
        self.last_input_token_count=0
        self._refresh_runtime_system_prompt()
        return True

    @staticmethod
    def _source_user_updates(history, canonical):
        return [{"role": "user", "content": text,
                 **({"_source_user_text": text} if "_source_user_text" in message else {})} for message in history
                if message.get("role") == "user" and (text := Agent._plain_message_text(message))
                and text != canonical
                and ("_source_user_text" in message or not text.lstrip().startswith("<stop_hook>"))]

    @staticmethod
    def _folded_history(canonical, history, summary, retained_tail, *, canonical_is_first):
        # Independent user revisions remain authoritative even when the summary omits them.
        updates = Agent._source_user_updates(history, canonical)
        task = [{"role": "user", "content": canonical, "_source_user_text": canonical}] if canonical else []
        memory = [{"role": "assistant", "content": format_folded_memory(summary)}]
        # An externally pinned current task follows older user requests from earlier episodes.
        return [*(task if canonical_is_first else []), *memory, *updates,
                *(task if not canonical_is_first else []), *retained_tail]

    @staticmethod
    def _plain_message_text(message: dict[str, Any]) -> str | None:
        if "_source_user_text" in message and message["_source_user_text"] is None:
            return None
        if isinstance(message.get("_source_user_text"), str):
            return message["_source_user_text"].strip()
        content = message.get("content")
        return Agent._strip_runtime_injections(content) if isinstance(content, str) else None

    def _canonical_issue(self, messages: list[dict[str, Any]]) -> str | None:
        if self._repair_task is not None:
            return self._repair_task.canonical_issue
        for message in messages:
            content = self._plain_message_text(message)
            if (
                message.get("role") == "user"
                and content
                and "<session-folded-memory>" not in content
                and "<stop_hook>" not in content
            ):
                return content
        return None

    def _record_folded_session_memory(self, trigger: str, summary: str) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trigger": trigger,
            "session_id": self.session_id,
            "summary": summary,
        }
        self._folded_session_memories.append(record)
        if self._session_persistence_enabled:
            try:
                save_folded_session_memory(self.session_id, _sanitize_for_utf8(record))
            except Exception:
                pass

    def _run_compression_pipeline(self)->None:
        return None

    #大结果持久化
    #如果工具返回的结果太大（超过 30KB），不要硬塞进上下文里，而是把它存成一个临时文件。
    # 然后在对话里只留一个‘文件路径’和‘内容预览’。如果模型后面还需要看完整内容，它可以再次调用工具去读取这个文件

    def _persist_large_result(self, tool_name: str, result: str, *,
                              threshold: int = 30 * 1024, preview_chars: int = 4000) -> str:
        #转换成字节
        if len(result.encode()) <= threshold:
            return result

        receipt_header = re.match(r"<tool_observation>([^\n]+)</tool_observation>", result)
        try:
            evidence_id = json.loads(receipt_header.group(1)).get("evidence_id") if receipt_header else None
        except (ValueError, AttributeError):
            evidence_id = None
        if not isinstance(evidence_id, str):
            evidence_id = None
        if evidence_id in self._observations:
            location = f"observation {evidence_id} (read_observation)"
        elif self.trace is not None:
            payload_ref = self.trace.store_payload({"tool": tool_name, "output": result})
            location = f"trace blob {payload_ref['sha256']}"
        else:
            d = Path.home() / ".sans" / "tool-results"
            d.mkdir(parents=True, exist_ok=True)
            filepath = d / f"{uuid.uuid4().hex}-{tool_name}.txt"
            filepath.write_text(result, encoding="utf-8")
            location = str(filepath)

        lines = result.split("\n")
        if len(result) <= 2 * preview_chars:
            preview = result
        else:
            preview = (
                f"{result[:preview_chars]}\n...[{len(result) - 2 * preview_chars} chars omitted]...\n"
                f"{result[-preview_chars:]}"
            )
        size_kb = len(result.encode()) / 1024
        instruction = (
            "Use read_observation with the evidence_id to retrieve omitted lines."
            if evidence_id in self._observations else "Run a narrower command if more detail is needed."
        )

        return (
            f"[Result too large ({size_kb:.1f} KB, {len(lines)} lines). "
            f"Full output preserved in {location}. "
            f"{instruction}]\n\n"
            f"Preview (head+tail, capped at {2 * preview_chars} chars):\n{preview}"
        )

    #执行工具入口

    async def _probe_repo_state(self) -> str | None:
        if self._repo_state_probe is None:
            timeout = min(10.0, self._finalization_deadline - time.monotonic()) if self._finalization_calls is not None else 10.0
            if timeout <= 0:
                return None
            return await asyncio.to_thread(repo_state_hash, Path.cwd(), timeout_seconds=timeout)
        if self._finalization_calls is not None and not inspect.iscoroutinefunction(self._repo_state_probe):
            # A foreign synchronous callback cannot be interrupted safely during bounded finalization.
            return None
        state = self._repo_state_probe()
        return await state if inspect.isawaitable(state) else state

    async def _retire_removed_diagnostics(self) -> None:
        repair = self._repair_state
        for evidence in repair.evidence:
            if not evidence.diagnostic:
                continue
            paths = self._diagnostic_checks.get(evidence.input_summary, [])
            if not paths:
                continue
            script = "import os,sys; sys.exit(0 if all(not os.path.lexists(p) for p in sys.argv[1:]) else 1)"
            removed = await self._execute_tool_call_impl("run_shell", {"command": python_command(script, paths, repair.task.repo_root)})
            if isinstance(removed, ShellResult) and not removed.failed:
                repair.retire_diagnostic(evidence.id)
                if self.trace:
                    self.trace.event("verification.diagnostic_retired", component="state", purpose="completion",
                                     attributes={"evidence_id": evidence.id, "repo_state": repair.current_repo_state, "paths": paths})

    async def _recheck_completion_failures(self) -> None:
        await self._retire_removed_diagnostics()
        repair = self._repair_state
        if self._finalization_calls is not None:
            # Reuse one unchanged failing check when the model cannot afford its post-patch verification.
            unresolved = set(repair.completion_decision().unresolved_verification_ids)
            for receipt in reversed(repair.verifications):
                if receipt.id not in unresolved or receipt.success:
                    continue
                evidence = next(item for item in repair.evidence if item.id == receipt.evidence_id)
                if evidence.diagnostic or not evidence.test_failure_observed or not evidence.check_fingerprint:
                    continue
                if await self._check_fingerprint(evidence.input_summary) != evidence.check_fingerprint:
                    continue
                await self._execute_tool_call("run_shell", {
                    "command": evidence.input_summary, "acceptance_ids": list(receipt.acceptance_ids),
                    "scope": receipt.scope, "oracle": receipt.oracle,
                }, runtime_owned=True)
                break
        decision = repair.completion_decision()
        # Comparisons can discharge regressions, but cannot supply the issue's repair witness.
        if not self._baseline_ref or any("unresolved verification" not in reason and "unscoped verification" not in reason for reason in decision.reasons):
            return
        for receipt in list(repair.verifications):
            if receipt.id not in decision.unresolved_verification_ids:
                continue
            evidence = next(e for e in repair.evidence if e.id == receipt.evidence_id)
            key = (repair.current_repo_state, evidence.input_summary)
            if evidence.diagnostic or key in self._baseline_attempts or not pytest_selection(evidence.input_summary, repair.task.repo_root):
                continue
            self._baseline_attempts.add(key)
            await self._execute_tool_call("run_shell", {
                "command": evidence.input_summary, "compare_baseline": True,
                "acceptance_ids": list(receipt.acceptance_ids), "scope": receipt.scope, "oracle": receipt.oracle,
            }, runtime_owned=True)

    async def _check_fingerprint(self, command: str) -> str | None:
        """Bind explicit pytest files and local test conditions in the execution environment."""
        if not self._is_verification_command(command):
            return None
        root = self._repair_state.task.repo_root if self._repair_state else str(Path.cwd())
        if "&&" in command:
            prefix, command = command.split("&&", 1)
            try:
                directory = shlex.split(prefix.replace("\\", "/"))
            except ValueError:
                return None
            if directory not in (["cd", "."], ["cd", root.replace("\\", "/")]) or "&&" in command:
                return None
        tokens = self._pytest_tokens(command)
        if not tokens:
            return None
        runner = next(i for i, token in enumerate(tokens) if token.rsplit("/", 1)[-1].lower() in {"pytest", "pytest.exe"})
        args = iter(tokens[runner + 1:])
        paths = []
        for token in args:
            if token in {"-p", "-k", "-m", "-c", "-o", "--tb", "--maxfail", "--color",
                         "--rootdir", "--confcutdir", "--config-file", "--inifile"}:
                next(args, None)
            elif token.startswith("-"):
                continue
            elif token.split("::", 1)[0].endswith(".py"):
                paths.append(token.split("::", 1)[0])
            else:
                # A directory/package selection would include sources we did not fingerprint.
                return None
        paths = sorted(set(paths))
        if not paths or len(paths) > 8:
            return None
        for path in paths:
            if path.startswith("/") or ":" in path or ".." in path.split("/"):
                return None
        interpreter = next((token for token in tokens[:runner]
                            if re.fullmatch(r"(?:.*/)?python(?:\d+(?:\.\d+)*)?(?:\.exe)?", token)), "python")
        payload = json.dumps([root, paths, tokens[runner + 1:]])
        result = await self._execute_tool_call_impl("run_shell", {
            "command": python_command(CHECK_FINGERPRINT_SCRIPT, [payload], root, interpreter),
            "timeout": 10_000,
        })
        if isinstance(result, ShellResult) and not result.failed:
            fingerprint = result.stdout.strip()
            if re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                return fingerprint
        return None

    def _read_observation(self, inp: dict) -> str:
        evidence_id = inp.get("evidence_id")
        if not isinstance(evidence_id, str) or evidence_id not in self._observations:
            return "Error: unknown evidence_id for this task."
        try:
            part = inp.get("part", "output")
            if part not in {"input", "output"}:
                raise ValueError("part must be input or output")
            output = (
                next(item.input_summary for item in self._repair_state.evidence if item.id == evidence_id)
                if part == "input" else self._observations[evidence_id]
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            return f"Error reading observation: {error}"
        return self._format_observation_page(evidence_id, output, inp)

    def _format_observation_page(self, evidence_id: str, output: str | dict, inp: dict, *, scope: str = "") -> str:
        """Reuse the same blob integrity check and paging for runtime observation readers."""
        try:
            if isinstance(output, dict):
                path = self.trace.blob_dir / output["sha256"]
                if path.is_symlink():
                    return "Error: observation blob must not be a symlink."
                raw = path.read_bytes()
                if len(raw) != output["bytes"] or hashlib.sha256(raw).hexdigest() != output["sha256"]:
                    return "Error: observation blob failed integrity check."
                output = json.loads(raw)["output"]
            window = format_file_window(output, inp)
            char_offset = inp.get("char_offset", 0)
            if type(char_offset) is not int or char_offset < 0:
                raise ValueError("char_offset must be a non-negative integer")
            chunk = window[char_offset:char_offset + 16000]
            remaining = len(window) > char_offset + len(chunk)
            return (
                f"Observation {evidence_id}; {scope}{len(output.split(chr(10)))} lines total; "
                f"next_char_offset={char_offset + len(chunk) if remaining else None}.\n{chunk}"
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            return f"Error reading observation: {error}"

    @activity("tool")
    async def _execute_tool_call(
        self, name: str, inp: dict, *, tool_batch_size: int = 1,
        runtime_owned: bool = False,
    ) -> str:
        if not runtime_owned:
            try:
                self._reserve_call("tool_calls", self.max_tool_calls)
            except _BudgetExceeded as error:
                print_info(f"Budget exceeded: {error}")
                return f"Tool execution skipped: {error}"

        async def invoke() -> str | ShellResult:
            if name == "compact_context" and tool_batch_size > 1:
                return (
                    "Error: compact_context must be called alone. "
                    "The other tool results were preserved; request compaction in a new turn."
                )
            if (
                self._repair_state is not None
                and name in {"write_file", "edit_file", "run_shell"}
                and not any(
                    case.source_kind == "issue"
                    for case in self._repair_state.acceptance_cases.values()
                )
            ):
                error = "Error: register at least one issue-derived acceptance case before run_shell or editing."
                return ShellResult(error=error, execution_started=False) if name == "run_shell" else error
            return await self._execute_tool_call_impl(name, inp)

        async def execute() -> str | ShellResult:
            try:
                result = await invoke()
            except Exception:
                if not runtime_owned:
                    self._record_tool_outcome(name, False)
                raise
            if name == "run_shell" and not isinstance(result, ShellResult):
                result = ShellResult(stdout=str(result), error="Execution status unknown: run_shell executor must return ShellResult.")
            if not runtime_owned:
                self._record_tool_outcome(name, not self._looks_like_tool_failure(name, result, result))
            tool_execution_result(result, self._looks_like_tool_failure(name, result, result))
            return result

        verification = name == "run_shell" and self._is_verification_command(str(inp.get("command") or ""))
        track_repo = (
            self.trace is not None
            or not self._stop_repo_state_seen
            or name in STATE_CHANGING_TOOLS
            or verification
        )
        before = await self._probe_repo_state() if track_repo else None
        check_fingerprint = None
        if verification and self._repair_state is not None and self._repair_state.acceptance_cases:
            try:
                check_fingerprint = await self._check_fingerprint(str(inp.get("command") or ""))
            except Exception:
                # A failed read cannot make a check trusted; the actual command still runs.
                check_fingerprint = None
        if self.trace is None:
            started = time.perf_counter()
            result = await execute()
            duration_ms = (time.perf_counter() - started) * 1000
            after = await self._probe_repo_state() if track_repo else before
            failed = self._looks_like_tool_failure(name, result, result)
            self._observe_stop_gate_tool(name, inp, result, before, after)
            evidence_id, state_error = self._record_repair_evidence(
                name, inp, result, failed=failed, duration_ms=duration_ms,
                before=before, after=after, runtime_owned=runtime_owned,
                check_fingerprint=check_fingerprint,
            )
            if before != after and self._repair_state is not None:
                await self._retire_removed_diagnostics()
            return self._format_tool_observation(
                name, result, failed=failed, duration_ms=duration_ms, before=before, after=after,
                evidence_id=evidence_id, state_error=state_error,
                check_fingerprint=check_fingerprint,
                command=str(inp.get("command") or "") if name == "run_shell" and self._repair_state is not None else None,
                preserve_output=self._repair_state is not None,
            )
        primitive = name in PRIMITIVE_TOOLS

        def state_attributes(after: str | None) -> dict[str, Any]:
            return {
                "tool_name": name,
                "primitive_operation": primitive,
                "repo_state_before": before,
                "repo_state_after": after,
                "state_changed": before is not None and after is not None and before != after,
            }
        with self.trace.span(
            "tool.call",
            component="tool",
            purpose=name,
            attributes={
                "tool_name": name,
                "primitive_operation": primitive,
                "repo_state_before": before,
            },
            payload={"input": inp},
        ) as span:
            try:
                started = time.perf_counter()
                result = await execute()
                duration_ms = (time.perf_counter() - started) * 1000
            except Exception as error:
                after = await self._probe_repo_state()
                self._observe_stop_gate_tool(name, inp, ShellResult(error=str(error)), before, after)
                self._record_repair_evidence(
                    name, inp, f"{type(error).__name__}: {error}", failed=True,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    before=before, after=after, runtime_owned=runtime_owned,
                )
                attributes = state_attributes(after)
                attributes.update(error_type=type(error).__name__, error_message=str(error))
                span.finish(
                    status="error",
                    attributes=attributes,
                )
                raise
            after = await self._probe_repo_state()
            failed = self._looks_like_tool_failure(name, result, result)
            self._observe_stop_gate_tool(name, inp, result, before, after)
            evidence_id, state_error = self._record_repair_evidence(
                name, inp, result, failed=failed, duration_ms=duration_ms,
                before=before, after=after, runtime_owned=runtime_owned,
                check_fingerprint=check_fingerprint,
            )
            if before != after and self._repair_state is not None:
                await self._retire_removed_diagnostics()
            attributes = state_attributes(after)
            progress_signature = hashlib.sha256(json.dumps(
                {"tool": name, "input": inp, "output": result, "repo_state": after},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
            ).encode("utf-8")).hexdigest()
            state_changing = name in STATE_CHANGING_TOOLS
            attributes["no_progress"] = (
                state_changing
                and progress_signature == self._last_progress_signature
            )
            if state_changing:
                self._last_progress_signature = progress_signature
            observation = self._format_tool_observation(
                name, result, failed=failed, duration_ms=duration_ms, before=before, after=after,
                evidence_id=evidence_id, state_error=state_error,
                check_fingerprint=check_fingerprint,
                command=str(inp.get("command") or "") if name == "run_shell" and self._repair_state is not None else None,
                preserve_output=self._repair_state is not None,
            )
            span.finish(
                status="error" if failed else "ok",
                attributes=attributes,
                payload={"output": observation, **({"execution": asdict(result)} if isinstance(result, ShellResult) else {})},
            )
            return observation

    def _record_repair_evidence(
        self, name: str, inp: dict, result: str | ShellResult, *, failed: bool, duration_ms: float,
        before: str | None, after: str | None, runtime_owned: bool = False,
        check_fingerprint: str | None = None,
    ) -> tuple[str | None, str | None]:
        if self._repair_state is None or name not in PRIMITIVE_TOOLS:
            return None, None
        exit_code: int | None = None
        if name == "run_shell":
            exit_code = result.exit_code if isinstance(result, ShellResult) else None
            input_summary = self._normalized_verification_command(inp.get("command") or "")
        elif name in {"write_file", "edit_file"}:
            summary: dict[str, Any] = {"file_path": str(inp.get("file_path") or "")}
            for field in ("content", "old_string", "new_string"):
                if field in inp:
                    value = str(inp[field])
                    summary[f"{field}_chars"] = len(value)
                    summary[f"{field}_sha256"] = hashlib.sha256(value.encode()).hexdigest()
            if "replace_all" in inp:
                summary["replace_all"] = bool(inp["replace_all"])
            input_summary = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        else:
            input_summary = json.dumps(inp, ensure_ascii=False, sort_keys=True, default=str)
        execution = asdict(result) if isinstance(result, ShellResult) else None
        test_outcome = result.test_outcome if isinstance(result, ShellResult) else "inconclusive"
        test_cases = result.pytest_report.cases if isinstance(result, ShellResult) and result.pytest_report else ()
        execution_started = result.execution_started if isinstance(result, ShellResult) else None
        if isinstance(result, ShellResult) and (result.exit_code is not None or result.timed_out):
            execution_started = True
        unchanged_baseline_failure = bool(isinstance(result, ShellResult) and result.unchanged_failure and check_fingerprint)
        prior_checks = [e for e in self._repair_state.evidence
                        if e.input_summary == input_summary and e.execution_started is not False]
        diagnostic = (
            input_summary in getattr(self, "_diagnostic_checks", {}) and bool(check_fingerprint)
            and not inp.get("acceptance_ids")
            and all(e.diagnostic for e in prior_checks)
        )
        result = str(result)
        preview = preview_output(result, 1500)
        evidence = None
        verification_start = len(self._repair_state.verifications)
        state_errors: list[str] = []
        command = str(inp.get("command") or "")
        recognized_test = name == "run_shell" and self._is_verification_command(command)
        pytest_not_executed = (
            recognized_test and exit_code in {4, 5}
            and bool(self._pytest_tokens(command))
        )
        verification_fields = {"acceptance_ids", "scope", "oracle"}
        has_claim = name == "run_shell" and bool(verification_fields.intersection(inp))
        try:
            evidence = self._repair_state.record_evidence(
                tool=name,
                status="error" if failed else "ok",
                exit_code=exit_code,
                duration_ms=duration_ms,
                repo_before=before,
                repo_after=after,
                input_summary=input_summary or "(empty input)",
                output_preview=preview,
                check_fingerprint=check_fingerprint,
                test_failure_observed=bool(check_fingerprint and test_outcome == "failed"),
                test_pass_observed=bool(check_fingerprint and test_outcome == "passed"),
                test_cases=test_cases,
                unchanged_baseline_failure=unchanged_baseline_failure,
                diagnostic=diagnostic,
                execution_started=execution_started,
            )
            self._observations[evidence.id] = (
                self.trace.store_payload({"output": result, "execution": execution}) if self.trace is not None else result
            )
            if pytest_not_executed:
                state_errors.append("pytest usage error or no tests collected; correct the command. Output is retained as evidence, not a failing test obligation.")
            elif recognized_test and self._pytest_tokens(command) and test_outcome == "inconclusive":
                state_errors.append("pytest result is inconclusive: require a fresh JUnit report with completed tests and a matching exit code. Retained as diagnostic evidence only.")
            elif has_claim:
                missing = [field for field in verification_fields if field not in inp]
                if missing:
                    raise ValueError("verification metadata missing: " + ", ".join(missing))
                requested_scope = str(inp["scope"])
                requested_oracle = str(inp["oracle"])
                if requested_scope == "diff" or requested_oracle == "diff_check":
                    raise ValueError("diff verification is runtime-owned")
                if recognized_test:
                    self._repair_state.record_verification(
                        evidence.id, inp.get("acceptance_ids") or (),
                        scope=requested_scope, oracle=requested_oracle,
                        runner_recognized=True,
                    )
                else:
                    state_errors.append(
                        "Command is not a recognized test runner; retained as diagnostic "
                        "evidence, not a test acceptance result or failing test obligation."
                    )
            elif runtime_owned and name == "run_shell":
                self._repair_state.record_verification(
                    evidence.id, (), scope="diff", oracle="diff_check",
                )
            elif recognized_test:
                self._repair_state.record_verification(
                    evidence.id, (), scope="focused", oracle="existing_test",
                    runner_recognized=True,
                )
        except (TypeError, ValueError) as error:
            state_errors.append(str(error))
            if (
                evidence is not None and recognized_test and has_claim
                and len(self._repair_state.verifications) == verification_start
            ):
                try:
                    self._repair_state.record_verification(
                        evidence.id, (), scope="focused", oracle="existing_test",
                        runner_recognized=True,
                    )
                except (TypeError, ValueError) as fallback_error:
                    state_errors.append(str(fallback_error))
        if self.trace is not None:
            verification_ids = [
                receipt.id for receipt in self._repair_state.verifications[verification_start:]
            ]
            self.trace.event(
                "state.change", component="state", purpose="repair_evidence",
                status="error" if state_errors else "ok",
                attributes={
                    "evidence_id": evidence.id if evidence else None,
                    "task_execution_id": self._repair_state.execution_id,
                    "repair_task_id": self._repair_state.task.task_id,
                    "repo_generation": evidence.repo_generation if evidence else None,
                },
                payload={
                    "evidence_id": evidence.id if evidence else None,
                    "observation_ref": self._observations.get(evidence.id) if evidence else None,
                    "verification_ids": verification_ids,
                    "error": "; ".join(state_errors) or None,
                },
            )
        return evidence.id if evidence else None, "; ".join(state_errors) or None

    @staticmethod
    def _format_tool_observation(
        name: str, result: str | ShellResult, *, failed: bool, duration_ms: float,
        before: str | None, after: str | None, evidence_id: str | None = None,
        state_error: str | None = None,
        command: str | None = None,
        check_fingerprint: str | None = None,
        preserve_output: bool = True,
    ) -> str:
        if name not in PRIMITIVE_TOOLS and name != "compact_context":
            return result
        exit_code: int | None = None
        if name == "run_shell":
            exit_code = result.exit_code if isinstance(result, ShellResult) else None
        metadata = {
            "tool": name,
            "status": "error" if failed else "ok",
            "exit_code": exit_code,
            "duration_ms": round(duration_ms, 1),
            "state_changed": before is not None and after is not None and before != after,
            "repo_state": after[:12] if after else None,
        }
        if isinstance(result, ShellResult):
            metadata.update(timed_out=result.timed_out, execution_error=result.error)
            metadata["pytest_report"] = {k: v for k, v in asdict(result.pytest_report).items() if k != "cases"} if result.pytest_report else None
            if result.baseline:
                metadata.update(baseline_ref=result.baseline_ref, baseline_exit_code=result.baseline.exit_code,
                                unchanged_baseline_failure=result.unchanged_failure, comparison_error=result.comparison_error)
            if result.comparison_error:
                metadata["comparison_error"] = result.comparison_error
        test_outcome = result.test_outcome if isinstance(result, ShellResult) else "inconclusive"
        result = str(result)
        if evidence_id is not None:
            metadata["evidence_id"] = evidence_id
        if command is not None:
            recognized = Agent._is_verification_command(command)
            metadata["verification_runner_recognized"] = recognized
            metadata["repair_witness_source_bound"] = bool(check_fingerprint)
            metadata["test_failure_observed"] = bool(check_fingerprint and test_outcome == "failed")
            metadata["test_pass_observed"] = bool(check_fingerprint and test_outcome == "passed")
            if recognized and not check_fingerprint:
                metadata["repair_witness_note"] = (
                    "No test-source fingerprint was recorded, so this result cannot form a repair witness. "
                    "Run one pytest command with readable relative .py targets from the repository root "
                    "(up to 8 files, each under 10000 lines). "
                    "Only a leading cd to that root (or cd .) is supported; other chained commands are not."
                )
            elif recognized and test_outcome == "inconclusive":
                metadata["repair_witness_note"] = (
                    "No qualifying pytest JUnit report was recorded. Read this output with read_observation; "
                    "run a direct pytest command without explicit --junitxml/--junit-xml or a standalone --, "
                    "so the harness can collect a fresh report. Fix collection errors or skipped tests. "
                    "Printed summaries and exit code alone are not a repair witness."
                )
            if not recognized:
                metadata["verification_note"] = (
                    "This result cannot satisfy test acceptance. Run the test directly without pipes, "
                    "shell substitutions, or collection/help-only flags. For a Python reproducer, "
                    "use a pytest test file; retrieve long output with read_observation."
                )
        if not preserve_output:
            result = _truncate_result(result)
        observation = (
            f"<tool_observation>{json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))}"
            f"</tool_observation>\n{result}"
        )
        if state_error:
            observation += f"\n<work_state_error>{_safe_utf8_text(state_error)}</work_state_error>"
        return observation

    async def _execute_tool_call_impl(self, name: str, inp: dict) -> str | ShellResult:
        if name == "run_shell":
            command = str(inp.get("command") or "")
            probe_paths = {path for paths in self._diagnostic_checks.values() for path in paths}
            test_paths = {token.split("::", 1)[0].removeprefix("./") for token in self._pytest_tokens(command)}
            if probe_paths.intersection(test_paths) and inp.get("acceptance_ids"):
                return ShellResult(error="A diagnostic probe cannot be promoted to acceptance evidence.", execution_started=False)
            root = self._repair_state.task.repo_root if self._repair_state else ""
            selected = pytest_selection(command, root)
            if selected and set(selected[1]).issubset(probe_paths):
                self._diagnostic_checks[self._normalized_verification_command(command)] = selected[1]
            if inp.get("compare_baseline") is True:
                if self._repair_state is None:
                    return ShellResult(error="Baseline comparison requires a repository repair task.", execution_started=False)
                return await compare_baseline(self._execute_tool_call_impl, command, self._repair_state.task.repo_root,
                                              self._baseline_ref, inp.get("timeout", 30_000))
            if inp.get("diagnostic") is True:
                root = self._repair_state.task.repo_root if self._repair_state else ""
                selected = pytest_selection(command, root)
                normalized = self._normalized_verification_command(command)
                paths = set(selected[1]) if selected else set()
                prior = [e for e in self._repair_state.evidence if e.check_fingerprint
                    and e.execution_started is not False and paths.intersection(
                    token.split("::", 1)[0].removeprefix("./") for token in self._pytest_tokens(e.input_summary)
                )] if self._repair_state else []
                if not selected or not self._baseline_ref or inp.get("acceptance_ids") or any(not e.diagnostic for e in prior):
                    return ShellResult(error="Declare diagnostic before its first run, with explicit new pytest files and no acceptance_ids.", execution_started=False)
                script = "import subprocess,sys; from pathlib import Path; paths=[Path(p) for p in sys.argv[2:]]; sys.exit(1) if any(not p.is_file() or any(q.is_symlink() for q in [p]+list(p.parents)) for p in paths) else None; r=subprocess.run(['git','ls-tree','-r','--name-only',sys.argv[1],'--']+sys.argv[2:],stdout=subprocess.PIPE); sys.exit(0 if r.returncode==0 and not r.stdout else 1)"
                new = await self._execute_tool_call_impl("run_shell", {"command": python_command(script, [self._baseline_ref, *selected[1]], root, selected[0][0])})
                if not isinstance(new, ShellResult) or new.failed:
                    return ShellResult(error="Existing tests cannot be declared diagnostic.", execution_started=False)
                self._diagnostic_checks[normalized] = selected[1]
            tokens = self._pytest_tokens(command)
            collect_report = bool(
                tokens and self._is_verification_command(command)
                and "--" not in tokens
                and not any(token.startswith(("--junitxml", "--junit-xml")) for token in tokens)
            )
            if collect_report or "_pytest_report" in inp:
                inp = {**inp, "_pytest_report": collect_report}
        if name == "read_observation":
            return self._read_observation(inp)
        if name == "compact_context":
            return await self._execute_compact_context_tool(inp)
        if name == "update_work_state":
            return self._execute_update_work_state_tool(inp)
        if name == "update_plan":
            return self._execute_update_plan_tool(inp)
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
            # Route MCP tool calls to the MCP manager
        if name not in PRIMITIVE_TOOLS and self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        if self._finalization_calls is not None:
            remaining_ms = int((self._finalization_deadline - time.monotonic()) * 1000)
            if name != "run_shell" or self._finalization_calls >= 8 or remaining_ms <= 0:
                raise RuntimeError("runtime finalization limit reached")
            self._finalization_calls += 1
            inp = {**inp, "timeout": min(inp.get("timeout", 30_000), remaining_ms)}
        if self._tool_executor is not None and name in PRIMITIVE_TOOLS:
            result = self._tool_executor(name, inp)
            return await result if inspect.isawaitable(result) else result
        result = await execute_tool(
            name, inp, self._read_file_state, preserve_output=self._repair_state is not None,
            structured_output=True,
        )
        if name in {"skill_create", "skill_evolve"}:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and parsed.get("ok"):
                    self._refresh_runtime_system_prompt()
            except Exception:
                pass
        return result

    async def _execute_compact_context_tool(self, inp: dict) -> str:
        reason = str(inp.get("reason") or "").strip()
        compacted = await self._compact_conversation(trigger="tool")
        if not compacted:
            return "No context compaction was performed; the existing conversation history was preserved."
        self._context_cleared = True
        suffix = f"\nReason: {reason}" if reason else ""
        return (
            "Context compacted into structured session memory. "
            "Continue from the folded memory now present in the conversation context."
            f"{suffix}"
        )


    async def _execute_skill_tool(self, inp: dict) -> str:
        from .skills import execute_skill
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))

        if not result:
            return f"Unknown skill: {inp.get('skill_name', '')}"
        if self.trace:
            self.trace.event("skill.body_load", component="skill", purpose="activation",
                             status="error" if result.get("error") else "ok",
                             payload=result["receipt"])
        if result.get("error"):
            return f"Error: {result['error']}. Rediscover the Skill after a source change."

        #fork 表示这个 skill 不直接把 prompt 塞回当前对话，而是要启动一个子 Agent 单独完成任务。
        if result["context"] == "fork":
            # result["allowed_tools"] - 直接访问
            tools = (
                [t for t in self.tools if t["name"] in  result["allowed_tools"] ]
                #result.get("allowed_tools") - 安全访问
                # 存在key：返回对应的值（可能是 None、[]、["tool1"] 等）
                # 不存在key：返回 None（不会抛异常）
                if result.get("allowed_tools")
                else  [t for t in self.tools if t["name"] != "agent"]
            )

            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = Agent(
                model=self.model,
                api_base=self._api_base,
                anthropic_base_url=self._anthropic_base_url,
                api_key=self._api_key,
                thinking=self.thinking,
                max_output_tokens=self.max_output_tokens,
                max_input_tokens_per_call=self.max_input_tokens_per_call,
                max_model_calls=self.max_model_calls,
                model_retry_limit=self.model_retry_limit,
                max_tool_calls=self.max_tool_calls,
                model_call_timeout_seconds=self.model_call_timeout_seconds,
                openai_extra_body=self._openai_extra_body,
                openai_stream_include_usage=self._openai_stream_include_usage,
                skills_enabled=self._skills_enabled_explicit,
                custom_system_prompt=result["prompt"],
                custom_tools=tools,
                tool_executor=self._tool_executor,
                repo_state_probe=self._repo_state_probe,
                is_sub_agent=True,
                trace=self.trace,
                model_purpose="subagent",
                _operation_counts=self._operation_counts,
                permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
            )
            try:
                sub_result = await sub_agent.run_once(inp.get("args") or "Execute this skill task.")
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                if sub_result.get("termination_reason"):
                    self.termination_reason = sub_result["termination_reason"]
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return sub_result["text"] or "(Skill produced no output)"
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return f"Skill fork error: {e}"

        # One bounded active procedure, shared with retrieval and preserved across folding.
        self._loaded_skill_context = result["loaded_context"]
        self._loaded_skill_receipt = result["receipt"]
        return (f'[Skill "{inp.get("skill_name", "")}" activated]\n'
                "Its verified full procedure is pinned in the current request context, replacing "
                "the previous active Skill. Reinvoke skill to switch procedures or arguments.")

    async def _execute_plan_mode_tool(self, name):
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path =  self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Entered plan mode (read-only). Plan file: " + self._plan_file_path)
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."
        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = self._plan_file_path
            # 交互式审批流程（如果有审批函数）
            if self._plan_approval_fn:
                result = self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice =="keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                #离开计划模式
                self._pre_plan_mode = target_mode
                self._pre_plan_mode = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None
                self._system_prompt = self._base_system_prompt
                if self.use_openai and self._openai_messages:
                    self._openai_messages[0]["content"] = self._system_prompt

                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(f"Plan approved. Context cleared, executing in {target_mode} mode.")
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )
                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )
            # 没有审批函数时的回退（例如子代理）
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt

            print_info("Exited plan mode. Restored to " + self.permission_mode + " mode.")
            return f"Exited plan mode. Permission mode restored to: {self.permission_mode}\n\n## Your Plan:\n{plan_content}"

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """清空历史信息，但是保留系统prompt."""
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.last_input_token_count = 0
        self._fold_last_time = 0.0
        self._fold_count = 0
        self._failed_compaction_signature = None
        self._tool_error_streak = 0
        self._same_tool_repeat_count = 0
        self._last_tool_name = ""

    async def _execute_agent_tool(self, inp:dict) -> str:
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")
        print_sub_agent_start(agent_type, description)

        config = get_sub_agent_config(agent_type)

        sub_agent = Agent(
            model=self.model,
            api_base=self._api_base,
            anthropic_base_url=self._anthropic_base_url,
            api_key=self._api_key,
            thinking=self.thinking,
            max_output_tokens=self.max_output_tokens,
            max_input_tokens_per_call=self.max_input_tokens_per_call,
            max_model_calls=self.max_model_calls,
            model_retry_limit=self.model_retry_limit,
            max_tool_calls=self.max_tool_calls,
            model_call_timeout_seconds=self.model_call_timeout_seconds,
            openai_extra_body=self._openai_extra_body,
            openai_stream_include_usage=self._openai_stream_include_usage,
            skills_enabled=self._skills_enabled_explicit,
            custom_system_prompt=config["system_prompt"],
            custom_tools=config["tools"],
            tool_executor=self._tool_executor,
            repo_state_probe=self._repo_state_probe,
            is_sub_agent=True,
            trace=self.trace,
            model_purpose="subagent",
            _operation_counts=self._operation_counts,
            permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
        )
        try:
            result = await sub_agent.run_once(prompt)
            self.total_input_tokens += result["tokens"]["input"]
            self.total_output_tokens += result["tokens"]["output"]
            if result.get("termination_reason"):
                self.termination_reason = result["termination_reason"]
            print_sub_agent_end(agent_type, description)
            return result["text"] or "(Sub-agent produced no output)"
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return f"Sub-agent error: {e}"

#--------------Anthropic 后端---------------
    def _user_message(self, content: str) -> dict[str, Any]:
        return {"role": "user", "content": content,
                "_source_user_text": self._current_user_source if self._current_user_source is not None else content}

    async def  _chat_anthropic(self, user_message: str) -> None:
        self._anthropic_messages = self._normalize_anthropic_messages(_sanitize_for_utf8(self._anthropic_messages))
        user_message = _safe_utf8_text(user_message)
        # 先把本轮用户输入放入 Anthropic 消息历史，后续每轮模型调用都会带上这段上下文。
        self._anthropic_messages.append(self._user_message(user_message))

        # 异步内存预取：主 agent 才需要查 memory，sub agent 不额外注入记忆。
        # 这里只启动后台任务，不阻塞当前模型调用流程。
        memory_prefetch:MemoryPrefetch | None = None
        if self._memory_enabled:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )
        while True:
            # 外部请求中止时，结束整个 agent loop。
            if self._aborted:
                break
            if self._stop_before_model_if_cost_exhausted():
                break

            # 每轮调用模型前尝试压缩上下文，避免消息历史过长。
            self._run_compression_pipeline()

            # 如果记忆预取任务已经完成，就把取回来的 memory 内容追加到最后一条用户消息里。
            # consumed 用来保证同一批 memory 只注入一次。
            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._anthropic_messages[-1] if self._anthropic_messages else None
                        if last and last.get("role") == "user":
                            content = last.get("content", "")
                            if isinstance(content, str):
                                # 字符串不可变，需要重新赋值回 message。
                                last["content"] = content + "\n\n" + injection_text
                            elif isinstance(content, list):
                                # list 是可变对象，append 会直接修改 last["content"] 指向的列表。
                                content.append({"type": "text", "text": injection_text})
                        else:
                            # 如果最后一条不是 user message，就单独追加一条用户消息承载 memory。
                            self._anthropic_messages.append({"role": "user", "content": injection_text, "_source_user_text": None})

                        for m in memories:
                            # 记录本 session 已经注入过的 memory，后续检索时可避免重复 surfaced。
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += m.size
                except:
                    # memory 注入失败不应该中断主对话流程。
                    pass

            if not self.is_sub_agent:
                start_spinner()


            # Execute the completed response in order; reads may depend on prior writes.
            output_marks = self._output_marks()
            try:
                response = await self._call_anthropic_stream()
            except _BudgetExceeded as error:
                if not self.is_sub_agent:
                    stop_spinner()
                print_info(f"Budget exceeded: {error}")
                break
            if not self.is_sub_agent:
                stop_spinner()

            # 记录本次模型调用的耗时点和 token 消耗，用于成本展示与预算控制。
            self.last_api_call_time = time.time()
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.last_input_token_count = response.usage.input_tokens
            self._record_provider_cost(response.usage)

            # Anthropic 的响应内容里可能混有 text block 和 tool_use block，这里只挑出工具调用。
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # 把模型返回的所有 content block 写入消息历史，后续 tool_result 要与这些 tool_use 对应。
            self._anthropic_messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })

            # 没有工具调用，说明模型已经给出最终回复，本轮对话结束。
            if not tool_uses:
                stop_reason = await self._run_stop_hooks()
                if stop_reason:
                    self._discard_output_since(output_marks)
                    self._anthropic_messages.append({
                        "role": "user",
                        "content": self._stop_hook_feedback(stop_reason),
                        "_source_user_text": None,
                    })
                    continue
                if self.termination_reason and self.termination_reason.get("kind") == "incomplete_response":
                    self._discard_output_since(output_marks)
                if not self.is_sub_agent:
                    print_cost(
                        self.total_input_tokens,
                        self.total_output_tokens,
                        self.total_provider_reported_cost if self._provider_cost_complete else None,
                    )
                break

            # 有工具调用时，进入下一轮工具执行。这里同时检查 turn/budget 限制。
            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                self.termination_reason = {"kind": budget["kind"]}
                print_info(f"Budget exceeded: {budget['reason']}")
                self._anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Tool execution skipped: {budget['reason']}",
                        }
                        for tu in tool_uses
                    ],
                })
                break


            # 收集本轮所有工具结果，之后作为 tool_result 消息回传给模型。
            tool_results: list[dict] = []
            context_break = False

            for tu in tool_uses:
                # context_break 表示某个工具执行期间清理了上下文，需要停止继续处理本轮剩余工具。
                if context_break or self._aborted:
                    break

                # 将工具入参转为普通 dict，便于权限检查、打印和实际执行。
                inp = dict(tu.input) if hasattr(tu, "items") else tu.input
                print_tool_call(tu.name, inp)

                perm = check_permission(tu.name, inp, self.permission_mode, self._plan_file_path)
                self._trace_permission(tu.name, inp, perm)
                if perm["action"] == "deny":
                    # 权限拒绝时，也要返回一个 tool_result，让模型知道该工具调用失败的原因。
                    print_info(f"Denied: {perm.get('message', '')}")
                    self._record_tool_outcome(tu.name, False)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                         "content": f"Action denied: {perm.get('message', '')}"})
                    continue

                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    # 高风险操作需要用户确认；同一个 message 确认过后会缓存，避免重复询问。
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        self._record_tool_outcome(tu.name, False)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": tu.id, "content": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])

                # 权限通过后执行工具，并把大输出持久化为可回传的摘要或引用。
                try:
                    raw = await self._execute_tool_call(
                        tu.name, inp, tool_batch_size=len(tool_uses),
                    )
                except Exception as e:
                    raw = f"Error executing tool: {e}"
                raw = _safe_utf8_text(raw)
                res = self._persist_large_result(tu.name, raw)
                print_tool_result(tu.name, res)

                if self._context_cleared:
                    # 工具执行过程中如果清理了上下文，就把结果作为新的用户消息写入，
                    # 并停止继续处理本轮剩余工具，避免旧上下文和新上下文混在一起。
                    self._context_cleared = False
                    self._anthropic_messages.append({"role": "user", "content": res, "_source_user_text": None})
                    context_break = True
                    break

                # Anthropic 要求 tool_result 使用 tool_use_id 对应到前面的 tool_use block。
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})

            if not context_break and tool_results:
                # Anthropic 要求 assistant/tool_use 后面紧跟一条 user/tool_result 消息，
                # 且这条消息必须包含本轮所有 tool_use 的对应结果。
                self._anthropic_messages.append({"role": "user", "content": tool_results})

            self._context_cleared = False
            if self.termination_reason:
                break

            # 工具结果可能很长，每轮工具执行后检查是否需要压缩上下文。
            self._refresh_runtime_system_prompt()
            await self._check_and_compact()

    @staticmethod
    def _block_to_dict(block) -> dict:
        if block.type == "text":
            return {"type": "text", "text": _safe_utf8_text(block.text)}
        if block.type == "tool_use":
            raw_input = dict(block.input) if hasattr(block.input, 'items') else block.input
            return {"type": "tool_use", "id": _safe_utf8_text(block.id), "name": _safe_utf8_text(block.name), "input": _sanitize_for_utf8(raw_input)}
        # Fallback
        return {"type": _safe_utf8_text(block.type)}

    async def _call_anthropic_stream(self):
        await self._check_and_compact()
        model_call_reserved = False
        request_messages = self._anthropic_request_messages()

        @activity("model")
        async def _provider_call(timing: dict[str, int]):
            nonlocal model_call_reserved
            max_output = self.max_output_tokens or _get_max_output_tokens(self.model)
            if self._thinking_mode == "disabled" and self.max_output_tokens is None:
                max_output = 16384
            tools = _sanitize_for_utf8(get_active_tool_definitions(self.tools))
            system = _safe_utf8_text(self._system_prompt)
            self._preflight_model_input(system=system, messages=request_messages, tools=tools)

            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output,
                "system": system,
                "tools": tools,
                "messages": request_messages,
            }
            #如果开启了思考模式，就给 Anthropic 请求加上 thinking 参数。
            if self._thinking_mode  in ("adaptive", "enabled"):
                create_params["thinking"]={"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True

            tool_blocks_by_index: dict[int, str] = {}

            if not model_call_reserved:
                self._reserve_call("model_calls", self.max_model_calls)
                model_call_reserved = True
            self._record_skill_exposure(request_messages)
            async with self._anthropic_client.messages.stream(**create_params)as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue
                    timing.setdefault("first_event_ns", time.monotonic_ns())
                    # 当事件是工具调用开始：
                    if event.type == "content_block_start":
                        cb = getattr(event, 'content_block', None)
                        #如果 block 类型是 tool_use，就记录这个工具调用：
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            stream_phase(f"Receiving tool arguments · {cb.name}")
                            tool_blocks_by_index[event.index] = cb.name
                    #当事件是内容增量，分三种情况。
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # 第一种，普通文本：模型输出正文时，
                        # 调用 _emit_text()。如果是普通交互，就打印；
                        # 如果是 run_once()，就写入 _output_buffer。
                        if hasattr(delta, "text"):
                            stream_phase("Writing response")
                            timing.setdefault("first_text_ns", time.monotonic_ns())
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        #第二种，thinking 内容：
                        #如果模型返回思考内容，也输出出来，并在开头加：[thinking]
                        elif hasattr(delta, 'thinking'):
                            stream_phase("Receiving reasoning", delta.thinking)
                            timing.setdefault("first_text_ns", time.monotonic_ns())
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n  [thinking] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        # The SDK assembles tool arguments; this branch only updates the display.
                        elif hasattr(delta, 'partial_json'):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                stream_phase(f"Receiving tool arguments · {tb}")
                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb:
                            timing.setdefault("first_tool_ns", time.monotonic_ns())
                final_message = await stream.get_final_message()

            #过滤思考的message（因为 thinking 内容一般不应该进入历史消息，否则后续上下文会变大，也可能不符合 API 消息格式要求。）
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message

        async def _do():
            if not model_call_reserved:
                self._raise_if_call_exhausted("model_calls", self.max_model_calls)
            timing: dict[str, int] = {}
            if self.trace is None:
                return await self._await_model_call(_provider_call(timing))
            started_ns = time.monotonic_ns()
            tools = _sanitize_for_utf8(get_active_tool_definitions(self.tools))
            model_attributes = self._model_trace_attributes("anthropic", tools)
            with self.trace.span(
                "model.call",
                component="model",
                purpose=self.model_purpose,
                attributes=model_attributes,
                payload={
                    "system": self._system_prompt,
                    "tools": tools,
                    "messages": request_messages,
                },
            ) as span:
                response = await self._await_model_call(_provider_call(timing))
                ended_ns = time.monotonic_ns()
                text = "".join(
                    _safe_utf8_text(getattr(block, "text", ""))
                    for block in response.content if getattr(block, "type", "") == "text"
                )
                tool_blocks = [
                    self._block_to_dict(block)
                    for block in response.content if getattr(block, "type", "") == "tool_use"
                ]
                span.finish(
                    attributes=model_attributes,
                    metrics={
                        "provider_usage": provider_usage(response.usage, "anthropic"),
                        "timing": timing_metrics(started_ns, ended_ns, **timing),
                        "input_sections": measure_model_input_sections(
                            self._prompt_sections, self._runtime_guidance,
                            request_messages, tools,
                        ),
                        "output_sections": measure_text_sections(
                            {"model_text": text, "tool_calls": json.dumps(tool_blocks, ensure_ascii=False)}
                        ),
                    },
                    payload={"text": text, "tool_calls": tool_blocks},
                )
                return response
#调用 _do()，如果遇到可重试错误，就由 _with_retry() 负责重试。
        return self._guard_anthropic_response(await _with_retry(_do, max_retries=self.model_retry_limit))

    #openAI后端

    async def _chat_openai(self, user_message:str) -> None:
        user_message = _safe_utf8_text(user_message)
        self._openai_messages.append(self._user_message(user_message))

        #预取句柄 MemoryPrefetch
        memory_prefetch: MemoryPrefetch | None = None
        if self._memory_enabled:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break
            if self._stop_before_model_if_cost_exhausted():
                break

            self._run_compression_pipeline()

            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        injection_text = _safe_utf8_text(injection_text)
                        last = self._openai_messages[-1] if self._openai_messages else None

                        if last and last.get("role") == "user":
                            last["content"] = (last.get("content") or "") + "\n\n" + injection_text
                        else:
                            self._openai_messages.append({"role": "user", "content": injection_text, "_source_user_text": None})

                        for m in memories:
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += len(m.content.encode())
                except Exception:
                    pass

            if not self.is_sub_agent:
                start_spinner()

            output_marks = self._output_marks()
            try:
                response = await self._call_openai_stream()
            except _BudgetExceeded as error:
                if not self.is_sub_agent:
                    stop_spinner()
                print_info(f"Budget exceeded: {error}")
                break

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()

            if response.get("usage"):
                self.total_input_tokens += response["usage"]["prompt_tokens"]
                self.total_output_tokens += response["usage"]["completion_tokens"]
                self.last_input_token_count = response["usage"]["prompt_tokens"]
                self._record_provider_cost(response["usage"])

            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            message = choice.get("message", {})

            tool_calls = list(message.get("tool_calls") or [])
            if len(tool_calls) > MAX_TOOL_CALLS_PER_RESPONSE:
                print_info(
                    f"Tool call batch limited to {MAX_TOOL_CALLS_PER_RESPONSE} "
                    f"of {len(tool_calls)} calls"
                )
                tool_calls = tool_calls[:MAX_TOOL_CALLS_PER_RESPONSE]
                message = {**message, "tool_calls": tool_calls}
            elif not tool_calls and "tool_calls" in message:
                message = {key: value for key, value in message.items() if key != "tool_calls"}
            self._openai_messages.append(message)

            if not tool_calls:
                stop_reason = await self._run_stop_hooks()
                if stop_reason:
                    self._discard_output_since(output_marks)
                    self._openai_messages.append({
                        "role": "user",
                        "content": self._stop_hook_feedback(stop_reason),
                        "_source_user_text": None,
                    })
                    continue
                if self.termination_reason and self.termination_reason.get("kind") == "incomplete_response":
                    self._discard_output_since(output_marks)
                if not self.is_sub_agent:
                    print_cost(
                        self.total_input_tokens,
                        self.total_output_tokens,
                        self.total_provider_reported_cost if self._provider_cost_complete else None,
                    )
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                self.termination_reason = {"kind": budget["kind"]}
                print_info(f"Budget exceeded: {budget['reason']}")
                break

            oai_checked: list[dict] = []
            for tc in tool_calls:
                if self._aborted:
                    break

                if tc.get("type") != "function":
                    continue

                fn_name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                    if not isinstance(inp, dict):
                        raise ValueError("tool arguments must be a JSON object")
                except (TypeError, ValueError):
                    self._record_tool_outcome(fn_name, False)
                    oai_checked.append({"tc": tc, "fn": fn_name, "inp": {}, "allowed": False,
                                        "result": "Tool arguments are not a complete JSON object; no tool was executed."})
                    continue

                print_tool_call(fn_name, inp)

                perm = check_permission(fn_name, inp, self.permission_mode, self._plan_file_path)
                self._trace_permission(fn_name, inp, perm)

                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    self._record_tool_outcome(fn_name, False)
                    oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False,
                                        "result": f"Action denied: {perm.get('message', '')}"})
                    continue
                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        self._record_tool_outcome(fn_name, False)
                        oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False,
                                            "result": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])
                oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": True})

            oai_batches: list[dict] = []
            for ct in oai_checked:
                safe = ct["allowed"] and ct["fn"] in CONCURRENCY_SAFE_TOOLS
                if safe and oai_batches and oai_batches[-1]["concurrent"]:
                    oai_batches[-1]["items"].append(ct)
                else:
                    oai_batches.append({"concurrent": safe, "items": [ct]})

            oai_context_break = False
            for batch in oai_batches:
                if oai_context_break or self._aborted:
                    break

                if batch["concurrent"]:
                    async def _run_oai_safe(ct_item: dict) -> tuple[dict, str]:
                        raw = await self._execute_tool_call(ct_item["fn"], ct_item["inp"])
                        raw = _safe_utf8_text(raw)
                        res = self._persist_large_result(ct_item["fn"], raw)
                        print_tool_result(ct_item["fn"], res)
                        return ct_item, res

                    results = await asyncio.gather(*[_run_oai_safe(ct) for ct in batch["items"]])
                    for ct_item, res in results:
                        self._openai_messages.append(
                            {"role": "tool", "tool_call_id": ct_item["tc"]["id"], "content": res})
                    if self.termination_reason:
                        break
                else:
                    for ct in batch["items"]:
                        if not ct["allowed"]:
                            self._openai_messages.append(
                                {"role": "tool", "tool_call_id": ct["tc"]["id"], "content": ct["result"]})
                            continue

                        raw = await self._execute_tool_call(
                            ct["fn"], ct["inp"], tool_batch_size=len(tool_calls),
                        )
                        raw = _safe_utf8_text(raw)
                        res = self._persist_large_result(ct["fn"], raw)
                        print_tool_result(ct["fn"], res)

                        if self._context_cleared:
                            self._context_cleared = False
                            self._openai_messages.append({"role": "user", "content": res, "_source_user_text": None})
                            oai_context_break = True
                            break

                        self._openai_messages.append(
                            {"role": "tool", "tool_call_id": ct["tc"]["id"], "content": res})
                        if self.termination_reason:
                            oai_context_break = True
                            break

            self._context_cleared = False
            if self.termination_reason:
                break
            self._refresh_runtime_system_prompt()
            await self._check_and_compact()

    async def _call_openai_stream(self) -> dict:
        await self._check_and_compact()
        model_call_reserved = False
        tools = _sanitize_for_utf8(_to_openai_tools(get_active_tool_definitions(self.tools)))
        messages = self._openai_request_messages()

        @activity("model")
        async def _provider_call(
            timing: dict[str, int], messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> dict:
            nonlocal model_call_reserved
            self._preflight_model_input(messages=messages, tools=tools)
            create_params: dict[str, Any] = {
                "model": self.model,
                "tools": tools,
                "messages": messages,
                "stream": True,
                "parallel_tool_calls": True,
            }
            if self._openai_stream_include_usage:
                create_params["stream_options"] = {"include_usage": True}
            if self.max_output_tokens is not None:
                create_params["max_tokens"] = self.max_output_tokens
            extra_body = self._openai_request_extra_body()
            if extra_body:
                create_params["extra_body"] = extra_body
            if not model_call_reserved:
                self._reserve_call("model_calls", self.max_model_calls)
                model_call_reserved = True
            self._record_skill_exposure(messages)
            stream = await self._openai_client.chat.completions.create(**create_params)

            content = ""
            reasoning_content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            def complete_tool_call(call: dict, *, allow_empty: bool) -> bool:
                if not call["name"]:
                    return False
                arguments = call["arguments"].strip()
                if not arguments:
                    return allow_empty
                try:
                    return isinstance(json.loads(arguments), dict)
                except (TypeError, json.JSONDecodeError):
                    return False

            async for chunk in stream:
                timing.setdefault("first_event_ns", time.monotonic_ns())
                if chunk.usage:
                    usage = chunk.usage

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                delta_reasoning = (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                ) if delta else None
                if delta_reasoning and delta_reasoning.strip() != REASONING_REPLAY_PLACEHOLDER:
                    stream_phase("Receiving reasoning", delta_reasoning)
                    reasoning_content += _safe_utf8_text(delta_reasoning)

                if delta and delta.content:
                    stream_phase("Writing response")
                    timing.setdefault("first_text_ns", time.monotonic_ns())
                    if first_text:
                        stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += _safe_utf8_text(delta.content)

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.id:
                                existing["id"] = _safe_utf8_text(tc.id)
                            if tc.function:
                                if tc.function.name:
                                    existing["name"] = _safe_utf8_text(tc.function.name)
                                if tc.function.arguments:
                                    existing["arguments"] += _safe_utf8_text(tc.function.arguments)
                        else:
                            tool_calls[tc.index] = {
                                "id": _safe_utf8_text(tc.id or ""),
                                "name": _safe_utf8_text((tc.function.name if tc.function else "") or ""),
                                "arguments": _safe_utf8_text((tc.function.arguments if tc.function else "") or ""),
                            }
                        candidate = tool_calls[tc.index]
                        stream_phase(f"Receiving tool arguments · {candidate['name'] or 'pending'}")
                        if complete_tool_call(candidate, allow_empty=False):
                            timing.setdefault("first_tool_ns", time.monotonic_ns())

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason == "tool_calls" and any(
                        complete_tool_call(call, allow_empty=True) for call in tool_calls.values()
                    ):
                        timing.setdefault("first_tool_ns", time.monotonic_ns())

            assembled = None
            if tool_calls:
                if any(complete_tool_call(call, allow_empty=True) for call in tool_calls.values()):
                    timing.setdefault("first_tool_ns", time.monotonic_ns())
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]
            if assembled and self._thinking_mode in ("adaptive", "enabled") and not reasoning_content:
                reasoning_content = REASONING_REPLAY_PLACEHOLDER
            if usage is None:
                raise RuntimeError("provider stream returned empty response or missing usage")

            completion_details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
            reasoning_tokens = getattr(usage, "reasoning_tokens", None) if usage is not None else None
            details_reasoning = (
                completion_details.get("reasoning_tokens")
                if isinstance(completion_details, dict)
                else getattr(completion_details, "reasoning_tokens", None)
            )
            if reasoning_tokens is not None and details_reasoning is None:
                completion_details = {"reasoning_tokens": reasoning_tokens}

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        **(
                            {"reasoning_content": reasoning_content}
                            if reasoning_content
                            else {}
                        ),
                        "tool_calls": assembled,
                    },
                    "finish_reason": finish_reason or "stop",
                    "raw_finish_reason": finish_reason or None,
                }],
                "usage": None if usage is None else {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "prompt_tokens_details": getattr(usage, "prompt_tokens_details", None),
                    "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
                    "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
                    "completion_tokens_details": completion_details,
                    "cost": getattr(usage, "cost", None),
                },
            }

        async def _do() -> dict:
            if not model_call_reserved:
                self._raise_if_call_exhausted("model_calls", self.max_model_calls)
            timing: dict[str, int] = {}
            if self.trace is None:
                return await self._await_model_call(_provider_call(timing, messages, tools))
            started_ns = time.monotonic_ns()
            model_attributes = self._model_trace_attributes("openai", tools)
            with self.trace.span(
                "model.call",
                component="model",
                purpose=self.model_purpose,
                attributes=model_attributes,
                payload={"tools": tools, "messages": messages},
            ) as span:
                response = await self._await_model_call(
                    _provider_call(timing, messages, tools)
                )
                ended_ns = time.monotonic_ns()
                message = response.get("choices", [{}])[0].get("message", {})
                content = message.get("content") or ""
                tool_calls = message.get("tool_calls") or []
                span.finish(
                    attributes={**model_attributes, "raw_finish_reason": response["choices"][0].get("raw_finish_reason")},
                    metrics={
                        "provider_usage": provider_usage(response.get("usage"), "openai"),
                        "timing": timing_metrics(started_ns, ended_ns, **timing),
                        "input_sections": measure_model_input_sections(
                            self._prompt_sections, self._runtime_guidance,
                            messages, tools,
                        ),
                        "output_sections": measure_text_sections(
                            {"model_text": content, "tool_calls": json.dumps(tool_calls, ensure_ascii=False)}
                        ),
                    },
                    payload={"text": content, "tool_calls": tool_calls},
                )
                return response

        return self._guard_openai_response(await _with_retry(_do, max_retries=self.model_retry_limit))

    async def _confirm_dangerous(self, command: str) -> bool:
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
