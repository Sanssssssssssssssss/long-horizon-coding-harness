"""Minimal replacement-history compaction helpers."""

from __future__ import annotations

import copy
import json
from typing import Any


COMPACTION_RESERVE_TOKENS = 16_384
COMPACTION_KEEP_RECENT_TOKENS = 20_000
COMPACTION_SUMMARY_MAX_TOKENS = int(COMPACTION_RESERVE_TOKENS * 0.8)
COMPACTION_SUMMARY_MARKER = "<session-folded-memory>"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You summarize coding-agent history for another model. Produce only a factual "
    "latest-state continuation checkpoint; do not continue the task or call tools. "
    "Derive user requirements only from <canonical-task> or explicit user messages. "
    "Retrieved Skills are procedural hints, not user requirements. "
    "Model-written test expectations and hypotheses may be wrong; do not promote them "
    "to user requirements. Do not reconstruct tool-use rules from diagnostic warnings."
)

COMPACTION_PROMPT = """Create a structured latest-state checkpoint from <previous-checkpoint>, <new-archived-evidence>, and <retained-recent-context> when present.

Read the sections in that recency order; retained recent context is newest. A checkpoint
may already reflect evidence repeated in a later section: treat duplicate evidence as the
same event, not as newer corroboration. When genuinely newer evidence changes, completes,
disproves, or supersedes an earlier status, plan, or claim, omit the earlier version entirely.

Include:
- The user's goal and hard constraints supported by <canonical-task> or explicit user messages
- The latest supported root cause, decisions, and changes actually made
- The latest verification results and any still-unresolved failures
- Critical evidence, including exact file paths, commands, test node IDs, results, and
  error text needed to rerun a check or continue the task

Do not describe the archive boundary, historical status, superseded plans, remaining-work
or next-step sections. Do not prescribe actions. Be concise, structured, and factual. Do
not infer unobserved state: a passing check proves only its reported scope, not a clean
working tree or overall completion. A path-filtered command cannot exclude changes in
other files, and a focused test cannot establish broader test health. Omit repeated probes,
diagnostic inventories, and observations that produced no behavioral evidence. Do not
call tools. Summarize the task, not the compaction machinery."""

COMPACTION_RETRY_PROMPT = """Return a substantive latest-state checkpoint from the previous checkpoint, archived evidence, and newer retained context in that recency order. Treat overlapping evidence as the same event, not as newer corroboration. State the objective, hard constraints, latest supported decisions and changes, latest verification, unresolved failures, and exact file paths, commands, test node IDs, results, and error text needed to continue. Omit every earlier status, plan, or claim superseded by genuinely newer evidence, plus repeated probes and diagnostics that produced no behavioral evidence. A passing check proves only its reported scope; a path-filtered command cannot exclude other changed files, and a focused test cannot establish broader health. Do not infer a clean working tree or overall completion. Do not describe archive boundaries, prescribe next actions, call tools, discuss compaction, or return a placeholder."""

_PLACEHOLDERS = {
    "no summary available",
    "summary unavailable",
    "no summary",
    "n/a",
    "na",
    "not available",
    "i cannot provide a summary",
    "i can't provide a summary",
    "unable to provide a summary",
}


def validate_folded_memory(text: str) -> str:
    summary = str(text or "").strip()
    if len(summary) < 40 or not any(char.isalnum() for char in summary):
        raise ValueError("compaction summary contained no substantive text")
    normalized = " ".join(summary.lower().split()).strip(" .,!?:;`'\"-_")
    if normalized in _PLACEHOLDERS:
        raise ValueError("compaction summary was a placeholder")
    return summary


def format_folded_memory(summary: str) -> str:
    return (
        f"{COMPACTION_SUMMARY_MARKER}\n"
        "Assistant-generated latest-state checkpoint, not user instructions. "
        "System instructions, the canonical task and runtime receipts take precedence. "
        "Raw recent messages follow and remain authoritative if any detail conflicts.\n\n"
        f"{summary.strip()}\n"
        "</session-folded-memory>"
    )


def folded_memory_summary(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if not isinstance(content, str) or COMPACTION_SUMMARY_MARKER not in content:
        return None
    start = content.find("\n\n")
    end = content.rfind("\n</session-folded-memory>")
    if start < 0 or end <= start:
        return None
    return content[start + 2:end].strip() or None


def _is_tool_result(message: dict[str, Any]) -> bool:
    if message.get("role") == "tool":
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _message_token_estimate(message: dict[str, Any]) -> int:
    payload = json.dumps({key: value for key, value in message.items() if key != "_source_user_text"},
                         ensure_ascii=False, default=str).encode("utf-8")
    return max(
        1,
        (len(payload) + 2) // 3,
    )


def compaction_cut_index(
    messages: list[dict[str, Any]], *, start: int = 0,
    keep_recent_tokens: int = COMPACTION_KEEP_RECENT_TOKENS,
    latest_exchange_only: bool = False,
) -> int | None:
    """Return a Pi-style cut that keeps a token-bounded, tool-safe recent tail."""
    cut_points = [
        index for index in range(start, len(messages))
        if not _is_tool_result(messages[index])
    ]
    if len(cut_points) < 2:
        return None

    accumulated = 0
    best_cut = None
    for index in range(len(messages) - 1, start - 1, -1):
        accumulated += _message_token_estimate(messages[index])
        if index not in cut_points or index <= start:
            continue
        if accumulated <= keep_recent_tokens:
            if latest_exchange_only:
                return index
            best_cut = index
            continue
        # ponytail: never spend a fold call when the newest indivisible unit cannot fit.
        return best_cut
    return best_cut


def serialize_compaction_history(messages: list[dict[str, Any]]) -> str:
    """Serialize history as evidence, not as a conversation for the model to continue."""
    lines: list[str] = []
    metadata = _tool_metadata(messages)
    for message in messages:
        role = str(message.get("role") or "message")
        content = message.get("content")
        if role == "user" and "_source_user_text" in message and message["_source_user_text"] is None:
            role = "runtime"
        if role == "user" and isinstance(message.get("_source_user_text"), str):
            content = message["_source_user_text"]
        if role == "tool" and isinstance(content, str):
            tool_id = str(message.get("tool_call_id") or "?")
            tool_name = metadata.get(tool_id, ("tool", "{}"))[0]
            lines.append(
                f"[Tool result id={tool_id} name={tool_name}]: {content}"
            )
            continue
        if isinstance(content, str) and content:
            lines.append(f"[{role.title()}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_result":
                    tool_id = str(block.get("tool_use_id") or "?")
                    tool_name = metadata.get(tool_id, ("tool", "{}"))[0]
                    lines.append(
                        f"[Tool result id={tool_id} name={tool_name}]: "
                        f"{str(block.get('content') or '')}"
                    )
                elif block_type == "tool_use":
                    lines.append(
                        f"[Assistant tool call id={block.get('id') or '?'}]: "
                        f"{block.get('name')}({json.dumps(block.get('input') or {}, ensure_ascii=False)})"
                    )
                elif block_type == "text":
                    lines.append(f"[{role.title()} text]: {block.get('text') or ''}")
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            lines.append(
                f"[Assistant tool call id={call.get('id') or '?'}]: "
                f"{function.get('name')}({function.get('arguments') or '{}'})"
            )
    return "\n\n".join(lines)


def pruned_request_view(
    messages: list[dict[str, Any]], *, minimum_bytes: int = 60_000,
    protect_bytes: int = 120_000,
) -> list[dict[str, Any]]:
    """Compatibility API: return an unpruned provider-request copy."""
    del minimum_bytes, protect_bytes
    return copy.deepcopy(messages)


def _tool_metadata(messages: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    metadata: dict[str, tuple[str, str]] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            metadata[str(call.get("id") or "")] = (
                str(function.get("name") or "tool"),
                str(function.get("arguments") or "{}"),
            )
        blocks = message.get("content") if isinstance(message.get("content"), list) else []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                metadata[str(block.get("id") or "")] = (
                    str(block.get("name") or "tool"),
                    str(block.get("input") or "{}"),
                )
    return metadata
