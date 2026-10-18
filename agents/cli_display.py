"""CLI activity rendering. Start/update/end follows pi's agent event lifecycle.

https://github.com/badlogic/pi-mono/tree/main/packages/agent
The context is installed by the CLI only; no display events enter model history.
"""
from __future__ import annotations

import asyncio
import shlex
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import PurePosixPath

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text

from .tool_result import ShellResult

current_display: ContextVar = ContextVar("cli_display", default=None)
_current_activity: ContextVar = ContextVar("cli_activity", default=None)


@dataclass(eq=False)
class Activity:
    kind: str
    label: str
    started: float = field(default_factory=time.monotonic)
    updated: float = field(default_factory=time.monotonic)
    preview: str = ""
    result: object = None
    failed: bool | None = None
    baseline: tuple[int, int] = (0, 0)
    tool_name: str = ""


class CliDisplay:
    def __init__(self, console):
        self.console = console
        self.animated = console.is_terminal and not console.is_dumb_terminal and not console.legacy_windows
        self.live = None
        self.active: list[Activity] = []
        self.pending_text = ""
        self.line_open = False
        self.transcript = []
        self.printed = 0
        self.tool_count = 0
        self.attention_count = 0

    def __enter__(self):
        self.token = current_display.set(self)
        return self

    def __exit__(self, *_exc):
        try:
            self.pause()
        finally:
            current_display.reset(self.token)

    def __rich_console__(self, console, options):
        from .ui import SPINNER_FRAMES

        active = list(self.active)
        now = time.monotonic()
        position = now % sum(delay for _, delay in SPINNER_FRAMES)
        expression = "open"
        for expression, delay in SPINNER_FRAMES:
            if position < delay:
                break
            position -= delay
        eyes = {"open": "● ●", "wink": "● ─", "closed": "─ ─"}[expression]
        tools = [item for item in active if item.kind == "tool"]
        current = (tools or active)[-1] if active else None
        label = current.label if current else "Ready"
        elapsed = now - current.started if current else 0
        preview = current.preview if current else ""
        preview = preview.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        content = Group(
            Text(label, style="#65c7ff", no_wrap=True, overflow="ellipsis"),
            Text.from_ansi(preview[-1] if preview else "", style="dim", no_wrap=True, overflow="ellipsis"),
            Text(f"{self.tool_count} tools finished · {len(tools)} active · {self.attention_count} need attention",
                 style="dim", no_wrap=True, overflow="ellipsis"),
        )
        header_height = min(5, max(3, options.max_height - 2))
        idle = f" · idle {now - current.updated:.1f}s" if current and options.max_width >= 64 else ""
        yield Panel(content, title=Text(f"sans  {eyes}", style="bold #65c7ff"),
                    title_align="left", subtitle=Text(f"{elapsed:.1f}s{idle} · Ctrl+C stop", style="#9aa7b0"),
                    subtitle_align="right", border_style="#426577", height=header_height)
        available = max(1, options.max_height - header_height)
        # Render only the tail during a turn; pause() preserves the complete transcript in scrollback.
        body = self.transcript[-available:]
        if self.pending_text.strip():
            body = [*body, Markdown(self.pending_text)]
        lines = console.render_lines(Group(*body), options.update(height=None), pad=False)
        for line in lines[-available:]:
            yield from line
            yield Segment.line()

    def write(self, *objects, **kwargs):
        if self.live is not None:
            self.flush_text()
            self.transcript.extend(self.console.render_str(obj) if isinstance(obj, str) else obj
                                   for obj in objects)
        else:
            self.console.print(*objects, **kwargs)

    def resume(self):
        if self.animated and self.live is None:
            self.live = Live(self, console=self.console, refresh_per_second=8,
                             screen=True, transient=True, redirect_stdout=False, redirect_stderr=False)
            self.live.start(refresh=True)

    def flush_text(self):
        if self.line_open:
            self.console.print()
            self.line_open = False
        if self.pending_text:
            pending, self.pending_text = self.pending_text, ""
            if pending.strip():
                self.transcript.append(Markdown(pending))

    def pause(self):
        self.flush_text()
        if self.live is not None:
            self.live.stop()
            self.live = None
        for entry in self.transcript[self.printed:]:
            self.console.print(entry)
        self.printed = len(self.transcript)
        if not self.active:
            self.transcript.clear()
            self.printed = 0

    def text(self, text):
        item = _current_activity.get()
        if item and item.kind == "model" and item.label == "Receiving reasoning":
            return  # Anthropic's existing output buffers still receive the original text.
        if not self.animated:
            self.console.print(Text(text), end="", soft_wrap=True)
            if text:
                self.line_open = not text.endswith("\n")
            return
        self.pending_text += text
        self.resume()

    def begin(self, kind, label):
        self.flush_text()
        self.active[:] = [item for item in self.active if item.kind != "retry"]
        if kind == "turn" and not self.active:
            self.tool_count = self.attention_count = 0
        item = Activity(kind, label)
        self.active.append(item)
        self.resume()
        return item

    def update(self, item, label=None, text=""):
        if item not in self.active:
            return
        if label and label != item.label:
            item.label = label
            item.preview = ""
        item.updated = time.monotonic()
        if text:
            item.preview = (item.preview + text)[-3000:]

    def finish(self, item, result=None, error=None):
        self.flush_text()
        self.active.remove(item)
        status, detail = "✓", ""
        if error is not None:
            status = "■" if isinstance(error, asyncio.CancelledError) else "✗"
            detail = "cancelled" if status == "■" else str(error)
        elif item.kind == "completion" and result:
            status, detail = "!", str(result)
        elif item.kind == "tool":
            status = "✗" if item.failed else "✓" if item.failed is False else "!"
            raw = item.result
            if isinstance(raw, ShellResult):
                detail = f"exit {raw.exit_code}" if raw.exit_code is not None else "exit unknown"
                if raw.timed_out:
                    detail = "timed out"
                if raw.error:
                    detail += f" · {raw.error}"
                if raw.pytest_report:
                    report = raw.pytest_report
                    if raw.test_outcome == "inconclusive" and not raw.failed:
                        status = "!"
                    detail += f" · tests {raw.test_outcome} ({report.collected} collected)"
                if raw.failed:
                    detail += "\n" + (raw.stderr or raw.stdout)[-1500:]
            elif raw is not None:
                detail = (f"{len(str(raw).splitlines())} output lines" if not item.failed and item.tool_name in
                          {"read_file", "list_files", "grep_search"} else str(raw)[:500])
            else:
                detail = "execution not observed"
        elif item.kind == "compaction" and not result:
            status, detail = "—", "no compaction needed"
        if item.kind == "tool":
            self.tool_count += 1
        if status in {"✗", "!", "■"}:
            self.attention_count += 1
        quiet = item.kind in {"model", "turn", "completion", "compaction"} or item.tool_name in {
            "read_file", "list_files", "grep_search", "update_plan", "update_work_state"}
        if not quiet or status in {"✗", "!", "■"}:
            style = "red" if status == "✗" else "yellow" if status in {"!", "■"} else "dim"
            label = item.label[:77] + "..." if len(item.label) > 80 else item.label
            self.write(Text(f"  {status} {label}  ·  {time.monotonic() - item.started:.1f}s"
                            + (f"  ·  {detail}" if detail else ""), style=style))
        if not self.active:
            self.pause()

    def cost(self, input_tokens, output_tokens, cost):
        self.flush_text()
        turn = next((item for item in self.active if item.kind == "turn"), None)
        base_in, base_out = turn.baseline if turn else (0, 0)
        price = "not reported" if cost is None else f"${cost:.8f}"
        self.write(Text(
            f"  Turn: input {input_tokens - base_in:,} · output {output_tokens - base_out:,} tokens\n"
            f"  Session: input {input_tokens:,} · output {output_tokens:,} tokens · cost {price}", style="dim"))


def activity(kind):
    """Observe an existing async boundary, without changing its return or exception."""
    def decorate(fn):
        @wraps(fn)
        async def wrapped(*args, **kwargs):
            display = current_display.get()
            if display is None:
                return await fn(*args, **kwargs)
            labels = {"turn": "Working", "model": "Waiting for model",
                      "completion": "Checking completion evidence", "compaction": "Compacting context"}
            label = labels.get(kind, kind)
            name = ""
            if kind == "tool":
                from .ui import _get_tool_summary
                name = args[1] if len(args) > 1 else kwargs["name"]
                inp = args[2] if len(args) > 2 else kwargs["inp"]
                summary = _get_tool_summary(name, inp)
                if name == "run_shell":
                    summary = str(inp.get("command") or "").strip()
                    try:
                        first = shlex.split(summary, posix=False)[0]
                        executable = PurePosixPath(first.strip("\"'").replace("\\", "/")).name
                        summary = executable + summary[len(first):]
                    except (ValueError, IndexError):
                        pass
                    summary = summary[:117] + "..." if len(summary) > 120 else summary
                label = f"{name}  {summary}"
            item = display.begin(kind, label)
            item.tool_name = name
            if kind == "turn":
                item.baseline = (args[0].total_input_tokens, args[0].total_output_tokens)
            token = _current_activity.set(item)
            result = error = None
            try:
                result = await fn(*args, **kwargs)
                return result
            except BaseException as caught:
                error = caught
                raise
            finally:
                display.finish(item, result, error)
                _current_activity.reset(token)
        return wrapped
    return decorate


def stream_phase(label, text=""):
    display, item = current_display.get(), _current_activity.get()
    if display is None or item is None:
        return False
    display.update(item, label, text)
    return True


def tool_execution_result(result, failed):
    item = _current_activity.get()
    if current_display.get() is not None and item is not None:
        item.result, item.failed = result, failed


def shell_output(text):
    display, item = current_display.get(), _current_activity.get()
    if display is not None and item is not None:
        display.update(item, text=text)


def cli_session(fn):
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        from . import ui
        from .cli_shell import run_shell
        from .tools import shell_runner
        with CliDisplay(ui.console):
            token = shell_runner.set(run_shell)
            try:
                return await fn(*args, **kwargs)
            finally:
                shell_runner.reset(token)
    return wrapped
