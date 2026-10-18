"""Terminal UI rendering — colored output, spinner, tool display."""

from __future__ import annotations

import sys
import threading

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from .cli_display import current_display

console = Console(highlight=False)


def _print(*objects, **kwargs) -> None:
    if display := current_display.get():
        display.write(*objects, **kwargs)
    else:
        console.print(*objects, **kwargs)


def _flush_text() -> None:
    if display := current_display.get():
        display.flush_text()

# ─── Basic output ──────────────────────────────────────────


def _safe_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _safe_stdout_write(text: object) -> None:
    sys.stdout.write(_safe_text(text))
    sys.stdout.flush()


# Terminal pixel portrait, inspired by Sans's Undertale dialogue sprite.
# Reference: https://www.spriters-resource.com/pc_computer/undertale/asset/76325/
SANS_FACE = (
    "      ############      ",
    "    ################    ",
    "   ##################   ",
    "  ####################  ",
    "  ###     ####     ###  ",
    " ###       ##       ### ",
    " ###   #   ##   #   ### ",
    " ###       ##       ### ",
    " ####     ####     #### ",
    " ##########  ########## ",
    "  ########    ########  ",
    "  ####################  ",
    "  ##              ##    ",
    "   ## # # # # # ###     ",
    "    ##############      ",
    "      ##########        ",
)


def sans_face(expression: str = "open") -> Text:
    pixels = [list(row) for row in SANS_FACE]
    for left in ((13,) if expression == "wink" else (4, 13) if expression == "closed" else ()):
        for y in range(4, 9):
            pixels[y][left:left + 7] = "#" * 7
        pixels[6][left + 1:left + 6] = " " * 5
    rows = []
    for top, bottom in zip(pixels[::2], pixels[1::2]):
        rows.append("".join(" ▄▀█"[(a == "#") * 2 + (b == "#")] for a, b in zip(top, bottom)))
    return Text("\n".join(rows), style="bold #f0f4ff on #10141c", no_wrap=True)


def print_welcome() -> None:
    title = Text("sans", style="bold #65c7ff")
    subtitle = Text("Evolvable Coding Agent CLI", style="bold cyan")

    commands = Table.grid(padding=(0, 2))
    commands.add_column(style="bold cyan", no_wrap=True)
    commands.add_column(style="dim")
    commands.add_row("/plan", "read-only planning workflow")
    commands.add_row("/skills", "list reusable skills")
    commands.add_row("/skill-create", "create a reusable skill")
    commands.add_row("/skill-stats", "show skill evolution stats")
    commands.add_row("/memory", "list long-term memories")
    commands.add_row("/compact", "compact current context")
    commands.add_row("/trace", "show this session's trace")
    commands.add_row("/exit, /quit", "quit the session")

    body = Table.grid(expand=True)
    body.add_row(Align.center(sans_face()))
    body.add_row(Align.center(title))
    body.add_row(Align.center(subtitle))
    body.add_row("")
    body.add_row(Panel(commands, title="Quick Commands", border_style="cyan", box=box.ROUNDED))

    _print()
    _print(Panel(
        body,
        title="[bold #65c7ff] sans · ready [/bold #65c7ff]",
        subtitle="[dim]Type your request below[/dim]",
        border_style="#65c7ff",
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    _print()


def print_user_prompt() -> None:
    if display := current_display.get():
        display.pause()
    _print("\n[bold #65c7ff]sans[/bold #65c7ff] [dim]❯[/dim] ", end="")


def print_assistant_text(text: str) -> None:
    if display := current_display.get():
        display.text(_safe_text(text))
        return
    _safe_stdout_write(text)


def print_tool_call(name: str, inp: dict) -> None:
    if current_display.get() is not None:
        return  # The execution boundary owns the CLI activity row.
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, inp)
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold yellow", no_wrap=True)
    table.add_column(style="white")
    table.add_row("tool", f"{icon} {name}")
    if summary:
        table.add_row("input", _safe_text(summary))
    _print(Panel(
        table,
        title="[bold yellow]Tool Call[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_tool_result(name: str, result: str) -> None:
    if current_display.get() is not None:
        return  # CLI receipts use the structured result before model formatting.
    result = _safe_text(result)
    if (name in ("edit_file", "write_file")) and not result.startswith("Error"):
        _print_file_change_result(name, result)
        return
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
    _print(Panel(
        _safe_text(truncated),
        title=f"[dim]{name} result[/dim]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def _print_file_change_result(_name: str, result: str) -> None:
    lines = result.split("\n")
    _print(Panel(
        _safe_text(lines[0]),
        title="[bold green]File Change[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    ))

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
        if not line.strip():
            continue
        if line.startswith("@@"):
            _print(f"[cyan]  {line}[/cyan]")
        elif line.startswith("- "):
            _print(f"[red]  {line}[/red]")
        elif line.startswith("+ "):
            _print(f"[green]  {line}[/green]")
        else:
            _print(f"[dim]  {line}[/dim]")
    if len(content_lines) > max_display:
        _print(f"[dim]  ... ({len(content_lines) - max_display} more lines)[/dim]")


def print_error(msg: str) -> None:
    _flush_text()
    _print(Panel(
        _safe_text(msg),
        title="[bold red]Error[/bold red]",
        border_style="red",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_confirmation(command: str) -> None:
    if display := current_display.get():
        display.pause()
    _print(Panel(
        _safe_text(command),
        title="[bold yellow]Dangerous command[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_divider() -> None:
    _flush_text()
    _print(Rule("[dim]turn complete[/dim]", style="dim"))


def print_cost(
    input_tokens: int,
    output_tokens: int,
    provider_reported_cost: float | None = None,
) -> None:
    if display := current_display.get():
        display.cost(input_tokens, output_tokens, provider_reported_cost)
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="white")
    table.add_row("input", f"{input_tokens} tokens")
    table.add_row("output", f"{output_tokens} tokens")
    table.add_row(
        "provider cost",
        "not reported" if provider_reported_cost is None else f"${provider_reported_cost:.8f}",
    )
    _print(Panel(table, title="Cost", border_style="cyan", box=box.ROUNDED))


def print_retry(attempt: int, max_retries: int, reason: str, delay: float = 0) -> None:
    if display := current_display.get():
        display.begin("retry", f"Retry {attempt}/{max_retries} · {reason} · waiting {delay:.1f}s")
        return
    _print(_safe_text(f"\n  [yellow]↻ Retry {attempt}/{max_retries}: {reason}[/yellow]"))


def print_info(msg: str) -> None:
    _flush_text()
    _print(Panel(
        _safe_text(msg),
        title="[bold cyan]Info[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_warning(msg: str) -> None:
    _flush_text()
    _print(Panel(
        _safe_text(msg),
        title="[bold yellow]Notice[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_goodbye() -> None:
    _print(Panel(
        Text("see you next time.", style="bold #65c7ff"),
        border_style="#65c7ff",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_interrupted() -> None:
    print_warning("Interrupted. Press Ctrl+C again to exit.")


# ─── Spinner ──────────────────────────────────────────────

SPINNER_FRAMES = (("open", 1.2), ("wink", 0.25), ("open", 0.8), ("closed", 0.35))

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()
_spinner_live: Live | None = None


def start_spinner(label: str = "Thinking") -> None:
    global _spinner_thread, _spinner_live
    if current_display.get() is not None:
        return
    if _spinner_thread is not None or not console.is_terminal or console.is_dumb_terminal:
        return
    _spinner_stop.clear()
    live = Live(console=console, auto_refresh=False, transient=True)
    _spinner_live = live
    live.start()

    def _run() -> None:
        frame = 0
        while not _spinner_stop.is_set():
            expression, delay = SPINNER_FRAMES[frame]
            live.update(Group(sans_face(expression), Text(f"  sans · {label.lower()}...", style="#65c7ff")), refresh=True)
            if _spinner_stop.wait(delay):
                break
            frame = (frame + 1) % len(SPINNER_FRAMES)

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    global _spinner_thread, _spinner_live
    if _spinner_thread is None:
        return
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    if _spinner_live is not None:
        _spinner_live.stop()
        _spinner_live = None


# ─── Plan approval display ──────────────────────────────────


def print_plan_for_approval(plan_content: str) -> None:
    if display := current_display.get():
        display.pause()
    lines = plan_content.split("\n")
    max_lines = 60
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n\n... ({len(lines) - max_lines} more lines)"
    _print(Panel(
        _safe_text(preview),
        title="[bold cyan]Plan for Approval[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def print_plan_approval_options() -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("choice", style="bold yellow", no_wrap=True)
    table.add_column("action", style="white")
    table.add_column("detail", style="dim")
    table.add_row("1", "Clear context and execute", "fresh start with auto-accept edits")
    table.add_row("2", "Execute", "keep context, auto-accept edits")
    table.add_row("3", "Manually approve edits", "keep context, confirm each edit")
    table.add_row("4", "Keep planning", "provide feedback to revise")
    _print(Panel(table, title="[bold yellow]Choose an option[/bold yellow]", border_style="yellow", box=box.ROUNDED))


# ─── Sub-agent display ──────────────────────────────────────


def print_sub_agent_start(agent_type: str, description: str) -> None:
    _print(Panel(
        _safe_text(description),
        title=f"[bold magenta]Sub-agent started: {agent_type}[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    _print(Panel(
        "completed",
        title=f"[bold magenta]Sub-agent finished: {agent_type}[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def print_memory_entries(memories: list[object]) -> None:
    table = Table(box=box.ROUNDED, header_style="bold cyan", border_style="cyan")
    table.add_column("Type", style="bold #f6c177", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Description", style="dim")
    for m in memories:
        table.add_row(
            _safe_text(getattr(m, "type", "")),
            _safe_text(getattr(m, "name", "")),
            _safe_text(getattr(m, "description", "")),
        )
    _print(Panel(table, title="[bold cyan]Memories[/bold cyan]", border_style="cyan", box=box.ROUNDED))


def print_skill_entries(skills: list[object]) -> None:
    table = Table(box=box.ROUNDED, header_style="bold cyan", border_style="cyan")
    table.add_column("Skill", style="bold #f6c177", no_wrap=True)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Mode", style="magenta", no_wrap=True)
    table.add_column("Description", style="white")
    for s in skills:
        name = getattr(s, "name", "")
        tag = f"/{name}" if getattr(s, "user_invocable", False) else name
        table.add_row(
            _safe_text(tag),
            _safe_text(getattr(s, "source", "")),
            _safe_text(getattr(s, "context", "")),
            _safe_text(getattr(s, "description", "")),
        )
    _print(Panel(table, title="[bold cyan]Skills[/bold cyan]", border_style="cyan", box=box.ROUNDED))


# ─── Tool icons and summaries ───────────────────────────────

_TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "🔧",
    "list_files": "📁",
    "grep_search": "🔍",
    "run_shell": "💻",
    "skill": "⚡",
    "skill_create": "✦",
    "skill_evolve": "🧬",
    "agent": "🤖",
}


def _get_tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔨")


def _get_tool_summary(name: str, inp: dict) -> str:
    if name == "read_file":
        return inp.get("file_path", "")
    if name == "write_file":
        return inp.get("file_path", "")
    if name == "edit_file":
        return inp.get("file_path", "")
    if name == "list_files":
        return inp.get("pattern", "")
    if name == "grep_search":
        return f'"{inp.get("pattern", "")}" in {inp.get("path", ".")}'
    if name == "run_shell":
        cmd = inp.get("command", "")
        return cmd[:60] + "..." if len(cmd) > 60 else cmd
    if name == "skill":
        return inp.get("skill_name", "")
    if name == "skill_create":
        return inp.get("name", "")
    if name == "skill_evolve":
        return inp.get("skill_name", "")
    if name == "agent":
        return f'[{inp.get("type", "general")}] {inp.get("description", "")}'
    return ""
