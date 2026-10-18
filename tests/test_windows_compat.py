from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.mcp_client import (
    McpConnection,
    McpManager,
    resolve_mcp_command,
)
from agents.tools import _grep_python, _grep_search, _run_shell, execute_tool


def run_tool(name: str, inp: dict, read_file_state: dict[str, float] | None = None) -> str:
    return asyncio.run(execute_tool(name, inp, read_file_state))


def test_run_shell_keeps_success_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agents.tools.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="tests passed\n"),
    )

    result = _run_shell({"command": "test"})
    assert result.exit_code == 0 and result.stderr == "tests passed\n"
    assert str(result) == "tests passed\n"


def test_grep_reports_execution_and_walk_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.tools.IS_WIN", False)
    monkeypatch.setattr(
        "agents.tools.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="denied"),
    )
    assert _grep_search({"pattern": "x", "path": "."}) == (
        "Error searching files (exit code 2): denied"
    )

    def denied(_: str) -> list[str]:
        raise PermissionError("denied")

    monkeypatch.setattr("agents.tools.os.listdir", denied)
    assert _grep_python("x", ".", None).startswith("Error searching files:")
    assert _grep_python("[", ".", None).startswith("Error searching files: invalid regex:")


def test_read_file_preserves_utf8_chinese(tmp_path: Path) -> None:
    path = tmp_path / "测试中文文件.md"
    content = "Sans 是一个自进化 Harness Agent。\n这是中文测试。"
    path.write_text(content, encoding="utf-8")

    result = run_tool("read_file", {"file_path": str(path)})

    assert "Sans 是一个自进化 Harness Agent。" in result
    assert "这是中文测试。" in result
    assert "鏄" not in result
    assert "涓" not in result


def test_read_file_honors_offset_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = run_tool("read_file", {"file_path": str(path), "offset": 1, "limit": 2})

    assert result == "   2 | two\n   3 | three"


def test_write_file_uses_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("utf8-write.txt")
    content = "你好 Sans\n中文写入正常"

    result = run_tool(
        "write_file",
        {"file_path": str(path), "content": content},
        {},
    )

    assert result.startswith("Successfully wrote")
    assert path.read_text(encoding="utf-8") == content
    assert path.read_bytes() == content.encode("utf-8")


def test_edit_file_preserves_utf8_and_read_before_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("utf8-edit.md")
    original = "Sans 是 Harness。"
    updated = "Sans 是 Agentic Harness。"
    path.write_text(original, encoding="utf-8")
    read_file_state: dict[str, float] = {}

    denied = run_tool(
        "edit_file",
        {"file_path": str(path), "old_string": "Harness", "new_string": "Agentic Harness"},
        read_file_state,
    )
    assert denied.startswith("Error: You must read this file before editing")

    run_tool("read_file", {"file_path": str(path)}, read_file_state)
    result = run_tool(
        "edit_file",
        {"file_path": str(path), "old_string": "Harness", "new_string": "Agentic Harness"},
        read_file_state,
    )

    assert result.startswith("Successfully edited")
    assert path.read_text(encoding="utf-8") == updated


def test_missing_mcp_command_is_actionable_and_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = McpManager()
    monkeypatch.setattr(
        manager,
        "_load_configs",
        lambda: {"context7": {"command": "definitely-not-existing-command"}},
    )

    asyncio.run(manager.load_and_connect())

    unavailable = manager.get_unavailable_servers()
    assert "context7" in unavailable
    assert 'MCP server "context7" could not start:' in unavailable["context7"]
    assert 'command "definitely-not-existing-command" was not found.' in unavailable["context7"]
    assert manager.get_tool_definitions() == []


def test_windows_npx_resolves_to_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = r"C:\Node\npx.cmd"

    def fake_which(command: str) -> str | None:
        return expected if command == "npx" else None

    monkeypatch.setattr("agents.mcp_client.os.name", "nt")
    monkeypatch.setattr("agents.mcp_client.shutil.which", fake_which)

    assert resolve_mcp_command("npx") == expected


def test_existing_python_command_still_resolves() -> None:
    resolved = resolve_mcp_command(sys.executable)

    assert os.path.normcase(resolved) == os.path.normcase(sys.executable)


def test_mcp_connection_uses_resolved_command(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = r"C:\Node\npx.cmd"
    calls: list[tuple[object, ...]] = []

    class DummyStream:
        async def readline(self) -> bytes:
            await asyncio.sleep(3600)
            return b""

    class DummyProcess:
        stdin = None
        stdout = DummyStream()

        def kill(self) -> None:
            pass

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> DummyProcess:
        calls.append(args)
        return DummyProcess()

    monkeypatch.setattr("agents.mcp_client.resolve_mcp_command", lambda command: expected)
    monkeypatch.setattr(
        "agents.mcp_client.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    async def exercise() -> None:
        connection = McpConnection("playwright", "npx", ["@playwright/mcp@latest"])
        await connection.connect()
        try:
            assert calls[0][0] == expected
            assert calls[0][1:] == ("@playwright/mcp@latest",)
        finally:
            connection.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("absolute_path", [False, True], ids=["relative", "absolute"])
def test_windows_grep_file_path_and_include_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute_path: bool
) -> None:
    monkeypatch.setattr("agents.tools.IS_WIN", True)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "module with spaces.py"
    target.write_text("# header\ndef dispatch():\n    mode = 'fast'\n", encoding="utf-8")
    (tmp_path / "neighbor.py").write_text("mode = 'sibling_only'\n", encoding="utf-8")
    path = str(target) if absolute_path else target.name
    query = {"path": path, "pattern": r"def dispatch|mode\s*=", "include": "*.py"}

    result = run_tool("grep_search", query)

    assert result.splitlines() == [
        f"{path}:2:def dispatch():",
        f"{path}:3:    mode = 'fast'",
    ]
    assert run_tool("grep_search", {**query, "include": "*.txt"}) == "No matches found."
