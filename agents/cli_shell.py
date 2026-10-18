"""Native shell streaming adapter, installed only for CLI sessions."""
from __future__ import annotations

import asyncio
import codecs
import locale
import os
import shlex
import signal
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from .cli_display import shell_output
from .tool_result import PytestReport, ShellResult


async def run_shell(inp):
    if not inp.get("_pytest_report"):
        return await _run_command(inp)
    cache = Path(__file__).resolve().parent.parent / "experiment/cache/pytest-reports"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        report = Path(temporary) / "junit.xml"
        path = f'"{report.as_posix()}"' if os.name == "nt" else shlex.quote(str(report))
        result = await _run_command({**inp, "command": f'{inp["command"]} --junitxml={path}'})
        return replace(result, pytest_report=PytestReport.read(report))


async def _kill_tree(process):
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        await killer.wait()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.returncode is None:
        process.kill()
    await process.wait()


async def _run_command(inp):
    output = [[], []]

    async def pump(reader, target):
        decoder = codecs.getincrementaldecoder(locale.getpreferredencoding(False))(errors="replace")
        while data := await reader.read(4096):
            text = decoder.decode(data)
            target.append(text)
            shell_output(text)
        tail = decoder.decode(b"", final=True)
        target.append(tail)
        if tail:
            shell_output(tail)

    timeout = inp.get("timeout", 30000)
    timed_out = False
    try:
        process = await asyncio.create_subprocess_shell(
            inp["command"], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}),
        )
    except (OSError, ValueError) as error:
        return ShellResult(error=f"Error: {error}")
    readers = asyncio.gather(pump(process.stdout, output[0]), pump(process.stderr, output[1]), process.wait())
    try:
        await asyncio.wait_for(asyncio.shield(readers), timeout / 1000)
    except TimeoutError:
        timed_out = True
        await _kill_tree(process)
        await readers
    except BaseException:
        await _kill_tree(process)
        await readers
        raise
    stdout, stderr = ("".join(parts).replace("\r\n", "\n").replace("\r", "\n") for parts in output)
    return ShellResult(stdout, stderr, None if timed_out else process.returncode,
                       timed_out=timed_out, error=f"Command timed out after {timeout}ms" if timed_out else None)
