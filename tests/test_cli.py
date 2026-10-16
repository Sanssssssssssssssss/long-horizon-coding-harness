import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from agents import main as cli
from agents.telemetry import read_trace
from agents import ui
from agents.cli_display import CliDisplay, current_display
from agents import cli_shell
from agents.tools import shell_runner, _run_shell
from rich.console import Console


def test_sans_animation_restores_terminal_and_skips_redirected_output(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    terminal = Console(file=output, force_terminal=True, width=80, color_system="truecolor")
    monkeypatch.setattr(ui, "console", terminal)
    monkeypatch.setattr(ui, "SPINNER_FRAMES", (("open", 0.01), ("wink", 0.01), ("closed", 0.01)))
    seen = set()
    closed = threading.Event()
    render_face = ui.sans_face

    def record_frame(expression="open"):
        seen.add(expression)
        if expression == "closed":
            closed.set()
        return render_face(expression)

    monkeypatch.setattr(ui, "sans_face", record_frame)
    previous_stdout = sys.stdout
    try:
        ui.start_spinner()
        first_thread = ui._spinner_thread
        ui.start_spinner()
        assert ui._spinner_thread is first_thread
        assert closed.wait(2), "The animation never reached the closed-eye frame"
    finally:
        ui.stop_spinner()
    assert seen == {"open", "wink", "closed"}
    assert first_thread is not None and not first_thread.is_alive()
    assert ui._spinner_live is None
    assert "\x1b[?25h" in output.getvalue()  # Cursor restored after the transient display.
    assert sys.stdout is previous_stdout
    terminal.print("TOOL_OUTPUT_AFTER_THINKING")
    assert output.getvalue().endswith("TOOL_OUTPUT_AFTER_THINKING\n")
    ui.stop_spinner()

    output.seek(0)
    output.truncate()
    monkeypatch.setattr(ui, "console", Console(file=output, force_terminal=False))
    ui.start_spinner()
    ui.stop_spinner()
    assert output.getvalue() == ""
    assert ui._spinner_thread is None


def test_cli_process_records_real_tools_resume_and_interactive_exit(tmp_path):
    """Exercise the executable, HTTP streaming, native shell and on-disk history."""
    repo = Path(__file__).resolve().parents[1]
    requests = []
    argv = [sys.executable, "-B", "-c", "print('Command failed (exit code 9)'); print('CLI_TOOL_OK')"]
    command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)

    class Provider(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            delta = {"content": "CLI_RESUME_OK" if len(requests) > 2 else "CLI_TOOL_OK"}
            finish = "stop"
            if len(requests) == 1:
                delta = {"tool_calls": [{"index": 0, "id": "call_local", "type": "function",
                         "function": {"name": "run_shell", "arguments": json.dumps({"command": command})}}]}
                finish = "tool_calls"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in (
                {"choices": [{"index": 0, "delta": {"reasoning_content": "Checking the command."}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                 "total_tokens": 110, "prompt_cache_hit_tokens": 20, "prompt_cache_miss_tokens": 80}},
            ):
                data = ("data: " + json.dumps(chunk) + "\n\n").encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")
            self.wfile.flush()

    isolated_home = tmp_path / "experiment" / "home"
    isolated_home.mkdir(parents=True)
    with ThreadingHTTPServer(("127.0.0.1", 0), Provider) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = {**os.environ, "APIKEY": "local-test-key", "API": f"http://127.0.0.1:{server.server_port}/v1",
               "PYTHONPATH": str(repo), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "HOME": str(isolated_home), "USERPROFILE": str(isolated_home),
               "SANS_DISABLE_MCP": "1", "SANS_DISABLE_MEMORY": "1", "SANS_AUTO_SKILL_EVOLUTION": "0",
               "SANS_DISABLE_SESSION_PERSISTENCE": "0", "NO_PROXY": "127.0.0.1,localhost"}

        def run(*args, input="", exit_code=0):
            result = subprocess.run([sys.executable, "-B", "-m", "agents.main", "--model", "local-test", *args],
                                    cwd=tmp_path, env=env, input=input, capture_output=True,
                                    text=True, encoding="utf-8", timeout=30)
            assert result.returncode == exit_code, result.stdout + result.stderr
            assert "an error occurred during closing" not in result.stderr, result.stderr
            return result

        try:
            result = run("--yolo", "Run the check and report its output.")
            assert "CLI_TOOL_OK" in result.stdout
            assert "Waiting for model" not in result.stdout  # Transient status never enters scrollback.
            assert "Receiving reasoning" not in result.stdout
            assert "Receiving tool arguments" not in result.stdout
            assert "Checking completion evidence" not in result.stdout
            assert "exit 0" in result.stdout
            assert "CLI_TOOL_OK\n" in result.stdout  # A response without a newline is closed before receipts.
            first = next((tmp_path / "experiment/runs/cli").iterdir())
            trace = read_trace(first / "trace.jsonl")
            summary = json.loads((first / "summary.json").read_text())
            assert summary["model_calls_started"] == 2
            assert summary["provider_usage"]["total_tokens"] == 220
            assert summary["calls_without_usage"] == 0
            tool = next(event for event in trace if event["event_type"] == "tool.call.end")
            payload = json.loads((first / "blobs" / tool["payload_ref"]["sha256"]).read_text())
            assert payload["execution"]["exit_code"] == 0
            assert "Command failed (exit code 9)" in payload["execution"]["stdout"]
            starts = {event["span_id"] for event in trace if event["event_type"].endswith(".start")}
            assert all(event["parent_span_id"] in starts for event in trace if event["parent_span_id"])
            assert trace[0]["event_type"] == "cli.run.start"
            assert trace[-1]["event_type"] == "cli.run.end"
            assert "local-test-key" not in (first / "trace.jsonl").read_text()
            for event in trace:
                if ref := event["payload_ref"]:
                    blob = (first / "blobs" / ref["sha256"]).read_bytes()
                    assert hashlib.sha256(blob).hexdigest() == ref["sha256"]
                    assert b"local-test-key" not in blob

            first_bytes = (first / "trace.jsonl").read_bytes()
            result = run("--resume", "Continue the previous conversation.")
            assert "CLI_RESUME_OK" in result.stdout
            assert any(message.get("tool_call_id") == "call_local" for message in requests[-1]["messages"])
            runs = list((tmp_path / "experiment/runs/cli").iterdir())
            resumed = next(path for path in runs if path != first)
            resumed_trace = read_trace(resumed / "trace.jsonl")
            assert resumed_trace[0]["attributes"]["resumed_from_session_id"] == trace[-1]["attributes"]["session_id"]
            assert (first / "trace.jsonl").read_bytes() == first_bytes
            assert not (isolated_home / ".sans/sessions").exists()

            interactive_result = run(input="/trace\n/exit\nTHIS_MUST_NOT_REACH_THE_MODEL\n")
            assert "sans" in interactive_result.stdout
            assert "Bear" not in interactive_result.stdout
            assert "mini-claude" not in interactive_result.stdout
            interactive = next(path for path in (tmp_path / "experiment/runs/cli").iterdir() if path not in runs)
            assert json.loads((interactive / "summary.json").read_text())["provider_usage"]["total_tokens"] == 0
            assert len(requests) == 3
            run("--no-trace", input="/quit\nTHIS_MUST_NOT_REACH_THE_MODEL\n")
            assert len(list((tmp_path / "experiment/runs/cli").iterdir())) == 3
            run("--max-cost", "0", "Check the execution limit.", exit_code=2)
            limited = next(path for path in (tmp_path / "experiment/runs/cli").iterdir()
                           if path not in [*runs, interactive])
            assert read_trace(limited / "trace.jsonl")[-1]["status"] == "incomplete"
            assert json.loads((limited / "summary.json").read_text())["exit_code"] == 2
            assert len(requests) == 3
            no_trace = run("--no-trace", "Continue without a trace.")
            assert "CLI_RESUME_OK" in no_trace.stdout
            assert "Waiting for model" not in no_trace.stdout
            assert len(requests) == 4
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _python_command(script):
    argv = [sys.executable, "-B", "-u", "-c", script]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def test_cli_shell_streams_before_exit_without_changing_execution_facts(tmp_path, monkeypatch):
    gate = tmp_path / "continue"
    command = _python_command(
        "import time,sys; from pathlib import Path; print('EARLY',flush=True); "
        f"gate=Path({str(gate)!r}); deadline=time.monotonic()+5; "
        "exec('while not gate.exists() and time.monotonic()<deadline: time.sleep(.01)'); "
        "print('Command failed (exit code 9)'); print('stderr-tail',file=sys.stderr); "
        "sys.exit(0 if gate.exists() else 8)"
    )

    async def exercise():
        early = asyncio.Event()
        chunks = []

        def update(text):
            chunks.append(text)
            if "EARLY" in "".join(chunks):
                early.set()

        monkeypatch.setattr(cli_shell, "shell_output", update)
        task = asyncio.create_task(cli_shell.run_shell({"command": command}))
        try:
            await asyncio.wait_for(early.wait(), 5)
            assert not task.done(), "Output was buffered until process exit"
            gate.touch()
            result = await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert result.exit_code == 0 and not result.failed
        assert result.stdout == "EARLY\nCommand failed (exit code 9)\n"
        assert result.stderr == "stderr-tail\n"
        assert "stderr-tail" in "".join(chunks)

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_cli_shell_timeout_and_cancel_kill_descendants(tmp_path, monkeypatch, cancel):
    escaped = tmp_path / "escaped"
    child = f"import time; from pathlib import Path; time.sleep(3); Path({str(escaped)!r}).touch()"
    command = _python_command(
        f"import subprocess,sys; p=subprocess.Popen([sys.executable,'-B','-c',{child!r}]); "
        "print('CHILD_STARTED',flush=True); p.wait()"
    )

    async def exercise():
        started = asyncio.Event()
        monkeypatch.setattr(cli_shell, "shell_output", lambda text: started.set() if "CHILD_STARTED" in text else None)
        task = asyncio.create_task(cli_shell.run_shell({"command": command, "timeout": 1500}))
        await asyncio.wait_for(started.wait(), 5)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.timed_out and result.failed and result.exit_code is None
            assert "CHILD_STARTED" in result.stdout
        await asyncio.sleep(3.1)
        assert not escaped.exists(), "A descendant survived cancellation/timeout"

    asyncio.run(exercise())


def test_cli_shell_preserves_pytest_receipts(tmp_path):
    test = tmp_path / "test_example.py"
    test.write_text("def test_pass(): assert True\ndef test_fail(): assert False\n")
    argv = [sys.executable, "-B", "-m", "pytest", str(test), "-q", "-p", "no:cacheprovider",
            "--basetemp", str(tmp_path / "pytest")]
    command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    original = _run_shell({"command": command, "_pytest_report": True})
    streamed = asyncio.run(cli_shell.run_shell({"command": command, "_pytest_report": True}))
    assert original.exit_code == streamed.exit_code == 1
    assert original.pytest_report == streamed.pytest_report
    assert streamed.test_outcome == "failed"


def test_cli_display_keeps_partial_text_and_restores_terminal_on_error(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=80, legacy_windows=False)
    with pytest.raises(RuntimeError, match="display-check"):
        with CliDisplay(console) as display:
            item = display.begin("model", "Waiting for model")
            display.text("First partial response")
            display.live.refresh()
            assert "First partial response" in output.getvalue()
            display.update(item, "Receiving tool arguments")
            display.live.refresh()
            console.width = 48
            display.live.refresh()
            raise RuntimeError("display-check")
    assert "\x1b[?25h" in output.getvalue()
    assert "\x1b[?1049h" in output.getvalue() and "\x1b[?1049l" in output.getvalue()
    assert current_display.get() is None and shell_runner.get() is None
    assert console._live_stack == []


def test_cli_panel_stays_above_chat_and_exports_only_durable_messages(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=80, height=18, color_system=None, legacy_windows=False)
    with CliDisplay(console) as display:
        turn = display.begin("turn", "Working")
        model = display.begin("model", "Waiting for model")
        display.text("**RESPONSE_START**\n\n" + "\n\n".join(f"Paragraph {i}" for i in range(30)))
        for index in range(40):
            display.update(model, f"Transient phase {index}")
        for width in (80, 40):
            frame = io.StringIO()
            Console(file=frame, width=width, height=18).print(display)
            lines = frame.getvalue().splitlines()
            assert "sans" in lines[0]
            assert "Transient phase 39" in lines[1]
            assert "Paragraph 29" in frame.getvalue()
            assert len(lines) <= 18
        display.finish(model)
        for index in range(20):
            read = display.begin("tool", f"read_file file_{index}.py")
            read.tool_name, read.result, read.failed = "read_file", "read contents", False
            display.finish(read)
        display.pause()  # Approval/input must restore the original terminal.
        exported = output.getvalue().rsplit("\x1b[?1049l", 1)[1]
        assert exported.count("RESPONSE_START") == 1
        assert "Paragraph 0" in exported and "Paragraph 29" in exported
        assert "Transient phase" not in exported and "read_file" not in exported
        display.resume()
        failed = display.begin("tool", "run_shell check")
        display.finish(failed, error=RuntimeError("CHECK_FAILED"))
        display.finish(turn)
        resumed = output.getvalue().rsplit("\x1b[?1049l", 1)[1]
        assert "CHECK_FAILED" in resumed
        assert "RESPONSE_START" not in resumed  # Resuming cannot duplicate old messages.
    assert current_display.get() is None and console._live_stack == []

    fallback = io.StringIO()
    with CliDisplay(Console(file=fallback, force_terminal=True, legacy_windows=True, color_system=None)) as display:
        item = display.begin("model", "Waiting for model")
        display.text("LEGACY_RESPONSE")
        display.finish(item)
    assert "LEGACY_RESPONSE" in fallback.getvalue()
    assert "Waiting for model" not in fallback.getvalue() and "\x1b[?1049" not in fallback.getvalue()


def test_cli_observes_incomplete_model_stream_without_changing_response(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    async def exercise():
        output = io.StringIO()
        agent = cli.Agent(api_base="http://unused.invalid/v1", api_key="test", is_sub_agent=True,
                          custom_tools=[], custom_system_prompt="fixed", skills_enabled=False)
        reasoning_seen, arguments_seen = asyncio.Event(), asyncio.Event()
        allow_arguments, allow_finish = asyncio.Event(), asyncio.Event()

        def chunk(**values):
            delta = SimpleNamespace(content=None, reasoning_content=None, tool_calls=None)
            delta.__dict__.update(values)
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=None)])

        async def stream():
            yield chunk(reasoning_content="Check the actual exit code.")
            reasoning_seen.set()
            await allow_arguments.wait()
            yield chunk(tool_calls=[SimpleNamespace(index=0, id="call-1",
                function=SimpleNamespace(name="run_shell", arguments='{"command":'))])
            arguments_seen.set()
            await allow_finish.wait()
            completed = chunk(tool_calls=[SimpleNamespace(index=0, id=None,
                function=SimpleNamespace(name=None, arguments='"echo ok"}'))])
            completed.choices[0].finish_reason = "tool_calls"
            yield completed
            yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12))

        async def create(**_kwargs):
            return stream()

        monkeypatch.setattr(agent._openai_client.chat.completions, "create", create)
        try:
            with CliDisplay(Console(file=output, force_terminal=True, width=90, legacy_windows=False)) as display:
                task = asyncio.create_task(agent._call_openai_stream())
                await asyncio.wait_for(reasoning_seen.wait(), 3)
                display.live.refresh()
                assert "Receiving reasoning" in output.getvalue() and not task.done()
                allow_arguments.set()
                await asyncio.wait_for(arguments_seen.wait(), 3)
                display.live.refresh()
                assert "Receiving tool arguments" in output.getvalue() and not task.done()
                allow_finish.set()
                displayed = await task
            original = await agent._call_openai_stream()
            assert original == displayed
            assert original["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"command":"echo ok"}'
            assert original["choices"][0]["message"]["reasoning_content"] == "Check the actual exit code."
        finally:
            allow_arguments.set()
            allow_finish.set()
            await agent.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure,code,status", [(RuntimeError, 1, "error"), (asyncio.CancelledError, 130, "cancelled")])
def test_cli_failure_retains_unknown_usage_and_closes_transports(tmp_path, monkeypatch, failure, code, status):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sans", "hello"])
    monkeypatch.setattr(cli, "_load_env_file", lambda: None)
    monkeypatch.setattr(cli, "_resolve_api_config", lambda _base: ("http://unused.invalid/v1", "test", True))
    closed = []
    shutdown = cli.Agent.shutdown

    async def chat(self, _prompt):
        self.trace.event("model.call.start", component="model", span_id="pending-call")
        raise failure("interrupted request")

    async def close(self):
        await shutdown(self)
        closed.append(True)

    monkeypatch.setattr(cli.Agent, "chat", chat)
    monkeypatch.setattr(cli.Agent, "shutdown", close)
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == code
    assert closed == [True]
    assert signal.getsignal(signal.SIGINT) == before
    run = next((tmp_path / "experiment/runs/cli").iterdir())
    assert read_trace(run / "trace.jsonl")[-1]["status"] == status
    summary = json.loads((run / "summary.json").read_text())
    assert summary["exit_code"] == code
    assert summary["calls_without_usage"] == 1
    assert summary["provider_usage"]["total_tokens"] is None
    assert summary["provider_usage"]["usage_complete"] is False


@pytest.mark.parametrize("idle", [False, True])
def test_repl_interrupt_cancels_active_chat_then_cleans_up(tmp_path, monkeypatch, idle):
    monkeypatch.chdir(tmp_path)
    agent = cli.Agent(api_base="http://unused.invalid/v1", api_key="test", is_sub_agent=True)
    before = signal.getsignal(signal.SIGINT)
    inputs = iter(["hello", "exit"])
    def read_input(*_args):
        if idle:
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)
            handler(signal.SIGINT, None)
        return next(inputs)

    monkeypatch.setattr("builtins.input", read_input)
    closed = []
    shutdown = agent.shutdown

    async def chat(_prompt):
        agent._current_task = asyncio.create_task(asyncio.sleep(60))
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        with pytest.raises(asyncio.CancelledError):
            await agent._current_task
        agent._current_task = None

    async def close():
        await shutdown()
        closed.append(True)

    monkeypatch.setattr(agent, "chat", chat)
    monkeypatch.setattr(agent, "shutdown", close)
    if idle:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(cli.run_cli(agent, None))
    else:
        asyncio.run(cli.run_cli(agent, None))
        assert agent._aborted
    assert closed == [True]
    assert signal.getsignal(signal.SIGINT) == before
