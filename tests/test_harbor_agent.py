from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harbor_agent import SansAgent, EnvironmentTools


class Result:
    def __init__(self, stdout: str = "", stderr: str = "", return_code: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class FakeEnvironment:
    environment_name = "terminal-bench/fake"
    context_id = None
    task_env_config = SimpleNamespace(workdir="/workspace")

    def __init__(self) -> None:
        self.files = {"/workspace/a.py": "old\n"}
        self.calls: list[tuple[str, object]] = []

    async def exec(self, command: str, **kwargs: object) -> Result:
        self.calls.append(("exec", (command, kwargs)))
        if "git diff --cached --binary --full-index" in command:
            self.files["/tmp/sans-final.patch"] = (
                "diff --git a/new.bin b/new.bin\nnew file mode 100644\nGIT binary patch\n"
            )
            return Result()
        if command == "rm -f -- /tmp/sans-final.patch":
            self.files.pop("/tmp/sans-final.patch", None)
            return Result()
        if command.startswith("cat -- "):
            path = command.removeprefix("cat -- ").strip("'")
            if path not in self.files:
                return Result(stderr="missing", return_code=1)
            return Result(stdout=self.files[path])
        if command.startswith("find "):
            return Result(stdout="/workspace/a.py\n")
        if command.startswith("grep "):
            return Result(stdout="/workspace/a.py:1:old\n")
        if "sha256sum" in command:
            return Result(stdout="abc123  -\n")
        if "pytest -q" in command:
            return Result(stdout="1 passed\n")
        if command.startswith("git status --short --branch"):
            return Result(stdout="## main\nabc123 latest\n")
        if command.startswith("if [ -f CLAUDE.md ]"):
            return Result(stdout="Use the repository test command.\n")
        return Result()

    async def is_file(self, path: str, user: object = None) -> bool:
        self.calls.append(("is_file", path))
        return path in self.files

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.calls.append(("upload_file", target_path))
        if target_path.startswith("/tmp/"):
            self.files[target_path] = Path(source_path).read_text(encoding="utf-8")
        else:
            self.files[target_path] = Path(source_path).read_text(encoding="utf-8")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        self.calls.append(("download_file", source_path))
        Path(target_path).write_text(self.files[source_path], encoding="utf-8")


def test_environment_tools_route_all_six_primitives_through_environment() -> None:
    async def exercise() -> tuple[FakeEnvironment, EnvironmentTools]:
        environment = FakeEnvironment()
        tools = EnvironmentTools(environment, "/workspace")

        assert "old" in await tools("read_file", {"file_path": "a.py"})
        environment.files["/workspace/lines.txt"] = "one\ntwo\nthree\nfour\n"
        assert await tools(
            "read_file", {"file_path": "lines.txt", "offset": 1, "limit": 2}
        ) == "   2 | two\n   3 | three"
        assert "Successfully edited" in await tools(
            "edit_file", {"file_path": "a.py", "old_string": "old", "new_string": "new"}
        )
        assert "Successfully wrote" in await tools(
            "write_file", {"file_path": "new.py", "content": "created\n"}
        )
        assert "a.py" in await tools("list_files", {"path": ".", "pattern": "*.py"})
        assert "a.py:1:old" in await tools("grep_search", {"path": ".", "pattern": "old"})
        result = await tools(
            "run_shell", {"command": "pytest -q", "timeout": 12_000}
        )
        assert result.exit_code == 0 and "1 passed" in result.stdout
        assert await tools.repo_state() == "abc123"
        return environment, tools

    environment, _ = asyncio.run(exercise())
    operations = {name for name, _ in environment.calls}
    assert {"exec", "is_file", "upload_file", "download_file"} <= operations
    shell_commands = [value[0] for name, value in environment.calls if name == "exec"]
    assert any(command.startswith("bash -e -o pipefail -c ") for command in shell_commands)
    assert all("/opt/miniconda3/envs/testbed/bin" not in command for command in shell_commands)


@pytest.mark.skipif(os.name == "nt", reason="The benchmark find adapter runs in POSIX containers")
def test_recursive_file_listing_includes_root_and_nested_tests(tmp_path):
    import subprocess

    class LocalEnvironment:
        async def exec(self, command, **kwargs):
            result = subprocess.run(command, shell=True, cwd=tmp_path, capture_output=True, text=True)
            return Result(result.stdout, result.stderr, result.returncode)

    for name in ("test.py", "pkg/test_nested.py", "pkg/other.py", "tests/conftest.py", ".git/test_hidden.py", "node_modules/test_hidden.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# sample\n")
    tools = EnvironmentTools(LocalEnvironment(), str(tmp_path))
    for pattern in ("**/test*.py", "**/**/test*.py"):
        listing = asyncio.run(tools._list_files({"pattern": pattern}))
        assert set(listing.splitlines()) == {"test.py", "pkg/test_nested.py"}


def test_environment_timeout_is_returned_to_agent() -> None:
    class TimeoutEnvironment(FakeEnvironment):
        async def exec(self, command: str, **kwargs: object) -> Result:
            try:
                raise TimeoutError()
            except TimeoutError:
                raise RuntimeError("Command timed out after 30 seconds")

    result = asyncio.run(EnvironmentTools(TimeoutEnvironment(), "/workspace")(
        "run_shell", {"command": "slow-test"}
    ))
    assert result.timed_out and result.exit_code is None
    assert str(result) == "Command timed out: Command timed out after 30 seconds"


def test_timeout_words_without_typed_cause_are_only_an_execution_error():
    class ErrorEnvironment(FakeEnvironment):
        async def exec(self, command, **kwargs):
            raise RuntimeError("not timed out: connection rejected")
    result = asyncio.run(EnvironmentTools(ErrorEnvironment(), "/workspace")("run_shell", {"command": "check"}))
    assert result.failed and not result.timed_out and result.exit_code is None


@pytest.mark.parametrize("xml", ["<testsuite><testcase><failure/></testcase></testsuite>", "not XML"])
def test_environment_collects_fresh_report_and_removes_it(xml):
    class ReportEnvironment(FakeEnvironment):
        async def exec(self, command, **kwargs):
            if command.startswith("bash "):
                tokens = shlex.split(shlex.split(command)[-1])
                self.report = next(token.split("=", 1)[1] for token in tokens if token.startswith("--junitxml="))
                self.files[self.report] = xml
                return Result(stdout="999 passed", return_code=1)
            if command.startswith("rm -f -- "):
                self.files.pop(shlex.split(command)[-1])
                return Result()
            return await super().exec(command, **kwargs)
    environment = ReportEnvironment()
    result = asyncio.run(EnvironmentTools(environment, "/workspace")("run_shell", {"command": "pytest test_case.py", "_pytest_report": True}))
    assert result.exit_code == 1 and result.stdout == "999 passed"
    assert result.test_outcome == ("failed" if xml.startswith("<") else "inconclusive")
    assert environment.report not in environment.files


@pytest.mark.parametrize("command", [
    "false; true",
    "pytest -q; git diff --check",
])
def test_environment_shell_stops_after_an_earlier_command_fails(command: str) -> None:
    class SequentialCommandEnvironment(FakeEnvironment):
        async def exec(self, wrapped: str, **kwargs: object) -> Result:
            self.calls.append(("exec", (wrapped, kwargs)))
            fail_fast = (
                wrapped.startswith("bash -e ")
                or wrapped.startswith("bash -eo ")
                or "set -e" in wrapped
            )
            return Result(
                stdout="later command succeeded\n",
                stderr="earlier command failed\n",
                return_code=1 if fail_fast else 0,
            )

    result = asyncio.run(EnvironmentTools(SequentialCommandEnvironment(), "/workspace")(
        "run_shell", {"command": command}
    ))

    assert result.exit_code == 1 and result.failed


def test_file_search_tools_preserve_no_match_and_real_errors() -> None:
    class SearchEnvironment(FakeEnvironment):
        async def exec(self, command: str, **kwargs: object) -> Result:
            if command.startswith("find "):
                return Result(stderr="find: denied", return_code=1)
            if command.startswith("grep "):
                return Result(stderr="grep: denied", return_code=2)
            return await super().exec(command, **kwargs)

    tools = EnvironmentTools(SearchEnvironment(), "/workspace")
    listing = asyncio.run(tools("list_files", {"path": ".", "pattern": "*.py"}))
    search = asyncio.run(tools("grep_search", {"path": ".", "pattern": "old"}))

    assert listing == "Error listing files (exit code 1): find: denied"
    assert search == "Error searching files (exit code 2): grep: denied"

    class NoMatchEnvironment(FakeEnvironment):
        async def exec(self, command: str, **kwargs: object) -> Result:
            if command.startswith("grep "):
                return Result(return_code=1)
            return await super().exec(command, **kwargs)

    no_match = asyncio.run(EnvironmentTools(NoMatchEnvironment(), "/workspace")(
        "grep_search", {"path": ".", "pattern": "absent"}
    ))
    assert no_match == "No matches found."


@pytest.mark.parametrize("usage_complete", [True, False])
def test_sans_agent_populates_harbor_context_without_leaking_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, usage_complete: bool,
) -> None:
    secret = "host-only-secret"
    seed = tmp_path / "seed" / "repo-workflow"
    seed.mkdir(parents=True)
    (seed / "SKILL.md").write_text("---\nname: repo-workflow\n---\nVerify once.\n")
    monkeypatch.setenv("SANS_BENCHMARK_API_KEY", secret)

    seen: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            assert kwargs["api_key"] == secret
            assert secret not in os.environ.values()
            assert (Path.home() / ".sans" / "skills" / "repo-workflow" / "SKILL.md").is_file()
            assert callable(kwargs["tool_executor"])
            assert callable(kwargs["repo_state_probe"])
            assert "git add -A -- ." in kwargs["diff_check_command"]
            assert "git diff --cached --check HEAD --" in kwargs["diff_check_command"]
            assert {tool["name"] for tool in kwargs["custom_tools"]} == {
                "read_file", "write_file", "edit_file", "list_files", "grep_search",
                "run_shell", "skill", "update_work_state", "read_observation",
            }
            assert "repository-task solver" in kwargs["custom_system_prompt"]
            assert kwargs["repair_task"].canonical_issue == "Fix it"
            assert kwargs["repair_task"].repo_root == "/workspace"
            assert "system_prompt_context" not in kwargs
            self.trace = kwargs["trace"]

        async def run_once(self, prompt: str) -> dict[str, object]:
            assert "Fix it" in prompt
            assert "smallest focused reproduction" in prompt
            assert "stop hook owns the final diff" in prompt
            assert "hard limit of 40 model calls and 40 tool calls" in prompt
            assert "Never issue more than 8 tool calls in one response" in prompt
            assert '"cwd": "/workspace"' in prompt
            assert "## main" in prompt
            assert "Use the repository test command." in prompt
            execution = await seen["tool_executor"](
                "run_shell", {"command": "pytest -q"}
            )
            assert execution.exit_code == 0 and "1 passed" in execution.stdout
            if not usage_complete:
                with self.trace.span("model.call", component="model", purpose="main") as span:
                    span.finish(status="error")
            with self.trace.span("model.call", component="model", purpose="main") as span:
                span.finish(metrics={"provider_usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_input_tokens": 60,
                    "cache_miss_tokens": 40,
                    "cache_creation_tokens": None,
                    "reasoning_tokens": 5,
                    "total_tokens": 120,
                    "provider_reported_cost": None,
                }})
            return {
                "text": "done",
                "tokens": {"input": 100, "output": 20},
                "model_calls": 1,
                "termination_reason": {"kind": "max_tool_calls"},
            }

        async def shutdown(self) -> None:
            assert secret not in os.environ.values()

    monkeypatch.setattr("agents.agent.Agent", FakeAgent)
    logs = tmp_path / "logs"
    bridge = SansAgent(
        logs_dir=logs,
        model_name="DeepSeek-V4-Flash",
        api_base="https://developer.amd.com.cn/radeon/api/v1",
        provider_revision="amd-test",
        thinking="true",
        skills_seed=str(seed.parent),
        online_evolution="false",
        max_turns="20",
        max_model_calls="40",
        max_tool_calls="40",
        max_input_tokens="32000",
        max_output_tokens="4096",
        model_call_timeout_seconds="180",
        cache_hit_price="0.0028",
        cache_miss_price="0.14",
        output_price="0.28",
    )
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        cost_usd=None,
        metadata=None,
    )
    environment = FakeEnvironment()
    environment.files["/opt/miniconda3/envs/testbed/bin/python"] = ""

    asyncio.run(bridge.setup(environment))
    asyncio.run(bridge.run("Fix it", environment, context))

    assert seen["thinking"] is True
    assert seen["max_turns"] == 20
    assert seen["max_model_calls"] == 40
    assert seen["max_tool_calls"] == 40
    assert seen["model_call_timeout_seconds"] == 180
    assert context.n_input_tokens == 100
    assert context.n_cache_tokens == 60
    assert context.n_output_tokens == 20
    assert context.cost_usd == (pytest.approx(0.000011368) if usage_complete else None)
    assert context.metadata["provider_usage"]["cache_miss_tokens"] == 40
    assert context.metadata["provider_usage"]["reasoning_tokens"] == 5
    assert context.metadata["provider_revision"] == "amd-test"
    assert context.metadata["observed_cost_usd"] == pytest.approx(0.000011368)
    assert context.metadata["cost_complete"] is usage_complete
    assert context.metadata["decision_model_calls"] == 1
    assert context.metadata["max_model_calls"] == 40
    assert context.metadata["max_tool_calls"] == 40
    assert context.metadata["termination_reason"] == {"kind": "max_tool_calls"}
    assert context.metadata["final_patch"] == "final.patch"
    assert context.metadata["final_patch_error"] is None
    assert (logs / "trace.jsonl").is_file()
    assert b"GIT binary patch" in (logs / "final.patch").read_bytes()
    shell_commands = [value[0] for name, value in environment.calls if name == "exec"]
    assert any(
        "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH" in command
        for command in shell_commands
    )
    assert any(
        "git add -A -- ." in command and "git diff --cached --binary --full-index" in command
        for command in shell_commands
    )
    assert all(secret not in path.read_text(encoding="utf-8", errors="replace")
               for path in logs.rglob("*") if path.is_file())
    assert os.environ["SANS_BENCHMARK_API_KEY"] == secret


@pytest.mark.parametrize("testbed_python", [False, True])
def test_setup_rejects_missing_pytest_in_the_actual_task_interpreter(tmp_path: Path, testbed_python: bool) -> None:
    class MissingPytest(FakeEnvironment):
        async def exec(self, command: str, **kwargs: object) -> Result:
            self.calls.append(("exec", (command, kwargs)))
            return Result(stderr="No module named pytest", return_code=1)

    environment = MissingPytest()
    if testbed_python:
        environment.files["/opt/miniconda3/envs/testbed/bin/python"] = ""
    bridge = SansAgent(logs_dir=tmp_path, model_name="model", api_base="https://unused.invalid/v1")
    with pytest.raises(RuntimeError, match="before starting model calls"):
        asyncio.run(bridge.setup(environment))
    assert bridge._workdir is None
    python = "/opt/miniconda3/envs/testbed/bin/python" if testbed_python else "python"
    assert environment.calls[-1] == ("exec", (f"{python} -m pytest --version", {"cwd": "/workspace", "timeout_sec": 30}))


def test_sans_agent_defaults_are_uncapped(tmp_path: Path) -> None:
    bridge = SansAgent(
        logs_dir=tmp_path,
        model_name="model",
        api_base="https://example.invalid/v1",
    )
    assert bridge.max_model_calls is None
    assert bridge.max_tool_calls is None
    assert bridge.version() == "0.1.3"


def test_final_patch_export_failure_is_an_infrastructure_error(tmp_path: Path) -> None:
    class BrokenExport(FakeEnvironment):
        async def exec(self, command: str, **kwargs: object) -> Result:
            if "git diff --cached --binary --full-index" in command:
                return Result(stderr="export failed", return_code=1)
            return await super().exec(command, **kwargs)

    bridge = SansAgent(
        logs_dir=tmp_path, model_name="model", api_base="https://example.invalid/v1",
    )
    bridge._workdir = "/workspace"
    with pytest.raises(RuntimeError, match="export failed"):
        asyncio.run(bridge._save_final_patch(BrokenExport()))
    assert (tmp_path / "final.patch").read_bytes() == b""


def test_sans_agent_requires_key_from_host_environment_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SANS_BENCHMARK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="must not be supplied through agent config"):
        SansAgent(
            logs_dir=tmp_path,
            model_name="model",
            api_base="https://example.invalid/v1",
            extra_env={"SANS_BENCHMARK_API_KEY": "configured-secret"},
        )

    bridge = SansAgent(
        logs_dir=tmp_path,
        model_name="model",
        api_base="https://example.invalid/v1",
    )
    with pytest.raises(RuntimeError, match="SANS_BENCHMARK_API_KEY"):
        asyncio.run(bridge.run("Do it", FakeEnvironment(), SimpleNamespace()))
