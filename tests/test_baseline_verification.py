import asyncio
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import pytest

from agents.agent import Agent
from agents.repair_state import EvidenceReceipt, TaskEnvelope
from agents.telemetry import repo_state_hash, TraceRecorder
from agents.tools import _run_shell
from agents.tool_result import ShellResult
from agents.verification import CHECK_FINGERPRINT_SCRIPT, compare_baseline, pytest_selection, python_command


def command(*args):
    argv = [sys.executable, "-m", "pytest", *args, "-q"]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def fixture_repo(root):
    (root / "tests").mkdir()
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "core.py").write_text("def answer():\n    return 0\n")
    (root / "tests/test_core.py").write_text(
        "from core import answer\n"
        "def test_issue():\n    assert answer() == 1\n"
        "def test_environment():\n    assert False, 'existing environment failure'\n"
        "def test_sibling():\n    assert answer() < 2\n"
    )
    for args in [("init", "-q"), ("add", "."), ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "base")]:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


@pytest.mark.parametrize("prefix", ["python", "python -B", "/usr/bin/python3.12 -B",
    "python -B -X utf8", "python -Xutf8", "python -X utf8 -B"])
def test_pytest_selection_preserves_interpreter_argv(prefix):
    text = f"cd /app/repo && {prefix} -m pytest -q tests/test_core.py::test_issue"
    argv, paths = pytest_selection(text, "/app/repo")
    assert argv == shlex.split(text.split("&&", 1)[1])
    assert paths == ["tests/test_core.py"]


@pytest.mark.parametrize("text", [
    "python -I -m pytest test_core.py -q",
    "python -B -B -m pytest test_core.py -q",
    "python -X dev -m pytest test_core.py -q",
    "python -X -m pytest test_core.py -q",
    "python -X utf8 -Xutf8 -m pytest test_core.py -q",
    "python -B -m pytest test_core.py -k issue",
    "python -B -m pytest tests -q",
    "python -B -m pytest ../test_core.py -q",
    "python -B -m pytest /outside/test_core.py -q",
    "cd /other && python -B -m pytest test_core.py -q",
    "python -B -m pytest test_core.py -q && echo done",
])
def test_bytecode_flag_does_not_relax_test_selection(text):
    assert pytest_selection(text, "/app/repo") is None


@pytest.mark.parametrize("config_args", [("-c", "custom.ini"), ("--config-file=custom.ini",)])
def test_explicit_config_replay_preserves_contract_imports_and_utf8_ids(tmp_path, monkeypatch, config_args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    fixture_repo(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "binding.py").write_text("MARKER = '合同'\n", encoding="utf-8")
    config = tmp_path / "custom.ini"
    config.write_text("[pytest]\npythonpath = source\naddopts = --junitprefix=配置\n"
                      "disable_test_id_escaping_and_forfeit_all_rights_to_community_support = true\n", encoding="utf-8")
    core = tmp_path / "core.py"
    core.write_text("from binding import MARKER\ndef answer():\n    return 0\n", encoding="utf-8")
    test = tmp_path / "tests/test_core.py"
    test.write_text("import pytest\nfrom core import answer, MARKER\n"
                    "@pytest.mark.parametrize('label', ['合同'], ids=['边界'])\n"
                    "def test_issue(label):\n    assert MARKER == label\n    assert answer() == 1\n"
                    "def test_config_contract(pytestconfig):\n"
                    "    assert pytestconfig.inipath.name == 'custom.ini'\n"
                    "    assert pytestconfig.getoption('junitprefix') == '配置'\n", encoding="utf-8")
    for args in [("add", "."), ("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                              "-c", "commit.gpgsign=false", "commit", "-qm", "explicit config contract")]:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    base = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    frozen = {path: path.read_bytes() for path in (config, test, source / "binding.py")}
    core.write_text("from binding import MARKER\ndef answer():\n    return 1\n", encoding="utf-8")
    selected_command = command("tests/test_core.py", *config_args)
    selected = pytest_selection(selected_command, str(tmp_path))
    assert selected is not None
    argv, paths = selected
    assert argv == shlex.split(selected_command.replace("\\", "/")) and paths == ["tests/test_core.py"]
    fingerprint_command = python_command(CHECK_FINGERPRINT_SCRIPT,
        [json.dumps([str(tmp_path), paths, argv])], str(tmp_path), sys.executable)

    async def execute(name, inp):
        return _run_shell(inp)

    pair = asyncio.run(compare_baseline(execute, selected_command, str(tmp_path), base, 30_000))
    assert pair.comparison_error is None, str(pair)
    assert pair.test_outcome == "passed" and pair.baseline.test_outcome == "failed"
    assert pair.pytest_report == pair.comparison_candidate.pytest_report
    assert pair.pytest_report.collected == pair.baseline.pytest_report.collected == 2
    assert any("配置" in identity and "边界" in identity for identity, _, _ in pair.pytest_report.cases)
    assert all(path.read_bytes() == raw for path, raw in frozen.items())
    assert "return 1" in core.read_text(encoding="utf-8")
    before = _run_shell({"command": fingerprint_command})
    assert not before.failed and len(before.stdout.strip()) == 64
    config.write_text(config.read_text(encoding="utf-8") + "# changed condition\n", encoding="utf-8")
    after = _run_shell({"command": fingerprint_command})
    assert not after.failed and before.stdout.strip() != after.stdout.strip()
    refused = asyncio.run(compare_baseline(execute, selected_command, str(tmp_path), base, 30_000))
    assert "unchanged test configuration/fixtures" in refused.comparison_error


@pytest.mark.parametrize("options", ["-c", "--config-file=", "-c ../outside.ini", "-c /outside.ini",
    "--config-file=C:/outside.ini", "-c custom.ini -c other.ini", "-c custom.ini -k issue",
    "-c custom.ini -m slow", "-c custom.ini -o addopts=-q", "-c custom.ini --rootdir=."])
def test_explicit_config_keeps_relative_and_selector_boundaries(options):
    assert pytest_selection(f"python -m pytest tests/test_core.py {options}", "/remote/repository") is None


def test_explicit_config_rejects_missing_and_symlink_escape_in_executor(tmp_path, monkeypatch):
    from agents.verification import PAIR_SCRIPT
    root = tmp_path / "repo"
    root.mkdir()
    base = fixture_repo(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "custom.ini").write_text("[pytest]\n", encoding="utf-8")
    link = root / "config-link"
    if os.name == "nt":
        linked = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/c", "mklink", "/J", str(link), str(outside)],
                                capture_output=True, timeout=10)
        assert linked.returncode == 0, linked.stderr
    else:
        link.symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    for config in ("missing.ini", "config-link/custom.ini"):
        argv, paths = pytest_selection(command("tests/test_core.py", "-c", config), str(root))
        fingerprint = _run_shell({"command": python_command(CHECK_FINGERPRINT_SCRIPT,
            [json.dumps([str(root), paths, argv])], str(root), sys.executable)})
        assert fingerprint.failed
        pair = _run_shell({"command": python_command(PAIR_SCRIPT,
            [json.dumps([str(root), base, argv, paths, 30])], str(root), sys.executable)})
        assert pair.failed and "explicit pytest configuration" in str(pair)
    assert (outside / "custom.ini").read_text(encoding="utf-8") == "[pytest]\n"


def test_ignored_explicit_config_cannot_be_carried_to_original_revision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    base = fixture_repo(tmp_path)
    (tmp_path / ".git/info/exclude").write_text("ignored.ini\n", encoding="utf-8")
    config = tmp_path / "ignored.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    assert not subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "ignored.ini"])
    core = tmp_path / "core.py"
    core.write_text("def answer():\n    return 1\n", encoding="utf-8")
    async def execute(name, inp):
        return _run_shell(inp)
    pair = asyncio.run(compare_baseline(execute, command("tests/test_core.py::test_issue", "-c", "ignored.ini"),
                                       str(tmp_path), base, 30_000))
    assert pair.test_outcome == "passed"
    assert pair.comparison_error is not None and "configuration missing from original revision" in pair.comparison_error
    assert pair.baseline is None and pair.comparison_candidate is None
    assert "return 1" in core.read_text(encoding="utf-8")
    assert config.read_text(encoding="utf-8") == "[pytest]\n"


@pytest.mark.parametrize("encoding_flags", [[], ["-X", "utf8"], ["-Xutf8"]])
def test_bytecode_flag_diagnostic_execution_and_baseline_transport(tmp_path, monkeypatch, encoding_flags):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    base = fixture_repo(tmp_path)
    agent = Agent(api_key="unused", api_base="http://unused.invalid", is_sub_agent=True,
                  custom_system_prompt="offline", custom_tools=[], skills_enabled=False,
                  repair_task=TaskEnvelope("small", "fix the answer", str(tmp_path)))
    agent._baseline_ref = base
    agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))
    agent._repair_state.register_acceptance_case("AC1", "fix the answer", "fix the answer")
    probe = tmp_path / "tests/test_bytecode_probe.py"
    probe.write_text("def test_probe():\n    assert True\n")
    agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))

    def selected(path):
        argv = [sys.executable, "-B", *encoding_flags, "-m", "pytest", "-q", path]
        return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)

    async def run():
        try:
            # Same metadata shape as the real quality check: execution and acceptance are separate.
            output = await agent._execute_tool_call("run_shell", {
                "command": selected("tests/test_bytecode_probe.py"), "diagnostic": True, "scope": "focused",
            })
            evidence = agent._repair_state.evidence[-1]
            assert evidence.diagnostic and evidence.exit_code == 0 and evidence.test_pass_observed
            assert "verification metadata missing" in output
            assert all(not item.acceptance_ids for item in agent._repair_state.verifications)
            assert agent._repair_state.completion_decision().unverified_acceptance_ids == ("AC1",)
            rejected = await agent._execute_tool_call_impl("run_shell", {
                "command": selected("tests/test_core.py"), "diagnostic": True,
            })
            assert "Existing tests cannot" in rejected.error
            rejected = await agent._execute_tool_call_impl("run_shell", {
                "command": selected("tests/test_bytecode_probe.py"), "acceptance_ids": ["AC1"],
            })
            assert "cannot be promoted" in rejected.error
            (tmp_path / "core.py").write_text("def answer():\n    return 1\n")
            async def execute(name, inp):
                return _run_shell(inp)
            pair = await compare_baseline(execute, selected("tests/test_core.py::test_issue"), str(tmp_path), base, 30_000)
            assert pair.test_outcome == "passed" and pair.baseline.test_outcome == "failed"
            assert (tmp_path / "core.py").read_text() == "def answer():\n    return 1\n"
        finally:
            await agent.shutdown()
    asyncio.run(run())


def test_rejected_diagnostic_can_retry_without_relabeling_executed_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    base = fixture_repo(tmp_path)
    agent = Agent(api_key="unused", api_base="http://unused.invalid", is_sub_agent=True,
                  custom_system_prompt="offline", custom_tools=[], skills_enabled=False,
                  repair_task=TaskEnvelope("retry", "fix the answer", str(tmp_path)))
    agent._baseline_ref = base
    agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))
    agent._repair_state.register_acceptance_case("AC1", "fix the answer", "fix the answer")
    for name, assertion in (("retry", "True"), ("red", "False"), ("timeout", "True"), ("transport", "True")):
        (tmp_path / f"tests/test_{name}.py").write_text(f"def test_probe():\n    assert {assertion}\n")
    agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))

    def selected(name):
        argv = [sys.executable, "-B", "-m", "pytest", "-q", f"tests/test_{name}.py"]
        return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)

    async def run():
        try:
            inp = {"command": selected("retry"), "diagnostic": True, "scope": "focused"}
            # Reproduce the former parser rejection before any pytest subprocess runs.
            with monkeypatch.context() as legacy:
                legacy.setattr("agents.agent.pytest_selection", lambda *args: None)
                rejected = await agent._execute_tool_call("run_shell", inp)
            original = agent._repair_state.evidence[-1]
            assert "before its first run" in rejected
            assert original.check_fingerprint and original.exit_code is None
            assert original.execution_started is False and not original.diagnostic
            legacy = asdict(original); legacy.pop("execution_started")
            unknown = EvidenceReceipt(**legacy)
            assert unknown.execution_started is None
            # Old snapshots do not establish whether execution started; retain their blocker.
            agent._repair_state._evidence[-1] = unknown
            rejected = await agent._execute_tool_call_impl("run_shell", inp)
            assert "before its first run" in rejected.error
            agent._repair_state._evidence[-1] = original
            await agent._execute_tool_call("run_shell", inp)
            retried = agent._repair_state.evidence[-1]
            assert retried.diagnostic and retried.exit_code == 0 and retried.test_pass_observed
            assert retried.execution_started is True
            assert agent._repair_state.evidence[-2] == original
            assert all(not item.acceptance_ids for item in agent._repair_state.verifications)

            await agent._execute_tool_call("run_shell", {"command": selected("red")})
            red = agent._repair_state.evidence[-1]
            assert red.exit_code == 1 and red.test_failure_observed and not red.diagnostic
            rejected = await agent._execute_tool_call_impl("run_shell", {
                "command": selected("red"), "diagnostic": True,
            })
            assert "before its first run" in rejected.error

            # Typed backend timeout: no exit code, but the process did execute.
            async def timed_executor(name, inp):
                return ShellResult(timed_out=True, error="backend timeout", execution_started=False) if inp.get("_pytest_report") else _run_shell(inp)
            agent._tool_executor = timed_executor
            await agent._execute_tool_call("run_shell", {"command": selected("timeout")})
            timeout = agent._repair_state.evidence[-1]
            assert timeout.execution_started is True and timeout.exit_code is None and not timeout.diagnostic
            saved = json.loads(json.dumps(agent._repair_state.snapshot()))["evidence"][-1]
            assert EvidenceReceipt(**saved).execution_started is True
            with pytest.raises(ValueError, match="execution_started"):
                replace(timeout, execution_started=1)
            with pytest.raises(ValueError, match="execution_started"):
                ShellResult(execution_started=1)
            rejected = await agent._execute_tool_call_impl("run_shell", {
                "command": selected("timeout"), "diagnostic": True,
            })
            assert "before its first run" in rejected.error
            assert red in agent._repair_state.evidence and timeout in agent._repair_state.evidence

            from benchmarks.harbor_agent import EnvironmentTools
            class LostTransport:
                calls = 0
                async def exec(self, *args, **kwargs):
                    self.calls += 1
                    raise ConnectionResetError("mock transport lost after remote dispatch")
            backend = LostTransport()
            result = await EnvironmentTools(backend, str(tmp_path))("run_shell", {"command": selected("transport")})
            assert backend.calls == 1 and result.execution_started is None
            assert result.exit_code is None and not result.timed_out
            current = agent._repair_state.current_repo_state
            agent._record_repair_evidence("run_shell", {"command": selected("transport")}, result,
                failed=True, duration_ms=0, before=current, after=current, check_fingerprint="mock-source-bound")
            transport = agent._repair_state.evidence[-1]
            assert transport.execution_started is None and not transport.diagnostic
            rejected = await agent._execute_tool_call_impl("run_shell", {
                "command": selected("transport"), "diagnostic": True,
            })
            assert "before its first run" in rejected.error
        finally:
            await agent.shutdown()
    asyncio.run(run())


@pytest.mark.parametrize("traced", [False, True])
def test_real_completion_preserves_old_red_retires_probe_and_blocks_new_regression(tmp_path, monkeypatch, traced):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    base = fixture_repo(tmp_path)
    agent = Agent(api_key="unused", api_base="http://unused.invalid", is_sub_agent=True,
                  custom_system_prompt="offline", custom_tools=[], skills_enabled=False,
                  trace=TraceRecorder(tmp_path.parent / (tmp_path.name + "-trace"), run_id="tiny", task_id="probe", attempt_id="1") if traced else None,
                  repair_task=TaskEnvelope("small", "fix the answer", str(tmp_path)))
    agent._baseline_ref = base
    agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))
    agent._repair_state.register_acceptance_case("AC1", "fix the answer", "fix the answer")
    labels = dict(acceptance_ids=["AC1"], scope="focused", oracle="issue_reproduction")
    async def run():
        rejected = await agent._execute_tool_call_impl("run_shell", {"command": command("tests/test_core.py"), "diagnostic": True})
        assert "Existing tests cannot" in rejected.error
        await agent._execute_tool_call("run_shell", {"command": command("tests/test_core.py::test_issue"), **labels})
        (tmp_path / "tests/test_probe.py").write_text("def test_probe():\n    assert False, 'diagnostic'\n")
        agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))
        await agent._execute_tool_call("run_shell", {"command": command("tests/test_probe.py"), "diagnostic": True})
        probe = agent._repair_state.evidence[-1]
        assert probe.diagnostic and probe.status == "error"
        # Real trajectory: a probe is edited, rerun with different flags, then removed.
        (tmp_path / "tests/test_probe.py").write_text("def test_probe():\n    assert False, 'revised diagnostic'\n")
        agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))
        await agent._execute_tool_call("run_shell", {"command": command("tests/test_probe.py")})
        revised_probe = agent._repair_state.evidence[-1]
        assert revised_probe.diagnostic and revised_probe.check_fingerprint != probe.check_fingerprint
        await agent._execute_tool_call("run_shell", {"command": command("tests/test_probe.py", "-s")})
        assert agent._repair_state.evidence[-1].diagnostic
        rejected = await agent._execute_tool_call_impl("run_shell", {"command": command("tests/test_probe.py"), **labels})
        assert "cannot be promoted" in rejected.error
        rejected = await agent._execute_tool_call_impl("run_shell", {"command": command("tests/test_probe.py", "-k", "test_probe"), **labels})
        assert "cannot be promoted" in rejected.error
        await agent._execute_tool_call("run_shell", {"command": python_command(
            "from pathlib import Path; Path('tests/test_probe.py').unlink()", [], str(tmp_path), sys.executable,
        )})
        # Deletion must clear the stale blocker before the next model request, not only at stop.
        assert revised_probe.id in agent._repair_state.snapshot()["retired_diagnostics"]
        assert "diagnostic_cleanup" not in agent._repair_state.render_runtime_context()
        (tmp_path / "core.py").write_text("def answer():\n    return 1\n")
        agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))
        await agent._execute_tool_call("run_shell", {"command": command("tests/test_core.py::test_issue"), **labels})
        await agent._execute_tool_call("run_shell", {"command": command("tests/test_core.py"), **labels})
        assert not agent._repair_state.completion_decision().complete
        reason = await agent._run_stop_hooks()
        assert reason is None, [(e.input_summary, e.output_preview) for e in agent._repair_state.evidence[-2:]]
        comparisons = [e for e in agent._repair_state.evidence if e.unchanged_baseline_failure]
        assert len(comparisons) == 1 and comparisons[0].status == "error" and comparisons[0].exit_code == 1
        assert probe in agent._repair_state.evidence
        assert probe.id in agent._repair_state.snapshot()["retired_diagnostics"]
        assert revised_probe.id in agent._repair_state.snapshot()["retired_diagnostics"]

        # A newly introduced sibling failure must stay blocking.
        (tmp_path / "core.py").write_text("def answer():\n    return 3\n")
        async def execute(name, inp):
            return _run_shell(inp)
        pair = await compare_baseline(execute, command("tests/test_core.py"), str(tmp_path), base, 30_000)
        assert pair.baseline and pair.pytest_report.failure == pair.baseline.pytest_report.failure + 1
        assert not pair.unchanged_failure
        assert "Original revision replay" in str(pair) and "existing environment failure" in str(pair)
        assert "return 3" in (tmp_path / "core.py").read_text()
        (tmp_path / "tests/test_late_probe.py").write_text("def test_late():\n    assert False\n")
        agent._repair_state.observe_repo_state(repo_state_hash(tmp_path))
        await agent._execute_tool_call("run_shell", {"command": command("tests/test_late_probe.py")})
        rejected = await agent._execute_tool_call_impl("run_shell", {
            "command": command("tests/test_late_probe.py", "-s"), "diagnostic": True,
        })
        assert "before its first run" in rejected.error
        await agent.shutdown()
    asyncio.run(run())


def test_comparison_requires_same_failures_and_complete_collection():
    from agents.tool_result import PytestReport, ShellResult
    base = PytestReport(2, failure=1, cases=(("a", "failure", "original"), ("b", "passed", "")))
    for current in (
        PytestReport(2, failure=1, cases=(("a", "passed", ""), ("b", "failure", "new"))),
        PytestReport(2, failure=1, cases=(("a", "failure", "different"), ("b", "passed", ""))),
        PytestReport(1, failure=1, cases=(("a", "failure", "original"),)),
    ):
        assert not ShellResult(exit_code=1, pytest_report=current, baseline=ShellResult(exit_code=1, pytest_report=base), baseline_ref="base").unchanged_failure


@pytest.mark.parametrize("bare_python", [False, True])
def test_baseline_replay_preserves_interpreter_dependencies_and_test_identities(tmp_path, monkeypatch, bare_python):
    parent = tmp_path / "suite"
    root = parent / "repository"
    root.mkdir(parents=True)
    fixture_repo(root)
    (root / "pytest.ini").unlink()
    (parent / "pytest.ini").write_text("[pytest]\n")
    source_path = root / "source"
    source_path.mkdir()
    (source_path / "source_dependency.py").write_text("value = 0\n")
    for args in [("add", "."), ("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                 "-c", "commit.gpgsign=false", "commit", "-qm", "ancestor pytest config")]:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    base = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dependency = tmp_path / "dependencies"
    dependency.mkdir()
    (dependency / "external_dependency.py").write_text("value = 7\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(dependency), str(source_path))))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--junitprefix=original")
    monkeypatch.setenv("PYTHONUTF8", "0")
    (root / "tests/test_core.py").write_text(
        "import sys, pytest\nfrom external_dependency import value\nfrom source_dependency import value as source_value\nfrom core import answer\n"
        "def test_issue():\n    assert answer() == 1\n"
        f"def test_environment():\n    assert sys.executable == {sys.executable!r}\n    assert value == 7\n"
        "def test_source_dependency():\n    assert source_value == 1\n"
        "def test_sibling():\n    assert answer() < 2\n"
        "@pytest.mark.parametrize('tag', ['中文'])\ndef test_unicode_中文_id(tag):\n    assert tag == '中文'\n",
        encoding="utf-8",
    )
    argv = ["python" if bare_python else sys.executable, "-B", "-m", "pytest", "tests/test_core.py", "-q"]
    selected = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    observed = []

    async def execute(name, inp):
        result = _run_shell(inp)
        if inp.get("_pytest_report"):
            observed.append(result)
        return result

    async def run():
        (root / "core.py").write_text("def answer():\n    return 1\n")
        (source_path / "source_dependency.py").write_text("value = 1\n")
        pair = await compare_baseline(execute, selected, str(root), base, 30_000)
        (tmp_path / "comparison.json").write_text(json.dumps(asdict(pair), indent=2), encoding="utf-8")
        assert pair.test_outcome == "passed"
        assert pair.comparison_error is None, (pair.pytest_report, pair.comparison_candidate.pytest_report, str(pair))
        assert pair.comparison_candidate.pytest_report == observed[-1].pytest_report
        assert pair.baseline.test_outcome == "failed" and pair.baseline.pytest_report.failure == 2
        assert all(case[0].startswith("original.repository.tests.test_core::") for case in pair.pytest_report.cases)
        assert any("中文" in case[0] for case in pair.pytest_report.cases)
        (root / "core.py").write_text("def answer():\n    return 3\n")
        (source_path / "source_dependency.py").write_text("value = 2\n")
        regression = await compare_baseline(execute, selected, str(root), base, 30_000)
        assert regression.comparison_error is None
        assert regression.pytest_report.failure == regression.baseline.pytest_report.failure + 1
        assert not regression.unchanged_failure
        assert (root / "core.py").read_text() == "def answer():\n    return 3\n"
    asyncio.run(run())


def test_remote_fingerprint_tracks_test_conditions_and_preserves_product_edits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_core.py").write_text("def test_core(answer):\n    assert answer == 1\n")
    (tmp_path / "core.py").write_text("def answer(): return 0\n")
    paths = ["tests/test_core.py"]

    def fingerprint(*args):
        payload = json.dumps([str(tmp_path), paths, [*paths, *args]])
        result = _run_shell({"command": python_command(CHECK_FINGERPRINT_SCRIPT, [payload], str(tmp_path), sys.executable)})
        assert not result.failed, str(result)
        value = result.stdout.strip()
        assert len(value) == 64 and int(value, 16) >= 0
        return value

    original = fingerprint()
    (tmp_path / "core.py").write_text("def answer(): return 1\n")
    (tmp_path / "tests/test_extra.py").write_text("def test_extra(): pass\n")
    assert fingerprint() == original
    for name in ("conftest.py", "tests/conftest.py", "pytest.ini", "tests/.pytest.ini", "pyproject.toml", "pytest.toml", ".pytest.toml", "setup.cfg", "tox.ini"):
        path = tmp_path / name
        path.write_text("original\n")
        added = fingerprint()
        assert added != original, name
        path.write_text("changed\n")
        assert fingerprint() != added, name
        path.unlink()
        assert fingerprint() == original, name
    custom = tmp_path / "custom.ini"
    custom.write_text("[pytest]\n")
    explicit = fingerprint("-c", "custom.ini", "--rootdir=tests", "--confcutdir", "tests")
    custom.write_text("[pytest]\naddopts = -q\n")
    assert fingerprint("-c", "custom.ini", "--rootdir=tests", "--confcutdir", "tests") != explicit


def test_remote_fingerprint_refuses_missing_and_external_test_conditions(tmp_path):
    (tmp_path / "test_core.py").write_text("def test_core(): pass\n")
    for paths, args in ((["missing.py"], []), (["test_core.py"], ["-c", "missing.ini"]),
                        (["test_core.py"], ["--rootdir", ".."]), (["test_core.py"], ["--confcutdir=.."])):
        payload = json.dumps([str(tmp_path), paths, args])
        result = _run_shell({"command": python_command(CHECK_FINGERPRINT_SCRIPT, [payload], str(tmp_path), sys.executable)})
        assert result.failed and not result.stdout.strip()
