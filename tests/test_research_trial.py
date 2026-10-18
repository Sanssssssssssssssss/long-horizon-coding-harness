"""Admission, immutable oracle and exception receipts for small paid research trials."""
import asyncio
from copy import deepcopy
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarks import research_trial as trial
from agents.tool_result import ShellResult


def graded(feature="passed", regression="passed", optional="skipped"):
    def suite(cases):
        return {"exit_code": int(any(status in {"failed", "error"} for _, status in cases)),
                "report": {"collected": len(cases), "failure": sum(s == "failed" for _, s in cases),
                           "error": sum(s == "error" for _, s in cases), "skipped": sum(s == "skipped" for _, s in cases),
                           "cases": cases}}
    return {"status": "graded", "suites": {"feature": suite([("target", feature)]),
            "regression": suite([("existing", regression), ("optional", optional)])}}


def test_hidden_pass_requires_full_feature_and_preserved_gold_statuses():
    gold = graded()
    calibration = {"gold_cases": trial.pilot.outcomes(gold)}
    assert trial.score(gold, calibration)["resolved"]
    assert not trial.score(graded(regression="failed"), calibration)["resolved"]
    assert not trial.score(graded(feature="skipped"), calibration)["resolved"]
    assert not trial.score(graded(optional="passed"), calibration)["resolved"]
    missing = deepcopy(gold)
    missing["suites"]["regression"]["report"]["cases"] = []
    assert not trial.score(missing, calibration)["resolved"]


def test_collection_error_is_labeled_and_cannot_establish_behavioral_failure():
    gold, baseline = graded(), graded(feature="error")
    result = trial.validate_calibration(baseline, gold, gold)
    assert result["initial_contract_missing"]
    assert result["initial_feature"]["failure"] == 0
    with pytest.raises(AssertionError, match="Gold calibration changed"):
        trial.validate_calibration(baseline, graded(regression="failed"), gold)
    with pytest.raises(AssertionError):
        trial.validate_calibration(graded(feature="error", regression="skipped"), gold, gold)


def test_protocol_rejects_changed_source_and_changed_image(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "ROOT", tmp_path)
    monkeypatch.setattr(trial, "CASES", {"tiny": {"key": "tiny", "split": "heldout"}})
    source, archive = tmp_path / "source.py", tmp_path / "source.zip"
    source.write_text("source")
    archive.write_bytes(b"frozen")
    record = {"name": "tiny", "key": "tiny", "split": "heldout", "image": "image-1",
              "instance_id": "tiny-1", "baseline": "baseline.tar", "gold": "gold.tar", "files": {}}
    monkeypatch.setattr(trial, "case_record", lambda _: deepcopy(record))
    protocol = {"version": 1, "model": trial.MODEL, "api_base": trial.BASE_URL,
                "limits": trial.LIMITS, "arms": list(trial.ARMS), "hidden_feedback_to_solver": False,
                "skills_enabled": False, "solver_profile": "B0", "cases": [deepcopy(record)],
                "source_files": {"source.py": trial.checkpoints.digest(source)},
                "source_archive": "source.zip", "source_archive_sha256": trial.checkpoints.digest(archive)}
    assert trial.validate_protocol(protocol, "tiny", "flat")["image"] == "image-1"
    protocol["cases"][0]["image"] = "image-2"
    with pytest.raises(AssertionError, match="Prepared task identity changed"):
        trial.validate_protocol(protocol, "tiny", "flat")
    protocol["cases"][0]["image"] = "image-1"
    source.write_text("changed")
    with pytest.raises(AssertionError, match="Frozen file changed"):
        trial.validate_protocol(protocol, "tiny", "flat")


def test_admission_error_retains_receipt_without_model_call(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    def reject(*_):
        raise ValueError("frozen inputs changed")
    monkeypatch.setattr(trial, "validate_protocol", reject)
    output = tmp_path / "rejected"
    receipt = asyncio.run(trial.run(protocol, "tiny", "flat", output))
    assert receipt["paid_model_calls"] == 0
    assert receipt["status"] == "incomplete"
    assert json.loads((output / "receipt.json").read_text())["error"]["message"] == "frozen inputs changed"


def test_summary_allowance_does_not_use_solve_decisions():
    class FakeSideQuery:
        async def _side_query_once(self, *args, **kwargs):
            assert kwargs["max_tokens"] <= 2048
            return await self._run_side_query_call(**kwargs)
        async def _run_side_query_call(self, **kwargs):
            async def invoke():
                return kwargs["purpose"]
            return await self._await_model_call(invoke())
        async def _await_model_call(self, call):
            return await call
    class Loop(trial.SummaryLimitMixin, FakeSideQuery):
        _trial_summary_calls = 0
        _trial_solve_attempts = 0
        _trial_call_purpose = None
    async def exercise():
        loop = Loop()
        for purpose in ["session_fold", "session_fold_evidence", "session_fold_plain_fallback"] * 2:
            assert await loop._side_query_once(purpose=purpose, max_tokens=8192) == purpose
        with pytest.raises(trial._BudgetExceeded, match="summary call allowance exhausted"):
            await loop._side_query_once(purpose="session_fold")
        assert await loop._run_side_query_call(purpose="verification") == "verification"
        assert loop._trial_summary_calls == 6
        assert loop._trial_solve_attempts == 1
    asyncio.run(exercise())


def test_trial_tools_register_issue_acceptance_before_shell_execution():
    names = {tool["name"] for tool in trial.get_solver_tool_definitions("B0")}
    assert {"update_work_state", "read_observation", "run_shell"} <= names
    assert "skill" not in names
    executed = []
    async def executor(name, inp):
        executed.append((name, inp))
        return ShellResult(stdout="observed source", exit_code=0)
    async def exercise():
        agent = trial._FoldingLoop(token_budget={"limit": 10000, "charged": 0},
            api_base="https://example.invalid/v1", api_key="test", model="test",
            custom_system_prompt="test", custom_tools=trial.get_solver_tool_definitions("B0"),
            skills_enabled=False, is_sub_agent=True, permission_mode="bypassPermissions",
            tool_executor=executor, repo_state_probe=lambda: "baseline",
            repair_task=trial.TaskEnvelope("tiny", "Implement the specified behavior.", "/testbed"))
        try:
            result = await agent._execute_tool_call("run_shell", {"command": "pwd"})
            assert "register at least one issue-derived acceptance" in result
            assert executed == []
            await agent._execute_tool_call("update_work_state", {"acceptance_cases": [{
                "id": "required", "behavior": "Implement the specified behavior.",
                "source_quote": "Implement the specified behavior.", "source_kind": "issue"}]})
            result = await agent._execute_tool_call("run_shell", {"command": "pwd"})
            assert "observed source" in result and len(executed) == 1
        finally:
            await agent.shutdown()
    asyncio.run(exercise())


def test_source_refreeze_reuses_only_intact_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "ROOT", tmp_path)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents/agent.py").write_text("new_source = True\n")
    prepared = tmp_path / "experiment/prepared"
    prepared.mkdir(parents=True)
    oracle, archive = prepared / "oracle.json", prepared / "source.zip"
    oracle.write_text("verified calibration")
    archive.write_bytes(b"old frozen source")
    record = {"name": "tiny", "image": "sealed-image", "instance_id": "tiny-1",
              "files": {"experiment/prepared/oracle.json": trial.checkpoints.digest(oracle)}}
    monkeypatch.setattr(trial, "case_record", lambda _: deepcopy(record))
    protocol = prepared / "protocol.json"
    protocol.write_text(json.dumps({"limits": trial.LIMITS, "model": trial.MODEL, "cases": [record],
        "source_archive": "experiment/prepared/source.zip", "source_archive_sha256": trial.checkpoints.digest(archive)}))
    trial.freeze_prepared(protocol, tmp_path / "experiment/refrozen")
    frozen = json.loads((tmp_path / "experiment/refrozen/protocol.json").read_text())
    assert frozen["reused_calibration_sha256"] == trial.checkpoints.digest(protocol)
    assert frozen["source_files"]["agents/agent.py"] == trial.checkpoints.digest(tmp_path / "agents/agent.py")
    oracle.write_text("changed environment result")
    with pytest.raises(AssertionError, match="Prepared environment changed"):
        trial.freeze_prepared(protocol, tmp_path / "experiment/rejected")


@pytest.fixture
def owned_keeper(monkeypatch):
    """Replace only the WSL launch with an owned native Python subprocess."""
    actual_popen = subprocess.Popen
    children = []
    def install(code=None):
        def launch(command, **kwargs):
            child = actual_popen([sys.executable, "-I", "-B", "-c", code or command[-1]],
                stdin=kwargs["stdin"], stdout=kwargs["stdout"], stderr=kwargs["stderr"])
            children.append(child)
            return child
        monkeypatch.setattr(trial, "os", SimpleNamespace(name="nt"))
        monkeypatch.setattr(trial.subprocess, "CREATE_NO_WINDOW", 0, raising=False)
        monkeypatch.setattr(trial.subprocess, "Popen", launch)
        def forbidden(*args, **kwargs):
            raise AssertionError("transport must not launch a separate WSL kill command")
        monkeypatch.setattr(trial.subprocess, "run", forbidden)
        return children
    yield install
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        for stream in (child.stdin, child.stdout):
            if stream and not stream.closed:
                stream.close()


@pytest.mark.parametrize("body_error", [False, True])
def test_transport_valid_pid_restores_backend_and_reaps_owned_client(owned_keeper, body_error):
    children = owned_keeper()
    previous = trial.checkpoints.docker
    try:
        with trial.transport(startup_timeout=1):
            assert trial.checkpoints.docker is trial.windows_docker
            assert children[0].poll() is None
            if body_error:
                raise LookupError("body failed")
    except LookupError as error:
        assert body_error and str(error) == "body failed"
    assert trial.checkpoints.docker is previous
    assert children[0].poll() == 0
    assert children[0].stdin.closed and children[0].stdout.closed


@pytest.mark.parametrize("response", ["", "not-a-pid", "0", "-1", "2147483648", "1" * 40])
def test_transport_eof_and_invalid_pid_fail_before_yield_and_clean_client(owned_keeper, response):
    code = ("pass" if not response else
            f"import sys; print({response!r},flush=True); sys.stdin.read()")
    children = owned_keeper(code)
    previous = trial.checkpoints.docker
    with pytest.raises(RuntimeError, match="valid PID"):
        with trial.transport(startup_timeout=1):
            pytest.fail("invalid keeper handshake reached the trial")
    assert trial.checkpoints.docker is previous
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_transport_silent_keeper_has_startup_deadline(owned_keeper):
    children = owned_keeper("import sys; sys.stdin.read()")
    previous = trial.checkpoints.docker
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        with trial.transport(startup_timeout=0.15):
            pytest.fail("silent keeper reached the trial")
    assert caught.value.timeout == 0.15
    assert time.monotonic() - started < 3
    assert trial.checkpoints.docker is previous
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_transport_stdin_close_error_still_reaps_owned_client(owned_keeper):
    children = owned_keeper()
    previous = trial.checkpoints.docker
    with trial.transport(startup_timeout=1):
        stream = children[0].stdin
        class CloseError:
            @property
            def closed(self):
                return stream.closed
            def close(self):
                stream.close()
                raise OSError("pipe close failed")
        children[0].stdin = CloseError()
    assert trial.checkpoints.docker is previous
    assert children[0].poll() == 0
    assert children[0].stdin.closed and children[0].stdout.closed


def test_transport_unresponsive_client_is_stopped_without_linux_kill(owned_keeper):
    children = owned_keeper("import time; time.sleep(30)")
    previous = trial.checkpoints.docker
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        with trial.transport(startup_timeout=0.1):
            pytest.fail("unresponsive keeper reached the trial")
    assert time.monotonic() - started < 9
    assert trial.checkpoints.docker is previous
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_transport_spawn_failure_restores_backend(monkeypatch):
    previous = trial.checkpoints.docker
    monkeypatch.setattr(trial, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(trial.subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    def reject(*args, **kwargs):
        raise OSError("client unavailable")
    monkeypatch.setattr(trial.subprocess, "Popen", reject)
    with pytest.raises(OSError, match="client unavailable"):
        with trial.transport():
            pytest.fail("spawn failure reached the trial")
    assert trial.checkpoints.docker is previous


def test_transport_non_windows_does_not_start_client(monkeypatch):
    previous = trial.checkpoints.docker
    monkeypatch.setattr(trial, "os", SimpleNamespace(name="posix"))
    def reject(*args, **kwargs):
        pytest.fail("non-Windows transport launched a client")
    monkeypatch.setattr(trial.subprocess, "Popen", reject)
    with trial.transport():
        assert trial.checkpoints.docker is previous
    assert trial.checkpoints.docker is previous
