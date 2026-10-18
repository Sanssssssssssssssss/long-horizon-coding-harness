"""Small frozen FeatureBench trials: model-free prepare, paid solve, isolated grading."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Thread
from types import SimpleNamespace
import zipfile

from agents.prompt import SOLVER_SYSTEM_PROMPT
from agents.agent import _BudgetExceeded
from agents.repair_state import TaskEnvelope
from agents.telemetry import TraceRecorder, read_trace, summarize_trace
from agents.tools import get_solver_tool_definitions
from benchmarks import checkpoints, context_pilot as pilot
from benchmarks.context_folding import _FoldingLoop
from benchmarks.harbor_agent import EnvironmentTools, _FULL_DIFF_CHECK, _isolated_home

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek/deepseek-v4.1-flash"
BASE_URL = "https://api.commandcode.ai/provider/v1"
ARMS = ("flat", "current_fold", "evidence_fold")
CASES = {
    "pydantic_arguments_schema": {
        "key": "test_experimental_arguments_schema", "split": "heldout",
        "research": "experiment/research/20261006-context-pilot",
        "artifacts": "experiment/runs/20261006-context-pilot/test_experimental_arguments_schema",
        "scope": "Unmodified FeatureBench task; reconstructed local Python environment, no leaderboard claim.",
        "prior_exposure": "Historical pilot runs exist; held out from current skill distillation, not a blind unseen task.",
    },
    "fastapi_local_scope": {
        "key": "test_compat", "split": "development",
        "research": "experiment/research/20261006-pilot-recovery/scoped-task",
        "artifacts": "experiment/runs/20261006-pilot-recovery/scoped-task/test_compat",
        "scope": "Local scope-corrected diagnostic; explicitly not an official FeatureBench score.",
        "prior_exposure": "Previously solved local diagnostic; development only.",
    },
}
LIMITS = {"solve_calls": 24, "summary_calls": 6, "total_model_calls": 30,
          "fold_every_decisions": 6, "case_tokens": 300000, "max_output_tokens": 8192,
          "checkpoint_summary_tokens": 2048,
          "max_input_tokens": 32768, "max_tool_calls": 120, "request_timeout_seconds": 120}
WSL = ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--exec"]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def artifact_path(value):
    path = Path(value).resolve()
    path.relative_to(ROOT / "experiment")
    return path


def linux_path(value):
    path = Path(value).resolve()
    path.relative_to(ROOT)
    return "/mnt/" + path.drive[0].lower() + str(path)[2:].replace("\\", "/")


def windows_docker(*args, **kwargs):
    return subprocess.run([*WSL, "docker", *map(str, args)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=90, check=True, **kwargs)


class WindowsWSLWorkspace(checkpoints.DockerWorkspace):
    # shortcut: driver-local WSL transport; move into checkpoints when a second live runner needs it.
    async def exec(self, command, cwd="/testbed", timeout_sec=30):
        result = await asyncio.to_thread(subprocess.run,
            [*WSL, "docker", "exec", "-w", cwd, self.name, "timeout", "-k", "2", str(timeout_sec), "bash", "-c", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_sec + 35)
        return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    async def upload_file(self, source, target):
        await asyncio.to_thread(windows_docker, "cp", linux_path(source), f"{self.name}:{target}")

    async def download_file(self, source, target):
        await asyncio.to_thread(windows_docker, "cp", f"{self.name}:{source}", linux_path(target))

    def snapshot(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        raw, normalized = directory / "capture.partial", directory / "canonical.partial"
        windows_docker("pause", self.name)
        try:
            with raw.open("xb") as stream:
                subprocess.run([*WSL, "docker", "cp", f"{self.name}:{self.root}/.", "-"],
                               stdout=stream, stderr=subprocess.PIPE, timeout=90, check=True)
        finally:
            windows_docker("unpause", self.name)
        try:
            checkpoints.canonical_tar(raw, normalized)
            target = directory / (checkpoints.digest(normalized) + ".tar")
            if target.exists():
                assert checkpoints.digest(target) == target.stem
            else:
                normalized.rename(target)
            return target
        finally:
            raw.unlink(missing_ok=True)
            normalized.unlink(missing_ok=True)


@contextmanager
def transport(startup_timeout=20):
    previous = checkpoints.docker
    keeper = None
    reader = None
    try:
        if os.name == "nt":
            keeper = subprocess.Popen([*WSL, "/usr/bin/python3", "-c",
                "import os,sys; print(os.getpid(),flush=True); sys.stdin.read()"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW)
            pid_lines = Queue()
            def read_pid():
                try:
                    line = keeper.stdout.readline(32)
                except OSError:
                    line = b""
                pid_lines.put(line)
            reader = Thread(target=read_pid, daemon=True)
            reader.start()
            try:
                line = pid_lines.get(timeout=startup_timeout)
            except Empty:
                raise subprocess.TimeoutExpired(keeper.args, startup_timeout) from None
            if not line.endswith(b"\n") or not line.strip().isdigit() or not 0 < int(line) < 2**31:
                raise RuntimeError("WSL keeper did not return a valid PID")
            checkpoints.docker = windows_docker
        yield
    finally:
        checkpoints.docker = previous
        if keeper is not None:
            try:
                keeper.stdin.close()
            except OSError:
                pass
            try:
                keeper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # WSL may be stuck; stop only the Windows client created here.
                keeper.terminate()
                try:
                    keeper.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    keeper.kill()
                    keeper.wait(timeout=5)
            finally:
                if reader is not None:
                    reader.join(timeout=1)
                if reader is None or not reader.is_alive():
                    keeper.stdout.close()


@contextmanager
def bind_case(record):
    previous = pilot.OUT, pilot.RESEARCH, pilot.workspace
    workspace = WindowsWSLWorkspace if os.name == "nt" else checkpoints.DockerWorkspace
    pilot.OUT = (ROOT / record["artifacts"]).parent
    pilot.RESEARCH = ROOT / record["research"]
    pilot.workspace = lambda image: workspace(image, memory="4g", cpus=2)
    try:
        yield
    finally:
        pilot.OUT, pilot.RESEARCH, pilot.workspace = previous


def case_record(name):
    record = dict(CASES[name], name=name)
    folder = ROOT / record["artifacts"]
    ready = json.loads((folder / "ready.json").read_text(encoding="utf-8"))
    spec_path = ROOT / record["research"] / "tasks" / (record["key"] + ".json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert checkpoints.digest(spec_path) == ready["task_sha256"]
    assert checkpoints.digest(folder / "checks.tar") == ready["checks_sha256"]
    files = [spec_path, folder / "ready.json", folder / "checks.tar"]
    for mode in ("baseline", "gold"):
        snapshot = folder / "calibration/snapshots" / ready[mode]
        if not snapshot.exists() and name == "fastapi_local_scope" and mode == "gold":
            snapshot = ROOT / "experiment/runs/20261006-context-pilot/test_compat/calibration/snapshots" / ready[mode]
        assert checkpoints.digest(snapshot) == snapshot.stem
        record[mode] = snapshot.relative_to(ROOT).as_posix()
        files.append(snapshot)
    record.update(image=ready["image"], instance_id=spec["instance_id"],
                  files={p.relative_to(ROOT).as_posix(): checkpoints.digest(p) for p in files})
    return record


def validate_calibration(original, gold, historical_gold):
    expected = pilot.outcomes(historical_gold)
    observed = pilot.outcomes(gold)
    assert observed == expected and gold["status"] == "graded", "Gold calibration changed"
    assert all(s["exit_code"] == 0 and s["report"]["collected"] > 0
               and s["report"]["failure"] == s["report"]["error"] == 0 for s in gold["suites"].values())
    assert original["status"] == "graded" and original["suites"]["feature"]["exit_code"] != 0
    regression = {k: v for k, v in expected.items() if k.startswith("regression:")}
    assert regression == {k: v for k, v in pilot.outcomes(original).items() if k.startswith("regression:")}
    assert original["suites"]["regression"]["exit_code"] == 0
    return {"initial_feature": original["suites"]["feature"]["report"],
            "initial_contract_missing": bool(original["suites"]["feature"]["report"]["error"]),
            "gold_cases": expected, "baseline_cases": pilot.outcomes(original)}


def score(receipt, calibration):
    observed = pilot.outcomes(receipt)
    expected = calibration["gold_cases"]
    feature = {k for k, v in expected.items() if k.startswith("feature:") and v == "passed"}
    regression = {k for k, v in expected.items() if k.startswith("regression:") and v == "passed"}
    passed = {k for k, v in observed.items() if v == "passed"}
    valid = receipt["status"] == "graded"
    resolved = valid and observed == expected and all(s["exit_code"] == 0 for s in receipt["suites"].values())
    return {"status": receipt["status"], "resolved": bool(resolved),
            "feature_passed": len(feature & passed), "feature_total": len(feature),
            "regression_passed": len(regression & passed), "regression_total": len(regression),
            "regressions": sorted(regression - passed),
            "skipped": sum(s["report"]["skipped"] for s in receipt["suites"].values() if s["report"])}


async def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with transport():
        for name in CASES:
            record = case_record(name)
            image = json.loads(checkpoints.docker("image", "inspect", record["image"]).stdout)[0]["Id"]
            assert image == record["image"]
            with bind_case(record):
                original = await pilot.evaluate(record["key"], image, ROOT / record["baseline"], output / name / "baseline")
                gold = await pilot.evaluate(record["key"], image, ROOT / record["gold"], output / name / "gold")
            ready = json.loads((ROOT / record["artifacts"] / "ready.json").read_text(encoding="utf-8"))
            record["calibration"] = validate_calibration(original, gold, ready["calibration"]["gold"][0])
            for path in (output / name).rglob("*"):
                if path.is_file():
                    record["files"][path.relative_to(ROOT).as_posix()] = checkpoints.digest(path)
            records.append(record)
    sources = {p.relative_to(ROOT).as_posix(): checkpoints.digest(p)
               for folder in ("agents", "benchmarks") for p in sorted((ROOT / folder).rglob("*.py"))}
    archive = output / "source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name in sources:
            bundle.write(ROOT / name, name)
    protocol = {"version": 1, "model": MODEL, "api_base": BASE_URL, "arms": list(ARMS),
                "limits": LIMITS, "cases": records, "source_files": sources,
                "source_archive": archive.relative_to(ROOT).as_posix(), "source_archive_sha256": checkpoints.digest(archive),
                "hidden_feedback_to_solver": False, "skills_enabled": False, "solver_profile": "B0",
                "scope": "Paired small-budget complete coding trials; development and heldout tasks reported separately."}
    save(output / "protocol.json", protocol)
    print(json.dumps({"prepared": [r["name"] for r in records], "model_calls": 0, "protocol": str(output / "protocol.json")}))


def freeze_prepared(protocol_path, output):
    """Reuse hash-verified environment calibration when only harness source changes."""
    previous = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert previous["limits"] == LIMITS and previous["model"] == MODEL
    assert checkpoints.digest(ROOT / previous["source_archive"]) == previous["source_archive_sha256"]
    for record in previous["cases"]:
        for name, digest in record["files"].items():
            assert checkpoints.digest(ROOT / name) == digest, "Prepared environment changed: " + name
        current = case_record(record["name"])
        assert (current["image"], current["instance_id"]) == (record["image"], record["instance_id"])
    output.mkdir(parents=True, exist_ok=False)
    sources = {p.relative_to(ROOT).as_posix(): checkpoints.digest(p)
        for folder in ("agents", "benchmarks") for p in sorted((ROOT / folder).rglob("*.py"))}
    archive = output / "source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name in sources:
            bundle.write(ROOT / name, name)
    frozen = {**previous, "source_files": sources,
        "source_archive": archive.relative_to(ROOT).as_posix(), "source_archive_sha256": checkpoints.digest(archive),
        "reused_calibration_protocol": protocol_path.relative_to(ROOT).as_posix(),
        "reused_calibration_sha256": checkpoints.digest(protocol_path)}
    save(output / "protocol.json", frozen)
    print(json.dumps({"source_files": len(sources), "model_calls": 0, "calibration_reused": True}), flush=True)


def validate_protocol(protocol, task, arm):
    assert protocol["version"] == 1 and protocol["model"] == MODEL and protocol["api_base"] == BASE_URL
    assert protocol["limits"] == LIMITS and protocol["arms"] == list(ARMS)
    assert protocol["hidden_feedback_to_solver"] is False and protocol["skills_enabled"] is False
    assert protocol["solver_profile"] == "B0"
    record = next(row for row in protocol["cases"] if row["name"] == task)
    assert arm in ARMS and task in CASES
    for key, value in CASES[task].items():
        assert record[key] == value, "Case identity or split changed"
    for name, digest in {**protocol["source_files"], **record["files"]}.items():
        assert checkpoints.digest(ROOT / name) == digest, "Frozen file changed: " + name
    assert checkpoints.digest(ROOT / protocol["source_archive"]) == protocol["source_archive_sha256"]
    current = case_record(task)
    for key in ("image", "instance_id", "baseline", "gold"):
        assert record[key] == current[key], "Prepared task identity changed"
    assert all(record["files"].get(name) == digest for name, digest in current["files"].items())
    return record


class SummaryLimitMixin:
    def _compaction_token_budgets(self):
        summary, recent, overhead = super()._compaction_token_budgets()
        return min(summary, LIMITS["checkpoint_summary_tokens"]), recent, overhead

    async def _side_query_once(self, *args, purpose=None, **kwargs):
        if purpose and purpose.startswith("session_fold"):
            kwargs["max_tokens"] = min(kwargs.get("max_tokens", LIMITS["checkpoint_summary_tokens"]),
                                       LIMITS["checkpoint_summary_tokens"])
        return await super()._side_query_once(*args, purpose=purpose, **kwargs)

    async def _run_side_query_call(self, *, purpose, **kwargs):
        previous = self._trial_call_purpose
        self._trial_call_purpose = purpose
        try:
            return await super()._run_side_query_call(purpose=purpose, **kwargs)
        finally:
            self._trial_call_purpose = previous

    async def _await_model_call(self, call):
        summary = bool(self._trial_call_purpose and self._trial_call_purpose.startswith("session_fold"))
        counter, limit = ("_trial_summary_calls", LIMITS["summary_calls"]) if summary else ("_trial_solve_attempts", LIMITS["solve_calls"])
        if getattr(self, counter) >= limit:
            call.close()
            if not summary:
                self.termination_reason = {"kind": "solve_calls"}
            raise _BudgetExceeded("Separate " + ("summary" if summary else "solve") + " call allowance exhausted")
        setattr(self, counter, getattr(self, counter) + 1)
        return await super()._await_model_call(call)


async def run(protocol_path, task, arm, output):
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"task": task, "arm": arm, "status": "incomplete", "paid_model_calls": 0}
    trace = None
    agent = None
    checkpoints_list = []
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        record = validate_protocol(protocol, task, arm)
        receipt.update(split=record["split"], scope=record["scope"], protocol_sha256=checkpoints.digest(protocol_path))
        receipt["prior_exposure"] = record["prior_exposure"]
        save(output / "protocol.json", protocol)
        from dotenv import dotenv_values
        from benchmarks.research_budget import ResearchBudget
        settings = dotenv_values(ROOT / ".env")
        assert str(settings.get("OPENAI_BASE_URL", "")).rstrip("/") == BASE_URL
        key = os.environ.get("SANS_BENCHMARK_API_KEY") or settings.get("OPENAI_API_KEY")
        assert key, "Missing local API credential"
        for name, value in list(os.environ.items()):
            if key in value:
                os.environ.pop(name)
        trace = TraceRecorder(output, run_id=output.name, task_id=record["instance_id"], attempt_id="001")
        spec = json.loads((ROOT / record["research"] / "tasks" / (record["key"] + ".json")).read_text(encoding="utf-8"))
        base_class = _FoldingLoop
        if arm == "evidence_fold":
            from benchmarks.folding_checkpoint import EvidenceFoldingLoop
            base_class = EvidenceFoldingLoop
        loop_class = type("ResearchTrialLoop", (SummaryLimitMixin, base_class), {})
        with transport(), bind_case(record), ResearchBudget(output, trace, allocation=LIMITS["case_tokens"]) as budget:
            image = json.loads(checkpoints.docker("image", "inspect", record["image"]).stdout)[0]["Id"]
            assert image == record["image"]
            with pilot.workspace(image) as env, _isolated_home(output):
                tools = EnvironmentTools(env, "/testbed", shell_path_prefix=pilot.BIN)
                captured = set()

                async def capture(force=False):
                    decisions = agent.decision_clock["completed"] if agent else 0
                    if not force and (decisions not in {0, 6, 12, 18, 24} or decisions in captured):
                        return
                    snapshot = await asyncio.to_thread(env.snapshot, output / "checkpoints")
                    checkpoints_list.append({"solve_calls": decisions, "snapshot": snapshot.name, "final": force,
                                             "usage": summarize_trace(read_trace(trace.trace_path))["provider_usage"]})
                    captured.add(decisions)
                    save(output / "checkpoints.json", checkpoints_list)
                    print(json.dumps({"phase": "checkpoint", "solve_calls": decisions, "final": force}), flush=True)

                async def before_call():
                    await capture()
                    await budget.before_call()

                agent = loop_class(token_budget=budget.token_budget, before_model_call=before_call,
                    auto_compact=arm != "flat", fold_every_decisions=6 if arm != "flat" else None,
                    max_decisions=24, model=MODEL, api_base=BASE_URL, api_key=key, thinking=True,
                    openai_extra_body={"reasoning_effort": "low"}, custom_system_prompt=SOLVER_SYSTEM_PROMPT,
                    custom_tools=get_solver_tool_definitions("B0"), tool_executor=tools, repo_state_probe=tools.repo_state,
                    repair_task=TaskEnvelope(record["instance_id"], spec["problem_statement"], "/testbed"),
                    diff_check_command=_FULL_DIFF_CHECK, skills_enabled=False, is_sub_agent=True,
                    permission_mode="bypassPermissions", trace=trace, max_model_calls=30, max_tool_calls=120,
                    max_output_tokens=8192, max_input_tokens_per_call=32768, model_call_timeout_seconds=120)
                agent._trial_summary_calls = 0
                agent._trial_solve_attempts = 0
                agent._trial_call_purpose = None
                prompt = ("Implement the public specification in /testbed, an offline Linux repository with Python, pytest and dependencies prepared. "
                          "Register acceptance from exact quotes in the specification, inspect source, implement, and run focused behavioral checks. "
                          "Use the installed environment. Hidden tests and original source archives are unavailable. "
                          "Keep diagnostic reversions in an isolated copy and leave the repaired repository in its correct final state.\n\n"
                          + spec["problem_statement"])
                try:
                    receipt["result"] = await agent.run_once(prompt)
                except Exception as error:
                    receipt["solve_error"] = {"type": type(error).__name__, "message": str(error)[:1500]}
                finally:
                    try:
                        await capture(force=True)
                        receipt.update(termination=agent.termination_reason, finish_reason=agent.finish_reason,
                                       repair_state=agent._repair_state.snapshot(), summary_call_attempts=agent._trial_summary_calls,
                                       solve_call_attempts=agent._trial_solve_attempts,
                                       solve_calls=agent.decision_clock["completed"], token_budget=dict(budget.token_budget))
                    finally:
                        await agent.shutdown()
            grades = []
            for point in checkpoints_list:
                graded = await pilot.evaluate(record["key"], image, output / "checkpoints" / point["snapshot"],
                                              output / "grading" / Path(point["snapshot"]).stem)
                grades.append({**point, "grade": score(graded, record["calibration"])})
            save(output / "grades.json", grades)
            receipt.update(status="interrupted" if receipt.get("solve_error") else "completed", final_grade=grades[-1]["grade"])
            receipt["source_unchanged"] = all(checkpoints.digest(ROOT / name) == digest for name, digest in protocol["source_files"].items())
            if not receipt["source_unchanged"]:
                receipt["status"] = "source_changed"
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)[:1500]}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if trace:
            events = read_trace(trace.trace_path)
            receipt.update(summarize_trace(events))
            calls = [e for e in events if e["event_type"] in {"model.call.end", "model.call.error"}
                     and (e.get("attributes") or {}).get("error_type") != "_BudgetExceeded"]
            receipt["paid_model_calls"] = len(calls)
            receipt["summary_calls"] = sum(str(e.get("purpose") or "").startswith("session_fold") for e in calls)
            receipt["solve_physical_requests"] = len(calls) - receipt["summary_calls"]
            receipt["natural_finish"] = any(e["event_type"] == "hook.stop" and e["status"] == "ok" for e in events)
        save(output / "receipt.json", receipt)
        print(json.dumps({k: receipt.get(k) for k in ("status", "task", "arm", "final_grade", "paid_model_calls", "error")}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "run"))
    parser.add_argument("--output", type=artifact_path, required=True)
    parser.add_argument("--protocol", type=artifact_path)
    parser.add_argument("--task", choices=tuple(CASES))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "prepare":
        asyncio.run(prepare(args.output))
    elif args.mode == "freeze":
        if not args.protocol:
            parser.error("freeze requires --protocol from a completed preparation")
        freeze_prepared(args.protocol, args.output)
    else:
        if not all((args.protocol, args.task, args.arm)):
            parser.error("run requires --protocol, --task and --arm")
        receipt = asyncio.run(run(args.protocol, args.task, args.arm, args.output))
        if receipt["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
