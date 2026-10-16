"""Frozen two-task FeatureBench pilot; preparation and grading are model-free."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import shlex
import time
import zipfile

from agents.prompt import SOLVER_SYSTEM_PROMPT
from agents.repair_state import TaskEnvelope
from agents.telemetry import TraceRecorder, read_trace, summarize_trace
from agents.tool_result import PytestReport
from agents.tools import get_solver_tool_definitions
from benchmarks.checkpoints import DockerWorkspace, digest, docker
from benchmarks.context_folding import ContextFoldingAgent, _FoldingLoop
from benchmarks.harbor_agent import EnvironmentTools, _FULL_DIFF_CHECK, _isolated_home

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "experiment/research/20261006-context-pilot"
OUT = ROOT / "experiment/runs/20261006-context-pilot"
KEYS = ("test_compat", "test_experimental_arguments_schema")
BIN = "/opt/miniconda3/envs/testbed/bin"
MODEL = "deepseek/deepseek-v4.1-flash"
BASE_URL = "https://api.commandcode.ai/provider/v1"
POINTS = {0, 5, 10, 20, 40, 80}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def task(key):
    return json.loads((RESEARCH / "tasks" / f"{key}.json").read_text(encoding="utf-8"))


async def checked(env, command, timeout=120):
    result = await env.exec(command, timeout_sec=timeout)
    if result.return_code:
        raise RuntimeError(f"Preparation failed ({result.return_code}): {result.stdout[-2000:]} {result.stderr[-2000:]}")
    return result.stdout


def workspace(image):
    return DockerWorkspace(image, memory="4g", cpus=2)


async def evaluate(key, image, snapshot, destination):
    """Grade frozen tests in a fresh container; never return these results to the solver."""
    destination = Path(destination)
    receipt_path = destination / "receipt.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    spec = task(key)
    if digest(snapshot) != snapshot.stem:
        raise ValueError("Snapshot hash mismatch")
    started = time.monotonic()
    receipt = {"snapshot": snapshot.name, "image": image, "suites": {}, "status": "unknown"}
    with workspace(image) as env:
        await env.upload_file(snapshot, "/tmp/candidate.tar")
        await env.upload_file(OUT / key / "checks.tar", "/tmp/checks.tar")
        # These literal paths are inside this disposable container only.
        await checked(env, "find /testbed -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; tar -xf /tmp/candidate.tar -C /testbed; tar -xf /tmp/checks.tar -C /testbed")
        for group, paths in (("feature", spec["FAIL_TO_PASS"]), ("regression", spec["PASS_TO_PASS"])):
            benchmark_option = "--benchmark-disable " if key == "test_experimental_arguments_schema" else ""
            command = (f"export PATH={BIN}:$PATH; python -m pytest -q -rA --tb=short --color=no "
                       f"-o addopts= {benchmark_option}--junitxml=/tmp/{group}.xml " + " ".join(map(shlex.quote, paths)))
            result = await env.exec(command, timeout_sec=120)
            (destination / f"{group}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            xml = await env.exec(f"cat /tmp/{group}.xml")
            (destination / f"{group}.xml").write_text(xml.stdout, encoding="utf-8")
            report = PytestReport.parse(xml.stdout) if xml.return_code == 0 else None
            receipt["suites"][group] = {"exit_code": result.return_code, "report": asdict(report) if report else None}
        receipt["status"] = "graded" if all(s["report"] and s["exit_code"] in (0, 1, 2) for s in receipt["suites"].values()) else "unknown"
    receipt["elapsed_seconds"] = time.monotonic() - started
    save(receipt_path, receipt)
    return receipt


async def prepare(key):
    spec = task(key)
    dest = OUT / key
    dest.mkdir(parents=True, exist_ok=False)
    source_tag = f"harness-featurebench-source:{key}"
    original_image = json.loads(docker("image", "inspect", source_tag).stdout)[0]["Id"]
    with workspace(original_image) as env:
        await checked(env, "cp -a /root/my_repo/. /testbed/; git init -q")
        if key == "test_experimental_arguments_schema":
            # Preserve this upstream test directory in the source-only snapshot format.
            await checked(env, "test \"$(readlink tests/pydantic_core)\" = ../pydantic-core/tests && "
                          "cp -aL tests/pydantic_core /tmp/pilot-core-tests && rm tests/pydantic_core && "
                          "mv /tmp/pilot-core-tests tests/pydantic_core")
        # Save original tests outside the candidate; the paid container never receives this archive.
        await checked(env, "tar -cf /tmp/checks.tar tests")
        await env.download_file("/tmp/checks.tar", dest / "checks.tar")
        gold = await asyncio.to_thread(env.snapshot, dest / "calibration/snapshots")
        patch = dest / "mask.patch"
        patch.write_text(spec["patch"] + ("" if spec["patch"].endswith("\n") else "\n"), encoding="utf-8")
        await env.upload_file(patch, "/tmp/mask.patch")
        await checked(env, "git apply --whitespace=fix /tmp/mask.patch")
        hidden = " ".join(shlex.quote(p) for p in spec["FAIL_TO_PASS"])
        await checked(env, f"rm -f -- {hidden}; rm -rf /root/my_repo /testbed/.git; rm -f /tmp/mask.patch /tmp/checks.tar")
        await checked(env, "git init -q; git config user.email pilot@localhost; git config user.name pilot; git add -A; git commit -qm masked-baseline")
        baseline = await asyncio.to_thread(env.snapshot, dest / "calibration/snapshots")
        # Keep the masked checkout as the only accessible project copy.
        seal = await checked(env, "test ! -e /root/my_repo; git log --format=%s; find /testbed -type l; du -sh /testbed; "
                             f"export PATH={BIN}:$PATH; python -m pytest --version")
        (dest / "environment.log").write_text(seal, encoding="utf-8")
        (dest / "requirements.freeze.txt").write_text(await checked(env, f"{BIN}/python -m pip freeze"), encoding="utf-8")
        tag = f"harness-context-pilot:{key.replace('_', '-')}-20261006"
        docker("commit", env.name, tag)
    image = json.loads(docker("image", "inspect", tag).stdout)[0]["Id"]
    calibration = {}
    for mode, snapshot in (("baseline", baseline), ("gold", gold)):
        calibration[mode] = []
        for repeat in (1, 2):
            result = await evaluate(key, image, snapshot, dest / "calibration" / f"{mode}-{repeat}")
            calibration[mode].append(result)
            print(json.dumps({"task": key, "mode": mode, "repeat": repeat,
                              "status": result["status"], "counts": {n: s["report"] and {k: s["report"][k] for k in ('collected','failure','error','skipped')} for n,s in result['suites'].items()} }), flush=True)
    for mode, results in calibration.items():
        assert results[0]["status"] == results[1]["status"] == "graded", results
        assert all(sorted((c[0], c[1]) for c in results[0]["suites"][s]["report"]["cases"]) ==
                   sorted((c[0], c[1]) for c in results[1]["suites"][s]["report"]["cases"])
                   for s in ("feature", "regression")), "Unstable environment"
    assert all(s["exit_code"] == 0 and s["report"]["collected"] > 0 for s in calibration["gold"][0]["suites"].values()), "Gold failed"
    assert calibration["baseline"][0]["suites"]["feature"]["exit_code"] != 0, "No initial failure"
    save(dest / "ready.json", {"image": image, "upstream_image": original_image, "task_sha256": digest(RESEARCH / "tasks" / f"{key}.json"),
                              "checks_sha256": digest(dest / "checks.tar"), "baseline": baseline.name, "gold": gold.name,
                              "calibration": calibration})


def freeze():
    for key in KEYS:
        assert (OUT / key / "ready.json").exists(), key
    files = sorted(p for folder in ("agents", "benchmarks", "tests") for p in (ROOT / folder).rglob("*.py"))
    files += list((RESEARCH / "tasks").glob("*.json"))
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    with zipfile.ZipFile(RESEARCH / "source-pilot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(ROOT / name, name)
    save(RESEARCH / "protocol.json", {
        "dataset_revision": "6ed8be1152d3a159317122141dcbdfcfa911ba34", "dataset": "FeatureBench v1.1 fast",
        "task_selection": "Two smaller masks from different repos among packaging/hatch/fastapi/pydantic; packaging and hatch masks exceed 50k bytes. No model outcomes used.",
        "tasks": list(KEYS), "arms": ["C0", "C1", "C2"], "model": MODEL, "api_base": BASE_URL,
        "arm_definitions": {
            "C0": "Flat execution with shared automatic compaction and accounting; memory and skills disabled.",
            "C1": "Serial branches; return summary plus complete child-visible message delta, including pre-compaction messages.",
            "C2": "Same serial branches; return unverified summary only, retaining workspace and structured execution evidence.",
        },
        "comparison": "C1 vs C2 isolates returned context; C0 additionally changes delegation. This does not test procedural memory or skill evolution.",
        "reasoning_effort": "low", "max_output_tokens": 8192, "input_estimate_limit": 65536,
        "max_total_tokens": 1000000, "total_budget": 6000000, "max_model_calls": 80, "max_tool_calls": 160,
        "model_timeout_seconds": 180, "trajectory_timeout_seconds": 2400,
        "branches": 8, "branch_decisions": 10, "auto_compact": True, "source_files": hashes,
        "sampling": "At safe model-call boundaries after 0/5/10/20/40/80 completed responses, plus final; compaction and child calls count. Offline grading never feeds the solver.",
        "scope": "Six single attempts for feasibility, not a statistically powered long-horizon effect estimate.",
    })
    print("Frozen source and six planned trajectories", flush=True)


async def live(key, arm, attempt="initial", unlimited=False, *, max_output_tokens=8192,
               model_timeout_seconds=180, page_branch_history=False):
    from dotenv import dotenv_values
    protocol = json.loads((RESEARCH / "protocol.json").read_text())
    if attempt == "initial":
        assert not unlimited, "Use a distinct attempt for changed limits"
        assert (max_output_tokens, model_timeout_seconds, page_branch_history) == (8192, 180, False)
        assert all(digest(ROOT / p) == h for p, h in protocol["source_files"].items()), "Frozen source changed"
    ready = json.loads((OUT / key / "ready.json").read_text())
    assert digest(RESEARCH / "tasks" / f"{key}.json") == ready["task_sha256"]
    assert digest(OUT / key / "checks.tar") == ready["checks_sha256"]
    run = OUT / key / arm / attempt
    run.mkdir(parents=True, exist_ok=False)
    values = dotenv_values(ROOT / ".env")
    assert str(values.get("OPENAI_BASE_URL", "")).rstrip("/") == BASE_URL
    api_key = os.environ.get("SANS_BENCHMARK_API_KEY") or values.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing existing credential")
    # Unknown usage needs an explicit receipt retaining its conservative budget charge.
    previous = []
    for path in OUT.glob("*/C[012]/*/receipt.json"):
        receipt = json.loads(path.read_text())
        if receipt["token_budget"].get("blocked") in ("unknown_usage_after_error", "missing_usage"):
            resolution = json.loads(path.with_name("usage-reconciliation.json").read_text())
            assert resolution["receipt_sha256"] == digest(path)
            assert resolution["retained_reservation"] == receipt["token_budget"]["charged"], "Unresolved usage"
        previous.append(receipt)
    spent = sum(p["token_budget"]["charged"] for p in previous)
    allowance = None if unlimited else min(protocol["max_total_tokens"], protocol["total_budget"] - spent)
    if allowance is not None and allowance <= 0:
        raise RuntimeError("Batch budget exhausted")
    save(run / "conditions.json", {"unlimited": unlimited, "image": ready["image"], "model": MODEL,
                                  "max_output_tokens": max_output_tokens, "input_estimate_limit": 65536,
                                  "model_timeout_seconds": model_timeout_seconds,
                                  "page_branch_history": page_branch_history,
                                  "max_total_tokens": allowance, "max_model_calls": None if unlimited else 80,
                                  "max_tool_calls": None if unlimited else 160,
                                  "max_branches": None if unlimited else 8, "branch_model_calls": 10,
                                  "trajectory_timeout_seconds": None if unlimited else 2400,
                                  "source_files": {p: digest(ROOT / p) for p in protocol["source_files"]}})
    spec = task(key)
    trace = TraceRecorder(run, run_id=f"{key}-{arm}-{attempt}", task_id=spec["instance_id"], attempt_id=attempt)
    checkpoints, captured = [], set()
    started = time.monotonic()
    result = error = None
    with workspace(ready["image"]) as env, _isolated_home(run):
        tools = EnvironmentTools(env, "/testbed", shell_path_prefix=BIN)

        async def capture(force=False):
            events = read_trace(trace.trace_path)
            responses = [e for e in events if e["event_type"] == "model.call.end" and e["status"] == "ok"]
            count = len(responses)
            if not force and (count not in POINTS or count in captured):
                return
            pending = asyncio.create_task(asyncio.to_thread(env.snapshot, run / "checkpoints"))
            try:
                snap = await asyncio.shield(pending)
            except asyncio.CancelledError:
                await pending
                raise
            point = {"calls": count, "main_calls": sum(e["purpose"] == "main" for e in responses),
                     "snapshot": snap.name, "final": force, "usage": summarize_trace(events)["provider_usage"]}
            checkpoints.append(point)
            captured.add(count)
            with (run / "checkpoints/manifest.jsonl").open("a") as f:
                f.write(json.dumps(point) + "\n")
            print(json.dumps({"task": key, "arm": arm, "attempt": attempt, "checkpoint": count, "usage": point["usage"]}), flush=True)

        options = dict(model=MODEL, api_base=BASE_URL, api_key=api_key, thinking=True,
                       openai_extra_body={"reasoning_effort": "low"}, custom_system_prompt=SOLVER_SYSTEM_PROMPT,
                       custom_tools=get_solver_tool_definitions("B1"), tool_executor=tools, repo_state_probe=tools.repo_state,
                       diff_check_command=_FULL_DIFF_CHECK, repair_task=TaskEnvelope(spec["instance_id"], spec["problem_statement"], "/testbed"),
                       skills_enabled=False, is_sub_agent=True, permission_mode="bypassPermissions", trace=trace,
                       max_model_calls=None if unlimited else 80, max_tool_calls=None if unlimited else 160,
                       max_output_tokens=max_output_tokens, max_input_tokens_per_call=65536,
                       model_call_timeout_seconds=model_timeout_seconds, auto_compact=True, before_model_call=capture)
        agent = (_FoldingLoop(token_budget={"limit": allowance, "charged": 0}, **options) if arm == "C0" else
                 ContextFoldingAgent(max_total_tokens=allowance, max_branches=None if unlimited else 8, branch_model_calls=10,
                                     retain_branch_history=arm == "C1", page_branch_history=page_branch_history, **options))
        prompt = ("Work in /testbed, an offline Linux checkout with Python and pytest installed. "
                  "Implement the public specification. Register acceptance using exact short quotes from it; "
                  "create focused behavioral tests as needed and label checks with acceptance_ids, scope and oracle. "
                  "Receive prerequisite observations before dependent actions. Existing regression tests remain available. "
                  "Do not access hidden tests, original source archives, network services or installed copies of this project's source.\n\n"
                  + spec["problem_statement"])
        try:
            await capture()
            result = await asyncio.wait_for(agent.run_once(prompt), timeout=None if unlimited else 2400)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        finally:
            try:
                await capture(force=True)
                diff = await env.exec("git diff --binary HEAD; git status --short")
                (run / "candidate.diff").write_text(diff.stdout, encoding="utf-8")
            finally:
                await agent.shutdown()
                events = read_trace(trace.trace_path)
                save(run / "receipt.json", {"arm": arm, "task": key, "attempt": attempt, "model": MODEL,
                                           "error": error, "result": result, "finish_reason": agent.finish_reason,
                                           "repair_state": agent._repair_state.snapshot(), "token_budget": agent.token_budget,
                                           "elapsed_seconds": time.monotonic() - started, **summarize_trace(events)})
    print(json.dumps({"task": key, "arm": arm, "error": error, "budget": agent.token_budget}), flush=True)


def outcomes(receipt):
    return {f"{suite}:{case[0]}": case[1] for suite, value in receipt["suites"].items()
            for case in (value["report"] or {}).get("cases", [])}


async def grade(key, arm, attempt="initial"):
    run = OUT / key / arm / attempt
    ready = json.loads((OUT / key / "ready.json").read_text())
    original = outcomes(ready["calibration"]["baseline"][0])
    gold = outcomes(ready["calibration"]["gold"][0])
    expected = {k for k, v in gold.items() if v == "passed"}
    preserved = {k for k in expected if original.get(k) == "passed"}
    targets = expected - preserved
    assert targets
    points = [json.loads(l) for l in (run / "checkpoints/manifest.jsonl").read_text().splitlines()]
    for point in points:
        snap = run / "checkpoints" / point["snapshot"]
        receipt = await evaluate(key, ready["image"], snap, run / "evaluations" / snap.stem)
        current = outcomes(receipt)
        passed = {k for k, v in current.items() if v == "passed"}
        valid = receipt["status"] == "graded"
        p_ok = preserved <= passed
        official_suites_ok = all(s["exit_code"] == 0 for s in receipt["suites"].values())
        point["grade"] = {"status": receipt["status"], "resolved": (expected <= passed and official_suites_ok) if valid else None,
                          "feature_passed": len(targets & passed), "feature_total": len(targets),
                          "regressions": len(preserved - passed), "preserved_total": len(preserved),
                          "completion": (len(targets & passed) / len(targets) if p_ok else 0) if valid else None}
    save(run / "evaluations.json", points)
    print(json.dumps({"task": key, "arm": arm, "final": points[-1]["grade"]}), flush=True)


def report():
    rows, curves = [], []
    for path in sorted(OUT.glob("*/C[012]/*/receipt.json")):
        receipt = json.loads(path.read_text())
        evaluation = path.with_name("evaluations.json")
        points = json.loads(evaluation.read_text()) if evaluation.exists() else []
        events = read_trace(path.with_name("trace.jsonl"))
        usage = receipt["provider_usage"]
        final = points[-1]["grade"] if points else {}
        natural = bool(receipt["result"] and receipt["result"]["termination_reason"] is None
                       and not receipt["error"] and receipt["finish_reason"] in ("stop", "end_turn"))
        row = {k: receipt[k] for k in ("task", "arm", "attempt", "model_calls", "tool_calls", "elapsed_seconds", "error", "token_budget")}
        row.update(usage=usage, final=final, natural_finish=natural,
                   false_finish=natural and final.get("resolved") is False,
                   branches=sum(e["event_type"] == "context.branch.start" for e in events),
                   compactions=sum(e["event_type"] == "model.call.end" and e["purpose"] == "session_fold" for e in events),
                   receipt=str(path.relative_to(ROOT)))
        rows.append(row)
        curves.extend({"task": receipt["task"], "arm": receipt["arm"], "attempt": receipt["attempt"],
                       "calls": p["calls"], "tokens": p["usage"].get("total_tokens"),
                       "completion": p["grade"]["completion"], "final": p["final"]} for p in points)
    total = sum(r["usage"].get("total_tokens") or 0 for r in rows)
    groups = {}
    for row in rows:
        key = f"{row['attempt']}/{row['arm']}"
        group = groups.setdefault(key, {"runs": 0, "passed": 0, "failed": 0, "unknown": 0, "known_tokens": 0})
        group["runs"] += 1
        resolved = row["final"].get("resolved")
        group["passed" if resolved is True else "failed" if resolved is False else "unknown"] += 1
        group["known_tokens"] += row["usage"].get("total_tokens") or 0
    summary = {"runs": rows, "known_tokens": total, "usage_complete": all(r["usage"]["usage_complete"] for r in rows),
               "model_calls": sum(r["model_calls"] for r in rows), "groups": groups}
    (RESEARCH / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (RESEARCH / "completion.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("task", "arm", "attempt", "calls", "tokens", "completion", "final"))
        writer.writeheader()
        writer.writerows(curves)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "freeze", "live", "grade", "report"))
    parser.add_argument("--task", choices=KEYS)
    parser.add_argument("--arm", choices=("C0", "C1", "C2"))
    parser.add_argument("--attempt", default="initial")
    parser.add_argument("--unlimited", action="store_true", help="Remove total token, call and trajectory time limits")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--model-timeout-seconds", type=int, default=180)
    parser.add_argument("--page-branch-history", action="store_true", help="Use observation paging for oversized C1 returns")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "report":
        report()
    elif args.mode == "prepare":
        asyncio.run(prepare(args.task))
    elif args.mode == "live":
        asyncio.run(live(args.task, args.arm, args.attempt, args.unlimited,
                         max_output_tokens=args.max_output_tokens, model_timeout_seconds=args.model_timeout_seconds,
                         page_branch_history=args.page_branch_history))
    else:
        asyncio.run(grade(args.task, args.arm, args.attempt))
