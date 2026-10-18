"""Cached local Harbor cases: prepare once, verify offline, then solve."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from agents.tool_result import PytestReport
from .runner import EXPERIMENT_ROOT, experiment_path
from .panel_run import COMMANDCODE_BASE, COMMANDCODE_MODEL, LIVE_AGENT, SECRET_ENV

REPO = EXPERIMENT_ROOT.parent
CACHE = EXPERIMENT_ROOT / "cache" / "benchmark"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_id(image: str) -> str:
    return subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True, stderr=subprocess.PIPE).strip()


def identity(task: Path) -> dict:
    spec = json.loads((task / "case.json").read_text(encoding="utf-8"))
    if digest(task / "instruction.md") != spec["instruction_sha256"]:
        raise ValueError("Task statement differs from its pinned UTF-8 source.")
    return {"image_id": image_id(spec["image"]), "task_files": {
        str(p.relative_to(task)): digest(p) for p in sorted(task.rglob("*")) if p.is_file()
    }}


def report(path: Path) -> dict:
    parsed = PytestReport.read(path)
    if parsed is None:
        raise ValueError(f"Missing or invalid pytest report: {path}")
    return asdict(parsed)


def validate_ready(receipt: dict, current: dict) -> None:
    if receipt.get("status") != "ready" or receipt.get("identity") != current:
        raise ValueError("Environment is not ready or changed; run prepare before paid solving.")
    if receipt.get("checked_modes") != ["nop", "oracle"] or not receipt.get("check_files"):
        raise ValueError("Both nop and oracle environment checks are required.")
    for path, expected in receipt["check_files"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Environment check evidence changed; run prepare again.")


def subnet() -> str:
    ids = subprocess.check_output(["docker", "network", "ls", "-q"], text=True).split()
    networks = json.loads(subprocess.check_output(["docker", "network", "inspect", *ids], text=True))
    used = [ipaddress.ip_network(c["Subnet"]) for n in networks for c in (n["IPAM"].get("Config") or []) if c.get("Subnet")]
    return str(next(n for n in ipaddress.ip_network("10.223.0.0/16").subnets(new_prefix=24)
                    if all(u.version != n.version or not u.overlaps(n) for u in used)))


def harbor(task: Path, mode: str, output: Path, frozen: dict, *, agent_options: dict | None = None) -> tuple[Path, dict]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "empty-skills").mkdir()
    compose = output / "docker.compose.yaml"
    compose.write_text('services:\n  main:\n    image: "' + frozen["image_id"] + '"\n    pull_policy: never\n'
                       'networks:\n  default:\n    internal: true\n    ipam:\n      config:\n        - subnet: ' + subnet() + '\n')
    agent = {"name": mode} if mode != "live" else {
        "import_path": LIVE_AGENT, "model_name": COMMANDCODE_MODEL,
        "kwargs": {"api_base": COMMANDCODE_BASE, "provider_revision": "commandcode-v1", "thinking": True,
                   "skills_seed": str(output / "empty-skills"), "model_call_timeout_seconds": 360},
    }
    if mode == "live" and agent_options:
        agent["kwargs"].update(agent_options)
    config = {"job_name": output.name, "jobs_dir": str(output / "jobs"), "n_concurrent_trials": 1,
              "retry": {"max_retries": 0}, "tasks": [{"path": str(task)}], "agents": [agent],
              "environment": {"type": "docker", "delete": True, "extra_docker_compose": [str(compose)]}}
    (output / "job.json").write_text(json.dumps(config, indent=2))
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    with (output / "harbor.log").open("w", encoding="utf-8") as log:
        run = subprocess.run([str(Path(sys.executable).with_name("harbor")), "run", "--config", str(output / "job.json")],
                             cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    results = list((output / "jobs").glob("*/*/result.json"))
    if run.returncode or len(results) != 1:
        raise RuntimeError(f"Harbor failed before a complete trial; see {output / 'harbor.log'}")
    result = json.loads(results[0].read_text(encoding="utf-8"))
    if result.get("exception_info"):
        raise RuntimeError(f"Trial incomplete; see {results[0]}")
    if identity(task) != frozen:
        raise ValueError("Task or image changed during execution.")
    return results[0].parent, result


def prepare(task: Path, output: Path) -> None:
    spec = json.loads((task / "case.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    try:
        image_id(spec["image"])
    except subprocess.CalledProcessError:
        if image_id(spec["toolchain_image"]) != spec["toolchain_id"]:
            raise ValueError("Pinned Python toolchain is missing or changed; see experiment/benchmarks/README.md")
        archive = experiment_path(REPO / spec["source_archive"])
        if digest(archive) != spec["source_sha256"]:
            raise ValueError("Cached source archive differs from the pinned case.")
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE) as directory:
            context = Path(directory)
            shutil.copyfile(archive, context / "source.tar")
            with (output / "build.log").open("w") as log:
                subprocess.run(["docker", "build", "--pull=false", "--network=none", "-t", spec["image"],
                                "-f", str(task / "environment" / "Dockerfile"), str(context)],
                               env=dict(os.environ, DOCKER_BUILDKIT="0"), stdout=log, stderr=subprocess.STDOUT, check=True)
    frozen = identity(task)
    ready = CACHE / task.name / "ready.json"
    check_files = {}
    for mode, expected_failures in [("nop", spec["regression_failures"]), ("oracle", 0)]:
        trial, result = harbor(task, mode, output / mode, frozen)
        verifier = trial / "verifier"
        for file, count, failures in [("existing.xml", spec["existing_tests"], 0),
                                      ("junit.xml", spec["regression_tests"], expected_failures)]:
            data = report(verifier / file)
            if (data["collected"], data["failure"], data["error"], data["skipped"]) != (count, failures, 0, 0):
                raise ValueError(f"Unexpected {mode} test result: {data}; environment not ready")
        facts = json.loads((verifier / "environment.json").read_text(encoding="utf-8"))
        if facts["history_count"] != 1 or (mode == "nop" and (facts["tree"] != spec["source_tree"] or facts["dirty"])):
            raise ValueError("Task source does not match its isolated baseline.")
        if result["verifier_result"]["rewards"]["reward"] != (0.0 if expected_failures else 1.0):
            raise ValueError("Verifier reward disagrees with expected calibration.")
        if mode == "oracle" and (trial / "agent" / "exit-code.txt").exists():
            raise ValueError("Reference solution failed.")
        for p in [trial / "result.json", *verifier.glob("*")]:
            if p.is_file():
                check_files[str(p)] = digest(p)
    ready.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "ready", "identity": frozen, "checked_modes": ["nop", "oracle"],
               "check_files": check_files, "prepared_at": datetime.now(timezone.utc).isoformat(), "model_calls": 0}
    (output / "ready.json").write_text(json.dumps(receipt, indent=2))
    ready.write_text(json.dumps(receipt, indent=2))
    print(f"READY: {task.name}; no model calls; receipt: {output / 'ready.json'}")


def live(task: Path, output: Path) -> None:
    frozen = identity(task)
    ready_path = CACHE / task.name / "ready.json"
    if not ready_path.is_file():
        raise ValueError("Environment not prepared; run prepare before paid solving.")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    validate_ready(ready, frozen)
    if not os.environ.get(SECRET_ENV):
        raise ValueError(f"Set {SECRET_ENV} in the host environment; never put it in case files.")
    source_hashes = {str(p.relative_to(REPO)): digest(p) for folder in ("agents", "benchmarks")
                     for p in (REPO / folder).rglob("*.py")}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if output.exists():
        raise FileExistsError("Run outputs are immutable; choose a new output directory.")
    receipt = {"status": "incomplete", "harness_commit": commit, "source_hashes": source_hashes,
               "identity": frozen, "readiness": ready}
    try:
        trial, result = harbor(task, "live", output, frozen)
        if any(digest(REPO / p) != h for p, h in source_hashes.items()):
            raise ValueError("Harness source changed during the run.")
        receipt.update(status="completed", trial=str(trial), verifier_result=result["verifier_result"])
    finally:
        from agents.telemetry import read_trace, summarize_trace
        traces = list(output.glob("jobs/*/*/agent/trace.jsonl"))
        events = read_trace(traces[0]) if len(traces) == 1 else []
        usage = summarize_trace(events)["provider_usage"]
        calls = sum(e["event_type"] == "model.call.start" for e in events)
        receipt.update(provider_usage=usage, model_calls=calls,
                       calls_without_usage=calls - (usage.get("usage_observed_model_calls") or 0),
                       natural_completion=any(e["event_type"] == "hook.stop" and e["status"] == "ok" for e in events))
        if output.is_dir():
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in receipt.items() if k not in {"source_hashes", "identity", "readiness"}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--case", type=experiment_path, required=True, help="Local, standard Harbor task")
    parser.add_argument("--output", type=experiment_path, required=True, help="New output directory; never overwritten")
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else live)(args.case, args.output)


if __name__ == "__main__":
    main()
