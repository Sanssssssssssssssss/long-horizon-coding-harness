"""Frozen execution-based Skill ablations on constructed batch adapters, not a public bench."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
import queue
from pathlib import Path
import signal
import subprocess
import socketserver
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from dotenv import dotenv_values

from agents.online_skill_eval import _generate_variant_response_async
from agents.telemetry import TraceRecorder
from benchmarks.context_folding import _FoldingLoop
from benchmarks.research_budget import ResearchBudget, save_json

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "experiment/research/20261009-skill-folding"
SOURCE = ROOT / "experiment/research/20261008-auto-research/phase34/published/sequence-skill-002/receipt.json"

WORKER = r'''import importlib.util,json,sys
from pathlib import Path
transport=sys.stdout
sys.stdout=sys.stderr
sys.path.insert(0,str(Path(__file__).parent))
spec=importlib.util.spec_from_file_location('candidate_adapter',Path(__file__).with_name('candidate_adapter.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def mutable_ids(value):
    if isinstance(value,(list,dict)):
        children=value.values() if isinstance(value,dict) else value
        return {id(value)} | {item for child in children for item in mutable_ids(child)}
    return set()
for line in sys.stdin:
    request=json.loads(line)
    values=request['values']
    before=json.dumps(values,sort_keys=True)
    try:
        result=getattr(module,request['function'])(iter(values) if request['single_use'] else values,**request['kwargs'])
        response={'result':result,'contract_error':None if isinstance(result,list) else 'Result must be a list',
                  'input_mutated':json.dumps(values,sort_keys=True)!=before,
                  'mutable_alias':bool(mutable_ids(values)&mutable_ids(result))}
    except Exception as error:
        response={'error':{'type':type(error).__name__,'message':str(error)},
                  'input_mutated':json.dumps(values,sort_keys=True)!=before,'mutable_alias':False}
    response['_request_id']=request['_request_id']
    transport.write(json.dumps(response,allow_nan=False)+'\n')
    transport.flush()
'''

PROXY = r'''import json,os,socket
class OracleTransportError(Exception):
    pass
def _call(function,values,kwargs):
    request={'token':os.environ['SKILL_ORACLE_TOKEN'],'function':function,'single_use':iter(values) is values,
             'values':list(values),'kwargs':kwargs}
    with socket.create_connection(('127.0.0.1',int(os.environ['SKILL_ORACLE_PORT'])),timeout=10) as connection:
        connection.sendall((json.dumps(request,allow_nan=False)+'\n').encode())
        response=json.loads(connection.makefile('rb').readline(1048576))
    if 'transport_error' in response:
        raise OracleTransportError(response['transport_error'])
    if response.get('input_mutated') or response.get('mutable_alias') or response.get('contract_error'):
        raise AssertionError('Candidate mutated input, retained mutable aliases, or violated the result type')
    if 'error' in response:
        error=response['error']
        kind={'ValueError':ValueError,'TypeError':TypeError,'RuntimeError':RuntimeError}.get(error['type'],RuntimeError)
        raise kind(error['message'])
    return response['result']
def decode_many(values,*,on_error='raise',**core_options):
    return _call('decode_many',values,dict(on_error=on_error,**core_options))
def decode_records(records,*,field='value',destination='decoded',on_error='raise',**core_options):
    return _call('decode_records',records,dict(field=field,destination=destination,on_error=on_error,**core_options))
'''


def fixtures():
    spec = importlib.util.spec_from_file_location("transfer_fixture", ROOT /
        "benchmarks/skill_transfer_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    originals = {name: getattr(module, name) for name in ("CORE", "TASK", "GOLD", "CHECKS")}
    cases = []
    for label, split, mapping in [
        ("measurement", "selection", {}),
        ("numeric_csv", "test", {"decode_cell": "parse_amount", "decode_many": "parse_amounts",
            "decode_records": "parse_rows", "adapter.py": "adapter.py", "measurement decoder": "CSV numeric parser"}),
        ("sensor_conversion", "test", {"decode_cell": "convert_reading", "decode_many": "convert_many",
            "decode_records": "convert_events", "measurement decoder": "sensor unit converter"}),
    ]:
        values = deepcopy(originals)
        for name in values:
            for old, new in mapping.items():
                values[name] = values[name].replace(old, new)
        # New held-out source contracts change accepted options; no memorized old signature.
        if split == "test":
            for name in values:
                values[name] = values[name].replace("offset", "bias").replace("as_int", "integral")
        cases.append({"id": label, "split": split, **values})
    return cases


def grade(case, code, destination):
    """Trusted assertions call a persistent candidate worker using JSON, never import its code."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    candidate, trusted = destination / "candidate", destination / "trusted"
    candidate.mkdir()
    trusted.mkdir()
    for name, text in {"core.py": case["CORE"], "candidate_adapter.py": code, "worker.py": WORKER}.items():
        (candidate / name).write_text(text, encoding="utf-8", newline="\n")
    functions = ["decode_many", "decode_records"]
    names = ["parse_amounts", "parse_rows"] if case["id"] == "numeric_csv" else ["convert_many", "convert_events"] if case["id"] == "sensor_conversion" else functions
    proxy = PROXY
    for old, new in zip(functions, names):
        proxy = proxy.replace(old, new)
    for name, text in {"adapter.py": proxy, "test_adapter.py": case["CHECKS"]}.items():
        (trusted / name).write_text(text, encoding="utf-8", newline="\n")
    oracle_files = {path.name: sha256(path.read_bytes()).hexdigest() for path in trusted.iterdir()}
    safe_env = {name: value for name, value in os.environ.items()
                if name in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC"}}
    safe_env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                    PYTEST_ADDOPTS="--rootdir=. -p no:cacheprovider")
    command = [sys.executable, "-B", "-m", "pytest", "-q", "test_adapter.py", "--junitxml=checks.xml"]
    token = uuid.uuid4().hex
    worker_log = (destination / "candidate.log").open("w", encoding="utf-8")
    worker = subprocess.Popen([sys.executable, "-I", "-B", "-u", "worker.py"], cwd=candidate, env=safe_env,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log, text=True, encoding="utf-8",
                              **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}))
    calls, transport_errors = [], []
    outgoing, incoming = queue.Queue(), queue.Queue()
    def write_requests():
        while (request := outgoing.get()) is not None:
            try:
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
            except Exception as error:
                incoming.put({"_request_id": request["_request_id"], "transport_error": str(error)})
    def read_responses():
        try:
            for line in worker.stdout:
                incoming.put(json.loads(line))
        except Exception as error:
            incoming.put({"transport_error": str(error)})
    for reader in (write_requests, read_responses):
        threading.Thread(target=reader, daemon=True).start()
    exchange = threading.Lock()
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline(1048576))
                if request.pop("token", None) != token:
                    raise ValueError("Invalid evaluator request token")
                calls.append(request["function"])
                request["_request_id"] = uuid.uuid4().hex
                if not exchange.acquire(timeout=8):
                    raise TimeoutError("Candidate response exchange timed out")
                try:
                    if worker.poll() is not None:
                        raise RuntimeError("Candidate worker exited")
                    outgoing.put(request)
                    deadline = time.monotonic() + 8
                    while True:
                        response = incoming.get(timeout=max(0, deadline - time.monotonic()))
                        if response.get("transport_error") or response.get("_request_id") == request["_request_id"]:
                            break
                finally:
                    exchange.release()
            except Exception as error:
                transport_errors.append(str(error))
                response = {"transport_error": str(error)}
            try:
                self.wfile.write((json.dumps(response) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
    class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        block_on_close = False
    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trusted_env = {**safe_env, "SKILL_ORACLE_PORT": str(server.server_address[1]), "SKILL_ORACLE_TOKEN": token}
    try:
        result = subprocess.run(command, cwd=trusted, env=trusted_env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=30)
        (destination / "execution.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        cases = list(ET.parse(trusted / "checks.xml").getroot().iter("testcase"))
        failures = [item.attrib["name"] for item in cases if item.find("failure") is not None or item.find("error") is not None]
        skipped = [item.attrib["name"] for item in cases if item.find("skipped") is not None]
        identities = sorted((item.attrib.get("classname", ""), item.attrib["name"]) for item in cases)
        expected = [tuple(item) for item in case.get("oracle_expected", identities)]
        identity_match = identities == expected and len(set(identities)) == len(identities)
        harness_unchanged = all(sha256((trusted / name).read_bytes()).hexdigest() == digest for name, digest in oracle_files.items())
        actual_core_sha = sha256((candidate / "core.py").read_bytes()).hexdigest()
        core_unchanged = actual_core_sha == sha256(case["CORE"].encode()).hexdigest()
        score = {"completed": not transport_errors, "passed": result.returncode == 0 and bool(cases) and not skipped and identity_match and harness_unchanged and core_unchanged and not transport_errors,
                 "checks_passed": len(cases) - len(failures) - len(skipped), "checks_total": len(cases),
                 "failures": failures, "skipped": skipped, "case_ids": identities, "identity_match": identity_match,
                 "harness_unchanged": harness_unchanged, "candidate_imported_by_pytest": False,
                 "worker_processes": 1, "worker_calls": len(calls), "transport_errors": transport_errors,
                 "oracle_sha256": sha256((case["CORE"] + case["CHECKS"] + WORKER + proxy).encode()).hexdigest(),
                 "core_sha256": sha256(case["CORE"].encode()).hexdigest()}
        score.update(actual_core_sha256=actual_core_sha, core_unchanged=core_unchanged)
    except (subprocess.TimeoutExpired, OSError, ET.ParseError) as error:
        score = {"completed": False, "passed": False, "error_type": type(error).__name__}
    finally:
        if worker.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"], capture_output=True,
                                   timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(worker.pid, signal.SIGKILL)
            except (OSError, subprocess.TimeoutExpired):
                worker.kill()
        outgoing.put(None)
        try:
            worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=3)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        worker_log.close()
    save_json(destination / "score.json", score)
    return score


def freeze(destination):
    destination.mkdir(parents=True, exist_ok=False)
    learned = json.loads(SOURCE.read_text(encoding="utf-8"))["result"]["candidate"]
    assert learned["source"] == "execution" and learned["evidence_ids"]
    candidate = deepcopy(learned)
    without_policy = deepcopy(candidate)
    without_policy["instructions"] = "\n".join(line for line in candidate["instructions"].splitlines()
                                              if not line.startswith("4."))
    cases = fixtures()
    payload = {"cases": cases, "variants": {"no_skill": {"instructions": ""},
        "grounded_skill": candidate, "without_empty_validation_rule": without_policy},
        "model": "deepseek/deepseek-v4.1-flash", "api_base": "https://api.commandcode.ai/provider/v1",
        "source_skill_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "scope": "Constructed same-family execution checks. Measurement selection case was studied before; two changed-name/option contracts are correlated held-out controls, not independent repository tasks. No public-benchmark score or statistical significance claim.",
        "promotion": "Research candidates remain inactive; semantic task success is required and a tie does not demonstrate utility.",
        "grader_visible_to_model": False, "reference_visible_to_model": False,
        "calls_per_cell": 1, "response_retries": 0, "max_output_tokens": 8192}
    for case in cases:
        positive = grade(case, case["GOLD"], destination / "calibration" / case["id"] / "gold")
        case["oracle_expected"] = positive["case_ids"]
        negative = grade(case, case["GOLD"].replace('    _policy(on_error)\n', ''),
                         destination / "calibration" / case["id"] / "missing_validation")
        assert positive["passed"] and not negative["passed"], (positive, negative)
    save_json(destination / "dataset.json", payload)
    save_json(destination / "manifest.json", {"sha256": sha256((destination / "dataset.json").read_bytes()).hexdigest()})
    print(json.dumps({"frozen_cases": len(cases), "variants": 3, "calibration": "passed"}), flush=True)


async def run(dataset, destination, *, selection_only=False):
    data = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == sha256((dataset / "dataset.json").read_bytes()).hexdigest()
    destination.mkdir(parents=True, exist_ok=False)
    settings = dotenv_values(ROOT / ".env")
    key = os.environ.get("SANS_BENCHMARK_API_KEY") or settings.get("OPENAI_API_KEY")
    assert key and str(settings.get("OPENAI_BASE_URL")).rstrip("/") == data["api_base"]
    for name, value in list(os.environ.items()):
        if key in value:
            os.environ.pop(name)
    trace = TraceRecorder(destination, run_id=destination.name, task_id="skill-transfer", attempt_id="1")
    selected = [case for case in data["cases"] if (case["split"] == "selection") == selection_only]
    order = [(case, arm) for case in selected for arm in data["variants"]]
    import random
    random.Random(20261009).shuffle(order)
    save_json(destination / "protocol.json", {"manifest_sha256": sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "order": [[case["id"], arm] for case, arm in order],
        "allocation": 100000, "model": data["model"], "scope": data["scope"], "source_activated": False})
    rows = []
    with ResearchBudget(destination, trace, 100000) as budget:
        model = _FoldingLoop(token_budget=budget.token_budget, before_model_call=budget.before_call,
            model=data["model"], api_base=data["api_base"], api_key=key, custom_system_prompt="Coding replay",
            custom_tools=[], skills_enabled=False, is_sub_agent=True, trace=trace,
            max_output_tokens=data["max_output_tokens"], max_input_tokens_per_call=32768,
            max_model_calls=len(order), model_call_timeout_seconds=120, thinking=True,
            openai_extra_body={"reasoning_effort": "low"})
        try:
            for case, arm in order:
                task = case["TASK"] + "\nUnchanged core source:\n" + case["CORE"]
                row = {"case": case["id"], "arm": arm, "split": case["split"]}
                try:
                    answer = await _generate_variant_response_async(variant={"snapshot": data["variants"][arm]},
                        sample={"messages": [{"role": "user", "content": task}]},
                        side_query=lambda system, user: model._side_query_once(system, user,
                            max_tokens=data["max_output_tokens"], purpose="skill_" + case["id"] + "_" + arm))
                    (destination / (case["id"] + "-" + arm + ".txt")).write_text(answer, encoding="utf-8")
                    row["score"] = grade(case, json.loads(answer)["code"], destination / "executions" / case["id"] / arm)
                except Exception as error:
                    row["score"] = {"completed": False, "passed": False, "error_type": type(error).__name__}
                rows.append(row)
                save_json(destination / "scores.json", rows)
                print(json.dumps(row), flush=True)
                if budget.token_budget.get("blocked"):
                    break
        finally:
            await model.shutdown()
        save_json(destination / "receipt.json", {"scores": rows, "usage": budget.settle(), "activated": False,
            "scheduled_cells": len(order), "attempted_cells": len(rows),
            "complete_matrix": len(rows) == len(order) and all(row["score"]["completed"] for row in rows)})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["freeze", "selection", "test"])
    parser.add_argument("--dataset", type=Path, default=RESEARCH / "skill-transfer-v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.dataset)
    else:
        if not args.out or not args.out.resolve().is_relative_to((ROOT / "experiment").resolve()):
            parser.error("--out must be a new directory within experiment/")
        asyncio.run(run(args.dataset, args.out, selection_only=args.command == "selection"))


if __name__ == "__main__":
    main()
