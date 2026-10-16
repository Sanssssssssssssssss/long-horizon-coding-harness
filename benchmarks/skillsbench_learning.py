"""Freeze train-only execution facts; optionally distill one inactive research Skill with one RPC."""
from __future__ import annotations

import argparse
import ast
import asyncio
import builtins
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import symtable
import tarfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.frontmatter import format_frontmatter
from agents.online_skill_evolution import extract_online_skill_candidate
from agents.telemetry import TraceRecorder
from benchmarks.context_folding import _FoldingLoop
from benchmarks.research_budget import ResearchBudget, save_json

ALLOCATE = 50000
ALLOWED_IMPORTS = {"re", "json", "typing", "collections", "dataclasses", "copy", "functools", "itertools", "__future__"}
GRAMMAR_CONSTANTS = {"End", "nodes", "edges", "id", "text", "speaker", "type", "from", "to", "line", "choice"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path, root=ROOT / "experiment"):
    path = Path(path).resolve()
    path.relative_to(root.resolve())
    if path.is_symlink():
        raise ValueError("Input may not be a symbolic link")
    return path


def checked(path, expected):
    path = artifact(path)
    if not expected or digest(path) != expected:
        raise ValueError(f"Frozen artifact changed: {path}")
    return path


def read_blob(run, reference, files):
    path = checked(run / "blobs" / reference["sha256"], reference["sha256"])
    if path.stat().st_size != reference["bytes"]:
        raise ValueError("Blob byte count changed")
    name = path.relative_to(ROOT).as_posix()
    seen = name in files
    files[name] = reference["sha256"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if seen:
        return value
    def nested(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key.endswith("_ref") and isinstance(child, dict) and {"sha256", "bytes"} <= set(child):
                    read_blob(run, child, files)
                else:
                    nested(child)
        elif isinstance(item, list):
            for child in item:
                nested(child)
    nested(value)
    return value


def load_run(run):
    run = artifact(run)
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("split") != "train" or receipt.get("status") not in {"completed", "interrupted"}
            or receipt.get("source_unchanged") is not True or receipt.get("bank_unchanged") is not True):
        raise ValueError("Only ended, unchanged train runs may be learned from")
    grade = receipt.get("grade") or {}
    if grade.get("valid_evaluation") is not True or grade.get("source_integrity") is not True:
        raise ValueError("A complete train evaluation with unchanged input is required")
    protocol_path = checked(run / "protocol.json", receipt["protocol_sha256"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    case = protocol["cases"]["train"]
    instruction_path = checked(ROOT / case["instruction"], case["instruction_sha256"])
    if "train" not in instruction_path.name:
        raise ValueError("Training instruction must be its frozen train file")
    bank_path = checked(run / "bank-manifest.json", receipt["bank_manifest_sha256"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("split") != "train" or bank.get("files") or bank.get("source") is not None:
        raise ValueError("Teacher may not contain canonical or human Skills")
    archive_path = checked(ROOT / protocol["source_archive"], protocol["source_archive_sha256"])
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(protocol["source_files"]):
            raise ValueError("Source archive member set changed")
        for name, sha in protocol["source_files"].items():
            if not re.fullmatch(r"(?:agents|benchmarks)/[^/]+\.py", name) or hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("Frozen runtime source changed")
    grade_path = run / "grading/score.json"
    if json.loads(grade_path.read_text(encoding="utf-8")) != grade:
        raise ValueError("Train grading receipt differs from the run")
    full_path, reward_path = run / "grading/pytest-cases.json", run / "grading/reward.txt"
    full = json.loads(full_path.read_text(encoding="utf-8"))
    if (full["cases"] != grade["cases"] or full["exitstatus"] != grade["pytest_exitstatus"]
            or full["collected"] != [case["name"] for case in full["cases"]]
            or float(reward_path.read_text()) != grade["reward"]):
        raise ValueError("Final train grade differs from its execution artifacts")
    if grade.get("resolved") and (grade["reward"] != 1 or full["exitstatus"] != 0 or any(case["status"] != "passed" for case in full["cases"])):
        raise ValueError("Successful teacher lacks complete passing evidence")
    snapshot = checked(ROOT / receipt["snapshot"], grade["snapshot_sha256"])
    artifact(snapshot, run)
    with tarfile.open(snapshot) as archive:
        members = {}
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name.startswith("app/"):
                name = name[4:]
            if member.isfile() and name in {"script.txt", "solution.py"}:
                if name in members:
                    raise ValueError("Duplicate snapshot input")
                members[name] = archive.extractfile(member).read()
    script = members.get("script.txt", b"")
    if hashlib.sha256(script).hexdigest() != case["input_sha256"] or grade.get("candidate_input_sha256") != case["input_sha256"]:
        raise ValueError("Teacher changed the train input")
    trace_path = run / "trace.jsonl"
    raw = trace_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Ended trace has a truncated tail")
    events = [json.loads(line) for line in raw.splitlines() if line]
    if not events or events[-1]["event_type"] != "agent.run_once.end":
        raise ValueError("Train trace has no terminal lifecycle event")
    files = {path.relative_to(ROOT).as_posix(): digest(path) for path in
             (receipt_path, protocol_path, instruction_path, bank_path, archive_path, grade_path, full_path, reward_path, snapshot, trace_path)}
    bodies = {event["seq"]: read_blob(run, event["payload_ref"], files)
              for event in events if event.get("payload_ref")}
    expected_scope = tuple(events[0][key] for key in ("run_id", "task_id", "attempt_id"))
    if expected_scope[:2] != (run.name, "skillsbench/dialogue-parser/train") or not isinstance(expected_scope[2], str) or not expected_scope[2]:
        raise ValueError("Trace is not a scoped training execution")
    if any(tuple(event[key] for key in ("run_id", "task_id", "attempt_id")) != expected_scope for event in events):
        raise ValueError("Trace crosses the training execution scope")
    requests = [event for event in events if event["event_type"] == "model.call.start"]
    if not requests or any(not event.get("payload_ref") or (event.get("attributes") or {}).get("model") != protocol["model"] for event in requests):
        raise ValueError("Model request provenance is incomplete")
    chat_inputs = [bodies[event["seq"]].get("user_message") for event in events if event["event_type"] == "agent.chat.start"]
    goal = instruction_path.read_text(encoding="utf-8")
    def strings(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [item for child in value.values() for item in strings(child)]
        if isinstance(value, list):
            return [item for child in value for item in strings(child)]
        return []
    request_texts = [text for event in requests for text in strings(bodies[event["seq"]])]
    if not any(goal in value for value in [*chat_inputs, *request_texts] if isinstance(value, str)):
        raise ValueError("Actual model request does not contain the frozen train goal")
    starts = {event["span_id"]: event for event in events if event["event_type"] == "tool.call.start"}
    facts = []
    for event in events:
        body = bodies.get(event["seq"], {})
        execution = body.get("execution")
        if event["event_type"] != "tool.call.end":
            continue
        observation = None
        if isinstance(execution, dict):
            if (type(execution.get("timed_out")) is not bool or execution.get("exit_code") is not None and type(execution["exit_code"]) is not int
                    or any(not isinstance(execution.get(key), str) for key in ("stdout", "stderr"))):
                raise ValueError("Malformed typed execution fact")
        else:
            match = re.match(r"^<tool_observation>(.*?)</tool_observation>\n?(.*)$", str(body.get("output") or ""), re.S)
            if match is None:
                continue
            metadata = json.loads(match[1])
            if metadata.get("tool") != (event.get("attributes") or {}).get("tool_name") or metadata.get("status") != event["status"]:
                raise ValueError("Typed observation disagrees with tool end")
            observation = {"metadata": metadata, "text": match[2]}
        start = starts.get(event["span_id"])
        if start is None or start["seq"] >= event["seq"]:
            raise ValueError("Tool end lacks a matching request")
        facts.append({"id": ":".join(expected_scope) + ":tool-" + str(event["seq"]),
                      "tool": (event.get("attributes") or {}).get("tool_name"), "status": event["status"],
                      "execution": execution, "input": bodies.get(start["seq"], {}).get("input"),
                      "observation": observation,
                      "operation_executed": event["status"] == "ok" and (event.get("attributes") or {}).get("primitive_operation") is True,
                      "process_executed": isinstance(execution, dict) and (type(execution.get("exit_code")) is int or execution.get("timed_out") is True),
                      "observation_ref": event["payload_ref"], "request_ref": start["payload_ref"]})
    facts.append({"id": ":".join(expected_scope) + ":official-train-grade", "tool": "official_train_verifier",
                  "status": "passed" if grade["resolved"] else "failed", "reward": grade["reward"],
                  "summary": grade["summary"], "cases": grade["cases"], "grade_ref": files[grade_path.relative_to(ROOT).as_posix()],
                  "scope": "Terminal train task evaluation; distinct from individual tool outcomes"})
    return {"run": run.relative_to(ROOT).as_posix(), "files": files, "protocol": protocol,
            "fresh_arm": receipt.get("fresh_arm", True),
            "goal": goal, "facts": facts,
            "resolved": grade["resolved"],
            "nodes": sorted(set(re.findall(r"^\[([^\]\r\n]+)\]\s*$", script.decode("utf-8"), re.M))),
            "solution": members.get("solution.py") if grade["resolved"] else None,
            "solution_ref": {"snapshot": snapshot.relative_to(ROOT).as_posix(), "snapshot_sha256": grade["snapshot_sha256"],
                             "member": "solution.py", "sha256": hashlib.sha256(members["solution.py"]).hexdigest()} if "solution.py" in members else None}


def request_packet(runs):
    goals = {run["goal"] for run in runs}
    if len(goals) != 1:
        raise ValueError("Learning inputs must share one frozen train goal")
    facts = []
    for run in runs:
        successful_writes = [fact for fact in run["facts"] if fact["tool"] in {"write_file", "edit_file"} and fact.get("operation_executed")][:2]
        selected = {fact["id"]: fact for fact in [*successful_writes, *run["facts"][-7:]]}
        for fact in selected.values():
            packed = dict(fact)
            if isinstance(packed.get("execution"), dict):
                original = packed["execution"]
                execution = {key: original.get(key) for key in ("exit_code", "timed_out", "error", "stdout", "stderr")}
                report = original.get("pytest_report")
                if isinstance(report, dict):
                    execution["pytest_report"] = {key: report.get(key) for key in ("collected", "failure", "error", "skipped")}
                for field in ("stdout", "stderr"):
                    value = str(execution.get(field) or "")
                    execution[field] = value[:700]
                    execution[field + "_omitted_chars"] = max(0, len(value) - 700)
                packed["execution"] = execution
            if isinstance(packed.get("observation"), dict):
                value = packed["observation"]["text"]
                packed["observation"] = {"metadata": packed["observation"]["metadata"], "text": value[:700],
                                         "omitted_chars": max(0, len(value) - 700)}
            if "input" in packed:
                packed["input"] = json.dumps(packed["input"], ensure_ascii=False)[:600]
            facts.append(packed)
    return {"messages": [{"role": "user", "content": next(iter(goals))}], "execution_evidence": facts,
            "hint": "Learn only from these train execution facts. Harness rejection without an exit code is not process execution. "
                    "Use source=execution and cite evidence_ids. Preserve failure-only scope when no train task succeeded. "
                    "Abstract all story identifiers, answers and runtime paths; do not assume general task improvement."}


def generic_helper(source, nodes):
    if source is None:
        return None
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError):
        return None
    if any(node not in GRAMMAR_CONSTANTS and node in text for node in nodes):
        return None
    if any(marker in text for marker in ("/tests", "/solution/", "/logs/verifier", "pytest.skip", "eval(", "exec(")):
        return None
    retained = []
    for item in tree.body:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in item.names] if isinstance(item, ast.Import) else [(item.module or "").split(".")[0]]
            if any(module not in ALLOWED_IMPORTS for module in modules):
                return None
            retained.append(item)
        elif isinstance(item, (ast.FunctionDef, ast.ClassDef)):
            retained.append(item)
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            value = item.value
            try:
                ast.literal_eval(value)
                safe = len(ast.unparse(value)) < 4096
            except (ValueError, TypeError):
                safe = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                        and isinstance(value.func.value, ast.Name) and value.func.value.id == "re"
                        and value.func.attr == "compile" and all(isinstance(arg, (ast.Constant, ast.Attribute)) for arg in value.args))
            if safe:
                retained.append(item)
    if not any(isinstance(item, ast.FunctionDef) and item.name == "parse_script" for item in retained):
        return None
    helper = ast.unparse(ast.Module(body=retained, type_ignores=[])) + "\n"
    symbols = symtable.symtable(helper, "learned_parser.py", "exec")
    bound = {symbol.get_name() for symbol in symbols.get_symbols() if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()} | set(dir(builtins))
    def unresolved(table):
        return any(symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in bound for symbol in table.get_symbols()) or any(unresolved(child) for child in table.get_children())
    return None if unresolved(symbols) else helper


def verify_helper(bank, image, output):
    """Import and exercise only a constructed new string in an owned, offline container."""
    import importlib.util
    path = ROOT / "benchmarks/skillsbench_prepare.py"
    spec = importlib.util.spec_from_file_location("learner_preparer", path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    name = "skill-helper-check-" + uuid.uuid4().hex[:12]
    code = "import learned_parser,json,uuid; a='Probe_'+uuid.uuid4().hex; b='Probe_'+uuid.uuid4().hex; text=f'[{a}]\\nGuide: Continue. -> {b}\\n[{b}]\\n1. Finish. -> End\\n'; value=learned_parser.parse_script(text); value=value.to_dict() if hasattr(value,'to_dict') else value; assert {a,b}=={item['id'] for item in value['nodes']}; assert any(item['from']==a and item['to']==b for item in value['edges']); assert any(item['from']==b and item['to']=='End' for item in value['edges']); print(json.dumps({'imported':learned_parser.__file__,'constructed_input_passed':True}))"
    try:
        result = prepare.docker(["run", "--name", name, "--network", "none", "--cpus", "2", "--memory", "4g",
            "--mount", f"type=bind,source={prepare.linux(bank)},target=/bank,readonly", "-e", "PYTHONPATH=/bank/scripts",
            "-e", "PYTHONDONTWRITEBYTECODE=1", image, "python", "-c", code])
        (output / "helper-check.json").write_text(result.stdout, encoding="utf-8")
        return True
    except Exception as error:
        save_json(output / "helper-check.json", {"passed": False, "error_type": type(error).__name__, "reason": str(error)[:1000]})
        return False
    finally:
        prepare.docker(["rm", "-f", name], check=False)


async def main(args, side_query=None):
    output = artifact(args.out)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "preparing", "activated": False, "model_api_calls": 0, "scope": "Train-only research Skill, not a canonical task result"}
    save_json(output / "receipt.json", receipt)
    agent = None
    try:
        runs = [load_run(path) for path in args.train_run]
        if len({run["run"] for run in runs}) != len(runs):
            raise ValueError("Duplicate training run")
        if len({run["protocol"]["model"] for run in runs}) != 1:
            raise ValueError("Training and reuse require the same model")
        packet = request_packet(runs)
        save_json(output / "request.json", packet)
        save_json(output / "evidence.json", [fact for run in runs for fact in run["facts"]])
        manifest = {"runs": [{"run": run["run"], "files": run["files"], "fresh_arm": run["fresh_arm"]} for run in runs],
                    "request_sha256": digest(output / "request.json"), "evidence_sha256": digest(output / "evidence.json"),
                    "learner_sha256": digest(__file__), "activated": False,
                    "distillation_source_files": {name: digest(ROOT / name) for name in
                    ("agents/online_skill_evolution.py", "agents/agent.py", "benchmarks/context_folding.py", "benchmarks/research_budget.py")}}
        save_json(output / "source-manifest.json", manifest)
        receipt.update(status="prepared", source_manifest_sha256=digest(output / "source-manifest.json"))
        if not args.extract:
            return receipt
        trace = TraceRecorder(output, run_id=output.name, task_id="skillsbench/dialogue-parser/learning", attempt_id="001")
        with ResearchBudget(output, trace, ALLOCATE) as budget:
            if side_query is None:
                from dotenv import dotenv_values
                settings = dotenv_values(ROOT / ".env")
                protocol = runs[0]["protocol"]
                if str(settings.get("OPENAI_BASE_URL", "")).rstrip("/") != protocol["api_base"]:
                    raise ValueError("Frozen model route changed")
                key = os.environ.get("SANS_BENCHMARK_API_KEY") or settings.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError("Missing local model credential")
                agent = _FoldingLoop(token_budget=budget.token_budget, before_model_call=budget.before_call,
                    model=protocol["model"], api_base=protocol["api_base"], api_key=key, thinking=True,
                    custom_system_prompt="Train-only skill distillation", custom_tools=[], skills_enabled=False,
                    is_sub_agent=True, trace=trace, max_model_calls=1, model_retry_limit=0,
                    max_output_tokens=8192, max_input_tokens_per_call=32768, model_call_timeout_seconds=120)
                async def side_query(system, user):
                    return await agent._side_query_once(system, user, max_tokens=8192, purpose="skill_train_distill", disable_thinking=True)
            try:
                candidate = await extract_online_skill_candidate(**packet, side_query=side_query)
            finally:
                if agent is not None:
                    await agent.shutdown()
                receipt["usage"] = budget.settle()
                receipt["model_api_calls"] = receipt["usage"]["physical_requests"]
        if candidate is None:
            receipt["status"] = "no_candidate"
            return receipt
        save_json(output / "candidate.json", asdict(candidate))
        nodes = {node for run in runs for node in run["nodes"]} - GRAMMAR_CONSTANTS
        content = "\n".join((candidate.name, candidate.description, candidate.when_to_use, candidate.instructions))
        if candidate.source != "execution" or any(node in content for node in nodes):
            raise ValueError("Candidate used an untrusted source label or retained training identifiers")
        bank = output / "bank/dialogue-parser-experience"
        bank.mkdir(parents=True)
        helper_run = next((run for run in runs if generic_helper(run["solution"], run["nodes"])), None)
        helper = generic_helper(helper_run["solution"], helper_run["nodes"]) if helper_run else None
        helper_status = "no_successful_solution_module" if not any(run["solution"] for run in runs) else "generic_or_dependency_guard_rejected"
        instructions = candidate.instructions
        if helper:
            (bank / "scripts").mkdir()
            (bank / "scripts/learned_parser.py").write_text(helper, encoding="utf-8")
            if verify_helper(bank, helper_run["protocol"]["cases"]["train"]["image"], output):
                helper_status = "passed_constructed_input"
                instructions += "\n\nA helper derived from a successful train execution is in ${CLAUDE_SKILL_DIR}/scripts/learned_parser.py. Import learned_parser and provide text to parse_script at runtime."
            else:
                (output / "rejected-helper.py").write_text(helper, encoding="utf-8")
                (bank / "scripts/learned_parser.py").unlink()
                helper = None
                helper_status = "constructed_input_check_failed"
        if not helper:
            instructions += "\n\nThis bank contains procedural lessons only. No reusable executable helper was established."
        if not any(run["resolved"] for run in runs):
            instructions += "\n\nThese are failure lessons from unsuccessful train tasks; no successful solution procedure or transfer gain was established."
        (bank / "SKILL.md").write_text(format_frontmatter({"name": "dialogue-parser-experience", "description": candidate.description,
            "when-to-use": candidate.when_to_use, "context": "inline", "user-invocable": "false", "source": "execution"}, instructions), encoding="utf-8")
        receipt.update(status="bank_frozen", bank=bank.relative_to(ROOT).as_posix(), helper=bool(helper),
                       helper_status=helper_status, helper_origin=helper_run["solution_ref"] if helper_run else None,
                       evidence_ids=candidate.evidence_ids,
                       bank_files={path.relative_to(bank).as_posix(): digest(path) for path in bank.rglob("*") if path.is_file()})
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)[:1500]})
        raise
    finally:
        save_json(output / "receipt.json", receipt)
        print(json.dumps({key: receipt.get(key) for key in ("status", "activated", "model_api_calls", "bank", "error")}))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--train-run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extract", action="store_true", help="Explicitly spend at most one extraction RPC and a 50,000-token allocation")
    asyncio.run(main(parser.parse_args()))
