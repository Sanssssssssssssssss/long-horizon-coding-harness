"""Train-only distillation may not silently cross scopes or accept ungrounded artifacts."""
import asyncio
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skillsbench_learning", REPO / "experiment/research/20261009-skill-folding/skillsbench/learn.py")
learn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learn)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def teacher(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "ROOT", tmp_path)
    run = tmp_path / "experiment/runs/train-fixture"
    run.mkdir(parents=True)
    blob = run / "blobs"
    blob.mkdir()
    goal = tmp_path / "experiment/runs/protocol/train-instruction.md"
    goal.parent.mkdir(parents=True)
    goal.write_text("Implement a generic parser.", encoding="utf-8")
    source = tmp_path / "agents/agent.py"
    source.parent.mkdir()
    source.write_text("pass\n")
    for name in ("agents/online_skill_evolution.py", "benchmarks/context_folding.py", "benchmarks/research_budget.py"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("pass\n")
    archive = goal.parent / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(source, "agents/agent.py")
    script = b"[TrainingOnlyNode]\nPerson: Stop. -> End\n"
    snapshot = run / "snapshots/candidate.tar"
    snapshot.parent.mkdir()
    with tarfile.open(snapshot, "w") as packed:
        member = tarfile.TarInfo("script.txt")
        member.size = len(script)
        packed.addfile(member, io.BytesIO(script))
    protocol = {"model": "model-a", "api_base": "https://example.invalid", "cases": {"train": {
        "instruction": goal.relative_to(tmp_path).as_posix(), "instruction_sha256": learn.digest(goal),
        "input_sha256": hashlib.sha256(script).hexdigest(), "image": "unused"}, "canonical": {"instruction": "MUST_NOT_READ"}},
        "source_archive": archive.relative_to(tmp_path).as_posix(), "source_archive_sha256": learn.digest(archive),
        "source_files": {"agents/agent.py": learn.digest(source)}}
    save(run / "protocol.json", protocol)
    save(run / "bank-manifest.json", {"split": "train", "files": {}, "source": None})
    cases = [{"name": "check_contract", "status": "failed", "phases": {"call": "failed"}}]
    grade = {"valid_evaluation": True, "source_integrity": True, "resolved": False, "reward": 0.0,
             "summary": {"tests": 1, "failed": 1}, "cases": cases, "pytest_exitstatus": 1,
             "snapshot_sha256": learn.digest(snapshot), "candidate_input_sha256": hashlib.sha256(script).hexdigest()}
    save(run / "grading/score.json", grade)
    save(run / "grading/pytest-cases.json", {"cases": cases, "collected": ["check_contract"], "exitstatus": 1})
    (run / "grading/reward.txt").write_text("0")
    def event(seq, kind, body=None, span=None, **attrs):
        ref = None
        if body is not None:
            raw = json.dumps(body).encode()
            sha = hashlib.sha256(raw).hexdigest()
            (blob / sha).write_bytes(raw)
            ref = {"sha256": sha, "bytes": len(raw)}
        return {"seq": seq, "event_type": kind, "payload_ref": ref, "span_id": span, "status": "error",
                "run_id": run.name, "task_id": "skillsbench/dialogue-parser/train", "attempt_id": "001", "attributes": attrs}
    events = [event(1, "agent.chat.start", {"user_message": "Wrapper\n\n" + goal.read_text()}),
              event(2, "model.call.start", {"messages": []}, model="model-a"),
              event(3, "tool.call.start", {"input": {"command": "python check.py"}}, span="tool"),
              event(4, "tool.call.end", {"execution": {"exit_code": 1, "timed_out": False, "stdout": "", "stderr": "AssertionError"}}, span="tool", tool_name="run_shell"),
              event(5, "agent.run_once.end")]
    (run / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    save(run / "receipt.json", {"split": "train", "status": "completed", "source_unchanged": True, "bank_unchanged": True,
         "grade": grade, "snapshot": snapshot.relative_to(tmp_path).as_posix(), "protocol_sha256": learn.digest(run / "protocol.json"),
         "bank_manifest_sha256": learn.digest(run / "bank-manifest.json")})
    return run


def test_frozen_packet_uses_goal_execution_and_terminal_train_grade(teacher):
    loaded = learn.load_run(teacher)
    packet = learn.request_packet([loaded])
    assert packet["messages"] == [{"role": "user", "content": "Implement a generic parser."}]
    assert packet["execution_evidence"][0]["execution"]["exit_code"] == 1
    assert packet["execution_evidence"][-1]["tool"] == "official_train_verifier"
    assert loaded["solution"] is None


def test_successful_write_is_an_operation_fact_without_claiming_a_shell_process(teacher):
    rows = [json.loads(line) for line in (teacher / "trace.jsonl").read_text().splitlines()]
    terminal = rows.pop()
    def blob(value):
        data = json.dumps(value).encode()
        sha = hashlib.sha256(data).hexdigest()
        (teacher / "blobs" / sha).write_bytes(data)
        return {"sha256": sha, "bytes": len(data)}
    common = {key: terminal[key] for key in ("run_id", "task_id", "attempt_id")}
    rows += [{**common, "seq": 5, "event_type": "tool.call.start", "span_id": "write", "status": "running", "attributes": {},
              "payload_ref": blob({"input": {"file_path": "solution.py", "content": "implementation"}})},
             {**common, "seq": 6, "event_type": "tool.call.end", "span_id": "write", "status": "ok",
              "attributes": {"tool_name": "write_file", "primitive_operation": True},
              "payload_ref": blob({"output": '<tool_observation>{"tool":"write_file","status":"ok"}</tool_observation>\nFile written'})},
             {**terminal, "seq": 7}]
    (teacher / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    facts = learn.request_packet([learn.load_run(teacher)])["execution_evidence"]
    written = next(fact for fact in facts if fact["tool"] == "write_file")
    assert written["operation_executed"] is True and written["process_executed"] is False
    assert written["observation"]["text"] == "File written"


@pytest.mark.parametrize("poison", ["canonical", "unfinished", "changed_input", "changed_blob", "changed_source", "cross_scope"])
def test_scope_and_hash_failures_are_rejected_before_extraction(teacher, poison):
    receipt = json.loads((teacher / "receipt.json").read_text())
    if poison in {"canonical", "unfinished", "changed_input"}:
        if poison == "canonical": receipt["split"] = "canonical"
        if poison == "unfinished": receipt["status"] = "incomplete"
        if poison == "changed_input": receipt["grade"]["source_integrity"] = False
        save(teacher / "receipt.json", receipt)
    elif poison == "changed_blob":
        next((teacher / "blobs").iterdir()).write_text("changed")
    elif poison == "changed_source":
        protocol = json.loads((teacher / "protocol.json").read_text())
        (learn.ROOT / protocol["source_archive"]).write_bytes(b"changed")
    else:
        rows = [json.loads(line) for line in (teacher / "trace.jsonl").read_text().splitlines()]
        rows[3]["task_id"] = "skillsbench/dialogue-parser/canonical"
        (teacher / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        learn.load_run(teacher)


def test_helper_keeps_regex_constants_and_rejects_unbound_or_instance_answers():
    good = b"import re\nHEADER=re.compile(r'^\\[([^]]+)\\]$',re.M)\ndef parse_script(text):\n return {'nodes':HEADER.findall(text),'edges':[]}\n"
    assert "HEADER = re.compile" in learn.generic_helper(good, ["TrainingOnlyNode"])
    assert learn.generic_helper(b"def parse_script(text):\n return missing_global\n", []) is None
    assert learn.generic_helper(b"def parse_script(text):\n return 'TrainingOnlyNode'\n", ["TrainingOnlyNode"]) is None


@pytest.mark.parametrize("source", ["execution", "user"])
def test_one_mock_rpc_freezes_only_grounded_inactive_failure_bank(teacher, monkeypatch, source):
    class FakeBudget:
        def __init__(self, *args): assert args[-1] == 50000
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settle(self): return {"physical_requests": 0, "known_tokens": 0, "unknown_usage_requests": 0}
    monkeypatch.setattr(learn, "ResearchBudget", FakeBudget)
    calls = []
    async def query(_system, body):
        calls.append(body)
        evidence = json.loads(body)["execution_evidence"][-1]["id"]
        return json.dumps({"skills": [{"name": "parser-workflow", "description": "Validate parser artifacts",
            "instructions": "Write and inspect both output artifacts before declaring the task complete.",
            "source": source, "evidence_ids": [evidence]}]})
    out = learn.ROOT / ("experiment/research/learning-" + source)
    args = SimpleNamespace(train_run=[teacher], out=out, extract=True)
    if source == "user":
        with pytest.raises(ValueError, match="source label"):
            asyncio.run(learn.main(args, query))
        assert not (out / "bank").exists()
    else:
        result = asyncio.run(learn.main(args, query))
        assert result["status"] == "bank_frozen" and result["activated"] is False
        assert "failure lessons" in (learn.ROOT / result["bank"] / "SKILL.md").read_text()
        assert not result["helper"]
    assert len(calls) == 1
