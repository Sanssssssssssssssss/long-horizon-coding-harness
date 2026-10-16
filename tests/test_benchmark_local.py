"""The paid entry must stay closed when its environment evidence is stale."""
import json
import pytest

import benchmarks.local as local


def test_live_rejects_stale_environment_before_harbor(tmp_path, monkeypatch):
    case = tmp_path / "case"
    case.mkdir()
    proof = tmp_path / "check.xml"
    proof.write_text("frozen preflight evidence")
    current = {"image_id": "sha256:original", "task_files": {"instruction.md": "original"}}
    ready = {"status": "ready", "identity": current, "checked_modes": ["nop", "oracle"],
             "check_files": {str(proof): local.digest(proof)}}
    (case / "ready.json").write_text(json.dumps(ready))
    monkeypatch.setattr(local, "CACHE", tmp_path)
    monkeypatch.setattr(local, "identity", lambda _: current)
    monkeypatch.setenv(local.SECRET_ENV, "unused-test-key")
    monkeypatch.setattr(local, "harbor", lambda *args: pytest.fail("paid runner must not start"))
    local.validate_ready(ready, current)
    for field, value in [("status", "failed"), ("checked_modes", ["nop"]), ("check_files", {}),
                         ("identity", {"image_id": "changed"})]:
        (case / "ready.json").write_text(json.dumps({**ready, field: value}))
        with pytest.raises(ValueError):
            local.live(case, tmp_path / "paid")
    (case / "ready.json").write_text(json.dumps(ready))
    proof.write_text("modified preflight evidence")
    with pytest.raises(ValueError, match="evidence changed"):
        local.live(case, tmp_path / "paid")
    assert not (tmp_path / "paid").exists()


def test_network_choice_handles_host_network_and_avoids_overlap(monkeypatch):
    def output(argv, **kwargs):
        if argv[2] == "ls":
            return "host\nbridge\n"
        return json.dumps([{"IPAM": {"Config": None}}, {"IPAM": {"Config": [{"Subnet": "10.223.0.0/24"}]}}])
    monkeypatch.setattr(local.subprocess, "check_output", output)
    assert local.subnet() == "10.223.1.0/24"


def test_statement_matches_pinned_utf8_even_when_mojibake_is_valid_unicode(tmp_path, monkeypatch):
    text = "Uppercase \u0401 \u0414 \u03a7"
    statement = tmp_path / "instruction.md"
    statement.write_text(text, encoding="utf-8")
    (tmp_path / "case.json").write_text(json.dumps({"image": "unused", "instruction_sha256": local.digest(statement)}))
    monkeypatch.setattr(local, "image_id", lambda _: "sha256:cached")
    assert local.identity(tmp_path)["image_id"] == "sha256:cached"
    statement.write_text(text.encode("utf-8").decode("latin1"), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned UTF-8"):
        local.identity(tmp_path)


def test_interrupted_run_keeps_usage_receipt(tmp_path, monkeypatch):
    case = tmp_path / "case"
    case.mkdir()
    proof = tmp_path / "check.xml"
    proof.write_text("frozen")
    current = {"image_id": "cached", "task_files": {}}
    (case / "ready.json").write_text(json.dumps({"status": "ready", "identity": current,
        "checked_modes": ["nop", "oracle"], "check_files": {str(proof): local.digest(proof)}}))
    monkeypatch.setattr(local, "CACHE", tmp_path)
    monkeypatch.setattr(local, "identity", lambda _: current)
    monkeypatch.setenv(local.SECRET_ENV, "unused-test-key")
    monkeypatch.setattr(local.subprocess, "check_output", lambda *a, **kw: "commit")
    def interrupted(task, mode, output, frozen):
        output.mkdir()
        raise RuntimeError("interrupted")
    monkeypatch.setattr(local, "harbor", interrupted)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        local.live(case, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["model_calls"] == 0
    assert not receipt["natural_completion"]
    assert "unused-test-key" not in (output / "receipt.json").read_text()
