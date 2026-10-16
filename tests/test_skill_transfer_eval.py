"""Generated adapter code cannot replace the trusted pytest assertions."""
import time
import pytest

from benchmarks import skill_transfer_eval as evaluation
from benchmarks.skill_transfer_eval import fixtures, grade


@pytest.mark.parametrize("index", range(3))
def test_all_frozen_contracts_pass_gold_through_one_json_worker(tmp_path, index):
    case = fixtures()[index]
    score = grade(case, case["GOLD"], tmp_path / case["id"])
    assert score["completed"] and score["passed"]
    assert score["checks_passed"] == score["checks_total"] == 17
    assert score["worker_processes"] == 1 and score["worker_calls"] == 17
    assert not score["candidate_imported_by_pytest"] and score["harness_unchanged"]
    assert score["core_unchanged"] and score["actual_core_sha256"] == score["core_sha256"]


def test_candidate_cannot_register_hooks_in_trusted_pytest(tmp_path):
    code = '''import inspect
class Bypass:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            item._obj = lambda **kwargs: None
for frame in inspect.stack():
    instance = frame.frame.f_locals.get('self')
    config = getattr(instance, 'config', None)
    if config is not None and hasattr(config, 'pluginmanager'):
        config.pluginmanager.register(Bypass(), 'adapter-bypass')
        break
def decode_many(*args, **kwargs):
    raise RuntimeError('deliberately broken')
def decode_records(*args, **kwargs):
    raise RuntimeError('deliberately broken')
'''
    score = grade(fixtures()[0], code, tmp_path / "exploit")
    assert score["completed"] and not score["passed"]
    assert score["checks_passed"] < 17 and score["harness_unchanged"]


@pytest.mark.parametrize("change", ["empty_policy", "shallow_copy", "input_mutation"])
def test_json_boundary_does_not_hide_missing_validation_or_mutable_aliases(tmp_path, change):
    case = fixtures()[0]
    code = case["GOLD"]
    if change == "empty_policy":
        code = code.replace("    _policy(on_error)\n", "")
    elif change == "shallow_copy":
        code = code.replace("deepcopy(record)", "dict(record)")
    else:
        code = code.replace("    _policy(on_error)\n    out = []\n    for record", "    _policy(on_error)\n    records[0]['changed'] = True\n    out = []\n    for record")
    score = grade(case, code, tmp_path / change)
    assert not score["passed"]


def test_worker_failure_is_not_a_candidate_runtime_exception_pass(tmp_path):
    score = grade(fixtures()[0], "this is invalid python !!!", tmp_path / "broken-worker")
    assert not score["completed"] and not score["passed"]
    assert score["checks_passed"] == 0 and score["transport_errors"]


def test_candidate_cannot_rewrite_the_supplied_core_even_when_checks_pass(tmp_path):
    case = fixtures()[0]
    changed = case["CORE"] + "\n# Changed by the candidate\n"
    code = "from pathlib import Path\nPath(__file__).with_name('core.py').write_text(" + repr(changed) + ",encoding='utf-8')\n" + case["GOLD"]
    score = grade(case, code, tmp_path / "changed-core")
    assert score["checks_passed"] == 17
    assert not score["core_unchanged"] and not score["passed"]


def test_timeout_does_not_wait_for_a_child_holding_worker_stdout(tmp_path, monkeypatch):
    code = '''import subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(6)'],stdout=sys.__stdout__)
Path('child.pid').write_text(str(child.pid))
def decode_many(*args,**kwargs):
    while True: time.sleep(1)
def decode_records(*args,**kwargs):
    while True: time.sleep(1)
'''
    original = evaluation.subprocess.run
    def limited(*args, **kwargs):
        if "pytest" in args[0]:
            kwargs["timeout"] = 1
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation.subprocess, "run", limited)
    started = time.monotonic()
    score = grade(fixtures()[0], code, tmp_path / "child-timeout")
    assert not score["completed"] and not score["passed"]
    assert score["error_type"] == "TimeoutExpired"
    assert time.monotonic() - started < 4
