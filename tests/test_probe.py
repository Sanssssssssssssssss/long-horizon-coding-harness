import pytest

from agents.tool_result import ShellResult
from benchmarks.probe import run_probe, summarize


@pytest.mark.parametrize("execution,xml,expected", [
    (ShellResult("999 passed", exit_code=0), "", "inconclusive"),
    (ShellResult(exit_code=0), "<testsuites><testsuite/></testsuites>", "inconclusive"),
    (ShellResult(exit_code=0), "<testsuite><testcase/></testsuite>", "passed"),
    (ShellResult(exit_code=1), "<testsuite><testcase><failure/></testcase></testsuite>", "failed"),
    (ShellResult(exit_code=1), "<testsuite><testcase><error/></testcase></testsuite>", "inconclusive"),
    (ShellResult(exit_code=0), "<testsuite><testcase><failure/></testcase></testsuite>", "inconclusive"),
    (ShellResult(exit_code=0), "<testsuite><testcase><skipped/></testcase></testsuite>", "inconclusive"),
    (ShellResult(timed_out=True), "<testsuite><testcase/></testsuite>", "inconclusive"),
])
def test_only_completed_pytest_assertions_are_classified(tmp_path, execution, xml, expected):
    report = tmp_path / "junit.xml"
    report.write_text(xml)
    assert summarize(execution, report)["status"] == expected


@pytest.mark.parametrize("label,timeout", [("probe", 1), ("../escape", 1), ("case", float("nan")), ("case", 0)])
def test_invalid_inputs_fail_before_execution(tmp_path, label, timeout):
    with pytest.raises(ValueError):
        run_probe(images={label: "unused"}, check=tmp_path / "missing.py",
                  output=tmp_path / "out", timeout=timeout)
    assert not (tmp_path / "out").exists()
