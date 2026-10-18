"""Execution facts stay structured until the model-facing rendering boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class PytestReport:
    collected: int
    failure: int = 0
    error: int = 0
    skipped: int = 0
    cases: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def read(cls, path: Path) -> "PytestReport | None":
        try:
            return cls.parse(path.read_bytes())
        except OSError:
            return None

    @classmethod
    def parse(cls, xml: str | bytes) -> "PytestReport | None":
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return None
        if root.tag not in {"testsuite", "testsuites"}:
            return None
        cases = list(root.iter("testcase"))
        outcomes = []
        for case in cases:
            status, signature = "passed", ""
            for tag in ("failure", "error", "skipped"):
                child = case.find(tag)
                if child is not None:
                    status = tag
                    message = child.get("message", "")
                    signature = sha256((child.get("type", "") + "\0" + message).encode()).hexdigest() if message else ""
                    break
            outcomes.append((case.get("classname", "") + "::" + case.get("name", ""), status, signature))
        return cls(len(cases), cases=tuple(outcomes), **{
            tag: sum(case.find(tag) is not None for case in cases)
            for tag in ("failure", "error", "skipped")
        })


@dataclass(frozen=True)
class ShellResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    error: str | None = None
    pytest_report: PytestReport | None = None
    baseline: ShellResult | None = None
    comparison_candidate: ShellResult | None = None
    baseline_ref: str | None = None
    comparison_error: str | None = None
    execution_started: bool | None = None

    def __post_init__(self) -> None:
        if self.execution_started is not None and type(self.execution_started) is not bool:
            raise ValueError("execution_started must be a boolean or None")

    @property
    def unchanged_failure(self) -> bool:
        base, current = self.baseline, self.pytest_report
        if not base or not self.baseline_ref or self.comparison_error:
            return False
        if self.test_outcome != "failed" or base.test_outcome != "failed":
            return False
        original = base.pytest_report
        if not current.cases or current.skipped or original.skipped:
            return False
        if any(c[1] == "failure" and not c[2] for c in current.cases):
            return False
        # Compare test identities and failure signatures, never just failure counts.
        if [c[0] for c in current.cases] != [c[0] for c in original.cases]:
            return False
        return all(now[1] == "passed" or now == old for old, now in zip(original.cases, current.cases))

    @property
    def failed(self) -> bool:
        return self.exit_code != 0 or self.timed_out or self.error is not None

    @property
    def test_outcome(self) -> str:
        report = self.pytest_report
        if report and report.collected and not self.timed_out and self.error is None and not report.error:
            if self.exit_code == 1 and report.failure:
                return "failed"
            if self.exit_code == 0 and not (report.failure or report.skipped):
                return "passed"
        return "inconclusive"

    def __str__(self) -> str:
        if self.error:
            prefix = self.error
        elif self.timed_out:
            prefix = "Command timed out"
        elif self.exit_code is None:
            prefix = "Error: command did not return an exit code"
        elif self.exit_code != 0:
            prefix = f"Command failed (exit code {self.exit_code})"
        else:
            prefix = ""
        stdout = f"\nStdout: {self.stdout}" if self.stdout else ""
        stderr = f"\nStderr: {self.stderr}" if self.stderr else ""
        output = prefix + stdout + stderr if prefix else self.stdout + self.stderr or "(no output)"
        if self.baseline:
            output += f"\n\nOriginal revision replay ({self.baseline_ref}):\n{self.baseline}"
        if self.comparison_error:
            output += f"\n\nComparison inconclusive: {self.comparison_error}"
        return output
