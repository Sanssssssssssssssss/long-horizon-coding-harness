"""Run one self-contained pytest probe against frozen, read-only Docker images.

This records observations; it does not decide whether a patch satisfies an issue.
Images must already contain the repository, its dependencies, and pytest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
import uuid

from agents.tool_result import PytestReport, ShellResult
from .runner import experiment_path


def summarize(result: ShellResult, report: Path) -> dict:
    """A printed 'passed' or an exit code alone is not an executed test result."""
    counts = PytestReport.read(report)
    if counts is None:
        return {"status": "inconclusive", "reason": "missing or invalid pytest XML"}
    return {"status": replace(result, pytest_report=counts).test_outcome, **asdict(counts)}


def run_probe(*, images: dict[str, str], check: Path, output: Path,
              workdir: str = "/testbed", python: str = "python", timeout: float = 120) -> dict:
    if not images or any(name == "probe" or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in images):
        raise ValueError("image labels must be simple names other than reserved 'probe'")
    if not workdir.startswith("/") or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("workdir must be absolute and timeout must be positive")
    check, output = check.resolve(), output.resolve()
    if "," in str(output):
        raise ValueError("output path cannot contain commas (Docker mount syntax)")
    source = check.read_bytes()
    # Resolve tags once before any execution. Never pull or rebuild an input image.
    frozen = {}
    for label, image in images.items():
        resolved = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", resolved):
            raise ValueError(f"invalid image identity for {label}")
        frozen[label] = resolved
    output.mkdir(parents=True, exist_ok=False)
    probe_dir = output / "probe"
    probe_dir.mkdir()
    (probe_dir / "test_probe.py").write_bytes(source)
    receipt = {
        "probe_sha256": hashlib.sha256(source).hexdigest(), "images": frozen,
        "workdir": workdir, "python": python, "timeout_seconds": timeout,
        "network": "none", "rootfs": "read-only", "runs": {},
        "interpretation": "Test outcomes only; semantic validity requires independent calibration.",
    }
    receipt_path = output / "result.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for label, image in frozen.items():
        report_dir = output / label
        report_dir.mkdir()
        container = "harness-probe-" + uuid.uuid4().hex
        command = [
            "docker", "run", "--name", container, "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,exec", "--workdir", workdir,
            "--mount", f"type=bind,src={probe_dir},dst=/probe,readonly",
            "--mount", f"type=bind,src={report_dir},dst=/probe-results",
            # ponytail: self-contained pytest only; wire plugins when a real probe requires them.
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            "--env", f"PYTHONPATH={workdir}", "--entrypoint", python, image,
            "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
            "--junitxml=/probe-results/junit.xml", "/probe/test_probe.py",
        ]
        started = time.monotonic()
        try:
            process = subprocess.run(command, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=timeout)
            result = ShellResult(process.stdout, process.stderr, process.returncode)
        except subprocess.TimeoutExpired as error:
            def text(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
            result = ShellResult(text(error.stdout), text(error.stderr), timed_out=True)
        finally:
            # Killing only the Docker client leaves the test running. Remove our container too.
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30)
        receipt["runs"][label] = {
            "execution": asdict(result), "duration_seconds": round(time.monotonic() - started, 3),
            "pytest": summarize(result, report_dir / "junit.xml"), "command": command,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, metavar="LABEL=IMAGE")
    parser.add_argument("--check", type=Path, required=True, help="Self-contained pytest file")
    parser.add_argument("--output", type=experiment_path, required=True, help="New directory under experiment/; never overwritten")
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--python", default="python", help="Interpreter inside each image")
    parser.add_argument("--timeout", type=float, default=120, help="Seconds per check execution")
    args = parser.parse_args()
    images = {}
    for entry in args.image:
        label, separator, image = entry.partition("=")
        if not separator or not image or label in images:
            parser.error("--image requires distinct LABEL=IMAGE entries")
        images[label] = image
    receipt = run_probe(images=images, check=args.check, output=args.output,
                        workdir=args.workdir, python=args.python, timeout=args.timeout)
    for label, run in receipt["runs"].items():
        print(f"{label}: {run['pytest']['status']}")
    print(f"Receipt: {args.output / 'result.json'}")


if __name__ == "__main__":
    main()
