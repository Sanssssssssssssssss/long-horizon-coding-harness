"""Paused Docker workspace snapshots and isolated grading for small Python cases."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
from types import SimpleNamespace
import uuid

from agents.tool_result import PytestReport, ShellResult


def docker(*args, **kwargs):
    return subprocess.run(["docker", *map(str, args)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=90, check=True, **kwargs)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_tar(source, destination):
    """Keep ignored/untracked files too; reject unsupported entries rather than lose them."""
    with tarfile.open(source) as incoming, tarfile.open(destination, "w") as outgoing:
        seen = set()
        for member in sorted(incoming.getmembers(), key=lambda m: m.name):
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe snapshot path: {member.name}")
            if any(p in {".git", "__pycache__", ".pytest_cache"} for p in path.parts):
                continue
            # ponytail: small source-only cases; add link handling before using linked repositories.
            if not (member.isfile() or member.isdir()) or path.as_posix() in seen:
                raise ValueError(f"Unsupported or duplicate snapshot entry: {member.name}")
            seen.add(path.as_posix())
            stream = incoming.extractfile(member) if member.isfile() else None
            member.name = path.as_posix()
            member.uid = member.gid = member.mtime = 0
            member.uname = member.gname = ""
            member.pax_headers = {}
            outgoing.addfile(member, stream)


class DockerWorkspace:
    """The minimal EnvironmentTools interface over one disposable, offline container."""

    def __init__(self, image, *, checks=None, snapshot=None, memory="768m", cpus=1, root="/testbed"):
        path = PurePosixPath(root)
        if not path.is_absolute() or str(path) == "/" or ".." in path.parts:
            raise ValueError("snapshot root must be a bounded absolute container directory")
        self.root = str(path)
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.checks = checks
        self.replay_snapshot = snapshot
        self.name = "sans-checkpoint-" + uuid.uuid4().hex

    def __enter__(self):
        mounts = [] if self.checks is None else [
            "--read-only", "--tmpfs", "/candidate:rw,exec", "--tmpfs", "/tmp:rw,exec",
            "--mount", f"type=bind,src={Path(self.checks).resolve()},dst=/checks,readonly",
            "--mount", f"type=bind,src={Path(self.replay_snapshot).resolve()},dst=/snapshot.tar,readonly",
        ]
        docker("run", "-d", "--name", self.name, "--network", "none", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "128", "--memory", self.memory,
               "--cpus", str(self.cpus), "--env", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "sleep",
               *mounts, self.image, "infinity")
        return self

    def __exit__(self, *exc):
        docker("rm", "-f", self.name)

    async def exec(self, command, cwd="/testbed", timeout_sec=30):
        def invoke():
            return subprocess.run(
                ["docker", "exec", "-w", cwd, self.name, "timeout", "-k", "2", str(timeout_sec),
                 "bash", "-c", command], capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout_sec + 10,
            )
        result = await asyncio.to_thread(invoke)
        return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    async def upload_file(self, source, target):
        await asyncio.to_thread(docker, "cp", source, f"{self.name}:{target}")

    async def download_file(self, source, target):
        await asyncio.to_thread(docker, "cp", f"{self.name}:{source}", target)

    async def is_file(self, path):
        return (await self.exec("test -f " + shlex.quote(path))).return_code == 0

    def snapshot(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        raw, normalized = directory / "capture.partial", directory / "canonical.partial"
        docker("pause", self.name)
        try:
            with raw.open("xb") as stream:
                subprocess.run(["docker", "cp", f"{self.name}:{self.root}/.", "-"], stdout=stream,
                               stderr=subprocess.PIPE, timeout=60, check=True)
        finally:
            docker("unpause", self.name)
        try:
            canonical_tar(raw, normalized)
            key = digest(normalized)
            target = directory / f"{key}.tar"
            if target.exists():
                if digest(target) != key:
                    raise ValueError("Existing snapshot hash mismatch")
            else:
                normalized.rename(target)
            return target
        finally:
            raw.unlink(missing_ok=True)
            normalized.unlink(missing_ok=True)


async def grade_snapshot(image, snapshot, checks, output):
    """Replay into an empty directory; frozen tests live outside the candidate tree."""
    snapshot, checks, output = Path(snapshot), Path(checks), Path(output)
    if digest(snapshot) != snapshot.stem:
        raise ValueError("Snapshot hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"snapshot_sha256": snapshot.stem, "image": image, "suites": {}, "status": "unknown"}
    with DockerWorkspace(image, checks=checks, snapshot=snapshot) as env:
        # Only locally canonicalized archives are accepted; validate again before container extraction.
        with tarfile.open(snapshot) as tar:
            names = set()
            for entry in tar:
                p = PurePosixPath(entry.name)
                if (p.is_absolute() or ".." in p.parts or not (entry.isfile() or entry.isdir())
                        or p.as_posix() in names):
                    raise ValueError("Unsafe replay archive")
                names.add(p.as_posix())
        setup = await env.exec("tar -xf /snapshot.tar -C /candidate")
        if setup.return_code:
            raise RuntimeError(setup.stderr)
        # Editable installation in the image must never silently fall back to the baseline package.
        imported = await env.exec(
            "PYTHONPATH=/candidate python -c " + shlex.quote(
                "import pathlib, slugify; assert pathlib.Path(slugify.__file__).resolve().is_relative_to('/candidate')"),
            cwd="/candidate",
        )
        receipt["import_check"] = vars(imported)
        if imported.return_code not in {0, 124, 137}:
            receipt["status"] = "candidate_failed"
        if imported.return_code == 0:
            for name in ("existing", "acceptance"):
                result = await env.exec(
                    "PYTHONPATH=/candidate PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest "
                    f"-q -p no:cacheprovider -o addopts= --noconftest --confcutdir=/checks "
                    f"--junitxml=/tmp/{name}.xml /checks/test_{name}.py", cwd="/candidate", timeout_sec=60,
                )
                (output / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
                xml = await env.exec(f"cat /tmp/{name}.xml")
                report = PytestReport.parse(xml.stdout) if xml.return_code == 0 else None
                (output / f"{name}.xml").write_text(xml.stdout, encoding="utf-8")
                outcome = ShellResult(result.stdout, result.stderr, result.return_code, pytest_report=report)
                receipt["suites"][name] = {"exit_code": result.return_code, "outcome": outcome.test_outcome,
                                           "report": asdict(report) if report else None,
                                           "tests_sha256": digest(checks / f"test_{name}.py")}
            if all(s["outcome"] in {"passed", "failed"} for s in receipt["suites"].values()):
                receipt["status"] = "graded"
            elif any(s["exit_code"] == 2 and s["report"] and s["report"]["error"]
                     for s in receipt["suites"].values()):
                receipt["status"] = "candidate_failed"
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
