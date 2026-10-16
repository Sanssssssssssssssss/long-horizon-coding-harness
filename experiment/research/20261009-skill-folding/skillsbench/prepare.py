"""Prepare dialogue-parser images and run unchanged official baseline/gold checks, without any model API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiment/cache/skillsbench-skillevolver-20261009"
WSL = ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--exec"]
BASE = "python:3.12.8-slim"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def linux(path):
    path = Path(path).resolve()
    path.relative_to(ROOT)
    return "/mnt/" + path.drive[0].lower() + str(path)[2:].replace("\\", "/")


def docker(args, *, log=None, timeout=180, check=True):
    result = subprocess.run([*WSL, "docker", *map(str, args)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if log:
        Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
    if check and result.returncode:
        raise RuntimeError(f"Docker command failed ({result.returncode}); see {log}: {result.stderr[-1500:]}")
    return result


def image_id(name):
    result = docker(["image", "inspect", name, "--format", "{{.Id}}"], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def official_inputs():
    manifest = json.loads((CACHE / "download-receipt.json").read_text(encoding="utf-8"))
    tasks = {"canonical": CACHE / "skillsbench/tasks/dialogue-parser",
             "train": CACHE / "skillevolver/bench-assets/tasks-train/dialogue-parser"}
    prefixes = [path.relative_to(CACHE).as_posix() + "/" for path in tasks.values()]
    sources = {}
    for row in manifest["files"]:
        if any(row["path"].startswith(prefix) for prefix in prefixes):
            path = CACHE / row["path"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"Official task changed: {path}")
            sources[row["path"]] = row["sha256"]
    for task in tasks.values():
        actual = {path.relative_to(CACHE).as_posix() for path in task.rglob("*") if path.is_file()}
        expected = {path for path in sources if path.startswith(task.relative_to(CACHE).as_posix() + "/")}
        if actual != expected:
            raise ValueError(f"Official task file set changed: {task}")
    return tasks, sources, manifest["repositories"]


def prepare_image(split, task, out, build_proxy):
    context = out / f"{split}-recipe"
    context.mkdir()
    shutil.copyfile(task / "environment/script.txt", context / "script.txt")
    recipe = (
        f"FROM {BASE}\n"
        "ARG BUILD_PROXY\nENV http_proxy=$BUILD_PROXY https_proxy=$BUILD_PROXY\n"
        "RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources "
        "&& apt-get -o Acquire::https::Timeout=20 -o Acquire::Retries=1 update "
        "&& apt-get -o Acquire::https::Timeout=20 -o Acquire::Retries=1 install -y --no-install-recommends graphviz bash git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN pip install --no-cache-dir graphviz==0.20.3 pytest==8.3.4 pytest-json-ctrf==0.3.6\n"
        "ENV http_proxy=\"\" https_proxy=\"\"\n"
        "COPY script.txt /app/script.txt\nWORKDIR /app\nCMD [\"bash\"]\n"
    )
    (context / "Dockerfile").write_text(recipe, encoding="utf-8")
    identity = hashlib.sha256((recipe + build_proxy + digest(context / "script.txt")).encode()).hexdigest()
    tag = f"harness-skillsbench-dialogue-{split}:{identity[:12]}"
    reused = image_id(tag) is not None
    if not reused:
        docker(["build", "--memory", "4g", "--cpu-period", "100000", "--cpu-quota", "200000",
                "--force-rm", "--build-arg", f"BUILD_PROXY={build_proxy}",
                "-t", tag, linux(context)], log=out / f"{split}-build.log", timeout=900)
    return {"tag": tag, "image_id": image_id(tag), "reused": reused, "recipe_sha256": digest(context / "Dockerfile"),
            "input_sha256": digest(context / "script.txt"), "base_image": BASE}


def calibration_plugin(out):
    support = out / "calibration-support"
    support.mkdir()
    (support / "calibration_identity.py").write_text('''import json
from pathlib import Path

collected = []
reports = {}

def pytest_collection_finish(session):
    collected.extend(item.nodeid for item in session.items)

def pytest_runtest_logreport(report):
    reports.setdefault(report.nodeid, {})[report.when] = report.outcome

def pytest_sessionfinish(session, exitstatus):
    cases = []
    for nodeid in collected:
        phases = reports.get(nodeid, {})
        if phases.get("setup") == "failed" or phases.get("teardown") == "failed":
            status = "error"
        elif "skipped" in phases.values():
            status = "skipped"
        elif phases.get("call") == "failed":
            status = "failed"
        elif all(phases.get(phase) == "passed" for phase in ("setup", "call", "teardown")):
            status = "passed"
        else:
            status = "incomplete"
        cases.append({"name": nodeid, "status": status, "phases": phases})
    Path("/logs/verifier/pytest-cases.json").write_text(json.dumps({"exitstatus": int(exitstatus), "collected": collected, "cases": cases}, indent=2))
''', encoding="utf-8")
    return support


def probe(image, command, log):
    name = "skillsbench-probe-" + uuid.uuid4().hex[:12]
    try:
        return docker(["run", "--name", name, "--network", "none", "--cpus", "2", "--memory", "4g",
                       image, *command], log=log)
    finally:
        docker(["rm", "-f", name], timeout=60, check=False)


def calibrate(split, mode, image, task, out, support):
    output = out / split / mode
    output.mkdir(parents=True)
    name = "skillsbench-calibration-" + uuid.uuid4().hex[:12]
    mounts = ["--mount", f"type=bind,source={linux(task / 'tests')},target=/tests,readonly",
              "--mount", f"type=bind,source={linux(output)},target=/logs/verifier",
              "--mount", f"type=bind,source={linux(support)},target=/calibration_support,readonly"]
    command = "bash /tests/test.sh"
    if mode == "gold":
        helper = CACHE / "skillsbench/tasks/dialogue-parser/environment/skills/dialogue_graph/scripts"
        mounts += ["--mount", f"type=bind,source={linux(task / 'solution')},target=/solution,readonly",
                   "--mount", f"type=bind,source={linux(helper)},target=/app/environment/skills/dialogue_graph/scripts,readonly"]
        command = "bash /solution/solve.sh && bash /tests/test.sh"
    try:
        result = docker(["run", "--name", name, "--network", "none", "--cpus", "2", "--memory", "4g",
                         "-e", "PIP_NO_INDEX=1", "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
                         "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTEST_PLUGINS=calibration_identity",
                         "-e", "PYTHONPATH=/calibration_support",
                         *mounts, image["image_id"], "bash", "-c", command],
                        log=output / "execution.log", timeout=240, check=False)
        if result.returncode:
            raise RuntimeError(f"{split}/{mode} execution failed; see {output / 'execution.log'}")
        ctrf = json.loads((output / "ctrf.json").read_text(encoding="utf-8"))["results"]
        full = json.loads((output / "pytest-cases.json").read_text(encoding="utf-8"))
        cases = full["cases"]
        if not cases or len(cases) != len(full["collected"]) or len({case["name"] for case in cases}) != len(cases):
            raise RuntimeError(f"{split}/{mode} has incomplete parameter testcase identities")
        reward = float((output / "reward.txt").read_text().strip())
        expected = 1.0 if mode == "gold" else 0.0
        if (reward != expected or (mode == "gold" and (full["exitstatus"] != 0 or any(case["status"] != "passed" for case in cases)))
                or (mode == "baseline" and (full["exitstatus"] != 1
                    or any(case["status"] in {"passed", "incomplete"} for case in cases)
                    or not any(case["status"] in {"failed", "error"} for case in cases)))):
            raise RuntimeError(f"{split}/{mode} reward {reward} did not satisfy {expected}; see {output}")
        score = {"reward": reward, "official_ctrf_summary": ctrf["summary"], "official_ctrf_functions": ctrf["tests"],
                 "pytest_exitstatus": full["exitstatus"], "cases": cases,
                 "summary": {"tests": len(cases), **{status: sum(case["status"] == status for case in cases)
                 for status in ("passed", "failed", "error", "skipped", "incomplete")}}, "network": "none",
                 "cpus": 2, "memory": "4g", "human_helper_exposed": mode == "gold",
                 "scope": "Independent official oracle calibration" if mode == "gold" else "Initial empty solver-state calibration",
                 "check_files": {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}}
        save(output / "score.json", score)
        return score
    finally:
        docker(["rm", "-f", name], timeout=60, check=False)


def prepare(out, build_proxy):
    out = out.resolve()
    out.relative_to(ROOT / "experiment")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "preparing", "model_api_calls": 0, "images": {}, "calibration": {},
               "scope": "SkillsBench dialogue-parser canonical and author train, original oracle and checks preserved"}
    save(out / "receipt.json", receipt)
    try:
        tasks, sources, revisions = official_inputs()
        receipt.update(official_files=sources, repositories=revisions, preparer_sha256=digest(__file__), build_proxy=build_proxy)
        support = calibration_plugin(out)
        receipt["calibration_observer_sha256"] = digest(support / "calibration_identity.py")
        for split, task in tasks.items():
            image = prepare_image(split, task, out, build_proxy)
            receipt["images"][split] = image
            probe_code = "import graphviz,pytest,sys,os,json; from importlib.metadata import version; " \
                    "assert sys.version_info[:3]==(3,12,8); " \
                    "assert not os.environ.get('http_proxy') and not os.environ.get('https_proxy'); " \
                    "assert not any(os.path.exists(p) for p in ['/solution','/tests','/app/environment/skills','/root/.claude/skills']); " \
                    "print(json.dumps({'python':sys.version,'graphviz':version('graphviz'),'pytest':version('pytest')," \
                    "'pytest-json-ctrf':version('pytest-json-ctrf'),'hidden_artifacts':False}))"
            probe(image["image_id"], ["python", "-c", probe_code], out / f"{split}-solver-probe.json")
            probe(image["image_id"], ["dot", "-V"], out / f"{split}-dot-version.txt")
            git_probe = """set -eu
git --version
git -c init.defaultBranch=main init /app
git add -- script.txt
git -c user.name=Calibration -c user.email=calibration@local.invalid -c commit.gpgsign=false commit -m 'Initial task input'
test -z "$(git status --porcelain)"
git rev-parse --verify HEAD
index=$(mktemp)
trap 'rm -f "$index"' EXIT
GIT_INDEX_FILE="$index" git read-tree HEAD
GIT_INDEX_FILE="$index" git add -A -- .
GIT_INDEX_FILE="$index" git diff --cached --check HEAD --
test -z "$(git diff --binary HEAD --)"
test -z "$(git ls-files --others --exclude-standard)"
printf 'Git initial status and full diff are clean\\n'
"""
            probe(image["image_id"], ["bash", "-c", git_probe], out / f"{split}-git-probe.txt")
            scores = {mode: calibrate(split, mode, image, task, out, support) for mode in ("baseline", "gold")}
            if sorted(row["name"] for row in scores["baseline"]["cases"]) != sorted(row["name"] for row in scores["gold"]["cases"]):
                raise RuntimeError(f"{split} baseline and gold testcase identities differ")
            receipt["calibration"][split] = scores
            save(out / "receipt.json", receipt)
        if official_inputs()[1] != sources:
            raise ValueError("Official sources changed during calibration")
        receipt["status"] = "ready"
        receipt["image_ids"] = {split: item["image_id"] for split, item in receipt["images"].items()}
    except Exception as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        save(out / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(out / 'receipt.json'), "model_api_calls": 0}), flush=True)


def validate_ready(path):
    path = path.resolve()
    path.relative_to(ROOT / "experiment")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "ready" or receipt.get("preparer_sha256") != digest(__file__):
        raise ValueError("Preparation is incomplete or its implementation changed")
    if receipt.get("official_files") != official_inputs()[1]:
        raise ValueError("Official task inputs changed")
    if digest(path.parent / "calibration-support/calibration_identity.py") != receipt["calibration_observer_sha256"]:
        raise ValueError("Calibration observer changed")
    for split in ("canonical", "train"):
        if image_id(receipt["image_ids"][split]) != receipt["images"][split]["image_id"]:
            raise ValueError(f"{split} prepared image is unavailable")
        if not (path.parent / f"{split}-git-probe.txt").is_file():
            raise ValueError(f"{split} git probe is missing")
        for mode in ("baseline", "gold"):
            for name, sha in receipt["calibration"][split][mode]["check_files"].items():
                if digest(path.parent / split / mode / name) != sha:
                    raise ValueError(f"{split}/{mode} calibration evidence changed")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="New output directory under experiment/")
    action.add_argument("--check-ready", type=Path, help="Verify an existing receipt and its local artifacts before paid solving")
    parser.add_argument("--build-proxy", default="", help="Existing verified local HTTP proxy, for container dependency installation only")
    args = parser.parse_args()
    if args.check_ready:
        verified = validate_ready(args.check_ready)
        print(json.dumps({"status": "ready", "verified_images": verified["image_ids"], "model_api_calls": 0}))
    else:
        prepare(args.out, args.build_proxy)
