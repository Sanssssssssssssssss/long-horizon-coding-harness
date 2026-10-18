"""Train one repository-execution Skill across six small coding fixtures."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .live import LiveAgentAdapter, _tree_hash
from .runner import BenchmarkRunner, TaskSpec, _safe_id, _write_artifact_index, EXPERIMENT_ROOT, experiment_path
from .smoke import _git


TARGET_SKILL = "repo_task_verification_workflow"


@dataclass(frozen=True)
class Fixture:
    task_id: str
    statement: str
    lesson: str
    files: dict[str, str]
    changed_file: str
    test_args: tuple[str, ...]


FIXTURES = (
    Fixture(
        "repo-root",
        "Fix ledger/money.py so format_cents always returns exactly two decimal places. "
        "Follow the repository's documented verification. Only modify ledger/money.py.",
        "Merge this durable lesson into the target Skill: first confirm the repository root, "
        "OS, active shell, language, and declared verification workflow; keep repository tool "
        "paths relative. Preserve prior useful instructions and never retain task-specific code.",
        {
            "README.md": "# Ledger\n\nFrom the repository root run:\n\n`python -m unittest discover -s tests -v`\n",
            "ledger/__init__.py": "",
            "ledger/money.py": "def format_cents(value: int) -> str:\n    return f\"{value / 100:.1f}\"\n",
            "tests/test_money.py": (
                "import unittest\n\nfrom ledger.money import format_cents\n\n\n"
                "class MoneyTests(unittest.TestCase):\n"
                "    def test_two_decimals(self):\n"
                "        self.assertEqual(format_cents(125), \"1.25\")\n"
                "        self.assertEqual(format_cents(5), \"0.05\")\n"
            ),
        },
        "ledger/money.py",
        ("-m", "unittest", "discover", "-s", "tests", "-v"),
    ),
    Fixture(
        "declared-runner",
        "Fix normalizer.py so clean_label trims, lowercases, and collapses whitespace. "
        "Use the repository-declared check. Only modify normalizer.py.",
        "Merge this durable lesson into the target Skill: read repository declarations before "
        "choosing the Python interpreter or test runner; do not guess a conventional command. "
        "Start with the smallest focused declared check.",
        {
            "README.md": (
                "# Labels\n\nRequired check:\n\n"
                "`python -m unittest discover -s checks -p \"check_*.py\" -v`\n"
            ),
            "normalizer.py": "def clean_label(value: str) -> str:\n    return value.strip().lower()\n",
            "checks/check_normalizer.py": (
                "import unittest\n\nfrom normalizer import clean_label\n\n\n"
                "class LabelChecks(unittest.TestCase):\n"
                "    def test_whitespace(self):\n"
                "        self.assertEqual(clean_label(\"  Hello   Wide   World  \"), \"hello wide world\")\n"
            ),
        },
        "normalizer.py",
        ("-m", "unittest", "discover", "-s", "checks", "-p", "check_*.py", "-v"),
    ),
    Fixture(
        "windows-shell",
        "Fix flags.py so parse_flag recognizes yes/true/1 case-insensitively. Only modify flags.py "
        "and complete the documented verification.",
        "Merge this durable lesson into the target Skill: identify the active shell before using "
        "shell syntax. Prefer one command per run_shell call instead of chaining commands with "
        "platform-specific separators. After a syntax failure, classify it and change strategy.",
        {
            "README.md": (
                "# Flags\n\nVerification commands:\n\n"
                "1. `python -m unittest discover -s tests -v`\n"
                "2. `git diff --check`\n"
            ),
            "flags.py": "def parse_flag(value: str) -> bool:\n    return value == \"yes\"\n",
            "tests/test_flags.py": (
                "import unittest\n\nfrom flags import parse_flag\n\n\n"
                "class FlagTests(unittest.TestCase):\n"
                "    def test_truthy(self):\n"
                "        for value in (\"yes\", \"YES\", \"true\", \"1\"):\n"
                "            self.assertTrue(parse_flag(value))\n"
                "        self.assertFalse(parse_flag(\"no\"))\n"
            ),
        },
        "flags.py",
        ("-m", "unittest", "discover", "-s", "tests", "-v"),
    ),
    Fixture(
        "failure-recovery",
        "Fix parser.py so parse_port accepts surrounding whitespace and rejects ports outside "
        "1..65535. Follow README.md and only modify parser.py.",
        "Merge this durable lesson into the target Skill: after a failed command, classify cwd, "
        "shell, interpreter, dependency, or real test failure before acting. Never repeat the same "
        "command in the same repository state when its failure output is unchanged.",
        {
            "README.md": "# Port parser\n\nChecks live in spec:\n\n`python -m unittest discover -s spec -v`\n",
            "package.json": "{\"name\":\"documentation-only\",\"private\":true}\n",
            "parser.py": "def parse_port(value: str) -> int:\n    return int(value)\n",
            "spec/test_parser.py": (
                "import unittest\n\nfrom parser import parse_port\n\n\n"
                "class ParserTests(unittest.TestCase):\n"
                "    def test_range(self):\n"
                "        self.assertEqual(parse_port(\" 8080 \"), 8080)\n"
                "        for value in (\"0\", \"65536\"):\n"
                "            with self.assertRaises(ValueError):\n"
                "                parse_port(value)\n"
            ),
        },
        "parser.py",
        ("-m", "unittest", "discover", "-s", "spec", "-v"),
    ),
    Fixture(
        "failure-classification",
        "Fix ratio.py so safe_ratio returns 0.0 for a zero denominator and otherwise returns the "
        "division result. This is a standard-library project; only modify ratio.py.",
        "Merge this durable lesson into the target Skill: distinguish environment, launcher, and "
        "dependency failures from genuine assertion or code failures. Fix code only for real test "
        "failures and do not install or change dependencies without evidence.",
        {
            "README.md": (
                "# Ratios\n\nNo third-party dependencies are required. Run:\n\n"
                "`python -m unittest discover -s tests -v`\n"
            ),
            "ratio.py": "def safe_ratio(left: float, right: float) -> float:\n    return left / right\n",
            "requirements-dev.txt": "# intentionally empty: unittest is in the standard library\n",
            "tests/test_ratio.py": (
                "import unittest\n\nfrom ratio import safe_ratio\n\n\n"
                "class RatioTests(unittest.TestCase):\n"
                "    def test_values(self):\n"
                "        self.assertEqual(safe_ratio(3, 2), 1.5)\n"
                "        self.assertEqual(safe_ratio(3, 0), 0.0)\n"
            ),
        },
        "ratio.py",
        ("-m", "unittest", "discover", "-s", "tests", "-v"),
    ),
    Fixture(
        "stop-when-green",
        "Fix greeting.py so greet trims the name before formatting Hello, <name>!. Only modify "
        "greeting.py and verify the result.",
        "Merge this durable lesson into the target Skill: once the focused check is green, run the "
        "required tests, git diff --check, inspect the final diff and newline, then stop. Do not keep "
        "exploring or repeat successful checks without new evidence.",
        {
            "README.md": "# Greeting\n\nRun `python -m unittest discover -s tests -v`.\n",
            "greeting.py": "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n",
            "tests/test_greeting.py": (
                "import unittest\n\nfrom greeting import greet\n\n\n"
                "class GreetingTests(unittest.TestCase):\n"
                "    def test_trimmed(self):\n"
                "        self.assertEqual(greet(\" Ada \"), \"Hello, Ada!\")\n"
            ),
        },
        "greeting.py",
        ("-m", "unittest", "discover", "-s", "tests", "-v"),
    ),
)


def _verify_single_skill(root: Path) -> str:
    files = list(root.glob("*/SKILL.md"))
    if len(files) != 1 or f"name: {TARGET_SKILL}" not in files[0].read_text(encoding="utf-8"):
        raise RuntimeError(f"training must contain exactly the target Skill: {TARGET_SKILL}")
    return _tree_hash(root)[0]


def _verified_training_patch(row: dict[str, Any], result: dict[str, Any]) -> bool:
    diagnostic = (result.get("evaluation") or {}).get("diagnostic") or {}
    return row.get("resolved") is True or diagnostic.get("resolved") is True


def _fixture(root: Path, spec: Fixture) -> tuple[Path, str]:
    root = root.resolve()
    if root.exists():
        if not root.is_dir() or root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
            raise RuntimeError(f"unsafe training fixture: {root}")
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
            raise RuntimeError(f"fixture git root mismatch: {spec.task_id}")
        if _git(root, "status", "--porcelain"):
            raise RuntimeError(f"fixture is dirty: {spec.task_id}")
        if sorted(_git(root, "ls-files").splitlines()) != sorted(spec.files):
            raise RuntimeError(f"fixture files changed: {spec.task_id}")
        for name, content in spec.files.items():
            if (root / name).read_text(encoding="utf-8") != content:
                raise RuntimeError(f"fixture content changed: {spec.task_id}/{name}")
        return root, _git(root, "rev-parse", "HEAD")

    for name, content in spec.files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", "--", ".")
    commit_env = dict(os.environ)
    commit_env.update({
        "GIT_AUTHOR_NAME": "Execution Training",
        "GIT_AUTHOR_EMAIL": "training@example.invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "Execution Training",
        "GIT_COMMITTER_EMAIL": "training@example.invalid",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    })
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", spec.task_id, env=commit_env)
    return root, _git(root, "rev-parse", "HEAD")


def _evaluator(spec: Fixture):
    def evaluate(*, workdir: Path, env: dict[str, str], **_: Any) -> dict[str, Any]:
        changed = sorted(set(
            _git(workdir, "diff", "--name-only", "HEAD").splitlines()
            + _git(workdir, "ls-files", "--others", "--exclude-standard").splitlines()
        ))
        tests = subprocess.run(
            [sys.executable, *spec.test_args], cwd=workdir, env=dict(env), text=True,
            encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        print(tests.stdout)
        diff = subprocess.run(
            ["git", "diff", "--check", "HEAD", "--"], cwd=workdir, env=dict(env),
            text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        source = (workdir / spec.changed_file).read_bytes()
        resolved = (
            changed == [spec.changed_file]
            and tests.returncode == 0
            and diff.returncode == 0
            and source.endswith(b"\n")
            and b"\r\n" not in source
        )
        return {
            "resolved": resolved,
            "status": "completed",
            "changed_files": changed,
            "tests_passed": tests.returncode == 0,
            "diff_check_passed": diff.returncode == 0,
            "clean_line_endings": source.endswith(b"\n") and b"\r\n" not in source,
        }

    return evaluate


async def run_training(
    *, runs_root: Path, training_id: str, api_base: str, api_key: str, model: str,
    provider_revision: str, skills_seed: Path, max_turns: int = 20,
    max_input_tokens: int = 32_000, max_output_tokens: int = 4_096,
    max_cost_usd: float | None = None, timeout_seconds: float = 900,
    start_step: int = 1, thinking: bool = False,
) -> dict[str, Any]:
    _safe_id("training_id", training_id)
    if not 1 <= start_step <= len(FIXTURES):
        raise ValueError(f"start_step must be between 1 and {len(FIXTURES)}")
    root = (runs_root / "execution-skill-training" / training_id).resolve()
    completed = root / "training-summary.json"
    if completed.is_file():
        return json.loads(completed.read_text(encoding="utf-8"))
    current_seed = skills_seed.expanduser().resolve()
    input_sha = _verify_single_skill(current_seed)
    lineage: list[dict[str, Any]] = []

    for index, spec in enumerate(FIXTURES[start_step - 1 :], start_step):
        repository, commit = _fixture(
            runs_root.resolve() / ".fixtures" / "execution-training-v1" / spec.task_id,
            spec,
        )
        adapter = LiveAgentAdapter(
            api_base=api_base, api_key=api_key, model=model,
            provider_revision=provider_revision, profile="B1",
            thinking=thinking,
            max_turns=max_turns, max_cost_usd=max_cost_usd,
            max_output_tokens=max_output_tokens, max_input_tokens=max_input_tokens,
            online_evolution=True, skills_seed=current_seed,
            evolution_target_skill=TARGET_SKILL, evolution_hint=spec.lesson,
            custom_tools=None, custom_system_prompt=None,
        )
        task = TaskSpec(
            task_id=spec.task_id, repository=repository, base_commit=commit,
            problem_statement=spec.statement, platform="windows",
            timeout_seconds=timeout_seconds, network_policy="enabled",
        )
        runner = BenchmarkRunner(
            runs_root=root, suite_name="steps", run_id=f"{index:02d}-{spec.task_id}",
            tasks=[task], agent=adapter, evaluator=_evaluator(spec), profile="B1",
            manifest={"suite_revision": "execution-training-v1", "evaluator_revision": "execution-training-v1"},
        )
        summary = await runner.run()
        row = summary["tasks"][0]
        attempt = runner.run_dir / "attempts" / spec.task_id / row["attempt_id"]
        result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
        learning = result.get("learning") or {}
        evolution = learning.get("evolution_result") or {}
        current_seed = attempt / str(learning.get("skills_path") or "")
        output_sha = _verify_single_skill(current_seed)
        if evolution.get("action") != "merge" or evolution.get("skill") != TARGET_SKILL:
            raise RuntimeError(f"fixture did not merge the target Skill: {spec.task_id}")
        lineage.append({
            "step": index, "task_id": spec.task_id, "input_skills_sha256": input_sha,
            "output_skills_sha256": output_sha,
            "run_dir": runner.run_dir.relative_to(runs_root.resolve()).as_posix(),
            "benchmark_status": row.get("status"),
            "patch_verified": _verified_training_patch(row, result),
            "termination_reason": result.get("termination_reason"),
            "metrics": summary["totals"],
        })
        input_sha = output_sha

    frozen = root / "frozen-skills"
    if frozen.exists():
        if _tree_hash(frozen)[0] != input_sha:
            raise RuntimeError("existing frozen Skill does not match completed training")
    else:
        shutil.copytree(current_seed, frozen)
    totals: dict[str, int | float | None] = {}
    for key in lineage[0]["metrics"]:
        values = [step["metrics"].get(key) for step in lineage]
        totals[key] = sum(values) if all(isinstance(value, (int, float)) for value in values) else None
    hit, miss = totals.get("cached_input_tokens"), totals.get("cache_miss_tokens")
    totals["cache_hit_rate"] = (
        hit / (hit + miss)
        if isinstance(hit, (int, float)) and isinstance(miss, (int, float)) and hit + miss > 0
        else None
    )
    report = {
        "schema_version": 1, "training_id": training_id, "target_skill": TARGET_SKILL,
        "start_step": start_step,
        "fixture_revision": "execution-training-v1", "steps": lineage,
        "totals": totals, "frozen_skills_path": "frozen-skills",
        "frozen_skills_sha256": input_sha,
    }
    temporary = completed.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, completed)
    _write_artifact_index(root)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=experiment_path, default=EXPERIMENT_ROOT / "runs")
    parser.add_argument("--training-id", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider-revision", required=True)
    parser.add_argument("--skills-seed", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=32_000)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = asyncio.run(run_training(
        runs_root=args.runs_root, training_id=args.training_id,
        api_base=args.api_base, api_key=getpass.getpass("API key (not stored): "),
        model=args.model, provider_revision=args.provider_revision,
        skills_seed=args.skills_seed, max_turns=args.max_turns,
        max_input_tokens=args.max_input_tokens, max_output_tokens=args.max_output_tokens,
        max_cost_usd=args.max_cost_usd, timeout_seconds=args.timeout_seconds,
        start_step=args.start_step, thinking=args.thinking,
    ))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
