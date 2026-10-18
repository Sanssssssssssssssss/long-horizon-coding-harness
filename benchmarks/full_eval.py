"""One product-profile coding eval with independent hidden checks."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .live import LiveAgentAdapter
from .runner import BenchmarkRunner, TaskSpec, EXPERIMENT_ROOT, experiment_path
from .smoke import _git


_SLUGGER = '''"""Small URL slug helper."""


def slugify(value: str) -> str:
    """Return a URL-safe slug."""
    return value.strip().lower().replace(" ", "-")
'''

_PUBLIC_TESTS = '''import unittest
from pathlib import Path

from slugger import slugify


class SlugifyTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation_and_runs(self):
        self.assertEqual(slugify("  release___candidate!!  "), "release-candidate")

    def test_unicode_and_empty(self):
        self.assertEqual(slugify("Café déjà vu!"), "cafe-deja-vu")
        self.assertEqual(slugify("---"), "untitled")
        self.assertEqual(slugify(""), "untitled")

    def test_source_has_final_newline(self):
        source = Path(__file__).resolve().parents[1] / "slugger.py"
        self.assertTrue(source.read_bytes().endswith(b"\\n"))


if __name__ == "__main__":
    unittest.main()
'''


async def run_full_eval(
    *, adapter: LiveAgentAdapter, runs_root: str | Path, run_id: str,
    timeout_seconds: float = 900,
) -> dict[str, Any]:
    output_root = Path(runs_root).expanduser().resolve()
    repository, base_commit = _fixture(output_root / ".fixtures" / "full-harness-eval-v2")
    task = TaskSpec(
        task_id="slugify-unicode",
        repository=repository,
        base_commit=base_commit,
        problem_statement=(
            "Fix slugger.slugify using only the Python standard library. Normalize Unicode "
            "with NFKD and discard combining marks/non-ASCII characters; lowercase the text; "
            "replace every run of characters outside [a-z0-9] with one hyphen; strip leading "
            "and trailing hyphens; and return 'untitled' when nothing remains. Only modify "
            "slugger.py. Run `python -m unittest discover -s tests -v` and `git diff --check` "
            "before finishing, and preserve the final LF newline. Durable training feedback: "
            "work from the repository root with relative paths, use Python UTF-8 mode only when "
            "console encoding blocks a check, and always verify the final diff. Learn this workflow, "
            "but never retain task-specific slug code as a Skill."
        ),
        platform="local",
        timeout_seconds=timeout_seconds,
        network_policy="enabled",
    )

    def evaluator(*, workdir: Path, env: Mapping[str, str], **_: Any) -> dict[str, Any]:
        changed = sorted(set(
            _git(workdir, "diff", "--name-only", "HEAD").splitlines()
            + _git(workdir, "ls-files", "--others", "--exclude-standard").splitlines()
        ))
        public = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=workdir, env=dict(env), text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        print(public.stdout)
        hidden_code = (
            "from slugger import slugify; "
            "cases={'Café déjà vu!':'cafe-deja-vu','  API___Version 2.0 ':'api-version-2-0',"
            "'naïve façade':'naive-facade','---':'untitled','':'untitled'}; "
            "bad={k:(slugify(k),v) for k,v in cases.items() if slugify(k)!=v}; "
            "assert not bad,bad"
        )
        hidden = subprocess.run(
            [sys.executable, "-c", hidden_code], cwd=workdir, env=dict(env), text=True,
            encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        print(hidden.stdout)
        diff_check = subprocess.run(
            ["git", "diff", "--check", "HEAD", "--"], cwd=workdir, env=dict(env),
            text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        source = (workdir / "slugger.py").read_bytes()
        clean_text = source.endswith(b"\n") and b"\r\n" not in source
        resolved = (
            changed == ["slugger.py"]
            and public.returncode == 0
            and hidden.returncode == 0
            and diff_check.returncode == 0
            and clean_text
        )
        return {
            "resolved": resolved,
            "status": "completed",
            "changed_files": changed,
            "public_tests_passed": public.returncode == 0,
            "hidden_tests_passed": hidden.returncode == 0,
            "diff_check_passed": diff_check.returncode == 0,
            "clean_line_endings": clean_text,
        }

    runner = BenchmarkRunner(
        runs_root=output_root,
        suite_name="harness-full-eval",
        run_id=run_id,
        tasks=[task],
        agent=adapter,
        evaluator=evaluator,
        profile="B1",
    )
    return await runner.run()


def _fixture(repository: Path) -> tuple[Path, str]:
    repository = repository.resolve()
    expected = {"slugger.py": _SLUGGER, "tests/test_slugger.py": _PUBLIC_TESTS}
    if repository.exists():
        is_junction = hasattr(repository, "is_junction") and repository.is_junction()
        if not repository.is_dir() or repository.is_symlink() or is_junction:
            raise RuntimeError(f"unsafe full-eval fixture path: {repository}")
        if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
            raise RuntimeError("full-eval fixture git root does not match its managed path")
        if _git(repository, "status", "--porcelain"):
            raise RuntimeError("full-eval fixture must remain clean")
        if sorted(_git(repository, "ls-files").splitlines()) != sorted(expected):
            raise RuntimeError("full-eval fixture tracked files do not match")
        for name, content in expected.items():
            if (repository / name).read_text(encoding="utf-8") != content:
                raise RuntimeError(f"full-eval fixture content mismatch: {name}")
        return repository, _git(repository, "rev-parse", "HEAD")

    (repository / "tests").mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "core.autocrlf", "false")
    for name, content in expected.items():
        (repository / name).write_text(content, encoding="utf-8", newline="\n")
    _git(repository, "add", "--", ".")
    commit_env = dict(os.environ)
    commit_env.update({
        "GIT_AUTHOR_NAME": "Harness Eval",
        "GIT_AUTHOR_EMAIL": "eval@example.invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "Harness Eval",
        "GIT_COMMITTER_EMAIL": "eval@example.invalid",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    })
    _git(repository, "-c", "commit.gpgsign=false", "commit", "-qm", "full harness eval v2", env=commit_env)
    return repository, _git(repository, "rev-parse", "HEAD")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=experiment_path, default=EXPERIMENT_ROOT / "runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider-revision", required=True)
    parser.add_argument("--stream-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--online-evolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skills-seed", type=Path)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=32_000)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    adapter = LiveAgentAdapter(
        api_base=args.api_base,
        api_key=getpass.getpass("API key (not stored): "),
        model=args.model,
        provider_revision=args.provider_revision,
        profile="B1",
        max_turns=args.max_turns,
        max_cost_usd=args.max_cost_usd,
        max_output_tokens=args.max_output_tokens,
        max_input_tokens=args.max_input_tokens,
        openai_stream_include_usage=args.stream_usage,
        online_evolution=args.online_evolution,
        skills_seed=args.skills_seed,
        custom_tools=None,
        custom_system_prompt=None,
        thinking=args.thinking,
    )
    summary = asyncio.run(run_full_eval(
        adapter=adapter, runs_root=args.runs_root, run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["counts"]["resolved"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
