"""Small, runtime-owned pytest comparisons; the working tree is never rewritten."""

from dataclasses import replace
import json
import re
import shlex
import subprocess

from .tool_result import PytestReport, ShellResult


def pytest_selection(command: str, root: str) -> tuple[list[str], list[str]] | None:
    if "&&" in command:
        prefix, command = command.split("&&", 1)
        try:
            directory = shlex.split(prefix.replace("\\", "/"))
        except ValueError:
            return None
        if directory not in (["cd", "."], ["cd", root.replace("\\", "/")]):
            return None
    if any(token in command for token in ("&&", "||", ";", "\n", "|", ">", "<", "`", "$")):
        return None
    try:
        argv = shlex.split(command.replace("\\", "/"))
    except ValueError:
        return None
    module_index, seen = 1, set()
    while module_index < len(argv):
        option = argv[module_index]
        width = 1
        if argv[module_index:module_index + 2] == ["-X", "utf8"]:
            option, width = "-Xutf8", 2
        if option not in {"-B", "-Xutf8"}:
            break
        if option in seen:
            return None
        seen.add(option)
        module_index += width
    if len(argv) < module_index + 3 or argv[module_index:module_index + 2] != ["-m", "pytest"] or not re.fullmatch(r"(?:.*/)?python(?:\d+(?:\.\d+)*)?(?:\.exe)?", argv[0]):
        return None
    paths, config_seen = [], False
    arguments = iter(argv[module_index + 2:])
    for arg in arguments:
        # ponytail: explicit files/nodes, local configuration and display flags only; unsupported selectors
        # remain blocking until a comparable runner is implemented.
        if arg in {"-q", "-s", "-v", "-vv"}:
            continue
        if arg in {"-c", "--config-file"} or arg.startswith("--config-file="):
            config = next(arguments, "") if "=" not in arg else arg.split("=", 1)[1]
            if config_seen or not config or config.startswith("/") or ":" in config or ".." in config.split("/"):
                return None
            config_seen = True
            continue
        path = arg.split("::", 1)[0]
        if arg.startswith("-") or not path.endswith(".py") or path.startswith("/") or ":" in path or ".." in path.split("/"):
            return None
        paths.append(path)
    return (argv, sorted(set(paths))) if paths and len(set(paths)) <= 8 else None


def python_command(script: str, args: list[str], root: str, python: str = "python") -> str:
    argv = [python, "-B", "-c", f"exec({script!r})", *args]
    return shlex.join(argv) if root.startswith("/") else subprocess.list2cmdline(argv)


DIFF_CHECK_SCRIPT = '''
import os, subprocess, sys, tempfile
from pathlib import Path
git_dir = Path(subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], text=True).strip()).resolve()
with tempfile.TemporaryDirectory(prefix='sans-diff-', dir=git_dir) as directory:
    Path(directory).resolve().relative_to(git_dir)
    check_env = dict(os.environ, GIT_INDEX_FILE=os.path.join(directory, 'index'))
    for arguments in [['read-tree', 'HEAD'], ['add', '-A', '--', '.'], ['diff', '--cached', '--check', 'HEAD', '--']]:
        result = subprocess.run(['git', *arguments], env=check_env)
        if result.returncode:
            sys.exit(result.returncode)
'''


CHECK_FINGERPRINT_SCRIPT = r'''
import hashlib, json, os, sys
from pathlib import Path
root, paths, args = json.loads(sys.argv[1])
root = Path(root).resolve(strict=True)
def within(value):
    path = Path(os.path.abspath(root / value))
    path.resolve().relative_to(root)
    path.relative_to(root)
    return path

files, directories = set(), {root}
def include_directory(path):
    path.relative_to(root)
    if not path.is_dir():
        raise ValueError('test condition directory does not exist: ' + str(path))
    while path != root:
        directories.add(path)
        path = path.parent

for name in paths:
    path = within(name)
    if not path.is_file():
        raise ValueError('selected test source is missing: ' + name)
    files.add(path)
    include_directory(path.parent)
if not files:
    raise ValueError('explicit test sources are required')
args = iter(args)
for arg in args:
    option, equal, value = arg.partition('=')
    if option in ('-c', '--config-file', '--inifile', '--rootdir', '--confcutdir'):
        value = value if equal else next(args)
    elif arg.startswith('-c') and not arg.startswith('--'):
        option, value = '-c', arg[2:]
    else:
        continue
    path = within(value)
    if option in ('--rootdir', '--confcutdir'):
        include_directory(path)
    else:
        if not path.is_file():
            raise ValueError('explicit pytest configuration is missing: ' + value)
        files.add(path)

# ponytail: hash local configuration candidates conservatively, including inactive
# files; trace actual loaded configuration if these invalidate useful witnesses.
for directory in directories:
    for name in ('conftest.py', 'pytest.toml', '.pytest.toml', 'pytest.ini', '.pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg'):
        files.add(directory / name)
digest = hashlib.sha256()
for path in sorted(files):
    path.resolve().relative_to(root)
    content = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    digest.update(json.dumps([path.relative_to(root).as_posix(), content]).encode())
print(digest.hexdigest())
'''


PAIR_SCRIPT = r'''
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from _pytest.config import get_config
root, base, argv, paths, timeout = json.loads(sys.argv[1])
root = Path(root).resolve()
argv[0] = sys.executable  # The transport was launched by the original shell/interpreter.
configuration_paths = set()
arguments = iter(argv[argv.index('-m') + 2:])
for arg in arguments:
    if arg in ('-c', '--config-file') or arg.startswith('--config-file='):
        name = next(arguments) if '=' not in arg else arg.split('=', 1)[1]
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('explicit pytest configuration must exist inside the repository: ' + name)
        configuration_paths.update((path.relative_to(root).as_posix(), path.resolve().relative_to(root).as_posix()))
config = get_config()
config.parse(argv[argv.index('-m') + 2:])
test_root, junit_prefix = config.rootpath, config.getoption('junitprefix') or ''
config._ensure_unconfigure()
if root.is_relative_to(test_root):
    junit_prefix = '.'.join(part for part in (junit_prefix, '.'.join(root.relative_to(test_root).parts)) if part)
def within(path, parent):
    return os.path.commonpath([str(path.resolve()), str(parent.resolve())]) == str(parent.resolve())
def git(*args):
    return subprocess.check_output(['git', '-C', str(root), *args])
changed = git('diff', '--name-only', base, '--').decode().splitlines()
extra = git('ls-files', '--others', '--exclude-standard').decode().splitlines()
for name in configuration_paths:
    if subprocess.run(['git', '-C', str(root), 'cat-file', '-e', base + ':' + name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise ValueError('comparison requires unchanged test configuration/fixtures: configuration missing from original revision: ' + name)
if configuration_paths.intersection(changed + extra):
    raise ValueError('comparison requires unchanged test configuration/fixtures: explicit pytest configuration')
# Test configuration and fixtures must be unchanged. Only selected test sources
# may be carried to the baseline to reproduce a newly written check.
for name in changed:
    if name not in paths and (not name.endswith('.py') or any(p.startswith('test') for p in Path(name).parts) or Path(name).name in ('conftest.py', 'setup.py')):
        raise ValueError('comparison requires unchanged test configuration/fixtures: ' + name)
ignored = shutil.ignore_patterns('.git', '.venv', '__pycache__', '.pytest_cache', 'experiment')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
results = {}
# Artifacts are temporary; the parent agent preserves the complete protocol in its trace.
cache = root / 'experiment' / 'cache' / 'verification'
if not within(cache, root):
    raise ValueError('comparison cache escapes repository')
cache.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=cache) as temporary:
    temporary = Path(temporary).resolve()
    seed = temporary / 'seed'
    shutil.copytree(root, seed, ignore=ignored, symlinks=True)
    work = temporary / 'repo'
    for label in ('baseline', 'candidate'):
        shutil.copytree(seed, work, symlinks=True)
        if label == 'baseline':
            for name in changed + extra:
                if name in paths:
                    continue
                target = work / name
                if not within(target, work):
                    raise ValueError('source path escapes comparison workspace')
                old = subprocess.run(['git', '-C', str(root), 'show', base + ':' + name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if old.returncode == 0:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(old.stdout)
                elif target.is_file():
                    target.unlink()
        if any(not (work / name).is_file() or not within(work / name, work) for name in configuration_paths):
            raise ValueError('explicit pytest configuration escapes comparison workspace')
        # The same interpreter, environment, argv, test bytes, path and ordering.
        # Each variant starts in a fresh copy and a fresh Python process.
        report = temporary / (label + '.xml')
        replay_root = work / test_root.relative_to(root) if test_root.is_relative_to(root) else work
        command = argv + ['--rootdir=' + str(replay_root), '-p', 'no:cacheprovider', '--junitxml=' + str(report)]
        if junit_prefix:
            command.append('--junitprefix=' + junit_prefix)
        run_env = dict(env)
        def map_path(value):
            path = Path(value or '.').resolve()
            if path.is_relative_to(root):
                relative = path.relative_to(root)
                if not relative.parts or relative.parts[0] not in ('.git', '.venv', '__pycache__', '.pytest_cache', 'experiment'):
                    return str(work / relative)
            return value
        if 'PYTHONPATH' in run_env:
            run_env['PYTHONPATH'] = os.pathsep.join(map_path(value) for value in run_env['PYTHONPATH'].split(os.pathsep))
        try:
            completed = subprocess.run(command, cwd=work, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, errors='replace', timeout=timeout)
            results[label] = dict(stdout=completed.stdout, stderr=completed.stderr, exit_code=completed.returncode,
                                  xml=report.read_text(encoding='utf-8') if report.exists() else '')
        except subprocess.TimeoutExpired:
            results[label] = dict(timed_out=True, xml='')
        assert work.parent == temporary and within(temporary, cache)
        shutil.rmtree(work)
print(json.dumps(results))
'''


async def compare_baseline(execute, command: str, root: str, base: str, timeout_ms: float) -> ShellResult:
    selected = pytest_selection(command, root)
    if not base or not selected:
        return ShellResult(error="Baseline comparison needs a captured clean base and explicit pytest files/nodes: `python -m pytest file.py::test_name -q`. Selectors such as -k/-m are unsupported; use the node ID or the whole file.")
    argv, paths = selected
    live = await execute("run_shell", {"command": command, "timeout": timeout_ms, "_pytest_report": True})
    if not isinstance(live, ShellResult) or live.test_outcome == "inconclusive":
        return ShellResult(error="Baseline comparison requires a completed test result in the working tree.")
    payload = json.dumps([root, base, argv, paths, timeout_ms / 1000])
    transport = await execute("run_shell", {"command": python_command(PAIR_SCRIPT, [payload], root, argv[0]),
                                           "timeout": timeout_ms * 2 + 60_000})
    if not isinstance(transport, ShellResult) or transport.failed:
        return replace(live, comparison_error="Baseline comparison could not run: " + str(transport))
    try:
        data = json.loads(transport.stdout)
        def result(row):
            row = dict(row)
            report = PytestReport.parse(row.pop("xml"))
            return ShellResult(**row, pytest_report=report)
        candidate = result(data["candidate"])
        mismatch = candidate.exit_code != live.exit_code or candidate.pytest_report != live.pytest_report or candidate.timed_out
        return replace(live, baseline=result(data["baseline"]), comparison_candidate=candidate, baseline_ref=base,
                       comparison_error="Candidate replay differed from the actual working tree." if mismatch else None)
    except (ValueError, KeyError, TypeError) as error:
        return replace(live, comparison_error="Invalid baseline execution protocol: " + str(error))
