"""Build the runtime wheel and exercise its installed CLI outside the source tree."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = ROOT / "experiment/cache"
    cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="install-", dir=cache))
    env = {**os.environ, "TMP": str(work), "TEMP": str(work), "TMPDIR": str(work),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    env.pop("PYTHONPATH", None)
    log = work / "install.log"

    def run(args, cwd=work):
        result = subprocess.run([str(arg) for arg in args], cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        with log.open("a", encoding="utf-8") as stream:
            stream.write(result.stdout)
        if result.returncode:
            raise RuntimeError(f"Installation check failed; see {log}")
        return result.stdout.strip()

    dist = work / "dist"
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", dist, ROOT])
    wheel, = dist.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "agents/main.py" in names
        assert all(name.startswith("agents/") or ".dist-info/" in name for name in names), names
        assert not any(name.endswith(".pyc") or "__pycache__" in name for name in names)
    installed = work / "venv"
    venv.EnvBuilder(with_pip=True).create(installed)
    binary = installed / ("Scripts" if os.name == "nt" else "bin")
    python = binary / ("python.exe" if os.name == "nt" else "python")
    cli = binary / ("sans.exe" if os.name == "nt" else "sans")
    run([python, "-m", "pip", "install", "-r", ROOT / "requirements.lock.txt", wheel])
    origin = run([python, "-I", "-c", "import agents; print(agents.__file__)"])
    assert Path(origin).resolve().is_relative_to(installed.resolve()), origin
    version = run([cli, "--version"])
    help_text = run([cli, "--help"])
    assert "--max-model-calls" in help_text and "/skill-candidates" in help_text
    receipt = {"status": "passed", "version": version, "installed_origin": origin,
               "wheel": str(wheel.relative_to(ROOT)), "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
               "api_calls": 0, "api_tokens": 0}
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**receipt, "receipt": str(work / "receipt.json")}))


if __name__ == "__main__":
    main()
