#!/bin/bash
set -u
cd /testbed
mkdir -p /logs/verifier
python -m pip check > /logs/verifier/environment.log 2>&1 || exit 1
python - <<'PY'
import json, pathlib, subprocess, sys, pytest, slugify
root=pathlib.Path.cwd()
assert pathlib.Path(slugify.__file__).resolve().is_relative_to(root)
def git(*args):
    return subprocess.check_output(['git',*args],text=True).strip()
facts={'python':sys.version,'pytest':pytest.__version__,'import_path':slugify.__file__,
       'git_version':git('--version'),'history_count':int(git('rev-list','--count','HEAD')),
       'tree':git('rev-parse','HEAD^{tree}'),'dirty':bool(git('status','--porcelain'))}
pathlib.Path('/logs/verifier/environment.json').write_text(json.dumps(facts,indent=2))
PY
if [ "$?" -ne 0 ]; then exit 1; fi
python -m pytest test.py -q --junitxml=/logs/verifier/existing.xml
existing=$?
python -m pytest /tests/upstream_test.py -q --junitxml=/logs/verifier/junit.xml
regression=$?
printf '%s %s\n' "$existing" "$regression" > /logs/verifier/exit-codes.txt
if [ "$existing" -eq 0 ] && [ "$regression" -eq 0 ]; then
    printf '1\n' > /logs/verifier/reward.txt
else
    printf '0\n' > /logs/verifier/reward.txt
fi
