"""Python diff checking covers untracked whitespace without changing the real index."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from agents.verification import DIFF_CHECK_SCRIPT


def test_diff_checker_retains_staging_and_checks_untracked_files(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'existing.py').write_text('value = 1\n', encoding='utf-8')
    for arguments in (['init', '-q'], ['add', '.'],
                      ['-c', 'user.name=DiffTest', '-c', 'user.email=diff@example.invalid', 'commit', '-qm', 'baseline']):
        subprocess.run(['git', *arguments], cwd=repo, capture_output=True, check=True)
    (repo / 'existing.py').write_text('value = 2\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'existing.py'], cwd=repo, check=True)
    original = (repo / '.git/index').read_bytes()
    (repo / 'untracked.py').write_text('invalid = 1  \n', encoding='utf-8')
    failed = subprocess.run([sys.executable, '-B', '-c', DIFF_CHECK_SCRIPT], cwd=repo,
                            capture_output=True, text=True, timeout=15)
    assert failed.returncode != 0 and 'untracked.py' in failed.stdout
    assert (repo / '.git/index').read_bytes() == original
    (repo / 'untracked.py').write_text('valid = 1\n', encoding='utf-8')
    passed = subprocess.run([sys.executable, '-B', '-c', DIFF_CHECK_SCRIPT], cwd=repo,
                            capture_output=True, text=True, timeout=15)
    assert passed.returncode == 0
    assert (repo / '.git/index').read_bytes() == original
    assert not list((repo / '.git').glob('sans-diff-*'))
