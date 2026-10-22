"""Versioned Skill activation with recoverable snapshots and guarded rollback."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from .session import atomic_write_json


def _digest(content: bytes | None) -> str:
    return hashlib.sha256(content).hexdigest() if content is not None else ""


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


@contextmanager
def skill_file_lock(path: Path):
    """OS locks are released on process exit; the lock file can remain in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Skill is being modified by another operation") from error
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _replace(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink()
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".candidate", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replace_skill_snapshot(path: Path, expected: str, content: bytes) -> None:
    with skill_file_lock(path):
        if _digest(_read(path)) != expected:
            raise ValueError("Skill changed since it was read")
        _replace(path, content)


def _record_transition(path: Path, before: bytes | None, after: bytes | None, **details) -> dict:
    from .skill_evolution import get_evolution_dir
    identity = uuid.uuid4().hex
    receipt = get_evolution_dir() / "activations" / f"{identity}.json"
    record = {"schema_version": 1, "id": identity, "time": datetime.now(timezone.utc).isoformat(),
              "workspace": str(Path.cwd().resolve()), "active_file": str(path.resolve()),
              "before_sha256": _digest(before), "after_sha256": _digest(after),
              "before_base64": base64.b64encode(before).decode("ascii") if before is not None else None,
              **details}
    # Retain recovery material before changing the active file. Current hashes
    # establish whether a prepared transition actually took effect after a crash.
    atomic_write_json(receipt, record)
    _replace(path, after)
    return {"activation_id": identity, "activation_receipt": str(receipt)}


def install_evaluated_skill(staged: dict, evaluation: dict, candidate_bytes: bytes) -> dict:
    if staged.get("workspace") != str(Path.cwd().resolve()):
        raise ValueError("candidate must be staged in this workspace")
    path = Path(staged["active_file"]).resolve()
    with skill_file_lock(path):
        before = _read(path)
        if _digest(before) != staged["baseline_sha256"] or _digest(candidate_bytes) != staged["candidate_sha256"]:
            raise ValueError("active or candidate snapshot changed after evaluation")
        return _record_transition(path, before, candidate_bytes, action="promote", skill=staged["skill"], evaluation=evaluation)


def rollback_skill_activation(identity: str) -> dict:
    from .skill_evolution import get_evolution_dir
    from .skills import reset_skill_cache
    try:
        if not re.fullmatch(r"[0-9a-f]{32}", identity):
            raise ValueError("invalid activation ID")
        record = json.loads((get_evolution_dir() / "activations" / f"{identity}.json").read_text(encoding="utf-8"))
        if (record.get("schema_version") != 1 or record.get("id") != identity
                or record.get("action") != "promote" or record.get("workspace") != str(Path.cwd().resolve())):
            raise ValueError("activation does not belong to this workspace")
        previous = base64.b64decode(record["before_base64"], validate=True) if record["before_base64"] is not None else None
        if _digest(previous) != record["before_sha256"]:
            raise ValueError("original Skill snapshot is corrupt")
        path = Path(record["active_file"])
        with skill_file_lock(path):
            current = _read(path)
            if _digest(current) != record["after_sha256"]:
                raise ValueError("active Skill changed after this activation")
            restored = _record_transition(path, current, previous, action="rollback", skill=record["skill"], rollback_of=identity)
        reset_skill_cache()
        return {"ok": True, "action": "rollback", "skill": record["skill"], "rollback_of": identity, **restored}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {"ok": False, "action": "rollback", "error": str(error)}


def list_skill_activations() -> list[dict]:
    from .skill_evolution import get_evolution_dir
    rows = []
    for path in (get_evolution_dir() / "activations").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["workspace"] != str(Path.cwd().resolve()):
                continue
            current = _digest(_read(Path(record["active_file"])))
            status = "active" if current == record["after_sha256"] else "superseded"
            if current == record["before_sha256"]:
                status = "not_applied_or_reverted"
            rows.append({key: record[key] for key in ("id", "time", "skill", "action")} | {"status": status})
        except (OSError, ValueError, KeyError, TypeError):
            rows.append({"id": path.stem, "time": "", "skill": "unknown", "action": "unknown", "status": "invalid_receipt"})
    return sorted(rows, key=lambda row: row["time"], reverse=True)
