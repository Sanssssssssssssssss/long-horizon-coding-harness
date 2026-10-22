
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from typing import Any
import json
import os
import tempfile

SESSION_DIR = Path.home() / ".sans" / "sessions"



def get_project_session_dir() -> Path:
    d = Path.cwd() / "experiment" / "runtime" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    payload = (json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_session(session_id: str, data: dict[str, Any]) -> None:
    if not session_id or Path(session_id).name != session_id or session_id in {".", ".."}:
        raise ValueError("invalid session id")
    atomic_write_json(get_project_session_dir() / f"{session_id}.json", data)


def save_folded_session_memory(session_id: str, record: dict[str, Any]) -> None:
    d = get_project_session_dir()
    line = json.dumps(record, ensure_ascii=False, default=str)
    with (d / f"{session_id}.folded-memory.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    (d / f"{session_id}.folded-memory.latest.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_session(session_id: str) -> dict[str, Any] | None:
    if not session_id or Path(session_id).name != session_id:
        return None
    path = get_project_session_dir() / f"{session_id}.json"
    if not path.exists():
        path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cwd = data.get("metadata", {}).get("cwd")
        return data if cwd and Path(cwd).resolve() == Path.cwd().resolve() else None
    except Exception:
        return None


def list_sessions() -> list[dict[str, Any]]:
    results = {}
    for directory in (SESSION_DIR, get_project_session_dir()):
        for path in directory.glob("*.json"):
            data = load_session(path.stem)
            if data and data["metadata"].get("id") == path.stem:
                results[path.stem] = data["metadata"]
    return list(results.values())


def get_latest_session_id() -> str | None:
    sessions = list_sessions()
    if not sessions:
        return None
    sessions.sort(key=lambda s: s.get("startTime", ""), reverse=True)
    return sessions[0].get("id")
