"""Serialized paid research runs with trace-derived, idempotent usage settlement."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agents.telemetry import read_trace
from agents.agent import _BudgetExceeded

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "experiment/research/20261008-auto-research/state.json"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(4):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) != 5 or attempt == 3:
                raise
            # Windows readers can briefly hold the destination without delete sharing.
            time.sleep(0.05)


def usage_from_trace(events):
    """Count physical completions/errors; admission refusals did not reach the provider."""
    known = unknown = reserve = requests = 0
    pending = []
    for call in events:
        if call["event_type"] == "budget.reserve":
            pending.append(call)
            continue
        if (call["event_type"] not in {"model.call.end", "model.call.error"}
                or (call.get("attributes") or {}).get("error_type") == "_BudgetExceeded"):
            continue
        requests += 1
        matching = [row for row in pending if call.get("span_id") and row.get("span_id") == call["span_id"]]
        # This runner serializes provider calls. Summary reservations inherit the parent purpose.
        candidates = matching or pending
        reservation = candidates[-1] if len(candidates) == 1 else None
        total = ((call.get("metrics") or {}).get("provider_usage") or {}).get("total_tokens")
        if type(total) is int and total >= 0:
            known += total
        else:
            unknown += 1
            if reservation is None:
                raise ValueError("Unmetered request lacks an admission reservation")
            reserve += 2 * reservation["attributes"]["reserved"]
        if reservation is not None:
            pending.remove(reservation)
    return {"known_tokens": known, "physical_requests": requests,
            "unknown_usage_requests": unknown, "unknown_reserved_tokens": reserve}


class ResearchBudget:
    """One ledger writer; interrupted runs retain their lock and evidence for reconciliation."""

    def __init__(self, run_dir, trace, allocation, *, state_path=STATE, lock_path=None):
        if type(allocation) is not int or allocation <= 0:
            raise ValueError("allocation must be positive")
        self.run_dir, self.trace = Path(run_dir), trace
        self.state_path = Path(state_path)
        self.lock_path = Path(lock_path or ROOT / "experiment/cache/research-api.lock")
        self.allocation = allocation
        self.token_budget = None
        self.usage = {}
        self._owned = False

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        self._owned = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "run": str(self.run_dir), "trace": str(self.trace.trace_path)}, stream)
        try:
            self.initial = json.loads(self.state_path.read_text(encoding="utf-8"))
            coverage = self.initial.get("unknown_usage_reservations", [])
            if self.initial["unknown_usage_requests"] != sum(row["requests"] for row in coverage):
                raise ValueError("Unknown usage must retain a budget reservation")
            charged = self.initial["known_api_tokens"] + sum(row["reserved_tokens"] for row in coverage)
            limit = min(self.initial["additional_api_token_limit"], charged + self.allocation)
            if charged >= limit:
                raise ValueError("Research budget exhausted")
            if usage_from_trace(read_trace(self.trace.trace_path))["physical_requests"]:
                raise ValueError("A paid run must start with a fresh trace")
            self.token_budget = {"limit": limit, "charged": charged}
            save_json(self.run_dir / "budget-admission.json", {
                "initial_known_tokens": self.initial["known_api_tokens"], "initial_charged": charged,
                "allocation": limit - charged, "global_limit": self.initial["additional_api_token_limit"],
                "unknown_reservations": coverage})
            return self
        except BaseException:
            self.lock_path.unlink()
            self._owned = False
            raise

    async def before_call(self):
        self.settle()
        if self.usage["unknown_usage_requests"]:
            raise _BudgetExceeded("Unmetered provider call: stop this experiment")

    def settle(self):
        usage = usage_from_trace(read_trace(self.trace.trace_path))
        state = dict(self.initial)
        state["known_api_tokens"] += usage["known_tokens"]
        state["physical_requests"] += usage["physical_requests"]
        state["unknown_usage_requests"] += usage["unknown_usage_requests"]
        state["unknown_usage_reservations"] = list(state.get("unknown_usage_reservations", []))
        if usage["unknown_usage_requests"]:
            state["unknown_usage_reservations"].append({"run": str(self.run_dir),
                "requests": usage["unknown_usage_requests"], "reserved_tokens": usage["unknown_reserved_tokens"],
                "scope": "Conservative admission charge; observed usage remains unknown"})
        state["next_work"] = "Inspect executed Skill and Folding outcomes and retained failures"
        save_json(self.state_path, state)
        self.usage = usage
        save_json(self.run_dir / "usage.json", {**usage, "token_budget": self.token_budget})
        return usage

    def __exit__(self, exc_type, exc, traceback):
        # Leave the lock on failed settlement so no later run spends against a stale ledger.
        self.settle()
        if self._owned:
            self.lock_path.unlink()
            self._owned = False
