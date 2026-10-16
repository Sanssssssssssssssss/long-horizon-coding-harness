"""Runtime-owned state for evidence-backed repository repair tasks."""

from __future__ import annotations

import uuid

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from difflib import get_close_matches
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
_CASE_SOURCES = {"issue", "project_rule", "existing_test"}
_HYPOTHESIS_STATUSES = {"active", "supported", "rejected"}
_EVIDENCE_STATUSES = {"ok", "error"}
_VERIFICATION_SCOPES = {"focused", "module", "related", "full", "diff"}
_ORACLES = {
    "diff_check",
    "issue_reproduction",
    "existing_test",
    "metamorphic",
    "sibling_implementation",
    "new_regression",
}


def _required_text(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(values: Sequence[object], name: str) -> tuple[str, ...]:
    result = tuple(_required_text(value, name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _operation_key(tool: str, input_summary: str) -> str:
    return sha256(f"{tool}\0{input_summary}".encode("utf-8")).hexdigest()


def preview_output(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit // 2] + "\n...[omitted]...\n" + text[-limit // 2:]


@dataclass(frozen=True, slots=True)
class TaskEnvelope:
    task_id: str
    canonical_issue: str
    repo_root: str
    requires_patch: bool = True
    requires_repair_witness: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _required_text(self.task_id, "task_id"))
        object.__setattr__(
            self, "canonical_issue", _required_text(self.canonical_issue, "canonical_issue")
        )
        object.__setattr__(self, "repo_root", _required_text(self.repo_root, "repo_root"))
        if type(self.requires_patch) is not bool:
            raise ValueError("requires_patch must be a boolean")
        if type(self.requires_repair_witness) is not bool:
            raise ValueError("requires_repair_witness must be a boolean")

    @property
    def issue_sha256(self) -> str:
        return sha256(self.canonical_issue.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AcceptanceCase:
    id: str
    behavior: str
    source_quote: str
    source_kind: str = "issue"
    source_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "acceptance id"))
        object.__setattr__(self, "behavior", _required_text(self.behavior, "behavior"))
        object.__setattr__(
            self, "source_quote", _required_text(self.source_quote, "source_quote")
        )
        if self.source_kind not in _CASE_SOURCES:
            raise ValueError(f"unsupported acceptance source: {self.source_kind}")
        object.__setattr__(self, "source_ref", _optional_text(self.source_ref))


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    claim: str
    prediction: str
    falsifier: str
    status: str
    evidence_ids: tuple[str, ...]
    revision: int = 1
    repo_generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "hypothesis id"))
        object.__setattr__(self, "claim", _required_text(self.claim, "claim"))
        object.__setattr__(self, "prediction", _required_text(self.prediction, "prediction"))
        object.__setattr__(self, "falsifier", _required_text(self.falsifier, "falsifier"))
        if self.status not in _HYPOTHESIS_STATUSES:
            raise ValueError(f"unsupported hypothesis status: {self.status}")
        object.__setattr__(self, "evidence_ids", _string_tuple(self.evidence_ids, "evidence_ids"))
        if self.status != "active" and not self.evidence_ids:
            raise ValueError(f"{self.status} hypotheses require evidence")


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    id: str
    sequence: int
    tool: str
    status: str
    exit_code: int | None
    duration_ms: float
    repo_before: str | None
    repo_after: str | None
    repo_generation: int
    input_summary: str
    operation_key: str
    output_preview: str
    check_fingerprint: str | None = None
    test_failure_observed: bool = False
    test_pass_observed: bool = False
    unchanged_baseline_failure: bool = False
    diagnostic: bool = False
    test_cases: tuple[tuple[str, str, str], ...] = ()
    execution_started: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "evidence id"))
        object.__setattr__(self, "tool", _required_text(self.tool, "tool"))
        object.__setattr__(
            self, "input_summary", _required_text(self.input_summary, "input_summary")
        )
        object.__setattr__(
            self, "operation_key", _required_text(self.operation_key, "operation_key")
        )
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence status: {self.status}")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("evidence sequence must be a positive integer")
        if type(self.repo_generation) is not int or self.repo_generation < 0:
            raise ValueError("evidence repository generation must be non-negative")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, (int, float)):
            raise ValueError("evidence duration must be numeric")
        if self.duration_ms < 0:
            raise ValueError("evidence counters and duration must be non-negative")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("exit_code must be an integer or None")
        if self.execution_started is not None and type(self.execution_started) is not bool:
            raise ValueError("execution_started must be a boolean or None")

    @property
    def succeeded(self) -> bool:
        return self.status == "ok" and self.exit_code in {None, 0}


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    id: str
    sequence: int
    evidence_id: str
    acceptance_ids: tuple[str, ...]
    scope: str
    oracle: str
    success: bool
    repo_state: str
    repo_generation: int
    operation_key: str
    runner_recognized: bool
    check_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "verification id"))
        object.__setattr__(self, "evidence_id", _required_text(self.evidence_id, "evidence_id"))
        object.__setattr__(
            self, "acceptance_ids", _string_tuple(self.acceptance_ids, "acceptance_ids")
        )
        object.__setattr__(self, "repo_state", _required_text(self.repo_state, "repo_state"))
        object.__setattr__(
            self, "operation_key", _required_text(self.operation_key, "operation_key")
        )
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("verification sequence must be a positive integer")
        if type(self.repo_generation) is not int or self.repo_generation < 0:
            raise ValueError("verification repository generation must be non-negative")
        if self.scope not in _VERIFICATION_SCOPES or self.oracle not in _ORACLES:
            raise ValueError("unsupported verification metadata")
        if (self.scope == "diff") != (self.oracle == "diff_check"):
            raise ValueError("diff scope and diff_check oracle must be used together")
        if type(self.success) is not bool or type(self.runner_recognized) is not bool:
            raise ValueError("verification flags must be booleans")


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    complete: bool
    reasons: tuple[str, ...]
    unverified_acceptance_ids: tuple[str, ...]
    unresolved_verification_ids: tuple[str, ...]


class RepairState:
    """One task's facts, with completion derived only from runtime receipts."""

    def __init__(self, task: TaskEnvelope, initial_repo_state: str | None) -> None:
        if not isinstance(task, TaskEnvelope):
            raise TypeError("task must be a TaskEnvelope")
        self._task = task
        self._execution_id = uuid.uuid4().hex
        self._initial_repo_state = _optional_text(initial_repo_state)
        self._current_repo_state = self._initial_repo_state
        self._repo_generation = 0
        self._cases: dict[str, AcceptanceCase] = {}
        self._hypotheses: dict[str, Hypothesis] = {}
        self._hypothesis_history: list[Hypothesis] = []
        self._evidence: list[EvidenceReceipt] = []
        self._verifications: list[VerificationReceipt] = []
        self._retired_diagnostics: dict[str, str] = {}

    @property
    def task(self) -> TaskEnvelope:
        return self._task

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def current_repo_state(self) -> str | None:
        return self._current_repo_state

    @property
    def repo_generation(self) -> int:
        return self._repo_generation

    @property
    def acceptance_cases(self) -> Mapping[str, AcceptanceCase]:
        return MappingProxyType(self._cases)

    @property
    def hypotheses(self) -> Mapping[str, Hypothesis]:
        return MappingProxyType(self._hypotheses)

    @property
    def evidence(self) -> tuple[EvidenceReceipt, ...]:
        return tuple(self._evidence)

    @property
    def verifications(self) -> tuple[VerificationReceipt, ...]:
        return tuple(self._verifications)

    def register_acceptance_case(
        self,
        case_id: str,
        behavior: str,
        source_quote: str,
        *,
        source_kind: str = "issue",
        source_ref: str | None = None,
    ) -> AcceptanceCase:
        case = AcceptanceCase(case_id, behavior, source_quote, source_kind, source_ref)
        if case.source_kind == "issue" and case.source_quote not in self.task.canonical_issue:
            canonical = self.task.canonical_issue
            match = re.search(r"\s+".join(re.escape(word) for word in case.source_quote.split()), canonical)
            near = match.group(0) if match else next(iter(get_close_matches(
                case.source_quote, canonical.splitlines(), n=1, cutoff=0.4)), "")
            hint = (f" Canonical excerpt: {near[:400]!r}. Copy exact text, preserve line breaks (JSON \\n)."
                    if near else " Copy an exact canonical_issue substring; environment labels are not issue quotes.")
            raise ValueError(
                f"acceptance {case.id!r}: issue acceptance source_quote must occur in canonical_issue exactly; "
                f"source_quote={case.source_quote[:240]!r}.{hint}"
            )
        existing = self._cases.get(case.id)
        if existing is not None:
            if existing != case:
                raise ValueError(f"acceptance case {case.id!r} is immutable once registered")
            return existing
        self._cases[case.id] = case
        return case

    def upsert_hypothesis(
        self,
        hypothesis_id: str,
        claim: str,
        prediction: str,
        falsifier: str,
        *,
        status: str = "active",
        evidence_ids: Sequence[str] = (),
    ) -> Hypothesis:
        hypothesis_id = _required_text(hypothesis_id, "hypothesis id")
        references = _string_tuple(evidence_ids, "evidence_ids")
        missing = [evidence_id for evidence_id in references if evidence_id not in self._evidence_by_id()]
        if missing:
            raise ValueError(f"unknown evidence id(s): {', '.join(missing)}")
        existing = self._hypotheses.get(hypothesis_id)
        changed_claim = existing is not None and (claim, prediction, falsifier) != (
            existing.claim, existing.prediction, existing.falsifier,
        )
        if changed_claim and status != "active" and not set(references).difference(existing.evidence_ids):
            raise ValueError("revised hypothesis requires new evidence or active status")
        hypothesis = Hypothesis(
            hypothesis_id, claim, prediction, falsifier, status, references,
            revision=existing.revision + 1 if existing else 1,
            repo_generation=self._repo_generation,
        )
        self._hypotheses[hypothesis.id] = hypothesis
        self._hypothesis_history.append(hypothesis)
        return hypothesis

    def observe_repo_state(self, repo_state: str) -> bool:
        observed = _required_text(repo_state, "repo_state")
        if self._current_repo_state is None:
            if self._initial_repo_state is None:
                self._initial_repo_state = observed
            self._current_repo_state = observed
            return False
        if observed == self._current_repo_state:
            return False
        self._current_repo_state = observed
        self._repo_generation += 1
        return True

    def record_evidence(
        self,
        *,
        tool: str,
        status: str,
        input_summary: str,
        exit_code: int | None = None,
        duration_ms: float = 0.0,
        repo_before: str | None = None,
        repo_after: str | None = None,
        output_preview: str = "",
        check_fingerprint: str | None = None,
        test_failure_observed: bool = False,
        test_pass_observed: bool = False,
        unchanged_baseline_failure: bool = False,
        diagnostic: bool = False,
        test_cases: tuple[tuple[str, str, str], ...] = (),
        execution_started: bool | None = None,
    ) -> EvidenceReceipt:
        tool = _required_text(tool, "tool")
        input_summary = _required_text(input_summary, "input_summary")
        if status not in _EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence status: {status}")
        before = _optional_text(repo_before)
        after = _optional_text(repo_after)
        if self._current_repo_state is None and before is not None:
            if self._initial_repo_state is None:
                self._initial_repo_state = before
            self._current_repo_state = before
        if before is not None and self._current_repo_state is not None:
            if before != self._current_repo_state:
                raise ValueError("repo_before does not match current_repo_state")
        if after is None and tool in {"write_file", "edit_file", "run_shell"}:
            self._current_repo_state = None
            self._repo_generation += 1
        elif after is None:
            after = before or self._current_repo_state
        if after is not None:
            self.observe_repo_state(after)

        sequence = len(self._evidence) + 1
        receipt = EvidenceReceipt(
            id=f"{self.execution_id}:e{sequence:04d}",
            sequence=sequence,
            tool=tool,
            status=status,
            exit_code=exit_code,
            duration_ms=float(duration_ms),
            repo_before=before,
            repo_after=self._current_repo_state,
            repo_generation=self._repo_generation,
            input_summary=input_summary,
            operation_key=_operation_key(tool, input_summary),
            output_preview=str(output_preview or ""),
            check_fingerprint=check_fingerprint,
            test_failure_observed=test_failure_observed,
            test_pass_observed=test_pass_observed,
            unchanged_baseline_failure=unchanged_baseline_failure,
            diagnostic=diagnostic,
            test_cases=test_cases,
            execution_started=execution_started,
        )
        self._evidence.append(receipt)
        return receipt

    def record_verification(
        self,
        evidence_id: str,
        acceptance_ids: Sequence[str],
        *,
        scope: str,
        oracle: str,
        runner_recognized: bool = False,
    ) -> VerificationReceipt:
        evidence_id = _required_text(evidence_id, "evidence_id")
        evidence = self._evidence_by_id().get(evidence_id)
        if evidence is None:
            raise ValueError(f"unknown evidence id: {evidence_id}")
        if evidence.tool != "run_shell":
            raise ValueError("verification evidence must come from run_shell")
        if evidence.exit_code is None:
            raise ValueError("run_shell verification evidence requires an explicit exit_code")
        if evidence.repo_before is None or evidence.repo_after is None:
            raise ValueError("verification evidence requires known repo_before and repo_after")
        if evidence.repo_before != evidence.repo_after:
            raise ValueError("verification command must not change repository state")
        case_ids = _string_tuple(acceptance_ids, "acceptance_ids")
        if scope not in _VERIFICATION_SCOPES:
            raise ValueError(f"unsupported verification scope: {scope}")
        if oracle not in _ORACLES:
            raise ValueError(f"unsupported verification oracle: {oracle}")
        if scope == "diff":
            if oracle != "diff_check":
                raise ValueError("diff verification requires the diff_check oracle")
        elif oracle == "diff_check":
            raise ValueError("diff_check oracle requires diff verification scope")
        # Empty acceptance_ids are runtime-owned, unscoped checks. They can block on red
        # and be cleared by an exact rerun, but never prove an acceptance case.
        missing = [case_id for case_id in case_ids if case_id not in self._cases]
        if missing:
            raise ValueError(f"unknown acceptance id(s): {', '.join(missing)}")
        if self._current_repo_state is None:
            raise ValueError("verification requires a known repo state")
        if evidence.repo_generation != self._repo_generation:
            raise ValueError("cannot verify with stale evidence")

        sequence = len(self._verifications) + 1
        receipt = VerificationReceipt(
            id=f"v{sequence:04d}",
            sequence=sequence,
            evidence_id=evidence.id,
            acceptance_ids=case_ids,
            scope=scope,
            oracle=oracle,
            success=evidence.succeeded,
            repo_state=self._current_repo_state,
            repo_generation=self._repo_generation,
            operation_key=evidence.operation_key,
            runner_recognized=runner_recognized,
            check_fingerprint=evidence.check_fingerprint,
        )
        self._verifications.append(receipt)
        return receipt

    def completion_decision(self) -> CompletionDecision:
        reasons: list[str] = []
        unverified: list[str] = []
        unresolved_ids: list[str] = []
        current_checks = [
            receipt for receipt in self._verifications
            if receipt.scope != "diff"
            and receipt.repo_generation == self._repo_generation
            and receipt.repo_state == self._current_repo_state
        ]
        if self._current_repo_state is None:
            reasons.append("repository state is unknown")
        if self.task.requires_patch and self._initial_repo_state is None:
            reasons.append("initial repository state is unknown")
        elif self.task.requires_patch and self._current_repo_state == self._initial_repo_state:
            reasons.append("repository has no net change from initial state")
        if not self._cases:
            reasons.append("no acceptance cases are registered")
        elif not any(case.source_kind == "issue" for case in self._cases.values()):
            reasons.append("no issue-derived acceptance case is registered")

        for case_id in self._cases:
            receipts = [
                receipt
                for receipt in self._verifications
                if case_id in receipt.acceptance_ids
                and receipt.scope != "diff"
            ]
            failures = self._unresolved_failures(receipts, current_checks)
            if failures:
                unresolved_ids.extend(receipt.id for receipt in failures)
                reasons.append(
                    f"acceptance {case_id} has unresolved verification failure(s): "
                    + ", ".join(receipt.id for receipt in failures)
                )
            successes = [
                receipt for receipt in receipts
                if receipt.success and receipt.runner_recognized
                and receipt in current_checks
            ]
            if not successes:
                unverified.append(case_id)
                reasons.append(
                    f"acceptance {case_id} has no current successful recognized test"
                )

        unscoped = [
            receipt
            for receipt in self._verifications
            if not receipt.acceptance_ids
            and receipt.scope != "diff"
        ]
        failed_unscoped = self._unresolved_failures(unscoped, current_checks)
        if failed_unscoped:
            unresolved_ids.extend(receipt.id for receipt in failed_unscoped)
            reasons.append(
                "unscoped verification pending recheck: "
                + ", ".join(receipt.id for receipt in failed_unscoped)
            )

        if not any(receipt.success and receipt.runner_recognized for receipt in current_checks):
            reasons.append("current repository state has no successful recognized test runner")

        # A model-provided oracle label cannot promote a green-only check to repair evidence.
        # No generation-0 requirement: a newly written reproducer may first fail at any generation.
        if self.task.requires_patch and self.task.requires_repair_witness and not self._repair_witnesses([
            receipt for receipt in current_checks if receipt.acceptance_ids
        ]):
            reasons.append(
                "no observed failing-to-passing check with unchanged test conditions and the "
                "original failed cases passing; rerun the same focused pytest cases before and "
                "after the repair. Missing, skipped or ambiguous testcase identities cannot "
                "prove a repair (a reproducer may be created first)"
            )

        if self.task.requires_patch:
            diff_receipts = [
                receipt
                for receipt in self._verifications
                if receipt.scope == "diff"
                and receipt.oracle == "diff_check"
                and receipt.repo_generation == self._repo_generation
                and receipt.repo_state == self._current_repo_state
            ]
            failed_diffs = self._unresolved_failures(diff_receipts)
            if failed_diffs:
                unresolved_ids.extend(receipt.id for receipt in failed_diffs)
                reasons.append(
                    "final diff has unresolved failure(s): "
                    + ", ".join(receipt.id for receipt in failed_diffs)
                )
            if not any(receipt.success for receipt in diff_receipts):
                reasons.append("current repository state has no successful diff verification")

        return CompletionDecision(
            complete=not reasons,
            reasons=tuple(reasons),
            unverified_acceptance_ids=tuple(dict.fromkeys(unverified)),
            unresolved_verification_ids=tuple(dict.fromkeys(unresolved_ids)),
        )

    def _unresolved_failures(
        self,
        receipts: Sequence[VerificationReceipt],
        resolutions: Sequence[VerificationReceipt] | None = None,
    ) -> list[VerificationReceipt]:
        resolutions = receipts if resolutions is None else resolutions
        evidence = self._evidence_by_id()
        replacements: dict[str, int] = {}
        for pair in self._repair_witnesses(resolutions):
            replacements[pair["passed"]] = max(
                replacements.get(pair["passed"], 0), evidence[pair["failed"]].sequence,
            )
        failures: list[VerificationReceipt] = []
        for failed in (receipt for receipt in receipts if not receipt.success):
            original = evidence[failed.evidence_id]
            if original.diagnostic and self._retired_diagnostics.get(original.id) == self._current_repo_state:
                continue
            if any(
                evidence[checked.evidence_id].unchanged_baseline_failure
                and checked.operation_key == failed.operation_key
                and checked.check_fingerprint == failed.check_fingerprint
                for checked in resolutions
            ):
                continue
            resolved = any(
                passed.success
                and passed.sequence > failed.sequence
                and passed.operation_key == failed.operation_key
                and (
                    failed.scope == "diff" or not original.test_failure_observed
                    or (self._failed_cases_passed(original, evidence[passed.evidence_id])
                        and ((failed.check_fingerprint
                              and passed.check_fingerprint == failed.check_fingerprint)
                             # A revised test must establish its own newer failing baseline.
                             or replacements.get(passed.evidence_id, 0) >= original.sequence))
                )
                for passed in resolutions
            )
            if not resolved:
                failures.append(failed)
        return failures

    def retire_diagnostic(self, evidence_id: str) -> None:
        """Called only after the runtime verifies removal of a predeclared new probe."""
        evidence = self._evidence_by_id()[evidence_id]
        if not evidence.diagnostic or self._current_repo_state is None:
            raise ValueError("only a runtime-verified diagnostic can be retired")
        self._retired_diagnostics[evidence_id] = self._current_repo_state

    @staticmethod
    def _failed_cases_passed(failed: EvidenceReceipt, passed: EvidenceReceipt) -> bool:
        if not failed.test_failure_observed or not passed.test_pass_observed:
            return False
        before = {identity: status for identity, status, _ in failed.test_cases}
        after = {identity: status for identity, status, _ in passed.test_cases}
        # Duplicate identities make it impossible to match individual executions.
        if len(before) != len(failed.test_cases) or len(after) != len(passed.test_cases):
            return False
        failures = [identity for identity, status in before.items() if status in {"failure", "error"}]
        return bool(failures) and all(
            identity.strip(": \t\r\n") and after.get(identity) == "passed"
            for identity in failures
        )

    def _repair_witnesses(self, current_checks: Sequence[VerificationReceipt]) -> list[dict[str, str]]:
        evidence = self._evidence_by_id()
        return [
            {"failed": failed.evidence_id, "passed": passed.evidence_id}
            for passed in current_checks if passed.success and passed.runner_recognized
            and passed.check_fingerprint
            and not evidence[passed.evidence_id].diagnostic
            for failed in self._verifications
            if not failed.success and failed.runner_recognized
            and self._failed_cases_passed(evidence[failed.evidence_id], evidence[passed.evidence_id])
            and not evidence[failed.evidence_id].diagnostic
            and failed.repo_generation < passed.repo_generation
            and failed.operation_key == passed.operation_key
            and failed.check_fingerprint == passed.check_fingerprint
        ]

    def render_runtime_context(self, *, latest: int = 4) -> str:
        """Render a bounded factual state receipt for tail-of-request injection."""
        if type(latest) is not int or latest <= 0:
            raise ValueError("latest must be a positive integer")
        decision = self.completion_decision()
        runtime_diff_reason = "current repository state has no successful diff verification"
        runtime_diff_pending = runtime_diff_reason in decision.reasons
        model_blockers = tuple(
            reason for reason in decision.reasons if reason != runtime_diff_reason
        )
        finish_gate: dict[str, Any] = {
            "ready": not model_blockers,
            "blockers": model_blockers,
        }
        if runtime_diff_pending and not model_blockers:
            finish_gate.update({
                "action": "finish_when_verified",
                "instruction": (
                    "Recorded obligations are satisfied. Before finishing, verify that your "
                    "assertions reject a plausible wrong fix and run a relevant sibling or "
                    "boundary check if still unchecked. If these checks already passed on "
                    "the final code, finish without repeating them; the stop hook will run "
                    "the runtime-owned final diff check."
                ),
            })
        hypotheses = list(self._hypotheses.values())[-3:]
        evidence_by_id = self._evidence_by_id()
        unresolved_ids = set(decision.unresolved_verification_ids)
        red_verifications: dict[str, list[str]] = {}
        for receipt in self._verifications:
            if receipt.id in unresolved_ids:
                red_verifications.setdefault(receipt.evidence_id, []).append(receipt.id)
        red_evidence_ids = set(red_verifications)
        recent_evidence = [
            receipt for receipt in self._evidence if receipt.id not in red_evidence_ids
        ][-latest:]
        recent_verifications = [
            receipt for receipt in self._verifications if receipt.id not in unresolved_ids
        ][-latest:]
        payload = {
            "task_mode": ("verification" if not self.task.requires_patch else
                          "repair" if self.task.requires_repair_witness else "artifact_build"),
            "check_requirements": {
                "authority": "Caller-owned task requirements; model hypotheses and labels remain claims.",
                "requires_patch": self.task.requires_patch,
                "requires_repair_witness": self.task.requires_patch and self.task.requires_repair_witness,
                "current_recognized_test_for_each_acceptance": True,
                "unresolved_failures_block": True,
                "verification_preserves_repository_state": True,
                "final_diff": self.task.requires_patch,
            },
            "unchanged_baseline_failures": [
                {"evidence_id": e.id, "input": e.input_summary, "outcome": "failed"}
                for e in self._evidence if e.unchanged_baseline_failure
                and e.repo_generation == self._repo_generation and e.repo_after == self._current_repo_state
            ],
            "canonical_task_sha256": self.task.issue_sha256,
            "acceptance": [
                {
                    "id": case.id,
                    "behavior": case.behavior,
                    "source_quote": case.source_quote,
                }
                for case in self._cases.values()
            ],
            "hypotheses": [
                {
                    "id": hypothesis.id,
                    "claim": hypothesis.claim,
                    "prediction": hypothesis.prediction,
                    "falsifier": hypothesis.falsifier,
                    "claimed_status": hypothesis.status,
                    "evidence_ids": hypothesis.evidence_ids,
                    "revision": hypothesis.revision,
                    "evidence_stale": any(
                        evidence_by_id[ref].repo_generation != self._repo_generation
                        for ref in hypothesis.evidence_ids
                    ),
                }
                for hypothesis in hypotheses
            ],
            "unresolved_reds": [
                {
                    "verification_ids": red_verifications[evidence_id],
                    "evidence_id": evidence_id,
                    "operation_key": evidence.operation_key,
                    "input": evidence.input_summary[:240],
                    "exit_code": evidence.exit_code,
                    "repo_generation": evidence.repo_generation,
                    "state": "failed" if evidence.repo_generation == self._repo_generation else "pending_recheck",
                    **({"diagnostic": True} if evidence.diagnostic else {}),
                    **({"failed_test_cases": [identity for identity, status, _ in evidence.test_cases
                                              if status in {"failure", "error"}][:10]}
                       if evidence.test_failure_observed else {}),
                    "output": preview_output(evidence.output_preview),
                }
                for evidence_id in red_verifications
                if (evidence := evidence_by_id.get(evidence_id)) is not None
            ],
            "latest_evidence": [
                {
                    "id": receipt.id,
                    "tool": receipt.tool,
                    "status": receipt.status,
                    "exit_code": receipt.exit_code,
                    "repo_generation": receipt.repo_generation,
                    "input": receipt.input_summary[:240],
                    "output": preview_output(receipt.output_preview),
                    "unchanged_baseline_failure": receipt.unchanged_baseline_failure,
                    "diagnostic": receipt.diagnostic,
                }
                for receipt in recent_evidence
            ],
            "latest_verifications": [
                {
                    "id": receipt.id,
                    "evidence_id": receipt.evidence_id,
                    "claimed_acceptance_ids": receipt.acceptance_ids,
                    "scope": receipt.scope,
                    "oracle": receipt.oracle,
                    "success": receipt.success,
                    "runner_recognized": receipt.runner_recognized,
                    "test_failure_observed": evidence_by_id[receipt.evidence_id].test_failure_observed,
                    "test_pass_observed": evidence_by_id[receipt.evidence_id].test_pass_observed,
                    "repo_generation": receipt.repo_generation,
                }
                for receipt in recent_verifications
            ],
            "finish_gate": finish_gate,
            "repair_witnesses": self._repair_witnesses([
                receipt for receipt in self._verifications
                if receipt.repo_generation == self._repo_generation
                and receipt.repo_state == self._current_repo_state
                and receipt.acceptance_ids
            ])[-3:],
            "observation_reader": "Use read_observation(evidence_id, offset, limit) for complete paged output.",
            "hypothesis_authority": "Model claims, not runtime-verified explanations; revisions are retained in the snapshot.",
            "recent_hypothesis_revisions": [
                {"id": item.id, "revision": item.revision, "claim": item.claim,
                 "prediction": item.prediction, "claimed_status": item.status}
                for item in self._hypothesis_history
                if item != self._hypotheses[item.id]
            ][-3:],
        }
        actionable = [item.id for item in hypotheses if item.status == "active"]
        changed_checks = []
        for failed in self._verifications:
            if (failed.id not in unresolved_ids or not failed.runner_recognized
                or not failed.check_fingerprint or evidence_by_id[failed.evidence_id].diagnostic):
                continue
            for passed in reversed(self._verifications):
                if (passed.sequence <= failed.sequence or not passed.runner_recognized
                    or passed.repo_generation != self._repo_generation
                    or passed.repo_state != self._current_repo_state
                    or passed.operation_key != failed.operation_key):
                    continue
                if (not passed.success or not passed.check_fingerprint
                    or passed.check_fingerprint == failed.check_fingerprint
                    or evidence_by_id[passed.evidence_id].diagnostic
                    or not self._failed_cases_passed(
                        evidence_by_id[failed.evidence_id], evidence_by_id[passed.evidence_id])):
                    break
                changed_checks.append({
                    "failed_verification_id": failed.id,
                    "current_passed_verification_id": passed.id,
                    "original_check_fingerprint": failed.check_fingerprint,
                    "current_check_fingerprint": passed.check_fingerprint,
                })
                break
        if changed_checks:
            payload["changed_check_conditions"] = {
                "pairs": changed_checks[-latest:],
                "instruction": (
                    "These checks passed under changed test conditions. Repeating green checks "
                    "alone cannot resolve their historical failures. Preserve supplied valid tests. "
                    "If you corrected a self-authored check, verify that it rejects a controlled wrong "
                    "implementation or artifact, restore the candidate, then rerun the same command "
                    "and testcase IDs with the corrected check unchanged. Preserve both receipts."
                ),
            }
        if any(evidence_by_id[ref].diagnostic for ref in red_evidence_ids):
            payload["diagnostic_cleanup"] = "Remove completed, predeclared probe files. Once the actual issue and regressions are verified, finish; the stop hook checks removal. Do not rewrite probes merely to make them green."
        current_green = any(
            receipt.success and receipt.runner_recognized and receipt.scope != "diff"
            and receipt.repo_generation == self._repo_generation
            for receipt in self._verifications
        )
        current_red = any(
            evidence_by_id[ref].test_failure_observed
            and evidence_by_id[ref].repo_generation == self._repo_generation
            for ref in red_evidence_ids
        )
        if self.task.requires_patch and not self.task.requires_repair_witness and model_blockers:
            payload["next_experiment"] = {
                "instruction": (
                    "Implement the smallest runnable component required by the public task. "
                    "Execute its entry point on the provided input to generate the requested artifacts; "
                    "record this as execution, separately from verification. Then run independent, "
                    "read-only pytest checks for every acceptance case on the current state. "
                    "Use recorded observations and failed checks to choose a correction; recheck all "
                    "unresolved failures before finishing."
                ),
            }
        elif self.task.requires_patch and self.task.requires_repair_witness and model_blockers and current_green and not current_red:
            rechecks = [
                {"evidence_id": receipt.evidence_id, "command": observed.input_summary,
                 "acceptance_ids": receipt.acceptance_ids, "scope": receipt.scope,
                 "oracle": receipt.oracle, "check_fingerprint": receipt.check_fingerprint}
                for receipt in self._verifications if receipt.id in unresolved_ids
                and not receipt.success and receipt.runner_recognized and receipt.check_fingerprint
                and (observed := evidence_by_id[receipt.evidence_id]).test_failure_observed
                and not observed.diagnostic
            ][-latest:]
            payload["verification_recovery"] = (
                "Current checks passed but verification evidence is incomplete. Inspect the blockers "
                "and read_observation: rerun the original failed testcase IDs with unchanged test "
                "conditions using the same command, or capture "
                "a corrected test's new failing baseline before repairing code. Check outcome flags "
                "and acceptance labels. Preserve the candidate implementation while diagnosing this gap."
            )
            if rechecks:
                payload["verification_rechecks"] = rechecks
                payload["verification_recovery"] = (
                    "Recorded failing baselines already exist. First rerun their original commands "
                    "on the current candidate with unchanged tests and the recorded acceptance labels. "
                    "A broader passing command does not replace this recheck. Preserve the candidate "
                    "implementation. If test conditions changed, establish a new failing baseline."
                )
        elif self.task.requires_patch and model_blockers and actionable:
            payload["next_experiment"] = {
                "hypothesis_ids": actionable,
                "instruction": (
                    "Use one of these predictions to choose the next experiment on current code. "
                    "If a failing check and code-grounded cause are available, make a small "
                    "candidate edit and test it. Otherwise run one check that can refute a "
                    "prediction. Reuse recorded observations; do not keep searching caches, "
                    "other installations, or unrelated history for a ready-made implementation."
                ),
            }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"<repair_state>{body}</repair_state>"

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task": {
                **asdict(self.task),
                "canonical_issue_sha256": self.task.issue_sha256,
            },
            "repo": {
                "initial_state": self._initial_repo_state,
                "current_state": self._current_repo_state,
                "generation": self._repo_generation,
            },
            "acceptance_cases": [asdict(case) for case in self._cases.values()],
            "hypotheses": [
                {**asdict(hypothesis), "evidence_ids": list(hypothesis.evidence_ids)}
                for hypothesis in self._hypotheses.values()
            ],
            "hypothesis_history": [asdict(hypothesis) for hypothesis in self._hypothesis_history],
            "evidence": [asdict(receipt) for receipt in self._evidence],
            "verifications": [
                {**asdict(receipt), "acceptance_ids": list(receipt.acceptance_ids)}
                for receipt in self._verifications
            ],
            "retired_diagnostics": dict(self._retired_diagnostics),
        }

    def _evidence_by_id(self) -> dict[str, EvidenceReceipt]:
        return {receipt.id: receipt for receipt in self._evidence}


__all__ = [
    "AcceptanceCase",
    "CompletionDecision",
    "EvidenceReceipt",
    "Hypothesis",
    "RepairState",
    "SCHEMA_VERSION",
    "TaskEnvelope",
    "VerificationReceipt",
]
