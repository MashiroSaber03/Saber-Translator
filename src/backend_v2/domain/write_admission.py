"""Chapter write-intent admission and upgrade rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WriteRequestKind(StrEnum):
    NEW_DOCUMENT_WRITE = "new_document_write"
    NEW_PAGE_OPERATION = "new_page_operation"
    NEW_IMPORT_LEASE = "new_import_lease"
    EXISTING_OPERATION_CLAIM = "existing_operation_claim"
    EXISTING_RENDER_CLAIM = "existing_render_claim"
    EXISTING_OPERATION_FOLLOWUP_RENDER = "existing_operation_followup_render"


class AdmissionDecision(StrEnum):
    ALLOW = "allow"
    CHAPTER_WRITE_PENDING = "chapter_write_pending"
    CHAPTER_LOCKED = "chapter_locked"


def decide_write_admission(
    *,
    has_intent: bool,
    has_lock: bool,
    request_kind: WriteRequestKind,
) -> AdmissionDecision:
    if has_lock:
        return AdmissionDecision.CHAPTER_LOCKED
    if not has_intent:
        return AdmissionDecision.ALLOW
    if request_kind in {
        WriteRequestKind.EXISTING_OPERATION_CLAIM,
        WriteRequestKind.EXISTING_RENDER_CLAIM,
        WriteRequestKind.EXISTING_OPERATION_FOLLOWUP_RENDER,
    }:
        return AdmissionDecision.ALLOW
    return AdmissionDecision.CHAPTER_WRITE_PENDING


@dataclass(frozen=True, slots=True)
class IntentFence:
    job_id: str
    worker_epoch_id: str
    intent_set_id: str
    intent_generation: int
    lease_token: str

    def __post_init__(self) -> None:
        if not all(
            (self.job_id, self.worker_epoch_id, self.intent_set_id, self.lease_token)
        ):
            raise ValueError("all intent fencing identities are required")
        if self.intent_generation < 1:
            raise ValueError("intent_generation must be positive")

    def matches(self, current: "IntentFence") -> bool:
        return self == current
