from __future__ import annotations

import pytest

from src.backend_v2.domain.drain import DrainCoordinator, DrainIntent
from src.backend_v2.domain.fencing import ExecutorFence, FencedExecution
from src.backend_v2.domain.object_journal import (
    JournalState,
    RecoveryAction,
    next_journal_state,
    recovery_action,
)
from src.backend_v2.domain.ordering import normalize_ordinals, reorder_subset
from src.backend_v2.domain.state_machines import (
    InvalidTransition,
    JobEvent,
    JobStatus,
    OperationEvent,
    OperationStatus,
    transition_job,
    transition_operation,
)
from src.backend_v2.domain.write_admission import (
    AdmissionDecision,
    IntentFence,
    WriteRequestKind,
    decide_write_admission,
)


def test_job_state_machine_distinguishes_resume_from_continue() -> None:
    assert transition_job(JobStatus.PAUSED, JobEvent.RESUME) is JobStatus.QUEUED
    assert transition_job(JobStatus.INTERRUPTED, JobEvent.CONTINUE) is JobStatus.QUEUED
    with pytest.raises(InvalidTransition):
        transition_job(JobStatus.INTERRUPTED, JobEvent.RESUME)
    with pytest.raises(InvalidTransition):
        transition_job(JobStatus.PAUSED, JobEvent.CONTINUE)


def test_cancel_intent_survives_worker_loss() -> None:
    assert (
        transition_job(JobStatus.CANCELLING, JobEvent.WORKER_LOST)
        is JobStatus.CANCELLED
    )


def test_operation_terminal_states_are_closed() -> None:
    assert (
        transition_operation(OperationStatus.PENDING, OperationEvent.CLAIM)
        is OperationStatus.RUNNING
    )
    assert (
        transition_operation(OperationStatus.RUNNING, OperationEvent.COMPLETE)
        is OperationStatus.COMPLETED
    )
    with pytest.raises(InvalidTransition):
        transition_operation(OperationStatus.COMPLETED, OperationEvent.CLAIM)


def test_zero_row_heartbeat_immediately_self_fences() -> None:
    fence = ExecutorFence()
    fence.observe_attempt_renewal("attempt-1", 0)
    assert not fence.may_publish("attempt-1")
    with pytest.raises(FencedExecution):
        fence.require_publish("attempt-1")

    fence.observe_epoch_renewal(0)
    assert not fence.may_admit()
    assert not fence.may_publish("another-attempt")
    with pytest.raises(FencedExecution):
        fence.require_admission()


def test_drain_requires_every_admitted_step_and_every_worker_slot() -> None:
    drain = DrainCoordinator(frozenset({("ocr", 0), ("translate", 0)}))
    drain.admit_step("step-1")
    drain.request(DrainIntent.PAUSE)
    assert not drain.admission_open
    with pytest.raises(RuntimeError):
        drain.admit_step("step-2")

    drain.mark_step_terminal("step-1")
    drain.acknowledge("ocr", 0)
    assert not drain.drained
    drain.acknowledge("translate", 0)
    assert drain.drained

    drain.request(DrainIntent.CANCEL)
    assert drain.intent is DrainIntent.CANCEL


def test_write_intent_blocks_new_writes_but_drains_existing_operations() -> None:
    assert (
        decide_write_admission(
            has_intent=True,
            has_lock=False,
            request_kind=WriteRequestKind.NEW_DOCUMENT_WRITE,
        )
        is AdmissionDecision.CHAPTER_WRITE_PENDING
    )
    assert (
        decide_write_admission(
            has_intent=True,
            has_lock=False,
            request_kind=WriteRequestKind.EXISTING_OPERATION_FOLLOWUP_RENDER,
        )
        is AdmissionDecision.ALLOW
    )
    assert (
        decide_write_admission(
            has_intent=True,
            has_lock=True,
            request_kind=WriteRequestKind.EXISTING_OPERATION_CLAIM,
        )
        is AdmissionDecision.CHAPTER_LOCKED
    )


def test_intent_fence_includes_generation_and_lease_identity() -> None:
    first = IntentFence("job", "epoch", "set", 3, "lease")
    assert first.matches(IntentFence("job", "epoch", "set", 3, "lease"))
    assert not first.matches(IntentFence("job", "epoch", "set", 4, "lease"))
    with pytest.raises(ValueError):
        IntentFence("job", "epoch", "set", 0, "lease")


def test_object_journal_recovery_covers_each_crash_window() -> None:
    assert next_journal_state(JournalState.STAGED) is JournalState.FILE_PUBLISHED
    assert (
        next_journal_state(JournalState.FILE_PUBLISHED)
        is JournalState.DATABASE_COMMITTED
    )
    with pytest.raises(ValueError):
        next_journal_state(JournalState.DATABASE_COMMITTED)

    assert (
        recovery_action(
            database_has_asset=False,
            final_file_exists=False,
            staging_file_exists=True,
        )
        is RecoveryAction.PUBLISH_STAGING_FILE
    )
    assert (
        recovery_action(
            database_has_asset=True,
            final_file_exists=False,
            staging_file_exists=False,
        )
        is RecoveryAction.MARK_ASSET_MISSING
    )


def test_ordering_helpers_are_total_and_do_not_move_fixed_prefix() -> None:
    assert normalize_ordinals(["a", "b"]) == {"a": 1, "b": 2}
    assert reorder_subset(
        complete_order=["running", "a", "b"],
        fixed_prefix=["running"],
        requested_sortable_order=["b", "a"],
    ) == ["running", "b", "a"]
    with pytest.raises(ValueError):
        normalize_ordinals(["a", "a"])
