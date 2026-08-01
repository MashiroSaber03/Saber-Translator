"""Closed job state machine."""

from __future__ import annotations

from enum import StrEnum


class InvalidTransition(ValueError):
    def __init__(self, current: StrEnum, event: StrEnum) -> None:
        super().__init__(f"{current.value} does not accept {event.value}")


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class JobEvent(StrEnum):
    CLAIM = "claim"
    REQUEST_PAUSE = "request_pause"
    REQUEST_CANCEL = "request_cancel"
    DRAIN_PAUSED = "drain_paused"
    DRAIN_CANCELLED = "drain_cancelled"
    COMPLETE = "complete"
    COMPLETE_WITH_ERRORS = "complete_with_errors"
    FAIL = "fail"
    WORKER_LOST = "worker_lost"
    RESUME = "resume"
    CONTINUE = "continue"


JOB_TRANSITIONS: dict[tuple[JobStatus, JobEvent], JobStatus] = {
    (JobStatus.QUEUED, JobEvent.CLAIM): JobStatus.RUNNING,
    (JobStatus.QUEUED, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLED,
    (JobStatus.RUNNING, JobEvent.REQUEST_PAUSE): JobStatus.PAUSING,
    (JobStatus.RUNNING, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLING,
    (JobStatus.RUNNING, JobEvent.COMPLETE): JobStatus.COMPLETED,
    (JobStatus.RUNNING, JobEvent.COMPLETE_WITH_ERRORS): JobStatus.COMPLETED_WITH_ERRORS,
    (JobStatus.RUNNING, JobEvent.FAIL): JobStatus.FAILED,
    (JobStatus.RUNNING, JobEvent.WORKER_LOST): JobStatus.INTERRUPTED,
    (JobStatus.PAUSING, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLING,
    (JobStatus.PAUSING, JobEvent.DRAIN_PAUSED): JobStatus.PAUSED,
    (JobStatus.PAUSING, JobEvent.WORKER_LOST): JobStatus.INTERRUPTED,
    (JobStatus.PAUSED, JobEvent.RESUME): JobStatus.QUEUED,
    (JobStatus.PAUSED, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLED,
    (JobStatus.CANCELLING, JobEvent.DRAIN_CANCELLED): JobStatus.CANCELLED,
    # A confirmed user cancel intent must not resurrect as interrupted.
    (JobStatus.CANCELLING, JobEvent.WORKER_LOST): JobStatus.CANCELLED,
    (JobStatus.INTERRUPTED, JobEvent.CONTINUE): JobStatus.QUEUED,
    (JobStatus.INTERRUPTED, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLED,
}


def transition_job(current: JobStatus, event: JobEvent) -> JobStatus:
    try:
        return JOB_TRANSITIONS[(current, event)]
    except KeyError as exc:
        raise InvalidTransition(current, event) from exc
