"""User-visible job control transitions."""

from __future__ import annotations

from enum import StrEnum


class InvalidTransition(ValueError):
    def __init__(self, current: StrEnum, event: StrEnum) -> None:
        super().__init__(f"{current.value} does not accept {event.value}")


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class JobEvent(StrEnum):
    REQUEST_PAUSE = "request_pause"
    REQUEST_CANCEL = "request_cancel"
    RESUME = "resume"
    CONTINUE = "continue"


JOB_TRANSITIONS: dict[tuple[JobStatus, JobEvent], JobStatus] = {
    (JobStatus.QUEUED, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLED,
    (JobStatus.RUNNING, JobEvent.REQUEST_PAUSE): JobStatus.PAUSED,
    (JobStatus.RUNNING, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLED,
    (JobStatus.PAUSED, JobEvent.RESUME): JobStatus.QUEUED,
    (JobStatus.PAUSED, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLED,
    (JobStatus.INTERRUPTED, JobEvent.CONTINUE): JobStatus.QUEUED,
    (JobStatus.INTERRUPTED, JobEvent.REQUEST_CANCEL): JobStatus.CANCELLED,
}


def transition_job(current: JobStatus, event: JobEvent) -> JobStatus:
    try:
        return JOB_TRANSITIONS[(current, event)]
    except KeyError as exc:
        raise InvalidTransition(current, event) from exc
