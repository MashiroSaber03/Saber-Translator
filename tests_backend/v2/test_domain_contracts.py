from __future__ import annotations

import pytest

from src.backend_v2.domain.state_machines import (
    InvalidTransition,
    JobEvent,
    JobStatus,
    transition_job,
)


def test_job_state_machine_distinguishes_resume_from_continue() -> None:
    assert (
        transition_job(JobStatus.RUNNING, JobEvent.REQUEST_PAUSE)
        is JobStatus.PAUSED
    )
    assert transition_job(JobStatus.PAUSED, JobEvent.RESUME) is JobStatus.QUEUED
    assert transition_job(JobStatus.INTERRUPTED, JobEvent.CONTINUE) is JobStatus.QUEUED
    with pytest.raises(InvalidTransition):
        transition_job(JobStatus.INTERRUPTED, JobEvent.RESUME)
    with pytest.raises(InvalidTransition):
        transition_job(JobStatus.PAUSED, JobEvent.CONTINUE)


@pytest.mark.parametrize(
    "status",
    [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.PAUSED,
        JobStatus.INTERRUPTED,
    ],
)
def test_cancel_is_an_immediate_terminal_transition(status: JobStatus) -> None:
    assert transition_job(status, JobEvent.REQUEST_CANCEL) is JobStatus.CANCELLED
