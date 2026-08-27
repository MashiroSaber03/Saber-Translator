"""Deterministic pet state selection based on top-level job types only."""

from __future__ import annotations

from enum import Enum
import time
from typing import Callable, Iterable, Mapping


class PetState(str, Enum):
    IDLE = "idle"
    GREETING = "greeting"
    STARTING = "starting"
    WAITING = "waiting"
    TRANSLATING = "translating"
    ANALYZING = "analyzing"
    TRANSFER = "transfer"
    PAUSED = "paused"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    DRAG_RIGHT = "drag_right"
    DRAG_LEFT = "drag_left"


TRANSLATION_JOB_KINDS = frozenset(
    {"translation", "remove_text", "detect", "style_apply"}
)
ANALYSIS_JOB_KINDS = frozenset(
    {"insight_analysis", "vector_rebuild", "continuation", "derived_rebuild", "plugin_agent"}
)
TRANSFER_JOB_KINDS = frozenset(
    {"text_import", "container_import", "web_extract", "web_import_commit", "export", "insight_export"}
)
RUNNING_STATUSES = frozenset({"running"})
PAUSED_STATUSES = frozenset({"paused"})
TERMINAL_REACTIONS = {
    "completed": PetState.SUCCESS,
    "completed_with_errors": PetState.WARNING,
    "failed": PetState.FAILED,
}


def pet_state_for_job_kind(kind: object) -> PetState:
    normalized = str(kind or "")
    if normalized in TRANSLATION_JOB_KINDS:
        return PetState.TRANSLATING
    if normalized in ANALYSIS_JOB_KINDS:
        return PetState.ANALYZING
    if normalized in TRANSFER_JOB_KINDS:
        return PetState.TRANSFER
    return PetState.WAITING


class PetStateMachine:
    """Latch one foreground job so atomic progress events never switch poses."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._foreground_job_id: str | None = None
        self._known_statuses: dict[str, str] = {}
        self._reaction: PetState | None = None
        self._reaction_until = 0.0
        self._initialized = False

    @property
    def foreground_job_id(self) -> str | None:
        return self._foreground_job_id

    def update(
        self,
        launcher_state: str,
        queue_jobs: Iterable[Mapping[str, object]],
        history_jobs: Iterable[Mapping[str, object]],
    ) -> PetState:
        queue = list(queue_jobs)
        history = list(history_jobs)
        all_jobs = [*queue, *history]
        next_statuses = {
            str(job.get("jobId")): str(job.get("status"))
            for job in all_jobs
            if job.get("jobId")
        }
        if self._initialized:
            self._capture_terminal_reaction(all_jobs)
        else:
            self._initialized = True
        self._known_statuses = next_statuses

        if launcher_state == "degraded":
            return PetState.FAILED
        if launcher_state == "starting":
            return PetState.STARTING
        if launcher_state == "stopping":
            return PetState.WAITING
        if launcher_state != "running":
            self._foreground_job_id = None
            return PetState.IDLE

        running = [job for job in queue if str(job.get("status")) in RUNNING_STATUSES]
        foreground = self._select_foreground(running)
        if foreground is not None:
            return pet_state_for_job_kind(foreground.get("kind"))

        paused = [job for job in queue if str(job.get("status")) in PAUSED_STATUSES]
        if paused:
            return PetState.PAUSED
        if any(str(job.get("status")) == "queued" for job in queue):
            return PetState.WAITING
        if any(str(job.get("status")) == "interrupted" for job in history):
            return PetState.PAUSED
        if self._reaction is not None and self._clock() < self._reaction_until:
            return self._reaction
        self._reaction = None
        return PetState.IDLE

    def _select_foreground(
        self,
        running: list[Mapping[str, object]],
    ) -> Mapping[str, object] | None:
        if not running:
            self._foreground_job_id = None
            return None
        for job in running:
            if str(job.get("jobId")) == self._foreground_job_id:
                return job
        running.sort(
            key=lambda job: str(job.get("startedAt") or job.get("createdAt") or "")
        )
        selected = running[0]
        self._foreground_job_id = str(selected.get("jobId"))
        return selected

    def _capture_terminal_reaction(
        self,
        jobs: list[Mapping[str, object]],
    ) -> None:
        for job in jobs:
            job_id = str(job.get("jobId") or "")
            status = str(job.get("status") or "")
            if not job_id or status not in TERMINAL_REACTIONS:
                continue
            previous = self._known_statuses.get(job_id)
            if previous is None or previous == status:
                continue
            self._reaction = TERMINAL_REACTIONS[status]
            self._reaction_until = self._clock() + 3.0
            return
