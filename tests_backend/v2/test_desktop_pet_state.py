from __future__ import annotations

from src.backend_v2.desktop.pet_state import PetState, PetStateMachine


def _job(
    job_id: str,
    kind: str,
    status: str,
    *,
    current_step: str = "",
    started_at: str = "2026-08-09T10:00:00Z",
) -> dict[str, object]:
    progress: dict[str, object] = {}
    if current_step:
        progress["currentStep"] = {"kind": current_step}
    return {
        "jobId": job_id,
        "kind": kind,
        "status": status,
        "startedAt": started_at,
        "progress": progress,
    }


def test_translation_pose_ignores_all_atomic_pipeline_steps() -> None:
    machine = PetStateMachine()
    steps = (
        "detect",
        "ocr",
        "color",
        "term_extract",
        "translate",
        "repair",
        "render",
        "save",
    )

    states = {
        machine.update("running", [_job("a", "translation", "running", current_step=step)], [])
        for step in steps
    }

    assert states == {PetState.TRANSLATING}
    assert machine.foreground_job_id == "a"


def test_analysis_pose_ignores_internal_progress_steps() -> None:
    machine = PetStateMachine()

    assert machine.update(
        "running",
        [_job("analysis", "insight_analysis", "running", current_step="page_vlm")],
        [],
    ) == PetState.ANALYZING
    assert machine.update(
        "running",
        [_job("analysis", "insight_analysis", "running", current_step="vector_rebuild")],
        [],
    ) == PetState.ANALYZING


def test_foreground_job_is_latched_across_snapshot_reordering() -> None:
    machine = PetStateMachine()
    translation = _job("translation", "translation", "running", started_at="2026-08-09T10:00:00Z")
    analysis = _job("analysis", "insight_analysis", "running", started_at="2026-08-09T10:01:00Z")

    assert machine.update("running", [translation, analysis], []) == PetState.TRANSLATING
    assert machine.update("running", [analysis, translation], []) == PetState.TRANSLATING
    assert machine.foreground_job_id == "translation"

    assert machine.update("running", [analysis], [dict(translation, status="completed")]) == PetState.ANALYZING
    assert machine.foreground_job_id == "analysis"


def test_lifecycle_states_override_task_category_without_treating_cancel_as_failure() -> None:
    now = [0.0]
    machine = PetStateMachine(clock=lambda: now[0])
    running = _job("a", "translation", "running")
    machine.update("running", [running], [])

    assert machine.update("running", [dict(running, status="paused")], []) == PetState.PAUSED
    assert machine.update("running", [dict(running, status="queued")], []) == PetState.WAITING
    assert machine.update("running", [], [dict(running, status="cancelled")]) == PetState.IDLE


def test_terminal_reaction_is_short_and_service_failure_has_priority() -> None:
    now = [10.0]
    machine = PetStateMachine(clock=lambda: now[0])
    running = _job("a", "translation", "running")
    machine.update("running", [running], [])

    completed = dict(running, status="completed")
    assert machine.update("running", [], [completed]) == PetState.SUCCESS
    assert machine.update("degraded", [], [completed]) == PetState.FAILED
    now[0] += 3.1
    assert machine.update("running", [], [completed]) == PetState.IDLE


def test_interrupted_job_is_not_hidden_by_newer_history() -> None:
    machine = PetStateMachine()
    history = [
        _job("done", "translation", "completed"),
        _job("interrupted", "translation", "interrupted"),
    ]

    assert machine.update("running", [], history) == PetState.PAUSED


def test_unknown_jobs_and_shutdown_use_the_neutral_waiting_pose() -> None:
    machine = PetStateMachine()

    assert machine.update("running", [_job("future", "future_job", "running")], []) == PetState.WAITING
    assert machine.update("stopping", [], []) == PetState.WAITING
