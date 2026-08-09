from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

import pytest
from flask import Flask
from sqlalchemy import insert, select, update
from sqlalchemy.exc import OperationalError as SqlAlchemyOperationalError

from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.events import JobEventBroadcaster
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    InvalidJobTransition,
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
    utcnow,
)
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.jobs.routes import create_jobs_blueprint
from src.backend_v2.jobs.worker_loop import (
    PARALLEL_PIPELINE_LEAD_WINDOW,
    AttemptHeartbeat,
    JobWorkerLoop,
)
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import (
    assets,
    chapter_write_intents,
    chapter_write_locks,
    credentials,
    credential_versions,
    job_artifacts,
    job_batches,
    job_config_snapshots,
    job_events,
    job_items,
    job_steps,
    jobs,
    metadata,
    operations,
    process_epochs,
    worker_leases,
)
from src.backend_v2.storage.seeding import seed_system_records


@pytest.fixture()
def job_platform(tmp_path: Path):
    engine = create_sqlite_engine(tmp_path / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(book_id=str(book["id"]), title="Chapter")
    worker_epoch_id = str(uuid.uuid4())
    worker_token = "worker-test-token"
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            epoch_id=worker_epoch_id,
            token=worker_token,
            role="worker",
            pid=123,
        )
    )
    repository = JobQueueRepository(engine, attempt_lease_seconds=10)
    try:
        yield engine, repository, book, chapter, worker_epoch_id
    finally:
        engine.dispose()


def _create_job(
    repository: JobQueueRepository,
    *,
    kind: str = "export",
    chapter_id: str | None = None,
    steps: tuple[str, ...] = ("package",),
) -> str:
    result = repository.create_batch(
        kind=kind,
        display_name=f"{kind} batch",
        specs=[
            JobSpec(
                kind=kind,
                config={"mode": "test"},
                chapter_id=chapter_id,
                items=(JobItemSpec(page_id=None, step_kinds=steps),),
                target_display={"chapter": "Chapter"} if chapter_id else {},
            )
        ],
    )
    return str(result["jobIds"][0])


def test_history_list_limit_counts_batches_not_member_jobs(job_platform) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    batch_ids: list[str] = []
    for index in range(3):
        result = repository.create_batch(
            kind="export",
            display_name=f"history batch {index}",
            specs=[
                JobSpec(
                    kind="export",
                    config={"index": index, "member": member},
                    items=(JobItemSpec(page_id=None, step_kinds=("package",)),),
                )
                for member in range(2)
            ],
        )
        batch_ids.append(str(result["batchId"]))
        now = utcnow()
        with engine.begin() as connection:
            connection.execute(
                update(jobs)
                .where(jobs.c.batch_id == result["batchId"])
                .values(
                    status="completed",
                    queue_rank=None,
                    finished_at=now,
                    updated_at=now,
                )
            )

    history = repository.list_jobs(scope="history", limit=2)
    returned_batch_ids = {str(row["batchId"]) for row in history["items"]}

    assert returned_batch_ids == set(batch_ids[-2:])
    assert len(history["items"]) == 4


def test_chapter_write_intent_drains_then_atomically_upgrades_and_releases(
    job_platform,
) -> None:
    engine, repository, _book, chapter, worker_epoch_id = job_platform
    job_id = _create_job(
        repository,
        kind="translation",
        chapter_id=str(chapter["id"]),
        steps=("detect", "ocr"),
    )

    # First pass establishes the admission barrier while the job remains queued.
    assert repository.claim_next(worker_epoch_id=worker_epoch_id) is None
    with engine.connect() as connection:
        intent = connection.execute(
            select(chapter_write_intents).where(
                chapter_write_intents.c.job_id == job_id
            )
        ).mappings().one()
        assert intent["chapter_id"] == chapter["id"]
        assert connection.execute(
            select(jobs.c.status).where(jobs.c.id == job_id)
        ).scalar_one() == "queued"

    fence = repository.claim_next(worker_epoch_id=worker_epoch_id)
    assert fence is not None
    with engine.connect() as connection:
        assert connection.execute(
            select(chapter_write_intents.c.job_id).where(
                chapter_write_intents.c.job_id == job_id
            )
        ).scalar_one_or_none() is None
        lock = connection.execute(
            select(chapter_write_locks).where(
                chapter_write_locks.c.job_id == job_id
            )
        ).mappings().one()
        assert lock["owner_attempt_id"] == fence.attempt_id

    while (step := repository.next_step(fence)) is not None:
        repository.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint={"kind": step["stepKind"]},
        )
    assert repository.finish_if_complete(fence) == "completed"
    with engine.connect() as connection:
        assert connection.execute(
            select(chapter_write_locks.c.job_id).where(
                chapter_write_locks.c.job_id == job_id
            )
        ).scalar_one_or_none() is None


def test_page_completed_event_identifies_the_published_page(job_platform) -> None:
    engine, repository, _book, chapter, worker_epoch_id = job_platform
    from src.backend_v2.storage.schema import pages

    page_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            insert(pages).values(
                id=page_id,
                chapter_id=chapter["id"],
                ordinal=1,
                logical_source_path="1.png",
            )
        )
    batch = repository.create_batch(
        kind="translation",
        display_name="translation batch",
        specs=[
            JobSpec(
                kind="translation",
                config={"mode": "test"},
                chapter_id=str(chapter["id"]),
                items=(
                    JobItemSpec(page_id=page_id, step_kinds=("save",)),
                ),
            )
        ],
    )
    job_id = str(batch["jobIds"][0])

    assert repository.claim_next(worker_epoch_id=worker_epoch_id) is None
    fence = repository.claim_next(worker_epoch_id=worker_epoch_id)
    assert fence is not None
    step = repository.next_step(fence)
    assert step is not None
    repository.complete_step(
        fence,
        step_id=str(step["stepId"]),
        checkpoint={"published": True},
    )

    events = repository.events_after(job_id=job_id)
    completed = [event for event in events if event["type"] == "page_completed"]
    assert len(completed) == 1
    assert completed[0]["payload"]["pageId"] == page_id


def test_write_intent_waits_for_preexisting_operation(job_platform) -> None:
    engine, repository, _book, chapter, worker_epoch_id = job_platform
    page_id = str(uuid.uuid4())
    with engine.begin() as connection:
        from src.backend_v2.storage.schema import pages

        connection.execute(
            insert(pages).values(
                id=page_id,
                chapter_id=chapter["id"],
                ordinal=1,
                logical_source_path="1.png",
            )
        )
        operation_id = str(uuid.uuid4())
        connection.execute(
            insert(operations).values(
                id=operation_id,
                kind="page_detect",
                executor_role="worker",
                status="pending",
                page_id=page_id,
                base_revision=1,
                request_json="{}",
            )
        )
    _create_job(
        repository,
        kind="translation",
        chapter_id=str(chapter["id"]),
    )
    assert repository.claim_next(worker_epoch_id=worker_epoch_id) is None
    assert repository.claim_next(worker_epoch_id=worker_epoch_id) is None
    with engine.begin() as connection:
        connection.execute(
            update(operations)
            .where(operations.c.id == operation_id)
            .values(status="completed")
        )
    assert repository.claim_next(worker_epoch_id=worker_epoch_id) is not None


def test_pause_resume_cancel_and_attempt_fencing(job_platform) -> None:
    engine, repository, _book, chapter, worker_epoch_id = job_platform
    job_id = _create_job(
        repository,
        kind="translation",
        chapter_id=str(chapter["id"]),
    )
    assert repository.claim_next(worker_epoch_id=worker_epoch_id) is None
    fence = repository.claim_next(worker_epoch_id=worker_epoch_id)
    assert fence is not None

    assert repository.request_pause(job_id)["status"] == "pausing"
    assert repository.request_pause(job_id)["status"] == "pausing"
    repository.acknowledge_drain(
        fence,
        pool_id="main",
        worker_slot=0,
        last_step_id=None,
    )
    assert repository.finalize_drain(
        fence, expected_slots={("main", 0)}
    ) == "paused"
    with engine.connect() as connection:
        assert connection.execute(
            select(chapter_write_locks.c.job_id).where(
                chapter_write_locks.c.job_id == job_id
            )
        ).scalar_one() == job_id

    assert repository.resume(job_id)["status"] == "queued"
    queue_snapshot = repository.list_jobs(scope="queue")
    queued_job = next(
        row for row in queue_snapshot["items"] if row["jobId"] == job_id
    )
    assert queued_job["blockedReason"] == "retained_chapter_lock"
    with pytest.raises(JobConflict):
        repository.reorder(
            ordered_job_ids=[job_id],
            base_revision=int(queue_snapshot["queueRevision"]),
        )
    with pytest.raises(InvalidJobTransition):
        repository.continue_interrupted(job_id)
    resumed = repository.claim_next(worker_epoch_id=worker_epoch_id)
    assert resumed is not None
    assert resumed.attempt_id != fence.attempt_id
    assert repository.request_cancel(job_id)["status"] == "cancelling"
    repository.acknowledge_drain(
        resumed,
        pool_id="main",
        worker_slot=0,
        last_step_id=None,
    )
    assert repository.finalize_drain(
        resumed, expected_slots={("main", 0)}
    ) == "cancelled"
    with pytest.raises(AttemptFenced):
        repository.control_status(resumed)
    with engine.connect() as connection:
        assert connection.execute(
            select(chapter_write_locks.c.job_id).where(
                chapter_write_locks.c.job_id == job_id
            )
        ).scalar_one_or_none() is None


def test_bad_attempt_token_cannot_publish_checkpoint(job_platform) -> None:
    _engine, repository, _book, _chapter, worker_epoch_id = job_platform
    _create_job(repository)
    fence = repository.claim_next(worker_epoch_id=worker_epoch_id)
    assert fence is not None
    step = repository.next_step(fence)
    assert step is not None
    forged = AttemptFence(
        job_id=fence.job_id,
        attempt_id=fence.attempt_id,
        lease_token="forged",
        worker_epoch_id=fence.worker_epoch_id,
        lease_expires_at=fence.lease_expires_at,
    )
    with pytest.raises(AttemptFenced):
        repository.complete_step(
            forged,
            step_id=str(step["stepId"]),
            checkpoint={"mustNotPublish": True},
        )


def test_zero_row_attempt_renewal_fences_all_late_writes(job_platform) -> None:
    engine, repository, _book, _chapter, worker_epoch_id = job_platform
    job_id = _create_job(repository)
    fence = repository.claim_next(worker_epoch_id=worker_epoch_id)
    assert fence is not None
    step = repository.next_step(fence)
    assert step is not None
    assert repository.renew_attempt(fence) is not None
    with engine.begin() as connection:
        from src.backend_v2.storage.schema import process_epochs

        connection.execute(
            update(process_epochs)
            .where(process_epochs.c.id == worker_epoch_id)
            .values(status="lost")
        )
    assert repository.renew_attempt(fence) is None
    with pytest.raises(AttemptFenced):
        repository.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint={"late": True},
        )
    assert repository.get_job(job_id)["status"] == "running"


def test_worker_loop_finishes_durable_job_without_browser(job_platform) -> None:
    _engine, repository, _book, _chapter, worker_epoch_id = job_platform
    job_id = _create_job(repository, steps=("one", "two"))
    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={
            "one": lambda _fence, step: {"done": step["stepKind"]},
            "two": lambda _fence, step: {"done": step["stepKind"]},
        },
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if repository.get_job(job_id)["status"] == "completed":
            break
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2)
    assert repository.get_job(job_id)["status"] == "completed"
    with repository.engine.connect() as connection:
        assert connection.execute(
            select(job_steps.c.status)
            .join(job_items, job_items.c.id == job_steps.c.job_item_id)
            .where(job_items.c.job_id == job_id)
            .order_by(job_steps.c.ordinal)
        ).scalars().all() == ["completed", "completed"]


def test_worker_loop_fails_instead_of_spinning_on_unregistered_step(
    job_platform,
) -> None:
    _engine, repository, _book, _chapter, worker_epoch_id = job_platform
    job_id = _create_job(repository, steps=("one", "missing-handler"))
    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"one": lambda _fence, step: {"done": step["stepKind"]}},
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if repository.get_job(job_id)["status"] == "failed":
            break
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2)

    job = repository.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == {
        "code": "UNSUPPORTED_STEP_KIND",
        "message": "Worker 没有以下步骤的处理器：missing-handler",
    }


def test_worker_rss_remains_bounded_for_large_durable_job_graphs(
    tmp_path: Path,
) -> None:
    measurements: list[dict[str, object]] = []
    for item_count in (100, 500, 1000):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests_backend.v2.worker_memory_probe",
                str(item_count),
                str(tmp_path / f"worker-memory-{item_count}.sqlite3"),
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=True,
            text=True,
            timeout=270,
        )
        measurements.append(json.loads(result.stdout.strip()))

    mib = 1024 * 1024
    for measurement in measurements:
        assert measurement["status"] == "completed"
        assert measurement["workerStopped"] is True
        assert int(measurement["peakDelta"]) < 96 * mib

    five_hundred = measurements[1]
    one_thousand = measurements[2]
    assert (
        int(one_thousand["peakDelta"]) - int(five_hundred["peakDelta"])
        < 32 * mib
    )


def test_job_failures_never_expose_frozen_credentials(job_platform) -> None:
    engine, repository, _book, _chapter, worker_epoch_id = job_platform
    canary = "CANARY-JOB-API-KEY-18462"
    credential_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            insert(credentials).values(
                id=credential_id,
                domain="translation",
                provider="canary",
            )
        )
        connection.execute(
            insert(credential_versions).values(
                id=version_id,
                credential_id=credential_id,
                version=1,
                secret_json=json.dumps({"apiKey": canary}),
                key_fingerprint="2" * 64,
            )
        )
    batch = repository.create_batch(
        kind="export",
        display_name="redaction",
        specs=[
            JobSpec(
                kind="export",
                config={"credentialVersionId": version_id},
                items=(
                    JobItemSpec(
                        page_id=None,
                        step_kinds=("package",),
                    ),
                ),
            )
        ],
    )
    job_id = str(batch["jobIds"][0])
    fence = repository.claim_next(worker_epoch_id=worker_epoch_id)
    assert fence is not None
    step = repository.next_step(fence)
    assert step is not None
    repository.fail_step(
        fence,
        step_id=str(step["stepId"]),
        code="PROVIDER_FAILED",
        message=f"provider rejected {canary}",
    )
    assert repository.finish_if_complete(fence) == "completed_with_errors"

    exposed = json.dumps(
        {
            "job": repository.get_job(job_id),
            "events": repository.events_after(job_id=job_id),
        },
        ensure_ascii=False,
    )
    assert canary not in exposed
    assert "[REDACTED]" in exposed


def test_shared_event_broadcaster_replays_and_fans_out(job_platform) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    existing_job = _create_job(repository)
    existing = repository.events_after(after=0)
    assert existing[-1]["jobId"] == existing_job

    broadcaster = JobEventBroadcaster(
        repository,
        epoch_repository=ProcessEpochRepository(engine),
        poll_seconds=0.01,
        subscriber_capacity=8,
    )
    broadcaster.start()
    subscription = broadcaster.subscribe()
    try:
        new_job = _create_job(repository)
        event = subscription.queue.get(timeout=2)
        assert event is not None
        assert event["jobId"] == new_job
        assert event["type"] == "job_created"
        assert "job" not in event
        assert "queueRevision" not in event

        snapshot = repository.job_snapshot(job_ids=[new_job])
        assert snapshot["items"][0]["jobId"] == new_job
        assert snapshot["items"][0]["status"] == "queued"
        assert int(snapshot["queueRevision"]) >= 1
    finally:
        broadcaster.unsubscribe(subscription)
        broadcaster.close()


def test_job_snapshot_route_reads_only_requested_current_projections(
    job_platform,
) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    first_job = _create_job(repository)
    second_job = _create_job(repository)
    broadcaster = JobEventBroadcaster(
        repository,
        epoch_repository=ProcessEpochRepository(engine),
    )
    app = Flask(__name__)
    app.register_blueprint(
        create_jobs_blueprint(engine=engine, broadcaster=broadcaster)
    )
    try:
        response = app.test_client().get(
            "/api/v2/jobs/snapshot",
            query_string=[
                ("job_id", second_job),
                ("job_id", first_job),
            ],
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert [item["jobId"] for item in payload["items"]] == [
            second_job,
            first_job,
        ]
        assert all(item["status"] == "queued" for item in payload["items"])
        assert int(payload["queueRevision"]) >= 1

        missing = app.test_client().get("/api/v2/jobs/snapshot")
        assert missing.status_code == 422
    finally:
        broadcaster.close()


def test_shared_event_poller_recovers_an_expired_worker_without_a_browser(
    job_platform,
) -> None:
    engine, repository, _book, _chapter, worker_epoch_id = job_platform
    expired_at = utcnow() - timedelta(seconds=1)
    with engine.begin() as connection:
        connection.execute(
            update(process_epochs)
            .where(process_epochs.c.id == worker_epoch_id)
            .values(lease_expires_at=expired_at)
        )
        connection.execute(
            update(worker_leases)
            .where(worker_leases.c.worker_epoch_id == worker_epoch_id)
            .values(lease_expires_at=expired_at)
        )

    epochs = ProcessEpochRepository(engine)
    broadcaster = JobEventBroadcaster(
        repository,
        epoch_repository=epochs,
        poll_seconds=0.01,
    )
    broadcaster.start()
    try:
        deadline = time.monotonic() + 2
        while (
            epochs.is_active_epoch(role="worker", epoch_id=worker_epoch_id)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert not epochs.is_active_epoch(
            role="worker",
            epoch_id=worker_epoch_id,
        )
    finally:
        broadcaster.close()


def test_job_attempt_heartbeat_exception_cannot_silently_lose_its_lease() -> None:
    class FailingRepository:
        def renew_attempt(self, _fence):
            raise RuntimeError("database unavailable")

    heartbeat = AttemptHeartbeat(
        FailingRepository(),  # type: ignore[arg-type]
        AttemptFence(
            job_id="00000000-0000-0000-0000-000000000001",
            attempt_id="00000000-0000-0000-0000-000000000002",
            lease_token="lease",
            worker_epoch_id="00000000-0000-0000-0000-000000000003",
            lease_expires_at=utcnow() + timedelta(seconds=10),
        ),
        interval_seconds=0.01,
    )

    heartbeat.start()
    try:
        assert heartbeat.fenced.wait(1)
    finally:
        heartbeat.stop()


def test_job_attempt_heartbeat_retries_sqlite_busy_without_fencing() -> None:
    class BusyThenRenewRepository:
        def __init__(self) -> None:
            self.calls = 0

        def renew_attempt(self, fence):
            self.calls += 1
            if self.calls == 1:
                raise SqlAlchemyOperationalError(
                    "UPDATE jobs",
                    (),
                    sqlite3.OperationalError("database is locked"),
                )
            return fence

    repository = BusyThenRenewRepository()
    heartbeat = AttemptHeartbeat(
        repository,  # type: ignore[arg-type]
        AttemptFence(
            job_id="00000000-0000-0000-0000-000000000001",
            attempt_id="00000000-0000-0000-0000-000000000002",
            lease_token="lease",
            worker_epoch_id="00000000-0000-0000-0000-000000000003",
            lease_expires_at=utcnow() + timedelta(seconds=10),
        ),
        interval_seconds=0.01,
    )

    heartbeat.start()
    try:
        deadline = time.monotonic() + 1
        while repository.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert repository.calls >= 2
        assert not heartbeat.fenced.is_set()
    finally:
        heartbeat.stop()


def test_job_attempt_heartbeat_fences_after_finite_sqlite_busy_retries() -> None:
    class AlwaysBusyRepository:
        def __init__(self) -> None:
            self.calls = 0

        def renew_attempt(self, _fence):
            self.calls += 1
            raise SqlAlchemyOperationalError(
                "UPDATE jobs",
                (),
                sqlite3.OperationalError("database is locked"),
            )

    repository = AlwaysBusyRepository()
    heartbeat = AttemptHeartbeat(
        repository,  # type: ignore[arg-type]
        AttemptFence(
            job_id="00000000-0000-0000-0000-000000000001",
            attempt_id="00000000-0000-0000-0000-000000000002",
            lease_token="lease",
            worker_epoch_id="00000000-0000-0000-0000-000000000003",
            lease_expires_at=utcnow() + timedelta(seconds=10),
        ),
        interval_seconds=0.01,
    )

    heartbeat.start()
    try:
        assert heartbeat.fenced.wait(1)
        assert repository.calls == 2
    finally:
        heartbeat.stop()


def test_failed_item_retry_creates_related_replacement_from_durable_facts(
    job_platform,
) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    created = repository.create_batch(
        kind="export",
        display_name="export source",
        specs=[
            JobSpec(
                kind="export",
                config={"format": "zip"},
                items=(
                    JobItemSpec(page_id=None, step_kinds=("package",)),
                    JobItemSpec(page_id=None, step_kinds=("package",)),
                ),
            )
        ],
    )
    source_id = str(created["jobIds"][0])
    with engine.begin() as connection:
        source_items = list(
            connection.execute(
                select(job_items.c.id)
                .where(job_items.c.job_id == source_id)
                .order_by(job_items.c.ordinal)
            ).scalars()
        )
        connection.execute(
            update(job_items)
            .where(job_items.c.id == source_items[0])
            .values(status="completed")
        )
        connection.execute(
            update(job_items)
            .where(job_items.c.id == source_items[1])
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        connection.execute(
            update(job_steps)
            .where(job_steps.c.job_item_id == source_items[1])
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.id == source_id)
            .values(status="completed_with_errors", queue_rank=None)
        )

    retried = JobRetryService(engine).retry(
        job_id=source_id,
        failed_only=True,
        strategy="original",
        idempotency_key="retry-fixture",
    )
    replacement_id = str(retried["jobIds"][0])
    detail = repository.get_job(replacement_id)
    assert retried["sourceJobId"] == source_id
    assert retried["failedOnly"] is True
    assert detail["retryOfJobId"] == source_id
    assert detail["retryMode"] == "original"
    assert detail["counts"]["total"] == 1


def test_batch_prioritize_cancel_and_continue_are_database_owned(
    job_platform,
) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    first = repository.create_batch(
        kind="export",
        display_name="first",
        specs=[
            JobSpec(
                kind="export",
                config={},
                items=(JobItemSpec(page_id=None, step_kinds=("package",)),),
            )
        ],
    )
    target = repository.create_batch(
        kind="export",
        display_name="target",
        specs=[
            JobSpec(
                kind="export",
                config={},
                items=(JobItemSpec(page_id=None, step_kinds=("package",)),),
            ),
            JobSpec(
                kind="export",
                config={},
                items=(JobItemSpec(page_id=None, step_kinds=("package",)),),
            ),
        ],
    )
    target_ids = [str(value) for value in target["jobIds"]]
    queue_revision = int(repository.list_jobs()["queueRevision"])
    repository.prioritize_batch(
        batch_id=str(target["batchId"]),
        base_revision=queue_revision,
    )
    assert [
        row["jobId"] for row in repository.list_jobs()["items"]
    ] == [*target_ids, str(first["jobIds"][0])]

    with engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == target_ids[0])
            .values(status="paused")
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.id == target_ids[1])
            .values(status="interrupted")
        )
    continued = repository.continue_batch(str(target["batchId"]))
    assert continued["continued"] == 2
    assert [row["jobId"] for row in continued["jobs"]] == target_ids
    assert all(row["status"] == "queued" for row in continued["jobs"])

    assert repository.cancel_batch_queued(str(target["batchId"])) == 2
    with engine.connect() as connection:
        assert connection.execute(
            select(job_items.c.status)
            .where(job_items.c.job_id.in_(target_ids))
            .order_by(job_items.c.job_id)
        ).scalars().all() == ["cancelled", "cancelled"]
        assert connection.execute(
            select(job_steps.c.status)
            .join(job_items, job_items.c.id == job_steps.c.job_item_id)
            .where(job_items.c.job_id.in_(target_ids))
            .order_by(job_items.c.job_id)
        ).scalars().all() == ["cancelled", "cancelled"]


def test_interrupted_job_is_listed_only_in_history(job_platform) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    job_id = _create_job(repository)
    with engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id)
            .values(status="interrupted", queue_rank=None)
        )

    assert all(
        row["jobId"] != job_id
        for row in repository.list_jobs(scope="queue")["items"]
    )
    assert [
        row["jobId"]
        for row in repository.list_jobs(scope="history")["items"]
        if row["jobId"] == job_id
    ] == [job_id]


def test_clear_history_deletes_retry_children_before_sources_and_protects_live_lineage(
    job_platform,
) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    source_id = _create_job(repository)
    with engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == source_id)
            .values(status="failed", queue_rank=None)
        )
    child = repository.create_batch(
        kind="export",
        display_name="retry child",
        specs=[
            JobSpec(
                kind="export",
                config={},
                items=(JobItemSpec(page_id=None, step_kinds=("package",)),),
                retry_of_job_id=source_id,
                retry_mode="original",
            )
        ],
    )
    child_id = str(child["jobIds"][0])
    assert repository.clear_history() == 0
    assert repository.get_job(source_id)["status"] == "failed"

    with engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == child_id)
            .values(status="failed", queue_rank=None)
        )
    assert repository.clear_history() == 2
    with pytest.raises(LookupError):
        repository.get_job(source_id)


def test_history_retains_latest_200_batches_and_cascades_old_members(
    job_platform,
) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    base = datetime(2025, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None)
    terminal_batches: list[str] = []
    terminal_jobs: list[str] = []
    with engine.begin() as connection:
        interrupted_batch = str(uuid.uuid4())
        interrupted_job = str(uuid.uuid4())
        connection.execute(
            insert(job_batches).values(
                id=interrupted_batch,
                kind="export",
                display_name="interrupted",
                status_summary_json='{"interrupted":1,"total":1}',
                created_at=base,
                updated_at=base,
            )
        )
        connection.execute(
            insert(jobs).values(
                id=interrupted_job,
                batch_id=interrupted_batch,
                kind="export",
                status="interrupted",
                config_json="{}",
                created_at=base,
                updated_at=base,
            )
        )
        for index in range(205):
            batch_id = str(uuid.uuid4())
            job_id = str(uuid.uuid4())
            created_at = base + timedelta(seconds=index + 1)
            terminal_batches.append(batch_id)
            terminal_jobs.append(job_id)
            connection.execute(
                insert(job_batches).values(
                    id=batch_id,
                    kind="export",
                    display_name=f"history-{index}",
                    status_summary_json='{"completed":1,"total":1}',
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
            connection.execute(
                insert(jobs).values(
                    id=job_id,
                    batch_id=batch_id,
                    kind="export",
                    status="completed",
                    config_json="{}",
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
            connection.execute(
                insert(job_config_snapshots).values(
                    job_id=job_id,
                    payload_json="{}",
                    schema_version=1,
                )
            )
            connection.execute(
                insert(job_events).values(
                    id=index + 1,
                    job_id=job_id,
                    event_type="job_completed",
                    payload_json="{}",
                    created_at=created_at,
                )
            )

    assert repository.prune_history() == 5
    with engine.connect() as connection:
        remaining_batches = set(
            connection.execute(select(job_batches.c.id)).scalars()
        )
        remaining_jobs = set(
            connection.execute(select(jobs.c.id)).scalars()
        )
        event_job_ids = set(
            connection.execute(select(job_events.c.job_id)).scalars()
        )
        snapshot_job_ids = set(
            connection.execute(
                select(job_config_snapshots.c.job_id)
            ).scalars()
        )
    assert interrupted_batch in remaining_batches
    assert interrupted_job in remaining_jobs
    assert set(terminal_batches[:5]).isdisjoint(remaining_batches)
    assert set(terminal_jobs[:5]).isdisjoint(remaining_jobs)
    assert len(set(terminal_batches) & remaining_batches) == 200
    assert len(set(terminal_jobs) & remaining_jobs) == 200
    assert event_job_ids == set(terminal_jobs[5:])
    assert snapshot_job_ids == set(terminal_jobs[5:])


def test_history_clear_preserves_unexpired_download_artifacts(
    job_platform,
) -> None:
    engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    job_id = _create_job(repository)
    asset_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
        hours=1
    )
    with engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id)
            .values(status="completed", queue_rank=None)
        )
        connection.execute(
            insert(assets).values(
                id=asset_id,
                relative_path=f"objects/aa/{asset_id}.zip",
                mime_type="application/zip",
                checksum="3" * 64,
                byte_size=1,
            )
        )
        connection.execute(
            insert(job_artifacts).values(
                job_id=job_id,
                kind="download",
                asset_id=asset_id,
                expires_at=expires_at,
            )
        )
    assert repository.clear_history() == 0
    assert repository.get_job(job_id)["status"] == "completed"
    with engine.begin() as connection:
        connection.execute(
            update(job_artifacts)
            .where(job_artifacts.c.job_id == job_id)
            .values(
                expires_at=datetime.now(timezone.utc).replace(tzinfo=None)
                - timedelta(seconds=1)
            )
        )
    assert repository.clear_history() == 1
    with pytest.raises(LookupError):
        repository.get_job(job_id)


def test_parallel_worker_uses_serial_stage_pools_with_cross_stage_overlap(
    job_platform,
) -> None:
    _engine, repository, _book, chapter, worker_epoch_id = job_platform
    created = repository.create_batch(
        kind="translation",
        display_name="parallel pipeline",
        specs=[
            JobSpec(
                kind="translation",
                chapter_id=str(chapter["id"]),
                config={
                    "executionMode": "parallel",
                    "deepLearningConcurrency": 2,
                },
                items=tuple(
                    JobItemSpec(page_id=None, step_kinds=("detect", "ocr"))
                    for _index in range(3)
                ),
            )
        ],
    )
    job_id = str(created["jobIds"][0])
    state_lock = threading.Lock()
    active = {"detect": 0, "ocr": 0}
    max_active = {"detect": 0, "ocr": 0}
    cross_stage_overlap = threading.Event()

    def handler(_fence, step):
        kind = str(step["stepKind"])
        with state_lock:
            active[kind] += 1
            max_active[kind] = max(max_active[kind], active[kind])
            if active["detect"] and active["ocr"]:
                cross_stage_overlap.set()
        time.sleep(0.06)
        with state_lock:
            active[kind] -= 1
        return {"done": kind}

    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"detect": handler, "ocr": handler},
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if repository.get_job(job_id)["status"] == "completed":
            break
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2)
    assert repository.get_job(job_id)["status"] == "completed"
    assert max_active == {"detect": 1, "ocr": 1}
    assert cross_stage_overlap.is_set()


def test_parallel_pipeline_limits_upstream_lead_to_fifty_items(
    job_platform,
) -> None:
    _engine, repository, _book, chapter, worker_epoch_id = job_platform
    page_count = PARALLEL_PIPELINE_LEAD_WINDOW + 1
    created = repository.create_batch(
        kind="translation",
        display_name="bounded parallel pipeline",
        specs=[
            JobSpec(
                kind="translation",
                chapter_id=str(chapter["id"]),
                config={
                    "executionMode": "parallel",
                    "deepLearningConcurrency": 1,
                },
                items=tuple(
                    JobItemSpec(page_id=None, step_kinds=("detect", "save"))
                    for _index in range(page_count)
                ),
            )
        ],
    )
    job_id = str(created["jobIds"][0])
    state_lock = threading.Lock()
    detected = 0
    reached_window = threading.Event()
    save_started = threading.Event()
    release_save = threading.Event()

    def handler(_fence, step):
        nonlocal detected
        kind = str(step["stepKind"])
        if kind == "detect":
            with state_lock:
                detected += 1
                if detected == PARALLEL_PIPELINE_LEAD_WINDOW:
                    reached_window.set()
            return {"done": kind}
        save_started.set()
        assert release_save.wait(5)
        return {"done": kind}

    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"detect": handler, "save": handler},
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert save_started.wait(2)
        assert reached_window.wait(5)
        time.sleep(0.25)
        with state_lock:
            assert detected == PARALLEL_PIPELINE_LEAD_WINDOW

        release_save.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if repository.get_job(job_id)["status"] == "completed":
                break
            time.sleep(0.01)
        assert repository.get_job(job_id)["status"] == "completed"
        with state_lock:
            assert detected == page_count
    finally:
        release_save.set()
        stop.set()
        thread.join(timeout=2)


def test_parallel_worker_quiesces_completed_pool_during_long_tail(
    job_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, repository, _book, chapter, worker_epoch_id = job_platform
    page_count = 4
    created = repository.create_batch(
        kind="translation",
        display_name="parallel long tail",
        specs=[
            JobSpec(
                kind="translation",
                chapter_id=str(chapter["id"]),
                config={
                    "executionMode": "parallel",
                    "deepLearningConcurrency": 1,
                },
                items=tuple(
                    JobItemSpec(page_id=None, step_kinds=("detect", "translate"))
                    for _index in range(page_count)
                ),
            )
        ],
    )
    job_id = str(created["jobIds"][0])
    state_lock = threading.Lock()
    next_step_calls = {"detect": 0, "translate": 0}
    detected = 0
    all_detected = threading.Event()
    translate_started = threading.Event()
    release_translate = threading.Event()
    original_next_step = repository.next_step

    def counted_next_step(
        fence,
        *,
        allowed_kinds=None,
        max_item_ordinal=None,
    ):
        kind = str(allowed_kinds[0])
        with state_lock:
            next_step_calls[kind] += 1
        return original_next_step(
            fence,
            allowed_kinds=allowed_kinds,
            max_item_ordinal=max_item_ordinal,
        )

    monkeypatch.setattr(repository, "next_step", counted_next_step)

    def handler(_fence, step):
        nonlocal detected
        kind = str(step["stepKind"])
        if kind == "detect":
            with state_lock:
                detected += 1
                if detected == page_count:
                    all_detected.set()
            return {"done": kind}
        translate_started.set()
        assert release_translate.wait(3)
        return {"done": kind}

    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"detect": handler, "translate": handler},
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert translate_started.wait(2)
        assert all_detected.wait(2)
        time.sleep(0.15)
        with state_lock:
            calls_after_quiescence = next_step_calls["detect"]
        time.sleep(0.3)
        with state_lock:
            calls_after_long_tail = next_step_calls["detect"]
        assert calls_after_long_tail == calls_after_quiescence
        assert calls_after_long_tail <= page_count + 1

        release_translate.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if repository.get_job(job_id)["status"] == "completed":
                break
            time.sleep(0.01)
        assert repository.get_job(job_id)["status"] == "completed"
    finally:
        release_translate.set()
        stop.set()
        thread.join(timeout=2)


def test_parallel_supervisor_uses_bounded_wait_during_running_step(
    job_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, repository, _book, chapter, worker_epoch_id = job_platform
    created = repository.create_batch(
        kind="translation",
        display_name="bounded supervisor polling",
        specs=[
            JobSpec(
                kind="translation",
                chapter_id=str(chapter["id"]),
                config={
                    "executionMode": "parallel",
                    "deepLearningConcurrency": 1,
                },
                items=(JobItemSpec(page_id=None, step_kinds=("detect",)),),
            )
        ],
    )
    job_id = str(created["jobIds"][0])
    started = threading.Event()
    release = threading.Event()
    active_count_calls = 0
    scoped_active_count_calls = 0
    count_lock = threading.Lock()
    original_active_step_counts = repository.active_step_counts

    def counted_active_step_counts(*args, **kwargs):
        nonlocal active_count_calls, scoped_active_count_calls
        with count_lock:
            active_count_calls += 1
            if kwargs.get("step_kind") is not None:
                scoped_active_count_calls += 1
        return original_active_step_counts(*args, **kwargs)

    monkeypatch.setattr(
        repository,
        "active_step_counts",
        counted_active_step_counts,
    )

    def handler(_fence, _step):
        started.set()
        assert release.wait(3)
        return {"done": True}

    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"detect": handler},
        safe_point=lambda: False,
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    try:
        assert started.wait(2)
        time.sleep(0.35)
        with count_lock:
            calls_while_running = active_count_calls
        assert calls_while_running <= 5
        assert scoped_active_count_calls == 0

        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if repository.get_job(job_id)["status"] == "completed":
                break
            time.sleep(0.01)
        assert repository.get_job(job_id)["status"] == "completed"
    finally:
        release.set()
        stop.set()
        thread.join(timeout=2)


def test_parallel_worker_defers_persistent_sqlite_busy_during_stage_admission(
    job_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, repository, _book, chapter, worker_epoch_id = job_platform
    created = repository.create_batch(
        kind="translation",
        display_name="parallel busy retry",
        specs=[
            JobSpec(
                kind="translation",
                chapter_id=str(chapter["id"]),
                config={
                    "executionMode": "parallel",
                    "deepLearningConcurrency": 1,
                },
                items=(JobItemSpec(page_id=None, step_kinds=("detect",)),),
            )
        ],
    )
    job_id = str(created["jobIds"][0])
    original_next_step = repository.next_step
    admission_calls = 0

    def busy_then_succeed(fence, *, allowed_kinds=None):
        nonlocal admission_calls
        admission_calls += 1
        if admission_calls <= 4:
            raise SqlAlchemyOperationalError(
                "BEGIN IMMEDIATE",
                (),
                sqlite3.OperationalError("database is locked"),
            )
        return original_next_step(fence, allowed_kinds=allowed_kinds)

    monkeypatch.setattr(repository, "next_step", busy_then_succeed)
    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"detect": lambda _fence, _step: {"done": True}},
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if repository.get_job(job_id)["status"] == "completed":
                break
            time.sleep(0.01)
        assert repository.get_job(job_id)["status"] == "completed"
        assert admission_calls == 5
    finally:
        stop.set()
        thread.join(timeout=2)


def test_parallel_worker_enforces_frozen_deep_learning_concurrency(
    job_platform,
) -> None:
    _engine, repository, _book, chapter, worker_epoch_id = job_platform
    created = repository.create_batch(
        kind="translation",
        display_name="bounded deep learning pipeline",
        specs=[
            JobSpec(
                kind="translation",
                chapter_id=str(chapter["id"]),
                config={
                    "executionMode": "parallel",
                    "deepLearningConcurrency": 1,
                },
                items=tuple(
                    JobItemSpec(page_id=None, step_kinds=("detect", "ocr"))
                    for _index in range(3)
                ),
            )
        ],
    )
    job_id = str(created["jobIds"][0])
    state_lock = threading.Lock()
    active_count = 0
    maximum_active = 0

    def handler(_fence, step):
        nonlocal active_count, maximum_active
        with state_lock:
            active_count += 1
            maximum_active = max(maximum_active, active_count)
        time.sleep(0.04)
        with state_lock:
            active_count -= 1
        return {"done": str(step["stepKind"])}

    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"detect": handler, "ocr": handler},
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if repository.get_job(job_id)["status"] == "completed":
            break
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2)
    assert repository.get_job(job_id)["status"] == "completed"
    assert maximum_active == 1


def test_parallel_progress_is_backend_owned_and_recovers_pool_lock_waiting(
    job_platform,
) -> None:
    _engine, repository, _book, chapter, worker_epoch_id = job_platform
    created = repository.create_batch(
        kind="translation",
        display_name="durable pool progress",
        specs=[
            JobSpec(
                kind="translation",
                chapter_id=str(chapter["id"]),
                config={
                    "executionMode": "parallel",
                    "deepLearningConcurrency": 1,
                },
                items=(
                    JobItemSpec(page_id=None, step_kinds=("detect",)),
                    JobItemSpec(page_id=None, step_kinds=("ocr",)),
                ),
            )
        ],
    )
    job_id = str(created["jobIds"][0])
    release = threading.Event()

    def handler(_fence, step):
        assert release.wait(3)
        return {"done": str(step["stepKind"])}

    stop = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"detect": handler, "ocr": handler, "render": handler},
        idle_poll_seconds=0.01,
    )
    thread = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    thread.start()
    snapshot: dict[str, object] = {}
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        snapshot = repository.get_job(job_id)["progress"]
        pools = snapshot.get("pools", [])
        if (
            len(pools) == 2
            and sum(int(pool["processing"]) for pool in pools) == 2
            and any(bool(pool["lockWaiting"]) for pool in pools)
        ):
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"lock-waiting progress was not persisted: {snapshot}")

    assert snapshot["executionMode"] == "parallel"
    assert [pool["kind"] for pool in snapshot["pools"]] == ["detect", "ocr"]
    assert all(int(pool["total"]) == 1 for pool in snapshot["pools"])
    assert all(int(pool["waiting"]) == 0 for pool in snapshot["pools"])

    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if repository.get_job(job_id)["status"] == "completed":
            break
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2)
    final = repository.get_job(job_id)["progress"]
    assert repository.get_job(job_id)["status"] == "completed"
    assert final["jobStatus"] == "completed"
    assert all(int(pool["completed"]) == 1 for pool in final["pools"])
    assert all(not bool(pool["lockWaiting"]) for pool in final["pools"])
