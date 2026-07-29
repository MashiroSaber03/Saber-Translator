from __future__ import annotations

from pathlib import Path
import threading
import time
import uuid

import pytest
from sqlalchemy import insert, select, update

from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.events import JobEventBroadcaster
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    InvalidJobTransition,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import (
    chapter_write_intents,
    chapter_write_locks,
    job_items,
    job_steps,
    jobs,
    metadata,
    operations,
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


def test_shared_event_broadcaster_replays_and_fans_out(job_platform) -> None:
    _engine, repository, _book, _chapter, _worker_epoch_id = job_platform
    existing_job = _create_job(repository)
    existing = repository.events_after(after=0)
    assert existing[-1]["jobId"] == existing_job

    broadcaster = JobEventBroadcaster(
        repository,
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
    finally:
        broadcaster.unsubscribe(subscription)
        broadcaster.close()


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
