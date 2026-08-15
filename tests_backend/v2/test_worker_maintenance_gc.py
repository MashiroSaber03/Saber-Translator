from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

import pytest
from sqlalchemy import delete, insert, select, update

from src.backend_v2.insight.gc import InsightReachabilityGarbageCollector
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_runs,
    assets,
    books,
    continuation_projects,
    idempotency_records,
    job_artifacts,
    job_batches,
    jobs,
    metadata,
    operation_artifacts,
    operations,
    timeline_versions,
    vector_generations,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.worker.maintenance import WorkerMaintenance


def _stored_job_progress(status: str) -> str:
    return json.dumps(
        {
            "executionMode": "sequential",
            "jobStatus": status,
            "totalItems": 0,
            "completedItems": 0,
            "failedItems": 0,
            "skippedItems": 0,
            "cancelledItems": 0,
            "pools": [],
        },
        separators=(",", ":"),
    )


@pytest.fixture()
def maintenance_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    try:
        yield data_root, engine
    finally:
        engine.dispose()


def _operation_values(
    operation_id: str,
    *,
    timestamp,
) -> dict[str, object]:
    return {
        "id": operation_id,
        "kind": "bubble_translate",
        "executor_role": "api",
        "status": "completed",
        "request_json": "{}",
        "result_json": "{}",
        "finished_at": timestamp,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def test_maintenance_prunes_bounded_ttl_records_without_active_work(
    maintenance_platform,
) -> None:
    data_root, engine = maintenance_platform
    now = utcnow()
    old = now - timedelta(days=31)
    recent = now - timedelta(days=1)
    batch_id = "active-batch"
    job_id = "active-job"
    asset_id = "expired-artifact-asset"
    with engine.begin() as connection:
        connection.execute(
            insert(operations),
            [
                _operation_values("old-operation", timestamp=old),
                _operation_values("idempotent-operation", timestamp=old),
                _operation_values("recent-operation", timestamp=recent),
            ],
        )
        connection.execute(
            insert(job_batches).values(
                id=batch_id,
                kind="export",
                display_name="active export",
            )
        )
        connection.execute(
            insert(jobs).values(
                id=job_id,
                batch_id=batch_id,
                kind="export",
                status="queued",
                queue_rank=1,
                config_json="{}",
                latest_progress_json=_stored_job_progress("queued"),
            )
        )
        connection.execute(
            insert(idempotency_records),
            [
                {
                    "scope": "operation",
                    "key": "still-visible",
                    "request_hash": "a" * 64,
                    "http_status": 202,
                    "response_json": "{}",
                    "resource_type": "operation",
                    "resource_id": "idempotent-operation",
                    "created_at": old,
                    "expires_at": now + timedelta(days=1),
                },
                {
                    "scope": "settings",
                    "key": "expired",
                    "request_hash": "b" * 64,
                    "http_status": 200,
                    "response_json": "{}",
                    "resource_type": "settings",
                    "resource_id": None,
                    "created_at": old,
                    "expires_at": now - timedelta(seconds=1),
                },
                {
                    "scope": "jobs",
                    "key": "active-batch",
                    "request_hash": "c" * 64,
                    "http_status": 201,
                    "response_json": "{}",
                    "resource_type": "job_batch",
                    "resource_id": batch_id,
                    "created_at": old,
                    "expires_at": now - timedelta(seconds=1),
                },
            ],
        )
        connection.execute(
            insert(assets).values(
                id=asset_id,
                relative_path="objects/expired.bin",
                mime_type="application/octet-stream",
                checksum="d" * 64,
                byte_size=1,
            )
        )
        connection.execute(
            insert(job_artifacts).values(
                job_id=job_id,
                kind="archive",
                asset_id=asset_id,
                expires_at=now - timedelta(seconds=1),
            )
        )
        connection.execute(
            insert(operation_artifacts).values(
                operation_id="recent-operation",
                kind="preview",
                asset_id=asset_id,
                expires_at=now - timedelta(seconds=1),
            )
        )

    maintenance = WorkerMaintenance(data_root=data_root, engine=engine)
    assert maintenance._prune_expired_artifacts() == {
        "jobArtifacts": 1,
        "operationArtifacts": 1,
    }
    assert maintenance._prune_terminal_operations() == 1
    assert maintenance._prune_idempotency_records() == 1

    with engine.connect() as connection:
        assert set(connection.execute(select(operations.c.id)).scalars()) == {
            "idempotent-operation",
            "recent-operation",
        }
        assert set(
            connection.execute(select(idempotency_records.c.key)).scalars()
        ) == {"still-visible", "active-batch"}
        assert connection.execute(select(job_artifacts.c.kind)).first() is None
        assert (
            connection.execute(select(operation_artifacts.c.kind)).first()
            is None
        )

    with engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id)
            .values(
                status="completed",
                queue_rank=None,
                finished_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            delete(idempotency_records).where(
                idempotency_records.c.key == "still-visible"
            )
        )
    assert maintenance._prune_idempotency_records() == 1
    assert maintenance._prune_terminal_operations() == 1


def _run_values(run_id: str, book_id: str, *, timestamp, job_id=None):
    return {
        "id": run_id,
        "book_id": book_id,
        "job_id": job_id,
        "scope": "full",
        "status": "completed",
        "config_json": "{}",
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def test_insight_gc_keeps_only_reachable_runs_and_partial_job_previews(
    maintenance_platform,
) -> None:
    _data_root, engine = maintenance_platform
    now = utcnow()
    old = now - timedelta(hours=2)
    recent = now - timedelta(minutes=30)
    book_id = "insight-gc-book"
    retained_job_id = "retained-insight-job"
    run_ids = {
        "orphan-run",
        "head-run",
        "continuation-run",
        "job-run",
        "derived-run",
        "recent-run",
        "abandoned-staging-run",
    }
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id=book_id, title="Insight GC")
        )
        connection.execute(
            insert(jobs).values(
                id=retained_job_id,
                kind="insight_analysis",
                status="completed",
                config_json="{}",
                latest_progress_json=_stored_job_progress("completed"),
                finished_at=old,
                created_at=old,
                updated_at=old,
            )
        )
        rows = [
            _run_values(run_id, book_id, timestamp=old)
            for run_id in run_ids - {"job-run", "recent-run", "abandoned-staging-run"}
        ]
        rows.extend(
            [
                _run_values(
                    "job-run",
                    book_id,
                    timestamp=old,
                    job_id=retained_job_id,
                ),
                _run_values("recent-run", book_id, timestamp=recent),
                {
                    **_run_values(
                        "abandoned-staging-run",
                        book_id,
                        timestamp=old,
                    ),
                    "status": "staging",
                },
            ]
        )
        connection.execute(insert(analysis_runs), rows)
        connection.execute(
            insert(analysis_heads).values(
                id="book-analysis-head",
                book_id=book_id,
                active_run_id="head-run",
                created_at=old,
                updated_at=old,
            )
        )
        connection.execute(
            insert(continuation_projects).values(
                id="continuation-project",
                book_id=book_id,
                source_run_id="continuation-run",
                payload_json="{}",
                created_at=old,
                updated_at=old,
            )
        )
        connection.execute(
            insert(analysis_artifacts),
            [
                {
                    "id": "orphan-artifact",
                    "book_id": book_id,
                    "run_id": "orphan-run",
                    "kind": "overview",
                    "status": "stale",
                    "is_active": False,
                    "dependency_fingerprint": "orphan",
                    "payload_json": "{}",
                    "created_at": old,
                    "updated_at": old,
                },
                {
                    "id": "active-artifact",
                    "book_id": book_id,
                    "run_id": "derived-run",
                    "kind": "compressed_context",
                    "status": "ready",
                    "is_active": True,
                    "dependency_fingerprint": "active",
                    "payload_json": "{}",
                    "created_at": old,
                    "updated_at": old,
                },
            ],
        )
        connection.execute(
            insert(timeline_versions).values(
                id="job-partial-timeline",
                book_id=book_id,
                run_id="job-run",
                mode="enhanced",
                status="stale",
                content_json="{}",
                dependency_fingerprint="partial",
                is_active=False,
                created_at=old,
                updated_at=old,
            )
        )
        connection.execute(
            insert(vector_generations).values(
                id="orphan-vector",
                book_id=book_id,
                run_id="orphan-run",
                generation=1,
                status="failed",
                dependency_fingerprint="orphan-vector",
                is_active=False,
                created_at=old,
                updated_at=old,
            )
        )

    collector = InsightReachabilityGarbageCollector(engine)
    assert collector.collect(now=now) == {
        "analysisArtifacts": 1,
        "timelineVersions": 0,
        "vectorGenerations": 1,
        "analysisRuns": 2,
    }
    with engine.connect() as connection:
        remaining_runs = set(
            connection.execute(select(analysis_runs.c.id)).scalars()
        )
        assert remaining_runs == {
            "head-run",
            "continuation-run",
            "job-run",
            "derived-run",
            "recent-run",
        }
        assert set(
            connection.execute(select(analysis_artifacts.c.id)).scalars()
        ) == {"active-artifact"}
        assert set(
            connection.execute(select(timeline_versions.c.id)).scalars()
        ) == {"job-partial-timeline"}
        assert connection.execute(select(vector_generations.c.id)).first() is None

    with engine.begin() as connection:
        connection.execute(delete(jobs).where(jobs.c.id == retained_job_id))
    second = collector.collect(now=now)
    assert second["timelineVersions"] == 1
    assert second["analysisRuns"] == 1
    with engine.connect() as connection:
        assert (
            connection.execute(
                select(analysis_runs.c.id).where(analysis_runs.c.id == "job-run")
            ).first()
            is None
        )
