from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, insert, inspect, text
from sqlalchemy.exc import IntegrityError

from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.lifecycle import (
    SCHEMA_REVISION,
    initialize_database,
    schema_smoke_test,
)
from src.backend_v2.storage.schema import (
    CURRENT_JOB_STATUSES,
    EXECUTING_JOB_STATUSES,
    JOB_STATUSES,
    NONTERMINAL_JOB_STATUSES,
    assets,
    books,
    chapters,
    job_batches,
    job_events,
    job_items,
    job_steps,
    jobs,
    metadata,
    pages,
    web_import_drafts,
)


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
def engine(tmp_path: Path):
    database_engine = create_sqlite_engine(tmp_path / "contract.sqlite3")
    metadata.create_all(database_engine)
    try:
        yield database_engine
    finally:
        database_engine.dispose()


def test_sqlite_runtime_pragmas_and_foreign_keys_are_clean(engine) -> None:
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert connection.execute(text("PRAGMA busy_timeout")).scalar_one() == 5000
        assert connection.execute(text("PRAGMA foreign_key_check")).all() == []


def test_schema_contains_backend_first_ownership_tables(engine) -> None:
    expected = {
        "assets",
        "books",
        "chapters",
        "pages",
        "page_assets",
        "jobs",
        "job_steps",
        "operations",
        "process_epochs",
        "chapter_write_locks",
        "object_commit_journal",
        "idempotency_records",
        "schema_metadata",
    }
    with engine.connect() as connection:
        actual = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
        }
    assert expected <= actual

    assert metadata.tables["plugins"].c.id.type.length == 100
    assert metadata.tables["plugin_versions"].c.plugin_id.type.length == 100
    assert metadata.tables["plugin_current_versions"].c.plugin_id.type.length == 100


def test_task_schema_vocabulary_is_exact() -> None:
    assert JOB_STATUSES == (
        "queued",
        "running",
        "paused",
        "cancelled",
        "completed",
        "completed_with_errors",
        "failed",
        "interrupted",
    )
    assert CURRENT_JOB_STATUSES == ("running", "paused")
    assert EXECUTING_JOB_STATUSES == ("running",)
    assert NONTERMINAL_JOB_STATUSES == (
        "queued",
        "running",
        "paused",
        "interrupted",
    )
    assert tuple(job_batches.c.keys()) == (
        "id",
        "owner_user_id",
        "display_name",
        "created_at",
    )
    assert "blocked_reason" not in jobs.c


def test_every_non_primary_foreign_key_has_a_leading_lookup_index() -> None:
    missing: list[str] = []
    for table in metadata.tables.values():
        indexed_leaders = {
            getattr(index.expressions[0], "name", None)
            for index in table.indexes
            if index.expressions
        }
        indexed_leaders.update(
            next(iter(constraint.columns)).name
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint) and constraint.columns
        )
        primary_keys = {column.name for column in table.primary_key.columns}
        for column in table.columns:
            if (
                column.foreign_keys
                and column.name not in primary_keys
                and column.name not in indexed_leaders
            ):
                missing.append(f"{table.name}.{column.name}")
    assert missing == []


def test_every_asset_reference_has_a_leading_lookup_index() -> None:
    missing: list[str] = []
    for table in metadata.tables.values():
        indexed_leaders = {
            getattr(index.expressions[0], "name", None)
            for index in table.indexes
            if index.expressions
        }
        indexed_leaders.update(
            next(iter(constraint.columns)).name
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint) and constraint.columns
        )
        primary_key_columns = list(table.primary_key.columns)
        if primary_key_columns:
            indexed_leaders.add(primary_key_columns[0].name)
        for column in table.columns:
            references_asset = any(
                foreign_key.column.table is assets
                for foreign_key in column.foreign_keys
            )
            if references_asset and column.name not in indexed_leaders:
                missing.append(f"{table.name}.{column.name}")
    assert missing == []


def test_large_dataset_hot_queries_use_declared_indexes(engine) -> None:
    now = datetime(2026, 7, 29, 12)
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="scale-book", kind="library", title="Scale")
        )
        connection.execute(
            insert(chapters).values(
                id="scale-chapter",
                book_id="scale-book",
                ordinal=1,
                title="Scale Chapter",
            )
        )
        connection.execute(
            insert(pages),
            [
                {
                    "id": f"page-{index:04d}",
                    "chapter_id": "scale-chapter",
                    "ordinal": index + 1,
                    "logical_source_path": f"{index + 1}.png",
                }
                for index in range(1000)
            ],
        )
        connection.execute(
            insert(assets),
            [
                {
                    "id": f"asset-{index:04d}",
                    "relative_path": f"objects/scale/{index:04d}.bin",
                    "mime_type": "application/octet-stream",
                    "checksum": f"{index:064x}",
                    "byte_size": 1,
                    "gc_marked_at": now if index % 2 else None,
                }
                for index in range(1000)
            ],
        )
        connection.execute(
            insert(job_batches),
            [
                {
                    "id": f"batch-{index:04d}",
                    "display_name": f"Batch {index}",
                }
                for index in range(200)
            ],
        )
        connection.execute(
            insert(jobs),
            [
                {
                    "id": f"job-{index:04d}",
                    "batch_id": f"batch-{index:04d}",
                    "kind": "export",
                    "status": "queued" if index < 10 else "completed",
                    "queue_rank": index + 1 if index < 10 else None,
                    "book_id": "scale-book",
                    "chapter_id": "scale-chapter",
                    "config_json": "{}",
                    "latest_progress_json": _stored_job_progress(
                        "queued" if index < 10 else "completed"
                    ),
                    "finished_at": None if index < 10 else now,
                }
                for index in range(200)
            ],
        )
        connection.execute(
            insert(job_events),
            [
                {
                    "id": index + 1,
                    "job_id": f"job-{index % 200:04d}",
                    "event_type": "progress",
                    "payload_json": "{}",
                }
                for index in range(10_000)
            ],
        )
        connection.execute(
            insert(job_items),
            [
                {
                    "id": f"scale-item-{index:04d}",
                    "job_id": "job-0000",
                    "ordinal": index + 1,
                    "status": "completed" if index < 900 else "pending",
                }
                for index in range(1000)
            ],
        )
        connection.execute(
            insert(job_steps),
            [
                {
                    "id": f"scale-step-{index:04d}",
                    "job_item_id": f"scale-item-{index:04d}",
                    "ordinal": 1,
                    "kind": "package",
                    "status": "completed" if index < 900 else "pending",
                }
                for index in range(1000)
            ],
        )
        connection.execute(
            insert(web_import_drafts),
            [
                {
                    "id": f"draft-{index:04d}",
                    "book_id": "scale-book",
                    "chapter_id": "scale-chapter",
                    "status": "ready",
                    "config_json": "{}",
                    "temp_relative_path": f"temp/drafts/{index:04d}",
                    "expires_at": now,
                }
                for index in range(200)
            ],
        )
        connection.exec_driver_sql("ANALYZE")

    def plan(sql: str, parameters: tuple[object, ...]) -> str:
        with engine.connect() as connection:
            rows = connection.exec_driver_sql(
                f"EXPLAIN QUERY PLAN {sql}",
                parameters,
            ).all()
        return "\n".join(str(row[3]) for row in rows)

    expectations = (
        (
            "SELECT id FROM pages WHERE chapter_id = ? ORDER BY ordinal",
            ("scale-chapter",),
            "sqlite_autoindex_pages_2",
        ),
        (
            "SELECT id FROM jobs WHERE status = ? ORDER BY queue_rank",
            ("queued",),
            "ix_jobs_queue_claim",
        ),
        (
            "SELECT id FROM jobs WHERE chapter_id = ? AND status = ?",
            ("scale-chapter", "queued"),
            "ix_jobs_chapter_status",
        ),
        (
            "SELECT id FROM jobs WHERE batch_id = ? AND status = ?",
            ("batch-0001", "completed"),
            "ix_jobs_batch_status",
        ),
        (
            "SELECT id FROM job_events WHERE job_id = ? AND id < ? "
            "ORDER BY id DESC LIMIT 50",
            ("job-0001", 9000),
            "ix_job_events_job_cursor",
        ),
        (
            "SELECT status, COUNT(*) FROM job_items WHERE job_id = ? "
            "GROUP BY status",
            ("job-0000",),
            "ix_job_items_job_status_ordinal",
        ),
        (
            "SELECT id FROM assets WHERE gc_marked_at <= ?",
            ("2026-07-29 12:00:00.000000",),
            "ix_assets_gc_marked_at",
        ),
        (
            "SELECT id FROM web_import_drafts "
            "WHERE chapter_id = ? AND status = ? AND expires_at > ?",
            ("scale-chapter", "ready", "2026-07-28 00:00:00"),
            "ix_web_import_drafts_chapter_status_expiry",
        ),
    )
    for sql, parameters, expected_index in expectations:
        detail = plan(sql, parameters)
        assert expected_index in detail, detail
        assert "SCAN " not in detail, detail


def test_history_targets_null_out_and_asset_producers_do_not_own_current_assets(
    engine,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO books(id, kind, title) "
                "VALUES ('history-book', 'library', 'History')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO chapters(id, book_id, ordinal, title) "
                "VALUES ('history-chapter', 'history-book', 1, 'Chapter')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO pages(id, chapter_id, ordinal, logical_source_path) "
                "VALUES ('history-page', 'history-chapter', 1, 'page.png')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO bubbles(id, page_id, ordinal, payload_json, "
                "updated_revision) "
                "VALUES ('history-bubble', 'history-page', 1, '{}', 1)"
            )
        )
        for asset_id in ("job-asset", "operation-asset", "render-asset"):
            connection.execute(
                text(
                    "INSERT INTO assets("
                    "id, relative_path, mime_type, checksum, byte_size"
                    ") VALUES (:id, :path, 'application/octet-stream', :checksum, 1)"
                ),
                {
                    "id": asset_id,
                    "path": f"objects/history/{asset_id}.bin",
                    "checksum": asset_id.ljust(64, "0"),
                },
            )
        connection.execute(
            text(
                "INSERT INTO jobs("
                "id, kind, status, book_id, chapter_id, page_id, "
                "config_json, latest_progress_json, target_display_json"
                ") VALUES ("
                "'history-job', 'export', 'completed', 'history-book', "
                "'history-chapter', 'history-page', '{}', :progress, "
                "'{\"book\":\"History\",\"chapter\":\"Chapter\"}'"
                ")"
            ),
            {"progress": _stored_job_progress("completed")},
        )
        connection.execute(
            text(
                "INSERT INTO job_items(id, job_id, ordinal) "
                "VALUES ('history-item', 'history-job', 1)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO job_steps(id, job_item_id, ordinal, kind) "
                "VALUES ('history-step', 'history-item', 1, 'package')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO operations("
                "id, kind, executor_role, status, page_id, bubble_id, "
                "base_revision, request_json"
                ") VALUES ("
                "'history-operation', 'bubble_color', 'worker', 'completed', "
                "'history-page', 'history-bubble', 1, '{}'"
                ")"
            )
        )
        connection.execute(
            text(
                "INSERT INTO operations("
                "id, kind, executor_role, status, page_id, base_revision, "
                "request_json"
                ") VALUES ("
                "'orphan-operation', 'page_detect', 'worker', 'completed', "
                "'history-page', 1, '{}'"
                ")"
            )
        )
        connection.execute(
            text(
                "INSERT INTO render_requests("
                "id, page_id, requested_revision, completed_revision, status"
                ") VALUES ('history-render', 'history-page', 1, 1, 'completed')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO page_assets("
                "page_id, role, asset_id, producer_job_step_id"
                ") VALUES ('history-page', 'clean', 'job-asset', 'history-step')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO page_assets("
                "page_id, role, asset_id, producer_operation_id"
                ") VALUES ("
                "'history-page', 'translated', 'operation-asset', "
                "'history-operation'"
                ")"
            )
        )
        connection.execute(
            text(
                "INSERT INTO page_assets("
                "page_id, role, asset_id, producer_render_request_id"
                ") VALUES ("
                "'history-page', 'text_mask', 'render-asset', 'history-render'"
                ")"
            )
        )

        connection.execute(
            text("DELETE FROM job_steps WHERE id = 'history-step'")
        )
        connection.execute(
            text("DELETE FROM operations WHERE id = 'history-operation'")
        )
        connection.execute(
            text("DELETE FROM render_requests WHERE id = 'history-render'")
        )
        producers = connection.execute(
            text(
                "SELECT role, producer_job_step_id, producer_operation_id, "
                "producer_render_request_id FROM page_assets "
                "WHERE page_id = 'history-page' ORDER BY role"
            )
        ).all()
        assert producers == [
            ("clean", None, None, None),
            ("text_mask", None, None, None),
            ("translated", None, None, None),
        ]
        assert connection.execute(
            text(
                "SELECT COUNT(*) FROM assets "
                "WHERE id IN ('job-asset','operation-asset','render-asset')"
            )
        ).scalar_one() == 3

        connection.execute(text("DELETE FROM books WHERE id = 'history-book'"))
        history = connection.execute(
            text(
                "SELECT book_id, chapter_id, page_id, target_display_json "
                "FROM jobs WHERE id = 'history-job'"
            )
        ).one()
        assert history[:3] == (None, None, None)
        assert json.loads(history.target_display_json) == {
            "book": "History",
            "chapter": "Chapter",
        }
        assert connection.execute(
            text(
                "SELECT page_id FROM operations "
                "WHERE id = 'orphan-operation'"
            )
        ).one() == (None,)


def test_formal_business_asset_links_restrict_direct_asset_deletion(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO books(id, kind, title) "
                "VALUES ('asset-book', 'library', 'Assets')"
            )
        )
        for asset_id in ("avatar-asset", "reference-asset"):
            connection.execute(
                text(
                    "INSERT INTO assets("
                    "id, relative_path, mime_type, checksum, byte_size"
                    ") VALUES (:id, :path, 'image/png', :checksum, 1)"
                ),
                {
                    "id": asset_id,
                    "path": f"objects/assets/{asset_id}.png",
                    "checksum": asset_id.ljust(64, "0"),
                },
            )
        connection.execute(
            text(
                "INSERT INTO studio_documents("
                "id, book_id, origin_type, title, avatar_asset_id"
                ") VALUES ("
                "'asset-document', 'asset-book', 'manual', 'Character', "
                "'avatar-asset'"
                ")"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_projects(id, book_id) "
                "VALUES ('asset-project', 'asset-book')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_characters(id, project_id, name) "
                "VALUES ('asset-character', 'asset-project', 'Character')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_character_forms("
                "id, character_id, name, reference_asset_id"
                ") VALUES ("
                "'asset-form', 'asset-character', 'Default', 'reference-asset'"
                ")"
            )
        )

    for asset_id in ("avatar-asset", "reference-asset"):
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    text("DELETE FROM assets WHERE id = :id"),
                    {"id": asset_id},
                )


def test_only_one_quick_workspace_can_exist(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO books "
                "(id, kind, title, created_at, updated_at) "
                "VALUES ('book-1', 'quick_workspace', 'Quick', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO books "
                    "(id, kind, title, created_at, updated_at) "
                    "VALUES ('book-2', 'quick_workspace', 'Quick 2', "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )


def test_only_one_current_job_can_exist(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO jobs "
                "(id, kind, status, config_json, latest_progress_json, "
                "created_at, updated_at) "
                "VALUES ('job-1', 'translation', 'running', '{}', :progress, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"progress": _stored_job_progress("running")},
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO jobs "
                    "(id, kind, status, config_json, latest_progress_json, "
                    "created_at, updated_at) "
                    "VALUES ('job-2', 'export', 'running', '{}', :progress, "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"progress": _stored_job_progress("running")},
            )


def test_continuation_current_image_flags_are_unique(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO books(id, kind, title) "
                "VALUES ('image-book', 'library', 'Images')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_projects(id, book_id) "
                "VALUES ('image-project', 'image-book')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_pages(id, project_id, ordinal) "
                "VALUES ('image-page', 'image-project', 1)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_characters(id, project_id, name) "
                "VALUES ('image-character', 'image-project', 'Character')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_character_forms(id, character_id, name) "
                "VALUES ('image-form', 'image-character', 'Default')"
            )
        )
        for index in range(8):
            connection.execute(
                text(
                    "INSERT INTO assets("
                    "id, relative_path, mime_type, checksum, byte_size"
                    ") VALUES (:id, :path, 'image/png', :checksum, 1)"
                ),
                {
                    "id": f"image-asset-{index}",
                    "path": f"objects/image-{index}.png",
                    "checksum": f"{index:064x}",
                },
            )
        connection.execute(
            text(
                "INSERT INTO continuation_image_versions("
                "id, continuation_page_id, asset_id, thumbnail_asset_id, "
                "version, is_active"
                ") VALUES ("
                "'page-version-1', 'image-page', 'image-asset-0', "
                "'image-asset-1', 1, 1)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO continuation_form_image_versions("
                "id, form_id, asset_id, thumbnail_asset_id, version, is_adopted"
                ") VALUES ("
                "'form-version-1', 'image-form', 'image-asset-4', "
                "'image-asset-5', 1, 1)"
            )
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO continuation_image_versions("
                    "id, continuation_page_id, asset_id, thumbnail_asset_id, "
                    "version, is_active"
                    ") VALUES ("
                    "'page-version-2', 'image-page', 'image-asset-2', "
                    "'image-asset-3', 2, 1)"
                )
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO continuation_form_image_versions("
                    "id, form_id, asset_id, thumbnail_asset_id, version, "
                    "is_adopted"
                    ") VALUES ("
                    "'form-version-2', 'image-form', 'image-asset-6', "
                    "'image-asset-7', 2, 1)"
                )
            )


def test_current_foundation_builds_the_exact_schema(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    initialized = initialize_database(data_root)
    database_path = initialized.database_path

    assert initialized.created is True
    assert initialized.schema_revision == SCHEMA_REVISION
    assert schema_smoke_test(database_path) == SCHEMA_REVISION

    engine = create_sqlite_engine(database_path)
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
        assert connection.execute(
            text("SELECT revision FROM schema_metadata WHERE singleton_id = 1")
        ).scalar_one() == SCHEMA_REVISION
        actual_tables = {
            str(row[0])
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            if not str(row[0]).startswith("sqlite_")
        }
        assert actual_tables == set(metadata.tables)
        inspector = inspect(connection)
        for table in metadata.tables.values():
            expected_checks = {
                str(constraint.name)
                for constraint in table.constraints
                if isinstance(constraint, CheckConstraint)
            }
            actual_checks = {
                str(constraint["name"])
                for constraint in inspector.get_check_constraints(table.name)
            }
            assert actual_checks == expected_checks, table.name
        index_sql = connection.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type='index' AND name='uq_jobs_one_current'"
            )
        ).scalar_one()
        assert "WHERE status IN" in index_sql
    engine.dispose()
