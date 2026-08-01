from __future__ import annotations

from datetime import timedelta
import json
from copy import deepcopy
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from alembic import command
from flask import Flask
from sqlalchemy import insert, select, text

from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.consistency import ConsistencyChecker
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import (
    DEFAULT_INSIGHT_SETTINGS,
    DEFAULT_WEB_IMPORT_SETTINGS,
    default_translation_settings,
)
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
    utcnow,
)
from src.backend_v2.storage.lifecycle import (
    migrate_database,
    schema_smoke_test,
    sqlite_backup,
)
import src.backend_v2.storage.lifecycle as lifecycle_module
from src.backend_v2.storage.platform_repositories import (
    CredentialEdit,
    FontRepository,
    ProviderRateLimiter,
    ProviderSettingMutation,
    RevisionConflict,
    SettingMutation,
    SettingsRepository,
)
from src.backend_v2.storage.schema import (
    assets,
    books,
    bubbles,
    chapter_write_intents,
    chapter_write_locks,
    chapters,
    credential_versions,
    credentials,
    jobs,
    metadata,
    object_commit_journal,
    operations,
    pages,
    process_epochs,
    render_requests,
)
from src.backend_v2.storage.seeding import (
    QUICK_WORKSPACE_BOOK_ID,
    QUICK_WORKSPACE_CHAPTER_ID,
    seed_system_records,
)
from src.backend_v2.storage.single_instance import (
    DataRootAlreadyLocked,
    DataRootLock,
)
from src.backend_v2.settings.validation import validate_setting_payload
from src.backend_v2.settings.diagnostics import ProviderDiagnostics
from src.backend_v2.settings.routes import create_settings_blueprint
from src.backend_v2.worker.maintenance import WorkerMaintenance


@pytest.fixture()
def platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    try:
        yield data_root, engine
    finally:
        engine.dispose()


def test_launcher_migration_seeds_one_persistent_quick_workspace(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    first = migrate_database(data_root)
    assert first.upgraded_to == "0017"
    assert not first.backup_created
    assert schema_smoke_test(first.database_path) == "0017"

    engine = create_sqlite_engine(first.database_path)
    with engine.connect() as connection:
        quick_books = connection.execute(
            select(books.c.id).where(books.c.kind == "quick_workspace")
        ).scalars().all()
        quick_chapters = connection.execute(
            select(chapters.c.id).where(chapters.c.book_id == quick_books[0])
        ).scalars().all()
    engine.dispose()
    assert quick_books == [QUICK_WORKSPACE_BOOK_ID]
    assert quick_chapters == [QUICK_WORKSPACE_CHAPTER_ID]

    second = migrate_database(data_root)
    assert second.backup_created
    assert not list((data_root / "runtime").glob("pre-migration-*.sqlite3"))


def test_0014_migrates_text_style_to_single_authoritative_facts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    config = lifecycle_module._alembic_config(database_path)
    command.upgrade(config, "0013")
    engine = create_sqlite_engine(database_path)
    seed_system_records(engine)
    with engine.begin() as connection:
        translation_payload = json.loads(
            connection.execute(
                text(
                    "SELECT payload_json FROM app_settings "
                    "WHERE domain = 'translation'"
                )
            ).scalar_one()
        )
        translation_payload["textStyle"] = {
            "fontFamily": "00000000-0000-0000-0000-000000000010",
            "fontSize": 99,
        }
        connection.execute(
            text(
                "UPDATE app_settings SET payload_json = :payload "
                "WHERE domain = 'translation'"
            ),
            {"payload": json.dumps(translation_payload)},
        )
        connection.execute(
            insert(books).values(id="legacy-book", kind="library", title="Legacy")
        )
        connection.execute(
            insert(chapters).values(
                id="legacy-chapter",
                book_id="legacy-book",
                ordinal=1,
                title="Legacy",
            )
        )
        connection.execute(
            insert(pages).values(
                id="legacy-page",
                chapter_id="legacy-chapter",
                ordinal=1,
                logical_source_path="legacy.png",
                default_font_id=None,
                page_style_defaults_json=json.dumps(
                    {
                        "fontFamily": "00000000-0000-0000-0000-000000000010",
                        "fontSize": 42,
                        "useAutoTextColor": False,
                    }
                ),
            )
        )
    engine.dispose()

    command.upgrade(config, "0014")
    with sqlite3.connect(database_path) as connection:
        translation = json.loads(
            connection.execute(
                "SELECT payload_json FROM app_settings "
                "WHERE domain = 'translation'"
            ).fetchone()[0]
        )
        default_font_id, page_style_json = connection.execute(
            "SELECT default_font_id, page_style_defaults_json "
            "FROM pages WHERE id = 'legacy-page'"
        ).fetchone()
    page_style = json.loads(page_style_json)
    assert "textStyle" not in translation
    assert default_font_id == "00000000-0000-0000-0000-000000000010"
    assert "fontFamily" not in page_style
    assert page_style["fontSize"] == 42
    assert set(page_style) == {
        "fontSize",
        "autoFontSize",
        "layoutDirection",
        "textColor",
        "fillColor",
        "inpaintMethod",
        "useAutoTextColor",
        "strokeEnabled",
        "strokeColor",
        "strokeWidth",
        "lineSpacing",
        "textAlign",
    }


def test_0015_separates_legacy_ai_review_from_structural_diagnostics(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-review.sqlite3"
    config = lifecycle_module._alembic_config(database_path)
    command.upgrade(config, "0014")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO books (id, kind, title) VALUES (?, ?, ?)",
            ("review-book", "library", "Review"),
        )
        connection.execute(
            "INSERT INTO studio_documents ("
            "id, book_id, origin_type, title, last_diagnostics_json, "
            "last_validated_at"
            ") VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (
                "review-document",
                "review-book",
                "manual",
                "Review",
                json.dumps(
                    {
                        "identity": {"name": "Review"},
                        "review": {
                            "notes": "审查完成",
                            "status": "approved",
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    command.upgrade(config, "0015")
    with sqlite3.connect(database_path) as connection:
        review, diagnostics, validated_at = connection.execute(
            "SELECT last_review_json, last_diagnostics_json, "
            "last_validated_at FROM studio_documents "
            "WHERE id = 'review-document'"
        ).fetchone()
    assert json.loads(review) == {
        "summary": "审查完成",
        "issues": [],
        "suggestions": [],
    }
    assert diagnostics is None
    assert validated_at is None


def test_0016_replaces_empty_app_settings_with_complete_backend_defaults(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "empty-app-settings.sqlite3"
    config = lifecycle_module._alembic_config(database_path)
    command.upgrade(config, "0015")
    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO app_settings "
            "(domain, revision, payload_json, schema_version) "
            "VALUES (?, 1, '{}', 1)",
            (("web_import",), ("insight",)),
        )

    command.upgrade(config, "0016")
    with sqlite3.connect(database_path) as connection:
        rows = dict(
            connection.execute(
                "SELECT domain, payload_json FROM app_settings "
                "WHERE domain IN ('web_import', 'insight')"
            )
        )
    assert json.loads(rows["web_import"]) == DEFAULT_WEB_IMPORT_SETTINGS
    assert json.loads(rows["insight"]) == DEFAULT_INSIGHT_SETTINGS


def test_0016_normalizes_saved_browser_settings_without_losing_user_values(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "saved-app-settings.sqlite3"
    config = lifecycle_module._alembic_config(database_path)
    command.upgrade(config, "0015")
    legacy_web_import = deepcopy(DEFAULT_WEB_IMPORT_SETTINGS)
    legacy_web_import["download"]["concurrency"] = 7
    legacy_web_import["firecrawl"]["apiKey"] = "browser-only-secret"
    legacy_web_import["agent"]["apiKey"] = "browser-only-secret"
    legacy_web_import["advanced"]["customCookie"] = "browser-only-secret"
    legacy_web_import["advanced"]["customHeaders"] = "browser-only-secret"
    partial_insight = {
        "analysis": {
            "batch": {
                "pagesPerBatch": 9,
            }
        },
        "vlm": {"provider": "openai"},
    }
    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO app_settings "
            "(domain, revision, payload_json, schema_version) "
            "VALUES (?, 1, ?, 1)",
            (
                ("web_import", json.dumps(legacy_web_import)),
                ("insight", json.dumps(partial_insight)),
            ),
        )

    command.upgrade(config, "0016")
    with sqlite3.connect(database_path) as connection:
        rows = dict(
            connection.execute(
                "SELECT domain, payload_json FROM app_settings "
                "WHERE domain IN ('web_import', 'insight')"
            )
        )
    web_import = json.loads(rows["web_import"])
    insight = json.loads(rows["insight"])
    assert web_import["download"]["concurrency"] == 7
    assert web_import["firecrawl"] == {}
    assert "apiKey" not in web_import["agent"]
    assert web_import["advanced"] == {"bypassProxy": False}
    assert insight["analysis"]["batch"]["pagesPerBatch"] == 9
    assert insight["analysis"]["batch"]["contextBatchCount"] == 3
    assert insight["vlm"]["provider"] == "openai"
    assert insight["chat"] == {"provider": "gemini", "useSameAsVlm": False}
    assert validate_setting_payload(
        "web_import",
        web_import,
        schema_version=1,
    ) == web_import
    assert validate_setting_payload(
        "insight",
        insight,
        schema_version=1,
    ) == insight


def test_custom_insight_architecture_requires_at_least_two_layers() -> None:
    payload = deepcopy(DEFAULT_INSIGHT_SETTINGS)
    payload["analysis"]["batch"]["architecturePreset"] = "custom"
    payload["analysis"]["batch"]["customLayers"] = []

    with pytest.raises(ValueError, match="must contain 2-8 layers"):
        validate_setting_payload("insight", payload, schema_version=1)


def test_0017_restores_only_accidentally_materialized_automatic_colors(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "automatic-color-repair.sqlite3"
    config = lifecycle_module._alembic_config(database_path)
    command.upgrade(config, "0016")
    page_style = {
        "fontSize": 26,
        "autoFontSize": True,
        "layoutDirection": "auto",
        "textColor": "#123456",
        "fillColor": "#ABCDEF",
        "inpaintMethod": "solid",
        "useAutoTextColor": False,
        "strokeEnabled": True,
        "strokeColor": "#FFFFFF",
        "strokeWidth": 3,
        "lineSpacing": 1.0,
        "textAlign": "start",
    }
    accidental = {
        "textColor": "#0A141E",
        "fillColor": "#F5F6F7",
        "autoFgColor": [10, 20, 30],
        "autoBgColor": [245, 246, 247],
    }
    manual_override = {
        "textColor": "#654321",
        "fillColor": "#FEDCBA",
        "autoFgColor": [10, 20, 30],
        "autoBgColor": [245, 246, 247],
    }
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO books (id, kind, title) VALUES (?, ?, ?)",
            ("color-book", "library", "Color"),
        )
        connection.execute(
            "INSERT INTO chapters (id, book_id, ordinal, title) "
            "VALUES (?, ?, ?, ?)",
            ("color-chapter", "color-book", 1, "Color"),
        )
        connection.execute(
            "INSERT INTO pages ("
            "id, chapter_id, ordinal, logical_source_path, "
            "document_revision, rendered_revision, render_status, "
            "page_style_defaults_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "color-page",
                "color-chapter",
                1,
                "color.png",
                4,
                4,
                "ready",
                json.dumps(page_style),
            ),
        )
        connection.execute(
            "INSERT INTO assets ("
            "id, relative_path, mime_type, checksum, byte_size"
            ") VALUES (?, ?, ?, ?, ?)",
            ("translated-asset", "objects/color.png", "image/png", "a" * 64, 1),
        )
        connection.execute(
            "INSERT INTO page_assets (page_id, role, asset_id) "
            "VALUES (?, 'translated', ?)",
            ("color-page", "translated-asset"),
        )
        connection.executemany(
            "INSERT INTO bubbles ("
            "id, page_id, ordinal, payload_json, updated_revision"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                (
                    "accidental-bubble",
                    "color-page",
                    1,
                    json.dumps(accidental),
                    4,
                ),
                (
                    "manual-bubble",
                    "color-page",
                    2,
                    json.dumps(manual_override),
                    4,
                ),
            ),
        )

    command.upgrade(config, "0017")
    with sqlite3.connect(database_path) as connection:
        rows = dict(
            connection.execute(
                "SELECT id, payload_json FROM bubbles "
                "WHERE page_id = 'color-page'"
            )
        )
        page = connection.execute(
            "SELECT document_revision, rendered_revision, render_status "
            "FROM pages WHERE id = 'color-page'"
        ).fetchone()
        render_request = connection.execute(
            "SELECT requested_revision, status FROM render_requests "
            "WHERE page_id = 'color-page'"
        ).fetchone()
    repaired = json.loads(rows["accidental-bubble"])
    preserved = json.loads(rows["manual-bubble"])
    assert repaired["textColor"] == "#123456"
    assert repaired["fillColor"] == "#ABCDEF"
    assert repaired["autoFgColor"] == [10, 20, 30]
    assert repaired["autoBgColor"] == [245, 246, 247]
    assert preserved["textColor"] == "#654321"
    assert preserved["fillColor"] == "#FEDCBA"
    assert page == (5, 4, "stale")
    assert render_request == (5, "pending")


def test_sqlite_backup_api_captures_committed_wal_pages(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    backup = tmp_path / "backup.sqlite3"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE facts(value TEXT NOT NULL)")
    writer.commit()
    writer.execute("INSERT INTO facts VALUES ('committed-in-wal')")
    writer.commit()

    sqlite_backup(source, backup)
    with sqlite3.connect(backup) as copied:
        assert copied.execute("SELECT value FROM facts").fetchall() == [
            ("committed-in-wal",)
        ]
    writer.close()


def test_migration_failure_restores_database_and_keeps_failure_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    migrated = migrate_database(data_root)
    with sqlite3.connect(migrated.database_path) as connection:
        connection.execute("CREATE TABLE rollback_probe(value TEXT NOT NULL)")
        connection.execute("INSERT INTO rollback_probe VALUES ('before-upgrade')")
        connection.commit()

    def fail_upgrade(_config, _revision) -> None:
        with sqlite3.connect(migrated.database_path) as connection:
            connection.execute("DROP TABLE rollback_probe")
            connection.commit()
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(lifecycle_module.command, "upgrade", fail_upgrade)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        migrate_database(data_root)

    with sqlite3.connect(migrated.database_path) as connection:
        assert connection.execute("SELECT value FROM rollback_probe").fetchall() == [
            ("before-upgrade",)
        ]
    assert list((data_root / "runtime").glob("pre-migration-*.sqlite3"))


def test_second_launcher_is_rejected_for_the_same_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    first = DataRootLock(data_root)
    second = DataRootLock(data_root)
    first.acquire()
    try:
        with pytest.raises(DataRootAlreadyLocked):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_worker_recovery_is_idempotent_and_preserves_chapter_lock(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration("worker-epoch", "worker-token", "worker", 123)
    repository.register(registration)
    now = utcnow()
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="book", kind="library", title="Book")
        )
        connection.execute(
            insert(chapters).values(
                id="chapter",
                book_id="book",
                ordinal=1,
                title="Chapter",
            )
        )
        connection.execute(
            insert(jobs).values(
                id="job",
                kind="translation",
                status="running",
                chapter_id="chapter",
                config_json="{}",
                worker_epoch_id=registration.epoch_id,
                attempt_id="attempt",
                lease_token="attempt-token",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(chapter_write_locks).values(
                chapter_id="chapter",
                job_id="job",
                lock_generation=1,
                owner_attempt_id="attempt",
                lease_token="attempt-token",
            )
        )
        connection.execute(
            insert(chapter_write_intents).values(
                chapter_id="chapter",
                job_id="job",
                intent_set_id="intent-set",
                intent_generation=1,
                worker_epoch_id=registration.epoch_id,
                lease_token="worker-token",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )

    first = repository.reconcile_dead_worker(registration.epoch_id)
    second = repository.reconcile_dead_worker(registration.epoch_id)
    assert first.changed and first.jobs_interrupted == 1 and first.intents_removed == 1
    assert not second.changed
    with engine.connect() as connection:
        job = connection.execute(
            select(jobs.c.status, jobs.c.attempt_id).where(jobs.c.id == "job")
        ).one()
        lock_count = connection.execute(
            select(text("COUNT(*)")).select_from(chapter_write_locks)
        ).scalar_one()
    assert job == ("interrupted", None)
    assert lock_count == 1


@pytest.mark.parametrize(
    "initial_status,expected_status,lock_is_retained",
    [
        ("pausing", "interrupted", True),
        ("cancelling", "cancelled", False),
    ],
)
def test_worker_recovery_resolves_drain_transition_states(
    platform,
    initial_status: str,
    expected_status: str,
    lock_is_retained: bool,
) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration(
        f"worker-{initial_status}",
        f"token-{initial_status}",
        "worker",
        456,
    )
    repository.register(registration)
    now = utcnow()
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="book", kind="library", title="Book")
        )
        connection.execute(
            insert(chapters).values(
                id="chapter",
                book_id="book",
                ordinal=1,
                title="Chapter",
            )
        )
        connection.execute(
            insert(jobs).values(
                id="job",
                kind="translation",
                status=initial_status,
                chapter_id="chapter",
                config_json="{}",
                worker_epoch_id=registration.epoch_id,
                attempt_id="attempt",
                lease_token="attempt-token",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(chapter_write_locks).values(
                chapter_id="chapter",
                job_id="job",
                lock_generation=1,
                owner_attempt_id="attempt",
                lease_token="attempt-token",
            )
        )

    result = repository.reconcile_dead_worker(registration.epoch_id)
    with engine.connect() as connection:
        status = connection.execute(
            select(jobs.c.status).where(jobs.c.id == "job")
        ).scalar_one()
        lock = connection.execute(
            select(chapter_write_locks.c.job_id)
        ).scalar_one_or_none()
    assert status == expected_status
    assert (lock is not None) is lock_is_retained
    assert result.jobs_interrupted == int(expected_status == "interrupted")
    assert result.jobs_cancelled == int(expected_status == "cancelled")


def test_api_recovery_fails_remote_work_and_requeues_safe_render(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration("api-epoch", "api-token", "api", 321)
    repository.register(registration)
    now = utcnow()
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="book", kind="library", title="Book")
        )
        connection.execute(
            insert(chapters).values(
                id="chapter", book_id="book", ordinal=1, title="Chapter"
            )
        )
        connection.execute(
            insert(pages).values(
                id="page",
                chapter_id="chapter",
                ordinal=1,
                logical_source_path="page.png",
            )
        )
        connection.execute(
            insert(bubbles).values(
                id="bubble",
                page_id="page",
                ordinal=1,
                payload_json="{}",
                updated_revision=1,
            )
        )
        connection.execute(
            insert(operations).values(
                id="operation",
                kind="bubble_translate",
                executor_role="api",
                status="running",
                page_id="page",
                bubble_id="bubble",
                base_revision=1,
                request_json="{}",
                executor_epoch_id=registration.epoch_id,
                attempt_id="attempt",
                lease_token="lease",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(render_requests).values(
                id="render",
                page_id="page",
                requested_revision=1,
                rendering_revision=1,
                status="running",
                executor_epoch_id=registration.epoch_id,
                attempt_id="render-attempt",
                lease_token="render-lease",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )

    result = repository.reconcile_dead_api(registration.epoch_id)
    assert result.operations_failed == 1
    assert result.renders_requeued == 1
    with engine.connect() as connection:
        operation = connection.execute(
            select(operations.c.status, operations.c.error_json).where(
                operations.c.id == "operation"
            )
        ).one()
        render = connection.execute(
            select(render_requests.c.status, render_requests.c.rendering_revision).where(
                render_requests.c.id == "render"
            )
        ).one()
    assert operation.status == "failed"
    assert json.loads(operation.error_json)["code"] == "API_EXECUTOR_LOST"
    assert render == ("pending", None)


def test_expired_or_replaced_epoch_cannot_be_renewed(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine, lease_seconds=3)
    registration = EpochRegistration("worker", "secret", "worker", 123)
    repository.register(registration)
    assert repository.renew(role="worker", epoch_id="worker", token="secret")
    with engine.begin() as connection:
        connection.execute(
            process_epochs.update()
            .where(process_epochs.c.id == "worker")
            .values(status="lost")
        )
    assert not repository.renew(role="worker", epoch_id="worker", token="secret")
    assert not repository.renew(role="worker", epoch_id="worker", token="wrong")


@pytest.mark.parametrize(
    "crash_point,committed",
    [
        ("staging_fsynced", False),
        ("journal_staged", False),
        ("file_published", False),
        ("journal_file_published", False),
        ("database_before_commit", False),
        ("database_committed", True),
    ],
)
def test_asset_publication_failure_windows_are_recoverable(
    platform,
    crash_point: str,
    committed: bool,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)

    def crash(point: str) -> None:
        if point == crash_point:
            raise RuntimeError(f"injected crash at {point}")

    with pytest.raises(RuntimeError, match=crash_point):
        storage.publish_bytes(
            f"payload-{crash_point}".encode(),
            extension="bin",
            mime_type="application/octet-stream",
            failpoint=crash,
        )

    storage.recover_journal(orphan_grace_seconds=0)
    with engine.connect() as connection:
        stored_assets = list(
            connection.execute(select(assets)).mappings()
        )
        assert connection.execute(select(object_commit_journal)).all() == []
    assert len(stored_assets) == int(committed)
    assert not list((data_root / "temp" / "staging").glob("*.part"))
    object_files = [
        path
        for path in (data_root / "objects").rglob("*")
        if path.is_file()
    ]
    assert len(object_files) == int(committed)
    if committed:
        stored_path = storage.resolve_relative_path(
            str(stored_assets[0]["relative_path"])
        )
        assert stored_path.read_bytes() == (
            f"payload-{crash_point}".encode()
        )


def test_integrity_scan_and_two_pass_gc_never_delete_referenced_assets(
    platform,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    unreferenced = storage.publish_bytes(
        b"unused", extension="bin", mime_type="application/octet-stream"
    )
    referenced = storage.publish_bytes(
        b"font", extension="ttf", mime_type="font/ttf"
    )
    FontRepository(engine).register_uploaded(
        asset_id=referenced.id,
        display_name="Uploaded",
    )
    storage.resolve_relative_path(referenced.relative_path).unlink()
    scan = storage.scan_integrity()
    assert scan.missing == 1

    first = storage.collect_garbage(grace_seconds=10, now=utcnow())
    second = storage.collect_garbage(
        grace_seconds=10,
        now=utcnow() + timedelta(seconds=11),
    )
    assert first.marked == 1
    assert second.deleted_rows == 1
    with engine.connect() as connection:
        remaining = set(connection.execute(select(assets.c.id)).scalars())
    assert unreferenced.id not in remaining
    assert referenced.id in remaining


def test_orphan_object_reconciliation_honors_database_journal_and_grace(
    platform,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    referenced = storage.publish_bytes(
        b"referenced",
        extension="bin",
        mime_type="application/octet-stream",
    )
    old_orphan = data_root / "objects" / "orphan.bin"
    young_orphan = data_root / "objects" / "young.bin"
    old_orphan.write_bytes(b"old")
    young_orphan.write_bytes(b"young")
    old_timestamp = (utcnow() - timedelta(hours=2)).timestamp()
    os.utime(old_orphan, (old_timestamp, old_timestamp))

    result = storage.reconcile_orphan_objects(grace_seconds=3600)

    assert result.scanned == 3
    assert result.deleted == 1
    assert result.protected == 1
    assert result.grace_retained == 1
    assert not old_orphan.exists()
    assert young_orphan.exists()
    assert storage.resolve_relative_path(referenced.relative_path).exists()


def test_consistency_checker_and_cli_report_storage_divergence(
    platform,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    missing = storage.publish_bytes(
        b"missing",
        extension="bin",
        mime_type="application/octet-stream",
    )
    storage.resolve_relative_path(missing.relative_path).unlink()
    orphan = data_root / "objects" / "orphan.bin"
    orphan.write_bytes(b"orphan")

    report = ConsistencyChecker(
        data_root=data_root,
        engine=engine,
    ).check(include_vectors=False)
    assert report.ok is False
    assert report.missing_asset_files == (missing.id,)
    assert report.integrity_status_mismatches == (missing.id,)
    assert report.orphan_object_files == ("objects/orphan.bin",)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_v2_consistency",
            "--data-dir",
            str(data_root),
            "--skip-vectors",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["missing_asset_files"] == [missing.id]


def test_worker_maintenance_runs_only_when_due(platform) -> None:
    data_root, engine = platform
    current = [100.0]
    maintenance = WorkerMaintenance(
        data_root=data_root,
        engine=engine,
        interval_seconds=60,
        clock=lambda: current[0],
    )

    assert maintenance.run_if_due(force=True) is True
    assert maintenance.run_if_due() is False
    current[0] += 60
    assert maintenance.run_if_due() is True


def test_worker_maintenance_continues_after_failed_action(
    platform,
    monkeypatch,
) -> None:
    data_root, engine = platform
    maintenance = WorkerMaintenance(
        data_root=data_root,
        engine=engine,
        interval_seconds=60,
    )
    completed: list[str] = []

    def fail_recovery():
        raise RuntimeError("broken journal fixture")

    monkeypatch.setattr(maintenance.storage, "recover_journal", fail_recovery)
    monkeypatch.setattr(
        maintenance.vector_store,
        "collect_orphan_collections",
        lambda _engine: completed.append("vector_gc"),
    )

    assert maintenance.run_if_due(force=True) is True
    assert completed == ["vector_gc"]


def test_settings_credentials_plugins_fonts_and_shared_limiter(platform) -> None:
    _data_root, engine = platform
    settings = SettingsRepository(engine)
    translation_payload = default_translation_settings()
    translation_payload["translation"]["provider"] = "custom"
    result = settings.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=translation_payload,
                base_revision=0,
                schema_version=3,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "never-return-me"},
                base_revision=0,
                client_ref="translation-fake",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={"modelName": "fake-model"},
                base_revision=0,
                credential_edit_ref="translation-fake",
            ),
        ),
    )
    credential_summary = result["credentials"][0]
    assert credential_summary["hasKey"] is True
    assert "never-return-me" not in json.dumps(result)
    assert "secret" not in json.dumps(settings.credential_summaries()).lower()

    credential_id = str(credential_summary["credentialId"])
    with engine.connect() as connection:
        version_id = connection.execute(
            select(credential_versions.c.id).where(
                credential_versions.c.credential_id == credential_id
            )
        ).scalar_one()
    assert settings.resolve_secret(version_id) == {"api_key": "never-return-me"}
    assert settings.resolve_current_secret(credential_id) == {
        "api_key": "never-return-me"
    }
    assert settings.resolve_provider_secret(
        domain="translation",
        provider="custom",
    ) == {"api_key": "never-return-me"}
    loaded = settings.load(domains=("translation",))
    assert loaded["providerSettings"] == [
        {
            "domain": "translation",
            "provider": "custom",
            "revision": 1,
            "schemaVersion": 1,
            "credentialVersionId": version_id,
            "payload": {"modelName": "fake-model"},
        }
    ]

    idempotent_body = {
        "settings": [
            {
                "domain": "proofreading",
                "payload": {"enabled": True},
                "baseRevision": 0,
            }
        ]
    }
    first, first_replayed = settings.save_transaction_idempotent(
        idempotency_key="settings-save-1",
        request_body=idempotent_body,
        settings=(
            SettingMutation(
                domain="proofreading",
                payload={"enabled": True},
                base_revision=0,
            ),
        ),
    )
    second, second_replayed = settings.save_transaction_idempotent(
        idempotency_key="settings-save-1",
        request_body=idempotent_body,
        settings=(
            SettingMutation(
                domain="proofreading",
                payload={"enabled": True},
                base_revision=0,
            ),
        ),
    )
    assert first == second
    assert first_replayed is False
    assert second_replayed is True

    with pytest.raises(RevisionConflict):
        settings.save_transaction(
            settings=(
                SettingMutation(
                    domain="translation",
                    payload=translation_payload,
                    base_revision=0,
                    schema_version=3,
                ),
            ),
            credentials_edits=(
                CredentialEdit(
                    domain="hq",
                    provider="custom",
                    secret={"api_key": "must-rollback"},
                    base_revision=0,
                ),
            ),
        )
    with engine.connect() as connection:
        assert connection.execute(select(credentials.c.id)).scalars().all() == [
            credential_id
        ]

    limiter = ProviderRateLimiter(engine)
    first = limiter.acquire(
        provider="fake",
        credential_version_id=version_id,
        rpm_limit=1,
    )
    second = limiter.acquire(
        provider="fake",
        credential_version_id=version_id,
        rpm_limit=1,
    )
    assert first.allowed and not second.allowed and second.retry_after_seconds > 0


def test_insight_provider_accepts_its_snake_case_openai_wire_contract(
    platform,
) -> None:
    _data_root, engine = platform
    settings = SettingsRepository(engine)
    settings.save_transaction(
        providers=(
            ProviderSettingMutation(
                domain="insight_vlm",
                provider="siliconflow",
                payload={
                    "modelName": "Qwen/Qwen3.6-27B",
                    "customBaseUrl": "",
                    "imageMaxSize": 1280,
                    "openaiOptions": {
                        "request": {
                            "force_json_output": False,
                            "temperature": 0.3,
                        },
                        "execution": {
                            "use_stream": True,
                            "rpm_limit": 0,
                            "transport_retries": 10,
                            "business_retries": 10,
                        },
                    },
                },
                base_revision=0,
            ),
        ),
    )

    loaded = settings.load(domains=("insight_vlm",))
    assert loaded["providerSettings"][0]["payload"]["openaiOptions"][
        "request"
    ]["force_json_output"] is False


def test_v2_provider_diagnostics_resolve_backend_credentials_and_routes(
    platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, engine = platform
    settings = SettingsRepository(engine)
    settings.save_transaction(
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "stored-only-on-server"},
                base_revision=0,
                client_ref="openai-key",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "gpt-test",
                    "customBaseUrl": "https://example.test/v1",
                },
                base_revision=0,
                credential_edit_ref="openai-key",
            ),
        ),
    )
    diagnostics = ProviderDiagnostics(settings)
    captured: dict[str, object] = {}

    def list_models(request):
        captured["api_key"] = request.api_key
        return [{"id": "gpt-test", "name": "gpt-test"}]

    monkeypatch.setattr(diagnostics.chat, "list_models", list_models)
    assert diagnostics.model_catalog(
        {
            "provider": "custom",
            "baseUrl": "https://example.test/v1",
            "domain": "translation",
        }
    ) == {
        "success": True,
        "models": [{"id": "gpt-test", "name": "gpt-test"}],
    }
    assert captured == {"api_key": "stored-only-on-server"}

    monkeypatch.setattr(
        ProviderDiagnostics,
        "model_catalog",
        lambda _self, body: {
            "success": True,
            "models": [{"id": str(body["provider"]), "name": "model"}],
        },
    )
    monkeypatch.setattr(
        ProviderDiagnostics,
        "connection_test",
        lambda _self, kind, _body: {
            "success": True,
            "message": f"{kind} ok",
        },
    )
    app = Flask("settings-diagnostics-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()

    catalog = client.post(
        "/api/v2/model-catalog",
        json={"provider": "openai", "domain": "translation"},
    )
    assert catalog.status_code == 200
    assert catalog.get_json()["models"][0]["id"] == "openai"
    tested = client.post(
        "/api/v2/connection-tests/llm",
        json={"provider": "openai", "domain": "translation"},
    )
    assert tested.status_code == 200
    assert tested.get_json() == {"success": True, "message": "llm ok"}
    unsupported = client.post(
        "/api/v2/connection-tests/not-real",
        json={},
    )
    assert unsupported.status_code == 422


def test_settings_http_transaction_persists_secret_without_returning_it(
    platform,
) -> None:
    data_root, engine = platform
    app = Flask("settings-credential-persistence-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()
    secret = "sk-must-never-return-to-browser"

    saved = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "save-translation-credential"},
        json={
            "settings": [],
            "bookSettings": [],
            "providerSettings": [{
                "domain": "translation",
                "provider": "deepseek",
                "payload": {"modelName": "deepseek-chat"},
                "baseRevision": 0,
                "credentialEditRef": "translation-deepseek",
            }],
            "credentialEdits": [{
                "domain": "translation",
                "provider": "deepseek",
                "secret": {"api_key": secret},
                "baseRevision": 0,
                "clientRef": "translation-deepseek",
            }],
        },
    )
    assert saved.status_code == 200
    assert secret not in saved.get_data(as_text=True)

    loaded = client.get("/api/v2/settings?domains=translation")
    assert loaded.status_code == 200
    document = loaded.get_json()
    assert secret not in loaded.get_data(as_text=True)
    assert document["credentials"] == [
        {
            "credentialId": document["credentials"][0]["credentialId"],
            "credentialVersionId": document["credentials"][0][
                "credentialVersionId"
            ],
            "currentVersion": 1,
            "domain": "translation",
            "hasKey": True,
            "provider": "deepseek",
            "revision": 1,
        }
    ]
    assert document["providerSettings"][0]["credentialVersionId"] == (
        document["credentials"][0]["credentialVersionId"]
    )
    assert SettingsRepository(engine).resolve_provider_secret(
        domain="translation",
        provider="deepseek",
    ) == {"api_key": secret}
