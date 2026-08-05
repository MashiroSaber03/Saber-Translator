from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import json
from copy import deepcopy
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from flask import Flask
from fontTools.ttLib import TTCollection, TTFont
from sqlalchemy import insert, select, text, update

from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.builtin_fonts import discover_bundled_fonts
from src.backend_v2.storage.consistency import ConsistencyChecker
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import (
    DEFAULT_INSIGHT_SETTINGS,
    default_translation_settings,
)
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
    utcnow,
)
from src.backend_v2.storage.lifecycle import (
    UnsupportedDataRoot,
    initialize_database,
    schema_smoke_test,
)
from src.backend_v2.storage.platform_repositories import (
    BookSettingMutation,
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
    fonts,
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


def test_launcher_initialization_seeds_one_persistent_quick_workspace(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    first = initialize_database(data_root)
    assert first.schema_revision == "v2_foundation_20260801"
    assert first.created is True
    assert first.upgraded is False
    assert schema_smoke_test(first.database_path) == "v2_foundation_20260801"

    engine = create_sqlite_engine(first.database_path)
    with engine.connect() as connection:
        quick_books = connection.execute(
            select(books.c.id).where(books.c.kind == "quick_workspace")
        ).scalars().all()
        quick_chapters = connection.execute(
            select(chapters.c.id).where(chapters.c.book_id == quick_books[0])
        ).scalars().all()
        seeded_fonts = connection.execute(
            select(
                fonts.c.id,
                fonts.c.builtin_key,
                fonts.c.display_name,
            ).where(fonts.c.kind == "builtin")
        ).mappings().all()
    engine.dispose()
    assert quick_books == [QUICK_WORKSPACE_BOOK_ID]
    assert quick_chapters == [QUICK_WORKSPACE_CHAPTER_ID]
    default_font = next(
        font for font in discover_bundled_fonts() if font.builtin_key == "default"
    )
    assert default_font.file_name == "思源黑体SourceHanSansK-Bold.TTF"
    assert default_font.display_name == "思源黑体"
    assert {
        (str(row["id"]), str(row["builtin_key"]), str(row["display_name"]))
        for row in seeded_fonts
    } == {
        (font.id, font.builtin_key, font.display_name)
        for font in discover_bundled_fonts()
    }

    # Repair databases created by the first backend-first font catalog, which
    # exposed a synthetic label instead of the real default resource name.
    engine = create_sqlite_engine(first.database_path)
    with engine.begin() as connection:
        connection.execute(
            update(fonts)
            .where(fonts.c.builtin_key == "default")
            .values(display_name="默认字体")
        )
    engine.dispose()

    second = initialize_database(data_root)
    assert second.created is False
    assert second.upgraded is False
    assert not list((data_root / "runtime").glob("pre-upgrade-*.sqlite3"))

    engine = create_sqlite_engine(second.database_path)
    try:
        listed_fonts = FontRepository(engine).list()
        assert len(listed_fonts) == len(discover_bundled_fonts())
        assert listed_fonts[0]["builtinKey"] == "default"
        assert listed_fonts[0]["displayName"] == "思源黑体"
    finally:
        engine.dispose()


@pytest.mark.parametrize("retired_revision", [None, "0017"])
def test_storage_initialization_rejects_nonformal_database_without_rewriting_it(
    tmp_path: Path,
    retired_revision: str | None,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    database_path = data_root / "saber.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel VALUES ('untouched')")
        if retired_revision is not None:
            connection.execute(
                "CREATE TABLE alembic_version(version_num VARCHAR(32) NOT NULL)"
            )
            connection.execute(
                "INSERT INTO alembic_version VALUES (?)",
                (retired_revision,),
            )

    with pytest.raises(UnsupportedDataRoot, match="旧数据不会被读取或迁移"):
        initialize_database(data_root)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT value FROM sentinel").fetchall() == [
            ("untouched",)
        ]
    assert not list((data_root / "runtime").glob("pre-upgrade-*.sqlite3"))


def test_storage_initialization_rejects_extra_nonformal_tables(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    initialized = initialize_database(data_root)
    with sqlite3.connect(initialized.database_path) as connection:
        connection.execute("CREATE TABLE retired_payload(value TEXT NOT NULL)")

    with pytest.raises(RuntimeError, match="unexpected=.*retired_payload"):
        initialize_database(data_root)


def test_custom_insight_architecture_requires_at_least_two_layers() -> None:
    payload = deepcopy(DEFAULT_INSIGHT_SETTINGS)
    payload["analysis"]["batch"]["architecturePreset"] = "custom"
    payload["analysis"]["batch"]["customLayers"] = []

    with pytest.raises(ValueError, match="must contain 2-8 layers"):
        validate_setting_payload("insight", payload, schema_version=1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("auxYoloConfThreshold", 1.01, "must be from 0 to 1"),
        ("auxYoloOverlapThreshold", -0.01, "must be from 0 to 1"),
        (
            "saberYoloRefineOverlapThreshold",
            101,
            "must be from 0 to 100",
        ),
    ],
)
def test_translation_detection_thresholds_use_one_current_unit(
    field: str,
    value: float,
    message: str,
) -> None:
    payload = default_translation_settings()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_setting_payload("translation", payload, schema_version=3)


def test_settings_load_rejects_noncurrent_persisted_schema_versions(
    platform,
) -> None:
    _data_root, engine = platform
    repository = SettingsRepository(engine)
    repository.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=default_translation_settings(),
                base_revision=0,
                schema_version=3,
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={"modelName": "current-model"},
                base_revision=0,
                schema_version=1,
            ),
        ),
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE app_settings SET schema_version = 2 "
                "WHERE domain = 'translation'"
            )
        )
    with pytest.raises(ValueError, match="translation settings schema version"):
        repository.load(domains=("translation",))

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE app_settings SET schema_version = 3 "
                "WHERE domain = 'translation'"
            )
        )
        connection.execute(
            text(
                "UPDATE provider_settings SET schema_version = 2 "
                "WHERE domain = 'translation' AND provider = 'custom'"
            )
        )
    with pytest.raises(ValueError, match="provider setting schema version"):
        repository.load(domains=("translation",))

    book_id = "current-schema-book"
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id=book_id, kind="library", title="Book")
        )
        connection.execute(
            text(
                "UPDATE provider_settings SET schema_version = 1 "
                "WHERE domain = 'translation' AND provider = 'custom'"
            )
        )
    repository.save_transaction(
        book_settings_edits=(
            BookSettingMutation(
                book_id=book_id,
                domain="insight",
                payload=deepcopy(DEFAULT_INSIGHT_SETTINGS),
                base_revision=0,
                schema_version=1,
            ),
        ),
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE book_settings SET schema_version = 2 "
                "WHERE book_id = :book_id AND domain = 'insight'"
            ),
            {"book_id": book_id},
        )
    with pytest.raises(ValueError, match="book setting schema version"):
        repository.load(domains=("insight",), book_id=book_id)


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
                schema_version=1,
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
                schema_version=1,
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
                schema_version=1,
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
        rpm_limit=5,
    )
    assert first.allowed and not second.allowed and second.retry_after_seconds > 0


def test_settings_http_rejects_unknown_transaction_fields(platform) -> None:
    data_root, engine = platform
    app = Flask("settings-strict-contract-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()

    top_level = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "settings-extra-top-level"},
        json={
            "settings": [],
            "bookSettings": [],
            "providerSettings": [],
            "credentialEdits": [],
            "legacySettings": [],
        },
    )
    assert top_level.status_code == 422
    assert "legacySettings" in top_level.get_data(as_text=True)

    nested = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "settings-extra-nested"},
        json={
            "settings": [
                {
                    "domain": "proofreading",
                    "payload": {"enabled": True},
                    "baseRevision": 0,
                    "schemaVersion": 1,
                    "legacyPayload": {},
                }
            ],
            "bookSettings": [],
            "providerSettings": [],
            "credentialEdits": [],
        },
    )
    assert nested.status_code == 422
    assert "legacyPayload" in nested.get_data(as_text=True)


def test_settings_http_accepts_true_type_collections(platform) -> None:
    data_root, engine = platform
    source_path = next(
        font.path for font in discover_bundled_fonts() if font.file_name == "ALGER.TTF"
    )
    source_font = TTFont(source_path)
    collection = TTCollection()
    collection.fonts = [source_font]
    payload = BytesIO()
    collection.save(payload)
    source_font.close()

    app = Flask("settings-font-collection-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()
    response = client.post(
        "/api/v2/fonts",
        headers={"Idempotency-Key": "upload-font-collection"},
        data={"file": (BytesIO(payload.getvalue()), "custom.ttc")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    uploaded_id = response.get_json()["id"]
    listed = client.get("/api/v2/fonts").get_json()["items"]
    assert any(
        item["id"] == uploaded_id
        and item["kind"] == "uploaded"
        and item["displayName"] == "custom"
        for item in listed
    )


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
                schema_version=1,
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
                schema_version=1,
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
        "models": [{"id": "gpt-test", "name": "gpt-test"}],
    }
    assert captured == {"api_key": "stored-only-on-server"}

    with pytest.raises(ValueError, match="exactly: api_key"):
        diagnostics.model_catalog(
            {
                "provider": "openai",
                "secret": {"apiKey": "retired-field-name"},
            }
        )

    monkeypatch.setattr(
        ProviderDiagnostics,
        "model_catalog",
        lambda _self, body: {
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
                    "schemaVersion": 1,
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
