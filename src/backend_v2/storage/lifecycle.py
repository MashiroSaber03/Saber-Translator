"""Launcher-owned initialization and integrity checks for the v2 data root."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from sqlalchemy import insert, select, text

from src.backend_v2.storage.database import (
    create_sqlite_engine,
    database_path_for,
)
from src.backend_v2.storage.schema import (
    DEFAULT_SCHEDULING_POLICY_JSON,
    metadata,
    schema_metadata,
)
from src.backend_v2.serialization import canonical_json
from src.backend_v2.storage.seeding import seed_system_records


SCHEMA_REVISION = "public_scheduler_direct_20260822_r2"
DIRECT_UPGRADE_REVISION = "public_user_policy_direct_20260822_r1"
REQUIRED_TABLES = frozenset(metadata.tables)


class UnsupportedDataRoot(RuntimeError):
    """The database is not a revision owned by the current formal schema."""


@dataclass(frozen=True, slots=True)
class StorageInitializationResult:
    database_path: Path
    schema_revision: str
    created: bool


def _database_revision(database_path: Path) -> str | None:
    try:
        with sqlite3.connect(database_path) as connection:
            has_version_table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'schema_metadata'"
            ).fetchone()
            if has_version_table is not None:
                rows = connection.execute(
                    "SELECT revision FROM schema_metadata WHERE singleton_id = 1"
                ).fetchall()
            else:
                return None
    except sqlite3.DatabaseError as exc:
        raise UnsupportedDataRoot(
            "data-v2/saber.sqlite3 不是当前架构的有效 SQLite 数据库"
        ) from exc
    if len(rows) != 1 or not isinstance(rows[0][0], str):
        return None
    return rows[0][0]


def schema_smoke_test(database_path: Path) -> str:
    engine = create_sqlite_engine(database_path)
    try:
        with engine.connect() as connection:
            integrity = connection.execute(text("PRAGMA integrity_check")).scalar_one()
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
            foreign_key_errors = connection.execute(text("PRAGMA foreign_key_check")).all()
            if foreign_key_errors:
                raise RuntimeError(f"SQLite foreign_key_check failed: {foreign_key_errors!r}")
            tables = {
                str(row[0])
                for row in connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                if not str(row[0]).startswith("sqlite_")
            }
            missing = REQUIRED_TABLES - tables
            unexpected = tables - REQUIRED_TABLES
            if missing or unexpected:
                raise RuntimeError(
                    "v2 schema table mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            revision = connection.execute(
                select(schema_metadata.c.revision).where(
                    schema_metadata.c.singleton_id == 1
                )
            ).scalar_one()
            return str(revision)
    finally:
        engine.dispose()


def _upgrade_previous_public_schema(database_path: Path) -> None:
    """Add scheduling policy and release paused jobs from the compute slot."""

    quoted_default = DEFAULT_SCHEDULING_POLICY_JSON.replace("'", "''")
    with sqlite3.connect(database_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(platform_config)")
        }
        if "public_user_policy_json" not in columns or "scheduler_policy_json" in columns:
            raise UnsupportedDataRoot(
                "旧公开版数据库的策略字段状态异常，未自动改写"
            )
        connection.execute(
            "ALTER TABLE platform_config ADD COLUMN "
            "scheduler_policy_json TEXT NOT NULL "
            f"DEFAULT '{quoted_default}'"
        )
        rows = connection.execute(
            "SELECT singleton_id, public_user_policy_json FROM platform_config"
        ).fetchall()
        for singleton_id, payload in rows:
            try:
                policy = json.loads(str(payload))
                parallel = policy["settings"]["parallel"]
                if not isinstance(parallel, dict):
                    raise TypeError
                parallel.pop("maxDeepLearningConcurrency", None)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise UnsupportedDataRoot(
                    "旧公开版数据库的普通用户策略无效，未自动改写"
                ) from exc
            connection.execute(
                "UPDATE platform_config SET public_user_policy_json = ? "
                "WHERE singleton_id = ?",
                (canonical_json(policy), singleton_id),
            )
        connection.execute("DROP INDEX uq_jobs_one_current")
        connection.execute(
            "CREATE UNIQUE INDEX uq_jobs_one_current ON jobs ((1)) "
            "WHERE status IN ('running','pausing','cancelling')"
        )
        updated = connection.execute(
            "UPDATE schema_metadata SET revision = ? "
            "WHERE singleton_id = 1 AND revision = ?",
            (SCHEMA_REVISION, DIRECT_UPGRADE_REVISION),
        )
        if updated.rowcount != 1:
            raise UnsupportedDataRoot("旧公开版数据库版本在升级时发生变化")


def initialize_database(data_root: Path) -> StorageInitializationResult:
    """Create the formal schema or validate an exact-current database.

    The immediately preceding public schema receives the small additive
    scheduling upgrade. Every other non-current database remains unsupported.
    """

    data_root.mkdir(parents=True, exist_ok=True)
    database_path = database_path_for(data_root)
    created = not database_path.exists() or database_path.stat().st_size == 0
    current_revision = None if created else _database_revision(database_path)
    if not created and current_revision == DIRECT_UPGRADE_REVISION:
        _upgrade_previous_public_schema(database_path)
        current_revision = SCHEMA_REVISION
    if not created and current_revision != SCHEMA_REVISION:
        raise UnsupportedDataRoot(
            "data-v2 不属于当前正式存储架构；旧数据不会被读取或迁移，"
            "请清空 data-v2 后重新启动"
        )

    if created:
        engine = create_sqlite_engine(database_path)
        try:
            metadata.create_all(engine)
            with engine.begin() as connection:
                connection.execute(
                    insert(schema_metadata).values(
                        singleton_id=1,
                        revision=SCHEMA_REVISION,
                    )
                )
        finally:
            engine.dispose()
    revision = schema_smoke_test(database_path)
    if revision != SCHEMA_REVISION:
        raise RuntimeError(
            f"database revision {revision!r} does not match "
            f"current revision {SCHEMA_REVISION!r}"
        )
    engine = create_sqlite_engine(database_path)
    try:
        seed_system_records(engine)
    finally:
        engine.dispose()
    return StorageInitializationResult(
        database_path=database_path,
        schema_revision=revision,
        created=created,
    )
