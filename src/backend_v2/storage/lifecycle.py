"""Launcher-owned initialization and integrity checks for the v2 data root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from sqlalchemy import insert, select, text

from src.backend_v2.storage.database import (
    create_sqlite_engine,
    database_path_for,
)
from src.backend_v2.runtime_profile import PROFILE_NAMES
from src.backend_v2.storage.schema import metadata, schema_metadata
from src.backend_v2.storage.seeding import seed_system_records


SCHEMA_REVISION = "backend_v2_browser_extension_sessions_20260830"
REQUIRED_TABLES = frozenset(metadata.tables)


class UnsupportedDataRoot(RuntimeError):
    """The database is not a revision owned by the current formal schema."""


@dataclass(frozen=True, slots=True)
class StorageInitializationResult:
    database_path: Path
    schema_revision: str
    created: bool


def _database_identity(database_path: Path) -> tuple[str, str] | None:
    try:
        with sqlite3.connect(database_path) as connection:
            has_version_table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'schema_metadata'"
            ).fetchone()
            if has_version_table is not None:
                rows = connection.execute(
                    "SELECT revision, runtime_profile FROM schema_metadata "
                    "WHERE singleton_id = 1"
                ).fetchall()
            else:
                return None
    except sqlite3.OperationalError:
        return None
    except sqlite3.DatabaseError as exc:
        raise UnsupportedDataRoot(
            "data-v2/saber.sqlite3 不是当前架构的有效 SQLite 数据库"
        ) from exc
    if (
        len(rows) != 1
        or not isinstance(rows[0][0], str)
        or not isinstance(rows[0][1], str)
    ):
        return None
    return str(rows[0][0]), str(rows[0][1])


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


def initialize_database(
    data_root: Path,
    *,
    profile_name: str = "local",
) -> StorageInitializationResult:
    """Create the formal schema or validate an exact-current database.

    Databases are never migrated, and a data root is permanently owned by the
    profile that created it.
    """

    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unsupported runtime profile: {profile_name!r}")
    data_root.mkdir(parents=True, exist_ok=True)
    database_path = database_path_for(data_root)
    created = not database_path.exists() or database_path.stat().st_size == 0
    current_identity = None if created else _database_identity(database_path)
    if not created and current_identity is None:
        raise UnsupportedDataRoot(
            "data-v2 不属于当前正式存储架构；旧数据不会被读取或迁移，"
            "请先备份数据目录，再手工清空 data-v2 后重新启动"
        )
    if not created:
        current_revision, current_profile = current_identity
        if current_revision != SCHEMA_REVISION:
            raise UnsupportedDataRoot(
                "data-v2 不属于当前正式存储架构；旧数据不会被读取或迁移，"
                "请先备份数据目录，再手工清空 data-v2 后重新启动"
            )
        if current_profile != profile_name:
            raise UnsupportedDataRoot(
                f"该数据目录属于 {current_profile} 模式，不能由 {profile_name} 模式使用"
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
                        runtime_profile=profile_name,
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
        seed_system_records(engine, profile_name=profile_name)
    finally:
        engine.dispose()
    return StorageInitializationResult(
        database_path=database_path,
        schema_revision=revision,
        created=created,
    )
