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
from src.backend_v2.storage.schema import metadata, schema_metadata
from src.backend_v2.storage.seeding import seed_system_records


SCHEMA_REVISION = "backend_v2_20260820"
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
            if has_version_table is None:
                return None
            rows = connection.execute(
                "SELECT revision FROM schema_metadata WHERE singleton_id = 1"
            ).fetchall()
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


def initialize_database(data_root: Path) -> StorageInitializationResult:
    """Create the formal schema or validate an exact-current database.

    Any existing database whose revision differs from the current foundation is
    rejected. Old data is never read, converted, upgraded, backed up, or stamped.
    """

    database_path = database_path_for(data_root)
    created = not database_path.exists() or database_path.stat().st_size == 0
    current_revision = None if created else _database_revision(database_path)
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
