"""Launcher-owned initialization and integrity checks for the v2 data root."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from sqlalchemy import insert

from src.backend_v2.storage.database import (
    create_sqlite_engine,
    database_path_for,
)
from src.backend_v2.runtime_profile import PROFILE_NAMES
from src.backend_v2.storage.schema import metadata, schema_metadata
from src.backend_v2.storage.seeding import seed_system_records


REQUIRED_TABLES = frozenset(metadata.tables)


class UnsupportedDataRoot(RuntimeError):
    """The database is invalid or belongs to another runtime profile."""


@dataclass(frozen=True, slots=True)
class StorageInitializationResult:
    database_path: Path
    created: bool


def _database_profile(database_path: Path) -> str | None:
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            rows = connection.execute(
                "SELECT runtime_profile FROM schema_metadata WHERE singleton_id = 1"
            ).fetchall()
    except sqlite3.OperationalError:
        return None
    except sqlite3.DatabaseError as exc:
        raise UnsupportedDataRoot(
            "data-v2/saber.sqlite3 不是当前架构的有效 SQLite 数据库"
        ) from exc
    if len(rows) != 1 or not isinstance(rows[0][0], str):
        return None
    return str(rows[0][0])


def schema_smoke_test(database_path: Path) -> None:
    with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(f"SQLite foreign_key_check failed: {foreign_key_errors!r}")
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not str(row[0]).startswith("sqlite_")
        }
        missing = REQUIRED_TABLES - tables
        unexpected = tables - REQUIRED_TABLES
        if missing or unexpected:
            raise RuntimeError(
                "v2 schema table mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

def initialize_database(
    data_root: Path,
    *,
    profile_name: str = "local",
) -> StorageInitializationResult:
    """Create the database or check its integrity and runtime profile.

    Databases are never migrated, and a data root is permanently owned by the
    profile that created it.
    """

    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unsupported runtime profile: {profile_name!r}")
    data_root.mkdir(parents=True, exist_ok=True)
    database_path = database_path_for(data_root)
    created = not database_path.exists() or database_path.stat().st_size == 0
    if not created:
        current_profile = _database_profile(database_path)
        if current_profile is None:
            raise UnsupportedDataRoot("数据库缺少有效的运行模式信息，可能尚未完成初始化")
        if current_profile != profile_name:
            raise UnsupportedDataRoot(
                f"该数据目录属于 {current_profile} 模式，不能由 {profile_name} 模式使用"
            )

    if created:
        engine = create_sqlite_engine(database_path)
        try:
            metadata.create_all(engine)
            seed_system_records(engine, profile_name=profile_name)
            # Publish the profile only after the initial records are committed.
            with engine.begin() as connection:
                connection.execute(
                    insert(schema_metadata).values(
                        singleton_id=1,
                        runtime_profile=profile_name,
                    )
                )
        finally:
            engine.dispose()
    schema_smoke_test(database_path)
    return StorageInitializationResult(
        database_path=database_path,
        created=created,
    )
