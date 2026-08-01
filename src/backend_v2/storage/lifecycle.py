"""Launcher-owned initialization and integrity checks for the v2 data root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import uuid

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from src.backend_v2.paths import project_root
from src.backend_v2.storage.database import (
    create_sqlite_engine,
    database_path_for,
    sqlite_url,
)
from src.backend_v2.storage.schema import metadata
from src.backend_v2.storage.seeding import seed_system_records


REQUIRED_TABLES = frozenset(metadata.tables) | {"alembic_version"}


class UnsupportedDataRoot(RuntimeError):
    """The database is not a revision owned by the current formal schema."""


@dataclass(frozen=True, slots=True)
class StorageInitializationResult:
    database_path: Path
    schema_revision: str
    created: bool
    upgraded: bool


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_connection, sqlite3.connect(
        destination
    ) as destination_connection:
        source_connection.backup(destination_connection)


def _alembic_config(database_path: Path) -> Config:
    config = Config()
    config.set_main_option(
        "script_location",
        str(project_root() / "src" / "backend_v2" / "storage" / "migrations"),
    )
    config.set_main_option("sqlalchemy.url", sqlite_url(database_path))
    return config


def _database_revision(database_path: Path) -> str | None:
    try:
        with sqlite3.connect(database_path) as connection:
            has_version_table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'alembic_version'"
            ).fetchone()
            if has_version_table is None:
                return None
            rows = connection.execute(
                "SELECT version_num FROM alembic_version"
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise UnsupportedDataRoot(
            "data-v2/saber.sqlite3 不是当前架构的有效 SQLite 数据库"
        ) from exc
    if len(rows) != 1 or not isinstance(rows[0][0], str):
        return None
    return rows[0][0]


def _formal_revisions(config: Config) -> tuple[str, frozenset[str]]:
    scripts = ScriptDirectory.from_config(config)
    head = scripts.get_current_head()
    if head is None:
        raise RuntimeError("formal v2 schema has no Alembic head")
    return head, frozenset(item.revision for item in scripts.walk_revisions())


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
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            return str(revision)
    finally:
        engine.dispose()


def initialize_database(data_root: Path) -> StorageInitializationResult:
    """Create the formal schema or advance an already-formal v2 revision.

    Databases without a revision present in this formal migration tree are
    rejected. No development-era or pre-v2 data is read, converted, or stamped.
    """

    database_path = database_path_for(data_root)
    created = not database_path.exists() or database_path.stat().st_size == 0
    config = _alembic_config(database_path)
    head, known_revisions = _formal_revisions(config)
    current_revision = None if created else _database_revision(database_path)
    if not created and current_revision not in known_revisions:
        raise UnsupportedDataRoot(
            "data-v2 不属于当前正式存储架构；旧数据不会被读取或迁移，"
            "请清空 data-v2 后重新启动"
        )

    upgraded = not created and current_revision != head
    backup_path: Path | None = None
    if upgraded:
        backup_path = (
            data_root
            / "runtime"
            / f"pre-upgrade-{uuid.uuid4().hex}.sqlite3"
        )
        _sqlite_backup(database_path, backup_path)

    try:
        if created or upgraded:
            command.upgrade(config, "head")
        revision = schema_smoke_test(database_path)
        if revision != head:
            raise RuntimeError(
                f"database revision {revision!r} does not match formal head {head!r}"
            )
        engine = create_sqlite_engine(database_path)
        try:
            seed_system_records(engine)
        finally:
            engine.dispose()
    except BaseException:
        if backup_path is not None and backup_path.exists():
            _sqlite_backup(backup_path, database_path)
        raise
    else:
        if backup_path is not None and backup_path.exists():
            backup_path.unlink()
        return StorageInitializationResult(
            database_path=database_path,
            schema_revision=revision,
            created=created,
            upgraded=upgraded,
        )
