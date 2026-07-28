"""Launcher-owned migration, SQLite backup, rollback, and schema smoke checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import uuid

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from src.backend_v2.paths import project_root
from src.backend_v2.storage.database import (
    create_sqlite_engine,
    database_path_for,
    sqlite_url,
)
from src.backend_v2.storage.seeding import seed_system_records


REQUIRED_TABLES = frozenset(
    {
        "alembic_version",
        "assets",
        "books",
        "chapters",
        "pages",
        "jobs",
        "operations",
        "process_epochs",
        "object_commit_journal",
        "credentials",
        "credential_versions",
        "prompts",
        "analysis_runs",
        "analysis_run_targets",
        "analysis_page_results",
        "analysis_heads",
        "analysis_layer_results",
        "analysis_artifacts",
        "timeline_versions",
        "vector_generations",
        "notes",
        "studio_documents",
        "studio_chat_sessions",
        "studio_messages",
        "plugins",
        "plugin_versions",
        "continuation_projects",
    }
)


@dataclass(frozen=True, slots=True)
class MigrationResult:
    database_path: Path
    upgraded_to: str
    backup_created: bool


def sqlite_backup(source: Path, destination: Path) -> None:
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
                row[0]
                for row in connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
            }
            missing = REQUIRED_TABLES - tables
            if missing:
                raise RuntimeError(f"v2 schema is missing required tables: {sorted(missing)}")
            revision = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            return str(revision)
    finally:
        engine.dispose()


def migrate_database(data_root: Path) -> MigrationResult:
    database_path = database_path_for(data_root)
    existed = database_path.exists() and database_path.stat().st_size > 0
    backup_path = (
        data_root
        / "runtime"
        / f"pre-migration-{uuid.uuid4().hex}.sqlite3"
    )
    if existed:
        sqlite_backup(database_path, backup_path)

    try:
        command.upgrade(_alembic_config(database_path), "head")
        revision = schema_smoke_test(database_path)
        engine = create_sqlite_engine(database_path)
        try:
            seed_system_records(engine)
        finally:
            engine.dispose()
    except BaseException:
        if existed and backup_path.exists():
            sqlite_backup(backup_path, database_path)
        raise
    else:
        if backup_path.exists():
            backup_path.unlink()
        return MigrationResult(
            database_path=database_path,
            upgraded_to=revision,
            backup_created=existed,
        )
