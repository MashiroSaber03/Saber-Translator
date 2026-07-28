from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.schema import metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        "worker_leases",
        "api_executor_leases",
        "chapter_write_intents",
        "chapter_write_locks",
        "object_commit_journal",
        "idempotency_records",
    }
    with engine.connect() as connection:
        actual = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
        }
    assert expected <= actual


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
                "(id, kind, status, config_json, created_at, updated_at) "
                "VALUES ('job-1', 'translation', 'running', '{}', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO jobs "
                    "(id, kind, status, config_json, created_at, updated_at) "
                    "VALUES ('job-2', 'export', 'pausing', '{}', "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )


def test_alembic_head_upgrades_and_downgrades_without_fk_damage(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "migration.sqlite3"
    environment = os.environ.copy()
    environment["SABER_V2_DATABASE_URL"] = (
        f"sqlite+pysqlite:///{database_path.resolve().as_posix()}"
    )
    command = [
        sys.executable,
        "-m",
        "alembic",
        "-c",
        str(PROJECT_ROOT / "alembic.ini"),
    ]
    subprocess.run(
        [*command, "upgrade", "head"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    engine = create_sqlite_engine(database_path)
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
        index_sql = connection.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type='index' AND name='uq_jobs_one_current'"
            )
        ).scalar_one()
        assert "WHERE status IN" in index_sql
    engine.dispose()

    subprocess.run(
        [*command, "downgrade", "base"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
