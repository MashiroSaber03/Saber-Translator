"""SQLite engine/session construction for v2."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker


DEFAULT_BUSY_TIMEOUT_MS = 5_000
DATABASE_FILENAME = "saber.sqlite3"


def sqlite_url(database_path: Path) -> str:
    return f"sqlite+pysqlite:///{database_path.resolve().as_posix()}"


def database_path_for(data_root: Path) -> Path:
    return data_root / DATABASE_FILENAME


def create_sqlite_engine(
    database_path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> Engine:
    if busy_timeout_ms < 1:
        raise ValueError("busy_timeout_ms must be positive")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        sqlite_url(database_path),
        future=True,
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection: sqlite3.Connection, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        finally:
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        with session.begin():
            yield session
    finally:
        session.close()


@contextmanager
def immediate_transaction(engine: Engine) -> Iterator[Connection]:
    """Open a short SQLite ``BEGIN IMMEDIATE`` transaction.

    This is reserved for check-then-write commands whose correctness depends on
    acquiring the write reservation before reading revisions or ordinals.
    """

    connection = engine.connect()
    try:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()
