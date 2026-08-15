"""SQLite engine/session construction for v2."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection


DEFAULT_BUSY_TIMEOUT_MS = 5_000
DATABASE_FILENAME = "saber.sqlite3"
SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT = 1
SQLITE_HEARTBEAT_BUSY_RETRY_DELAY_SECONDS = 0.1


def is_sqlite_busy_error(exc: BaseException) -> bool:
    """Recognize SQLite writer contention through DBAPI/SQLAlchemy wrappers."""

    candidates: list[BaseException] = [exc]
    seen: set[int] = set()
    while candidates:
        current = candidates.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, sqlite3.OperationalError):
            code = getattr(current, "sqlite_errorcode", None)
            if isinstance(code, int) and (code & 0xFF) in {
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            }:
                return True
            message = str(current).lower()
            if any(
                marker in message
                for marker in (
                    "database is locked",
                    "database table is locked",
                    "database schema is locked",
                )
            ):
                return True
        for nested in (
            getattr(current, "orig", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(nested, BaseException):
                candidates.append(nested)
    return False


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


@contextmanager
def read_transaction(engine: Engine) -> Iterator[Connection]:
    """Read several SQLite tables from one consistent WAL snapshot."""

    connection = engine.connect()
    try:
        connection.exec_driver_sql("BEGIN")
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()
