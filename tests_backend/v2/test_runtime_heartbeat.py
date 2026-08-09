from __future__ import annotations

import sqlite3
import threading
import time

from sqlalchemy.exc import OperationalError as SqlAlchemyOperationalError

from src.backend_v2.runtime_heartbeat import EpochHeartbeat
from src.backend_v2.runtime_identity import RuntimeIdentity


class _Repository:
    def __init__(self, result: bool | BaseException) -> None:
        self.result = result
        self.called = threading.Event()
        self.calls = 0

    def renew(self, **_kwargs: object) -> bool:
        self.calls += 1
        self.called.set()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _heartbeat(
    repository: _Repository,
    on_fenced: threading.Event,
) -> EpochHeartbeat:
    return EpochHeartbeat(
        repository,  # type: ignore[arg-type]
        role="api",
        identity=RuntimeIdentity("epoch", "token"),
        interval_seconds=0.01,
        on_fenced=on_fenced.set,
    )


def test_zero_row_epoch_renewal_fences_and_calls_shutdown() -> None:
    repository = _Repository(False)
    shutdown = threading.Event()
    heartbeat = _heartbeat(repository, shutdown)

    heartbeat.start()
    assert shutdown.wait(1)
    heartbeat.stop()

    assert repository.called.is_set()
    assert not heartbeat.healthy


def test_epoch_renewal_exception_cannot_silently_kill_heartbeat() -> None:
    repository = _Repository(RuntimeError("database unavailable"))
    shutdown = threading.Event()
    heartbeat = _heartbeat(repository, shutdown)

    heartbeat.start()
    assert shutdown.wait(1)
    heartbeat.stop()

    assert repository.called.is_set()
    assert not heartbeat.healthy


def _sqlite_busy_error() -> SqlAlchemyOperationalError:
    return SqlAlchemyOperationalError(
        "UPDATE process_epochs",
        (),
        sqlite3.OperationalError("database is locked"),
    )


def test_epoch_heartbeat_retries_one_sqlite_lock_without_shutdown() -> None:
    class BusyThenRenewRepository(_Repository):
        def __init__(self) -> None:
            super().__init__(True)

        def renew(self, **kwargs: object) -> bool:
            if self.calls == 0:
                self.calls += 1
                self.called.set()
                raise _sqlite_busy_error()
            return super().renew(**kwargs)

    repository = BusyThenRenewRepository()
    shutdown = threading.Event()
    heartbeat = _heartbeat(repository, shutdown)

    heartbeat.start()
    try:
        deadline = time.monotonic() + 1
        while repository.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert repository.calls >= 2
        assert heartbeat.healthy
        assert not shutdown.is_set()
    finally:
        heartbeat.stop()


def test_epoch_heartbeat_fences_after_finite_sqlite_busy_retries() -> None:
    repository = _Repository(_sqlite_busy_error())
    shutdown = threading.Event()
    heartbeat = _heartbeat(repository, shutdown)

    heartbeat.start()
    assert shutdown.wait(1)
    heartbeat.stop()

    assert repository.calls == 2
    assert not heartbeat.healthy
