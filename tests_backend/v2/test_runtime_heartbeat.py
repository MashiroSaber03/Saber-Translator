from __future__ import annotations

import threading

from src.backend_v2.runtime_heartbeat import EpochHeartbeat
from src.backend_v2.runtime_identity import RuntimeIdentity


class _Repository:
    def __init__(self, result: bool | BaseException) -> None:
        self.result = result
        self.called = threading.Event()

    def renew(self, **_kwargs: object) -> bool:
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
