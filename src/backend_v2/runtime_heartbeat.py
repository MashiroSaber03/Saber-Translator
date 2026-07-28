"""Role heartbeat loop that turns zero-row renewal into immediate self-fencing."""

from __future__ import annotations

from collections.abc import Callable
import threading

from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.epochs import ProcessEpochRepository


class EpochHeartbeat:
    def __init__(
        self,
        repository: ProcessEpochRepository,
        *,
        role: str,
        identity: RuntimeIdentity,
        interval_seconds: float = 2.0,
        on_fenced: Callable[[], None] | None = None,
    ) -> None:
        if role not in {"api", "worker"}:
            raise ValueError("only API and Worker have runtime heartbeat loops")
        self._repository = repository
        self._role = role
        self._identity = identity
        self._interval_seconds = interval_seconds
        self._on_fenced = on_fenced
        self._stop = threading.Event()
        self._fenced = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def healthy(self) -> bool:
        return not self._fenced.is_set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("heartbeat already started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"v2-{self._role}-epoch-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            renewed = self._repository.renew(
                role=self._role,  # type: ignore[arg-type]
                epoch_id=self._identity.epoch_id,
                token=self._identity.epoch_token,
            )
            if renewed:
                continue
            self._fenced.set()
            self._stop.set()
            if self._on_fenced is not None:
                self._on_fenced()
            return

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self._interval_seconds * 2))
