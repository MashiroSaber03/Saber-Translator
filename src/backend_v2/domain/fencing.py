"""In-process reaction to authoritative heartbeat/lease fencing."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


class FencedExecution(RuntimeError):
    """Raised when a stale executor tries to schedule or publish work."""


@dataclass(slots=True)
class ExecutorFence:
    """Local mirror of whether this executor may continue.

    Database CAS remains authoritative.  This object closes admission as soon
    as a heartbeat reports zero affected rows, avoiding unnecessary work before
    the final publish CAS would reject it.
    """

    _epoch_valid: bool = True
    _admission_open: bool = True
    _poisoned_attempts: set[str] = field(default_factory=set)
    _lock: RLock = field(default_factory=RLock)

    def observe_epoch_renewal(self, affected_rows: int) -> None:
        if affected_rows not in (0, 1):
            raise ValueError("epoch renewal must affect zero or one row")
        if affected_rows == 1:
            return
        with self._lock:
            self._epoch_valid = False
            self._admission_open = False

    def observe_attempt_renewal(self, attempt_id: str, affected_rows: int) -> None:
        if not attempt_id:
            raise ValueError("attempt_id is required")
        if affected_rows not in (0, 1):
            raise ValueError("attempt renewal must affect zero or one row")
        if affected_rows == 0:
            with self._lock:
                self._poisoned_attempts.add(attempt_id)

    def may_admit(self) -> bool:
        with self._lock:
            return self._epoch_valid and self._admission_open

    def may_publish(self, attempt_id: str) -> bool:
        with self._lock:
            return self._epoch_valid and attempt_id not in self._poisoned_attempts

    def require_admission(self) -> None:
        if not self.may_admit():
            raise FencedExecution("executor epoch lost; new work admission is closed")

    def require_publish(self, attempt_id: str) -> None:
        if not self.may_publish(attempt_id):
            raise FencedExecution(f"attempt {attempt_id} no longer owns publish rights")
