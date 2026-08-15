"""Launcher-issued process identities for v2 roles."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from collections.abc import Callable


API_EPOCH_ID_ENV = "SABER_V2_API_EPOCH_ID"
API_EPOCH_TOKEN_ENV = "SABER_V2_API_EPOCH_TOKEN"
WORKER_EPOCH_ID_ENV = "SABER_V2_WORKER_EPOCH_ID"
WORKER_EPOCH_TOKEN_ENV = "SABER_V2_WORKER_EPOCH_TOKEN"
LAUNCHER_PID_ENV = "SABER_V2_LAUNCHER_PID"


@dataclass(slots=True)
class LauncherParentMonitor:
    """Stop a POSIX role when its owning Launcher process disappears."""

    stop_event: threading.Event
    thread: threading.Thread

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def _watch_launcher_parent(
    launcher_pid: int,
    on_parent_lost: Callable[[], None],
    stop_event: threading.Event,
    *,
    get_parent_pid: Callable[[], int] = os.getppid,
    interval_seconds: float = 0.5,
) -> None:
    while not stop_event.wait(interval_seconds):
        if get_parent_pid() != launcher_pid:
            on_parent_lost()
            return


def start_launcher_parent_monitor(
    on_parent_lost: Callable[[], None],
    *,
    test_mode: bool,
) -> LauncherParentMonitor | None:
    """Watch the direct Launcher parent where Windows Job Objects are unavailable."""

    if test_mode or os.name == "nt":
        return None
    raw_pid = os.environ.get(LAUNCHER_PID_ENV, "")
    try:
        launcher_pid = int(raw_pid)
    except ValueError as exc:
        raise RuntimeError("Launcher-issued parent process identity is missing") from exc
    if launcher_pid <= 0 or os.getppid() != launcher_pid:
        raise RuntimeError("Launcher-issued parent process identity is invalid")
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_watch_launcher_parent,
        args=(launcher_pid, on_parent_lost, stop_event),
        name="saber-launcher-parent-monitor",
        daemon=True,
    )
    thread.start()
    return LauncherParentMonitor(stop_event=stop_event, thread=thread)


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    epoch_id: str
    epoch_token: str
    test_mode: bool = False

    @classmethod
    def for_api(cls, *, test_mode: bool) -> "RuntimeIdentity":
        return cls._from_environment(
            epoch_id_name=API_EPOCH_ID_ENV,
            epoch_token_name=API_EPOCH_TOKEN_ENV,
            role="API",
            test_mode=test_mode,
        )

    @classmethod
    def for_worker(cls, *, test_mode: bool) -> "RuntimeIdentity":
        return cls._from_environment(
            epoch_id_name=WORKER_EPOCH_ID_ENV,
            epoch_token_name=WORKER_EPOCH_TOKEN_ENV,
            role="Worker",
            test_mode=test_mode,
        )

    @classmethod
    def _from_environment(
        cls,
        *,
        epoch_id_name: str,
        epoch_token_name: str,
        role: str,
        test_mode: bool,
    ) -> "RuntimeIdentity":
        epoch_id = os.environ.get(epoch_id_name, "")
        epoch_token = os.environ.get(epoch_token_name, "")
        if epoch_id and epoch_token:
            return cls(epoch_id=epoch_id, epoch_token=epoch_token)
        if test_mode:
            return cls(epoch_id=f"test-{role.lower()}", epoch_token="test-only", test_mode=True)
        raise RuntimeError(
            f"{role} requires a Launcher-issued epoch identity; "
            "direct startup is only allowed with --test-mode"
        )
