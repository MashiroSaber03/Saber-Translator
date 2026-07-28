"""Data-root scoped, non-blocking single-instance lock."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import BinaryIO


class DataRootAlreadyLocked(RuntimeError):
    pass


@dataclass(slots=True)
class DataRootLock:
    data_root: Path
    _handle: BinaryIO | None = None

    @property
    def lock_path(self) -> Path:
        return self.data_root / "runtime" / "launcher.lock"

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("data-root lock is already acquired by this object")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        try:
            handle.seek(0)
            try:
                first_byte = handle.read(1)
            except PermissionError as exc:
                raise DataRootAlreadyLocked(
                    "another Saber Translator Launcher already owns this data-v2 root"
                ) from exc
            if first_byte == b"":
                handle.seek(0)
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            self._lock_byte(handle)
            owner = json.dumps({"pid": os.getpid()}, sort_keys=True).encode("ascii")
            handle.seek(1)
            handle.truncate()
            handle.write(owner)
            handle.flush()
        except BaseException:
            handle.close()
            raise
        self._handle = handle

    @staticmethod
    def _lock_byte(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                handle.flush()
                os.lseek(handle.fileno(), 0, os.SEEK_SET)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise DataRootAlreadyLocked(
                    "another Saber Translator Launcher already owns this data-v2 root"
                ) from exc
            return

        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DataRootAlreadyLocked(
                "another Saber Translator Launcher already owns this data-v2 root"
            ) from exc

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                handle.flush()
                os.lseek(handle.fileno(), 0, os.SEEK_SET)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "DataRootLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()
