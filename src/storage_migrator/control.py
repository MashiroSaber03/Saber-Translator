"""Stable coordination outside the swappable data root; no business DB access."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import time
import uuid

import psutil

from .contracts import StorageError


def reject_links(path: Path) -> None:
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise StorageError(f"拒绝链接或重解析目录: {path}")


def control_root(root: Path) -> Path:
    root = root.resolve()
    path = root.parent / f".{root.name}.saber-storage"
    reject_links(path)
    return path


def is_empty_root(root: Path) -> bool:
    if not root.exists():
        return True
    for path in root.rglob("*"):
        reject_links(path)
        if path.is_file() and path.relative_to(root).parts[0] not in {"logs", "runtime"}:
            return False
    return True


def atomic_json(path: Path, payload: dict) -> None:
    reject_links(path.parent)
    reject_links(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    reject_links(temporary)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class DataRootAlreadyLocked(StorageError):
    pass


class DataRootLock:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self._handle = None

    @property
    def lock_path(self) -> Path:
        return control_root(self.data_root) / "launcher.lock"

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("data-root lock is already acquired by this object")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        reject_links(self.lock_path)
        handle = self.lock_path.open("a+b")
        try:
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise DataRootAlreadyLocked("another Saber Translator Launcher/Desktop already owns this data root；请先退出已有实例") from exc
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.close()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def wait_for_children(root: Path, *, timeout: float = 5.0) -> None:
    folder = control_root(root) / "processes"
    reject_links(folder)
    if not folder.exists():
        return
    deadline = time.monotonic() + timeout
    while True:
        live = []
        for path in folder.glob("*.json"):
            reject_links(path)
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("data_root") != str(root.resolve()):
                raise StorageError("进程记录与数据目录不匹配")
            try:
                process = psutil.Process(record["pid"])
                alive = process.create_time() == record["created_at"] and process.is_running()
            except psutil.NoSuchProcess:
                alive = False
            except psutil.AccessDenied as exc:
                raise StorageError("无法确认旧进程是否退出，拒绝转换") from exc
            if alive:
                live.append(record["pid"])
            else:
                path.unlink(missing_ok=True)
        if not live:
            return
        if time.monotonic() >= deadline:
            raise StorageError(f"旧 API/Worker 尚未退出: {live}；请先关闭旧程序")
        time.sleep(0.1)


@contextmanager
def registered_process(root: Path, role: str):
    folder = control_root(root) / "processes"
    reject_links(folder)
    folder.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    path = folder / f"{role}-{process.pid}.json"
    atomic_json(path, {"data_root": str(root.resolve()), "pid": process.pid, "created_at": process.create_time(), "role": role})
    # Keep the record until the OS process has exited, including log teardown.
    # The next owner removes records whose PID + creation time are no longer live.
    yield


def business_ready(root: Path) -> bool:
    control = control_root(root)
    pending = control / "upgrade-pending.json"
    if not pending.exists():
        return True
    try:
        operation_id = json.loads(pending.read_text(encoding="utf-8"))["id"]
        if str(uuid.UUID(operation_id)) != operation_id:
            return False
        state = json.loads((control / "operations" / operation_id / "state.json").read_text(encoding="utf-8"))
        return (state.get("id") == operation_id and state.get("root") == str(root.resolve())
                and state.get("stage") in {"committed", "cleaned"})
    except (OSError, ValueError, KeyError, TypeError):
        return False
