"""Offline full-copy conversion with a write-ahead directory publication record."""

from contextlib import closing, contextmanager
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Callable
import uuid

from src.version import STORAGE_VERSION
from .contracts import StorageError, contract_sql, file_hash, read_database, read_identity, validate_data
from .control import DataRootLock, atomic_json, control_root, reject_links, wait_for_children, is_empty_root
from .registry import MIGRATIONS, SOURCE_PREPARERS, VERSION_VALIDATORS, migration_chain
from .paths import filesystem_path


LOGGER = logging.getLogger(__name__)
OWNER_FILE = ".saber-migration-owner.json"


class StorageManager:
    def __init__(self, root: Path, profile: str, *, target=STORAGE_VERSION, steps=MIGRATIONS,
                 sql_for=contract_sql, progress: Callable[[str], None] | None = None,
                 failpoint: Callable[[str], None] | None = None, preparers=SOURCE_PREPARERS):
        reject_links(root)
        self.root = root.resolve()
        self.profile = profile
        self.target = target
        self.steps = steps
        self.preparers = preparers
        self.sql_for = sql_for
        self.progress = progress or (lambda message: LOGGER.info(message))
        self.failpoint = failpoint or (lambda _stage: None)
        self.lock = DataRootLock(self.root)
        self.control = control_root(self.root)
        self.pending = self.control / "upgrade-pending.json"
        self.rolled_back = False

    def __enter__(self):
        self.lock.acquire()
        try:
            wait_for_children(self.root)
        except BaseException:
            self.lock.release()
            raise
        return self

    def __exit__(self, *_args):
        self.lock.release()

    def _require_lock(self):
        if self.lock._handle is None:
            raise StorageError("转换必须持有数据目录独占锁")

    def check(self) -> dict:
        if not (self.root / "saber.sqlite3").exists():
            if not is_empty_root(self.root):
                raise StorageError("数据库缺失但目录仍有旧数据；拒绝覆盖，请选择新的数据目录")
            return {"status": "new", "targetVersion": self.target}
        source, profile = read_identity(self.root)
        if profile != self.profile:
            raise StorageError(f"该数据目录属于 {profile} 模式，不能由 {self.profile} 模式使用")
        chain = migration_chain(source, self.target, self.steps)
        self._validate_version(self.root, source, files=self.pending.exists())
        return {"status": "current" if not chain else "upgrade_required", "sourceVersion": source,
                "targetVersion": self.target, "steps": [step.target for step in chain]}

    def _validate_version(self, root: Path, version: str, *, files=False):
        root = filesystem_path(root)
        validate_data(root, self.sql_for(version), files=files)
        validator = VERSION_VALIDATORS.get(version)
        if validator:
            with closing(read_database(root / "saber.sqlite3")) as db:
                db.execute("BEGIN")
                validator(db, root=root, files=files)

    def _operations(self):
        folder = filesystem_path(self.control / "operations")
        reject_links(folder)
        if folder.exists():
            for op in sorted(folder.iterdir()):
                reject_links(op)
                if not op.is_dir():
                    continue
                try:
                    uuid.UUID(op.name)
                except ValueError:
                    continue
                state_file = op / "state.json"
                reject_links(state_file)
                if not state_file.is_file():
                    if any(op.iterdir()):
                        raise StorageError(f"转换记录缺失，拒绝猜测或初始化空库: {op}")
                    continue
                state = json.loads(state_file.read_text(encoding="utf-8"))
                if state.get("id") != op.name or state.get("root") != str(self.root):
                    raise StorageError("转换目录身份不匹配，拒绝修改")
                if state.get("profile") != self.profile:
                    raise StorageError("转换记录的运行模式不匹配，拒绝修改")
                yield op, state

    def _state(self, op: Path, state: dict, stage: str) -> None:
        state["stage"] = stage
        atomic_json(op / "state.json", state)
        self.progress(f"存储升级：{stage}")
        self.failpoint(stage)

    def _marker(self, path: Path, state: dict) -> None:
        atomic_json(path / OWNER_FILE, {"id": state["id"], "root": state["root"]})

    def _owned(self, path: Path, state: dict) -> None:
        reject_links(path)
        marker = path / OWNER_FILE
        reject_links(marker)
        expected = {"id": state["id"], "root": state["root"]}
        try:
            if not marker.exists():
                # A kill between fsync and replace can leave a complete marker.
                # Recover it only after validating the same ownership identity.
                temporary = marker.with_suffix(marker.suffix + ".tmp")
                reject_links(temporary)
                if json.loads(temporary.read_text(encoding="utf-8")) == expected:
                    os.replace(temporary, marker)
            identity = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError(f"无法读取转换目录身份，拒绝删除/替换: {path}") from exc
        if identity != expected:
            raise StorageError(f"无法核验转换目录，拒绝删除/替换: {path}")

    def _remove(self, path: Path, state: dict) -> None:
        path = filesystem_path(path)
        reject_links(path)
        if not path.exists():
            return
        if not path.resolve().is_relative_to(filesystem_path(self.control / "operations" / state["id"]).resolve()):
            raise StorageError("清理目标越界")
        if not any(path.iterdir()):
            path.rmdir()
            return
        self._owned(path, state)
        for child in path.rglob("*"):
            reject_links(child)
        # Keep the marker until the last unlink, making interrupted deletion retryable.
        for child in path.iterdir():
            if child.name == OWNER_FILE:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        (path / OWNER_FILE).unlink()
        path.rmdir()

    def _rollback(self, op: Path, state: dict) -> None:
        backup, work, discarded = op / "backup", op / "work", op / "discarded"
        if backup.exists():
            self._owned(backup, state)
            if self.root.exists():
                self._owned(self.root, state)
                if discarded.exists():
                    raise StorageError("存在未决恢复目录，请检查转换日志")
                os.replace(self.root, discarded)
            os.replace(backup, self.root)
            (self.root / OWNER_FILE).unlink(missing_ok=True)
        elif state["stage"] in {"old_moved", "installed", "starting"}:
            if not self.root.exists() or read_identity(self.root)[0] != state["source"]:
                raise StorageError("恢复所需备份缺失，拒绝猜测或初始化空库")
            (self.root / OWNER_FILE).unlink(missing_ok=True)
        elif self.root.exists() and (self.root / OWNER_FILE).exists():
            self._owned(self.root, state)
            (self.root / OWNER_FILE).unlink()
        self.pending.unlink(missing_ok=True)
        self._state(op, state, "rolled_back")
        self.rolled_back = True
        for path in (work, discarded):
            try:
                self._remove(path, state)
            except (OSError, StorageError):
                LOGGER.warning("恢复完成，临时目录清理待重试: %s", path)

    def recover(self) -> None:
        operations = list(self._operations())
        if self.pending.exists():
            try:
                pending_id = json.loads(self.pending.read_text(encoding="utf-8"))["id"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise StorageError("升级门禁记录无效，拒绝初始化或开放业务") from exc
            if pending_id not in {state["id"] for _, state in operations}:
                raise StorageError("升级门禁缺少对应恢复记录，拒绝初始化或开放业务")
        if sum(state["stage"] not in {"committed", "cleaned", "rolled_back"} for _, state in operations) > 1:
            raise StorageError("存在多个未决转换，拒绝猜测恢复顺序")
        for op, state in operations:
            stage = state["stage"]
            if stage not in {"copying", "converting", "validated", "old_moved", "installed", "starting", "committed", "cleaned", "rolled_back"}:
                raise StorageError("无法识别转换恢复阶段，拒绝修改数据")
            if stage in {"committed", "cleaned", "rolled_back"}:
                continue
            if stage == "installed" and self.root.exists() and read_identity(self.root) == (self.target, self.profile):
                self._owned(self.root, state)
                atomic_json(self.pending, {"id": state["id"]})
                continue
            self._rollback(op, state)
            raise StorageError("上次存储准备被中断，已恢复旧数据；请重新启动以重试。")

    def prepare(self, initialize: Callable) -> dict:
        self._require_lock()
        try:
            return self._prepare(initialize)
        except Exception:
            self.startup_failed(include_installed=True)
            raise

    def _prepare(self, initialize: Callable) -> dict:
        self.recover()
        result = self.check()
        if result["status"] == "new":
            self.progress("正在初始化三段式版本存储")
            initialize(self.root, profile_name=self.profile)
            self.check()
            return {**result, "status": "created"}
        if result["status"] == "current":
            if self.target == STORAGE_VERSION:
                initialize(self.root, profile_name=self.profile)
            return result
        chain = migration_chain(result["sourceVersion"], self.target, self.steps)
        if all(step.convert is None for step in chain):
            for step in chain:
                self._validate_version(self.root, step.target)
                if step.validate:
                    step.validate(self.root)
            with closing(sqlite3.connect(self.root / "saber.sqlite3")) as db, db:
                db.execute("UPDATE schema_metadata SET storage_version=? WHERE singleton_id=1", (self.target,))
            return {**result, "status": "updated"}
        prepare_source = self.preparers.get(result["sourceVersion"])
        if prepare_source is None:
            raise StorageError("缺少源版本任务终止规则，拒绝执行实际转换")
        source_root = filesystem_path(self.root)
        size = 0
        for path in source_root.rglob("*"):
            reject_links(path)
            if path.is_file():
                size += path.stat().st_size
        if shutil.disk_usage(self.root.parent).free < size + max(size // 10, 256 * 1024 * 1024):
            raise StorageError("磁盘空间不足，无法创建完整转换副本；原数据未修改")
        op = filesystem_path(self.control / "operations" / str(uuid.uuid4()))
        reject_links(op.parent)
        op.mkdir(parents=True)
        state = {"id": op.name, "root": str(self.root), "profile": self.profile,
                 "source": result["sourceVersion"], "target": self.target}
        atomic_json(op / "state.json", {**state, "stage": "copying"})
        work = op / "work"
        work.mkdir()
        self._marker(work, state)
        try:
            self._state(op, state, "copying")
            for path in source_root.rglob("*"):
                relative = path.relative_to(source_root)
                if relative.parts[0] == "runtime" or (len(relative.parts) == 1 and relative.name in {"saber.sqlite3", "saber.sqlite3-wal", "saber.sqlite3-shm", OWNER_FILE}):
                    continue
                destination = work / relative
                if path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                    if file_hash(path) != file_hash(destination):
                        raise StorageError(f"工作副本校验失败: {relative}")
                    self.progress(f"正在复制存储文件：{relative}")
            with closing(read_database(self.root / "saber.sqlite3")) as source, closing(sqlite3.connect(work / "saber.sqlite3")) as target:
                source.backup(target)
            self._state(op, state, "converting")
            prepare_source(work)
            for step in chain:
                if step.convert:
                    step.convert(work)
                self._validate_version(work, step.target, files=True)
                if step.validate:
                    step.validate(work)
                with closing(sqlite3.connect(work / "saber.sqlite3")) as db, db:
                    db.execute("UPDATE schema_metadata SET storage_version=? WHERE singleton_id=1", (step.target,))
                if read_identity(work) != (step.target, self.profile):
                    raise StorageError("转换后的存储身份不一致")
            self._state(op, state, "validated")
            self._marker(self.root, state)
            os.replace(self.root, op / "backup")
            self._state(op, state, "old_moved")
            os.replace(work, self.root)
            atomic_json(self.pending, {"id": state["id"]})
            self._state(op, state, "installed")
            return {**result, "status": "awaiting_health"}
        except Exception:
            self._rollback(op, state)
            raise

    def before_start(self) -> None:
        self._require_lock()
        for op, state in self._operations():
            if state["stage"] == "installed":
                self._state(op, state, "starting")

    @contextmanager
    def startup_guard(self):
        try:
            yield
        except Exception:
            self.startup_failed(include_installed=True)
            raise
        finally:
            self.startup_failed()

    def startup_failed(self, *, include_installed: bool = False) -> None:
        self._require_lock()
        for op, state in self._operations():
            if state["stage"] == "starting" or (include_installed and state["stage"] == "installed"):
                wait_for_children(self.root)
                self._rollback(op, state)

    def confirm_ready(self) -> list[str]:
        self._require_lock()
        for op, state in self._operations():
            if state["stage"] == "installed":
                raise StorageError("尚未开始后端就绪验收，不能提交升级")
            if state["stage"] == "starting":
                self._owned(self.root, state)
                self._state(op, state, "committed")
        return self.cleanup()

    def cleanup(self) -> list[str]:
        self._require_lock()
        errors = []
        for op, state in self._operations():
            if state["stage"] not in {"committed", "rolled_back"}:
                continue
            try:
                if state["stage"] == "committed":
                    if self.pending.exists():
                        pending = json.loads(self.pending.read_text(encoding="utf-8"))
                        if pending.get("id") == state["id"]:
                            self.pending.unlink()
                    marker = self.root / OWNER_FILE
                    if marker.exists():
                        identity = json.loads(marker.read_text(encoding="utf-8"))
                        if identity == {"id": state["id"], "root": state["root"]}:
                            marker.unlink()
                for name in ("backup", "work", "discarded"):
                    self._remove(op / name, state)
                state.pop("cleanup_error", None)
                self._state(op, state, "cleaned")
            except (OSError, StorageError, ValueError) as exc:
                message = f"升级成功或恢复完成，备份清理待重试: {op} ({exc})"
                errors.append(message)
                LOGGER.warning(message)
                state["cleanup_error"] = str(exc)
                try:
                    atomic_json(op / "state.json", state)
                except OSError:
                    LOGGER.exception("无法保存清理失败原因；下次启动仍会重试")
        return errors
