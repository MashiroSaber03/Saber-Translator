"""Minimal Launcher for the isolated v2 API and Worker processes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
import uuid
import webbrowser

from src.backend_v2.logging_config import LOG_LEVEL_ENV, configure_backend_logging
from src.backend_v2.paths import (
    DATA_ROOT_ENV,
    data_root_fingerprint,
    ensure_data_root,
    project_root,
    resolve_data_root,
)
from src.backend_v2.runtime_identity import (
    API_EPOCH_ID_ENV,
    API_EPOCH_TOKEN_ENV,
    WORKER_EPOCH_ID_ENV,
    WORKER_EPOCH_TOKEN_ENV,
)
from src.backend_v2.launcher.windows_job import ChildProcessJob
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.lifecycle import migrate_database
from src.backend_v2.storage.single_instance import DataRootLock


MAX_CONSECUTIVE_RESTARTS = 3
API_HEALTH_CHECK_INTERVAL_SECONDS = 1.0
API_HEALTH_FAILURE_LIMIT = 3
RESTART_STABILITY_SECONDS = 30.0
LOGGER = logging.getLogger("saber.launcher")


@dataclass(slots=True)
class ManagedChild:
    role: str
    process: subprocess.Popen[str]
    registration: EpochRegistration
    restart_count: int = 0
    ready_at: float = 0.0
    next_health_check_at: float = 0.0
    health_failures: int = 0


def _role_command(role: str, *, data_root: Path, host: str, port: int) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [sys.executable]
    else:
        command = [sys.executable, str(project_root() / "saber_v2.py")]
    return [
        *command,
        "--role",
        role,
        "--data-dir",
        str(data_root),
        "--host",
        host,
        "--port",
        str(port),
    ]


def _new_registration(role: str, *, pid: int = 0) -> EpochRegistration:
    if role not in {"api", "worker", "launcher"}:
        raise ValueError(f"unsupported v2 process role: {role}")
    return EpochRegistration(
        epoch_id=str(uuid.uuid4()),
        token=secrets.token_urlsafe(32),
        role=role,  # type: ignore[arg-type]
        pid=pid,
    )


def _child_environment(
    data_root: Path,
    role: str,
    registration: EpochRegistration,
    *,
    log_level: str | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment[DATA_ROOT_ENV] = str(data_root)
    if log_level:
        environment[LOG_LEVEL_ENV] = log_level
    if registration.role != role:
        raise ValueError("child role and epoch registration role differ")
    if role == "api":
        environment.update(
            {
                API_EPOCH_ID_ENV: registration.epoch_id,
                API_EPOCH_TOKEN_ENV: registration.token,
            }
        )
        environment.pop(WORKER_EPOCH_ID_ENV, None)
        environment.pop(WORKER_EPOCH_TOKEN_ENV, None)
    elif role == "worker":
        environment.update(
            {
                WORKER_EPOCH_ID_ENV: registration.epoch_id,
                WORKER_EPOCH_TOKEN_ENV: registration.token,
            }
        )
        environment.pop(API_EPOCH_ID_ENV, None)
        environment.pop(API_EPOCH_TOKEN_ENV, None)
    else:
        raise ValueError(f"unsupported v2 child role: {role}")
    return environment


def _spawn(command: list[str], environment: dict[str, str]) -> subprocess.Popen[str]:
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        command,
        cwd=str(project_root()),
        env=environment,
        text=True,
        creationflags=creation_flags,
    )


def _wait_for_api(
    port: int,
    *,
    expected_epoch_id: str,
    child: subprocess.Popen[str],
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/api/v2/health"
    while time.monotonic() < deadline:
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(f"v2 API exited during startup with code {return_code}")
        try:
            with urlopen(url, timeout=1.0) as response:
                payload = json.loads(response.read())
                if response.status == 200 and payload.get("epochId") == expected_epoch_id:
                    return
        except (OSError, URLError):
            time.sleep(0.1)
    raise RuntimeError(f"v2 API did not become healthy within {timeout_seconds:.0f}s")


def _api_is_healthy(port: int, *, expected_epoch_id: str) -> bool:
    url = f"http://127.0.0.1:{port}/api/v2/health"
    try:
        with urlopen(url, timeout=0.5) as response:
            payload = json.loads(response.read())
    except (OSError, URLError, ValueError):
        return False
    return bool(
        response.status == 200
        and isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("epochId") == expected_epoch_id
    )


def _api_health_requires_restart(
    managed: ManagedChild,
    *,
    port: int,
    now: float,
) -> bool:
    if managed.role != "api" or now < managed.next_health_check_at:
        return False
    managed.next_health_check_at = now + API_HEALTH_CHECK_INTERVAL_SECONDS
    if _api_is_healthy(
        port,
        expected_epoch_id=managed.registration.epoch_id,
    ):
        managed.health_failures = 0
        return False
    managed.health_failures += 1
    LOGGER.warning(
        "API 运行期健康检查失败：epoch=%s，连续失败=%s/%s",
        managed.registration.epoch_id[:8],
        managed.health_failures,
        API_HEALTH_FAILURE_LIMIT,
    )
    return managed.health_failures >= API_HEALTH_FAILURE_LIMIT


def _worker_epoch_requires_restart(
    managed: ManagedChild,
    *,
    repository: ProcessEpochRepository,
    now: float,
) -> bool:
    if managed.role != "worker" or now < managed.next_health_check_at:
        return False
    managed.next_health_check_at = now + API_HEALTH_CHECK_INTERVAL_SECONDS
    try:
        return not repository.is_active_epoch(
            role="worker",
            epoch_id=managed.registration.epoch_id,
        )
    except Exception:
        LOGGER.exception(
            "读取 Worker epoch 状态失败，将在下一轮重试：epoch=%s",
            managed.registration.epoch_id[:8],
        )
        return False


def _reset_restart_count_after_stable_run(
    managed: ManagedChild,
    *,
    now: float,
) -> None:
    if (
        managed.restart_count > 0
        and now - managed.ready_at >= RESTART_STABILITY_SECONDS
    ):
        LOGGER.info(
            "%s 子进程已稳定运行 %.0f 秒，连续重启计数清零",
            managed.role.upper(),
            RESTART_STABILITY_SECONDS,
        )
        managed.restart_count = 0


def _wait_for_worker(
    data_root: Path,
    *,
    expected_epoch_id: str,
    child: subprocess.Popen[str],
    timeout_seconds: float = 30.0,
) -> None:
    marker = data_root / "runtime" / "worker-ready.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(f"v2 Worker exited during startup with code {return_code}")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        if payload.get("epochId") == expected_epoch_id:
            return
        time.sleep(0.1)
    raise RuntimeError(f"v2 Worker did not become healthy within {timeout_seconds:.0f}s")


def _stop_children(children: list[subprocess.Popen[str]]) -> None:
    for child in children:
        if child.poll() is None:
            LOGGER.info("正在停止子进程 pid=%s", child.pid)
            child.terminate()
    deadline = time.monotonic() + 5.0
    for child in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            LOGGER.warning("子进程 pid=%s 未按时退出，执行强制停止", child.pid)
            child.kill()
            child.wait(timeout=2.0)


def _probe_payload(data_root: Path, host: str, port: int) -> dict[str, Any]:
    return {
        "role": "launcher",
        "status": "ready",
        "dataRoot": str(data_root),
        "dataRootFingerprint": data_root_fingerprint(data_root),
        "apiCommand": _role_command("api", data_root=data_root, host=host, port=port),
        "workerCommand": _role_command("worker", data_root=data_root, host=host, port=port),
    }


def _reconcile_all_previous_epochs(repository: ProcessEpochRepository) -> None:
    for epoch_id in repository.active_epochs("api"):
        repository.reconcile_dead_api(epoch_id)
    for epoch_id in repository.active_epochs("worker"):
        repository.reconcile_dead_worker(epoch_id)


def _start_child(
    *,
    role: str,
    data_root: Path,
    host: str,
    port: int,
    repository: ProcessEpochRepository,
    child_job: ChildProcessJob,
    restart_count: int,
    log_level: str | None = None,
) -> ManagedChild:
    registration = _new_registration(role)
    repository.register(registration)
    started_at = time.monotonic()
    process: subprocess.Popen[str] | None = None
    LOGGER.info(
        "正在启动 %s 子进程（epoch=%s，restart=%s）",
        role.upper(),
        registration.epoch_id[:8],
        restart_count,
    )
    try:
        process = _spawn(
            _role_command(role, data_root=data_root, host=host, port=port),
            _child_environment(
                data_root,
                role,
                registration,
                log_level=log_level,
            ),
        )
        if not repository.bind_pid(registration, process.pid):
            process.terminate()
            process.wait(timeout=5)
            raise RuntimeError(f"could not bind {role} pid to its epoch")
        child_job.assign(process)
        if role == "api":
            _wait_for_api(
                port,
                expected_epoch_id=registration.epoch_id,
                child=process,
            )
        else:
            _wait_for_worker(
                data_root,
                expected_epoch_id=registration.epoch_id,
                child=process,
            )
    except BaseException:
        LOGGER.exception(
            "%s 子进程启动失败（epoch=%s）",
            role.upper(),
            registration.epoch_id[:8],
        )
        if process is not None:
            _stop_children([process])
        if role == "api":
            repository.reconcile_dead_api(registration.epoch_id)
        else:
            repository.reconcile_dead_worker(registration.epoch_id)
        raise
    LOGGER.info(
        "%s 子进程已就绪：pid=%s，epoch=%s，耗时=%.2fs",
        role.upper(),
        process.pid,
        registration.epoch_id[:8],
        time.monotonic() - started_at,
    )
    ready_at = time.monotonic()
    return ManagedChild(
        role=role,
        process=process,
        registration=registration,
        restart_count=restart_count,
        ready_at=ready_at,
        next_health_check_at=ready_at + API_HEALTH_CHECK_INTERVAL_SECONDS,
    )


def _start_child_with_retries(
    *,
    role: str,
    data_root: Path,
    host: str,
    port: int,
    repository: ProcessEpochRepository,
    child_job: ChildProcessJob,
    restart_count: int,
    log_level: str | None = None,
) -> ManagedChild:
    current_restart_count = restart_count
    while current_restart_count <= MAX_CONSECUTIVE_RESTARTS:
        try:
            return _start_child(
                role=role,
                data_root=data_root,
                host=host,
                port=port,
                repository=repository,
                child_job=child_job,
                restart_count=current_restart_count,
                log_level=log_level,
            )
        except Exception:
            current_restart_count += 1
            if current_restart_count > MAX_CONSECUTIVE_RESTARTS:
                break
            LOGGER.warning(
                "%s 子进程启动失败，将执行第 %s/%s 次连续重启",
                role.upper(),
                current_restart_count,
                MAX_CONSECUTIVE_RESTARTS,
            )
            time.sleep(0.25)
    raise RuntimeError(
        f"v2 {role} exceeded {MAX_CONSECUTIVE_RESTARTS} consecutive restarts"
    )


def run_launcher(args: object) -> int:
    data_root = ensure_data_root(resolve_data_root(args.data_dir))
    host = args.host
    port = args.port

    if args.probe:
        print(json.dumps(_probe_payload(data_root, host, port), sort_keys=True))
        return 0

    log_level = args.log_level
    log_path = configure_backend_logging(
        role="launcher",
        data_root=data_root,
        console_level=log_level,
    )
    LOGGER.info(
        "Saber-Translator Backend-First V2 启动中：pid=%s，Python=%s",
        os.getpid(),
        sys.version.split()[0],
    )
    LOGGER.info(
        "运行参数：data_root=%s，监听=%s:%s，日志=%s",
        data_root,
        host,
        port,
        log_path,
    )
    with DataRootLock(data_root):
        LOGGER.info("已取得数据目录单实例锁")
        migration = migrate_database(data_root)
        LOGGER.info(
            "数据库迁移与完整性检查完成：revision=%s，升级前备份=%s",
            migration.upgraded_to,
            "是" if migration.backup_created else "否",
        )
        engine = create_sqlite_engine(database_path_for(data_root))
        repository = ProcessEpochRepository(engine)
        object_storage = AssetStorageService(data_root, engine)
        launcher_registration = _new_registration("launcher", pid=os.getpid())
        repository.register(launcher_registration)
        children: dict[str, ManagedChild] = {}
        try:
            _reconcile_all_previous_epochs(repository)
            LOGGER.info("已完成历史进程租约与中断任务恢复")
            recovered = object_storage.recover_journal()
            integrity = object_storage.scan_integrity()
            LOGGER.info(
                "对象存储检查完成：恢复日志=%s，检查对象=%s，缺失=%s，恢复=%s",
                recovered,
                integrity.checked,
                integrity.missing,
                integrity.restored,
            )

            with ChildProcessJob() as child_job:
                try:
                    for role in ("api", "worker"):
                        children[role] = _start_child_with_retries(
                            role=role,
                            data_root=data_root,
                            host=host,
                            port=port,
                            repository=repository,
                            child_job=child_job,
                            restart_count=0,
                            log_level=log_level,
                        )

                    if not args.no_browser:
                        webbrowser.open_new(f"http://127.0.0.1:{port}/")
                        LOGGER.info(
                            "已请求打开浏览器：http://127.0.0.1:%s/",
                            port,
                        )
                    LOGGER.info(
                        "后端全部就绪：本机 http://127.0.0.1:%s/，局域网监听 %s:%s",
                        port,
                        host,
                        port,
                    )

                    while True:
                        now = time.monotonic()
                        for role, managed in list(children.items()):
                            _reset_restart_count_after_stable_run(
                                managed,
                                now=now,
                            )
                            return_code = managed.process.poll()
                            if (
                                return_code is None
                                and _api_health_requires_restart(
                                    managed,
                                    port=port,
                                    now=now,
                                )
                            ):
                                LOGGER.error(
                                    "API 进程仍存活但健康检查持续失败，"
                                    "先终止旧进程再执行 epoch 恢复：pid=%s，epoch=%s",
                                    managed.process.pid,
                                    managed.registration.epoch_id[:8],
                                )
                                _stop_children([managed.process])
                                return_code = managed.process.poll()
                            if (
                                return_code is None
                                and _worker_epoch_requires_restart(
                                    managed,
                                    repository=repository,
                                    now=now,
                                )
                            ):
                                LOGGER.error(
                                    "Worker epoch 已失效但旧进程仍存活，"
                                    "正在终止旧进程后重启：pid=%s，epoch=%s",
                                    managed.process.pid,
                                    managed.registration.epoch_id[:8],
                                )
                                _stop_children([managed.process])
                                return_code = managed.process.poll()
                            if return_code is None:
                                continue
                            LOGGER.warning(
                                "%s 子进程意外退出：pid=%s，exit_code=%s",
                                role.upper(),
                                managed.process.pid,
                                return_code,
                            )
                            if role == "api":
                                repository.reconcile_dead_api(
                                    managed.registration.epoch_id
                                )
                            else:
                                repository.reconcile_dead_worker(
                                    managed.registration.epoch_id
                                )
                            restart_count = managed.restart_count + 1
                            if restart_count > MAX_CONSECUTIVE_RESTARTS:
                                raise RuntimeError(
                                    f"v2 {role} exceeded "
                                    f"{MAX_CONSECUTIVE_RESTARTS} consecutive restarts"
                                )
                            children[role] = _start_child_with_retries(
                                role=role,
                                data_root=data_root,
                                host=host,
                                port=port,
                                repository=repository,
                                child_job=child_job,
                                restart_count=restart_count,
                                log_level=log_level,
                            )
                        time.sleep(0.25)
                except KeyboardInterrupt:
                    LOGGER.info("收到终止信号，准备关闭后端")
                    return 0
                finally:
                    _stop_children(
                        [managed.process for managed in children.values()]
                    )
                    for managed in children.values():
                        repository.close(managed.registration)
        except BaseException:
            LOGGER.exception("Launcher 运行失败")
            raise
        finally:
            repository.close(launcher_registration)
            engine.dispose()
            LOGGER.info("Launcher 已关闭")
