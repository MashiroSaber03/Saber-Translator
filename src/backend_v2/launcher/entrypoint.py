"""Minimal Launcher for the isolated v2 API and Worker processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen
import uuid
import webbrowser

import psutil

from src.backend_v2.auth.credential_broker import (
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    CredentialLeaseBroker,
)
from src.backend_v2.logging_config import (
    LOG_LEVEL_ENV,
    STREAM_FRAME_ENV,
    configure_backend_logging,
)
from src.backend_v2.local_models import normalize_resident_models
from src.backend_v2.browser_extension.auth import (
    BROWSER_EXTENSION_ENABLED_ENV,
    BROWSER_EXTENSION_TOKEN_ENV,
)
from src.backend_v2.paths import (
    DATA_ROOT_ENV,
    data_root_fingerprint,
    ensure_data_root,
    project_root,
    resolve_data_root,
)
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.runtime_identity import (
    API_EPOCH_ID_ENV,
    API_EPOCH_TOKEN_ENV,
    LAUNCHER_PID_ENV,
    INTERNAL_HEALTH_TOKEN_HEADER,
    WORKER_RECYCLE_EXIT_CODE,
    WORKER_EPOCH_ID_ENV,
    WORKER_EPOCH_TOKEN_ENV,
)
from src.backend_v2.runtime_profile import (
    PROFILE_ENV,
    PUBLIC_HOST_ENV,
    resolve_public_host,
    resolve_runtime_profile,
    validate_profile_bind_host,
)
from src.backend_v2.launcher.windows_job import ChildProcessJob
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import (
    create_sqlite_engine,
    database_path_for,
    is_sqlite_busy_error,
)
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.lifecycle import initialize_database
from src.backend_v2.storage.single_instance import DataRootLock
from src.shared.user_logging import STREAM_FRAME_PREFIX, inline_log_text, user_log


MAX_CONSECUTIVE_RESTARTS = 3
API_HEALTH_CHECK_INTERVAL_SECONDS = 1.0
API_HEALTH_FAILURE_LIMIT = 3
RESTART_STABILITY_SECONDS = 30.0
PREVIOUS_CHILD_EXIT_TIMEOUT_SECONDS = 5.0
TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT_ENV = "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"
WORKER_CUDNN_V8_API_LRU_CACHE_LIMIT = "1000"
RESIDENT_MODEL_WORKER_READY_TIMEOUT_SECONDS = 600.0
LOGGER = logging.getLogger("saber.launcher")


class LauncherState(str, Enum):
    """Stable service states shared by the CLI and desktop shell."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"


class _LauncherStopRequested(Exception):
    """Internal control flow used to cancel a startup wait promptly."""


@dataclass(frozen=True, slots=True)
class LauncherConfig:
    data_root: Path
    host: str
    port: int
    profile: str = "local"
    log_level: str | None = None
    open_browser: bool = False
    resident_models: tuple[str, ...] = ()
    browser_extension_enabled: bool = False
    browser_extension_token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resident_models",
            normalize_resident_models(self.resident_models),
        )


@dataclass(frozen=True, slots=True)
class LauncherStatus:
    state: LauncherState
    message: str
    api_pid: int | None = None
    worker_pid: int | None = None


StatusCallback = Callable[[LauncherStatus], None]
ChildOutputCallback = Callable[[str, str], None]


@dataclass(slots=True)
class ManagedChild:
    role: str
    process: subprocess.Popen[str]
    registration: EpochRegistration
    restart_count: int = 0
    ready_at: float = 0.0
    next_health_check_at: float = 0.0
    health_failures: int = 0


def _role_command(
    role: str,
    *,
    data_root: Path,
    host: str,
    port: int,
    profile: str,
    resident_models: tuple[str, ...] = (),
) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [sys.executable]
    else:
        command = [sys.executable, str(project_root() / "saber_v2.py")]
    role_command = [
        *command,
        "--role",
        role,
        "--data-dir",
        str(data_root),
        "--host",
        host,
        "--port",
        str(port),
        "--profile",
        profile,
    ]
    if role == "worker":
        for model_id in normalize_resident_models(resident_models):
            role_command.extend(("--resident-model", model_id))
    return role_command


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
    profile: str = "local",
    log_level: str | None = None,
    credential_broker_url: str | None = None,
    credential_broker_token: str | None = None,
    stream_frames: bool = False,
    browser_extension_enabled: bool = False,
    browser_extension_token: str = "",
) -> dict[str, str]:
    environment = os.environ.copy()
    # The desktop supervisor decodes the captured pipe as UTF-8.  Windows can
    # otherwise make redirected Python stdout use the active ANSI code page,
    # which irreversibly corrupts Chinese log messages in the GUI.
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment[DATA_ROOT_ENV] = str(data_root)
    environment[LAUNCHER_PID_ENV] = str(os.getpid())
    environment[PROFILE_ENV] = profile
    if log_level:
        environment[LOG_LEVEL_ENV] = log_level
    if stream_frames:
        environment[STREAM_FRAME_ENV] = "1"
    else:
        environment.pop(STREAM_FRAME_ENV, None)
    if credential_broker_url and credential_broker_token:
        environment[BROKER_URL_ENV] = credential_broker_url
        environment[BROKER_TOKEN_ENV] = credential_broker_token
    else:
        environment.pop(BROKER_URL_ENV, None)
        environment.pop(BROKER_TOKEN_ENV, None)
    if registration.role != role:
        raise ValueError("child role and epoch registration role differ")
    if role == "api" and browser_extension_enabled:
        if not browser_extension_token:
            raise ValueError("enabled browser extension integration requires a token")
        environment[BROWSER_EXTENSION_ENABLED_ENV] = "1"
        environment[BROWSER_EXTENSION_TOKEN_ENV] = browser_extension_token
    else:
        environment.pop(BROWSER_EXTENSION_ENABLED_ENV, None)
        environment.pop(BROWSER_EXTENSION_TOKEN_ENV, None)
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
                # 48px OCR uses variable-width batches. Bound cuDNN's host-side
                # execution-plan cache before the Worker imports PyTorch.
                TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT_ENV: (
                    WORKER_CUDNN_V8_API_LRU_CACHE_LIMIT
                ),
            }
        )
        environment.pop(API_EPOCH_ID_ENV, None)
        environment.pop(API_EPOCH_TOKEN_ENV, None)
    else:
        raise ValueError(f"unsupported v2 child role: {role}")
    return environment


def _spawn(
    command: list[str],
    environment: dict[str, str],
    *,
    capture_output: bool = False,
) -> subprocess.Popen[str]:
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        command,
        cwd=str(project_root()),
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        bufsize=1,
        creationflags=creation_flags,
    )


def _start_output_reader(
    process: subprocess.Popen[str],
    *,
    role: str,
    callback: ChildOutputCallback | None,
) -> None:
    if callback is None or process.stdout is None:
        return

    def read_output() -> None:
        try:
            for line in process.stdout:
                raw_line = line.rstrip("\r\n")
                rendered = (
                    raw_line
                    if raw_line.startswith(STREAM_FRAME_PREFIX)
                    else redact_sensitive_text(raw_line, redact_paths=False)
                )
                if rendered:
                    try:
                        callback(role, rendered)
                    except Exception:
                        LOGGER.exception("转发 %s 子进程日志失败", role.upper())
        finally:
            process.stdout.close()

    thread = threading.Thread(
        target=read_output,
        name=f"saber-{role}-output",
        daemon=True,
    )
    thread.start()


def _raise_if_stop_requested(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise _LauncherStopRequested


def _read_api_health(
    port: int,
    *,
    expected_epoch_token: str,
    timeout_seconds: float,
) -> tuple[int, object]:
    request = Request(
        f"http://127.0.0.1:{port}/api/v2/health",
        headers={INTERNAL_HEALTH_TOKEN_HEADER: expected_epoch_token},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.status, json.loads(response.read())


def _wait_for_api(
    port: int,
    *,
    expected_epoch_id: str,
    expected_epoch_token: str,
    child: subprocess.Popen[str],
    timeout_seconds: float = 30.0,
    stop_event: threading.Event | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _raise_if_stop_requested(stop_event)
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(f"v2 API exited during startup with code {return_code}")
        try:
            status, payload = _read_api_health(
                port,
                expected_epoch_token=expected_epoch_token,
                timeout_seconds=1.0,
            )
            if (
                status == 200
                and isinstance(payload, dict)
                and payload.get("epochId") == expected_epoch_id
            ):
                return
        except (OSError, URLError, ValueError):
            if stop_event is None:
                time.sleep(0.1)
            elif stop_event.wait(0.1):
                raise _LauncherStopRequested
    raise RuntimeError(f"v2 API did not become healthy within {timeout_seconds:.0f}s")


def _api_is_healthy(
    port: int,
    *,
    expected_epoch_id: str,
    expected_epoch_token: str,
) -> bool:
    try:
        status, payload = _read_api_health(
            port,
            expected_epoch_token=expected_epoch_token,
            timeout_seconds=0.5,
        )
    except (OSError, URLError, ValueError):
        return False
    return bool(
        status == 200
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
        expected_epoch_token=managed.registration.token,
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


def _try_reconcile_dead_child(
    repository: ProcessEpochRepository,
    *,
    role: str,
    epoch_id: str,
) -> bool:
    try:
        if role == "api":
            repository.reconcile_dead_api(epoch_id)
        else:
            repository.reconcile_dead_worker(epoch_id)
    except Exception as error:
        if not is_sqlite_busy_error(error):
            raise
        LOGGER.debug(
            "%s 退出清理遇到 SQLite 写锁竞争，将在下一轮重试：epoch=%s",
            role.upper(),
            epoch_id[:8],
        )
        return False
    return True


def _reset_restart_count_after_stable_run(
    managed: ManagedChild,
    *,
    now: float,
) -> None:
    if (
        managed.restart_count > 0
        and now - managed.ready_at >= RESTART_STABILITY_SECONDS
    ):
        LOGGER.debug(
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
    stop_event: threading.Event | None = None,
) -> None:
    marker = data_root / "runtime" / "worker-ready.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _raise_if_stop_requested(stop_event)
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(f"v2 Worker exited during startup with code {return_code}")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if stop_event is None:
                time.sleep(0.1)
            elif stop_event.wait(0.1):
                raise _LauncherStopRequested
            continue
        if payload.get("epochId") == expected_epoch_id:
            return
        if stop_event is None:
            time.sleep(0.1)
        elif stop_event.wait(0.1):
            raise _LauncherStopRequested
    raise RuntimeError(f"v2 Worker did not become healthy within {timeout_seconds:.0f}s")


def _stop_children(children: list[subprocess.Popen[str]]) -> None:
    descendants: list[psutil.Process] = []
    for child in children:
        if child.poll() is None:
            LOGGER.debug("正在停止子进程 pid=%s", child.pid)
            try:
                child_descendants = psutil.Process(child.pid).children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                child_descendants = []
            descendants.extend(child_descendants)
            for descendant in reversed(child_descendants):
                try:
                    descendant.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if child.poll() is None:
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
    if descendants:
        _, alive = psutil.wait_procs(descendants, timeout=2.0)
        for descendant in alive:
            try:
                descendant.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if alive:
            psutil.wait_procs(alive, timeout=2.0)


def _probe_payload(
    data_root: Path,
    host: str,
    port: int,
    profile: str = "local",
    resident_models: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "role": "launcher",
        "status": "ready",
        "dataRoot": str(data_root),
        "dataRootFingerprint": data_root_fingerprint(data_root),
        "profile": profile,
        "apiCommand": _role_command(
            "api", data_root=data_root, host=host, port=port, profile=profile
        ),
        "workerCommand": _role_command(
            "worker",
            data_root=data_root,
            host=host,
            port=port,
            profile=profile,
            resident_models=resident_models,
        ),
    }


def _is_expected_previous_child(pid: int, *, role: str, data_root: Path) -> bool:
    if pid <= 0:
        return False
    try:
        command = psutil.Process(pid).cmdline()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True
    try:
        role_index = command.index("--role")
        data_index = command.index("--data-dir")
    except ValueError:
        return False
    if role_index + 1 >= len(command) or command[role_index + 1] != role:
        return False
    if data_index + 1 >= len(command):
        return False
    try:
        child_data_root = Path(command[data_index + 1]).expanduser().resolve()
    except OSError:
        return False
    return child_data_root == data_root


def _wait_for_previous_children_to_exit(
    repository: ProcessEpochRepository,
    *,
    data_root: Path,
    stop_event: threading.Event,
) -> None:
    previous = [
        (role, epoch_id, pid)
        for role in ("api", "worker")
        for epoch_id, pid in repository.active_epoch_processes(role)
        if _is_expected_previous_child(pid, role=role, data_root=data_root)
    ]
    if not previous:
        return
    deadline = time.monotonic() + PREVIOUS_CHILD_EXIT_TIMEOUT_SECONDS
    while previous and time.monotonic() < deadline:
        if stop_event.wait(0.05):
            raise _LauncherStopRequested
        previous = [
            item
            for item in previous
            if _is_expected_previous_child(
                item[2],
                role=item[0],
                data_root=data_root,
            )
        ]
    if previous:
        rendered = ", ".join(
            f"{role} pid={pid} epoch={epoch_id[:8]}"
            for role, epoch_id, pid in previous
        )
        raise RuntimeError(f"previous backend child did not exit: {rendered}")


def _reconcile_all_previous_epochs(repository: ProcessEpochRepository) -> None:
    for epoch_id in repository.active_epochs("api"):
        repository.reconcile_dead_api(epoch_id)
    for epoch_id in repository.active_epochs("worker"):
        repository.reconcile_dead_worker(epoch_id)
    recovered = repository.reconcile_orphaned_work()
    if recovered:
        LOGGER.warning(
            "启动时已收敛 %s 个非活动进程遗留的运行中工作",
            len(recovered),
        )


def _start_child(
    *,
    role: str,
    data_root: Path,
    host: str,
    port: int,
    profile: str,
    repository: ProcessEpochRepository,
    child_job: ChildProcessJob,
    restart_count: int,
    log_level: str | None = None,
    credential_broker_url: str | None = None,
    credential_broker_token: str | None = None,
    resident_models: tuple[str, ...] = (),
    output_callback: ChildOutputCallback | None = None,
    stop_event: threading.Event | None = None,
    browser_extension_enabled: bool = False,
    browser_extension_token: str = "",
) -> ManagedChild:
    _raise_if_stop_requested(stop_event)
    registration = _new_registration(role)
    started_at = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        repository.register(registration)
        LOGGER.debug(
            "正在启动 %s 子进程（epoch=%s，restart=%s）",
            role.upper(),
            registration.epoch_id[:8],
            restart_count,
        )
        process = _spawn(
            _role_command(
                role,
                data_root=data_root,
                host=host,
                port=port,
                profile=profile,
                resident_models=resident_models,
            ),
            _child_environment(
                data_root,
                role,
                registration,
                profile=profile,
                log_level=log_level,
                credential_broker_url=credential_broker_url,
                credential_broker_token=credential_broker_token,
                stream_frames=output_callback is not None,
                browser_extension_enabled=browser_extension_enabled,
                browser_extension_token=browser_extension_token,
            ),
            capture_output=output_callback is not None,
        )
        _start_output_reader(
            process,
            role=role,
            callback=output_callback,
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
                expected_epoch_token=registration.token,
                child=process,
                stop_event=stop_event,
            )
        else:
            _wait_for_worker(
                data_root,
                expected_epoch_id=registration.epoch_id,
                child=process,
                timeout_seconds=(
                    RESIDENT_MODEL_WORKER_READY_TIMEOUT_SECONDS
                    if resident_models
                    else 30.0
                ),
                stop_event=stop_event,
            )
    except BaseException as error:
        if isinstance(error, _LauncherStopRequested):
            LOGGER.debug(
                "正在取消 %s 子进程启动（epoch=%s）",
                role.upper(),
                registration.epoch_id[:8],
            )
        else:
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
    LOGGER.debug(
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
    profile: str = "local",
    repository: ProcessEpochRepository,
    child_job: ChildProcessJob,
    restart_count: int,
    log_level: str | None = None,
    credential_broker_url: str | None = None,
    credential_broker_token: str | None = None,
    resident_models: tuple[str, ...] = (),
    output_callback: ChildOutputCallback | None = None,
    stop_event: threading.Event | None = None,
    browser_extension_enabled: bool = False,
    browser_extension_token: str = "",
) -> ManagedChild:
    current_restart_count = restart_count
    while current_restart_count <= MAX_CONSECUTIVE_RESTARTS:
        _raise_if_stop_requested(stop_event)
        try:
            return _start_child(
                role=role,
                data_root=data_root,
                host=host,
                port=port,
                profile=profile,
                repository=repository,
                child_job=child_job,
                restart_count=current_restart_count,
                log_level=log_level,
                credential_broker_url=credential_broker_url,
                credential_broker_token=credential_broker_token,
                resident_models=resident_models,
                output_callback=output_callback,
                stop_event=stop_event,
                browser_extension_enabled=browser_extension_enabled,
                browser_extension_token=browser_extension_token,
            )
        except _LauncherStopRequested:
            raise
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
            if stop_event is None:
                time.sleep(0.25)
            elif stop_event.wait(0.25):
                raise _LauncherStopRequested
    raise RuntimeError(
        f"v2 {role} exceeded {MAX_CONSECUTIVE_RESTARTS} consecutive restarts"
    )


class LauncherSupervisor:
    """UI-independent owner for the complete API/Worker lifecycle."""

    def __init__(
        self,
        config: LauncherConfig,
        *,
        status_callback: StatusCallback | None = None,
        output_callback: ChildOutputCallback | None = None,
    ) -> None:
        self.config = config
        self._status_callback = status_callback
        self._output_callback = output_callback
        self._stop_event = threading.Event()
        self._run_lock = threading.Lock()
        self._status = LauncherStatus(LauncherState.STOPPED, "后端未启动")

    @property
    def status(self) -> LauncherStatus:
        return self._status

    def request_stop(self) -> None:
        self._stop_event.set()

    def _publish(
        self,
        state: LauncherState,
        message: str,
        children: dict[str, ManagedChild] | None = None,
    ) -> None:
        children = children or {}
        status = LauncherStatus(
            state=state,
            message=message,
            api_pid=(children.get("api").process.pid if children.get("api") else None),
            worker_pid=(
                children.get("worker").process.pid if children.get("worker") else None
            ),
        )
        self._status = status
        if self._status_callback is not None:
            self._status_callback(status)

    def run(self) -> int:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("launcher supervisor is already running")
        clean_exit = False
        children: dict[str, ManagedChild] = {}
        config = self.config
        credential_broker: CredentialLeaseBroker | None = None
        try:
            self._publish(LauncherState.STARTING, "正在初始化后端")
            _raise_if_stop_requested(self._stop_event)
            with DataRootLock(config.data_root):
                LOGGER.debug("已取得数据目录单实例锁")
                storage = initialize_database(
                    config.data_root,
                    profile_name=config.profile,
                )
                LOGGER.debug(
                    "数据库初始化与完整性检查完成：revision=%s，新建=%s",
                    storage.schema_revision,
                    "是" if storage.created else "否",
                )
                engine = create_sqlite_engine(database_path_for(config.data_root))
                repository = ProcessEpochRepository(engine)
                object_storage = AssetStorageService(config.data_root, engine)
                launcher_registration = _new_registration("launcher", pid=os.getpid())
                repository.register(launcher_registration)
                try:
                    _wait_for_previous_children_to_exit(
                        repository,
                        data_root=config.data_root,
                        stop_event=self._stop_event,
                    )
                    _reconcile_all_previous_epochs(repository)
                    LOGGER.debug("已完成历史进程租约与中断任务恢复")
                    recovered = object_storage.recover_journal()
                    integrity = object_storage.scan_integrity()
                    LOGGER.debug(
                        "对象存储检查完成：恢复日志=%s，检查对象=%s，缺失=%s，恢复=%s",
                        recovered,
                        integrity.checked,
                        integrity.missing,
                        integrity.restored,
                    )

                    if resolve_runtime_profile(config.profile).browser_credentials:
                        credential_broker = CredentialLeaseBroker()
                        credential_broker.start()
                        LOGGER.debug("浏览器密钥内存服务已启动（仅监听 127.0.0.1）")

                    with ChildProcessJob() as child_job:
                        try:
                            for role in ("api", "worker"):
                                children[role] = _start_child_with_retries(
                                    role=role,
                                    data_root=config.data_root,
                                    host=config.host,
                                    port=config.port,
                                    profile=config.profile,
                                    repository=repository,
                                    child_job=child_job,
                                    restart_count=0,
                                    log_level=config.log_level,
                                    credential_broker_url=(
                                        credential_broker.url
                                        if credential_broker is not None
                                        else None
                                    ),
                                    credential_broker_token=(
                                        credential_broker.token
                                        if credential_broker is not None
                                        else None
                                    ),
                                    resident_models=config.resident_models,
                                    output_callback=self._output_callback,
                                    stop_event=self._stop_event,
                                    browser_extension_enabled=(
                                        config.browser_extension_enabled
                                    ),
                                    browser_extension_token=(
                                        config.browser_extension_token
                                    ),
                                )

                            _raise_if_stop_requested(self._stop_event)
                            if config.open_browser:
                                webbrowser.open_new(f"http://127.0.0.1:{config.port}/")
                                LOGGER.debug(
                                    "已请求打开浏览器：http://127.0.0.1:%s/",
                                    config.port,
                                )
                            user_log(
                                "system",
                                f"后端已就绪｜网页 http://127.0.0.1:{config.port}/ ｜"
                                f"监听 {config.host}:{config.port}",
                            )
                            self._publish(
                                LauncherState.RUNNING,
                                "API 与 Worker 运行正常",
                                children,
                            )

                            while not self._stop_event.wait(0.25):
                                now = time.monotonic()
                                for role, managed in list(children.items()):
                                    recovery_was_requested = False
                                    _reset_restart_count_after_stable_run(
                                        managed,
                                        now=now,
                                    )
                                    return_code = managed.process.poll()
                                    if (
                                        return_code is None
                                        and _api_health_requires_restart(
                                            managed,
                                            port=config.port,
                                            now=now,
                                        )
                                    ):
                                        LOGGER.error(
                                            "API 进程仍存活但健康检查持续失败，"
                                            "先终止旧进程再执行 epoch 恢复：pid=%s，epoch=%s",
                                            managed.process.pid,
                                            managed.registration.epoch_id[:8],
                                        )
                                        user_log(
                                            "warning",
                                            "接口进程健康检查持续失败，正在自动恢复",
                                        )
                                        self._publish(
                                            LauncherState.DEGRADED,
                                            "API 健康检查失败，正在恢复",
                                            children,
                                        )
                                        recovery_was_requested = True
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
                                        user_log(
                                            "warning",
                                            "任务执行器运行权已失效，正在自动重启",
                                        )
                                        self._publish(
                                            LauncherState.DEGRADED,
                                            "Worker 状态异常，正在恢复",
                                            children,
                                        )
                                        recovery_was_requested = True
                                        _stop_children([managed.process])
                                        return_code = managed.process.poll()
                                    if return_code is None:
                                        continue
                                    controlled_recycle = (
                                        role == "worker"
                                        and return_code == WORKER_RECYCLE_EXIT_CODE
                                    )
                                    log_child_exit = (
                                        LOGGER.info
                                        if controlled_recycle
                                        else LOGGER.warning
                                    )
                                    log_child_exit(
                                        "%s 子进程%s：pid=%s，exit_code=%s",
                                        role.upper(),
                                        (
                                            "已受控回收"
                                            if controlled_recycle
                                            else "意外退出"
                                        ),
                                        managed.process.pid,
                                        return_code,
                                    )
                                    if not recovery_was_requested:
                                        role_label = (
                                            "接口进程"
                                            if role == "api"
                                            else "任务执行器"
                                        )
                                        if controlled_recycle:
                                            user_log(
                                                "system",
                                                "任务执行器已主动回收卡住的处理器，"
                                                "正在自动恢复",
                                            )
                                        else:
                                            user_log(
                                                "warning",
                                                f"{role_label}意外退出｜"
                                                f"退出码 {return_code}｜正在自动恢复",
                                            )
                                    if not _try_reconcile_dead_child(
                                        repository,
                                        role=role,
                                        epoch_id=managed.registration.epoch_id,
                                    ):
                                        continue
                                    restart_count = managed.restart_count + 1
                                    if restart_count > MAX_CONSECUTIVE_RESTARTS:
                                        raise RuntimeError(
                                            f"v2 {role} exceeded "
                                            f"{MAX_CONSECUTIVE_RESTARTS} consecutive restarts"
                                        )
                                    self._publish(
                                        LauncherState.DEGRADED,
                                        f"{role.upper()} 已退出，正在重启",
                                        children,
                                    )
                                    children[role] = _start_child_with_retries(
                                        role=role,
                                        data_root=config.data_root,
                                        host=config.host,
                                        port=config.port,
                                        profile=config.profile,
                                        repository=repository,
                                        child_job=child_job,
                                        restart_count=restart_count,
                                        log_level=config.log_level,
                                        credential_broker_url=(
                                            credential_broker.url
                                            if credential_broker is not None
                                            else None
                                        ),
                                        credential_broker_token=(
                                            credential_broker.token
                                            if credential_broker is not None
                                            else None
                                        ),
                                        resident_models=config.resident_models,
                                        output_callback=self._output_callback,
                                        stop_event=self._stop_event,
                                        browser_extension_enabled=(
                                            config.browser_extension_enabled
                                        ),
                                        browser_extension_token=(
                                            config.browser_extension_token
                                        ),
                                    )
                                    self._publish(
                                        LauncherState.RUNNING,
                                        "API 与 Worker 运行正常",
                                        children,
                                    )
                            clean_exit = True
                        except KeyboardInterrupt:
                            user_log("system", "收到终止信号，正在关闭后端")
                            clean_exit = True
                        finally:
                            self._publish(
                                LauncherState.STOPPING,
                                "正在停止 API 与 Worker",
                                children,
                            )
                            _stop_children(
                                [managed.process for managed in children.values()]
                            )
                            for managed in children.values():
                                repository.close(managed.registration)
                finally:
                    if credential_broker is not None:
                        credential_broker.close()
                        LOGGER.debug("浏览器密钥内存服务已清空并停止")
                    repository.close(launcher_registration)
                    engine.dispose()
                    user_log("system", "后端已关闭")
        except _LauncherStopRequested:
            clean_exit = True
            user_log("system", "已取消后端启动")
        except BaseException as error:
            self._publish(LauncherState.DEGRADED, f"后端运行失败：{error}")
            LOGGER.exception("Launcher 运行失败")
            user_log(
                "error",
                f"后端运行失败｜{inline_log_text(error)}",
            )
            raise
        finally:
            if clean_exit:
                self._publish(LauncherState.STOPPED, "后端已停止")
            self._run_lock.release()
        return 0


def run_launcher(args: object) -> int:
    profile = resolve_runtime_profile(getattr(args, "profile", "local"))
    resident_models = normalize_resident_models(
        getattr(args, "resident_model", ())
    )
    explicit_data_root = getattr(args, "data_dir", None)
    if profile.name == "public" and not explicit_data_root:
        raise ValueError("public profile requires an explicit --data-dir")
    public_host = resolve_public_host(profile)
    validate_profile_bind_host(profile, args.host)
    if public_host is not None:
        os.environ[PUBLIC_HOST_ENV] = public_host
    os.environ[PROFILE_ENV] = profile.name
    data_root = ensure_data_root(resolve_data_root(explicit_data_root))
    host = args.host
    port = args.port

    if args.probe:
        print(
            json.dumps(
                _probe_payload(
                    data_root,
                    host,
                    port,
                    profile.name,
                    resident_models,
                ),
                sort_keys=True,
            )
        )
        return 0

    log_level = args.log_level
    log_path = configure_backend_logging(
        role="launcher",
        data_root=data_root,
        console_level=log_level,
    )
    user_log(
        "system",
        f"Saber-Translator 正在启动｜Python {sys.version.split()[0]}",
    )
    LOGGER.debug(
        "运行参数：profile=%s，data_root=%s，监听=%s:%s，日志=%s",
        profile.name,
        data_root,
        host,
        port,
        log_path,
    )
    return LauncherSupervisor(
        LauncherConfig(
            data_root=data_root,
            host=host,
            port=port,
            profile=profile.name,
            log_level=log_level,
            open_browser=not args.no_browser,
            resident_models=resident_models,
        )
    ).run()
