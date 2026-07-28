"""Minimal Launcher for the isolated v2 API and Worker processes."""

from __future__ import annotations

from dataclasses import dataclass
import json
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


@dataclass(slots=True)
class ManagedChild:
    role: str
    process: subprocess.Popen[str]
    registration: EpochRegistration
    restart_count: int = 0


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
    registration: EpochRegistration | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment[DATA_ROOT_ENV] = str(data_root)
    identity = registration or _new_registration(role)
    if identity.role != role:
        raise ValueError("child role and epoch registration role differ")
    if role == "api":
        environment.update(
            {
                API_EPOCH_ID_ENV: identity.epoch_id,
                API_EPOCH_TOKEN_ENV: identity.token,
            }
        )
        environment.pop(WORKER_EPOCH_ID_ENV, None)
        environment.pop(WORKER_EPOCH_TOKEN_ENV, None)
    elif role == "worker":
        environment.update(
            {
                WORKER_EPOCH_ID_ENV: identity.epoch_id,
                WORKER_EPOCH_TOKEN_ENV: identity.token,
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
            child.terminate()
    deadline = time.monotonic() + 5.0
    for child in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
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
) -> ManagedChild:
    registration = _new_registration(role)
    repository.register(registration)
    try:
        process = _spawn(
            _role_command(role, data_root=data_root, host=host, port=port),
            _child_environment(data_root, role, registration),
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
        if role == "api":
            repository.reconcile_dead_api(registration.epoch_id)
        else:
            repository.reconcile_dead_worker(registration.epoch_id)
        raise
    return ManagedChild(
        role=role,
        process=process,
        registration=registration,
        restart_count=restart_count,
    )


def run_launcher(args: object) -> int:
    data_root = ensure_data_root(resolve_data_root(getattr(args, "data_dir", None)))
    host = str(getattr(args, "host", "0.0.0.0"))
    port = int(getattr(args, "port", 5000))

    if getattr(args, "probe", False):
        print(json.dumps(_probe_payload(data_root, host, port), sort_keys=True))
        return 0

    with DataRootLock(data_root):
        migrate_database(data_root)
        engine = create_sqlite_engine(database_path_for(data_root))
        repository = ProcessEpochRepository(engine)
        object_storage = AssetStorageService(data_root, engine)
        launcher_registration = _new_registration("launcher", pid=os.getpid())
        repository.register(launcher_registration)
        children: dict[str, ManagedChild] = {}
        try:
            _reconcile_all_previous_epochs(repository)
            object_storage.recover_journal()
            object_storage.scan_integrity()

            with ChildProcessJob() as child_job:
                try:
                    for role in ("api", "worker"):
                        children[role] = _start_child(
                            role=role,
                            data_root=data_root,
                            host=host,
                            port=port,
                            repository=repository,
                            child_job=child_job,
                            restart_count=0,
                        )

                    if not getattr(args, "no_browser", False):
                        webbrowser.open_new(f"http://127.0.0.1:{port}/")

                    while True:
                        for role, managed in list(children.items()):
                            return_code = managed.process.poll()
                            if return_code is None:
                                continue
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
                            children[role] = _start_child(
                                role=role,
                                data_root=data_root,
                                host=host,
                                port=port,
                                repository=repository,
                                child_job=child_job,
                                restart_count=restart_count,
                            )
                        time.sleep(0.25)
                except KeyboardInterrupt:
                    return 0
                finally:
                    _stop_children(
                        [managed.process for managed in children.values()]
                    )
                    for managed in children.values():
                        repository.close(managed.registration)
        finally:
            repository.close(launcher_registration)
            engine.dispose()
