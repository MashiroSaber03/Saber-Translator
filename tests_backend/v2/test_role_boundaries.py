from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import psutil
import pytest
from sqlalchemy import update

from src.backend_v2.launcher.entrypoint import _child_environment
from src.backend_v2.runtime_identity import (
    API_EPOCH_ID_ENV,
    API_EPOCH_TOKEN_ENV,
    WORKER_EPOCH_ID_ENV,
    WORKER_EPOCH_TOKEN_ENV,
)
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.schema import process_epochs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "saber_v2.py"


def _clean_role_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        API_EPOCH_ID_ENV,
        API_EPOCH_TOKEN_ENV,
        WORKER_EPOCH_ID_ENV,
        WORKER_EPOCH_TOKEN_ENV,
    ):
        environment.pop(name, None)
    return environment


def _run_probe(role: str, data_root: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--role",
            role,
            "--data-dir",
            str(data_root),
            "--test-mode",
            "--probe",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_role_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until(predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("condition did not become true before timeout")


def test_api_probe_loads_only_v2_routes_and_no_worker_modules(tmp_path: Path) -> None:
    result = _run_probe("api", tmp_path / "api")

    assert result["role"] == "api"
    assert result["forbiddenModules"] == []
    routes = result["routes"]
    assert "/api/v2/health" in routes
    assert "/api/v2/openapi.json" in routes
    assert all(route.startswith("/api/v2/") for route in routes)


def test_worker_and_launcher_resolve_the_same_explicit_data_root(tmp_path: Path) -> None:
    worker = _run_probe("worker", tmp_path / "shared")
    launcher = _run_probe("launcher", tmp_path / "shared")

    assert worker["dataRootFingerprint"] == launcher["dataRootFingerprint"]
    assert launcher["apiCommand"][0] == sys.executable
    assert launcher["workerCommand"][0] == sys.executable


@pytest.mark.parametrize("role", ["api", "worker"])
def test_direct_production_role_startup_requires_launcher_identity(
    role: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--role",
            role,
            "--data-dir",
            str(tmp_path / role),
            "--probe",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_role_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "Launcher-issued epoch identity" in completed.stderr


def test_launcher_exposes_only_the_target_roles_secret(tmp_path: Path) -> None:
    polluted = _clean_role_environment()
    polluted.update(
        {
            API_EPOCH_ID_ENV: "stale-api-id",
            API_EPOCH_TOKEN_ENV: "stale-api-token",
            WORKER_EPOCH_ID_ENV: "stale-worker-id",
            WORKER_EPOCH_TOKEN_ENV: "stale-worker-token",
        }
    )
    original = os.environ.copy()
    os.environ.clear()
    os.environ.update(polluted)
    try:
        api = _child_environment(tmp_path, "api")
        worker = _child_environment(tmp_path, "worker")
    finally:
        os.environ.clear()
        os.environ.update(original)

    assert API_EPOCH_ID_ENV in api and API_EPOCH_TOKEN_ENV in api
    assert WORKER_EPOCH_ID_ENV not in api and WORKER_EPOCH_TOKEN_ENV not in api
    assert WORKER_EPOCH_ID_ENV in worker and WORKER_EPOCH_TOKEN_ENV in worker
    assert API_EPOCH_ID_ENV not in worker and API_EPOCH_TOKEN_ENV not in worker
    with pytest.raises(ValueError):
        _child_environment(tmp_path, "renderer")


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
def test_launcher_health_and_kill_on_close(tmp_path: Path) -> None:
    port = _free_port()
    data_root = tmp_path / "runtime"
    process = subprocess.Popen(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--role",
            "launcher",
            "--data-dir",
            str(data_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-browser",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_role_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    child_pids: list[int] = []
    try:
        def healthy() -> bool:
            try:
                with urlopen(
                    f"http://127.0.0.1:{port}/api/v2/health",
                    timeout=0.5,
                ) as response:
                    return response.status == 200
            except OSError:
                return False

        _wait_until(healthy, timeout=30)
        marker_path = data_root / "runtime" / "worker-ready.json"
        _wait_until(marker_path.exists)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["dataRootFingerprint"]

        launcher = psutil.Process(process.pid)
        child_pids = [child.pid for child in launcher.children(recursive=True)]
        assert int(marker["pid"]) in child_pids
        assert len(child_pids) >= 2
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)

    def all_children_gone() -> bool:
        return all(not psutil.pid_exists(pid) for pid in child_pids)

    _wait_until(all_children_gone, timeout=10)


@pytest.mark.skipif(os.name != "nt", reason="Windows process supervision integration")
def test_worker_self_fences_and_launcher_restarts_it_without_restarting_api(
    tmp_path: Path,
) -> None:
    port = _free_port()
    data_root = tmp_path / "runtime"
    process = subprocess.Popen(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--role",
            "launcher",
            "--data-dir",
            str(data_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-browser",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_role_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    descendants: list[int] = []
    try:
        marker_path = data_root / "runtime" / "worker-ready.json"

        def initial_state_ready() -> bool:
            if not marker_path.exists():
                return False
            try:
                with urlopen(
                    f"http://127.0.0.1:{port}/api/v2/health",
                    timeout=0.5,
                ) as response:
                    return response.status == 200
            except OSError:
                return False

        _wait_until(initial_state_ready, timeout=30)
        initial_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        with urlopen(
            f"http://127.0.0.1:{port}/api/v2/health",
            timeout=1,
        ) as response:
            api_epoch = json.loads(response.read())["epochId"]

        engine = create_sqlite_engine(database_path_for(data_root))
        with engine.begin() as connection:
            connection.execute(
                update(process_epochs)
                .where(process_epochs.c.id == initial_marker["epochId"])
                .values(status="lost")
            )
        engine.dispose()

        def worker_restarted() -> bool:
            try:
                current = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            return current.get("epochId") != initial_marker["epochId"]

        _wait_until(worker_restarted, timeout=20)
        _wait_until(lambda: not psutil.pid_exists(int(initial_marker["pid"])), timeout=10)
        with urlopen(
            f"http://127.0.0.1:{port}/api/v2/health",
            timeout=1,
        ) as response:
            assert json.loads(response.read())["epochId"] == api_epoch
        descendants = [
            child.pid
            for child in psutil.Process(process.pid).children(recursive=True)
        ]
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)

    _wait_until(
        lambda: all(not psutil.pid_exists(pid) for pid in descendants),
        timeout=10,
    )
