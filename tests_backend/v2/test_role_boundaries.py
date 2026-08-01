from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import urlopen

import psutil
import pytest
from sqlalchemy import select, update

from src.backend_v2.launcher.entrypoint import (
    API_HEALTH_CHECK_INTERVAL_SECONDS,
    API_HEALTH_FAILURE_LIMIT,
    MAX_CONSECUTIVE_RESTARTS,
    RESTART_STABILITY_SECONDS,
    ManagedChild,
    _api_health_requires_restart,
    _child_environment,
    _reset_restart_count_after_stable_run,
    _start_child_with_retries,
)
from src.backend_v2.runtime_identity import (
    API_EPOCH_ID_ENV,
    API_EPOCH_TOKEN_ENV,
    WORKER_EPOCH_ID_ENV,
    WORKER_EPOCH_TOKEN_ENV,
)
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import EpochRegistration
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
    assert "/api/v2/system/server-info" in routes
    assert "/api/v2/openapi.json" in routes
    assert "/" in routes
    assert "/<path:path>" in routes
    assert "/js/<path:filename>" in routes
    assert "/assets/<path:filename>" in routes
    assert all(
        route.startswith("/api/v2/")
        or route in {
            "/",
            "/<path:path>",
            "/js/<path:filename>",
            "/assets/<path:filename>",
        }
        for route in routes
    )


def test_api_epoch_heartbeat_starts_before_application_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.backend_v2.api import entrypoint
    from src.backend_v2.runtime_identity import RuntimeIdentity

    events: list[str] = []

    class FakeEngine:
        def dispose(self) -> None:
            events.append("engine_disposed")

    class FakeEpochRepository:
        def __init__(self, _engine: object) -> None:
            pass

        def validate(self, **_kwargs: object) -> bool:
            return True

    class FakeHeartbeat:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            events.append("heartbeat_started")

        def stop(self) -> None:
            events.append("heartbeat_stopped")

    class FakeRuntime:
        def close(self) -> None:
            events.append("runtime_closed")

    class FakeUrlMap:
        @staticmethod
        def iter_rules() -> list[SimpleNamespace]:
            return [SimpleNamespace(rule="/api/v2/health")]

    fake_app = SimpleNamespace(
        url_map=FakeUrlMap(),
        extensions={"saber_v2_runtime": FakeRuntime()},
    )

    def create_app(_settings: object) -> object:
        events.append("app_initialized")
        assert events[0] == "heartbeat_started"
        return fake_app

    monkeypatch.setattr(
        entrypoint.RuntimeIdentity,
        "for_api",
        classmethod(
            lambda _cls, **_kwargs: RuntimeIdentity("api-epoch", "token")
        ),
    )
    monkeypatch.setattr(entrypoint, "ProcessEpochRepository", FakeEpochRepository)
    monkeypatch.setattr(entrypoint, "EpochHeartbeat", FakeHeartbeat)
    monkeypatch.setattr(entrypoint, "create_sqlite_engine", lambda _path: FakeEngine())
    monkeypatch.setattr(entrypoint, "create_api_app", create_app)
    monkeypatch.setattr(entrypoint, "loaded_forbidden_api_modules", lambda: [])

    result = entrypoint.run_api(
        SimpleNamespace(
            data_dir=str(tmp_path / "api-heartbeat"),
            probe=True,
            test_mode=False,
            host="127.0.0.1",
            port=5000,
            log_level=None,
        )
    )

    assert result == 0
    assert events == [
        "heartbeat_started",
        "app_initialized",
        "heartbeat_stopped",
        "runtime_closed",
        "engine_disposed",
    ]


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
        api_registration = EpochRegistration(
            epoch_id="api-test",
            token="api-token",
            role="api",
            pid=0,
        )
        worker_registration = EpochRegistration(
            epoch_id="worker-test",
            token="worker-token",
            role="worker",
            pid=0,
        )
        api = _child_environment(tmp_path, "api", api_registration)
        worker = _child_environment(tmp_path, "worker", worker_registration)
    finally:
        os.environ.clear()
        os.environ.update(original)

    assert API_EPOCH_ID_ENV in api and API_EPOCH_TOKEN_ENV in api
    assert WORKER_EPOCH_ID_ENV not in api and WORKER_EPOCH_TOKEN_ENV not in api
    assert WORKER_EPOCH_ID_ENV in worker and WORKER_EPOCH_TOKEN_ENV in worker
    assert API_EPOCH_ID_ENV not in worker and API_EPOCH_TOKEN_ENV not in worker
    with pytest.raises(ValueError):
        _child_environment(tmp_path, "renderer", api_registration)


def test_launcher_requires_repeated_api_health_failures_before_restart(
    monkeypatch,
) -> None:
    registration = EpochRegistration(
        epoch_id="api-epoch",
        token="token",
        role="api",
        pid=123,
    )
    managed = ManagedChild(
        role="api",
        process=object(),  # type: ignore[arg-type]
        registration=registration,
        ready_at=0.0,
        next_health_check_at=0.0,
    )
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint._api_is_healthy",
        lambda _port, *, expected_epoch_id: False,
    )

    for failure in range(1, API_HEALTH_FAILURE_LIMIT):
        now = failure * API_HEALTH_CHECK_INTERVAL_SECONDS
        assert not _api_health_requires_restart(managed, port=5000, now=now)
        assert managed.health_failures == failure

    now = API_HEALTH_FAILURE_LIMIT * API_HEALTH_CHECK_INTERVAL_SECONDS
    assert _api_health_requires_restart(managed, port=5000, now=now)
    assert managed.health_failures == API_HEALTH_FAILURE_LIMIT


def test_launcher_resets_only_stable_consecutive_restart_count() -> None:
    managed = ManagedChild(
        role="worker",
        process=object(),  # type: ignore[arg-type]
        registration=EpochRegistration(
            epoch_id="worker-epoch",
            token="token",
            role="worker",
            pid=456,
        ),
        restart_count=2,
        ready_at=100.0,
    )

    _reset_restart_count_after_stable_run(
        managed,
        now=100.0 + RESTART_STABILITY_SECONDS - 0.1,
    )
    assert managed.restart_count == 2

    _reset_restart_count_after_stable_run(
        managed,
        now=100.0 + RESTART_STABILITY_SECONDS,
    )
    assert managed.restart_count == 0


def test_launcher_retries_child_startup_up_to_the_consecutive_limit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    attempts: list[int] = []
    expected = object()

    def fake_start_child(**kwargs):
        attempts.append(int(kwargs["restart_count"]))
        if len(attempts) < 3:
            raise RuntimeError("startup failed")
        return expected

    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint._start_child",
        fake_start_child,
    )
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint.time.sleep",
        lambda _seconds: None,
    )

    result = _start_child_with_retries(
        role="api",
        data_root=tmp_path,
        host="127.0.0.1",
        port=5000,
        repository=object(),  # type: ignore[arg-type]
        child_job=object(),  # type: ignore[arg-type]
        restart_count=0,
    )

    assert result is expected
    assert attempts == [0, 1, 2]


def test_launcher_stops_after_the_consecutive_startup_retry_limit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    attempts: list[int] = []

    def fail_start_child(**kwargs):
        attempts.append(int(kwargs["restart_count"]))
        raise RuntimeError("startup failed")

    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint._start_child",
        fail_start_child,
    )
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="exceeded"):
        _start_child_with_retries(
            role="worker",
            data_root=tmp_path,
            host="127.0.0.1",
            port=5000,
            repository=object(),  # type: ignore[arg-type]
            child_job=object(),  # type: ignore[arg-type]
            restart_count=0,
        )

    assert attempts == list(range(MAX_CONSECUTIVE_RESTARTS + 1))


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


@pytest.mark.skipif(os.name != "nt", reason="Windows process supervision integration")
def test_api_self_fences_and_launcher_restarts_it_without_restarting_worker(
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

        def initial_state() -> tuple[str, dict[str, object]] | None:
            if not marker_path.exists():
                return None
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                with urlopen(
                    f"http://127.0.0.1:{port}/api/v2/health",
                    timeout=0.5,
                ) as response:
                    payload = json.loads(response.read())
            except (OSError, json.JSONDecodeError):
                return None
            if response.status != 200:
                return None
            return str(payload["epochId"]), marker

        state: tuple[str, dict[str, object]] | None = None

        def capture_initial_state() -> bool:
            nonlocal state
            state = initial_state()
            return state is not None

        _wait_until(capture_initial_state, timeout=30)
        assert state is not None
        initial_api_epoch, initial_worker_marker = state

        engine = create_sqlite_engine(database_path_for(data_root))
        with engine.begin() as connection:
            initial_api_pid = int(
                connection.execute(
                    select(process_epochs.c.pid).where(
                        process_epochs.c.id == initial_api_epoch
                    )
                ).scalar_one()
            )
            connection.execute(
                update(process_epochs)
                .where(process_epochs.c.id == initial_api_epoch)
                .values(status="lost")
            )
        engine.dispose()

        replacement_epoch: str | None = None

        def api_restarted() -> bool:
            nonlocal replacement_epoch
            try:
                with urlopen(
                    f"http://127.0.0.1:{port}/api/v2/health",
                    timeout=0.5,
                ) as response:
                    payload = json.loads(response.read())
            except (OSError, json.JSONDecodeError):
                return False
            replacement_epoch = str(payload.get("epochId", ""))
            return response.status == 200 and replacement_epoch != initial_api_epoch

        _wait_until(api_restarted, timeout=20)
        _wait_until(lambda: not psutil.pid_exists(initial_api_pid), timeout=10)
        current_worker_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert current_worker_marker["epochId"] == initial_worker_marker["epochId"]
        assert current_worker_marker["pid"] == initial_worker_marker["pid"]
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
