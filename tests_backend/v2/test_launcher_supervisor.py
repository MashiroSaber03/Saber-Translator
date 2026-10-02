from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import sys

import psutil
import pytest
from sqlalchemy import select, update

from src.backend_v2.browser_extension.auth import (
    BROWSER_EXTENSION_ENABLED_ENV,
    BROWSER_EXTENSION_TOKEN_ENV,
)
from src.backend_v2.launcher.entrypoint import (
    LauncherConfig,
    LauncherState,
    LauncherSupervisor,
    RESIDENT_MODEL_WORKER_READY_TIMEOUT_SECONDS,
    _child_environment,
    _new_registration,
    _read_api_health,
    _role_command,
    _spawn,
    _start_child,
)
from src.backend_v2.logging_config import STREAM_FRAME_ENV
from src.backend_v2.runtime_identity import INTERNAL_HEALTH_TOKEN_HEADER
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.lifecycle import initialize_database
from src.backend_v2.storage.schema import process_epochs
from src.backend_v2.timestamps import utcnow
from src.storage_migrator.contracts import StorageError
from src.storage_migrator.control import DataRootLock, atomic_json, control_root, registered_process, wait_for_children
from src.storage_migrator.runner import StorageManager


def test_child_process_logs_are_forced_to_utf8() -> None:
    registration = _new_registration("api")

    environment = _child_environment("api", registration)

    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"

    process = _spawn(
        [sys.executable, "-c", "print('中文子进程日志')"],
        environment,
        capture_output=True,
    )
    output, _stderr = process.communicate(timeout=10)

    assert process.returncode == 0
    assert output.strip() == "中文子进程日志"


def test_desktop_captured_children_enable_stream_frames() -> None:
    registration = _new_registration("worker")

    direct = _child_environment("worker", registration)
    captured = _child_environment(
        "worker",
        registration,
        stream_frames=True,
    )

    assert STREAM_FRAME_ENV not in direct
    assert captured[STREAM_FRAME_ENV] == "1"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console creation")
def test_captured_backend_process_has_no_console() -> None:
    import os

    process = _spawn(
        [sys.executable, "-c",
         "import ctypes; print(ctypes.windll.kernel32.GetConsoleWindow())"],
        os.environ.copy(), capture_output=True,
    )
    output, _ = process.communicate(timeout=10)
    assert process.returncode == 0
    assert output.strip() == "0"


def test_browser_extension_secret_is_passed_only_to_the_api() -> None:
    token = "browser-extension-secret"
    api = _child_environment(
        "api",
        _new_registration("api"),
        browser_extension_enabled=True,
        browser_extension_token=token,
    )
    worker = _child_environment(
        "worker",
        _new_registration("worker"),
        browser_extension_enabled=True,
        browser_extension_token=token,
    )

    assert api[BROWSER_EXTENSION_ENABLED_ENV] == "1"
    assert api[BROWSER_EXTENSION_TOKEN_ENV] == token
    assert BROWSER_EXTENSION_ENABLED_ENV not in worker
    assert BROWSER_EXTENSION_TOKEN_ENV not in worker
    with pytest.raises(ValueError, match="requires a token"):
        _child_environment(
            "api",
            _new_registration("api"),
            browser_extension_enabled=True,
        )


def test_launcher_authenticates_its_internal_api_health_request(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b'{"status":"ok","epochId":"api-epoch"}'

    def open_health(request, *, timeout):
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint.urlopen",
        open_health,
    )

    status, payload = _read_api_health(
        5000,
        expected_epoch_token="epoch-token",
        timeout_seconds=0.5,
    )

    assert status == 200
    assert payload == {"status": "ok", "epochId": "api-epoch"}
    assert captured["headers"] == {
        INTERNAL_HEALTH_TOKEN_HEADER.lower(): "epoch-token"
    }
    assert captured["timeout"] == 0.5


def test_launcher_passes_resident_models_only_to_the_worker(tmp_path) -> None:
    config = LauncherConfig(
        data_root=tmp_path,
        host="127.0.0.1",
        port=5000,
        resident_models=("manga_ocr", "detector_yolo", "manga_ocr"),
    )

    worker = _role_command(
        "worker",
        data_root=tmp_path,
        host=config.host,
        port=config.port,
        profile=config.profile,
        resident_models=config.resident_models,
    )
    api = _role_command(
        "api",
        data_root=tmp_path,
        host=config.host,
        port=config.port,
        profile=config.profile,
        resident_models=config.resident_models,
    )

    assert config.resident_models == ("detector_yolo", "manga_ocr")
    assert worker[-4:] == [
        "--resident-model",
        "detector_yolo",
        "--resident-model",
        "manga_ocr",
    ]
    assert "--resident-model" not in api


def test_resident_worker_gets_a_preload_aware_startup_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    waited: dict[str, object] = {}

    class FakeProcess:
        pid = 1234

        @staticmethod
        def poll():
            return None

    class FakeRepository:
        @staticmethod
        def register(_registration) -> None:
            return None

        @staticmethod
        def bind_pid(_registration, _pid) -> bool:
            return True

    class FakeJob:
        @staticmethod
        def assign(_process) -> None:
            return None

    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint._spawn",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint._start_output_reader",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint._wait_for_worker",
        lambda *_args, **kwargs: waited.update(kwargs),
    )

    _start_child(
        role="worker",
        data_root=tmp_path,
        host="127.0.0.1",
        port=5000,
        profile="local",
        repository=FakeRepository(),  # type: ignore[arg-type]
        child_job=FakeJob(),  # type: ignore[arg-type]
        restart_count=0,
        resident_models=("detector_yolo",),
    )

    assert waited["timeout_seconds"] == (
        RESIDENT_MODEL_WORKER_READY_TIMEOUT_SECONDS
    )


def test_stop_requested_before_run_is_not_lost(tmp_path) -> None:
    statuses = []
    supervisor = LauncherSupervisor(
        LauncherConfig(
            data_root=tmp_path / "data",
            host="127.0.0.1",
            port=5000,
        ),
        status_callback=statuses.append,
    )

    supervisor.request_stop()

    assert supervisor.run() == 0
    assert [status.state for status in statuses] == [
        LauncherState.STARTING,
        LauncherState.STOPPED,
    ]
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("reuse_manager", [False, True])
def test_initial_start_and_desktop_restart_check_children_before_prepare(tmp_path, monkeypatch, reuse_manager):
    root = tmp_path / "data"
    manager = StorageManager(root, "local")
    monkeypatch.setattr("src.storage_migrator.runner.wait_for_children", lambda root, **kwargs: wait_for_children(root, timeout=0, **kwargs))
    monkeypatch.setattr("src.backend_v2.launcher.entrypoint.wait_for_children", lambda root, **kwargs: wait_for_children(root, timeout=0, **kwargs))

    def unexpected_prepare(*_args):
        pytest.fail("Database preparation must not run while a previous child is alive")

    monkeypatch.setattr(StorageManager, "prepare", unexpected_prepare)
    with manager if reuse_manager else nullcontext():
        with registered_process(root, "api"):
            supervisor = LauncherSupervisor(
                LauncherConfig(data_root=root, host="127.0.0.1", port=5000),
                storage_manager=manager if reuse_manager else None,
            )
            with pytest.raises(StorageError, match="尚未退出"):
                supervisor.run()


@pytest.mark.parametrize("reuse_manager", [False, True])
def test_stale_epoch_and_reused_protected_pid_do_not_block_startup(tmp_path, monkeypatch, reuse_manager):
    root = tmp_path / "data"
    initialize_database(root)
    engine = create_sqlite_engine(database_path_for(root))
    try:
        repository = ProcessEpochRepository(engine)
        repository.register(EpochRegistration("old-api", "test-token", "api", 123))
        with engine.begin() as connection:
            connection.execute(update(process_epochs).values(lease_expires_at=utcnow() - timedelta(days=1)))

        class ReusedProcess:
            def __init__(self, pid):
                assert pid == 123

            def create_time(self):
                return 2.0

            def cmdline(self):
                raise psutil.AccessDenied(123)

        monkeypatch.setattr(psutil, "Process", ReusedProcess)
        manager = StorageManager(root, "local")
        with manager if reuse_manager else nullcontext():
            marker = control_root(root) / "processes" / "api-123.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(marker, {"data_root": str(root.resolve()), "pid": 123, "created_at": 1.0, "role": "api"})

            class ReadyToStart(Exception):
                pass

            def start_child(**_kwargs):
                assert not marker.exists()
                with engine.connect() as connection:
                    assert connection.execute(select(process_epochs.c.status).where(process_epochs.c.id == "old-api")).scalar_one() == "lost"
                raise ReadyToStart

            monkeypatch.setattr("src.backend_v2.launcher.entrypoint._start_child_with_retries", start_child)
            supervisor = LauncherSupervisor(
                LauncherConfig(data_root=root, host="127.0.0.1", port=5000),
                storage_manager=manager if reuse_manager else None,
            )
            with pytest.raises(ReadyToStart):
                supervisor.run()
    finally:
        engine.dispose()


@pytest.mark.parametrize("reuse_manager", [False, True])
def test_cancel_previous_child_wait_stops_without_error_or_preparing_database(tmp_path, monkeypatch, reuse_manager):
    root = tmp_path / "data"
    manager = StorageManager(root, "local")
    statuses = []
    supervisor = LauncherSupervisor(
        LauncherConfig(data_root=root, host="127.0.0.1", port=5000),
        storage_manager=manager if reuse_manager else None,
        status_callback=statuses.append,
    )
    def unexpected_prepare(*_args):
        pytest.fail("Cancelled startup must not prepare the database")

    monkeypatch.setattr(StorageManager, "prepare", unexpected_prepare)
    monkeypatch.setattr("src.storage_migrator.control.time.sleep", lambda _: supervisor.request_stop())
    with manager if reuse_manager else nullcontext():
        with registered_process(root, "worker"):
            assert supervisor.run() == 0
            assert statuses[-1].state == LauncherState.STOPPED
            assert all(status.state != LauncherState.DEGRADED for status in statuses)
    with DataRootLock(root):
        pass
