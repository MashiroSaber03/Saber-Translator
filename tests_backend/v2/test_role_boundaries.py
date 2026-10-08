from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import urlopen

import psutil
import pytest
from sqlalchemy import select

from src.backend_v2.launcher.entrypoint import (
    MAX_CONSECUTIVE_RESTARTS,
    RESTART_STABILITY_SECONDS,
    TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT_ENV,
    WORKER_CUDNN_V8_API_LRU_CACHE_LIMIT,
    ManagedChild,
    _child_environment,
    _reset_restart_count_after_stable_run,
    _start_child_with_retries,
    _stop_children,
    _try_reconcile_dead_child,
)
from src.backend_v2.runtime_identity import (
    API_BIND_FAILED_EXIT_CODE,
    API_EPOCH_ID_ENV,
    API_EPOCH_TOKEN_ENV,
    LAUNCHER_PID_ENV,
    WORKER_EPOCH_ID_ENV,
    WORKER_EPOCH_TOKEN_ENV,
    _watch_launcher_parent,
)
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import EpochRegistration
from src.backend_v2.storage.schema import process_epochs
from src.backend_v2.worker.entrypoint import _insight_layer_handler
from src.backend_v2.jobs.repository import JobItemSpec, JobQueueRepository, JobSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "saber_v2.py"


def test_worker_dynamic_insight_layer_handler_has_no_fixed_layer_cap() -> None:
    handler = object()
    service = SimpleNamespace(handle=handler)

    assert _insight_layer_handler("insight_build_layer_0", service) is handler
    assert _insight_layer_handler("insight_build_layer_8", service) is handler
    assert _insight_layer_handler("insight_build_layer_128", service) is handler
    assert _insight_layer_handler("insight_build_layer_08", service) is None
    assert _insight_layer_handler("insight_build_layer_invalid", service) is None


def _clean_role_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        API_EPOCH_ID_ENV,
        API_EPOCH_TOKEN_ENV,
        WORKER_EPOCH_ID_ENV,
        WORKER_EPOCH_TOKEN_ENV,
        LAUNCHER_PID_ENV,
        TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT_ENV,
    ):
        environment.pop(name, None)
    return environment


def _run_probe(
    role: str,
    data_root: Path,
    *extra_args: str,
) -> dict[str, object]:
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
            *extra_args,
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


def test_api_validates_identity_before_application_initialization(
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
            events.append("identity_validated")
            return True

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
        assert events[0] == "identity_validated"
        return fake_app

    monkeypatch.setattr(
        entrypoint.RuntimeIdentity,
        "for_api",
        classmethod(
            lambda _cls, **_kwargs: RuntimeIdentity("api-epoch", "token")
        ),
    )
    monkeypatch.setattr(entrypoint, "ProcessEpochRepository", FakeEpochRepository)
    monkeypatch.setattr(entrypoint, "create_sqlite_engine", lambda _path: FakeEngine())
    monkeypatch.setattr(entrypoint, "create_api_app", create_app)
    monkeypatch.setattr(entrypoint, "loaded_forbidden_api_modules", lambda: [])
    monkeypatch.setattr(entrypoint, "start_launcher_parent_monitor", lambda *_args, **_kwargs: None)

    from src.backend_v2.storage.lifecycle import initialize_database
    initialize_database(tmp_path / "api-identity")
    result = entrypoint.run_api(
        SimpleNamespace(
            data_dir=str(tmp_path / "api-identity"),
            probe=True,
            test_mode=False,
            host="127.0.0.1",
            port=5000,
            log_level=None,
        )
    )

    assert result == 0
    assert events == [
        "identity_validated",
        "app_initialized",
        "runtime_closed",
        "engine_disposed",
    ]


def test_worker_and_launcher_resolve_the_same_explicit_data_root(tmp_path: Path) -> None:
    worker = _run_probe("worker", tmp_path / "shared")
    launcher = _run_probe("launcher", tmp_path / "shared")

    assert worker["dataRootFingerprint"] == launcher["dataRootFingerprint"]
    assert launcher["apiCommand"][0] == sys.executable
    assert launcher["workerCommand"][0] == sys.executable


def test_launcher_probe_propagates_repeated_resident_model_options(
    tmp_path: Path,
) -> None:
    launcher = _run_probe(
        "launcher",
        tmp_path / "resident",
        "--resident-model",
        "manga_ocr",
        "--resident-model",
        "detector_yolo",
    )

    worker_command = launcher["workerCommand"]
    assert worker_command[-4:] == [
        "--resident-model",
        "detector_yolo",
        "--resident-model",
        "manga_ocr",
    ]
    assert "--resident-model" not in launcher["apiCommand"]


@pytest.mark.parametrize("role", ["api", "desktop"])
def test_non_worker_roles_reject_resident_model_options(
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
            "--test-mode",
            "--probe",
            "--resident-model",
            "detector_yolo",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_role_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "only supported by launcher and worker roles" in completed.stderr


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


def test_launcher_exposes_only_the_target_roles_secret() -> None:
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
        api = _child_environment("api", api_registration)
        worker = _child_environment("worker", worker_registration)
    finally:
        os.environ.clear()
        os.environ.update(original)

    assert API_EPOCH_ID_ENV in api and API_EPOCH_TOKEN_ENV in api
    assert WORKER_EPOCH_ID_ENV not in api and WORKER_EPOCH_TOKEN_ENV not in api
    assert WORKER_EPOCH_ID_ENV in worker and WORKER_EPOCH_TOKEN_ENV in worker
    assert API_EPOCH_ID_ENV not in worker and API_EPOCH_TOKEN_ENV not in worker
    assert TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT_ENV not in api
    assert (
        worker[TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT_ENV]
        == WORKER_CUDNN_V8_API_LRU_CACHE_LIMIT
    )
    assert api[LAUNCHER_PID_ENV] == str(os.getpid())
    assert worker[LAUNCHER_PID_ENV] == str(os.getpid())
    with pytest.raises(ValueError):
        _child_environment("renderer", api_registration)


def test_posix_parent_monitor_stops_after_launcher_parent_changes() -> None:
    stop_event = threading.Event()
    parent_pids = iter((1234, 4321))
    lost = []

    _watch_launcher_parent(
        1234,
        lambda: lost.append(True),
        stop_event,
        get_parent_pid=lambda: next(parent_pids),
        interval_seconds=0,
    )

    assert lost == [True]


def test_stop_children_terminates_descendants_before_wrapper(monkeypatch) -> None:
    events: list[str] = []

    class FakeDescendant:
        def terminate(self) -> None:
            events.append("descendant-terminate")

        def kill(self) -> None:
            events.append("descendant-kill")

    class FakePsutilRoot:
        def children(self, *, recursive: bool) -> list[FakeDescendant]:
            assert recursive is True
            return [descendant]

    class FakeChild:
        pid = 123

        def poll(self):
            return None

        def terminate(self) -> None:
            events.append("wrapper-terminate")

        def wait(self, *, timeout: float):
            events.append("wrapper-wait")
            return 0

        def kill(self) -> None:
            events.append("wrapper-kill")

    descendant = FakeDescendant()
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint.psutil.Process",
        lambda _pid: FakePsutilRoot(),
    )
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint.psutil.wait_procs",
        lambda processes, timeout: (list(processes), []),
    )

    _stop_children([FakeChild()])  # type: ignore[list-item]

    assert events == [
        "descendant-terminate",
        "wrapper-terminate",
        "wrapper-wait",
    ]


def test_launcher_retries_dead_child_reconciliation_after_sqlite_busy() -> None:
    class BusyRepository:
        def reconcile_dead_worker(self, _epoch_id: str) -> None:
            raise sqlite3.OperationalError("database is locked")

    assert not _try_reconcile_dead_child(
        BusyRepository(),  # type: ignore[arg-type]
        role="worker",
        epoch_id="worker-epoch",
    )


def test_launcher_does_not_hide_non_busy_reconciliation_errors() -> None:
    class BrokenRepository:
        def reconcile_dead_api(self, _epoch_id: str) -> None:
            raise RuntimeError("reconciliation failed")

    with pytest.raises(RuntimeError, match="reconciliation failed"):
        _try_reconcile_dead_child(
            BrokenRepository(),  # type: ignore[arg-type]
            role="api",
            epoch_id="api-epoch",
        )


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
    cause = RuntimeError("startup failed")

    def fail_start_child(**kwargs):
        attempts.append(int(kwargs["restart_count"]))
        raise cause

    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint._start_child",
        fail_start_child,
    )
    monkeypatch.setattr(
        "src.backend_v2.launcher.entrypoint.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="启动失败.*startup failed") as caught:
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
    assert caught.value.__cause__ is cause


def test_launcher_port_conflict_stops_after_one_api_attempt(tmp_path: Path) -> None:
    from src.backend_v2.storage.lifecycle import initialize_database

    root = tmp_path / "data"
    initialize_database(root)
    project = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = subprocess.run(
            [sys.executable, str(project / "saber_v2.py"), "--role", "launcher",
             "--data-dir", str(root), "--port", str(listener.getsockname()[1])],
            cwd=project, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"}, timeout=20,
        )
    assert result.returncode == API_BIND_FAILED_EXIT_CODE
    assert "无法监听端口" in result.stdout + result.stderr
    assert "第 1/3 次连续重启" not in result.stdout + result.stderr
    with sqlite3.connect(root / "saber.sqlite3") as database:
        assert database.execute("SELECT count(*) FROM process_epochs WHERE role='api'").fetchone()[0] == 1
        assert database.execute("SELECT count(*) FROM process_epochs WHERE role='worker'").fetchone()[0] == 0


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


@pytest.mark.skipif(os.name != "nt", reason="Windows native GIL blocking")
def test_live_api_and_worker_survive_native_blocking_and_publish_result(tmp_path: Path) -> None:
    # Keep the production Launcher, API, Worker and job loop. Only replace a
    # test handler and add a test route to reproduce a native call holding GIL.
    wrapper = tmp_path / "blocking_roles.py"
    wrapper.write_text(
        f"import sys\nsys.path.insert(0, {str(PROJECT_ROOT)!r})\n" + '''
import ctypes
from pathlib import Path
from flask import jsonify
from src.backend_v2.launcher import entrypoint as launcher
original_command = launcher._role_command
def role_command(*args, **kwargs):
    command = original_command(*args, **kwargs)
    command[1] = __file__
    return command
launcher._role_command = role_command
role = sys.argv[sys.argv.index('--role') + 1]
def block():
    sleep = ctypes.PyDLL('kernel32').Sleep
    sleep.argtypes = [ctypes.c_ulong]
    sleep.restype = None
    sleep(15000)
if role == 'worker':
    from src.backend_v2.jobs.worker_loop import JobWorkerLoop
    original_init = JobWorkerLoop.__init__
    def package(*_args):
        root = Path(sys.argv[sys.argv.index('--data-dir') + 1])
        (root / 'native-block-started').touch()
        block()
        return {'nativeCallCompleted': True}
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.handlers['package'] = package
    JobWorkerLoop.__init__ = initialize
elif role == 'api':
    from src.backend_v2.api import entrypoint as api
    original_app = api.create_api_app
    def create_app(*args, **kwargs):
        app = original_app(*args, **kwargs)
        def native_block():
            block()
            return jsonify({'nativeCallCompleted': True})
        app.add_url_rule('/__test_native_block', view_func=native_block)
        return app
    api.create_api_app = create_app
import saber_v2
raise SystemExit(saber_v2.main())
''', encoding="utf-8",
    )
    port = _free_port()
    root = tmp_path / "runtime"
    process = subprocess.Popen(
        [sys.executable, str(wrapper), "--role", "launcher", "--data-dir",
         str(root), "--host", "127.0.0.1", "--port", str(port), "--no-browser"],
        cwd=PROJECT_ROOT, env=_clean_role_environment(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    engine = None
    api_results = []
    api_thread = None
    child_pids = []
    try:
        _wait_until(lambda: (root / "runtime/worker-ready.json").exists(), timeout=30)
        engine = create_sqlite_engine(database_path_for(root))
        with engine.connect() as connection:
            initial_epochs = connection.execute(
                select(process_epochs.c.id, process_epochs.c.pid)
                .where(process_epochs.c.role.in_(("api", "worker")))
            ).all()
            assert connection.execute(
                select(process_epochs.c.id).where(process_epochs.c.role == "launcher")
            ).all() == []
        child_pids = [int(row.pid) for row in initial_epochs]
        assert len(initial_epochs) == 2
        repository = JobQueueRepository(engine)
        job = repository.create_batch(
            display_name="Native blocking regression",
            specs=[JobSpec(kind="export", config={"mode": "test"},
                           items=(JobItemSpec(page_id=None, step_kinds=("package",)),))],
        )
        job_id = str(job["jobIds"][0])
        def block_api():
            with urlopen(f"http://127.0.0.1:{port}/__test_native_block", timeout=40) as response:
                api_results.append(json.loads(response.read()))
        api_thread = threading.Thread(target=block_api, daemon=True)
        api_thread.start()
        _wait_until(lambda: (root / "native-block-started").exists(), timeout=15)
        time.sleep(13)
        assert process.poll() is None
        assert all(psutil.pid_exists(pid) for pid in child_pids)
        with engine.connect() as connection:
            epochs = connection.execute(
                select(process_epochs.c.id, process_epochs.c.status)
                .where(process_epochs.c.role.in_(("api", "worker")))
            ).all()
        assert epochs == [(row.id, "active") for row in initial_epochs]
        _wait_until(lambda: repository.get_job(job_id)["status"] == "completed", timeout=15)
        api_thread.join(timeout=15)
        assert api_results == [{"nativeCallCompleted": True}]
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
        if api_thread is not None:
            api_thread.join(timeout=1)
        if engine is not None:
            engine.dispose()
    _wait_until(lambda: all(not psutil.pid_exists(pid) for pid in child_pids), timeout=10)


@pytest.mark.skipif(os.name != "nt", reason="Windows process supervision integration")
def test_queued_job_waits_for_running_auxiliary_work_without_restarting_worker(tmp_path: Path) -> None:
    wrapper = tmp_path / "slow_auxiliary.py"
    wrapper.write_text(
        f"import sys\nsys.path.insert(0, {str(PROJECT_ROOT)!r})\n" + '''
import time
from pathlib import Path
from src.backend_v2.launcher import entrypoint as launcher
original_command = launcher._role_command
def role_command(*args, **kwargs):
    command = original_command(*args, **kwargs)
    command[1] = __file__
    return command
launcher._role_command = role_command
if sys.argv[sys.argv.index('--role') + 1] == 'worker':
    from src.backend_v2.operations.executor import WorkerOperationRunner
    from src.backend_v2.jobs.worker_loop import JobWorkerLoop
    root = Path(sys.argv[sys.argv.index('--data-dir') + 1])
    original_run_one = WorkerOperationRunner.run_one
    def run_one(self):
        trigger = root / 'auxiliary-trigger'
        if trigger.exists():
            trigger.unlink()
            (root / 'auxiliary-started').touch()
            time.sleep(6)
            (root / 'auxiliary-finished').touch()
            return True
        return original_run_one(self)
    WorkerOperationRunner.run_one = run_one
    original_init = JobWorkerLoop.__init__
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.handlers['package'] = lambda *_args: {'done': True}
    JobWorkerLoop.__init__ = initialize
import saber_v2
raise SystemExit(saber_v2.main())
''', encoding="utf-8",
    )
    port = _free_port()
    root = tmp_path / "runtime"
    process = subprocess.Popen(
        [sys.executable, str(wrapper), "--role", "launcher", "--data-dir",
         str(root), "--host", "127.0.0.1", "--port", str(port), "--no-browser"],
        cwd=PROJECT_ROOT, env=_clean_role_environment(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    engine = None
    children = []
    try:
        marker_path = root / "runtime/worker-ready.json"
        _wait_until(marker_path.exists, timeout=30)
        original = json.loads(marker_path.read_text(encoding="utf-8"))
        children = [child.pid for child in psutil.Process(process.pid).children(recursive=True)]
        (root / "auxiliary-trigger").touch()
        _wait_until(lambda: (root / "auxiliary-started").exists())
        engine = create_sqlite_engine(database_path_for(root))
        repository = JobQueueRepository(engine)
        result = repository.create_batch(
            display_name="Queued behind auxiliary work",
            specs=[JobSpec(kind="export", config={"mode": "test"},
                           items=(JobItemSpec(page_id=None, step_kinds=("package",)),))],
        )
        job_id = str(result["jobIds"][0])
        _wait_until(lambda: repository.get_job(job_id)["status"] == "completed", timeout=20)
        assert (root / "auxiliary-finished").exists()
        current = json.loads(marker_path.read_text(encoding="utf-8"))
        assert current["epochId"] == original["epochId"]
        assert current["pid"] == original["pid"]
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
        if engine is not None:
            engine.dispose()
    _wait_until(lambda: all(not psutil.pid_exists(pid) for pid in children), timeout=10)


@pytest.mark.skipif(os.name != "nt", reason="Windows process supervision integration")
def test_launcher_restarts_exited_worker_without_restarting_api(
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

        psutil.Process(int(initial_marker["pid"])).terminate()

        def worker_restarted() -> bool:
            try:
                current = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            return current.get("epochId") != initial_marker["epochId"]

        _wait_until(worker_restarted, timeout=20)
        with sqlite3.connect(database_path_for(data_root)) as database:
            assert database.execute("SELECT status FROM process_epochs WHERE id=?", (initial_marker["epochId"],)).fetchone()[0] == "lost"
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
def test_launcher_restarts_exited_api_without_restarting_worker(
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
        engine.dispose()
        psutil.Process(initial_api_pid).terminate()

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
        with sqlite3.connect(database_path_for(data_root)) as database:
            assert database.execute("SELECT status FROM process_epochs WHERE id=?", (initial_api_epoch,)).fetchone()[0] == "lost"
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
