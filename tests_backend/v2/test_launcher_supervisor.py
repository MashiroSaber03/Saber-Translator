from __future__ import annotations

import sys

from src.backend_v2.launcher.entrypoint import (
    LauncherConfig,
    LauncherState,
    LauncherSupervisor,
    _child_environment,
    _new_registration,
    _spawn,
)


def test_child_process_logs_are_forced_to_utf8(tmp_path) -> None:
    registration = _new_registration("api")

    environment = _child_environment(tmp_path, "api", registration)

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
