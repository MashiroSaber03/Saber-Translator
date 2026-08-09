from __future__ import annotations

from src.backend_v2.launcher.entrypoint import (
    LauncherConfig,
    LauncherState,
    LauncherSupervisor,
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
