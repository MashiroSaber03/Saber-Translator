"""Background failures must reach the launcher as exit codes, not native dialogs."""
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.backend_v2.dispatch import dispatch


@pytest.mark.parametrize("role", ["api", "worker"])
def test_child_failure_logs_cause_and_returns_failure(role, monkeypatch, caplog):
    def fail(_args):
        raise PermissionError("test socket bind denied")

    monkeypatch.setitem(sys.modules, f"src.backend_v2.{role}.entrypoint",
                        SimpleNamespace(**{f"run_{role}": fail}))
    with caplog.at_level(logging.ERROR):
        assert dispatch(["--role", role, "--test-mode"]) == 1
    records = [r for r in caplog.records if r.name == f"saber.{role}"]
    assert len(records) == 1 and records[0].exc_info[0] is PermissionError
    assert "test socket bind denied" in records[0].getMessage()
    assert any(getattr(r, "saber_user_log", False) and "test socket bind denied" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("role", ["api", "worker"])
@pytest.mark.parametrize("exit_code", [0, 73])
def test_child_exit_code_is_preserved(role, exit_code, monkeypatch):
    monkeypatch.setitem(sys.modules, f"src.backend_v2.{role}.entrypoint",
                        SimpleNamespace(**{f"run_{role}": lambda _args: exit_code}))
    assert dispatch(["--role", role, "--test-mode"]) == exit_code


def test_worker_failure_is_logged_once_at_dispatch(tmp_path, monkeypatch, caplog):
    from src.backend_v2.storage.lifecycle import initialize_database
    from src.backend_v2.worker import entrypoint

    root = tmp_path / "data"
    initialize_database(root)

    def fail(*_args):
        raise RuntimeError("test worker startup failure")

    monkeypatch.setattr(entrypoint, "_write_ready_marker", fail)
    monkeypatch.setattr(entrypoint, "configure_backend_logging", lambda **_kwargs: None)
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *_args: None)
    with caplog.at_level(logging.ERROR):
        assert dispatch(["--role", "worker", "--test-mode", "--data-dir", str(root)]) == 1
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 2
    assert sum(record.exc_info is not None for record in errors) == 1
    assert sum(bool(getattr(record, "saber_user_log", False)) for record in errors) == 1


@pytest.mark.parametrize("role", ["api", "worker"])
@pytest.mark.parametrize("error", [SystemExit(7), KeyboardInterrupt()])
def test_child_explicit_exit_and_interrupt_are_not_swallowed(role, error, monkeypatch):
    def stop(_args):
        raise error

    monkeypatch.setitem(sys.modules, f"src.backend_v2.{role}.entrypoint",
                        SimpleNamespace(**{f"run_{role}": stop}))
    with pytest.raises(type(error)) as caught:
        dispatch(["--role", role])
    assert caught.value is error


def test_api_port_conflict_exits_and_keeps_product_and_diagnostic_logs(tmp_path):
    from src.backend_v2.storage.lifecycle import initialize_database

    root = tmp_path / "data"
    initialize_database(root)
    project = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = subprocess.run(
            [sys.executable, str(project / "saber_v2.py"), "--role", "api",
             "--test-mode", "--data-dir", str(root),
             "--port", str(listener.getsockname()[1])],
            cwd=project, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"}, timeout=20,
        )
    assert result.returncode == 1
    assert "API 进程启动或运行失败" in result.stdout
    assert "API 进程启动或运行失败" in (root / "logs/saber-api.log").read_text("utf-8")
    diagnostics = (root / "logs/saber-api-diagnostic.log").read_text("utf-8")
    assert "Traceback (most recent call last)" in diagnostics
    assert "bind_server_socket" in diagnostics
