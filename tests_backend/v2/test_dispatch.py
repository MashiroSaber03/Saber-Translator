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
from src.backend_v2.runtime_identity import API_BIND_FAILED_EXIT_CODE


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


@pytest.mark.parametrize("reuse_address", [False, True])
def test_api_port_conflict_exits_and_keeps_product_and_diagnostic_logs(tmp_path, reuse_address):
    from src.backend_v2.storage.lifecycle import initialize_database

    root = tmp_path / "data"
    initialize_database(root)
    project = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse_address))
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = subprocess.run(
            [sys.executable, str(project / "saber_v2.py"), "--role", "api",
             "--test-mode", "--data-dir", str(root),
             "--port", str(listener.getsockname()[1])],
            cwd=project, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"}, timeout=20,
        )
    assert result.returncode == API_BIND_FAILED_EXIT_CODE
    assert "无法监听" in result.stdout
    assert "API 服务已就绪" not in result.stdout
    assert "无法监听" in (root / "logs/saber-api.log").read_text("utf-8")
    diagnostics = (root / "logs/saber-api-diagnostic.log").read_text("utf-8")
    assert "Traceback (most recent call last)" in diagnostics
    assert "OSError" in diagnostics


@pytest.mark.skipif(os.name != "nt", reason="Windows exclusive listener")
@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "0.0.0.0", "*"])
def test_windows_api_listener_keeps_exclusive_binding(host):
    from waitress.adjustments import Adjustments
    from src.backend_v2.api.entrypoint import _create_http_server
    from src.backend_v2.runtime_profile import resolve_runtime_profile

    addresses = Adjustments(host=host, port=0).listen
    for family, socktype, proto, address in addresses:
        if family == socket.AF_INET6:
            try:
                with socket.socket(family, socktype, proto) as probe:
                    probe.bind(address)
            except OSError:
                pytest.skip("IPv6 binding unavailable")
    server = _create_http_server(lambda _environ, _start_response: [], host=host, port=0, profile=resolve_runtime_profile("local"))
    try:
        listeners = server.adj.sockets
        assert {s.getsockname()[0] for s in listeners} == {address[0] for _, _, _, address in addresses}
        for listener in listeners:
            assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE) == 1
            if listener.family == socket.AF_INET6:
                assert listener.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
            with socket.socket(listener.family, socket.SOCK_STREAM) as contender:
                contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                with pytest.raises(OSError):
                    contender.bind(listener.getsockname())
    finally:
        server.close()
        server.task_dispatcher.shutdown(cancel_pending=True, timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows exclusive listener")
def test_windows_api_listener_releases_earlier_bindings_on_failure(monkeypatch):
    from waitress.adjustments import Adjustments
    from src.backend_v2.api.entrypoint import _create_http_server
    from src.backend_v2.runtime_profile import resolve_runtime_profile

    # Two real loopback addresses reproduce a conflict after the first bind.
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        addresses = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, ("127.0.0.2", port)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, ("127.0.0.1", port)),
        ]
        monkeypatch.setattr(Adjustments, "__init__", lambda self, **_kwargs: setattr(self, "listen", addresses))
        with pytest.raises(OSError):
            _create_http_server(lambda _environ, _start_response: [], host="localhost", port=port, profile=resolve_runtime_profile("local"))
        with socket.socket() as released:
            released.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            released.bind(("127.0.0.2", port))
