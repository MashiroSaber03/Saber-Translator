"""Windows GUI processes must retain redirected logs without a console."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.backend_v2.paths import project_root


@pytest.mark.skipif(sys.platform != "win32", reason="Windows windowed interpreter")
def test_pythonw_restores_pipes_without_allocating_console():
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    script = """
import ctypes
import sys
from src.backend_v2.standard_streams import restore_standard_streams
restore_standard_streams()
assert ctypes.windll.kernel32.GetConsoleWindow() == 0
print(sys.stdin.readline().strip(), flush=True)
sys.stdout.close()
print('中文错误日志', file=sys.stderr, flush=True)
"""
    result = subprocess.run(
        [str(pythonw), "-c", script], cwd=project_root(), input="中文后台日志\n",
        capture_output=True, encoding="utf-8", timeout=20,
        # Match the UTF-8 environment used by the actual backend launcher.
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "中文后台日志"
    assert result.stderr.strip() == "中文错误日志"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows windowed interpreter")
def test_pythonw_preserves_combined_log_pipe():
    result = subprocess.run(
        [str(Path(sys.executable).with_name("pythonw.exe")), "-c",
         "import sys; from src.backend_v2.standard_streams import restore_standard_streams; "
         "restore_standard_streams(); print('first', flush=True); sys.stdout.close(); "
         "print('second', file=sys.stderr, flush=True)"],
        cwd=project_root(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", timeout=20, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.splitlines() == ["first", "second"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows windowed interpreter")
def test_pythonw_entrypoint_retains_version_output():
    from src.version import APP_VERSION

    result = subprocess.run(
        [str(Path(sys.executable).with_name("pythonw.exe")), "saber_v2.py", "--version"],
        cwd=project_root(), capture_output=True, encoding="utf-8", timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == APP_VERSION
