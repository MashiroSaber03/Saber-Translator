"""Check each packaged role initializes outside the source checkout."""

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def probe_model_install(command: list[str], directory: Path) -> None:
    """Exercise the real launcher/API/worker without any preinstalled weights."""
    models = directory / "models"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def api(path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(base + "/api/v2/" + path, data=data,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    env = {**os.environ, "SABER_MODELS_DIR": str(models), "PYTHONUTF8": "1"}
    with (directory / "launcher.log").open("wb") as log:
        process = subprocess.Popen(
            [*command, "--role", "launcher", "--no-browser", "--test-mode",
             "--data-dir", str(directory / "runtime"), "--host", "127.0.0.1", "--port", str(port)],
            cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Packaged launcher exited before becoming ready")
                try:
                    api("health")
                    break
                except URLError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Packaged launcher did not become ready")
                    time.sleep(.5)
            assert api("system/capabilities")["installedModels"] == []
            assert not any(model["installed"] for model in api("system/models")["models"])
            api("system/models/install", {"modelIds": ["detector_yolo"], "source": "github"})
            deadline = time.monotonic() + 180
            while True:
                task = api("system/models")["task"]
                if task["status"] not in {"running", "cancelling"}:
                    assert task["status"] == "completed", task
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Packaged model installation did not finish: {task}")
                time.sleep(.5)
            assert api("system/capabilities")["installedModels"] == ["detector_yolo"]
            manifest = json.loads((Path(__file__).resolve().parents[2] / "src/shared/model_manifest.json").read_text(encoding="utf-8"))
            model = next(model for model in manifest["models"] if model["id"] == "detector_yolo")
            for file in model["files"]:
                installed = models / file["path"]
                assert installed.stat().st_size == file["size"]
                with installed.open("rb") as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == file["sha256"]
            print(json.dumps({"status": "ready", "weightsFreeStartup": True, "installedModels": ["detector_yolo"]}))
        except BaseException:
            print((directory / "launcher.log").read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
            raise
        finally:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            elif process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)


def main() -> None:
    command = [str(Path(argument).resolve()) for argument in sys.argv[1:]]
    if not command:
        raise SystemExit("Usage: probe_package.py EXECUTABLE [ENTRYPOINT]")
    manifest = json.loads((Path(__file__).resolve().parents[2] / "version.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="saber-package-probe-") as directory:
        version = subprocess.run([*command, "--version"], cwd=directory, capture_output=True, text=True, timeout=120, check=True)
        if version.stdout.strip() != manifest["version"]:
            raise RuntimeError("Packaged version differs from release manifest")
        for role in ("desktop", "launcher", "api", "worker"):
            result = subprocess.run(
                [*command, "--role", role, "--probe", "--test-mode",
                 "--data-dir", str(Path(directory) / role)],
                cwd=directory, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=120,
            )
            print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            result.check_returncode()
            payload = json.loads(result.stdout)
            if payload.get("role") != role or payload.get("status") != "ready":
                raise RuntimeError(f"Unexpected {role} probe result: {payload}")
        for action in ("upgrade", "check"):
            result = subprocess.run(
                [*command, "--role", "storage-migrator", "--action", action,
                 "--data-dir", str(Path(directory) / "storage")],
                cwd=directory, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=120, check=True,
            )
            payload = json.loads(result.stdout)
            expected = "created" if action == "upgrade" else "current"
            if payload.get("status") != expected or payload.get("targetVersion") != manifest["storageVersion"]:
                raise RuntimeError(f"Unexpected storage probe result: {payload}")
        probe_model_install(command, Path(directory))


if __name__ == "__main__":
    main()
