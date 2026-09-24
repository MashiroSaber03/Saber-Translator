"""Check each packaged role initializes outside the source checkout."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


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


if __name__ == "__main__":
    main()
