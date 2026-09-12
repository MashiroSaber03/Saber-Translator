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
    with tempfile.TemporaryDirectory(prefix="saber-package-probe-") as directory:
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


if __name__ == "__main__":
    main()
