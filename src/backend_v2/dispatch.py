"""Command-line role dispatch for the isolated v2 runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.backend_v2.local_models import LOCAL_MODEL_IDS
from src.backend_v2.runtime_profile import PROFILE_NAMES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Saber Translator backend-first v2")
    parser.add_argument(
        "--role",
        choices=("desktop", "launcher", "api", "worker"),
        default="desktop",
        help="Process role. The packaged executable defaults to the desktop shell.",
    )
    parser.add_argument("--data-dir", help="Explicit v2 data root.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="local",
        help="Fixed runtime policy profile.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Allow direct API/Worker startup without Launcher-issued epoch identity.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Initialize the selected role, print a JSON smoke result, and exit.",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level. Detailed DEBUG logs are always kept in log files.",
    )
    parser.add_argument(
        "--resident-model",
        action="append",
        choices=LOCAL_MODEL_IDS,
        default=[],
        help=(
            "Preload a built-in local model and keep it resident in the Worker. "
            "Repeat this option to select multiple models."
        ),
    )
    return parser


def dispatch(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resident_model and args.role not in {"launcher", "worker"}:
        parser.error("--resident-model is only supported by launcher and worker roles")

    if args.role == "api":
        from src.backend_v2.api.entrypoint import run_api

        return run_api(args)

    if args.role == "worker":
        from src.backend_v2.worker.entrypoint import run_worker

        return run_worker(args)

    if args.role == "desktop":
        if args.profile != "local":
            parser.error("the desktop role only supports the local profile")
        from src.backend_v2.desktop.entrypoint import run_desktop

        return run_desktop(args)

    from src.backend_v2.launcher.entrypoint import run_launcher

    return run_launcher(args)
