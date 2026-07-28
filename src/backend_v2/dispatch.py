"""Command-line role dispatch for the isolated v2 runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Saber Translator backend-first v2")
    parser.add_argument(
        "--role",
        choices=("launcher", "api", "worker"),
        default="launcher",
        help="Process role. The packaged executable defaults to launcher.",
    )
    parser.add_argument("--data-dir", help="Explicit v2 data root.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
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
    return parser


def dispatch(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.role == "api":
        from src.backend_v2.api.entrypoint import run_api

        return run_api(args)

    if args.role == "worker":
        from src.backend_v2.worker.entrypoint import run_worker

        return run_worker(args)

    from src.backend_v2.launcher.entrypoint import run_launcher

    return run_launcher(args)
