"""Offline storage CLI; imports initialization only for a fresh data root."""

import json

from src.backend_v2.paths import resolve_data_root
from .contracts import StorageError
from .runner import StorageManager


def run_migrator(args) -> int:
    if args.profile == "public" and not args.data_dir:
        raise ValueError("public profile requires an explicit --data-dir")
    manager = StorageManager(resolve_data_root(args.data_dir), args.profile)
    try:
        if args.action == "check":
            result = manager.check()
        else:
            def initialize_database(*positional, **keyword):
                from src.backend_v2.storage.lifecycle import initialize_database as initialize
                return initialize(*positional, **keyword)

            with manager:
                result = manager.prepare(initialize_database)
        print(json.dumps({**result, "diagnostics": str(manager.control)}, ensure_ascii=False))
        return 0
    except (StorageError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "diagnostics": str(manager.control)}, ensure_ascii=False))
        return 1
