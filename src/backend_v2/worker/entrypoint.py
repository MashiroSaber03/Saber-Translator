"""Minimal v2 Worker process lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import threading
import time

from src.backend_v2.paths import data_root_fingerprint, ensure_data_root, resolve_data_root
from src.backend_v2.runtime_heartbeat import EpochHeartbeat
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import ProcessEpochRepository


def _write_ready_marker(data_root: Path, identity: RuntimeIdentity) -> None:
    runtime_dir = data_root / "runtime"
    marker = runtime_dir / "worker-ready.json"
    temporary = runtime_dir / f".worker-ready-{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "epochId": identity.epoch_id,
                "dataRootFingerprint": data_root_fingerprint(data_root),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(marker)


def run_worker(args: object) -> int:
    data_root = ensure_data_root(resolve_data_root(getattr(args, "data_dir", None)))
    identity = RuntimeIdentity.for_worker(test_mode=bool(getattr(args, "test_mode", False)))
    engine = None
    repository = None
    if not identity.test_mode:
        engine = create_sqlite_engine(database_path_for(data_root))
        repository = ProcessEpochRepository(engine)
        if not repository.validate(
            role="worker",
            epoch_id=identity.epoch_id,
            token=identity.epoch_token,
        ):
            engine.dispose()
            raise RuntimeError("Launcher-issued Worker epoch is missing, expired, or invalid")

    if getattr(args, "probe", False):
        print(
            json.dumps(
                {
                    "role": "worker",
                    "status": "ready",
                    "epochId": identity.epoch_id,
                    "dataRootFingerprint": data_root_fingerprint(data_root),
                },
                sort_keys=True,
            )
        )
        if engine is not None:
            engine.dispose()
        return 0

    _write_ready_marker(data_root, identity)
    stop_event = threading.Event()
    heartbeat = (
        EpochHeartbeat(
            repository,
            role="worker",
            identity=identity,
            on_fenced=stop_event.set,
        )
        if repository is not None
        else None
    )

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if heartbeat is not None:
        heartbeat.start()
    try:
        while not stop_event.wait(timeout=0.5):
            time.monotonic()
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if engine is not None:
            engine.dispose()

    return 75 if heartbeat is not None and not heartbeat.healthy else 0
