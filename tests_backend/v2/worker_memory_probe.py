"""Isolated RSS probe for the durable job scheduler.

This helper is intentionally not named like a pytest module.  The parent test
starts one fresh process per graph size so allocator history from a previous
sample cannot hide a linear memory trend.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import threading
import time
import uuid

import psutil
from sqlalchemy import select

from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import jobs, metadata
from src.backend_v2.storage.seeding import seed_system_records


def main() -> int:
    item_count = int(sys.argv[1])
    database_path = Path(sys.argv[2])
    engine = create_sqlite_engine(database_path)
    metadata.create_all(engine)
    seed_system_records(engine)

    worker_epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            epoch_id=worker_epoch_id,
            token="worker-memory-probe",
            role="worker",
            pid=psutil.Process().pid,
        )
    )
    repository = JobQueueRepository(engine, attempt_lease_seconds=120)
    created = repository.create_batch(
        kind="export",
        display_name=f"{item_count}-item memory probe",
        specs=(
            JobSpec(
                kind="export",
                config={"mode": "memory-probe"},
                items=tuple(
                    JobItemSpec(page_id=None, step_kinds=("package",))
                    for _index in range(item_count)
                ),
            ),
        ),
    )
    job_id = str(created["jobIds"][0])
    del created
    gc.collect()

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    samples = 0

    def sample() -> None:
        nonlocal peak_rss, samples
        peak_rss = max(peak_rss, process.memory_info().rss)
        samples += 1

    def handler(_fence, _step):
        sample()
        return {"done": True}

    stop_event = threading.Event()
    loop = JobWorkerLoop(
        repository,
        worker_epoch_id=worker_epoch_id,
        handlers={"package": handler},
        idle_poll_seconds=0.005,
    )
    worker_thread = threading.Thread(
        target=loop.run,
        args=(stop_event,),
        daemon=True,
    )
    worker_thread.start()
    deadline = time.monotonic() + 180
    status = "queued"
    while time.monotonic() < deadline:
        sample()
        with engine.connect() as connection:
            status = str(
                connection.execute(
                    select(jobs.c.status).where(jobs.c.id == job_id)
                ).scalar_one()
            )
        if status in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(0.01)

    stop_event.set()
    worker_thread.join(timeout=5)
    sample()
    engine.dispose()
    gc.collect()
    final_rss = process.memory_info().rss
    print(
        json.dumps(
            {
                "baselineRss": baseline_rss,
                "finalRss": final_rss,
                "itemCount": item_count,
                "peakDelta": peak_rss - baseline_rss,
                "peakRss": peak_rss,
                "samples": samples,
                "status": status,
                "workerStopped": not worker_thread.is_alive(),
            },
            separators=(",", ":"),
        )
    )
    return 0 if status == "completed" and not worker_thread.is_alive() else 1


if __name__ == "__main__":
    raise SystemExit(main())
