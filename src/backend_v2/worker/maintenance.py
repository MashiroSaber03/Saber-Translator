"""Worker-owned periodic storage and vector reconciliation."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Callable

from sqlalchemy import Engine

from src.backend_v2.insight.derived import InsightVectorStore
from src.backend_v2.storage.assets import AssetStorageService


LOGGER = logging.getLogger(__name__)


class WorkerMaintenance:
    """Run bounded maintenance only at Worker scheduler safe points."""

    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        interval_seconds: float = 600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("maintenance interval must be positive")
        self.storage = AssetStorageService(data_root, engine)
        self.vector_store = InsightVectorStore(data_root)
        self.engine = engine
        self.interval_seconds = interval_seconds
        self.clock = clock
        self._next_run = 0.0
        self.last_errors: tuple[str, ...] = ()

    def run_if_due(self, *, force: bool = False) -> bool:
        now = self.clock()
        if not force and now < self._next_run:
            return False
        started_at = time.monotonic()
        LOGGER.info("Worker 后台维护开始")
        errors: list[str] = []
        actions = (
            ("recover_journal", self.storage.recover_journal),
            ("scan_integrity", self.storage.scan_integrity),
            ("collect_garbage", self.storage.collect_garbage),
            ("reconcile_orphan_objects", self.storage.reconcile_orphan_objects),
            (
                "collect_orphan_vector_collections",
                lambda: self.vector_store.collect_orphan_collections(self.engine),
            ),
        )
        for name, action in actions:
            try:
                result = action()
                LOGGER.debug("后台维护完成：action=%s result=%s", name, result)
            except Exception:
                errors.append(name)
                LOGGER.exception("Worker maintenance action failed: %s", name)
        self.last_errors = tuple(errors)
        self._next_run = now + self.interval_seconds
        if errors:
            LOGGER.warning(
                "Worker 后台维护结束：duration=%.2fs failed=%s",
                time.monotonic() - started_at,
                ",".join(errors),
            )
        else:
            LOGGER.info(
                "Worker 后台维护结束：duration=%.2fs",
                time.monotonic() - started_at,
            )
        return True
