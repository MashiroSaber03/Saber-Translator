"""Worker-owned periodic storage and vector reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Callable

from sqlalchemy import Engine, delete, exists, select

from src.backend_v2.insight.derived import InsightVectorStore
from src.backend_v2.insight.qa import TransientRequestRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    jobs,
    web_import_drafts,
)
from src.backend_v2.timestamps import utcnow


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
        self.data_root = data_root.resolve()
        self.storage = AssetStorageService(data_root, engine)
        self.vector_store = InsightVectorStore(data_root)
        self.transient_requests = TransientRequestRepository(engine)
        self.jobs = JobQueueRepository(engine)
        self.engine = engine
        self.interval_seconds = interval_seconds
        self.clock = clock
        self._next_run = 0.0

    def run_if_due(self, *, force: bool = False) -> bool:
        now = self.clock()
        if not force and now < self._next_run:
            return False
        started_at = time.monotonic()
        LOGGER.info("Worker 后台维护开始")
        errors: list[str] = []
        actions = (
            ("recover_journal", self.storage.recover_journal),
            ("prune_import_temp", self._prune_import_temp),
            ("collect_garbage", self.storage.collect_garbage),
            ("reconcile_orphan_objects", self.storage.reconcile_orphan_objects),
            ("prune_job_history", self.jobs.prune_history),
            ("prune_transient_requests", self.transient_requests.prune),
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

    def _prune_import_temp(self) -> dict[str, int]:
        now = utcnow()
        removed_drafts = self._prune_web_import_drafts(now)
        removed_containers = self._prune_container_inputs(now)
        return {
            "webImportDrafts": removed_drafts,
            "containerInputs": removed_containers,
        }

    def _prune_web_import_drafts(self, now: datetime) -> int:
        with immediate_transaction(self.engine) as connection:
            expired_ids = [
                str(value)
                for value in connection.execute(
                    select(web_import_drafts.c.id).where(
                        web_import_drafts.c.expires_at <= now,
                        ~exists(
                            select(jobs.c.id).where(
                                jobs.c.web_import_draft_id
                                == web_import_drafts.c.id,
                                jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                            )
                        ),
                    )
                ).scalars()
            ]
            if expired_ids:
                connection.execute(
                    delete(web_import_drafts).where(
                        web_import_drafts.c.id.in_(expired_ids)
                    )
                )
            live_ids = {
                str(value)
                for value in connection.execute(
                    select(web_import_drafts.c.id)
                ).scalars()
            }

        root = self.data_root / "temp" / "web-import"
        for draft_id in expired_ids:
            shutil.rmtree(root / draft_id, ignore_errors=True)
        cutoff = time.time() - timedelta(hours=24).total_seconds()
        if root.is_dir():
            for child in root.iterdir():
                if (
                    child.is_dir()
                    and child.name not in live_ids
                    and child.stat().st_mtime <= cutoff
                ):
                    shutil.rmtree(child, ignore_errors=True)
        return len(expired_ids)

    def _prune_container_inputs(self, now: datetime) -> int:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        jobs.c.status,
                        jobs.c.config_json,
                        jobs.c.finished_at,
                        jobs.c.updated_at,
                    ).where(jobs.c.kind == "container_import")
                ).mappings()
            )

        root = (self.data_root / "temp" / "container-import").resolve()
        protected: set[Path] = set()
        deadlines: dict[Path, datetime] = {}
        for row in rows:
            try:
                config = json.loads(str(row["config_json"]))
            except (TypeError, ValueError):
                continue
            if not isinstance(config, dict):
                continue
            for key in ("containerRelativePath", "extractedRelativePath"):
                value = config.get(key)
                if not isinstance(value, str) or not value:
                    continue
                path = (self.data_root / value).resolve()
                try:
                    path.relative_to(root)
                except ValueError:
                    continue
                if path == root:
                    continue
                if str(row["status"]) in NONTERMINAL_JOB_STATUSES:
                    protected.add(path)
                    continue
                finished_at = row["finished_at"] or row["updated_at"]
                deadline = finished_at + timedelta(hours=24)
                previous = deadlines.get(path)
                if previous is None or deadline > previous:
                    deadlines[path] = deadline

        removed = 0
        cutoff = time.time() - timedelta(hours=24).total_seconds()
        if not root.is_dir():
            return removed
        for child in root.iterdir():
            path = child.resolve()
            if path in protected:
                continue
            deadline = deadlines.get(path)
            if deadline is not None:
                if deadline > now:
                    continue
            elif child.stat().st_mtime > cutoff:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "无法删除过期导入临时文件，下次维护重试：%s",
                    child,
                )
                continue
            removed += 1
        return removed
