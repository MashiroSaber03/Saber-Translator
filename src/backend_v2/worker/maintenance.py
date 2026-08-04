"""Worker-owned periodic storage and vector reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Callable

from sqlalchemy import Engine, delete, exists, func, or_, select, tuple_

from src.backend_v2.insight.derived import InsightVectorStore
from src.backend_v2.insight.gc import InsightReachabilityGarbageCollector
from src.backend_v2.insight.qa import TransientRequestRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    idempotency_records,
    job_artifacts,
    jobs,
    operation_artifacts,
    operations,
    web_import_drafts,
)
from src.backend_v2.timestamps import utcnow


LOGGER = logging.getLogger(__name__)

TERMINAL_OPERATION_STATUSES = ("completed", "failed", "cancelled")
OPERATION_RETENTION = timedelta(days=30)
MAINTENANCE_DELETE_LIMIT = 500


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
        self.insight_gc = InsightReachabilityGarbageCollector(engine)
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
            ("prune_job_history", self.jobs.prune_history),
            ("prune_expired_artifacts", self._prune_expired_artifacts),
            ("prune_terminal_operations", self._prune_terminal_operations),
            ("prune_idempotency_records", self._prune_idempotency_records),
            ("collect_insight_garbage", self.insight_gc.collect),
            ("collect_garbage", self.storage.collect_garbage),
            ("reconcile_orphan_objects", self.storage.reconcile_orphan_objects),
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

    def _prune_expired_artifacts(self) -> dict[str, int]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            job_keys = list(
                connection.execute(
                    select(job_artifacts.c.job_id, job_artifacts.c.kind)
                    .where(job_artifacts.c.expires_at <= now)
                    .order_by(job_artifacts.c.expires_at)
                    .limit(MAINTENANCE_DELETE_LIMIT)
                )
            )
            if job_keys:
                connection.execute(
                    delete(job_artifacts).where(
                        tuple_(job_artifacts.c.job_id, job_artifacts.c.kind).in_(
                            job_keys
                        )
                    )
                )
            operation_keys = list(
                connection.execute(
                    select(
                        operation_artifacts.c.operation_id,
                        operation_artifacts.c.kind,
                    )
                    .where(operation_artifacts.c.expires_at <= now)
                    .order_by(operation_artifacts.c.expires_at)
                    .limit(MAINTENANCE_DELETE_LIMIT)
                )
            )
            if operation_keys:
                connection.execute(
                    delete(operation_artifacts).where(
                        tuple_(
                            operation_artifacts.c.operation_id,
                            operation_artifacts.c.kind,
                        ).in_(operation_keys)
                    )
                )
        return {
            "jobArtifacts": len(job_keys),
            "operationArtifacts": len(operation_keys),
        }

    def _prune_terminal_operations(self) -> int:
        now = utcnow()
        cutoff = now - OPERATION_RETENTION
        with immediate_transaction(self.engine) as connection:
            operation_ids = [
                str(value)
                for value in connection.execute(
                    select(operations.c.id)
                    .where(
                        operations.c.status.in_(TERMINAL_OPERATION_STATUSES),
                        func.coalesce(
                            operations.c.finished_at,
                            operations.c.updated_at,
                            operations.c.created_at,
                        )
                        <= cutoff,
                        ~exists(
                            select(operation_artifacts.c.operation_id).where(
                                operation_artifacts.c.operation_id
                                == operations.c.id,
                                or_(
                                    operation_artifacts.c.expires_at.is_(None),
                                    operation_artifacts.c.expires_at > now,
                                ),
                            )
                        ),
                        ~exists(
                            select(idempotency_records.c.key).where(
                                idempotency_records.c.resource_type
                                == "operation",
                                idempotency_records.c.resource_id
                                == operations.c.id,
                                idempotency_records.c.expires_at > now,
                            )
                        ),
                    )
                    .order_by(
                        func.coalesce(
                            operations.c.finished_at,
                            operations.c.updated_at,
                            operations.c.created_at,
                        ),
                        operations.c.id,
                    )
                    .limit(MAINTENANCE_DELETE_LIMIT)
                ).scalars()
            ]
            if operation_ids:
                connection.execute(
                    delete(operations).where(operations.c.id.in_(operation_ids))
                )
        return len(operation_ids)

    def _prune_idempotency_records(self) -> int:
        now = utcnow()
        active_operation = operations.alias("active_idempotent_operation")
        active_batch_job = jobs.alias("active_idempotent_batch_job")
        active_draft_job = jobs.alias("active_idempotent_draft_job")
        linked_draft = web_import_drafts.alias("idempotent_web_import_draft")
        with immediate_transaction(self.engine) as connection:
            protected = or_(
                exists(
                    select(active_operation.c.id).where(
                        idempotency_records.c.resource_type == "operation",
                        active_operation.c.id == idempotency_records.c.resource_id,
                        active_operation.c.status.in_(("pending", "running")),
                    )
                ),
                exists(
                    select(active_batch_job.c.id).where(
                        idempotency_records.c.resource_type == "job_batch",
                        active_batch_job.c.batch_id
                        == idempotency_records.c.resource_id,
                        active_batch_job.c.status.in_(NONTERMINAL_JOB_STATUSES),
                    )
                ),
                exists(
                    select(linked_draft.c.id).where(
                        idempotency_records.c.resource_type
                        == "web_import_draft",
                        linked_draft.c.id == idempotency_records.c.resource_id,
                        or_(
                            linked_draft.c.expires_at > now,
                            exists(
                                select(active_draft_job.c.id).where(
                                    active_draft_job.c.web_import_draft_id
                                    == linked_draft.c.id,
                                    active_draft_job.c.status.in_(
                                        NONTERMINAL_JOB_STATUSES
                                    ),
                                )
                            ),
                        ),
                    )
                ),
            )
            keys = list(
                connection.execute(
                    select(idempotency_records.c.scope, idempotency_records.c.key)
                    .where(
                        idempotency_records.c.expires_at <= now,
                        ~protected,
                    )
                    .order_by(idempotency_records.c.expires_at)
                    .limit(MAINTENANCE_DELETE_LIMIT)
                )
            )
            if keys:
                connection.execute(
                    delete(idempotency_records).where(
                        tuple_(
                            idempotency_records.c.scope,
                            idempotency_records.c.key,
                        ).in_(keys)
                    )
                )
        return len(keys)

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
