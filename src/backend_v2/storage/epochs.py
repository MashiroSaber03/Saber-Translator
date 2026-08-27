"""Authoritative process epoch, heartbeat, and recovery repository."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import secrets
from typing import Literal

from sqlalchemy import Engine, exists, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.timestamps import utcnow
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    jobs,
    operation_events,
    operations,
    pages,
    process_epochs,
    render_requests,
    transient_requests,
)


ProcessRole = Literal["launcher", "api", "worker"]
REMOTE_API_OPERATION_KINDS = frozenset(
    {"bubble_translate", "studio_generate", "studio_chat", "studio_summary"}
)


def hash_epoch_token(token: str) -> str:
    if not token:
        raise ValueError("epoch token is required")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EpochRegistration:
    epoch_id: str
    token: str
    role: ProcessRole
    pid: int


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    epoch_id: str
    role: ProcessRole
    jobs_interrupted: int = 0
    operations_failed: int = 0
    operations_requeued: int = 0
    renders_requeued: int = 0
    transient_requests_requeued: int = 0
    transient_requests_cancelled: int = 0
    changed: bool = False


class ProcessEpochRepository:
    def __init__(self, engine: Engine, *, lease_seconds: int = 12) -> None:
        if lease_seconds < 3:
            raise ValueError("lease_seconds must be at least 3")
        self.engine = engine
        self.lease_seconds = lease_seconds

    def register(self, registration: EpochRegistration) -> None:
        if (
            not isinstance(registration.pid, int)
            or isinstance(registration.pid, bool)
            or registration.pid < 0
        ):
            raise ValueError("epoch pid must be non-negative")
        now = utcnow()
        expires_at = now + timedelta(seconds=self.lease_seconds)
        with immediate_transaction(self.engine) as connection:
            connection.execute(
                insert(process_epochs).values(
                    id=registration.epoch_id,
                    role=registration.role,
                    token_hash=hash_epoch_token(registration.token),
                    pid=registration.pid,
                    status="active",
                    heartbeat_at=now,
                    lease_expires_at=expires_at,
                )
            )

    def validate(
        self,
        *,
        role: ProcessRole,
        epoch_id: str,
        token: str,
    ) -> bool:
        now = utcnow()
        with self.engine.connect() as connection:
            stored_hash = connection.execute(
                select(process_epochs.c.token_hash).where(
                    process_epochs.c.id == epoch_id,
                    process_epochs.c.role == role,
                    process_epochs.c.status == "active",
                    process_epochs.c.lease_expires_at > now,
                )
            ).scalar_one_or_none()
        return stored_hash is not None and secrets.compare_digest(
            stored_hash,
            hash_epoch_token(token),
        )

    def bind_pid(self, registration: EpochRegistration, pid: int) -> bool:
        if pid < 1:
            raise ValueError("child pid must be positive")
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == registration.epoch_id,
                    process_epochs.c.role == registration.role,
                    process_epochs.c.token_hash
                    == hash_epoch_token(registration.token),
                    process_epochs.c.status == "active",
                )
                .values(pid=pid, updated_at=utcnow())
            ).rowcount
        return changed == 1

    def renew(
        self,
        *,
        role: Literal["api", "worker"],
        epoch_id: str,
        token: str,
    ) -> bool:
        now = utcnow()
        expires_at = now + timedelta(seconds=self.lease_seconds)
        token_hash = hash_epoch_token(token)
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == epoch_id,
                    process_epochs.c.role == role,
                    process_epochs.c.status == "active",
                    process_epochs.c.token_hash == token_hash,
                    process_epochs.c.lease_expires_at > now,
                )
                .values(
                    heartbeat_at=now,
                    lease_expires_at=expires_at,
                    updated_at=now,
                )
            ).rowcount
        return changed == 1

    def active_epochs(self, role: Literal["api", "worker"]) -> list[str]:
        with self.engine.connect() as connection:
            return list(
                connection.execute(
                    select(process_epochs.c.id).where(
                        process_epochs.c.role == role,
                        process_epochs.c.status == "active",
                    )
                ).scalars()
            )

    def active_epoch_processes(
        self,
        role: Literal["api", "worker"],
    ) -> list[tuple[str, int]]:
        with self.engine.connect() as connection:
            return [
                (str(epoch_id), int(pid))
                for epoch_id, pid in connection.execute(
                    select(process_epochs.c.id, process_epochs.c.pid).where(
                        process_epochs.c.role == role,
                        process_epochs.c.status == "active",
                    )
                )
            ]

    def is_active_epoch(
        self,
        *,
        role: Literal["api", "worker"],
        epoch_id: str,
    ) -> bool:
        now = utcnow()
        with self.engine.connect() as connection:
            return bool(
                connection.execute(
                    select(
                        exists().where(
                            process_epochs.c.id == epoch_id,
                            process_epochs.c.role == role,
                            process_epochs.c.status == "active",
                            process_epochs.c.lease_expires_at > now,
                        )
                    )
                ).scalar()
            )

    def reconcile_dead_worker(self, epoch_id: str) -> ReconcileResult:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            epoch_changed = connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == epoch_id,
                    process_epochs.c.role == "worker",
                    process_epochs.c.status == "active",
                )
                .values(
                    status="lost",
                    recovery_completed_at=now,
                    updated_at=now,
                )
            ).rowcount
            interrupted, requeued, transient_requeued, transient_cancelled = (
                self._converge_worker_work(
                    connection,
                    epoch_id=epoch_id,
                    now=now,
                    reason="WORKER_EPOCH_LOST",
                )
            )
        return ReconcileResult(
            epoch_id=epoch_id,
            role="worker",
            jobs_interrupted=interrupted,
            operations_requeued=requeued,
            transient_requests_requeued=transient_requeued,
            transient_requests_cancelled=transient_cancelled,
            changed=bool(
                epoch_changed
                or interrupted
                or requeued
                or transient_requeued
                or transient_cancelled
            ),
        )

    def reconcile_dead_api(self, epoch_id: str) -> ReconcileResult:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            epoch_changed = connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == epoch_id,
                    process_epochs.c.role == "api",
                    process_epochs.c.status == "active",
                )
                .values(
                    status="lost",
                    recovery_completed_at=now,
                    updated_at=now,
                )
            ).rowcount
            failed, requeued, renders = self._converge_api_work(
                connection,
                epoch_id=epoch_id,
                now=now,
                reason="API_EPOCH_LOST",
            )
        return ReconcileResult(
            epoch_id=epoch_id,
            role="api",
            operations_failed=failed,
            operations_requeued=requeued,
            renders_requeued=renders,
            changed=bool(epoch_changed or failed or requeued or renders),
        )

    def reconcile_orphaned_work(self) -> list[ReconcileResult]:
        """Converge active work whose recorded executor is no longer live."""

        with self.engine.connect() as connection:
            worker_epoch_ids = {
                str(value)
                for value in connection.execute(
                    select(jobs.c.worker_epoch_id)
                    .where(
                        jobs.c.status == "running",
                        jobs.c.worker_epoch_id.is_not(None),
                    )
                    .distinct()
                ).scalars()
            }
            worker_epoch_ids.update(
                str(value)
                for value in connection.execute(
                    select(operations.c.executor_epoch_id)
                    .where(
                        operations.c.status == "running",
                        operations.c.executor_role == "worker",
                        operations.c.executor_epoch_id.is_not(None),
                    )
                    .distinct()
                ).scalars()
            )
            worker_epoch_ids.update(
                str(value)
                for value in connection.execute(
                    select(transient_requests.c.worker_epoch_id)
                    .where(
                        transient_requests.c.status == "running",
                        transient_requests.c.worker_epoch_id.is_not(None),
                    )
                    .distinct()
                ).scalars()
            )
            api_epoch_ids = {
                str(value)
                for value in connection.execute(
                    select(operations.c.executor_epoch_id)
                    .where(
                        operations.c.status == "running",
                        operations.c.executor_role == "api",
                        operations.c.executor_epoch_id.is_not(None),
                    )
                    .distinct()
                ).scalars()
            }
            api_epoch_ids.update(
                str(value)
                for value in connection.execute(
                    select(render_requests.c.executor_epoch_id)
                    .where(
                        render_requests.c.status == "running",
                        render_requests.c.executor_epoch_id.is_not(None),
                    )
                    .distinct()
                ).scalars()
            )

        results: list[ReconcileResult] = []
        for epoch_id in sorted(worker_epoch_ids):
            if not self.is_active_epoch(role="worker", epoch_id=epoch_id):
                result = self.reconcile_dead_worker(epoch_id)
                if result.changed:
                    results.append(result)
        for epoch_id in sorted(api_epoch_ids):
            if not self.is_active_epoch(role="api", epoch_id=epoch_id):
                result = self.reconcile_dead_api(epoch_id)
                if result.changed:
                    results.append(result)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            worker_counts = self._converge_worker_work(
                connection,
                epoch_id=None,
                now=now,
                reason="WORKER_EPOCH_MISSING",
            )
            api_counts = self._converge_api_work(
                connection,
                epoch_id=None,
                now=now,
                reason="API_EPOCH_MISSING",
            )
        if any(worker_counts):
            interrupted, requeued, transient_requeued, transient_cancelled = (
                worker_counts
            )
            results.append(
                ReconcileResult(
                    epoch_id="missing",
                    role="worker",
                    jobs_interrupted=interrupted,
                    operations_requeued=requeued,
                    transient_requests_requeued=transient_requeued,
                    transient_requests_cancelled=transient_cancelled,
                    changed=True,
                )
            )
        if any(api_counts):
            failed, requeued, renders = api_counts
            results.append(
                ReconcileResult(
                    epoch_id="missing",
                    role="api",
                    operations_failed=failed,
                    operations_requeued=requeued,
                    renders_requeued=renders,
                    changed=True,
                )
            )
        return results

    def _converge_worker_work(
        self,
        connection: Connection,
        *,
        epoch_id: str | None,
        now: datetime,
        reason: str,
    ) -> tuple[int, int, int, int]:
        interrupted = JobQueueRepository(
            self.engine
        ).reconcile_lost_worker_jobs(
            connection,
            worker_epoch_id=epoch_id,
            now=now,
            reason=reason,
        )
        epoch_matches = (
            operations.c.executor_epoch_id.is_(None)
            if epoch_id is None
            else operations.c.executor_epoch_id == epoch_id
        )
        worker_operation_ids = [
            str(value)
            for value in connection.execute(
                select(operations.c.id).where(
                    epoch_matches,
                    operations.c.executor_role == "worker",
                    operations.c.status == "running",
                )
            ).scalars()
        ]
        requeued = connection.execute(
            update(operations)
            .where(
                epoch_matches,
                operations.c.executor_role == "worker",
                operations.c.status == "running",
            )
            .values(
                status="pending",
                executor_epoch_id=None,
                attempt_id=None,
                started_at=None,
                error_json=None,
                updated_at=now,
            )
        ).rowcount
        for operation_id in worker_operation_ids:
            connection.execute(
                insert(operation_events).values(
                    operation_id=operation_id,
                    type="operation_requeued",
                    payload_json=json.dumps(
                        {"reason": reason},
                        separators=(",", ":"),
                    ),
                    created_at=now,
                )
            )
        transient_epoch_matches = (
            transient_requests.c.worker_epoch_id.is_(None)
            if epoch_id is None
            else transient_requests.c.worker_epoch_id == epoch_id
        )
        transient_requeued = connection.execute(
            update(transient_requests)
            .where(
                transient_epoch_matches,
                transient_requests.c.status == "running",
                transient_requests.c.connection_open.is_(True),
            )
            .values(
                status="pending",
                worker_epoch_id=None,
                attempt_id=None,
                updated_at=now,
            )
        ).rowcount
        transient_cancelled = connection.execute(
            update(transient_requests)
            .where(
                transient_epoch_matches,
                transient_requests.c.status == "running",
                transient_requests.c.connection_open.is_(False),
            )
            .values(
                status="cancelled",
                worker_epoch_id=None,
                attempt_id=None,
                completed_at=now,
                updated_at=now,
            )
        ).rowcount
        return interrupted, requeued, transient_requeued, transient_cancelled

    def _converge_api_work(
        self,
        connection: Connection,
        *,
        epoch_id: str | None,
        now: datetime,
        reason: str,
    ) -> tuple[int, int, int]:
        epoch_matches = (
            operations.c.executor_epoch_id.is_(None)
            if epoch_id is None
            else operations.c.executor_epoch_id == epoch_id
        )
        remote_operation_ids = [
            str(value)
            for value in connection.execute(
                select(operations.c.id).where(
                    epoch_matches,
                    operations.c.executor_role == "api",
                    operations.c.status == "running",
                    operations.c.kind.in_(REMOTE_API_OPERATION_KINDS),
                )
            ).scalars()
        ]
        retryable_operation_ids = [
            str(value)
            for value in connection.execute(
                select(operations.c.id).where(
                    epoch_matches,
                    operations.c.executor_role == "api",
                    operations.c.status == "running",
                    ~operations.c.kind.in_(REMOTE_API_OPERATION_KINDS),
                )
            ).scalars()
        ]
        error = {
            "code": "API_EXECUTOR_LOST",
            "message": "API executor exited before publishing a result",
        }
        failed = connection.execute(
            update(operations)
            .where(
                epoch_matches,
                operations.c.executor_role == "api",
                operations.c.status == "running",
                operations.c.kind.in_(REMOTE_API_OPERATION_KINDS),
            )
            .values(
                status="failed",
                error_json=json.dumps(error, separators=(",", ":")),
                executor_epoch_id=None,
                attempt_id=None,
                finished_at=now,
                updated_at=now,
            )
        ).rowcount
        for operation_id in remote_operation_ids:
            connection.execute(
                insert(operation_events).values(
                    operation_id=operation_id,
                    type="operation_failed",
                    payload_json=json.dumps(
                        {"status": "failed", "error": error},
                        separators=(",", ":"),
                    ),
                    created_at=now,
                )
            )
        requeued = connection.execute(
            update(operations)
            .where(
                epoch_matches,
                operations.c.executor_role == "api",
                operations.c.status == "running",
                ~operations.c.kind.in_(REMOTE_API_OPERATION_KINDS),
            )
            .values(
                status="pending",
                executor_epoch_id=None,
                attempt_id=None,
                started_at=None,
                error_json=None,
                updated_at=now,
            )
        ).rowcount
        for operation_id in retryable_operation_ids:
            connection.execute(
                insert(operation_events).values(
                    operation_id=operation_id,
                    type="operation_requeued",
                    payload_json=json.dumps(
                        {"reason": reason},
                        separators=(",", ":"),
                    ),
                    created_at=now,
                )
            )
        render_epoch_matches = (
            render_requests.c.executor_epoch_id.is_(None)
            if epoch_id is None
            else render_requests.c.executor_epoch_id == epoch_id
        )
        render_page_ids = [
            str(value)
            for value in connection.execute(
                select(render_requests.c.page_id).where(
                    render_epoch_matches,
                    render_requests.c.status == "running",
                )
            ).scalars()
        ]
        renders = connection.execute(
            update(render_requests)
            .where(
                render_epoch_matches,
                render_requests.c.status == "running",
            )
            .values(
                status="pending",
                executor_epoch_id=None,
                attempt_id=None,
                rendering_revision=None,
                error_json=None,
                updated_at=now,
            )
        ).rowcount
        if render_page_ids:
            connection.execute(
                update(pages)
                .where(
                    pages.c.id.in_(render_page_ids),
                    pages.c.render_status == "rendering",
                )
                .values(render_status="stale", updated_at=now)
            )
        return failed, requeued, renders

    def close(self, registration: EpochRegistration) -> bool:
        now = utcnow()
        token_hash = hash_epoch_token(registration.token)
        with immediate_transaction(self.engine) as connection:
            stored_token_hash = connection.execute(
                select(process_epochs.c.token_hash).where(
                    process_epochs.c.id == registration.epoch_id,
                    process_epochs.c.role == registration.role,
                )
            ).scalar_one_or_none()
            if stored_token_hash is None or not secrets.compare_digest(
                str(stored_token_hash), token_hash
            ):
                return False
            epoch_changed = connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == registration.epoch_id,
                    process_epochs.c.role == registration.role,
                    process_epochs.c.token_hash == token_hash,
                    process_epochs.c.status == "active",
                )
                .values(
                    status="closed",
                    recovery_completed_at=now,
                    updated_at=now,
                )
            ).rowcount
            work_changed = False
            if registration.role == "worker":
                counts = self._converge_worker_work(
                    connection,
                    epoch_id=registration.epoch_id,
                    now=now,
                    reason="WORKER_STOPPED",
                )
                work_changed = any(counts)
            elif registration.role == "api":
                counts = self._converge_api_work(
                    connection,
                    epoch_id=registration.epoch_id,
                    now=now,
                    reason="API_STOPPED",
                )
                work_changed = any(counts)
        return bool(epoch_changed or work_changed)
