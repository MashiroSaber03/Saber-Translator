"""Authoritative process epoch, heartbeat, and recovery repository."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import secrets
from typing import Literal

from sqlalchemy import Engine, exists, insert, select, update

from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.timestamps import utcnow
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
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
    jobs_cancelled: int = 0
    operations_failed: int = 0
    operations_requeued: int = 0
    renders_requeued: int = 0
    transient_requests_requeued: int = 0
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

    def expired_worker_epochs(self) -> list[str]:
        now = utcnow()
        with self.engine.connect() as connection:
            return [
                str(value)
                for value in connection.execute(
                    select(process_epochs.c.id)
                    .where(
                        process_epochs.c.role == "worker",
                        process_epochs.c.status == "active",
                        process_epochs.c.lease_expires_at <= now,
                    )
                ).scalars()
            ]

    def reconcile_dead_worker(self, epoch_id: str) -> ReconcileResult:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            closed = connection.execute(
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
            )
            if closed.rowcount != 1:
                return ReconcileResult(epoch_id=epoch_id, role="worker")

            interrupted, cancelled = JobQueueRepository(
                self.engine
            ).reconcile_lost_worker_jobs(
                connection,
                worker_epoch_id=epoch_id,
                now=now,
            )
            worker_operation_ids = [
                str(value)
                for value in connection.execute(
                    select(operations.c.id).where(
                        operations.c.executor_epoch_id == epoch_id,
                        operations.c.executor_role == "worker",
                        operations.c.status == "running",
                    )
                ).scalars()
            ]
            requeued = connection.execute(
                update(operations)
                .where(
                    operations.c.executor_epoch_id == epoch_id,
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
                            {"reason": "WORKER_EPOCH_LOST"},
                            separators=(",", ":"),
                        ),
                        created_at=now,
                    )
                )
            transient_requeued = connection.execute(
                update(transient_requests)
                .where(
                    transient_requests.c.worker_epoch_id == epoch_id,
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
        return ReconcileResult(
            epoch_id=epoch_id,
            role="worker",
            jobs_interrupted=interrupted,
            jobs_cancelled=cancelled,
            operations_requeued=requeued,
            transient_requests_requeued=transient_requeued,
            changed=True,
        )

    def reconcile_dead_api(self, epoch_id: str) -> ReconcileResult:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            closed = connection.execute(
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
            )
            if closed.rowcount != 1:
                return ReconcileResult(epoch_id=epoch_id, role="api")

            remote_operation_ids = [
                str(value)
                for value in connection.execute(
                    select(operations.c.id).where(
                        operations.c.executor_epoch_id == epoch_id,
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
                        operations.c.executor_epoch_id == epoch_id,
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
                    operations.c.executor_epoch_id == epoch_id,
                    operations.c.executor_role == "api",
                    operations.c.status == "running",
                    operations.c.kind.in_(REMOTE_API_OPERATION_KINDS),
                )
                .values(
                    status="failed",
                    error_json=json.dumps(
                        error,
                        separators=(",", ":"),
                    ),
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
                    operations.c.executor_epoch_id == epoch_id,
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
                            {"reason": "API_EPOCH_LOST"},
                            separators=(",", ":"),
                        ),
                        created_at=now,
                    )
                )
            render_page_ids = [
                str(value)
                for value in connection.execute(
                    select(render_requests.c.page_id).where(
                        render_requests.c.executor_epoch_id == epoch_id,
                        render_requests.c.status == "running",
                    )
                ).scalars()
            ]
            renders = connection.execute(
                update(render_requests)
                .where(
                    render_requests.c.executor_epoch_id == epoch_id,
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
        return ReconcileResult(
            epoch_id=epoch_id,
            role="api",
            operations_failed=failed,
            operations_requeued=requeued,
            renders_requeued=renders,
            changed=True,
        )

    def close(self, registration: EpochRegistration) -> bool:
        now = utcnow()
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == registration.epoch_id,
                    process_epochs.c.role == registration.role,
                    process_epochs.c.token_hash == hash_epoch_token(registration.token),
                    process_epochs.c.status == "active",
                )
                .values(status="closed", updated_at=now)
            ).rowcount
        return changed == 1
