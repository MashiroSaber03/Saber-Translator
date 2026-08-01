"""Authoritative process epoch, heartbeat, and recovery repository."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import secrets
from typing import Literal

from sqlalchemy import Engine, delete, exists, func, insert, select, update

from src.backend_v2.timestamps import utcnow
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    api_executor_leases,
    chapter_write_intents,
    chapter_write_locks,
    job_events,
    job_steps,
    jobs,
    operations,
    process_epochs,
    render_requests,
    worker_leases,
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
    intents_removed: int = 0
    changed: bool = False


class ProcessEpochRepository:
    def __init__(self, engine: Engine, *, lease_seconds: int = 12) -> None:
        if lease_seconds < 3:
            raise ValueError("lease_seconds must be at least 3")
        self.engine = engine
        self.lease_seconds = lease_seconds

    def register(self, registration: EpochRegistration) -> None:
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
            if registration.role == "api":
                connection.execute(
                    insert(api_executor_leases).values(
                        api_epoch_id=registration.epoch_id,
                        lease_token=registration.token,
                        heartbeat_at=now,
                        lease_expires_at=expires_at,
                    )
                )
            elif registration.role == "worker":
                connection.execute(
                    insert(worker_leases).values(
                        worker_epoch_id=registration.epoch_id,
                        lease_token=registration.token,
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
        lease_table = api_executor_leases if role == "api" else worker_leases
        epoch_column = (
            lease_table.c.api_epoch_id
            if role == "api"
            else lease_table.c.worker_epoch_id
        )
        try:
            with self.engine.begin() as connection:
                epoch_result = connection.execute(
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
                )
                lease_result = connection.execute(
                    update(lease_table)
                    .where(
                        epoch_column == epoch_id,
                        lease_table.c.lease_token == token,
                        lease_table.c.lease_expires_at > now,
                    )
                    .values(heartbeat_at=now, lease_expires_at=expires_at)
                )
                if epoch_result.rowcount != 1 or lease_result.rowcount != 1:
                    raise _RenewalLost
        except _RenewalLost:
            return False
        return True

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

    def is_active_epoch(
        self,
        *,
        role: Literal["api", "worker"],
        epoch_id: str,
    ) -> bool:
        with self.engine.connect() as connection:
            return bool(
                connection.execute(
                    select(
                        exists().where(
                            process_epochs.c.id == epoch_id,
                            process_epochs.c.role == role,
                            process_epochs.c.status == "active",
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
                    .join(
                        worker_leases,
                        worker_leases.c.worker_epoch_id == process_epochs.c.id,
                    )
                    .where(
                        process_epochs.c.role == "worker",
                        process_epochs.c.status == "active",
                        process_epochs.c.lease_expires_at <= now,
                        worker_leases.c.lease_expires_at <= now,
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

            affected_jobs = list(
                connection.execute(
                    select(
                        jobs.c.id,
                        jobs.c.status,
                        jobs.c.attempt_id,
                    ).where(
                        jobs.c.worker_epoch_id == epoch_id,
                        jobs.c.status.in_(("running", "pausing", "cancelling")),
                    )
                ).mappings()
            )
            attempt_ids = [
                str(row["attempt_id"])
                for row in affected_jobs
                if row["attempt_id"] is not None
            ]
            if attempt_ids:
                connection.execute(
                    update(job_steps)
                    .where(
                        job_steps.c.status == "running",
                        job_steps.c.attempt_id.in_(attempt_ids),
                    )
                    .values(
                        status="pending",
                        attempt_id=None,
                        updated_at=now,
                    )
                )
            interrupted = connection.execute(
                update(jobs)
                .where(
                    jobs.c.worker_epoch_id == epoch_id,
                    jobs.c.status.in_(("running", "pausing")),
                )
                .values(
                    status="interrupted",
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    worker_epoch_id=None,
                    updated_at=now,
                )
            ).rowcount
            cancelled = connection.execute(
                update(jobs)
                .where(
                    jobs.c.worker_epoch_id == epoch_id,
                    jobs.c.status == "cancelling",
                )
                .values(
                    status="cancelled",
                    queue_rank=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    worker_epoch_id=None,
                    finished_at=now,
                    updated_at=now,
                )
            ).rowcount
            cancelled_ids = [
                str(row["id"])
                for row in affected_jobs
                if row["status"] == "cancelling"
            ]
            if cancelled_ids:
                connection.execute(
                    delete(chapter_write_locks).where(
                        chapter_write_locks.c.job_id.in_(cancelled_ids)
                    )
                )
            next_event_id = int(
                connection.execute(
                    select(func.coalesce(func.max(job_events.c.id), 0))
                ).scalar_one()
            )
            for row in affected_jobs:
                next_event_id += 1
                final_status = (
                    "cancelled"
                    if row["status"] == "cancelling"
                    else "interrupted"
                )
                connection.execute(
                    insert(job_events).values(
                        id=next_event_id,
                        job_id=row["id"],
                        event_type=f"job_{final_status}",
                        payload_json=json.dumps(
                            {
                                "reason": "WORKER_EPOCH_LOST",
                                "workerEpochId": epoch_id,
                            },
                            separators=(",", ":"),
                        ),
                        payload_schema_version=1,
                        created_at=now,
                    )
                )
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
                    lease_token=None,
                    lease_expires_at=None,
                    started_at=None,
                    error_json=None,
                    updated_at=now,
                )
            ).rowcount
            intents_removed = connection.execute(
                delete(chapter_write_intents).where(
                    chapter_write_intents.c.worker_epoch_id == epoch_id
                )
            ).rowcount
            connection.execute(
                delete(worker_leases).where(
                    worker_leases.c.worker_epoch_id == epoch_id
                )
            )
        return ReconcileResult(
            epoch_id=epoch_id,
            role="worker",
            jobs_interrupted=interrupted,
            jobs_cancelled=cancelled,
            operations_requeued=requeued,
            intents_removed=intents_removed,
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
                        {
                            "code": "API_EXECUTOR_LOST",
                            "message": "API executor exited before publishing a result",
                        },
                        separators=(",", ":"),
                    ),
                    executor_epoch_id=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    finished_at=now,
                    updated_at=now,
                )
            ).rowcount
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
                    lease_token=None,
                    lease_expires_at=None,
                    started_at=None,
                    error_json=None,
                    updated_at=now,
                )
            ).rowcount
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
                    lease_token=None,
                    lease_expires_at=None,
                    rendering_revision=None,
                    error_json=None,
                    updated_at=now,
                )
            ).rowcount
            connection.execute(
                delete(api_executor_leases).where(
                    api_executor_leases.c.api_epoch_id == epoch_id
                )
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
            if registration.role == "api":
                connection.execute(
                    delete(api_executor_leases).where(
                        api_executor_leases.c.api_epoch_id == registration.epoch_id
                    )
                )
            elif registration.role == "worker":
                connection.execute(
                    delete(worker_leases).where(
                        worker_leases.c.worker_epoch_id == registration.epoch_id
                    )
                )
        return changed == 1


class _RenewalLost(RuntimeError):
    pass
