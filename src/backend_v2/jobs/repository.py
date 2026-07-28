"""Transactional repository for the single durable v2 job queue.

The database is the scheduler's source of truth.  In-memory queues only carry
IDs that can be reconstructed after a Worker restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import hashlib
import secrets
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

from sqlalchemy import (
    Engine,
    and_,
    delete,
    exists,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Connection

from src.backend_v2.domain.state_machines import (
    InvalidTransition,
    JobEvent,
    JobStatus,
    transition_job,
)
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    CURRENT_JOB_STATUSES,
    JOB_KINDS,
    NONTERMINAL_JOB_STATUSES,
    chapter_write_intents,
    chapter_write_locks,
    chapters,
    import_leases,
    idempotency_records,
    assets,
    job_asset_inputs,
    job_artifacts,
    job_batches,
    job_config_snapshots,
    job_credential_snapshots,
    job_drain_acks,
    job_events,
    job_items,
    job_font_snapshots,
    job_plugin_snapshots,
    job_steps,
    jobs,
    operations,
    page_assets,
    pages,
    process_epochs,
    queue_state,
    render_requests,
    analysis_runs,
    analysis_run_targets,
)


TERMINAL_JOB_STATUSES = (
    "cancelled",
    "completed",
    "completed_with_errors",
    "failed",
)
WRITE_JOB_KINDS = frozenset(
    {
        "translation",
        "remove_text",
        "detect",
        "style_apply",
        "text_import",
        "container_import",
        "web_import_commit",
    }
)
ACTIVE_OPERATION_STATUSES = ("pending", "running")
ACTIVE_RENDER_STATUSES = ("pending", "running")


class JobNotFound(LookupError):
    pass


class JobConflict(RuntimeError):
    pass


class InvalidJobTransition(JobConflict):
    pass


class AttemptFenced(JobConflict):
    pass


@dataclass(frozen=True, slots=True)
class JobItemSpec:
    page_id: str | None
    step_kinds: tuple[str, ...]
    asset_inputs: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.step_kinds:
            raise ValueError("every job item requires at least one step")
        if any(not value or len(value) > 64 for value in self.step_kinds):
            raise ValueError("step kinds must be non-empty and at most 64 characters")


@dataclass(frozen=True, slots=True)
class JobSpec:
    kind: str
    config: Mapping[str, Any]
    items: tuple[JobItemSpec, ...]
    book_id: str | None = None
    chapter_id: str | None = None
    page_id: str | None = None
    analysis_run_id: str | None = None
    continuation_project_id: str | None = None
    web_import_draft_id: str | None = None
    target_display: Mapping[str, Any] | None = None
    credential_snapshots: Mapping[str, str] | None = None
    font_snapshots: Mapping[str, str] | None = None
    plugin_snapshots: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.kind not in JOB_KINDS:
            raise ValueError(f"unsupported job kind: {self.kind}")
        if not self.items:
            raise ValueError("job requires at least one item")
        if self.kind in WRITE_JOB_KINDS and not self.chapter_id:
            raise ValueError(f"{self.kind} jobs require a chapter_id")


@dataclass(frozen=True, slots=True)
class AttemptFence:
    job_id: str
    attempt_id: str
    lease_token: str
    worker_epoch_id: str
    lease_expires_at: datetime


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


def _credential_version_references(
    value: Mapping[str, Any],
) -> dict[str, str]:
    references: dict[str, str] = {}

    def visit(current: object, path: tuple[str, ...]) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                key_text = str(key)
                next_path = (*path, key_text)
                if key_text == "credentialVersionId" and isinstance(child, str):
                    role = ".".join(path) or "default"
                    if len(role) > 64:
                        role = hashlib.sha256(role.encode("utf-8")).hexdigest()
                    references[role] = child
                else:
                    visit(child, next_path)
        elif isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                visit(child, (*path, str(index)))

    visit(value, ())
    return references


class JobQueueRepository:
    """Own all queue ordering, transition, checkpoint, and lock transactions."""

    def __init__(self, engine: Engine, *, attempt_lease_seconds: int = 30) -> None:
        if attempt_lease_seconds < 3:
            raise ValueError("attempt_lease_seconds must be at least 3")
        self.engine = engine
        self.attempt_lease_seconds = attempt_lease_seconds

    def create_batch(
        self,
        *,
        kind: str,
        display_name: str,
        specs: Sequence[JobSpec],
        idempotency_scope: str | None = None,
        idempotency_key: str | None = None,
        idempotency_payload: Mapping[str, Any] | None = None,
        transaction_initializer: (
            Callable[[Connection, str], None] | None
        ) = None,
        transaction_hook: (
            Callable[[Connection, str, Sequence[str]], None] | None
        ) = None,
    ) -> dict[str, object]:
        if not specs:
            raise ValueError("a batch requires at least one job")
        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("batch display name is required")
        now = utcnow()
        batch_id = str(uuid.uuid4())
        created_ids: list[str] = []
        if bool(idempotency_scope) != bool(idempotency_key):
            raise ValueError("idempotency scope and key must be provided together")
        request_hash = (
            hashlib.sha256(
                _json(dict(idempotency_payload or {})).encode("utf-8")
            ).hexdigest()
            if idempotency_scope
            else None
        )
        try:
            with immediate_transaction(self.engine) as connection:
                if idempotency_scope and idempotency_key and request_hash:
                    replay = connection.execute(
                        select(
                            idempotency_records.c.request_hash,
                            idempotency_records.c.response_json,
                        ).where(
                            idempotency_records.c.scope == idempotency_scope,
                            idempotency_records.c.key == idempotency_key,
                            idempotency_records.c.expires_at > now,
                        )
                    ).mappings().one_or_none()
                    if replay is not None:
                        if replay["request_hash"] != request_hash:
                            raise JobConflict(
                                "Idempotency-Key was reused for different job input"
                            )
                        return json.loads(replay["response_json"])
                connection.execute(
                    insert(job_batches).values(
                        id=batch_id,
                        kind=kind,
                        display_name=normalized_name,
                        status_summary_json=_json(
                            {"total": len(specs), "queued": len(specs)}
                        ),
                        created_at=now,
                        updated_at=now,
                    )
                )
                if transaction_initializer is not None:
                    transaction_initializer(connection, batch_id)
                next_rank = int(
                    connection.execute(
                        select(func.coalesce(func.max(jobs.c.queue_rank), 0))
                    ).scalar_one()
                )
                for spec in specs:
                    next_rank += 1
                    job_id = str(uuid.uuid4())
                    created_ids.append(job_id)
                    connection.execute(
                        insert(jobs).values(
                            id=job_id,
                            batch_id=batch_id,
                            kind=spec.kind,
                            status="queued",
                            queue_rank=next_rank,
                            book_id=spec.book_id,
                            chapter_id=spec.chapter_id,
                            page_id=spec.page_id,
                            analysis_run_id=spec.analysis_run_id,
                            continuation_project_id=(
                                spec.continuation_project_id
                            ),
                            web_import_draft_id=spec.web_import_draft_id,
                            config_json=_json(dict(spec.config)),
                            config_schema_version=1,
                            latest_progress_json=_json(
                                {
                                    "completedItems": 0,
                                    "failedItems": 0,
                                    "totalItems": len(spec.items),
                                }
                            ),
                            target_display_json=_json(dict(spec.target_display or {})),
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    connection.execute(
                        insert(job_config_snapshots).values(
                            job_id=job_id,
                            payload_json=_json(dict(spec.config)),
                            schema_version=1,
                        )
                    )
                    credential_refs = {
                        **_credential_version_references(spec.config),
                        **dict(spec.credential_snapshots or {}),
                    }
                    if credential_refs:
                        connection.execute(
                            insert(job_credential_snapshots),
                            [
                                {
                                    "job_id": job_id,
                                    "credential_version_id": version_id,
                                    "role": role,
                                }
                                for role, version_id in credential_refs.items()
                            ],
                        )
                    if spec.font_snapshots:
                        connection.execute(
                            insert(job_font_snapshots),
                            [
                                {
                                    "job_id": job_id,
                                    "font_id": font_id,
                                    "role": role,
                                }
                                for role, font_id in spec.font_snapshots.items()
                            ],
                        )
                    if spec.plugin_snapshots:
                        connection.execute(
                            insert(job_plugin_snapshots),
                            [
                                {
                                    "job_id": job_id,
                                    "plugin_version_id": version_id,
                                    "config_json": _json(dict(plugin_config)),
                                }
                                for version_id, plugin_config in (
                                    spec.plugin_snapshots.items()
                                )
                            ],
                        )
                    for item_ordinal, item_spec in enumerate(spec.items, start=1):
                        item_id = str(uuid.uuid4())
                        connection.execute(
                            insert(job_items).values(
                                id=item_id,
                                job_id=job_id,
                                ordinal=item_ordinal,
                                page_id=item_spec.page_id,
                                status="pending",
                                created_at=now,
                                updated_at=now,
                            )
                        )
                        connection.execute(
                            insert(job_steps),
                            [
                                {
                                    "id": str(uuid.uuid4()),
                                    "job_item_id": item_id,
                                    "ordinal": step_ordinal,
                                    "kind": step_kind,
                                    "status": "pending",
                                    "checkpoint_schema_version": 1,
                                    "created_at": now,
                                    "updated_at": now,
                                }
                                for step_ordinal, step_kind in enumerate(
                                    item_spec.step_kinds, start=1
                                )
                            ],
                        )
                        if item_spec.asset_inputs:
                            connection.execute(
                                insert(job_asset_inputs),
                                [
                                    {
                                        "job_id": job_id,
                                        "asset_id": asset_id,
                                        "role": role,
                                        "binding_phase": "create",
                                        "job_item_id": item_id,
                                    }
                                    for role, asset_id in item_spec.asset_inputs.items()
                                ],
                            )
                    self._append_event(
                        connection,
                        job_id=job_id,
                        event_type="job_created",
                        payload={"batchId": batch_id, "queueRank": next_rank},
                        now=now,
                    )
                if transaction_hook is not None:
                    transaction_hook(
                        connection,
                        batch_id,
                        tuple(created_ids),
                    )
                self._bump_queue_revision(connection, now)
                response = {
                    "batchId": batch_id,
                    "jobIds": created_ids,
                    "status": "queued",
                }
                if idempotency_scope and idempotency_key and request_hash:
                    connection.execute(
                        insert(idempotency_records).values(
                            scope=idempotency_scope,
                            key=idempotency_key,
                            request_hash=request_hash,
                            http_status=202,
                            response_json=_json(response),
                            resource_type="job_batch",
                            resource_id=batch_id,
                            created_at=now,
                            expires_at=now + timedelta(days=7),
                        )
                    )
        except IntegrityError as exc:
            raise JobConflict("a conflicting nonterminal job already exists") from exc
        return response

    def list_jobs(
        self,
        *,
        scope: str = "queue",
        status: str | None = None,
        kind: str | None = None,
        book_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, object]:
        if scope not in {"queue", "history"}:
            raise ValueError("scope must be queue or history")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        conditions = []
        if scope == "queue":
            conditions.append(jobs.c.status.in_(NONTERMINAL_JOB_STATUSES))
        else:
            conditions.append(
                jobs.c.status.in_((*TERMINAL_JOB_STATUSES, "interrupted"))
            )
        if status:
            conditions.append(jobs.c.status == status)
        if kind:
            conditions.append(jobs.c.kind == kind)
        if book_id:
            conditions.append(jobs.c.book_id == book_id)
        order = (
            (jobs.c.queue_rank.asc(), jobs.c.created_at.asc())
            if scope == "queue"
            else (jobs.c.finished_at.desc(), jobs.c.created_at.desc())
        )
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(jobs, job_batches.c.display_name.label("batch_display_name"))
                .join(job_batches, job_batches.c.id == jobs.c.batch_id, isouter=True)
                .where(*conditions)
                .order_by(*order)
                .limit(limit)
            ).mappings()
            revision = connection.execute(
                select(queue_state.c.queue_revision).where(
                    queue_state.c.singleton_id == 1
                )
            ).scalar_one()
            return {
                "items": [self._job_dto(row) for row in rows],
                "queueRevision": int(revision),
            }

    def get_job(self, job_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            job = connection.execute(
                select(jobs, job_batches.c.display_name.label("batch_display_name"))
                .join(job_batches, job_batches.c.id == jobs.c.batch_id, isouter=True)
                .where(jobs.c.id == job_id)
            ).mappings().one_or_none()
            if job is None:
                raise JobNotFound("job not found")
            item_rows = connection.execute(
                select(job_items)
                .where(job_items.c.job_id == job_id)
                .order_by(job_items.c.ordinal)
            ).mappings()
            items: list[dict[str, object]] = []
            for row in item_rows:
                step_rows = connection.execute(
                    select(job_steps)
                    .where(job_steps.c.job_item_id == row["id"])
                    .order_by(job_steps.c.ordinal)
                ).mappings()
                items.append(
                    {
                        "itemId": row["id"],
                        "ordinal": row["ordinal"],
                        "pageId": row["page_id"],
                        "status": row["status"],
                        "result": _load_json(row["result_json"], None),
                        "error": _load_json(row["error_json"], None),
                        "steps": [
                            {
                                "stepId": step["id"],
                                "ordinal": step["ordinal"],
                                "kind": step["kind"],
                                "status": step["status"],
                                "checkpoint": _load_json(
                                    step["checkpoint_json"], None
                                ),
                                "error": _load_json(step["error_json"], None),
                            }
                            for step in step_rows
                        ],
                    }
                )
            artifact_rows = list(
                connection.execute(
                    select(job_artifacts).where(
                        job_artifacts.c.job_id == job_id
                    )
                ).mappings()
            )
        result = self._job_dto(job)
        result["items"] = items
        result["artifacts"] = [
            {
                "kind": row["kind"],
                "assetId": row["asset_id"],
                "url": f"/api/v2/assets/{row['asset_id']}",
                "expiresAt": self._iso(row["expires_at"]),
            }
            for row in artifact_rows
        ]
        return result

    def get_batch(self, batch_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            batch = connection.execute(
                select(job_batches).where(job_batches.c.id == batch_id)
            ).mappings().one_or_none()
            if batch is None:
                raise JobNotFound("job batch not found")
            member_rows = connection.execute(
                select(jobs)
                .where(jobs.c.batch_id == batch_id)
                .order_by(jobs.c.queue_rank, jobs.c.created_at)
            ).mappings()
        return {
            "batchId": batch["id"],
            "kind": batch["kind"],
            "displayName": batch["display_name"],
            "summary": _load_json(batch["status_summary_json"], {}),
            "jobs": [self._job_dto(row) for row in member_rows],
            "createdAt": self._iso(batch["created_at"]),
        }

    def events_after(
        self,
        *,
        after: int = 0,
        job_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        if after < 0:
            raise ValueError("event cursor must be nonnegative")
        if limit < 1 or limit > 1000:
            raise ValueError("event limit must be between 1 and 1000")
        condition = job_events.c.id > after
        if job_id:
            condition = and_(condition, job_events.c.job_id == job_id)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(job_events)
                .where(condition)
                .order_by(job_events.c.id)
                .limit(limit)
            ).mappings()
        return [
                {
                    "eventId": int(row["id"]),
                    "jobId": row["job_id"],
                    "type": row["event_type"],
                    "payload": _load_json(row["payload_json"], {}),
                    "createdAt": self._iso(row["created_at"]),
                }
            for row in rows
        ]

    def latest_event_id(self) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    select(func.coalesce(func.max(job_events.c.id), 0))
                ).scalar_one()
            )

    def reorder(self, *, ordered_job_ids: Sequence[str], base_revision: int) -> int:
        if not ordered_job_ids or len(set(ordered_job_ids)) != len(ordered_job_ids):
            raise ValueError("orderedJobIds must contain unique job IDs")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            revision = int(
                connection.execute(
                    select(queue_state.c.queue_revision).where(
                        queue_state.c.singleton_id == 1
                    )
                ).scalar_one()
            )
            if revision != base_revision:
                raise JobConflict("queue revision changed")
            sortable = list(
                connection.execute(
                    select(jobs.c.id)
                    .where(
                        jobs.c.status == "queued",
                        ~jobs.c.id.in_(
                            select(chapter_write_intents.c.job_id)
                        ),
                        ~jobs.c.id.in_(select(chapter_write_locks.c.job_id)),
                    )
                    .order_by(jobs.c.queue_rank)
                ).scalars()
            )
            if set(sortable) != set(ordered_job_ids):
                raise JobConflict(
                    "only the complete ordinary queued set may be reordered"
                )
            # Avoid transient UNIQUE(queue_rank) collisions.
            for temporary_rank, job_id in enumerate(ordered_job_ids, start=1):
                connection.execute(
                    update(jobs)
                    .where(jobs.c.id == job_id)
                    .values(queue_rank=-temporary_rank, updated_at=now)
                )
            prefix_max = int(
                connection.execute(
                    select(func.coalesce(func.max(jobs.c.queue_rank), 0)).where(
                        or_(
                            jobs.c.status != "queued",
                            jobs.c.id.not_in(ordered_job_ids),
                        ),
                        jobs.c.queue_rank.is_not(None),
                    )
                ).scalar_one()
            )
            for offset, job_id in enumerate(ordered_job_ids, start=1):
                connection.execute(
                    update(jobs)
                    .where(jobs.c.id == job_id)
                    .values(queue_rank=prefix_max + offset, updated_at=now)
                )
                self._append_event(
                    connection,
                    job_id=job_id,
                    event_type="job_reordered",
                    payload={"queueRank": prefix_max + offset},
                    now=now,
                )
            return self._bump_queue_revision(connection, now)

    def request_pause(self, job_id: str) -> dict[str, object]:
        return self._command(job_id, JobEvent.REQUEST_PAUSE)

    def request_cancel(self, job_id: str) -> dict[str, object]:
        return self._command(job_id, JobEvent.REQUEST_CANCEL)

    def resume(self, job_id: str) -> dict[str, object]:
        return self._command(job_id, JobEvent.RESUME)

    def continue_interrupted(self, job_id: str) -> dict[str, object]:
        return self._command(job_id, JobEvent.CONTINUE)

    def cancel_all_queued(self) -> int:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            ids = list(
                connection.execute(
                    select(jobs.c.id).where(jobs.c.status == "queued")
                ).scalars()
            )
            for job_id in ids:
                connection.execute(
                    update(jobs)
                    .where(jobs.c.id == job_id, jobs.c.status == "queued")
                    .values(
                        status="cancelled",
                        queue_rank=None,
                        finished_at=now,
                        updated_at=now,
                    )
                )
                self._release_write_reservations(connection, job_id)
                self._sync_domain_terminal(
                    connection,
                    job_id=str(job_id),
                    status="cancelled",
                    now=now,
                )
                self._append_event(
                    connection,
                    job_id=job_id,
                    event_type="job_cancelled",
                    payload={"source": "cancel_all_queued"},
                    now=now,
                )
            if ids:
                self._bump_queue_revision(connection, now)
            return len(ids)

    def clear_history(self) -> int:
        with immediate_transaction(self.engine) as connection:
            removable = list(
                connection.execute(
                    select(jobs.c.id).where(jobs.c.status.in_(TERMINAL_JOB_STATUSES))
                ).scalars()
            )
            if removable:
                connection.execute(delete(jobs).where(jobs.c.id.in_(removable)))
            return len(removable)

    def claim_next(self, *, worker_epoch_id: str) -> AttemptFence | None:
        """Claim the next executable job or advance its write-intent barrier."""

        now = utcnow()
        expires = now + timedelta(seconds=self.attempt_lease_seconds)
        with immediate_transaction(self.engine) as connection:
            self._assert_worker_epoch(connection, worker_epoch_id, now)
            current = connection.execute(
                select(jobs.c.id).where(jobs.c.status.in_(CURRENT_JOB_STATUSES))
            ).scalar_one_or_none()
            if current is not None:
                return None

            candidates = list(
                connection.execute(
                    select(jobs)
                    .where(jobs.c.status == "queued")
                    .order_by(jobs.c.queue_rank, jobs.c.created_at)
                ).mappings()
            )
            for candidate in candidates:
                job_id = str(candidate["id"])
                if candidate["kind"] in WRITE_JOB_KINDS:
                    reservation = self._advance_write_reservation(
                        connection,
                        candidate=candidate,
                        worker_epoch_id=worker_epoch_id,
                        now=now,
                        expires=expires,
                    )
                    if reservation == "draining":
                        return None
                    if reservation == "blocked":
                        continue
                fence = self._claim_row(
                    connection,
                    candidate=candidate,
                    worker_epoch_id=worker_epoch_id,
                    now=now,
                    expires=expires,
                )
                return fence
        return None

    def renew_attempt(self, fence: AttemptFence) -> AttemptFence | None:
        now = utcnow()
        expires = now + timedelta(seconds=self.attempt_lease_seconds)
        with self.engine.begin() as connection:
            result = connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                    jobs.c.worker_epoch_id == fence.worker_epoch_id,
                    jobs.c.lease_expires_at > now,
                    jobs.c.status.in_(("running", "pausing", "cancelling")),
                    exists(
                        select(process_epochs.c.id).where(
                            process_epochs.c.id == fence.worker_epoch_id,
                            process_epochs.c.role == "worker",
                            process_epochs.c.status == "active",
                            process_epochs.c.lease_expires_at > now,
                        )
                    ),
                )
                .values(lease_expires_at=expires, updated_at=now)
            )
            if result.rowcount != 1:
                return None
            connection.execute(
                update(chapter_write_intents)
                .where(
                    chapter_write_intents.c.job_id == fence.job_id,
                    chapter_write_intents.c.worker_epoch_id
                    == fence.worker_epoch_id,
                    chapter_write_intents.c.lease_token == fence.lease_token,
                )
                .values(lease_expires_at=expires)
            )
        return AttemptFence(
            job_id=fence.job_id,
            attempt_id=fence.attempt_id,
            lease_token=fence.lease_token,
            worker_epoch_id=fence.worker_epoch_id,
            lease_expires_at=expires,
        )

    def control_status(self, fence: AttemptFence) -> str:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(jobs.c.status).where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                    jobs.c.worker_epoch_id == fence.worker_epoch_id,
                )
            ).scalar_one_or_none()
        if value is None:
            raise AttemptFenced("job attempt lost execution rights")
        return str(value)

    def bind_item_inputs(
        self,
        fence: AttemptFence,
        *,
        item_id: str,
        page_id: str,
        roles: Sequence[str],
    ) -> dict[str, dict[str, object]]:
        """Freeze current page assets for one item and return their metadata."""

        if not roles:
            return {}
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running",),
            )
            owns_item = connection.execute(
                select(job_items.c.id).where(
                    job_items.c.id == item_id,
                    job_items.c.job_id == fence.job_id,
                    job_items.c.page_id == page_id,
                )
            ).scalar_one_or_none()
            if owns_item is None:
                raise JobConflict("job item does not own the requested page")
            result: dict[str, dict[str, object]] = {}
            for role in roles:
                existing = connection.execute(
                    select(
                        assets.c.id,
                        assets.c.relative_path,
                        assets.c.mime_type,
                        assets.c.checksum,
                        assets.c.width,
                        assets.c.height,
                    )
                    .join(
                        job_asset_inputs,
                        job_asset_inputs.c.asset_id == assets.c.id,
                    )
                    .where(
                        job_asset_inputs.c.job_id == fence.job_id,
                        job_asset_inputs.c.job_item_id == item_id,
                        job_asset_inputs.c.role == role,
                    )
                ).mappings().one_or_none()
                if existing is None:
                    current = connection.execute(
                        select(
                            assets.c.id,
                            assets.c.relative_path,
                            assets.c.mime_type,
                            assets.c.checksum,
                            assets.c.width,
                            assets.c.height,
                        )
                        .join(
                            page_assets,
                            page_assets.c.asset_id == assets.c.id,
                        )
                        .where(
                            page_assets.c.page_id == page_id,
                            page_assets.c.role == role,
                        )
                    ).mappings().one_or_none()
                    if current is None:
                        raise JobConflict(
                            f"page has no current {role} asset to bind"
                        )
                    connection.execute(
                        insert(job_asset_inputs).values(
                            job_id=fence.job_id,
                            asset_id=current["id"],
                            role=role,
                            binding_phase="item_start",
                            job_item_id=item_id,
                        )
                    )
                    existing = current
                result[role] = dict(existing)
            return result

    def attempt_config(self, fence: AttemptFence) -> dict[str, object]:
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            value = connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == fence.job_id)
            ).scalar_one()
        loaded = _load_json(value, {})
        if not isinstance(loaded, dict):
            raise JobConflict("job configuration snapshot is invalid")
        return loaded

    def append_plugin_event(
        self,
        fence: AttemptFence,
        *,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int:
        if not event_type.startswith("plugin_") or len(event_type) > 64:
            raise ValueError("plugin event type is invalid")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            return self._append_event(
                connection,
                job_id=fence.job_id,
                event_type=event_type,
                payload=payload,
                now=now,
            )

    def plugin_stage_completed(
        self,
        fence: AttemptFence,
        *,
        hook: str,
    ) -> bool:
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            rows = connection.execute(
                select(job_events.c.payload_json).where(
                    job_events.c.job_id == fence.job_id,
                    job_events.c.event_type
                    == "plugin_stage_completed",
                )
            ).scalars()
            return any(
                _load_json(str(payload), {}).get("hook") == hook
                for payload in rows
            )

    def active_step_counts(self, fence: AttemptFence) -> tuple[int, int]:
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            rows = list(connection.execute(
                select(job_steps.c.status, func.count())
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == fence.job_id,
                    job_steps.c.status.in_(("pending", "running")),
                )
                .group_by(job_steps.c.status)
            ))
        counts = {str(status): int(count) for status, count in rows}
        return counts.get("pending", 0), counts.get("running", 0)

    def next_step(
        self,
        fence: AttemptFence,
        *,
        allowed_kinds: Sequence[str] | None = None,
    ) -> dict[str, object] | None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(connection, fence, now, allowed_statuses=("running",))
            prior_step = job_steps.alias("prior_step")
            conditions = [
                jobs.c.id == fence.job_id,
                job_items.c.status.in_(("pending", "running")),
                job_steps.c.status == "pending",
                ~exists(
                    select(prior_step.c.id).where(
                        prior_step.c.job_item_id == job_steps.c.job_item_id,
                        prior_step.c.ordinal < job_steps.c.ordinal,
                        prior_step.c.status.in_(("pending", "running")),
                    )
                ),
            ]
            if allowed_kinds:
                conditions.append(job_steps.c.kind.in_(tuple(allowed_kinds)))
            row = connection.execute(
                select(
                    job_steps.c.id.label("step_id"),
                    job_steps.c.kind.label("step_kind"),
                    job_steps.c.ordinal.label("step_ordinal"),
                    job_items.c.id.label("item_id"),
                    job_items.c.ordinal.label("item_ordinal"),
                    job_items.c.page_id,
                    jobs.c.kind.label("job_kind"),
                    jobs.c.config_json,
                )
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .join(jobs, jobs.c.id == job_items.c.job_id)
                .where(*conditions)
                .order_by(job_items.c.ordinal, job_steps.c.ordinal)
                .limit(1)
            ).mappings().one_or_none()
            if row is None:
                return None
            connection.execute(
                update(job_items)
                .where(job_items.c.id == row["item_id"])
                .values(status="running", updated_at=now)
            )
            claimed = connection.execute(
                update(job_steps)
                .where(
                    job_steps.c.id == row["step_id"],
                    job_steps.c.status == "pending",
                )
                .values(
                    status="running",
                    attempt_id=fence.attempt_id,
                    updated_at=now,
                )
            )
            if claimed.rowcount != 1:
                raise AttemptFenced("step was claimed by another attempt")
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type="step_started",
                payload={
                    "itemId": row["item_id"],
                    "pageId": row["page_id"],
                    "stepId": row["step_id"],
                    "stepKind": row["step_kind"],
                },
                now=now,
            )
            return {
                "jobId": fence.job_id,
                "jobKind": row["job_kind"],
                "config": _load_json(row["config_json"], {}),
                "itemId": row["item_id"],
                "itemOrdinal": row["item_ordinal"],
                "pageId": row["page_id"],
                "stepId": row["step_id"],
                "stepOrdinal": row["step_ordinal"],
                "stepKind": row["step_kind"],
            }

    def complete_step(
        self,
        fence: AttemptFence,
        *,
        step_id: str,
        checkpoint: Mapping[str, Any],
        input_fingerprint: str | None = None,
        publisher: Callable[[Connection], None] | None = None,
    ) -> None:
        self._finish_step(
            fence,
            step_id=step_id,
            status="completed",
            checkpoint=checkpoint,
            error=None,
            input_fingerprint=input_fingerprint,
            publisher=publisher,
        )

    def fail_step(
        self,
        fence: AttemptFence,
        *,
        step_id: str,
        code: str,
        message: str,
        publisher: Callable[[Connection], None] | None = None,
    ) -> None:
        self._finish_step(
            fence,
            step_id=step_id,
            status="failed",
            checkpoint=None,
            error={"code": code, "message": message},
            input_fingerprint=None,
            publisher=publisher,
        )

    def finish_if_complete(self, fence: AttemptFence) -> str | None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running",),
            )
            active_steps = int(
                connection.execute(
                    select(func.count())
                    .select_from(job_steps)
                    .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                    .where(
                        job_items.c.job_id == fence.job_id,
                        job_steps.c.status.in_(("pending", "running")),
                    )
                ).scalar_one()
            )
            if active_steps:
                return None
            failed = int(
                connection.execute(
                    select(func.count())
                    .select_from(job_items)
                    .where(
                        job_items.c.job_id == fence.job_id,
                        job_items.c.status == "failed",
                    )
                ).scalar_one()
            )
            final = "completed_with_errors" if failed else "completed"
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                )
                .values(
                    status=final,
                    queue_rank=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    worker_epoch_id=None,
                    finished_at=now,
                    latest_progress_json=_json(
                        self._progress_snapshot(connection, fence.job_id)
                    ),
                    updated_at=now,
                )
            )
            self._release_write_reservations(connection, fence.job_id)
            self._sync_domain_terminal(
                connection,
                job_id=fence.job_id,
                status=final,
                now=now,
            )
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type="job_finished",
                payload={"status": final, "failedItems": failed},
                now=now,
            )
            self._bump_queue_revision(connection, now)
            self._refresh_batch_summary(connection, fence.job_id, now)
            return final

    def write_progress(
        self,
        fence: AttemptFence,
        progress: Mapping[str, Any],
    ) -> None:
        now = utcnow()
        with self.engine.begin() as connection:
            result = connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                    jobs.c.worker_epoch_id == fence.worker_epoch_id,
                    jobs.c.lease_expires_at > now,
                    jobs.c.status.in_(("running", "pausing", "cancelling")),
                )
                .values(latest_progress_json=_json(dict(progress)), updated_at=now)
            )
            if result.rowcount != 1:
                raise AttemptFenced("job progress write was fenced")

    def acknowledge_drain(
        self,
        fence: AttemptFence,
        *,
        pool_id: str,
        worker_slot: int,
        last_step_id: str | None,
    ) -> None:
        if worker_slot < 0:
            raise ValueError("worker_slot must be nonnegative")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            status = self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("pausing", "cancelling"),
            )
            connection.execute(
                insert(job_drain_acks)
                .values(
                    job_id=fence.job_id,
                    attempt_id=fence.attempt_id,
                    pool_id=pool_id,
                    worker_slot=worker_slot,
                    last_step_id=last_step_id,
                    created_at=now,
                )
                .prefix_with("OR REPLACE")
            )
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type="drain_acknowledged",
                payload={
                    "status": status,
                    "poolId": pool_id,
                    "workerSlot": worker_slot,
                },
                now=now,
            )

    def finalize_drain(
        self,
        fence: AttemptFence,
        *,
        expected_slots: Iterable[tuple[str, int]],
    ) -> str:
        expected = set(expected_slots)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            status = self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("pausing", "cancelling"),
            )
            active = int(
                connection.execute(
                    select(func.count())
                    .select_from(job_steps)
                    .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                    .where(
                        job_items.c.job_id == fence.job_id,
                        job_steps.c.status == "running",
                    )
                ).scalar_one()
            )
            actual = set(
                connection.execute(
                    select(
                        job_drain_acks.c.pool_id,
                        job_drain_acks.c.worker_slot,
                    ).where(
                        job_drain_acks.c.job_id == fence.job_id,
                        job_drain_acks.c.attempt_id == fence.attempt_id,
                    )
                ).tuples()
            )
            if active or actual != expected:
                raise JobConflict("job has not reached a fully acknowledged safe point")
            final = "paused" if status == "pausing" else "cancelled"
            values: dict[str, object] = {
                "status": final,
                "attempt_id": None,
                "lease_token": None,
                "lease_expires_at": None,
                "worker_epoch_id": None,
                "updated_at": now,
            }
            if final == "cancelled":
                values.update(queue_rank=None, finished_at=now)
                connection.execute(
                    update(job_steps)
                    .where(
                        job_steps.c.status == "pending",
                        job_steps.c.job_item_id.in_(
                            select(job_items.c.id).where(
                                job_items.c.job_id == fence.job_id
                            )
                        ),
                    )
                    .values(status="cancelled", updated_at=now)
                )
                connection.execute(
                    update(job_items)
                    .where(
                        job_items.c.job_id == fence.job_id,
                        job_items.c.status == "pending",
                    )
                    .values(status="cancelled", updated_at=now)
                )
                self._release_write_reservations(connection, fence.job_id)
                self._sync_domain_terminal(
                    connection,
                    job_id=fence.job_id,
                    status="cancelled",
                    now=now,
                )
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                )
                .values(**values)
            )
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type=f"job_{final}",
                payload={"source": "worker_drain"},
                now=now,
            )
            self._bump_queue_revision(connection, now)
            self._refresh_batch_summary(connection, fence.job_id, now)
            return final

    def fail_job(
        self,
        fence: AttemptFence,
        *,
        code: str,
        message: str,
    ) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing"),
            )
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                )
                .values(
                    status="failed",
                    queue_rank=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    worker_epoch_id=None,
                    finished_at=now,
                    latest_progress_json=_json(
                        {"error": {"code": code, "message": message}}
                    ),
                    updated_at=now,
                )
            )
            self._release_write_reservations(connection, fence.job_id)
            self._sync_domain_terminal(
                connection,
                job_id=fence.job_id,
                status="failed",
                now=now,
            )
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type="job_failed",
                payload={"code": code, "message": message},
                now=now,
            )
            self._bump_queue_revision(connection, now)
            self._refresh_batch_summary(connection, fence.job_id, now)

    def _finish_step(
        self,
        fence: AttemptFence,
        *,
        step_id: str,
        status: str,
        checkpoint: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
        input_fingerprint: str | None,
        publisher: Callable[[Connection], None] | None,
    ) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            job_status = self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            step = connection.execute(
                select(job_steps.c.job_item_id).where(
                    job_steps.c.id == step_id,
                    job_steps.c.status == "running",
                    job_steps.c.attempt_id == fence.attempt_id,
                    job_steps.c.job_item_id.in_(
                        select(job_items.c.id).where(
                            job_items.c.job_id == fence.job_id
                        )
                    ),
                )
            ).scalar_one_or_none()
            if step is None:
                raise AttemptFenced("step completion was fenced")
            if publisher is not None:
                publisher(connection)
            connection.execute(
                update(job_steps)
                .where(
                    job_steps.c.id == step_id,
                    job_steps.c.attempt_id == fence.attempt_id,
                )
                .values(
                    status=status,
                    input_fingerprint=input_fingerprint,
                    checkpoint_json=_json(dict(checkpoint)) if checkpoint else None,
                    error_json=_json(dict(error)) if error else None,
                    updated_at=now,
                )
            )
            item_id = str(step)
            if status == "failed":
                connection.execute(
                    update(job_items)
                    .where(job_items.c.id == item_id)
                    .values(status="failed", error_json=_json(dict(error or {})), updated_at=now)
                )
                connection.execute(
                    update(job_steps)
                    .where(
                        job_steps.c.job_item_id == item_id,
                        job_steps.c.status == "pending",
                    )
                    .values(status="skipped", updated_at=now)
                )
                event_type = "page_failed"
            else:
                pending = int(
                    connection.execute(
                        select(func.count())
                        .select_from(job_steps)
                        .where(
                            job_steps.c.job_item_id == item_id,
                            job_steps.c.status.in_(("pending", "running")),
                        )
                    ).scalar_one()
                )
                if pending == 0:
                    connection.execute(
                        update(job_items)
                        .where(job_items.c.id == item_id)
                        .values(
                            status="completed",
                            result_json=_json({"lastCheckpoint": dict(checkpoint or {})}),
                            updated_at=now,
                        )
                    )
                    event_type = "page_completed"
                else:
                    event_type = "step_completed"
            snapshot = self._progress_snapshot(connection, fence.job_id)
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                    jobs.c.status == job_status,
                )
                .values(latest_progress_json=_json(snapshot), updated_at=now)
            )
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type=event_type,
                payload={
                    "itemId": item_id,
                    "stepId": step_id,
                    "status": status,
                    "progress": snapshot,
                },
                now=now,
            )

    def _command(self, job_id: str, event: JobEvent) -> dict[str, object]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(jobs).where(jobs.c.id == job_id)
            ).mappings().one_or_none()
            if row is None:
                raise JobNotFound("job not found")
            current = JobStatus(str(row["status"]))

            # Repeated in-flight pause/cancel requests are idempotent.
            if event is JobEvent.REQUEST_PAUSE and current is JobStatus.PAUSING:
                return self._job_dto(row)
            if event is JobEvent.REQUEST_CANCEL and current in {
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
            }:
                return self._job_dto(row)
            try:
                new_status = transition_job(current, event)
            except InvalidTransition as exc:
                raise InvalidJobTransition(str(exc)) from exc

            values: dict[str, object] = {
                "status": new_status.value,
                "updated_at": now,
            }
            if new_status in {JobStatus.CANCELLED, JobStatus.QUEUED}:
                values.update(
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    worker_epoch_id=None,
                )
            if new_status is JobStatus.CANCELLED:
                values.update(queue_rank=None, finished_at=now)
                self._release_write_reservations(connection, job_id)
                self._sync_domain_terminal(
                    connection,
                    job_id=job_id,
                    status="cancelled",
                    now=now,
                )
            connection.execute(
                update(jobs).where(jobs.c.id == job_id).values(**values)
            )
            self._append_event(
                connection,
                job_id=job_id,
                event_type=f"job_{event.value}",
                payload={"from": current.value, "to": new_status.value},
                now=now,
            )
            self._bump_queue_revision(connection, now)
            self._refresh_batch_summary(connection, job_id, now)
            updated = dict(row)
            updated.update(values)
            return self._job_dto(updated)

    @staticmethod
    def _sync_domain_terminal(
        connection: Connection,
        *,
        job_id: str,
        status: str,
        now: datetime,
    ) -> None:
        """Converge staging domain runs when their owning job terminates."""

        run_id = connection.execute(
            select(analysis_runs.c.id).where(
                analysis_runs.c.job_id == job_id,
                analysis_runs.c.status == "staging",
            )
        ).scalar_one_or_none()
        if run_id is None:
            return
        rows = list(
            connection.execute(
                select(
                    analysis_run_targets.c.page_id_snapshot,
                    analysis_run_targets.c.status,
                ).where(analysis_run_targets.c.run_id == run_id)
            )
        )
        success_count = sum(
            1 for _page_id, target_status in rows
            if str(target_status) == "completed"
        )
        missing = [
            str(page_id)
            for page_id, target_status in rows
            if str(target_status) != "completed"
        ]
        terminal = "cancelled" if status == "cancelled" else "failed"
        connection.execute(
            update(analysis_runs)
            .where(
                analysis_runs.c.id == run_id,
                analysis_runs.c.status == "staging",
            )
            .values(
                status=terminal,
                success_count=success_count,
                failed_count=len(rows) - success_count,
                missing_page_ids_json=_json(missing),
                updated_at=now,
            )
        )

    def _advance_write_reservation(
        self,
        connection: Any,
        *,
        candidate: Mapping[str, Any],
        worker_epoch_id: str,
        now: datetime,
        expires: datetime,
    ) -> str:
        job_id = str(candidate["id"])
        chapter_ids = self._target_chapter_ids(connection, candidate)
        if not chapter_ids:
            raise JobConflict("chapter-writing job has no target chapters")

        existing_locks = list(
            connection.execute(
                select(chapter_write_locks).where(
                    chapter_write_locks.c.chapter_id.in_(chapter_ids)
                )
            ).mappings()
        )
        foreign_lock = next(
            (row for row in existing_locks if row["job_id"] != job_id), None
        )
        if foreign_lock is not None:
            self._set_blocked(
                connection,
                job_id=job_id,
                reason="blocked_by_job",
                blocked_job_id=str(foreign_lock["job_id"]),
                import_lease_id=None,
                now=now,
            )
            return "blocked"
        if len(existing_locks) == len(chapter_ids):
            return "ready"

        own_intents = list(
            connection.execute(
                select(chapter_write_intents).where(
                    chapter_write_intents.c.job_id == job_id
                )
            ).mappings()
        )
        if own_intents:
            if {str(row["chapter_id"]) for row in own_intents} != set(chapter_ids):
                raise JobConflict("job owns an incomplete write-intent set")
            connection.execute(
                update(chapter_write_intents)
                .where(
                    chapter_write_intents.c.job_id == job_id,
                    chapter_write_intents.c.worker_epoch_id == worker_epoch_id,
                )
                .values(lease_expires_at=expires)
            )
            if self._old_write_chains_active(connection, chapter_ids):
                return "draining"
            attempt_id = str(uuid.uuid4())
            lease_token = secrets.token_urlsafe(32)
            for intent in own_intents:
                connection.execute(
                    insert(chapter_write_locks).values(
                        chapter_id=intent["chapter_id"],
                        job_id=job_id,
                        lock_generation=intent["intent_generation"],
                        owner_attempt_id=attempt_id,
                        lease_token=lease_token,
                        created_at=now,
                    )
                )
            connection.execute(
                delete(chapter_write_intents).where(
                    chapter_write_intents.c.job_id == job_id
                )
            )
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id, jobs.c.status == "queued")
                .values(
                    attempt_id=attempt_id,
                    lease_token=lease_token,
                    lease_expires_at=expires,
                    worker_epoch_id=worker_epoch_id,
                    blocked_reason=None,
                    blocked_by_job_id=None,
                    blocked_by_import_lease_id=None,
                    updated_at=now,
                )
            )
            self._append_event(
                connection,
                job_id=job_id,
                event_type="chapter_write_lock_acquired",
                payload={"chapterIds": chapter_ids},
                now=now,
            )
            return "ready_preclaimed"

        active_import = connection.execute(
            select(import_leases.c.id)
            .where(
                import_leases.c.chapter_id.in_(chapter_ids),
                import_leases.c.expires_at > now,
            )
            .limit(1)
        ).scalar_one_or_none()
        if active_import is not None:
            self._set_blocked(
                connection,
                job_id=job_id,
                reason="blocked_by_import_lease",
                blocked_job_id=None,
                import_lease_id=str(active_import),
                now=now,
            )
            return "blocked"
        foreign_intent = connection.execute(
            select(chapter_write_intents.c.job_id)
            .where(
                chapter_write_intents.c.chapter_id.in_(chapter_ids),
                chapter_write_intents.c.job_id != job_id,
            )
            .limit(1)
        ).scalar_one_or_none()
        if foreign_intent is not None:
            self._set_blocked(
                connection,
                job_id=job_id,
                reason="blocked_by_job",
                blocked_job_id=str(foreign_intent),
                import_lease_id=None,
                now=now,
            )
            return "blocked"

        intent_set_id = str(uuid.uuid4())
        lease_token = secrets.token_urlsafe(32)
        for chapter_id in chapter_ids:
            connection.execute(
                update(chapters)
                .where(chapters.c.id == chapter_id)
                .values(
                    write_intent_generation=chapters.c.write_intent_generation + 1,
                    updated_at=now,
                )
            )
            generation = int(
                connection.execute(
                    select(chapters.c.write_intent_generation).where(
                        chapters.c.id == chapter_id
                    )
                ).scalar_one()
            )
            connection.execute(
                insert(chapter_write_intents).values(
                    chapter_id=chapter_id,
                    job_id=job_id,
                    intent_set_id=intent_set_id,
                    intent_generation=generation,
                    worker_epoch_id=worker_epoch_id,
                    lease_token=lease_token,
                    lease_expires_at=expires,
                    created_at=now,
                )
            )
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id, jobs.c.status == "queued")
            .values(
                blocked_reason="draining_immediate_writes",
                blocked_by_job_id=None,
                blocked_by_import_lease_id=None,
                worker_epoch_id=worker_epoch_id,
                lease_token=lease_token,
                lease_expires_at=expires,
                updated_at=now,
            )
        )
        self._append_event(
            connection,
            job_id=job_id,
            event_type="chapter_write_intent_created",
            payload={"chapterIds": chapter_ids, "intentSetId": intent_set_id},
            now=now,
        )
        return "draining"

    def _claim_row(
        self,
        connection: Any,
        *,
        candidate: Mapping[str, Any],
        worker_epoch_id: str,
        now: datetime,
        expires: datetime,
    ) -> AttemptFence:
        job_id = str(candidate["id"])
        attempt_id = candidate.get("attempt_id") or str(uuid.uuid4())
        lease_token = candidate.get("lease_token") or secrets.token_urlsafe(32)
        result = connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id, jobs.c.status == "queued")
            .values(
                status="running",
                attempt_id=attempt_id,
                lease_token=lease_token,
                lease_expires_at=expires,
                worker_epoch_id=worker_epoch_id,
                blocked_reason=None,
                blocked_by_job_id=None,
                blocked_by_import_lease_id=None,
                started_at=func.coalesce(jobs.c.started_at, now),
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            raise JobConflict("job claim lost a queue race")
        connection.execute(
            update(chapter_write_locks)
            .where(chapter_write_locks.c.job_id == job_id)
            .values(owner_attempt_id=attempt_id, lease_token=lease_token)
        )
        self._append_event(
            connection,
            job_id=job_id,
            event_type="job_started",
            payload={"attemptId": attempt_id},
            now=now,
        )
        self._bump_queue_revision(connection, now)
        return AttemptFence(
            job_id=job_id,
            attempt_id=str(attempt_id),
            lease_token=str(lease_token),
            worker_epoch_id=worker_epoch_id,
            lease_expires_at=expires,
        )

    @staticmethod
    def _target_chapter_ids(
        connection: Any,
        candidate: Mapping[str, Any],
    ) -> list[str]:
        if candidate.get("chapter_id"):
            return [str(candidate["chapter_id"])]
        if candidate.get("page_id"):
            value = connection.execute(
                select(pages.c.chapter_id).where(
                    pages.c.id == candidate["page_id"]
                )
            ).scalar_one_or_none()
            return [str(value)] if value else []
        return list(
            connection.execute(
                select(pages.c.chapter_id)
                .join(job_items, job_items.c.page_id == pages.c.id)
                .where(job_items.c.job_id == candidate["id"])
                .distinct()
            ).scalars()
        )

    @staticmethod
    def _old_write_chains_active(connection: Any, chapter_ids: Sequence[str]) -> bool:
        active_operation = connection.execute(
            select(operations.c.id)
            .join(pages, pages.c.id == operations.c.page_id)
            .where(
                pages.c.chapter_id.in_(chapter_ids),
                operations.c.status.in_(ACTIVE_OPERATION_STATUSES),
            )
            .limit(1)
        ).scalar_one_or_none()
        if active_operation is not None:
            return True
        active_render = connection.execute(
            select(render_requests.c.id)
            .join(pages, pages.c.id == render_requests.c.page_id)
            .where(
                pages.c.chapter_id.in_(chapter_ids),
                render_requests.c.status.in_(ACTIVE_RENDER_STATUSES),
            )
            .limit(1)
        ).scalar_one_or_none()
        return active_render is not None

    @staticmethod
    def _set_blocked(
        connection: Any,
        *,
        job_id: str,
        reason: str,
        blocked_job_id: str | None,
        import_lease_id: str | None,
        now: datetime,
    ) -> None:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id, jobs.c.status == "queued")
            .values(
                blocked_reason=reason,
                blocked_by_job_id=blocked_job_id,
                blocked_by_import_lease_id=import_lease_id,
                updated_at=now,
            )
        )

    @staticmethod
    def _assert_worker_epoch(
        connection: Any,
        worker_epoch_id: str,
        now: datetime,
    ) -> None:
        epoch = connection.execute(
            select(process_epochs.c.id).where(
                process_epochs.c.id == worker_epoch_id,
                process_epochs.c.role == "worker",
                process_epochs.c.status == "active",
                process_epochs.c.lease_expires_at > now,
            )
        ).scalar_one_or_none()
        if epoch is None:
            raise AttemptFenced("Worker epoch is inactive or expired")

    @staticmethod
    def _assert_attempt(
        connection: Any,
        fence: AttemptFence,
        now: datetime,
        *,
        allowed_statuses: Sequence[str],
    ) -> str:
        row = connection.execute(
            select(jobs.c.status).where(
                jobs.c.id == fence.job_id,
                jobs.c.attempt_id == fence.attempt_id,
                jobs.c.lease_token == fence.lease_token,
                jobs.c.worker_epoch_id == fence.worker_epoch_id,
                jobs.c.lease_expires_at > now,
                jobs.c.status.in_(allowed_statuses),
                exists(
                    select(process_epochs.c.id).where(
                        process_epochs.c.id == fence.worker_epoch_id,
                        process_epochs.c.role == "worker",
                        process_epochs.c.status == "active",
                        process_epochs.c.lease_expires_at > now,
                    )
                ),
            )
        ).scalar_one_or_none()
        if row is None:
            raise AttemptFenced("job attempt lost execution rights")
        return str(row)

    @staticmethod
    def _release_write_reservations(connection: Any, job_id: str) -> None:
        connection.execute(
            delete(chapter_write_intents).where(
                chapter_write_intents.c.job_id == job_id
            )
        )
        connection.execute(
            delete(chapter_write_locks).where(chapter_write_locks.c.job_id == job_id)
        )

    @staticmethod
    def _progress_snapshot(connection: Any, job_id: str) -> dict[str, int]:
        rows = connection.execute(
            select(job_items.c.status, func.count().label("count"))
            .where(job_items.c.job_id == job_id)
            .group_by(job_items.c.status)
        )
        counts = {str(status): int(count) for status, count in rows}
        total = sum(counts.values())
        return {
            "totalItems": total,
            "completedItems": counts.get("completed", 0),
            "failedItems": counts.get("failed", 0),
            "cancelledItems": counts.get("cancelled", 0),
        }

    @staticmethod
    def _job_dto(row: Mapping[str, Any]) -> dict[str, object]:
        return {
            "jobId": row["id"],
            "batchId": row.get("batch_id"),
            "batchDisplayName": row.get("batch_display_name"),
            "kind": row["kind"],
            "status": row["status"],
            "queueRank": row.get("queue_rank"),
            "bookId": row.get("book_id"),
            "chapterId": row.get("chapter_id"),
            "pageId": row.get("page_id"),
            "blockedReason": row.get("blocked_reason"),
            "blockedByJobId": row.get("blocked_by_job_id"),
            "progress": _load_json(row.get("latest_progress_json"), {}),
            "target": _load_json(row.get("target_display_json"), {}),
            "createdAt": JobQueueRepository._iso(row.get("created_at")),
            "startedAt": JobQueueRepository._iso(row.get("started_at")),
            "finishedAt": JobQueueRepository._iso(row.get("finished_at")),
        }

    @staticmethod
    def _iso(value: datetime | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _append_event(
        connection: Any,
        *,
        job_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        now: datetime,
    ) -> int:
        event_id = int(
            connection.execute(
                select(func.coalesce(func.max(job_events.c.id), 0) + 1)
            ).scalar_one()
        )
        connection.execute(
            insert(job_events).values(
                id=event_id,
                job_id=job_id,
                event_type=event_type,
                payload_json=_json(dict(payload)),
                payload_schema_version=1,
                created_at=now,
            )
        )
        return event_id

    @staticmethod
    def _bump_queue_revision(connection: Any, now: datetime) -> int:
        connection.execute(
            update(queue_state)
            .where(queue_state.c.singleton_id == 1)
            .values(
                queue_revision=queue_state.c.queue_revision + 1,
                updated_at=now,
            )
        )
        return int(
            connection.execute(
                select(queue_state.c.queue_revision).where(
                    queue_state.c.singleton_id == 1
                )
            ).scalar_one()
        )

    @staticmethod
    def _refresh_batch_summary(connection: Any, job_id: str, now: datetime) -> None:
        batch_id = connection.execute(
            select(jobs.c.batch_id).where(jobs.c.id == job_id)
        ).scalar_one_or_none()
        if batch_id is None:
            return
        rows = connection.execute(
            select(jobs.c.status, func.count())
            .where(jobs.c.batch_id == batch_id)
            .group_by(jobs.c.status)
        )
        counts = {str(status): int(count) for status, count in rows}
        connection.execute(
            update(job_batches)
            .where(job_batches.c.id == batch_id)
            .values(
                status_summary_json=_json(
                    {"total": sum(counts.values()), **counts}
                ),
                updated_at=now,
            )
        )
