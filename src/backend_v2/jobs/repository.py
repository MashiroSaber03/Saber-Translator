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

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import iso_utc as _iso, utcnow
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Connection

from src.backend_v2.domain.state_machines import (
    InvalidTransition,
    JobEvent,
    JobStatus,
    transition_job,
)
from src.backend_v2.redaction import (
    credential_version_references,
    redact_sensitive_text,
    redact_sensitive_value,
    secret_values_from_json,
)
from src.backend_v2.plugins.snapshots import enabled_plugin_snapshots
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    CURRENT_JOB_STATUSES,
    JOB_KINDS,
    NONTERMINAL_JOB_STATUSES,
    chapter_write_intents,
    chapter_write_locks,
    chapters,
    credential_versions,
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
    job_step_asset_outputs,
    jobs,
    operations,
    page_assets,
    pages,
    process_epochs,
    queue_state,
    render_requests,
    worker_leases,
    analysis_runs,
    analysis_run_targets,
    web_import_drafts,
)


TERMINAL_JOB_STATUSES = (
    "cancelled",
    "completed",
    "completed_with_errors",
    "failed",
)
HISTORY_JOB_STATUSES = (*TERMINAL_JOB_STATUSES, "interrupted")
QUEUE_JOB_STATUSES = ("queued", *CURRENT_JOB_STATUSES)
HISTORY_BATCH_LIMIT = 200
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
    retry_of_job_id: str | None = None
    retry_mode: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in JOB_KINDS:
            raise ValueError(f"unsupported job kind: {self.kind}")
        if not self.items:
            raise ValueError("job requires at least one item")
        if self.kind in WRITE_JOB_KINDS and not self.chapter_id:
            raise ValueError(f"{self.kind} jobs require a chapter_id")
        if bool(self.retry_of_job_id) != bool(self.retry_mode):
            raise ValueError("retry lineage requires both source job and mode")
        if self.retry_mode not in {None, "current", "original"}:
            raise ValueError("retry mode must be current or original")


@dataclass(frozen=True, slots=True)
class AttemptFence:
    job_id: str
    attempt_id: str
    lease_token: str
    worker_epoch_id: str
    lease_expires_at: datetime


def _load_json(value: str | None, default: object) -> object:
    if not value:
        return default
    return json.loads(value)


def _job_secret_values(connection: Connection, job_id: str) -> tuple[str, ...]:
    values: set[str] = set()
    secret_rows = connection.execute(
        select(credential_versions.c.secret_json)
        .join(
            job_credential_snapshots,
            job_credential_snapshots.c.credential_version_id
            == credential_versions.c.id,
        )
        .where(job_credential_snapshots.c.job_id == job_id)
    ).scalars()
    for secret_json in secret_rows:
        values.update(secret_values_from_json(str(secret_json)))
    return tuple(sorted(values, key=len, reverse=True))


class JobQueueRepository:
    """Own all queue ordering, transition, checkpoint, and lock transactions."""

    def __init__(self, engine: Engine, *, attempt_lease_seconds: int = 30) -> None:
        if attempt_lease_seconds < 3:
            raise ValueError("attempt_lease_seconds must be at least 3")
        self.engine = engine
        self.attempt_lease_seconds = attempt_lease_seconds

    def idempotency_replay(
        self,
        *,
        scope: str,
        key: str,
        payload: Mapping[str, Any],
    ) -> dict[str, object] | None:
        """Read a committed command response before resolving mutable targets."""
        if not scope or not key:
            raise ValueError("idempotency scope and key are required")
        request_hash = hashlib.sha256(
            _json(dict(payload)).encode("utf-8")
        ).hexdigest()
        with self.engine.connect() as connection:
            replay = connection.execute(
                select(
                    idempotency_records.c.request_hash,
                    idempotency_records.c.response_json,
                ).where(
                    idempotency_records.c.scope == scope,
                    idempotency_records.c.key == key,
                    idempotency_records.c.expires_at > utcnow(),
                )
            ).mappings().one_or_none()
        if replay is None:
            return None
        if replay["request_hash"] != request_hash:
            raise JobConflict(
                "Idempotency-Key was reused for different job input"
            )
        return json.loads(str(replay["response_json"]))

    def create_batch(
        self,
        *,
        kind: str,
        display_name: str,
        specs: Sequence[JobSpec],
        response_extra: Mapping[str, object] | None = None,
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
                current_plugin_snapshots = (
                    enabled_plugin_snapshots(connection)
                    if any(spec.plugin_snapshots is None for spec in specs)
                    else {}
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
                            retry_of_job_id=spec.retry_of_job_id,
                            retry_mode=spec.retry_mode,
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
                        **credential_version_references(spec.config),
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
                    effective_plugin_snapshots = (
                        spec.plugin_snapshots
                        if spec.plugin_snapshots is not None
                        else current_plugin_snapshots
                    )
                    if effective_plugin_snapshots:
                        connection.execute(
                            insert(job_plugin_snapshots),
                            [
                                {
                                    "job_id": job_id,
                                    "plugin_version_id": version_id,
                                    "config_json": _json(dict(plugin_config)),
                                }
                                for version_id, plugin_config in (
                                    effective_plugin_snapshots.items()
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
                    initial_progress = self._progress_snapshot(connection, job_id)
                    connection.execute(
                        update(jobs)
                        .where(jobs.c.id == job_id)
                        .values(latest_progress_json=_json(initial_progress))
                    )
                    self._append_event(
                        connection,
                        job_id=job_id,
                        event_type="job_created",
                        payload={
                            "batchId": batch_id,
                            "queueRank": next_rank,
                            "progress": initial_progress,
                        },
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
                if response_extra:
                    reserved = {"batchId", "jobIds", "status"} & set(response_extra)
                    if reserved:
                        raise ValueError(
                            "response_extra cannot replace batch response fields"
                        )
                    response.update(dict(response_extra))
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
            conditions.append(jobs.c.status.in_(QUEUE_JOB_STATUSES))
        else:
            conditions.append(jobs.c.status.in_(HISTORY_JOB_STATUSES))
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
            statement = (
                select(
                    jobs,
                    job_batches.c.display_name.label("batch_display_name"),
                    exists().where(
                        chapter_write_locks.c.job_id == jobs.c.id
                    ).label("holds_chapter_lock"),
                )
                .join(job_batches, job_batches.c.id == jobs.c.batch_id, isouter=True)
                .where(*conditions)
            )
            if scope == "history":
                limited_batch_ids = (
                    select(job_batches.c.id)
                    .join(jobs, jobs.c.batch_id == job_batches.c.id)
                    .where(*conditions)
                    .group_by(job_batches.c.id, job_batches.c.created_at)
                    .order_by(
                        job_batches.c.created_at.desc(),
                        job_batches.c.id.desc(),
                    )
                    .limit(limit)
                )
                statement = statement.where(
                    jobs.c.batch_id.in_(limited_batch_ids)
                ).order_by(
                    job_batches.c.created_at.desc(),
                    job_batches.c.id.desc(),
                    *order,
                )
            else:
                statement = statement.order_by(*order).limit(limit)
            rows = connection.execute(statement).mappings()
            revision = connection.execute(
                select(queue_state.c.queue_revision).where(
                    queue_state.c.singleton_id == 1
                )
            ).scalar_one()
            event_cursor = int(
                connection.execute(
                    select(func.coalesce(func.max(job_events.c.id), 0))
                ).scalar_one()
            )
            now = utcnow()
            worker_online = bool(
                connection.execute(
                    select(
                        exists().where(
                            worker_leases.c.worker_epoch_id
                            == process_epochs.c.id,
                            process_epochs.c.role == "worker",
                            process_epochs.c.status == "active",
                            process_epochs.c.lease_expires_at > now,
                            worker_leases.c.lease_expires_at > now,
                        )
                    )
                ).scalar()
            )
            return {
                "items": [self._job_dto(row) for row in rows],
                "queueRevision": int(revision),
                "eventCursor": event_cursor,
                "workerOnline": worker_online,
            }

    def get_job(self, job_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            job = connection.execute(
                select(
                    jobs,
                    job_batches.c.display_name.label("batch_display_name"),
                    exists().where(
                        chapter_write_locks.c.job_id == jobs.c.id
                    ).label("holds_chapter_lock"),
                )
                .join(job_batches, job_batches.c.id == jobs.c.batch_id, isouter=True)
                .where(jobs.c.id == job_id)
            ).mappings().one_or_none()
            if job is None:
                raise JobNotFound("job not found")
            item_rows = list(connection.execute(
                select(job_items)
                .where(job_items.c.job_id == job_id)
                .order_by(job_items.c.ordinal)
            ).mappings())
            items: list[dict[str, object]] = []
            failed_items: list[dict[str, object]] = []
            for row in item_rows:
                step_rows = list(connection.execute(
                    select(job_steps)
                    .where(job_steps.c.job_item_id == row["id"])
                    .order_by(job_steps.c.ordinal)
                ).mappings())
                item_error = _load_json(row["error_json"], None)
                serialized_steps = [
                    {
                        "stepId": step["id"],
                        "ordinal": step["ordinal"],
                        "kind": step["kind"],
                        "status": step["status"],
                        "checkpoint": _load_json(step["checkpoint_json"], None),
                        "error": _load_json(step["error_json"], None),
                    }
                    for step in step_rows
                ]
                serialized_item = {
                    "itemId": row["id"],
                    "ordinal": row["ordinal"],
                    "pageId": row["page_id"],
                    "status": row["status"],
                    "result": _load_json(row["result_json"], None),
                    "error": item_error,
                    "steps": serialized_steps,
                }
                items.append(serialized_item)
                if str(row["status"]) == "failed":
                    failed_step = next(
                        (
                            step
                            for step in serialized_steps
                            if step["status"] == "failed"
                        ),
                        None,
                    )
                    failed_items.append(
                        {
                            "itemId": row["id"],
                            "ordinal": row["ordinal"],
                            "pageId": row["page_id"],
                            "stepId": (
                                failed_step["stepId"] if failed_step else None
                            ),
                            "stepKind": (
                                failed_step["kind"] if failed_step else None
                            ),
                            "error": (
                                item_error
                                if item_error is not None
                                else (
                                    failed_step["error"]
                                    if failed_step
                                    else None
                                )
                            ),
                        }
                    )
            artifact_rows = list(
                connection.execute(
                    select(job_artifacts).where(
                        job_artifacts.c.job_id == job_id
                    )
                ).mappings()
            )
            resource_rows = list(
                connection.execute(
                    select(
                        job_step_asset_outputs.c.job_step_id,
                        job_step_asset_outputs.c.role,
                        job_step_asset_outputs.c.asset_id,
                        assets.c.mime_type,
                        assets.c.byte_size,
                        assets.c.integrity_status,
                    )
                    .join(
                        job_steps,
                        job_steps.c.id
                        == job_step_asset_outputs.c.job_step_id,
                    )
                    .join(
                        job_items,
                        job_items.c.id == job_steps.c.job_item_id,
                    )
                    .join(
                        assets,
                        assets.c.id == job_step_asset_outputs.c.asset_id,
                    )
                    .where(job_items.c.job_id == job_id)
                    .order_by(job_items.c.ordinal, job_steps.c.ordinal)
                ).mappings()
            )
            recent_event_rows = list(
                connection.execute(
                    select(job_events)
                    .where(job_events.c.job_id == job_id)
                    .order_by(job_events.c.id.desc())
                    .limit(50)
                ).mappings()
            )
        result = self._job_dto(job)
        progress = result["progress"]
        counts: dict[str, int] = {}
        for row in item_rows:
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + 1
        result["counts"] = {
            "total": len(item_rows),
            "pending": counts.get("pending", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "skipped": counts.get("skipped", 0),
            "cancelled": counts.get("cancelled", 0),
        }
        result["durationMs"] = self._duration_ms(
            job.get("started_at"),
            job.get("finished_at"),
            running=str(job["status"]) in NONTERMINAL_JOB_STATUSES,
        )
        result["error"] = (
            progress.get("error")
            if isinstance(progress, Mapping)
            else None
        )
        if result["error"] is None and failed_items:
            result["error"] = failed_items[0]["error"]
        result["configSummary"] = self._config_summary(
            _load_json(job.get("config_json"), {})
        )
        result["items"] = items
        result["failedItems"] = failed_items
        result["artifacts"] = [
            {
                "kind": row["kind"],
                "assetId": row["asset_id"],
                "url": f"/api/v2/assets/{row['asset_id']}",
                "expiresAt": _iso(row["expires_at"]),
            }
            for row in artifact_rows
        ]
        result["resources"] = [
            {
                "stepId": row["job_step_id"],
                "role": row["role"],
                "assetId": row["asset_id"],
                "url": f"/api/v2/assets/{row['asset_id']}",
                "mimeType": row["mime_type"],
                "byteSize": int(row["byte_size"]),
                "integrityStatus": row["integrity_status"],
            }
            for row in resource_rows
        ]
        result["recentEvents"] = [
            self._event_dto(row) for row in reversed(recent_event_rows)
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
                select(
                    jobs,
                    exists().where(
                        chapter_write_locks.c.job_id == jobs.c.id
                    ).label("holds_chapter_lock"),
                )
                .where(jobs.c.batch_id == batch_id)
                .order_by(jobs.c.queue_rank, jobs.c.created_at)
            ).mappings()
        return {
            "batchId": batch["id"],
            "kind": batch["kind"],
            "displayName": batch["display_name"],
            "summary": _load_json(batch["status_summary_json"], {}),
            "jobs": [self._job_dto(row) for row in member_rows],
            "createdAt": _iso(batch["created_at"]),
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
            rows = list(
                connection.execute(
                    select(job_events)
                    .where(condition)
                    .order_by(job_events.c.id)
                    .limit(limit)
                ).mappings()
            )
        return [self._event_dto(row) for row in rows]

    def job_snapshot(self, *, job_ids: Sequence[str]) -> dict[str, object]:
        unique_ids = tuple(dict.fromkeys(str(value) for value in job_ids if value))
        if not unique_ids:
            raise ValueError("at least one job_id is required")
        if len(unique_ids) > 200:
            raise ValueError("at most 200 job IDs may be read at once")
        with self.engine.connect() as connection:
            snapshots = self._job_snapshots(
                connection,
                job_ids=set(unique_ids),
            )
            revision = int(
                connection.execute(
                    select(queue_state.c.queue_revision).where(
                        queue_state.c.singleton_id == 1
                    )
                ).scalar_one()
            )
        return {
            "items": [snapshots[value] for value in unique_ids if value in snapshots],
            "queueRevision": revision,
        }

    def events_before(
        self,
        *,
        before: int,
        job_id: str,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        if before < 1:
            raise ValueError("event cursor must be positive")
        if limit < 1 or limit > 1000:
            raise ValueError("event limit must be between 1 and 1000")
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(job_events)
                    .where(
                        job_events.c.job_id == job_id,
                        job_events.c.id < before,
                    )
                    .order_by(job_events.c.id.desc())
                    .limit(limit)
                ).mappings()
            )
        return [self._event_dto(row) for row in reversed(rows)]

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
            self._reorder_ordinary(
                connection,
                ordered_job_ids=ordered_job_ids,
                now=now,
            )
            return self._bump_queue_revision(connection, now)

    def prioritize_batch(self, *, batch_id: str, base_revision: int) -> int:
        """Move ordinary queued members to the front without splitting their order."""

        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            exists_batch = connection.execute(
                select(job_batches.c.id).where(job_batches.c.id == batch_id)
            ).scalar_one_or_none()
            if exists_batch is None:
                raise JobNotFound("job batch not found")
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
                        ~jobs.c.id.in_(select(chapter_write_intents.c.job_id)),
                        ~jobs.c.id.in_(select(chapter_write_locks.c.job_id)),
                    )
                    .order_by(jobs.c.queue_rank, jobs.c.created_at)
                ).scalars()
            )
            members = [
                str(job_id)
                for job_id in connection.execute(
                    select(jobs.c.id)
                    .where(
                        jobs.c.batch_id == batch_id,
                        jobs.c.id.in_(sortable),
                    )
                    .order_by(jobs.c.queue_rank, jobs.c.created_at)
                ).scalars()
            ]
            if not members:
                raise JobConflict("job batch has no ordinary queued members")
            member_set = set(members)
            ordered = [*members, *(value for value in sortable if value not in member_set)]
            self._reorder_ordinary(
                connection,
                ordered_job_ids=ordered,
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
                self._cancel_queued_job(
                    connection,
                    job_id=str(job_id),
                    source="cancel_all_queued",
                    now=now,
                )
            if ids:
                self._bump_queue_revision(connection, now)
            return len(ids)

    def cancel_batch_queued(self, batch_id: str) -> int:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            exists_batch = connection.execute(
                select(job_batches.c.id).where(job_batches.c.id == batch_id)
            ).scalar_one_or_none()
            if exists_batch is None:
                raise JobNotFound("job batch not found")
            ids = list(
                connection.execute(
                    select(jobs.c.id)
                    .where(
                        jobs.c.batch_id == batch_id,
                        jobs.c.status == "queued",
                    )
                    .order_by(jobs.c.queue_rank, jobs.c.created_at)
                ).scalars()
            )
            for job_id in ids:
                self._cancel_queued_job(
                    connection,
                    job_id=str(job_id),
                    source="cancel_batch_queued",
                    now=now,
                )
            if ids:
                self._bump_queue_revision(connection, now)
            return len(ids)

    def continue_batch(self, batch_id: str) -> dict[str, object]:
        """Resume paused and continue interrupted members in batch order."""

        now = utcnow()
        updated_jobs: list[dict[str, object]] = []
        with immediate_transaction(self.engine) as connection:
            exists_batch = connection.execute(
                select(job_batches.c.id).where(job_batches.c.id == batch_id)
            ).scalar_one_or_none()
            if exists_batch is None:
                raise JobNotFound("job batch not found")
            rows = list(
                connection.execute(
                    select(jobs)
                    .where(
                        jobs.c.batch_id == batch_id,
                        jobs.c.status.in_(("paused", "interrupted")),
                    )
                    .order_by(jobs.c.queue_rank, jobs.c.created_at)
                ).mappings()
            )
            if not rows:
                raise JobConflict("job batch has no paused or interrupted members")
            for row in rows:
                event = (
                    JobEvent.RESUME
                    if str(row["status"]) == "paused"
                    else JobEvent.CONTINUE
                )
                values = {
                    "status": "queued",
                    "attempt_id": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "worker_epoch_id": None,
                    "updated_at": now,
                }
                progress = _load_json(row["latest_progress_json"], {})
                progress["jobStatus"] = "queued"
                values["latest_progress_json"] = _json(progress)
                connection.execute(
                    update(jobs).where(jobs.c.id == row["id"]).values(**values)
                )
                self._append_event(
                    connection,
                    job_id=str(row["id"]),
                    event_type=f"job_{event.value}",
                    payload={
                        "from": row["status"],
                        "to": "queued",
                        "source": "batch",
                        "progress": progress,
                    },
                    now=now,
                )
                updated = dict(row)
                updated.update(values)
                updated_jobs.append(self._job_dto(updated))
            self._bump_queue_revision(connection, now)
            self._refresh_batch_summary(connection, str(rows[0]["id"]), now)
        return {"continued": len(updated_jobs), "jobs": updated_jobs}

    def clear_history(self) -> int:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            candidates = {
                str(value)
                for value in connection.execute(
                    select(jobs.c.id).where(
                        jobs.c.status.in_(TERMINAL_JOB_STATUSES)
                    )
                ).scalars()
            }
            if not candidates:
                return 0
            return self._delete_history_jobs(
                connection,
                candidates=candidates,
                now=now,
            )

    def prune_history(self, *, max_batches: int = HISTORY_BATCH_LIMIT) -> int:
        if max_batches < 1:
            raise ValueError("history batch limit must be positive")
        with immediate_transaction(self.engine) as connection:
            return self._prune_history_batches(
                connection,
                now=utcnow(),
                max_batches=max_batches,
            )

    @staticmethod
    def _delete_history_jobs(
        connection: Any,
        *,
        candidates: set[str],
        now: datetime,
    ) -> int:
        if not candidates:
            return 0
        protected = {
            str(value)
            for value in connection.execute(
                select(job_artifacts.c.job_id).where(
                    job_artifacts.c.job_id.in_(candidates),
                    or_(
                        job_artifacts.c.expires_at.is_(None),
                        job_artifacts.c.expires_at > now,
                    ),
                )
            ).scalars()
        }
        retry_edges = [
            (str(child_id), str(source_id))
            for child_id, source_id in connection.execute(
                select(jobs.c.id, jobs.c.retry_of_job_id).where(
                    jobs.c.retry_of_job_id.is_not(None)
                )
            )
        ]
        protected.update(
            source_id
            for child_id, source_id in retry_edges
            if source_id in candidates and child_id not in candidates
        )
        changed = True
        while changed:
            changed = False
            for child_id, source_id in retry_edges:
                if (
                    child_id in protected
                    and source_id in candidates
                    and source_id not in protected
                ):
                    protected.add(source_id)
                    changed = True
        remaining = candidates - protected
        removable_count = len(remaining)
        while remaining:
            referenced = {
                source_id
                for child_id, source_id in retry_edges
                if child_id in remaining and source_id in remaining
            }
            leaves = remaining - referenced
            if not leaves:
                raise JobConflict("job retry lineage contains a cycle")
            connection.execute(delete(jobs).where(jobs.c.id.in_(leaves)))
            remaining -= leaves
        connection.execute(
            delete(job_batches).where(
                ~exists(
                    select(jobs.c.id).where(
                        jobs.c.batch_id == job_batches.c.id
                    )
                )
            )
        )
        return removable_count

    @staticmethod
    def _prune_history_batches(
        connection: Any,
        *,
        now: datetime,
        max_batches: int = HISTORY_BATCH_LIMIT,
    ) -> int:
        member = jobs.alias("history_member")
        nonhistory_member = jobs.alias("nonhistory_member")
        history_batch_ids = [
            str(value)
            for value in connection.execute(
                select(job_batches.c.id)
                .where(
                    exists(
                        select(member.c.id).where(
                            member.c.batch_id == job_batches.c.id
                        )
                    ),
                    ~exists(
                        select(nonhistory_member.c.id).where(
                            nonhistory_member.c.batch_id == job_batches.c.id,
                            nonhistory_member.c.status.not_in(
                                HISTORY_JOB_STATUSES
                            ),
                        )
                    ),
                )
                .order_by(
                    job_batches.c.created_at.desc(),
                    job_batches.c.id.desc(),
                )
            ).scalars()
        ]
        old_batch_ids = set(history_batch_ids[max_batches:])
        if not old_batch_ids:
            return 0
        interrupted_batches = {
            str(value)
            for value in connection.execute(
                select(jobs.c.batch_id).where(
                    jobs.c.batch_id.in_(old_batch_ids),
                    jobs.c.status == "interrupted",
                )
            ).scalars()
        }
        deletable_batches = old_batch_ids - interrupted_batches
        if not deletable_batches:
            return 0
        candidates = {
            str(value)
            for value in connection.execute(
                select(jobs.c.id).where(
                    jobs.c.batch_id.in_(deletable_batches),
                    jobs.c.status.in_(TERMINAL_JOB_STATUSES),
                )
            ).scalars()
        }
        if not candidates:
            return 0

        # A protected job retains its complete batch. Re-evaluate because
        # retaining a retry child can in turn protect an older source batch.
        while True:
            protected: set[str] = set()
            artifact_protected = {
                str(value)
                for value in connection.execute(
                    select(job_artifacts.c.job_id).where(
                        job_artifacts.c.job_id.in_(candidates),
                        or_(
                            job_artifacts.c.expires_at.is_(None),
                            job_artifacts.c.expires_at > now,
                        ),
                    )
                ).scalars()
            }
            protected.update(artifact_protected)
            retry_edges = [
                (str(child_id), str(source_id))
                for child_id, source_id in connection.execute(
                    select(jobs.c.id, jobs.c.retry_of_job_id).where(
                        jobs.c.retry_of_job_id.is_not(None)
                    )
                )
            ]
            protected.update(
                source_id
                for child_id, source_id in retry_edges
                if source_id in candidates and child_id not in candidates
            )
            changed = True
            while changed:
                changed = False
                for child_id, source_id in retry_edges:
                    if (
                        child_id in protected
                        and source_id in candidates
                        and source_id not in protected
                    ):
                        protected.add(source_id)
                        changed = True
            if not protected:
                break
            protected_batches = {
                str(value)
                for value in connection.execute(
                    select(jobs.c.batch_id).where(
                        jobs.c.id.in_(protected),
                        jobs.c.batch_id.is_not(None),
                    )
                ).scalars()
            }
            next_candidates = {
                str(row["id"])
                for row in connection.execute(
                    select(jobs.c.id, jobs.c.batch_id).where(
                        jobs.c.id.in_(candidates)
                    )
                ).mappings()
                if str(row["batch_id"]) not in protected_batches
            }
            if next_candidates == candidates:
                break
            candidates = next_candidates
            if not candidates:
                return 0
        return JobQueueRepository._delete_history_jobs(
            connection,
            candidates=candidates,
            now=now,
        )

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

    def redact_attempt_message(
        self,
        fence: AttemptFence,
        message: object,
    ) -> str:
        """Scrub an exception before a domain publisher persists it."""

        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            secret_values = _job_secret_values(connection, fence.job_id)
        return redact_sensitive_text(message, secret_values=secret_values)

    def append_worker_event(
        self,
        fence: AttemptFence,
        *,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int:
        if (
            not event_type.startswith(("plugin_", "web_import_"))
            or len(event_type) > 64
        ):
            raise ValueError("worker event type is invalid")
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

    def completed_plugin_stages(
        self,
        fence: AttemptFence,
    ) -> set[tuple[str, str | None]]:
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
            result: set[tuple[str, str | None]] = set()
            for payload in rows:
                data = _load_json(str(payload), {})
                hook = data.get("hook")
                if not isinstance(hook, str):
                    continue
                item_id = data.get("itemId")
                result.add(
                    (hook, str(item_id) if item_id is not None else None)
                )
            return result

    def complete_plugin_stage(
        self,
        fence: AttemptFence,
        *,
        hook: str,
        scope: str,
        item_id: str | None = None,
        page_id: str | None = None,
        job_config: Mapping[str, Any] | None = None,
        outcome: str = "completed",
    ) -> None:
        if outcome not in {"completed", "failed"}:
            raise ValueError("plugin stage outcome is invalid")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            if job_config is not None:
                changed = connection.execute(
                    update(jobs)
                    .where(
                        jobs.c.id == fence.job_id,
                        jobs.c.attempt_id == fence.attempt_id,
                        jobs.c.lease_token == fence.lease_token,
                    )
                    .values(config_json=_json(dict(job_config)), updated_at=now)
                )
                if changed.rowcount != 1:
                    raise AttemptFenced("plugin job config write was fenced")
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type="plugin_stage_completed",
                payload={
                    "hook": hook,
                    "scope": scope,
                    "outcome": outcome,
                    "itemId": item_id,
                    "pageId": page_id,
                },
                now=now,
            )

    def active_step_counts(
        self,
        fence: AttemptFence,
        *,
        step_kind: str | None = None,
    ) -> tuple[int, int]:
        """Return pending/running counts, optionally scoped to one stage pool."""

        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            conditions = [
                job_items.c.job_id == fence.job_id,
                job_steps.c.status.in_(("pending", "running")),
            ]
            if step_kind is not None:
                conditions.append(job_steps.c.kind == step_kind)
            rows = list(
                connection.execute(
                    select(job_steps.c.status, func.count())
                    .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                    .where(*conditions)
                    .group_by(job_steps.c.status)
                )
            )
        counts = {str(status): int(count) for status, count in rows}
        return counts.get("pending", 0), counts.get("running", 0)

    def item_statuses(
        self,
        fence: AttemptFence,
        item_ids: Sequence[str],
    ) -> dict[str, str]:
        if not item_ids:
            return {}
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            rows = connection.execute(
                select(job_items.c.id, job_items.c.status).where(
                    job_items.c.job_id == fence.job_id,
                    job_items.c.id.in_(tuple(item_ids)),
                )
            )
            return {str(item_id): str(status) for item_id, status in rows}

    def terminal_page_items(
        self,
        fence: AttemptFence,
    ) -> list[dict[str, str]]:
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            rows = connection.execute(
                select(
                    job_items.c.id,
                    job_items.c.page_id,
                    job_items.c.status,
                )
                .where(
                    job_items.c.job_id == fence.job_id,
                    job_items.c.page_id.is_not(None),
                    job_items.c.status.in_(
                        ("completed", "failed", "skipped", "cancelled")
                    ),
                )
                .order_by(job_items.c.ordinal)
            ).mappings()
            return [
                {
                    "itemId": str(row["id"]),
                    "pageId": str(row["page_id"]),
                    "status": str(row["status"]),
                }
                for row in rows
            ]

    def pending_step_kinds(self, fence: AttemptFence) -> tuple[str, ...]:
        """Return the durable step kinds that still need a Worker handler."""

        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            rows = connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == fence.job_id,
                    job_steps.c.status == "pending",
                )
                .distinct()
                .order_by(job_steps.c.kind)
            ).scalars()
            return tuple(str(kind) for kind in rows)

    def step_kinds(self, fence: AttemptFence) -> tuple[str, ...]:
        """Return only the durable pools that belong to this job graph."""

        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            rows = connection.execute(
                select(
                    job_steps.c.kind,
                    func.min(job_steps.c.ordinal).label("first_ordinal"),
                )
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == fence.job_id)
                .group_by(job_steps.c.kind)
                .order_by("first_ordinal", job_steps.c.kind)
            )
            return tuple(str(kind) for kind, _ordinal in rows)

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
            first_boundary_step = job_steps.alias("first_boundary_step")
            last_boundary_step = job_steps.alias("last_boundary_step")
            proofread_barrier_step = job_steps.alias("proofread_barrier_step")
            proofread_barrier_item = job_items.alias("proofread_barrier_item")
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
                or_(
                    job_steps.c.kind != "render",
                    ~exists(
                        select(proofread_barrier_step.c.id)
                        .select_from(
                            proofread_barrier_step.join(
                                proofread_barrier_item,
                                proofread_barrier_item.c.id
                                == proofread_barrier_step.c.job_item_id,
                            )
                        )
                        .where(
                            proofread_barrier_item.c.job_id == jobs.c.id,
                            proofread_barrier_step.c.kind == "proofread",
                            proofread_barrier_step.c.status.in_(
                                ("pending", "running")
                            ),
                        )
                        .correlate(jobs)
                    ),
                ),
            ]
            if allowed_kinds:
                conditions.append(job_steps.c.kind.in_(tuple(allowed_kinds)))
            row = connection.execute(
                select(
                    job_steps.c.id.label("step_id"),
                    job_steps.c.kind.label("step_kind"),
                    job_steps.c.ordinal.label("step_ordinal"),
                    job_steps.c.checkpoint_json,
                    job_items.c.id.label("item_id"),
                    job_items.c.ordinal.label("item_ordinal"),
                    job_items.c.page_id,
                    jobs.c.kind.label("job_kind"),
                    jobs.c.config_json,
                    ~exists(
                        select(first_boundary_step.c.id).where(
                            first_boundary_step.c.job_item_id
                            == job_steps.c.job_item_id,
                            first_boundary_step.c.ordinal
                            < job_steps.c.ordinal,
                        )
                    ).label("is_first_step"),
                    ~exists(
                        select(last_boundary_step.c.id).where(
                            last_boundary_step.c.job_item_id
                            == job_steps.c.job_item_id,
                            last_boundary_step.c.ordinal
                            > job_steps.c.ordinal,
                        )
                    ).label("is_last_step"),
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
            snapshot = self._progress_after_step_started(
                connection,
                fence.job_id,
                row,
            )
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                )
                .values(latest_progress_json=_json(snapshot), updated_at=now)
            )
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type="step_started",
                payload={
                    "itemId": row["item_id"],
                    "pageId": row["page_id"],
                    "stepId": row["step_id"],
                    "stepKind": row["step_kind"],
                    "progress": snapshot,
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
                "checkpoint": _load_json(row["checkpoint_json"], {}),
                "isFirstStep": bool(row["is_first_step"]),
                "isLastStep": bool(row["is_last_step"]),
            }

    def next_step_batch(
        self,
        fence: AttemptFence,
        *,
        step_kind: str,
        limit: int,
    ) -> list[dict[str, object]]:
        """Claim one durable step batch at a shared round boundary.

        A partial batch is only released when no page with the same step ordinal
        is still waiting on an upstream step. This gives the Worker bounded
        buffering without using an in-memory page list as scheduler state.
        """

        if not step_kind or limit < 1 or limit > 32:
            raise ValueError("step batch kind/limit is invalid")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(connection, fence, now, allowed_statuses=("running",))
            prior_step = job_steps.alias("batch_prior_step")
            first_boundary_step = job_steps.alias("batch_first_boundary_step")
            last_boundary_step = job_steps.alias("batch_last_boundary_step")
            ready_condition = ~exists(
                select(prior_step.c.id).where(
                    prior_step.c.job_item_id == job_steps.c.job_item_id,
                    prior_step.c.ordinal < job_steps.c.ordinal,
                    prior_step.c.status.in_(("pending", "running")),
                )
            )
            base_conditions = (
                jobs.c.id == fence.job_id,
                job_items.c.status.in_(("pending", "running")),
                job_steps.c.status == "pending",
                job_steps.c.kind == step_kind,
            )
            first_ordinal = connection.execute(
                select(func.min(job_steps.c.ordinal))
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .join(jobs, jobs.c.id == job_items.c.job_id)
                .where(*base_conditions, ready_condition)
            ).scalar_one_or_none()
            if first_ordinal is None:
                return []

            rows = list(
                connection.execute(
                    select(
                        job_steps.c.id.label("step_id"),
                        job_steps.c.kind.label("step_kind"),
                        job_steps.c.ordinal.label("step_ordinal"),
                        job_items.c.id.label("item_id"),
                        job_items.c.ordinal.label("item_ordinal"),
                        job_items.c.page_id,
                        jobs.c.kind.label("job_kind"),
                        jobs.c.config_json,
                        ~exists(
                            select(first_boundary_step.c.id).where(
                                first_boundary_step.c.job_item_id
                                == job_steps.c.job_item_id,
                                first_boundary_step.c.ordinal
                                < job_steps.c.ordinal,
                            )
                        ).label("is_first_step"),
                        ~exists(
                            select(last_boundary_step.c.id).where(
                                last_boundary_step.c.job_item_id
                                == job_steps.c.job_item_id,
                                last_boundary_step.c.ordinal
                                > job_steps.c.ordinal,
                            )
                        ).label("is_last_step"),
                    )
                    .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                    .join(jobs, jobs.c.id == job_items.c.job_id)
                    .where(
                        *base_conditions,
                        job_steps.c.ordinal == int(first_ordinal),
                        ready_condition,
                    )
                    .order_by(job_items.c.ordinal)
                    .limit(limit)
                ).mappings()
            )
            if len(rows) < limit:
                blocked_prior = job_steps.alias("blocked_prior_step")
                blocked = int(
                    connection.execute(
                        select(func.count())
                        .select_from(job_steps)
                        .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                        .where(
                            job_items.c.job_id == fence.job_id,
                            job_items.c.status.in_(("pending", "running")),
                            job_steps.c.status == "pending",
                            job_steps.c.kind == step_kind,
                            job_steps.c.ordinal == int(first_ordinal),
                            exists(
                                select(blocked_prior.c.id).where(
                                    blocked_prior.c.job_item_id
                                    == job_steps.c.job_item_id,
                                    blocked_prior.c.ordinal < job_steps.c.ordinal,
                                    blocked_prior.c.status.in_(("pending", "running")),
                                )
                            ),
                        )
                    ).scalar_one()
                )
                if blocked:
                    return []

            claimed_steps: list[dict[str, object]] = []
            for row in rows:
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
                    raise AttemptFenced("batch step was claimed by another attempt")
                claimed_steps.append(
                    {
                        "jobId": fence.job_id,
                        "jobKind": row["job_kind"],
                        "config": _load_json(row["config_json"], {}),
                        "itemId": row["item_id"],
                        "itemOrdinal": row["item_ordinal"],
                        "pageId": row["page_id"],
                        "stepId": row["step_id"],
                        "stepOrdinal": row["step_ordinal"],
                        "stepKind": row["step_kind"],
                        "isFirstStep": bool(row["is_first_step"]),
                        "isLastStep": bool(row["is_last_step"]),
                    }
                )
            snapshot = self._load_progress_snapshot(connection, fence.job_id)
            for row in rows:
                if not self._mutate_progress_step_started(snapshot, row):
                    snapshot = self._progress_snapshot(
                        connection,
                        fence.job_id,
                    )
                    break
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                )
                .values(latest_progress_json=_json(snapshot), updated_at=now)
            )
            for row in rows:
                self._append_event(
                    connection,
                    job_id=fence.job_id,
                    event_type="step_started",
                    payload={
                        "itemId": row["item_id"],
                        "pageId": row["page_id"],
                        "stepId": row["step_id"],
                        "stepKind": row["step_kind"],
                        "batchOrdinal": int(first_ordinal),
                        "progress": snapshot,
                    },
                    now=now,
                )
            return claimed_steps

    def ready_step_ordinal(
        self,
        fence: AttemptFence,
        *,
        step_kind: str,
    ) -> int | None:
        return self.ready_step_ordinals(
            fence,
            step_kinds=(step_kind,),
        ).get(step_kind)

    def ready_step_ordinals(
        self,
        fence: AttemptFence,
        *,
        step_kinds: Sequence[str],
    ) -> dict[str, int]:
        """Return claimable rounds for a bounded set of batch stage pools.

        This read-only admission preflight is shared by the parallel
        coordinator.  Each subsequent claim still revalidates every condition
        inside ``BEGIN IMMEDIATE``.
        """

        unique_kinds = tuple(dict.fromkeys(step_kinds))
        if not unique_kinds:
            return {}
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(connection, fence, now, allowed_statuses=("running",))
            prior_step = job_steps.alias("ready_prior_step")
            proofread_barrier_step = job_steps.alias("ready_proofread_barrier_step")
            proofread_barrier_item = job_items.alias("ready_proofread_barrier_item")
            rows = connection.execute(
                select(
                    job_steps.c.kind,
                    func.min(job_steps.c.ordinal),
                )
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == fence.job_id,
                    job_items.c.status.in_(("pending", "running")),
                    job_steps.c.status == "pending",
                    job_steps.c.kind.in_(unique_kinds),
                    ~exists(
                        select(prior_step.c.id).where(
                            prior_step.c.job_item_id == job_steps.c.job_item_id,
                            prior_step.c.ordinal < job_steps.c.ordinal,
                            prior_step.c.status.in_(("pending", "running")),
                        )
                    ),
                    or_(
                        job_steps.c.kind != "render",
                        ~exists(
                            select(proofread_barrier_step.c.id)
                            .select_from(
                                proofread_barrier_step.join(
                                    proofread_barrier_item,
                                    proofread_barrier_item.c.id
                                    == proofread_barrier_step.c.job_item_id,
                                )
                            )
                            .where(
                                proofread_barrier_item.c.job_id == fence.job_id,
                                proofread_barrier_step.c.kind == "proofread",
                                proofread_barrier_step.c.status.in_(
                                    ("pending", "running")
                                ),
                            )
                        ),
                    ),
                )
                .group_by(job_steps.c.kind)
            )
            return {str(kind): int(ordinal) for kind, ordinal in rows}

    def checkpoint_step(
        self,
        fence: AttemptFence,
        *,
        step_id: str,
        checkpoint: Mapping[str, Any],
        publisher: Callable[[Connection], None] | None = None,
    ) -> str:
        """Persist an intra-step safe point and yield when control was requested."""

        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            job_status = self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            safe_checkpoint = redact_sensitive_value(
                dict(checkpoint),
                secret_values=(),
            )
            step = connection.execute(
                select(
                    job_steps.c.job_item_id,
                    job_items.c.page_id,
                )
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_steps.c.id == step_id,
                    job_steps.c.status == "running",
                    job_steps.c.attempt_id == fence.attempt_id,
                    job_items.c.job_id == fence.job_id,
                )
            ).mappings().one_or_none()
            if step is None:
                raise AttemptFenced("step checkpoint was fenced")
            if publisher is not None:
                publisher(connection)
            yielding = job_status in {"pausing", "cancelling"}
            values: dict[str, object] = {
                "checkpoint_json": _json(safe_checkpoint),
                "updated_at": now,
            }
            if yielding:
                values.update(status="pending", attempt_id=None)
            connection.execute(
                update(job_steps)
                .where(
                    job_steps.c.id == step_id,
                    job_steps.c.attempt_id == fence.attempt_id,
                )
                .values(**values)
            )
            if yielding:
                connection.execute(
                    update(job_items)
                    .where(job_items.c.id == step["job_item_id"])
                    .values(status="pending", updated_at=now)
                )
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
                event_type="step_checkpointed",
                payload={
                    "itemId": str(step["job_item_id"]),
                    "pageId": (
                        str(step["page_id"])
                        if step["page_id"] is not None
                        else None
                    ),
                    "stepId": step_id,
                    "yielded": yielding,
                    "checkpoint": safe_checkpoint,
                    "progress": snapshot,
                },
                now=now,
            )
            return job_status

    def complete_step(
        self,
        fence: AttemptFence,
        *,
        step_id: str,
        checkpoint: Mapping[str, Any],
        input_fingerprint: str | None = None,
        publisher: Callable[[Connection], None] | None = None,
        defer_on_control: bool = False,
    ) -> bool:
        return self._finish_step(
            fence,
            step_id=step_id,
            status="completed",
            checkpoint=checkpoint,
            error=None,
            input_fingerprint=input_fingerprint,
            publisher=publisher,
            defer_on_control=defer_on_control,
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
            defer_on_control=False,
        )

    def fail_terminal_item(
        self,
        fence: AttemptFence,
        *,
        item_id: str,
        code: str,
        message: str,
    ) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            job_status = self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            row = connection.execute(
                select(job_items.c.status, job_items.c.page_id).where(
                    job_items.c.id == item_id,
                    job_items.c.job_id == fence.job_id,
                )
            ).mappings().one_or_none()
            if row is None:
                raise AttemptFenced("plugin pipeline item no longer exists")
            if row["status"] == "failed":
                return
            if row["status"] not in {"completed", "skipped", "cancelled"}:
                raise AttemptFenced("plugin pipeline item is not terminal")
            safe_error = redact_sensitive_value(
                {"code": code, "message": message},
                secret_values=_job_secret_values(connection, fence.job_id),
            )
            connection.execute(
                update(job_items)
                .where(job_items.c.id == item_id)
                .values(
                    status="failed",
                    error_json=_json(safe_error),
                    updated_at=now,
                )
            )
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
                event_type="page_failed",
                payload={
                    "itemId": item_id,
                    "pageId": row["page_id"],
                    "status": "failed",
                    "error": safe_error,
                    "progress": snapshot,
                },
                now=now,
            )

    def skip_remaining_item(
        self,
        fence: AttemptFence,
        *,
        step_id: str,
        reason: str,
    ) -> None:
        """Skip the current step and every downstream step for one page item."""

        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            job_status = self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            item_id = connection.execute(
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
            if item_id is None:
                raise AttemptFenced("step skip was fenced")
            connection.execute(
                update(job_steps)
                .where(
                    job_steps.c.job_item_id == item_id,
                    job_steps.c.status.in_(("pending", "running")),
                )
                .values(
                    status="skipped",
                    checkpoint_json=_json({"skipped": True, "reason": reason}),
                    attempt_id=fence.attempt_id,
                    updated_at=now,
                )
            )
            connection.execute(
                update(job_items)
                .where(job_items.c.id == item_id)
                .values(
                    status="skipped",
                    result_json=_json({"skipped": True, "reason": reason}),
                    updated_at=now,
                )
            )
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
                event_type="page_skipped",
                payload={
                    "itemId": str(item_id),
                    "stepId": step_id,
                    "reason": reason,
                    "progress": snapshot,
                },
                now=now,
            )

    def completion_status(self, fence: AttemptFence) -> str | None:
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running",),
            )
            outcome = self._completion_outcome(connection, fence.job_id)
        return outcome[0] if outcome is not None else None

    def finish_if_complete(self, fence: AttemptFence) -> str | None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running",),
            )
            outcome = self._completion_outcome(connection, fence.job_id)
            if outcome is None:
                return None
            final, failed = outcome
            final_progress = self._progress_snapshot(
                connection,
                fence.job_id,
                job_status=final,
            )
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
                    latest_progress_json=_json(final_progress),
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
                payload={
                    "status": final,
                    "failedItems": failed,
                    "progress": final_progress,
                },
                now=now,
            )
            self._bump_queue_revision(connection, now)
            self._refresh_batch_summary(connection, fence.job_id, now)
            return final

    @staticmethod
    def _completion_outcome(
        connection: Any,
        job_id: str,
    ) -> tuple[str, int] | None:
        active_steps = int(
            connection.execute(
                select(func.count())
                .select_from(job_steps)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == job_id,
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
                    job_items.c.job_id == job_id,
                    job_items.c.status == "failed",
                )
            ).scalar_one()
        )
        return (
            "completed_with_errors" if failed else "completed",
            failed,
        )

    def write_pipeline_progress(
        self,
        fence: AttemptFence,
        *,
        lock_waiting: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Overwrite the recoverable backend projection of pipeline state.

        The database step graph remains authoritative.  The only in-process
        signal accepted here is whether a claimed deep-learning step is waiting
        for the shared admission semaphore; all counts and current pages are
        reconstructed from durable rows.
        """

        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            snapshot = self._progress_snapshot(
                connection,
                fence.job_id,
                lock_waiting=lock_waiting,
            )
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
                .values(latest_progress_json=_json(snapshot), updated_at=now)
            )
            if result.rowcount != 1:
                raise AttemptFenced("job pipeline progress write was fenced")
            self._append_event(
                connection,
                job_id=fence.job_id,
                event_type="pipeline_progress",
                payload={"progress": snapshot},
                now=now,
            )
            return snapshot

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
            final_progress = self._progress_snapshot(
                connection,
                fence.job_id,
                job_status=final,
            )
            values["latest_progress_json"] = _json(final_progress)
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
                payload={
                    "source": "worker_drain",
                    "progress": final_progress,
                },
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
            message = redact_sensitive_text(
                message,
                secret_values=_job_secret_values(connection, fence.job_id),
            )
            failed_progress = self._progress_snapshot(
                connection,
                fence.job_id,
                job_status="failed",
            )
            failed_progress["error"] = {
                "code": code,
                "message": message,
            }
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
                    latest_progress_json=_json(failed_progress),
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
                payload={
                    "code": code,
                    "message": message,
                    "progress": failed_progress,
                },
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
        defer_on_control: bool,
    ) -> bool:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            job_status = self._assert_attempt(
                connection,
                fence,
                now,
                allowed_statuses=("running", "pausing", "cancelling"),
            )
            secret_values = (
                _job_secret_values(connection, fence.job_id)
                if error is not None
                else ()
            )
            safe_checkpoint = (
                redact_sensitive_value(
                    dict(checkpoint),
                    secret_values=secret_values,
                )
                if checkpoint is not None
                else None
            )
            safe_error = (
                redact_sensitive_value(
                    dict(error),
                    secret_values=secret_values,
                )
                if error is not None
                else None
            )
            step = connection.execute(
                select(
                    job_steps.c.job_item_id,
                    job_steps.c.kind,
                    job_steps.c.ordinal.label("step_ordinal"),
                    job_items.c.ordinal.label("item_ordinal"),
                    job_items.c.page_id,
                )
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_steps.c.id == step_id,
                    job_steps.c.status == "running",
                    job_steps.c.attempt_id == fence.attempt_id,
                    job_items.c.job_id == fence.job_id,
                )
            ).mappings().one_or_none()
            if step is None:
                raise AttemptFenced("step completion was fenced")
            if publisher is not None:
                publisher(connection)
            if defer_on_control and job_status in {"pausing", "cancelling"}:
                connection.execute(
                    update(job_steps)
                    .where(
                        job_steps.c.id == step_id,
                        job_steps.c.attempt_id == fence.attempt_id,
                    )
                    .values(
                        status="pending",
                        attempt_id=None,
                        checkpoint_json=(
                            _json(safe_checkpoint)
                            if safe_checkpoint
                            else None
                        ),
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(job_items)
                    .where(job_items.c.id == step["job_item_id"])
                    .values(status="pending", updated_at=now)
                )
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
                    event_type="step_checkpointed",
                    payload={
                        "itemId": str(step["job_item_id"]),
                        "pageId": (
                            str(step["page_id"])
                            if step["page_id"] is not None
                            else None
                        ),
                        "stepId": step_id,
                        "yielded": True,
                        "checkpoint": safe_checkpoint or {},
                        "progress": snapshot,
                    },
                    now=now,
                )
                return False
            connection.execute(
                update(job_steps)
                .where(
                    job_steps.c.id == step_id,
                    job_steps.c.attempt_id == fence.attempt_id,
                )
                .values(
                    status=status,
                    input_fingerprint=input_fingerprint,
                    checkpoint_json=(
                        _json(safe_checkpoint)
                        if safe_checkpoint
                        else None
                    ),
                    error_json=_json(safe_error) if safe_error else None,
                    updated_at=now,
                )
            )
            item_id = str(step["job_item_id"])
            item_completed = False
            if status == "failed":
                connection.execute(
                    update(job_items)
                    .where(job_items.c.id == item_id)
                    .values(
                        status="failed",
                        error_json=_json(safe_error or {}),
                        updated_at=now,
                    )
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
                    item_completed = True
                    connection.execute(
                        update(job_items)
                        .where(job_items.c.id == item_id)
                        .values(
                            status="completed",
                            result_json=_json(
                                {"lastCheckpoint": safe_checkpoint or {}}
                            ),
                            updated_at=now,
                        )
                    )
                    event_type = "page_completed"
                else:
                    event_type = "step_completed"
            if status == "completed":
                snapshot = self._progress_after_step_finished(
                    connection,
                    fence.job_id,
                    step_id=step_id,
                    step=step,
                    item_completed=item_completed,
                )
            else:
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
                    "pageId": (
                        str(step["page_id"])
                        if step["page_id"] is not None
                        else None
                    ),
                    "stepId": step_id,
                    "status": status,
                    "progress": snapshot,
                },
                now=now,
            )
            return True

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
            progress = _load_json(row["latest_progress_json"], {})
            progress["jobStatus"] = new_status.value
            values["latest_progress_json"] = _json(progress)
            connection.execute(
                update(jobs).where(jobs.c.id == job_id).values(**values)
            )
            self._append_event(
                connection,
                job_id=job_id,
                event_type=f"job_{event.value}",
                payload={
                    "from": current.value,
                    "to": new_status.value,
                    "progress": progress,
                },
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
        """Converge domain state when its owning job terminates."""

        job = connection.execute(
            select(
                jobs.c.kind,
                jobs.c.web_import_draft_id,
            ).where(jobs.c.id == job_id)
        ).mappings().one_or_none()
        if job is not None and job["web_import_draft_id"] is not None:
            draft_source_status = {
                "web_extract": "extracting",
                "web_import_commit": "committing",
            }.get(str(job["kind"]))
            draft_terminal_status = {
                "cancelled": "cancelled",
                "completed_with_errors": "failed",
                "failed": "failed",
            }.get(status)
            if draft_source_status and draft_terminal_status:
                connection.execute(
                    update(web_import_drafts)
                    .where(
                        web_import_drafts.c.id
                        == str(job["web_import_draft_id"]),
                        web_import_drafts.c.status == draft_source_status,
                    )
                    .values(
                        status=draft_terminal_status,
                        revision=web_import_drafts.c.revision + 1,
                        updated_at=now,
                    )
                )

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
        progress = _load_json(candidate["latest_progress_json"], {})
        progress["jobStatus"] = "running"
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
                started_at=func.coalesce(jobs.c.started_at, now),
                latest_progress_json=_json(progress),
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
            payload={"attemptId": attempt_id, "progress": progress},
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
        now: datetime,
    ) -> None:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id, jobs.c.status == "queued")
            .values(
                blocked_reason=reason,
                blocked_by_job_id=blocked_job_id,
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

    def _cancel_queued_job(
        self,
        connection: Connection,
        *,
        job_id: str,
        source: str,
        now: datetime,
    ) -> None:
        item_ids = select(job_items.c.id).where(job_items.c.job_id == job_id)
        connection.execute(
            update(job_steps)
            .where(
                job_steps.c.job_item_id.in_(item_ids),
                job_steps.c.status.in_(("pending", "running")),
            )
            .values(status="cancelled", updated_at=now)
        )
        connection.execute(
            update(job_items)
            .where(
                job_items.c.job_id == job_id,
                job_items.c.status.in_(("pending", "running")),
            )
            .values(status="cancelled", updated_at=now)
        )
        snapshot = self._progress_snapshot(
            connection,
            job_id,
            job_status="cancelled",
        )
        result = connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id, jobs.c.status == "queued")
            .values(
                status="cancelled",
                queue_rank=None,
                latest_progress_json=_json(snapshot),
                finished_at=now,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            raise JobConflict("queued job changed during cancellation")
        self._release_write_reservations(connection, job_id)
        self._sync_domain_terminal(
            connection,
            job_id=job_id,
            status="cancelled",
            now=now,
        )
        self._append_event(
            connection,
            job_id=job_id,
            event_type="job_cancelled",
            payload={"source": source, "progress": snapshot},
            now=now,
        )
        self._refresh_batch_summary(connection, job_id, now)

    def _reorder_ordinary(
        self,
        connection: Connection,
        *,
        ordered_job_ids: Sequence[str],
        now: datetime,
    ) -> None:
        # Move through a disjoint positive range to avoid transient UNIQUE
        # collisions without violating the queue-rank CHECK constraint.
        temporary_start = int(
            connection.execute(
                select(func.coalesce(func.max(jobs.c.queue_rank), 0))
            ).scalar_one()
        )
        for offset, job_id in enumerate(ordered_job_ids, start=1):
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(queue_rank=temporary_start + offset, updated_at=now)
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
            queue_rank = prefix_max + offset
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_id)
                .values(queue_rank=queue_rank, updated_at=now)
            )
            self._append_event(
                connection,
                job_id=job_id,
                event_type="job_reordered",
                payload={"queueRank": queue_rank},
                now=now,
            )

    @staticmethod
    def _load_progress_snapshot(
        connection: Any,
        job_id: str,
    ) -> dict[str, Any]:
        value = connection.execute(
            select(jobs.c.latest_progress_json).where(jobs.c.id == job_id)
        ).scalar_one()
        loaded = _load_json(value, {})
        return dict(loaded) if isinstance(loaded, Mapping) else {}

    @staticmethod
    def _refresh_current_step(snapshot: dict[str, Any]) -> bool:
        pools = snapshot.get("pools")
        if not isinstance(pools, list):
            return False
        current_steps: list[dict[str, Any]] = []
        for pool in pools:
            if not isinstance(pool, dict) or not isinstance(
                pool.get("kind"),
                str,
            ):
                return False
            current = pool.get("current")
            if not isinstance(current, list):
                return False
            for value in current:
                if not isinstance(value, dict):
                    return False
                try:
                    item_ordinal = int(value["itemOrdinal"])
                    step_ordinal = int(value["stepOrdinal"])
                except (KeyError, TypeError, ValueError):
                    return False
                current_steps.append(
                    {
                        "kind": pool["kind"],
                        **value,
                        "_sort": (item_ordinal, step_ordinal),
                    }
                )
        if current_steps:
            first = min(current_steps, key=lambda value: value["_sort"])
            first.pop("_sort", None)
            snapshot["currentStep"] = first
        else:
            snapshot.pop("currentStep", None)
        return True

    @staticmethod
    def _mutate_progress_step_started(
        snapshot: dict[str, Any],
        step: Mapping[str, Any],
    ) -> bool:
        pools = snapshot.get("pools")
        if not isinstance(pools, list):
            return False
        kind = str(step["step_kind"])
        pool = next(
            (
                value
                for value in pools
                if isinstance(value, dict) and value.get("kind") == kind
            ),
            None,
        )
        if pool is None or not isinstance(pool.get("current"), list):
            return False
        try:
            waiting = int(pool["waiting"])
            processing = int(pool["processing"])
        except (KeyError, TypeError, ValueError):
            return False
        if waiting < 1:
            return False
        pool["waiting"] = waiting - 1
        pool["processing"] = processing + 1
        pool["current"].append(
            {
                "itemId": str(step["item_id"]),
                "pageId": (
                    str(step["page_id"])
                    if step.get("page_id") is not None
                    else None
                ),
                "itemOrdinal": int(step["item_ordinal"]),
                "stepId": str(step["step_id"]),
                "stepOrdinal": int(step["step_ordinal"]),
            }
        )
        return JobQueueRepository._refresh_current_step(snapshot)

    @staticmethod
    def _progress_after_step_started(
        connection: Any,
        job_id: str,
        step: Mapping[str, Any],
    ) -> dict[str, Any]:
        snapshot = JobQueueRepository._load_progress_snapshot(
            connection,
            job_id,
        )
        if JobQueueRepository._mutate_progress_step_started(snapshot, step):
            return snapshot
        return JobQueueRepository._progress_snapshot(connection, job_id)

    @staticmethod
    def _progress_after_step_finished(
        connection: Any,
        job_id: str,
        *,
        step_id: str,
        step: Mapping[str, Any],
        item_completed: bool,
    ) -> dict[str, Any]:
        snapshot = JobQueueRepository._load_progress_snapshot(
            connection,
            job_id,
        )
        pools = snapshot.get("pools")
        if not isinstance(pools, list):
            return JobQueueRepository._progress_snapshot(connection, job_id)
        kind = str(step["kind"])
        pool = next(
            (
                value
                for value in pools
                if isinstance(value, dict) and value.get("kind") == kind
            ),
            None,
        )
        if pool is None or not isinstance(pool.get("current"), list):
            return JobQueueRepository._progress_snapshot(connection, job_id)
        try:
            processing = int(pool["processing"])
            completed = int(pool["completed"])
            completed_items = int(snapshot["completedItems"])
        except (KeyError, TypeError, ValueError):
            return JobQueueRepository._progress_snapshot(connection, job_id)
        if processing < 1:
            return JobQueueRepository._progress_snapshot(connection, job_id)
        current = [
            value
            for value in pool["current"]
            if isinstance(value, dict) and str(value.get("stepId")) != step_id
        ]
        if len(current) == len(pool["current"]):
            return JobQueueRepository._progress_snapshot(connection, job_id)
        pool["current"] = current
        pool["processing"] = processing - 1
        pool["completed"] = completed + 1
        if item_completed:
            snapshot["completedItems"] = completed_items + 1
        if not JobQueueRepository._refresh_current_step(snapshot):
            return JobQueueRepository._progress_snapshot(connection, job_id)
        return snapshot

    @staticmethod
    def _progress_snapshot(
        connection: Any,
        job_id: str,
        *,
        lock_waiting: Mapping[str, bool] | None = None,
        job_status: str | None = None,
    ) -> dict[str, Any]:
        rows = connection.execute(
            select(job_items.c.status, func.count().label("count"))
            .where(job_items.c.job_id == job_id)
            .group_by(job_items.c.status)
        )
        counts = {str(status): int(count) for status, count in rows}
        total = sum(counts.values())
        job_row = connection.execute(
            select(
                jobs.c.status,
                jobs.c.config_json,
                jobs.c.latest_progress_json,
            ).where(jobs.c.id == job_id)
        ).mappings().one()
        config = _load_json(job_row["config_json"], {})
        execution_mode = str(config.get("executionMode", "sequential"))
        effective_lock_waiting = dict(lock_waiting or {})
        if lock_waiting is None:
            previous = _load_json(job_row["latest_progress_json"], {})
            previous_pools = previous.get("pools")
            if isinstance(previous_pools, list):
                effective_lock_waiting = {
                    str(pool.get("kind")): bool(pool.get("lockWaiting", False))
                    for pool in previous_pools
                    if isinstance(pool, Mapping) and pool.get("kind")
                }

        step_rows = connection.execute(
            select(
                job_steps.c.kind,
                job_steps.c.status,
                func.count().label("count"),
                func.min(job_steps.c.ordinal).label("first_ordinal"),
            )
            .join(job_items, job_items.c.id == job_steps.c.job_item_id)
            .where(job_items.c.job_id == job_id)
            .group_by(job_steps.c.kind, job_steps.c.status)
        )
        pool_counts: dict[str, dict[str, int]] = {}
        pool_ordinals: dict[str, int] = {}
        for kind, status, count, first_ordinal in step_rows:
            name = str(kind)
            pool_counts.setdefault(name, {})[str(status)] = int(count)
            ordinal = int(first_ordinal)
            pool_ordinals[name] = min(pool_ordinals.get(name, ordinal), ordinal)

        running_rows = connection.execute(
            select(
                job_steps.c.kind,
                job_steps.c.id.label("step_id"),
                job_steps.c.ordinal.label("step_ordinal"),
                job_items.c.id.label("item_id"),
                job_items.c.page_id,
                job_items.c.ordinal.label("item_ordinal"),
            )
            .join(job_items, job_items.c.id == job_steps.c.job_item_id)
            .where(
                job_items.c.job_id == job_id,
                job_steps.c.status == "running",
            )
            .order_by(job_items.c.ordinal, job_steps.c.ordinal)
        ).mappings()
        current_by_pool: dict[str, list[dict[str, Any]]] = {}
        for row in running_rows:
            current_by_pool.setdefault(str(row["kind"]), []).append(
                {
                    "itemId": str(row["item_id"]),
                    "pageId": (
                        str(row["page_id"]) if row["page_id"] is not None else None
                    ),
                    "itemOrdinal": int(row["item_ordinal"]),
                    "stepId": str(row["step_id"]),
                    "stepOrdinal": int(row["step_ordinal"]),
                }
            )

        pools: list[dict[str, Any]] = []
        for kind in sorted(
            pool_counts,
            key=lambda value: (pool_ordinals.get(value, 0), value),
        ):
            statuses = pool_counts[kind]
            current = current_by_pool.get(kind, [])
            pools.append(
                {
                    "kind": kind,
                    "total": sum(statuses.values()),
                    "completed": statuses.get("completed", 0),
                    "failed": statuses.get("failed", 0),
                    "skipped": statuses.get("skipped", 0),
                    "waiting": statuses.get("pending", 0),
                    "processing": statuses.get("running", 0),
                    "lockWaiting": bool(effective_lock_waiting.get(kind, False)),
                    "current": current,
                }
            )

        current_steps = [
            {
                "kind": kind,
                **current,
            }
            for kind, values in current_by_pool.items()
            for current in values
        ]
        current_steps.sort(
            key=lambda value: (
                int(value["itemOrdinal"]),
                int(value["stepOrdinal"]),
            )
        )
        snapshot: dict[str, Any] = {
            "executionMode": execution_mode,
            "jobStatus": job_status or str(job_row["status"]),
            "totalItems": total,
            "completedItems": counts.get("completed", 0),
            "failedItems": counts.get("failed", 0),
            "skippedItems": counts.get("skipped", 0),
            "cancelledItems": counts.get("cancelled", 0),
            "pools": pools,
        }
        if current_steps:
            snapshot["currentStep"] = current_steps[0]
        return snapshot

    @staticmethod
    def _config_summary(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            return {}
        scalar_keys = (
            "mode",
            "executionMode",
            "sourceLanguage",
            "targetLanguage",
            "format",
            "scope",
            "method",
            "repairMethod",
            "batchSize",
            "pageCount",
            "selectedFields",
        )
        summary: dict[str, object] = {
            key: value[key]
            for key in scalar_keys
            if key in value
            and isinstance(value[key], (str, int, float, bool, list))
        }
        for section_name in (
            "translation",
            "ocr",
            "agent",
            "inpainting",
            "style",
        ):
            section = value.get(section_name)
            if not isinstance(section, Mapping):
                continue
            safe_section = {
                key: section[key]
                for key in (
                    "provider",
                    "model",
                    "model_name",
                    "method",
                    "batchSize",
                    "fontId",
                    "selectedFields",
                )
                if key in section
                and isinstance(section[key], (str, int, float, bool, list))
            }
            if "model_name" in safe_section and "model" not in safe_section:
                safe_section["model"] = safe_section.pop("model_name")
            if safe_section:
                summary[section_name] = safe_section
        rounds = value.get("proofreadingRounds")
        if isinstance(rounds, list):
            summary["proofreadingRounds"] = [
                {
                    key: round_config[key]
                    for key in ("roundIndex", "name", "provider", "model", "batchSize")
                    if key in round_config
                    and isinstance(
                        round_config[key],
                        (str, int, float, bool),
                    )
                }
                for round_config in rounds
                if isinstance(round_config, Mapping)
            ]
        return summary

    @staticmethod
    def _duration_ms(
        started_at: datetime | str | None,
        finished_at: datetime | str | None,
        *,
        running: bool,
    ) -> int | None:
        started = JobQueueRepository._datetime(started_at)
        if started is None:
            return None
        finished = JobQueueRepository._datetime(finished_at)
        if finished is None and running:
            finished = utcnow()
        if finished is None:
            return None
        return max(0, int((finished - started).total_seconds() * 1000))

    @staticmethod
    def _datetime(value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    @staticmethod
    def _job_dto(row: Mapping[str, Any]) -> dict[str, object]:
        blocked_reason = row.get("blocked_reason")
        if (
            blocked_reason is None
            and row.get("status") == "queued"
            and bool(row.get("holds_chapter_lock"))
        ):
            blocked_reason = "retained_chapter_lock"
        return {
            "jobId": row["id"],
            "batchId": row.get("batch_id"),
            "batchDisplayName": row.get("batch_display_name"),
            "kind": row["kind"],
            "retryOfJobId": row.get("retry_of_job_id"),
            "retryMode": row.get("retry_mode"),
            "status": row["status"],
            "queueRank": row.get("queue_rank"),
            "bookId": row.get("book_id"),
            "chapterId": row.get("chapter_id"),
            "pageId": row.get("page_id"),
            "blockedReason": blocked_reason,
            "blockedByJobId": row.get("blocked_by_job_id"),
            "progress": _load_json(row.get("latest_progress_json"), {}),
            "target": _load_json(row.get("target_display_json"), {}),
            "createdAt": _iso(row.get("created_at")),
            "startedAt": _iso(row.get("started_at")),
            "finishedAt": _iso(row.get("finished_at")),
        }

    @staticmethod
    def _event_dto(
        row: Mapping[str, Any],
    ) -> dict[str, object]:
        return {
            "eventId": int(row["id"]),
            "jobId": row["job_id"],
            "type": row["event_type"],
            "payload": _load_json(row["payload_json"], {}),
            "createdAt": _iso(row["created_at"]),
        }

    @staticmethod
    def _job_snapshots(
        connection: Any,
        *,
        job_ids: set[str],
    ) -> dict[str, dict[str, object]]:
        if not job_ids:
            return {}
        rows = connection.execute(
            select(
                jobs,
                job_batches.c.display_name.label("batch_display_name"),
                exists().where(
                    chapter_write_locks.c.job_id == jobs.c.id
                ).label("holds_chapter_lock"),
            )
            .join(
                job_batches,
                job_batches.c.id == jobs.c.batch_id,
                isouter=True,
            )
            .where(jobs.c.id.in_(job_ids))
        ).mappings()
        return {
            str(row["id"]): JobQueueRepository._job_dto(row)
            for row in rows
        }

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
                payload_json=_json(redact_sensitive_value(dict(payload))),
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
