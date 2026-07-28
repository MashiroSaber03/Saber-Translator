"""Fenced persistence for saved operations and coalescing render requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
from typing import Any
import uuid

from sqlalchemy import Engine, delete, exists, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Connection

from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.plugins.snapshots import enabled_plugin_snapshots
from src.backend_v2.storage.schema import (
    OPERATION_KINDS,
    bubbles,
    chapter_write_intents,
    chapter_write_locks,
    idempotency_records,
    operation_asset_inputs,
    operation_credential_snapshots,
    operation_events,
    operation_plugin_snapshots,
    operations,
    page_assets,
    pages,
    process_epochs,
    render_requests,
)


PUBLIC_PAGE_OPERATION_KINDS = frozenset(
    {"bubble_ocr", "bubble_color", "page_detect", "bubble_translate"}
)
WORKER_OPERATION_KINDS = frozenset(
    {"bubble_ocr", "bubble_color", "page_detect"}
)
API_REMOTE_OPERATION_KINDS = frozenset(
    {"bubble_translate", "studio_generate", "studio_chat", "studio_summary"}
)


class OperationNotFound(LookupError):
    pass


class OperationConflict(RuntimeError):
    pass


class OperationLocked(RuntimeError):
    pass


class OperationFenced(OperationConflict):
    pass


@dataclass(frozen=True, slots=True)
class OperationFence:
    operation_id: str
    attempt_id: str
    lease_token: str
    executor_epoch_id: str
    executor_role: str
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class RenderFence:
    render_request_id: str
    page_id: str
    rendering_revision: int
    attempt_id: str
    lease_token: str
    api_epoch_id: str
    lease_expires_at: datetime


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


def _credential_version_references(
    value: Mapping[str, Any],
) -> dict[str, str]:
    references: dict[str, str] = {}

    def visit(current: object, path: tuple[str, ...]) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                key_text = str(key)
                next_path = (*path, key_text)
                if (
                    key_text == "credentialVersionId"
                    and isinstance(child, str)
                ):
                    role = ".".join(path) or "default"
                    if len(role) > 64:
                        role = hashlib.sha256(
                            role.encode("utf-8")
                        ).hexdigest()
                    references[role] = child
                else:
                    visit(child, next_path)
        elif isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                visit(child, (*path, str(index)))

    visit(value, ())
    return references


class OperationRepository:
    def __init__(self, engine: Engine, *, attempt_lease_seconds: int = 30) -> None:
        if attempt_lease_seconds < 3:
            raise ValueError("attempt_lease_seconds must be at least 3")
        self.engine = engine
        self.attempt_lease_seconds = attempt_lease_seconds

    def create_page_operation(
        self,
        *,
        page_id: str,
        kind: str,
        base_revision: int,
        bubble_id: str | None,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        if kind not in PUBLIC_PAGE_OPERATION_KINDS:
            raise ValueError(f"unsupported public page operation kind: {kind}")
        if base_revision < 1:
            raise ValueError("baseRevision must be positive")
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("Idempotency-Key is required")
        if kind in {"bubble_ocr", "bubble_color", "bubble_translate"}:
            if not bubble_id:
                raise ValueError(f"{kind} requires bubbleId")
        elif bubble_id is not None:
            raise ValueError("page_detect does not accept bubbleId")

        request_payload = {
            "kind": kind,
            "baseRevision": base_revision,
            "bubbleId": bubble_id,
            "payload": dict(payload),
        }
        request_hash = hashlib.sha256(
            _json(request_payload).encode("utf-8")
        ).hexdigest()
        scope = f"page-operation:{page_id}"
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            replay = self._idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay, True
            page = connection.execute(
                select(pages.c.chapter_id, pages.c.document_revision).where(
                    pages.c.id == page_id
                )
            ).mappings().one_or_none()
            if page is None:
                raise OperationNotFound("page not found")
            self._assert_new_page_write_allowed(
                connection, str(page["chapter_id"])
            )
            if int(page["document_revision"]) != base_revision:
                raise OperationConflict("page document revision changed")
            if bubble_id is not None:
                exists_bubble = connection.execute(
                    select(bubbles.c.id).where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.page_id == page_id,
                    )
                ).scalar_one_or_none()
                if exists_bubble is None:
                    raise OperationNotFound("bubble does not belong to page")

            operation_id = str(uuid.uuid4())
            executor_role = (
                "worker" if kind in WORKER_OPERATION_KINDS else "api"
            )
            try:
                connection.execute(
                    insert(operations).values(
                        id=operation_id,
                        kind=kind,
                        executor_role=executor_role,
                        status="pending",
                        page_id=page_id,
                        bubble_id=bubble_id,
                        base_revision=base_revision,
                        request_json=_json(request_payload),
                        request_schema_version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
            except IntegrityError as exc:
                raise OperationConflict(
                    "page already has an active write operation"
                ) from exc
            self._bind_page_assets(
                connection,
                operation_id=operation_id,
                page_id=page_id,
                roles=self._input_roles(kind),
            )
            credential_refs = _credential_version_references(
                request_payload
            )
            if credential_refs:
                connection.execute(
                    insert(operation_credential_snapshots),
                    [
                        {
                            "operation_id": operation_id,
                            "credential_version_id": version_id,
                            "role": role,
                        }
                        for role, version_id in credential_refs.items()
                    ],
                )
            if executor_role == "worker":
                self._snapshot_plugins(
                    connection,
                    operation_id=operation_id,
                )
            response = {
                "operationId": operation_id,
                "kind": kind,
                "status": "pending",
                "executorRole": executor_role,
            }
            connection.execute(
                insert(idempotency_records).values(
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    http_status=202,
                    response_json=_json(response),
                    resource_type="operation",
                    resource_id=operation_id,
                    created_at=now,
                    expires_at=now + timedelta(days=7),
                )
            )
            return response, False

    def create_internal(
        self,
        *,
        kind: str,
        executor_role: str,
        request_payload: Mapping[str, Any],
        page_id: str | None = None,
        bubble_id: str | None = None,
        studio_document_id: str | None = None,
        studio_session_id: str | None = None,
        base_revision: int | None = None,
        base_generation: int | None = None,
        input_assets: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        if kind not in OPERATION_KINDS:
            raise ValueError(f"unsupported operation kind: {kind}")
        if executor_role not in {"api", "worker"}:
            raise ValueError("executor_role must be api or worker")
        now = utcnow()
        operation_id = str(uuid.uuid4())
        try:
            with immediate_transaction(self.engine) as connection:
                if page_id:
                    chapter_id = connection.execute(
                        select(pages.c.chapter_id).where(pages.c.id == page_id)
                    ).scalar_one_or_none()
                    if chapter_id is None:
                        raise OperationNotFound("page not found")
                    self._assert_new_page_write_allowed(
                        connection, str(chapter_id)
                    )
                connection.execute(
                    insert(operations).values(
                        id=operation_id,
                        kind=kind,
                        executor_role=executor_role,
                        status="pending",
                        page_id=page_id,
                        bubble_id=bubble_id,
                        studio_document_id=studio_document_id,
                        studio_session_id=studio_session_id,
                        base_revision=base_revision,
                        base_generation=base_generation,
                        request_json=_json(dict(request_payload)),
                        request_schema_version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
                if input_assets:
                    connection.execute(
                        insert(operation_asset_inputs),
                        [
                            {
                                "operation_id": operation_id,
                                "role": role,
                                "asset_id": asset_id,
                            }
                            for role, asset_id in input_assets.items()
                        ],
                    )
                if executor_role == "worker":
                    self._snapshot_plugins(
                        connection,
                        operation_id=operation_id,
                    )
        except IntegrityError as exc:
            raise OperationConflict("an active operation already targets this entity") from exc
        return self.get(operation_id)

    def create_page_repair(
        self,
        *,
        page_id: str,
        base_revision: int,
        method: str,
        fill_color: str | None,
        mask_asset_id: str,
        mask_checksum: str,
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        if method not in {"solid", "lama_mpe", "litelama", "restore_source"}:
            raise ValueError("unsupported page repair method")
        if method != "restore_source" and (
            not isinstance(fill_color, str) or not fill_color
        ):
            raise ValueError("fillColor is required for this repair method")
        payload = {
            "method": method,
            "fillColor": fill_color if method != "restore_source" else None,
            "repairRevision": base_revision + 1,
        }
        request_hash = self._page_repair_request_hash(
            payload=payload,
            mask_checksum=mask_checksum,
        )
        scope = f"page-repair:{page_id}"
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            replay = self._idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay, True
            page = connection.execute(
                select(pages.c.chapter_id, pages.c.document_revision).where(
                    pages.c.id == page_id
                )
            ).mappings().one_or_none()
            if page is None:
                raise OperationNotFound("page not found")
            self._assert_new_page_write_allowed(
                connection, str(page["chapter_id"])
            )
            if int(page["document_revision"]) != base_revision:
                raise OperationConflict("page document revision changed")
            source_id = connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "source",
                )
            ).scalar_one_or_none()
            parent_clean_id = connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "clean",
                )
            ).scalar_one_or_none()
            if source_id is None:
                raise OperationConflict("page has no source asset")
            operation_id = str(uuid.uuid4())
            repair_revision = base_revision + 1
            executor_role = (
                "api" if method in {"solid", "restore_source"} else "worker"
            )
            try:
                connection.execute(
                    insert(operations).values(
                        id=operation_id,
                        kind="page_repair",
                        executor_role=executor_role,
                        status="pending",
                        page_id=page_id,
                        base_revision=repair_revision,
                        request_json=_json(payload),
                        request_schema_version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
            except IntegrityError as exc:
                raise OperationConflict(
                    "page already has an active write operation"
                ) from exc
            inputs = {
                "source": str(source_id),
                "repair_mask": mask_asset_id,
            }
            if parent_clean_id is not None:
                inputs["parent_clean"] = str(parent_clean_id)
            connection.execute(
                insert(operation_asset_inputs),
                [
                    {
                        "operation_id": operation_id,
                        "role": role,
                        "asset_id": asset_id,
                    }
                    for role, asset_id in inputs.items()
                ],
            )
            if executor_role == "worker":
                self._snapshot_plugins(
                    connection,
                    operation_id=operation_id,
                )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == base_revision,
                )
                .values(
                    document_revision=repair_revision,
                    render_status="awaiting_repair",
                    updated_at=now,
                )
            )
            response = {
                "operationId": operation_id,
                "kind": "page_repair",
                "status": "pending",
                "executorRole": executor_role,
                "documentRevision": repair_revision,
            }
            connection.execute(
                insert(idempotency_records).values(
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    http_status=202,
                    response_json=_json(response),
                    resource_type="operation",
                    resource_id=operation_id,
                    created_at=now,
                    expires_at=now + timedelta(days=7),
                )
            )
        return response, False

    @staticmethod
    def _snapshot_plugins(
        connection: Connection,
        *,
        operation_id: str,
    ) -> None:
        snapshots = enabled_plugin_snapshots(connection)
        if not snapshots:
            return
        connection.execute(
            insert(operation_plugin_snapshots),
            [
                {
                    "operation_id": operation_id,
                    "plugin_version_id": version_id,
                    "config_json": _json(dict(snapshot)),
                }
                for version_id, snapshot in snapshots.items()
            ],
        )

    def find_page_repair_replay(
        self,
        *,
        page_id: str,
        base_revision: int,
        method: str,
        fill_color: str | None,
        mask_checksum: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        if method not in {"solid", "lama_mpe", "litelama", "restore_source"}:
            raise ValueError("unsupported page repair method")
        payload = {
            "method": method,
            "fillColor": fill_color if method != "restore_source" else None,
            "repairRevision": base_revision + 1,
        }
        request_hash = self._page_repair_request_hash(
            payload=payload,
            mask_checksum=mask_checksum,
        )
        with self.engine.connect() as connection:
            return self._idempotency_replay(
                connection,
                scope=f"page-repair:{page_id}",
                key=idempotency_key,
                request_hash=request_hash,
                now=utcnow(),
            )

    def get(self, operation_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(operations).where(operations.c.id == operation_id)
            ).mappings().one_or_none()
        if row is None:
            raise OperationNotFound("operation not found")
        return self._dto(row)

    def events_after(
        self,
        operation_id: str,
        *,
        after: int = 0,
        limit: int = 500,
    ) -> list[dict[str, object]]:
        if limit < 1 or limit > 2000:
            raise ValueError("limit must be between 1 and 2000")
        with self.engine.connect() as connection:
            if connection.execute(
                select(operations.c.id).where(
                    operations.c.id == operation_id
                )
            ).scalar_one_or_none() is None:
                raise OperationNotFound("operation not found")
            rows = list(
                connection.execute(
                    select(operation_events)
                    .where(
                        operation_events.c.operation_id == operation_id,
                        operation_events.c.id > max(0, after),
                    )
                    .order_by(operation_events.c.id)
                    .limit(limit)
                ).mappings()
            )
        return [
            {
                "eventId": int(row["id"]),
                "operationId": str(row["operation_id"]),
                "type": str(row["type"]),
                "payload": _load_json(row["payload_json"], {}),
                "createdAt": (
                    row["created_at"].isoformat() + "Z"
                    if hasattr(row["created_at"], "isoformat")
                    else str(row["created_at"])
                ),
            }
            for row in rows
        ]

    def append_event(
        self,
        fence: OperationFence,
        *,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int:
        if not event_type or len(event_type) > 64:
            raise ValueError("operation event type is invalid")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_fence(connection, fence, now)
            cursor = connection.execute(
                insert(operation_events).values(
                    operation_id=fence.operation_id,
                    type=event_type,
                    payload_json=_json(dict(payload)),
                    created_at=now,
                )
            ).inserted_primary_key[0]
        return int(cursor)

    def claim_next(
        self,
        *,
        executor_role: str,
        executor_epoch_id: str,
        allowed_kinds: Sequence[str],
    ) -> tuple[OperationFence, dict[str, object]] | None:
        if executor_role not in {"api", "worker"}:
            raise ValueError("executor_role must be api or worker")
        if not allowed_kinds:
            return None
        now = utcnow()
        expires = now + timedelta(seconds=self.attempt_lease_seconds)
        with immediate_transaction(self.engine) as connection:
            self._assert_epoch(
                connection,
                role=executor_role,
                epoch_id=executor_epoch_id,
                now=now,
            )
            row = connection.execute(
                select(operations)
                .where(
                    operations.c.executor_role == executor_role,
                    operations.c.status == "pending",
                    operations.c.kind.in_(allowed_kinds),
                )
                .order_by(operations.c.created_at)
                .limit(1)
            ).mappings().one_or_none()
            if row is None:
                return None
            attempt_id = str(uuid.uuid4())
            lease_token = secrets.token_urlsafe(32)
            changed = connection.execute(
                update(operations)
                .where(
                    operations.c.id == row["id"],
                    operations.c.status == "pending",
                )
                .values(
                    status="running",
                    executor_epoch_id=executor_epoch_id,
                    attempt_id=attempt_id,
                    lease_token=lease_token,
                    lease_expires_at=expires,
                    started_at=now,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                return None
            connection.execute(
                insert(operation_events).values(
                    operation_id=row["id"],
                    type="operation_started",
                    payload_json=_json({"status": "running"}),
                    created_at=now,
                )
            )
            dto = self._dto(dict(row))
            dto["status"] = "running"
            dto["inputs"] = {
                str(role): str(asset_id)
                for role, asset_id in connection.execute(
                    select(
                        operation_asset_inputs.c.role,
                        operation_asset_inputs.c.asset_id,
                    ).where(
                        operation_asset_inputs.c.operation_id == row["id"]
                    )
                )
            }
            return (
                OperationFence(
                    operation_id=str(row["id"]),
                    attempt_id=attempt_id,
                    lease_token=lease_token,
                    executor_epoch_id=executor_epoch_id,
                    executor_role=executor_role,
                    lease_expires_at=expires,
                ),
                dto,
            )

    def renew(self, fence: OperationFence) -> OperationFence | None:
        now = utcnow()
        expires = now + timedelta(seconds=self.attempt_lease_seconds)
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(operations)
                .where(
                    operations.c.id == fence.operation_id,
                    operations.c.status == "running",
                    operations.c.attempt_id == fence.attempt_id,
                    operations.c.lease_token == fence.lease_token,
                    operations.c.executor_epoch_id == fence.executor_epoch_id,
                    operations.c.executor_role == fence.executor_role,
                    operations.c.lease_expires_at > now,
                    exists(
                        select(process_epochs.c.id).where(
                            process_epochs.c.id == fence.executor_epoch_id,
                            process_epochs.c.role == fence.executor_role,
                            process_epochs.c.status == "active",
                            process_epochs.c.lease_expires_at > now,
                        )
                    ),
                )
                .values(lease_expires_at=expires, updated_at=now)
            )
        if changed.rowcount != 1:
            return None
        return OperationFence(
            operation_id=fence.operation_id,
            attempt_id=fence.attempt_id,
            lease_token=fence.lease_token,
            executor_epoch_id=fence.executor_epoch_id,
            executor_role=fence.executor_role,
            lease_expires_at=expires,
        )

    def complete(
        self,
        fence: OperationFence,
        *,
        result: Mapping[str, Any],
        publisher: Callable[[Connection, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._finish(fence, result=result, error=None, publisher=publisher)

    def fail(
        self,
        fence: OperationFence,
        *,
        code: str,
        message: str,
    ) -> None:
        self._finish(
            fence,
            result=None,
            error={"code": code, "message": message},
            publisher=None,
        )

    def _finish(
        self,
        fence: OperationFence,
        *,
        result: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
        publisher: Callable[[Connection, Mapping[str, Any]], None] | None,
    ) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            row = self._assert_fence(connection, fence, now)
            if row["page_id"] is not None and row["base_revision"] is not None:
                revision = connection.execute(
                    select(pages.c.document_revision).where(
                        pages.c.id == row["page_id"]
                    )
                ).scalar_one_or_none()
                if revision != row["base_revision"]:
                    raise OperationFenced("page revision changed before operation publish")
            if publisher is not None:
                publisher(connection, row)
            status = "completed" if error is None else "failed"
            if error is not None and row["kind"] == "page_repair":
                connection.execute(
                    update(pages)
                    .where(
                        pages.c.id == row["page_id"],
                        pages.c.document_revision == row["base_revision"],
                    )
                    .values(render_status="repair_failed", updated_at=now)
                )
            changed = connection.execute(
                update(operations)
                .where(
                    operations.c.id == fence.operation_id,
                    operations.c.status == "running",
                    operations.c.attempt_id == fence.attempt_id,
                    operations.c.lease_token == fence.lease_token,
                    operations.c.executor_epoch_id == fence.executor_epoch_id,
                )
                .values(
                    status=status,
                    result_json=_json(dict(result)) if result is not None else None,
                    error_json=_json(dict(error)) if error is not None else None,
                    executor_epoch_id=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    finished_at=now,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise OperationFenced("operation terminal write was fenced")
            connection.execute(
                insert(operation_events).values(
                    operation_id=fence.operation_id,
                    type=(
                        "operation_completed"
                        if error is None
                        else "operation_failed"
                    ),
                    payload_json=_json(
                        {
                            "status": status,
                            **(
                                {"result": dict(result)}
                                if result is not None
                                else {"error": dict(error or {})}
                            ),
                        }
                    ),
                    created_at=now,
                )
            )

    @staticmethod
    def _assert_new_page_write_allowed(
        connection: Connection,
        chapter_id: str,
    ) -> None:
        intent = connection.execute(
            select(chapter_write_intents.c.chapter_id).where(
                chapter_write_intents.c.chapter_id == chapter_id
            )
        ).scalar_one_or_none()
        lock = connection.execute(
            select(chapter_write_locks.c.chapter_id).where(
                chapter_write_locks.c.chapter_id == chapter_id
            )
        ).scalar_one_or_none()
        if intent is not None:
            raise OperationLocked("chapter_write_pending")
        if lock is not None:
            raise OperationLocked("chapter_locked")

    @staticmethod
    def _input_roles(kind: str) -> tuple[str, ...]:
        if kind in {"bubble_ocr", "bubble_color", "page_detect"}:
            return ("source",)
        return ()

    @staticmethod
    def _bind_page_assets(
        connection: Connection,
        *,
        operation_id: str,
        page_id: str,
        roles: Sequence[str],
    ) -> None:
        if not roles:
            return
        rows = connection.execute(
            select(page_assets.c.role, page_assets.c.asset_id).where(
                page_assets.c.page_id == page_id,
                page_assets.c.role.in_(roles),
            )
        )
        found = {str(role): str(asset_id) for role, asset_id in rows}
        missing = set(roles) - set(found)
        if missing:
            raise OperationConflict(
                f"page is missing required assets: {', '.join(sorted(missing))}"
            )
        connection.execute(
            insert(operation_asset_inputs),
            [
                {
                    "operation_id": operation_id,
                    "role": role,
                    "asset_id": asset_id,
                }
                for role, asset_id in found.items()
            ],
        )

    @staticmethod
    def _assert_epoch(
        connection: Connection,
        *,
        role: str,
        epoch_id: str,
        now: datetime,
    ) -> None:
        value = connection.execute(
            select(process_epochs.c.id).where(
                process_epochs.c.id == epoch_id,
                process_epochs.c.role == role,
                process_epochs.c.status == "active",
                process_epochs.c.lease_expires_at > now,
            )
        ).scalar_one_or_none()
        if value is None:
            raise OperationFenced(f"{role} epoch is inactive or expired")

    @staticmethod
    def _assert_fence(
        connection: Connection,
        fence: OperationFence,
        now: datetime,
    ) -> Mapping[str, Any]:
        row = connection.execute(
            select(operations).where(
                operations.c.id == fence.operation_id,
                operations.c.status == "running",
                operations.c.attempt_id == fence.attempt_id,
                operations.c.lease_token == fence.lease_token,
                operations.c.executor_epoch_id == fence.executor_epoch_id,
                operations.c.executor_role == fence.executor_role,
                operations.c.lease_expires_at > now,
                exists(
                    select(process_epochs.c.id).where(
                        process_epochs.c.id == fence.executor_epoch_id,
                        process_epochs.c.role == fence.executor_role,
                        process_epochs.c.status == "active",
                        process_epochs.c.lease_expires_at > now,
                    )
                ),
            )
        ).mappings().one_or_none()
        if row is None:
            raise OperationFenced("operation attempt lost execution rights")
        return row

    @staticmethod
    def _idempotency_replay(
        connection: Connection,
        *,
        scope: str,
        key: str,
        request_hash: str,
        now: datetime,
    ) -> dict[str, object] | None:
        row = connection.execute(
            select(
                idempotency_records.c.request_hash,
                idempotency_records.c.response_json,
            ).where(
                idempotency_records.c.scope == scope,
                idempotency_records.c.key == key,
                idempotency_records.c.expires_at > now,
            )
        ).mappings().one_or_none()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise OperationConflict(
                "Idempotency-Key was reused for a different operation"
            )
        return json.loads(row["response_json"])

    @staticmethod
    def _page_repair_request_hash(
        *,
        payload: Mapping[str, Any],
        mask_checksum: str,
    ) -> str:
        canonical = {
            "request": dict(payload),
            "maskChecksum": mask_checksum,
        }
        return hashlib.sha256(_json(canonical).encode("utf-8")).hexdigest()

    @staticmethod
    def _dto(row: Mapping[str, Any]) -> dict[str, object]:
        return {
            "operationId": row["id"],
            "kind": row["kind"],
            "executorRole": row["executor_role"],
            "status": row["status"],
            "pageId": row.get("page_id"),
            "bubbleId": row.get("bubble_id"),
            "studioDocumentId": row.get("studio_document_id"),
            "studioSessionId": row.get("studio_session_id"),
            "baseRevision": row.get("base_revision"),
            "baseGeneration": row.get("base_generation"),
            "request": _load_json(row.get("request_json"), {}),
            "result": _load_json(row.get("result_json"), None),
            "error": _load_json(row.get("error_json"), None),
            "createdAt": _iso(row.get("created_at")),
            "startedAt": _iso(row.get("started_at")),
            "finishedAt": _iso(row.get("finished_at")),
        }


class RenderRequestRepository:
    def __init__(self, engine: Engine, *, attempt_lease_seconds: int = 30) -> None:
        self.engine = engine
        self.attempt_lease_seconds = attempt_lease_seconds

    def upsert(
        self,
        connection: Connection,
        *,
        page_id: str,
        requested_revision: int,
        existing_chain: bool = False,
    ) -> str:
        page = connection.execute(
            select(pages.c.chapter_id, pages.c.document_revision).where(
                pages.c.id == page_id
            )
        ).mappings().one_or_none()
        if page is None:
            raise OperationNotFound("page not found")
        if int(page["document_revision"]) != requested_revision:
            raise OperationConflict("render revision is not the current document")
        if not existing_chain:
            OperationRepository._assert_new_page_write_allowed(
                connection, str(page["chapter_id"])
            )
        existing = connection.execute(
            select(
                render_requests.c.id,
                render_requests.c.status,
            ).where(
                render_requests.c.page_id == page_id
            )
        ).mappings().one_or_none()
        now = utcnow()
        if existing is None:
            request_id = str(uuid.uuid4())
            connection.execute(
                insert(render_requests).values(
                    id=request_id,
                    page_id=page_id,
                    requested_revision=requested_revision,
                    status="pending",
                    created_at=now,
                    updated_at=now,
                )
            )
            return request_id
        values: dict[str, object] = {
            "requested_revision": requested_revision,
            "error_json": None,
            "updated_at": now,
        }
        if existing["status"] != "running":
            values["status"] = "pending"
        connection.execute(
            update(render_requests)
            .where(render_requests.c.id == existing["id"])
            .values(**values)
        )
        return str(existing["id"])

    def claim_next(self, *, api_epoch_id: str) -> RenderFence | None:
        now = utcnow()
        expires = now + timedelta(seconds=self.attempt_lease_seconds)
        with immediate_transaction(self.engine) as connection:
            OperationRepository._assert_epoch(
                connection, role="api", epoch_id=api_epoch_id, now=now
            )
            row = connection.execute(
                select(render_requests)
                .where(render_requests.c.status == "pending")
                .order_by(render_requests.c.updated_at)
                .limit(1)
            ).mappings().one_or_none()
            if row is None:
                return None
            page = connection.execute(
                select(pages.c.render_status).where(pages.c.id == row["page_id"])
            ).scalar_one_or_none()
            if page in {"awaiting_repair", "repair_failed"}:
                return None
            attempt_id = str(uuid.uuid4())
            lease_token = secrets.token_urlsafe(32)
            changed = connection.execute(
                update(render_requests)
                .where(
                    render_requests.c.id == row["id"],
                    render_requests.c.status == "pending",
                )
                .values(
                    status="running",
                    rendering_revision=row["requested_revision"],
                    executor_epoch_id=api_epoch_id,
                    attempt_id=attempt_id,
                    lease_token=lease_token,
                    lease_expires_at=expires,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                return None
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == row["page_id"],
                    pages.c.document_revision == row["requested_revision"],
                )
                .values(render_status="rendering", updated_at=now)
            )
            return RenderFence(
                render_request_id=str(row["id"]),
                page_id=str(row["page_id"]),
                rendering_revision=int(row["requested_revision"]),
                attempt_id=attempt_id,
                lease_token=lease_token,
                api_epoch_id=api_epoch_id,
                lease_expires_at=expires,
            )

    def renew(self, fence: RenderFence) -> RenderFence | None:
        now = utcnow()
        expires = now + timedelta(seconds=self.attempt_lease_seconds)
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(render_requests)
                .where(
                    render_requests.c.id == fence.render_request_id,
                    render_requests.c.status == "running",
                    render_requests.c.rendering_revision
                    == fence.rendering_revision,
                    render_requests.c.attempt_id == fence.attempt_id,
                    render_requests.c.lease_token == fence.lease_token,
                    render_requests.c.executor_epoch_id == fence.api_epoch_id,
                    render_requests.c.lease_expires_at > now,
                    exists(
                        select(process_epochs.c.id).where(
                            process_epochs.c.id == fence.api_epoch_id,
                            process_epochs.c.role == "api",
                            process_epochs.c.status == "active",
                            process_epochs.c.lease_expires_at > now,
                        )
                    ),
                )
                .values(lease_expires_at=expires, updated_at=now)
            )
        if changed.rowcount != 1:
            return None
        return RenderFence(
            render_request_id=fence.render_request_id,
            page_id=fence.page_id,
            rendering_revision=fence.rendering_revision,
            attempt_id=fence.attempt_id,
            lease_token=fence.lease_token,
            api_epoch_id=fence.api_epoch_id,
            lease_expires_at=expires,
        )

    def complete(
        self,
        fence: RenderFence,
        *,
        publisher: Callable[[Connection], None],
    ) -> bool:
        """Publish only if this is still the page's newest requested revision."""

        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(render_requests).where(
                    render_requests.c.id == fence.render_request_id,
                    render_requests.c.status == "running",
                    render_requests.c.rendering_revision
                    == fence.rendering_revision,
                    render_requests.c.attempt_id == fence.attempt_id,
                    render_requests.c.lease_token == fence.lease_token,
                    render_requests.c.executor_epoch_id == fence.api_epoch_id,
                    render_requests.c.lease_expires_at > now,
                    exists(
                        select(process_epochs.c.id).where(
                            process_epochs.c.id == fence.api_epoch_id,
                            process_epochs.c.role == "api",
                            process_epochs.c.status == "active",
                            process_epochs.c.lease_expires_at > now,
                        )
                    ),
                )
            ).mappings().one_or_none()
            if row is None:
                raise OperationFenced("render attempt lost execution rights")
            document_revision = connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == fence.page_id
                )
            ).scalar_one_or_none()
            if (
                row["requested_revision"] != fence.rendering_revision
                or document_revision != fence.rendering_revision
            ):
                connection.execute(
                    update(render_requests)
                    .where(render_requests.c.id == fence.render_request_id)
                    .values(
                        status="pending",
                        rendering_revision=None,
                        executor_epoch_id=None,
                        attempt_id=None,
                        lease_token=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                return False
            publisher(connection)
            connection.execute(
                update(render_requests)
                .where(render_requests.c.id == fence.render_request_id)
                .values(
                    status="completed",
                    completed_revision=fence.rendering_revision,
                    executor_epoch_id=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == fence.page_id,
                    pages.c.document_revision == fence.rendering_revision,
                )
                .values(
                    rendered_revision=fence.rendering_revision,
                    render_status="ready",
                    updated_at=now,
                )
            )
            return True

    def fail(self, fence: RenderFence, *, code: str, message: str) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(render_requests)
                .where(
                    render_requests.c.id == fence.render_request_id,
                    render_requests.c.status == "running",
                    render_requests.c.attempt_id == fence.attempt_id,
                    render_requests.c.lease_token == fence.lease_token,
                    render_requests.c.executor_epoch_id == fence.api_epoch_id,
                    render_requests.c.lease_expires_at > now,
                )
                .values(
                    status="failed",
                    error_json=_json({"code": code, "message": message}),
                    executor_epoch_id=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise OperationFenced("render failure write was fenced")
            connection.execute(
                update(pages)
                .where(pages.c.id == fence.page_id)
                .values(render_status="render_failed", updated_at=now)
            )


def _iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
