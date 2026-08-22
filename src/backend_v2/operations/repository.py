"""Fenced persistence for saved operations and coalescing render requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import re
from typing import Any
import uuid

from sqlalchemy import Engine, exists, insert, select, update
from sqlalchemy.exc import IntegrityError

from src.backend_v2.auth.constants import LOCAL_USER_ID
from src.backend_v2.auth.ownership import effective_owner_id
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import iso_utc as _iso, utcnow
from src.backend_v2.redaction import (
    credential_version_references,
    redact_sensitive_value,
    secret_values_from_json,
)
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.plugins.snapshots import enabled_plugin_snapshots
from src.backend_v2.storage.schema import (
    bubbles,
    chapter_write_locks,
    credential_versions,
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
_REPAIR_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


class OperationNotFound(LookupError):
    pass


class OperationConflict(RuntimeError):
    pass


class OperationLocked(RuntimeError):
    pass


class OperationFenced(OperationConflict):
    pass


class OperationDataInvalid(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OperationFence:
    operation_id: str
    attempt_id: str
    executor_epoch_id: str
    executor_role: str
    owner_user_id: str = LOCAL_USER_ID


@dataclass(frozen=True, slots=True)
class RenderFence:
    render_request_id: str
    page_id: str
    rendering_revision: int
    attempt_id: str
    api_epoch_id: str
    owner_user_id: str = LOCAL_USER_ID


def _load_required_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise OperationDataInvalid(f"{field} is missing")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise OperationDataInvalid(f"{field} contains invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise OperationDataInvalid(f"{field} must contain a JSON object")
    return dict(decoded)


def _load_optional_object(
    value: object,
    field: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return _load_required_object(value, field)


def _operation_secret_values(
    connection: Connection,
    operation_id: str,
) -> tuple[str, ...]:
    values: set[str] = set()
    secret_rows = connection.execute(
        select(credential_versions.c.secret_json)
        .join(
            operation_credential_snapshots,
            operation_credential_snapshots.c.credential_version_id
            == credential_versions.c.id,
        )
        .where(
            operation_credential_snapshots.c.operation_id == operation_id
        )
    ).scalars()
    for secret_json in secret_rows:
        values.update(secret_values_from_json(str(secret_json)))
    return tuple(sorted(values, key=len, reverse=True))


class OperationRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

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
        if not isinstance(kind, str) or kind not in PUBLIC_PAGE_OPERATION_KINDS:
            raise ValueError(f"unsupported public page operation kind: {kind}")
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 1
        ):
            raise ValueError("baseRevision must be positive")
        if not isinstance(payload, Mapping):
            raise ValueError("operation payload must be an object")
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 200
        ):
            raise ValueError("Idempotency-Key is required")
        if kind in {"bubble_ocr", "bubble_color", "bubble_translate"}:
            if not isinstance(bubble_id, str) or not bubble_id:
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
                        owner_user_id=effective_owner_id(),
                        kind=kind,
                        executor_role=executor_role,
                        status="pending",
                        page_id=page_id,
                        bubble_id=bubble_id,
                        base_revision=base_revision,
                        request_json=_json(request_payload),
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
            credential_refs = credential_version_references(
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
                    owner_user_id=effective_owner_id(),
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

    def create_page_repair(
        self,
        *,
        page_id: str,
        base_revision: int,
        method: str,
        fill_color: str | None,
        disable_resize: bool,
        settings_snapshot: object,
        mask_asset_id: str,
        mask_checksum: str,
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        self.validate_page_repair_identity(
            base_revision=base_revision,
            method=method,
            fill_color=fill_color,
        )
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 200
        ):
            raise ValueError("Idempotency-Key is required")
        request_identity = {
            "method": method,
            "repairRevision": base_revision + 1,
        }
        if method == "solid":
            request_identity["fillColor"] = fill_color
        request_hash = self._page_repair_request_hash(
            payload=request_identity,
            mask_checksum=mask_checksum,
        )
        payload = dict(request_identity)
        if method in {"lama_mpe", "litelama"}:
            if not isinstance(disable_resize, bool):
                raise ValueError("disableResize must be boolean")
            if not isinstance(settings_snapshot, Mapping):
                raise ValueError("settingsSnapshot must be an object")
            payload.update(
                {
                    "disableResize": disable_resize,
                    "settingsSnapshot": dict(settings_snapshot),
                }
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
                        owner_user_id=effective_owner_id(),
                        kind="page_repair",
                        executor_role=executor_role,
                        status="pending",
                        page_id=page_id,
                        base_revision=repair_revision,
                        request_json=_json(payload),
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
            changed = connection.execute(
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
            if changed.rowcount != 1:
                raise OperationConflict("page document revision changed")
            connection.execute(
                update(bubbles)
                .where(bubbles.c.page_id == page_id)
                .values(
                    updated_revision=repair_revision,
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
                    owner_user_id=effective_owner_id(),
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
        self.validate_page_repair_identity(
            base_revision=base_revision,
            method=method,
            fill_color=fill_color,
        )
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 200
        ):
            raise ValueError("Idempotency-Key is required")
        payload = {
            "method": method,
            "repairRevision": base_revision + 1,
        }
        if method == "solid":
            payload["fillColor"] = fill_color
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
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be a non-negative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > 2000
        ):
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
                        operation_events.c.id > after,
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
                "payload": _load_required_object(
                    row["payload_json"],
                    "operation_events.payload_json",
                ),
                "createdAt": row["created_at"].isoformat() + "Z",
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
        if (
            not isinstance(event_type, str)
            or not event_type
            or len(event_type) > 64
        ):
            raise ValueError("operation event type is invalid")
        if not isinstance(payload, Mapping):
            raise ValueError("operation event payload must be an object")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_fence(connection, fence, now)
            secret_values = _operation_secret_values(
                connection,
                fence.operation_id,
            )
            cursor = connection.execute(
                insert(operation_events).values(
                    operation_id=fence.operation_id,
                    type=event_type,
                    payload_json=_json(
                        redact_sensitive_value(
                            dict(payload),
                            secret_values=secret_values,
                        )
                    ),
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
        claim_conditions = (
            operations.c.executor_role == executor_role,
            operations.c.status == "pending",
            operations.c.kind.in_(tuple(allowed_kinds)),
        )
        now = utcnow()
        with self.engine.connect() as connection:
            self._assert_epoch(
                connection,
                role=executor_role,
                epoch_id=executor_epoch_id,
                now=now,
            )
            pending_id = connection.execute(
                select(operations.c.id)
                .where(*claim_conditions)
                .order_by(operations.c.created_at)
                .limit(1)
            ).scalar_one_or_none()
        if pending_id is None:
            return None

        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_epoch(
                connection,
                role=executor_role,
                epoch_id=executor_epoch_id,
                now=now,
            )
            row = connection.execute(
                select(operations)
                .where(*claim_conditions)
                .order_by(operations.c.created_at)
                .limit(1)
            ).mappings().one_or_none()
            if row is None:
                return None
            try:
                dto = self._dto(row)
            except OperationDataInvalid as exc:
                error = {
                    "code": "OPERATION_DATA_INVALID",
                    "message": str(exc),
                }
                try:
                    preserved_request = _load_required_object(
                        row["request_json"],
                        "operations.request_json",
                    )
                except OperationDataInvalid:
                    preserved_request = {
                        "discardedInvalidStoredRequest": True,
                    }
                connection.execute(
                    update(operations)
                    .where(
                        operations.c.id == row["id"],
                        operations.c.status == "pending",
                    )
                    .values(
                        status="failed",
                        request_json=_json(preserved_request),
                        result_json=None,
                        error_json=_json(error),
                        finished_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(operation_events).values(
                        operation_id=row["id"],
                        type="operation_failed",
                        payload_json=_json(
                            {"status": "failed", "error": error}
                        ),
                        created_at=now,
                    )
                )
                return None
            attempt_id = str(uuid.uuid4())
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
                    executor_epoch_id=executor_epoch_id,
                    executor_role=executor_role,
                    owner_user_id=str(row["owner_user_id"]),
                ),
                dto,
            )

    @staticmethod
    def validate_page_repair_identity(
        *,
        base_revision: object,
        method: object,
        fill_color: object,
    ) -> tuple[int, str, str | None]:
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 1
        ):
            raise ValueError("baseRevision must be positive")
        if not isinstance(method, str) or method not in {
            "solid",
            "lama_mpe",
            "litelama",
            "restore_source",
        }:
            raise ValueError("unsupported page repair method")
        if method != "solid":
            if fill_color is not None:
                raise ValueError(f"{method} does not accept fillColor")
            return base_revision, method, None
        if (
            not isinstance(fill_color, str)
            or _REPAIR_COLOR_PATTERN.fullmatch(fill_color) is None
        ):
            raise ValueError("fillColor must be a #RRGGBB color")
        return base_revision, method, fill_color

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
            secret_values = _operation_secret_values(
                connection,
                fence.operation_id,
            )
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
            safe_result = (
                redact_sensitive_value(
                    dict(result),
                    secret_values=secret_values,
                )
                if result is not None
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
                    operations.c.executor_epoch_id == fence.executor_epoch_id,
                )
                .values(
                    status=status,
                    result_json=(
                        _json(safe_result)
                        if safe_result is not None
                        else None
                    ),
                    error_json=(
                        _json(safe_error)
                        if safe_error is not None
                        else None
                    ),
                    executor_epoch_id=None,
                    attempt_id=None,
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
                                {"result": safe_result}
                                if safe_result is not None
                                else {"error": safe_error or {}}
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
        lock = connection.execute(
            select(chapter_write_locks.c.chapter_id).where(
                chapter_write_locks.c.chapter_id == chapter_id
            )
        ).scalar_one_or_none()
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
                operations.c.executor_epoch_id == fence.executor_epoch_id,
                operations.c.executor_role == fence.executor_role,
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
                idempotency_records.c.owner_user_id == effective_owner_id(),
                idempotency_records.c.expires_at > now,
            )
        ).mappings().one_or_none()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise OperationConflict(
                "Idempotency-Key was reused for a different operation"
            )
        return _load_required_object(
            row["response_json"],
            "idempotency_records.response_json",
        )

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
        request = _load_required_object(
            row["request_json"],
            "operations.request_json",
        )
        result = _load_optional_object(
            row["result_json"],
            "operations.result_json",
        )
        error = _load_optional_object(
            row["error_json"],
            "operations.error_json",
        )
        status = row["status"]
        if status in {"pending", "running", "cancelled"}:
            valid_terminal_shape = result is None and error is None
        elif status == "completed":
            valid_terminal_shape = result is not None and error is None
        elif status == "failed":
            valid_terminal_shape = result is None and error is not None
        else:
            valid_terminal_shape = False
        if not valid_terminal_shape:
            raise OperationDataInvalid(
                "operation status/result/error state is invalid"
            )
        return {
            "operationId": row["id"],
            "kind": row["kind"],
            "executorRole": row["executor_role"],
            "status": status,
            "pageId": row["page_id"],
            "bubbleId": row["bubble_id"],
            "studioDocumentId": row["studio_document_id"],
            "studioSessionId": row["studio_session_id"],
            "baseRevision": row["base_revision"],
            "baseGeneration": row["base_generation"],
            "request": request,
            "result": result,
            "error": error,
            "createdAt": _iso(row["created_at"]),
            "startedAt": _iso(row["started_at"]),
            "finishedAt": _iso(row["finished_at"]),
        }


class RenderRequestRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def upsert(
        self,
        connection: Connection,
        *,
        page_id: str,
        requested_revision: int,
        existing_chain: bool = False,
    ) -> str:
        if (
            isinstance(requested_revision, bool)
            or not isinstance(requested_revision, int)
            or requested_revision < 1
        ):
            raise ValueError("requested_revision must be a positive integer")
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
                    owner_user_id=effective_owner_id(),
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
        # The render executor polls frequently.  Avoid taking SQLite's write
        # reservation when there is no eligible work; the transactional query
        # below remains the authoritative claim and safely handles races.
        now = utcnow()
        with self.engine.connect() as connection:
            OperationRepository._assert_epoch(
                connection,
                role="api",
                epoch_id=api_epoch_id,
                now=now,
            )
            has_pending = connection.execute(
                select(render_requests.c.id)
                .join(pages, pages.c.id == render_requests.c.page_id)
                .where(
                    render_requests.c.status == "pending",
                    pages.c.render_status.not_in(
                        ("awaiting_repair", "repair_failed")
                    ),
                )
                .limit(1)
            ).scalar_one_or_none()
        if has_pending is None:
            return None

        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            OperationRepository._assert_epoch(
                connection, role="api", epoch_id=api_epoch_id, now=now
            )
            row = connection.execute(
                select(render_requests)
                .join(pages, pages.c.id == render_requests.c.page_id)
                .where(
                    render_requests.c.status == "pending",
                    pages.c.render_status.not_in(
                        ("awaiting_repair", "repair_failed")
                    ),
                )
                .order_by(render_requests.c.updated_at)
                .limit(1)
            ).mappings().one_or_none()
            if row is None:
                return None
            attempt_id = str(uuid.uuid4())
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
                api_epoch_id=api_epoch_id,
                owner_user_id=str(row["owner_user_id"]),
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
                    render_requests.c.executor_epoch_id == fence.api_epoch_id,
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
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(pages)
                    .where(
                        pages.c.id == fence.page_id,
                        pages.c.document_revision == row["requested_revision"],
                        pages.c.render_status == "rendering",
                    )
                    .values(render_status="stale", updated_at=now)
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
            row = connection.execute(
                select(render_requests).where(
                    render_requests.c.id == fence.render_request_id,
                    render_requests.c.status == "running",
                    render_requests.c.rendering_revision
                    == fence.rendering_revision,
                    render_requests.c.attempt_id == fence.attempt_id,
                    render_requests.c.executor_epoch_id == fence.api_epoch_id,
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
                raise OperationFenced("render failure write was fenced")
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
                        error_json=None,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(pages)
                    .where(
                        pages.c.id == fence.page_id,
                        pages.c.document_revision == row["requested_revision"],
                        pages.c.render_status == "rendering",
                    )
                    .values(render_status="stale", updated_at=now)
                )
                return

            changed = connection.execute(
                update(render_requests)
                .where(
                    render_requests.c.id == fence.render_request_id,
                    render_requests.c.status == "running",
                    render_requests.c.rendering_revision
                    == fence.rendering_revision,
                    render_requests.c.attempt_id == fence.attempt_id,
                    render_requests.c.executor_epoch_id == fence.api_epoch_id,
                )
                .values(
                    status="failed",
                    error_json=_json(
                        redact_sensitive_value(
                            {"code": code, "message": message}
                        )
                    ),
                    executor_epoch_id=None,
                    attempt_id=None,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise OperationFenced("render failure write was fenced")
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == fence.page_id,
                    pages.c.document_revision == fence.rendering_revision,
                )
                .values(render_status="render_failed", updated_at=now)
            )
