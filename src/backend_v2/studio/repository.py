"""Transactional Character Studio documents, sessions, and saved operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
import hashlib
import json
from typing import Any
import uuid

from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from src.backend_v2.operations.repository import (
    OperationFence,
    OperationFenced,
    OperationRepository,
    utcnow,
)
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    assets,
    books,
    idempotency_records,
    operation_asset_inputs,
    operation_credential_snapshots,
    operation_events,
    operations,
    studio_chat_sessions,
    studio_documents,
    studio_message_assets,
    studio_messages,
)
from src.backend_v2.studio.model import (
    from_storage,
    new_document,
    normalize_document,
    to_storage,
)
from src.backend_v2.studio.pure import run_state_tasks


ACTIVE_OPERATION_STATUSES = ("pending", "running")


class StudioNotFound(LookupError):
    pass


class StudioConflict(RuntimeError):
    pass


class StudioBusy(StudioConflict):
    pass


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


def _default_chat_runtime_state() -> dict[str, Any]:
    return {
        "event_counts": {
            "message_received": 0,
            "message_sent": 0,
        },
        "matched_lorebook_ids": [],
    }


class StudioRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.operations = OperationRepository(engine)

    def index(self, *, book_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            self._assert_book(connection, book_id)
            rows = list(
                connection.execute(
                    select(studio_documents)
                    .where(studio_documents.c.book_id == book_id)
                    .order_by(
                        studio_documents.c.updated_at.desc(),
                        studio_documents.c.title,
                    )
                ).mappings()
            )
        return {
            "bookId": book_id,
            "documents": [
                {
                    "documentId": str(row["id"]),
                    "title": str(row["title"]),
                    "kind": str(row["origin_type"]),
                    "revision": int(row["revision"]),
                    "avatarAssetId": row["avatar_asset_id"],
                    "hasAvatar": row["avatar_asset_id"] is not None,
                    "sourceCharacter": row["source_character"],
                    "tags": _load(row["tags_json"], []),
                    "isFavorite": bool(row["is_favorite"]),
                    "updatedAt": str(row["updated_at"]),
                }
                for row in rows
            ],
        }

    def create_document(
        self,
        *,
        book_id: str,
        title: str,
        document: Mapping[str, Any] | None = None,
        kind: str = "manual",
        avatar_asset_id: str | None = None,
        idempotency_key: str | None = None,
        idempotency_request: Mapping[str, Any] | None = None,
        idempotency_scope: str | None = None,
    ) -> dict[str, Any]:
        canonical = normalize_document(
            book_id=book_id,
            title=title,
            document=document or new_document(book_id, title=title),
        )
        canonical["origin"] = {
            **_mapping(canonical.get("origin")),
            "type": kind,
        }
        canonical_title, storage_values = to_storage(canonical)
        document_id = str(uuid.uuid4())

        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            self._assert_book(connection, book_id)
            if avatar_asset_id is not None:
                self._assert_assets(connection, [avatar_asset_id])
            connection.execute(
                insert(studio_documents).values(
                    id=document_id,
                    book_id=book_id,
                    title=canonical_title,
                    avatar_asset_id=avatar_asset_id,
                    revision=1,
                    **storage_values,
                    schema_version=2,
                    created_at=now,
                    updated_at=now,
                )
            )
            return (
                self._document_from_connection(connection, document_id),
                document_id,
            )

        result, _replayed = self._execute_short_command(
            scope=(
                idempotency_scope
                or f"POST:createStudioDocument:{book_id}"
            ),
            key=idempotency_key,
            request=idempotency_request
            or {
                "bookId": book_id,
                "title": canonical_title,
                "kind": kind,
                "document": canonical,
                "avatarAssetId": avatar_asset_id,
            },
            http_status=201,
            resource_type="studio_document",
            mutation=mutate,
        )
        return result

    def get_document(self, document_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            return self._document_from_connection(
                connection,
                document_id,
            )

    def update_document(
        self,
        *,
        document_id: str,
        base_revision: int,
        title: str | None,
        document: Mapping[str, Any],
        idempotency_key: str | None = None,
        idempotency_request: Mapping[str, Any] | None = None,
        idempotency_scope: str | None = None,
    ) -> dict[str, Any]:
        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            row = self._assert_document(connection, document_id)
            canonical = normalize_document(
                book_id=str(row["book_id"]),
                title=title,
                document=document,
            )
            canonical_title, storage_values = to_storage(canonical)
            changed = connection.execute(
                update(studio_documents)
                .where(
                    studio_documents.c.id == document_id,
                    studio_documents.c.revision == base_revision,
                )
                .values(
                    title=canonical_title,
                    **storage_values,
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise StudioConflict("studio document revision changed")
            return (
                self._document_from_connection(connection, document_id),
                document_id,
            )

        updated, replayed = self._execute_short_command(
            scope=(
                idempotency_scope
                or f"PUT:updateStudioDocument:{document_id}"
            ),
            key=idempotency_key,
            request=idempotency_request
            or {
                "documentId": document_id,
                "baseRevision": base_revision,
                "title": title,
                "document": dict(document),
            },
            http_status=200,
            resource_type="studio_document",
            mutation=mutate,
        )
        if not replayed:
            self._align_active_draft(updated)
            return self.get_document(document_id)
        return updated

    def validate_document(
        self,
        *,
        document_id: str,
        base_revision: int,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        from src.backend_v2.studio.pure import build_diagnostics_report

        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            row = self._assert_document(connection, document_id)
            if int(row["revision"]) != base_revision:
                raise StudioConflict("studio document revision changed")
            document = from_storage(row)
            report = build_diagnostics_report(document)
            status = _mapping(document.get("status"))
            status["last_diagnostics"] = report
            status["last_validated_at"] = (
                now.replace(microsecond=0).isoformat() + "Z"
            )
            document["status"] = status
            title, storage_values = to_storage(
                normalize_document(
                    book_id=str(row["book_id"]),
                    title=str(row["title"]),
                    document=document,
                )
            )
            changed = connection.execute(
                update(studio_documents)
                .where(
                    studio_documents.c.id == document_id,
                    studio_documents.c.revision == base_revision,
                )
                .values(
                    title=title,
                    **storage_values,
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise StudioConflict("studio document revision changed")
            return (
                {
                    "documentRevision": base_revision + 1,
                    "diagnostics": report,
                },
                document_id,
            )

        result, _replayed = self._execute_short_command(
            scope=f"POST:validateStudioDocument:{document_id}",
            key=idempotency_key,
            request={
                "documentId": document_id,
                "baseRevision": base_revision,
            },
            http_status=200,
            resource_type="studio_document",
            mutation=mutate,
        )
        return result

    def set_avatar(
        self,
        *,
        document_id: str,
        base_revision: int,
        asset_id: str | None,
        idempotency_key: str | None = None,
        idempotency_request: Mapping[str, Any] | None = None,
        idempotency_scope: str | None = None,
    ) -> dict[str, Any]:
        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            return (
                self._set_avatar_on_connection(
                    connection,
                    document_id=document_id,
                    base_revision=base_revision,
                    asset_id=asset_id,
                    now=now,
                ),
                document_id,
            )

        result, _replayed = self._execute_short_command(
            scope=(
                idempotency_scope
                or f"POST:setStudioAvatar:{document_id}"
            ),
            key=idempotency_key,
            request=idempotency_request
            or {
                "documentId": document_id,
                "baseRevision": base_revision,
                "assetId": asset_id,
            },
            http_status=200,
            resource_type="studio_document",
            mutation=mutate,
        )
        return result

    def delete_document(
        self,
        document_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        def mutate(
            connection: Connection,
            _now,
        ) -> tuple[dict[str, Any], str]:
            self._assert_document(connection, document_id)
            if self._active_operation(
                connection,
                document_id=document_id,
            ) or connection.execute(
                select(operations.c.id)
                .join(
                    studio_chat_sessions,
                    studio_chat_sessions.c.id
                    == operations.c.studio_session_id,
                )
                .where(
                    studio_chat_sessions.c.document_id == document_id,
                    operations.c.status.in_(ACTIVE_OPERATION_STATUSES),
                )
                .limit(1)
            ).scalar_one_or_none() is not None:
                raise StudioBusy("studio document has an active operation")
            connection.execute(
                delete(studio_documents).where(
                    studio_documents.c.id == document_id
                )
            )
            return {"deleted": True, "documentId": document_id}, document_id

        result, _replayed = self._execute_short_command(
            scope=f"DELETE:deleteStudioDocument:{document_id}",
            key=idempotency_key,
            request={"documentId": document_id},
            http_status=200,
            resource_type="studio_document",
            mutation=mutate,
        )
        return result

    def create_session(
        self,
        *,
        document_id: str,
        title: str,
        base_index_revision: int | None = None,
        greeting: str | None = None,
        greeting_source: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        idempotency_request: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid4())

        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            document = self._assert_document(connection, document_id)
            session_work: dict[str, Any] = {
                "variables": {},
                "_runtime": _default_chat_runtime_state(),
            }
            initial_runtime_log = run_state_tasks(
                session_work,
                from_storage(document).get("stateTasks", []),
                event="initialization",
            )
            initial_variables = _mapping(session_work.get("variables"))
            initial_runtime = _mapping(session_work.get("_runtime"))
            if (
                base_index_revision is not None
                and int(document["chat_index_revision"])
                != base_index_revision
            ):
                raise StudioConflict(
                    "studio chat session index revision changed"
                )
            active = connection.execute(
                select(studio_chat_sessions.c.id).where(
                    studio_chat_sessions.c.document_id == document_id,
                    studio_chat_sessions.c.archived_at.is_(None),
                )
            ).scalar_one_or_none()
            if active is not None:
                if self._active_operation(
                    connection,
                    session_id=str(active),
                ):
                    raise StudioBusy("active studio session is busy")
                connection.execute(
                    update(studio_chat_sessions)
                    .where(studio_chat_sessions.c.id == active)
                    .values(archived_at=now, updated_at=now)
                )
            connection.execute(
                insert(studio_chat_sessions).values(
                    id=session_id,
                    document_id=document_id,
                    title=title or f"{document['title']} 对话",
                    revision=1,
                    generation=1,
                    greeting_source_json=_json(dict(greeting_source or {})),
                    variables_json=_json(initial_variables),
                    summary_blocks_json="[]",
                    summary_generation=0,
                    runtime_state_json=_json(initial_runtime),
                    runtime_schema_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            if greeting:
                connection.execute(
                    insert(studio_messages).values(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        ordinal=1,
                        role="assistant",
                        content=greeting,
                        runtime_log=_json(initial_runtime_log),
                        variables_snapshot_json=_json(initial_variables),
                        generation_meta_json=_json(
                            {
                                "source": "greeting",
                                "runtimeState": initial_runtime,
                            }
                        ),
                        metadata_json="{}",
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(studio_chat_sessions)
                    .where(studio_chat_sessions.c.id == session_id)
                    .values(revision=2)
                )
            connection.execute(
                update(studio_documents)
                .where(studio_documents.c.id == document_id)
                .values(
                    chat_index_revision=(
                        studio_documents.c.chat_index_revision + 1
                    )
                )
            )
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one()
            return (
                self._session_dto(
                    connection,
                    session,
                    self._message_rows(connection, session_id),
                ),
                session_id,
            )

        result, _replayed = self._execute_short_command(
            scope=f"POST:createStudioSession:{document_id}",
            key=idempotency_key,
            request=idempotency_request
            or {
                "documentId": document_id,
                "baseIndexRevision": base_index_revision,
                "title": title,
                "greeting": greeting,
                "greetingSource": dict(greeting_source or {}),
            },
            http_status=201,
            resource_type="studio_session",
            mutation=mutate,
        )
        return result

    def ensure_active_session(
        self,
        *,
        document_id: str,
        title: str,
        greeting: str | None = None,
        greeting_source: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically return the active session or bootstrap it once."""
        session_id = str(uuid.uuid4())
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            document = self._assert_document(connection, document_id)
            active = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.document_id == document_id,
                    studio_chat_sessions.c.archived_at.is_(None),
                )
            ).mappings().one_or_none()
            if active is not None:
                return self._session_dto(
                    connection,
                    active,
                    self._message_rows(connection, str(active["id"])),
                )
            connection.execute(
                insert(studio_chat_sessions).values(
                    id=session_id,
                    document_id=document_id,
                    title=title or f"{document['title']} 对话",
                    revision=1,
                    generation=1,
                    greeting_source_json=_json(dict(greeting_source or {})),
                    variables_json="{}",
                    summary_blocks_json="[]",
                    summary_generation=0,
                    runtime_state_json=_json(
                        {
                            "event_counts": {
                                "message_received": 0,
                                "message_sent": 0,
                            },
                            "matched_lorebook_ids": [],
                        }
                    ),
                    runtime_schema_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            if greeting:
                connection.execute(
                    insert(studio_messages).values(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        ordinal=1,
                        role="assistant",
                        content=greeting,
                        runtime_log="",
                        variables_snapshot_json="{}",
                        generation_meta_json=_json({"source": "greeting"}),
                        metadata_json="{}",
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(studio_chat_sessions)
                    .where(studio_chat_sessions.c.id == session_id)
                    .values(revision=2)
                )
            connection.execute(
                update(studio_documents)
                .where(studio_documents.c.id == document_id)
                .values(
                    chat_index_revision=(
                        studio_documents.c.chat_index_revision + 1
                    )
                )
            )
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one()
            return self._session_dto(
                connection,
                session,
                self._message_rows(connection, session_id),
            )

    def import_session(
        self,
        *,
        document_id: str,
        base_index_revision: int | None = None,
        payload: Mapping[str, Any],
        idempotency_key: str | None = None,
        idempotency_request: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw_messages = payload.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("session messages must be an array")
        normalized_messages: list[dict[str, Any]] = []
        all_asset_ids: list[str] = []
        for raw in raw_messages:
            if not isinstance(raw, Mapping):
                raise ValueError("each session message must be an object")
            role = str(raw.get("role", ""))
            if role not in {"system", "user", "assistant"}:
                raise ValueError("session message role is invalid")
            asset_ids = raw.get("assetIds", [])
            if not isinstance(asset_ids, list) or not all(
                isinstance(value, str) for value in asset_ids
            ):
                raise ValueError("message assetIds must be a string array")
            all_asset_ids.extend(asset_ids)
            normalized_messages.append(
                {
                    "source_id": str(
                        raw.get(
                            "messageId",
                            raw.get("message_id", ""),
                        )
                        or ""
                    ),
                    "role": role,
                    "content": str(raw.get("content", "")),
                    "runtime_log": raw.get(
                        "runtimeLog",
                        raw.get("runtime_log", []),
                    ),
                    "variables_snapshot": raw.get(
                        "variablesSnapshot",
                        raw.get("variables_snapshot", {}),
                    ),
                    "generation_meta": raw.get(
                        "generationMeta",
                        raw.get("generation_meta", {}),
                    ),
                    "metadata": raw.get("metadata", {}),
                    "asset_ids": list(asset_ids),
                }
            )
        session_id = str(uuid.uuid4())

        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            document = self._assert_document(connection, document_id)
            if (
                base_index_revision is not None
                and int(document["chat_index_revision"])
                != base_index_revision
            ):
                raise StudioConflict(
                    "studio chat session index revision changed"
                )
            self._assert_assets(
                connection,
                list(dict.fromkeys(all_asset_ids)),
            )
            active = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.document_id == document_id,
                    studio_chat_sessions.c.archived_at.is_(None),
                )
            ).mappings().one_or_none()
            if active is not None:
                if self._active_operation(
                    connection,
                    session_id=str(active["id"]),
                ):
                    raise StudioBusy("active studio session is busy")
                connection.execute(
                    update(studio_chat_sessions)
                    .where(studio_chat_sessions.c.id == active["id"])
                    .values(archived_at=now, updated_at=now)
                )
            summary_blocks = payload.get(
                "summaryBlocks",
                payload.get("summary_blocks", []),
            )
            variables = payload.get("variables", {})
            runtime_state = payload.get(
                "runtimeState",
                payload.get("_runtime", {}),
            )
            connection.execute(
                insert(studio_chat_sessions).values(
                    id=session_id,
                    document_id=document_id,
                    title=str(payload.get("title") or "导入对话"),
                    revision=1,
                    generation=1,
                    greeting_source_json=_json(
                        _mapping(
                            payload.get(
                                "greetingSource",
                                payload.get("greeting_source", {}),
                            )
                        )
                    ),
                    variables_json=_json(_mapping(variables)),
                    summary_blocks_json=_json(
                        list(summary_blocks)
                        if isinstance(summary_blocks, list)
                        else []
                    ),
                    summary_generation=int(
                        payload.get(
                            "summaryGeneration",
                            payload.get("summary_generation", 0),
                        )
                        or 0
                    ),
                    runtime_state_json=_json(_mapping(runtime_state)),
                    runtime_schema_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            imported_message_ids: dict[str, str] = {}
            for ordinal, message in enumerate(
                normalized_messages,
                start=1,
            ):
                message_id = str(uuid.uuid4())
                if message["source_id"]:
                    imported_message_ids[message["source_id"]] = message_id
                connection.execute(
                    insert(studio_messages).values(
                        id=message_id,
                        session_id=session_id,
                        ordinal=ordinal,
                        role=message["role"],
                        content=message["content"],
                        runtime_log=_json(message["runtime_log"]),
                        variables_snapshot_json=_json(
                            message["variables_snapshot"]
                        ),
                        generation_meta_json=_json(
                            message["generation_meta"]
                        ),
                        metadata_json=_json(message["metadata"]),
                        created_at=now,
                        updated_at=now,
                    )
                )
                if message["asset_ids"]:
                    connection.execute(
                        insert(studio_message_assets),
                        [
                            {
                                "message_id": message_id,
                                "asset_id": asset_id,
                                "ordinal": asset_ordinal,
                            }
                            for asset_ordinal, asset_id in enumerate(
                                message["asset_ids"],
                                start=1,
                            )
                        ],
                    )
            source_summary_through_id = str(
                payload.get(
                    "summaryThroughMessageId",
                    payload.get("summary_through_message_id", ""),
                )
                or ""
            )
            if source_summary_through_id:
                summary_through_id = imported_message_ids.get(
                    source_summary_through_id
                )
                if summary_through_id is None:
                    raise ValueError(
                        "summaryThroughMessageId does not reference an imported message"
                    )
                connection.execute(
                    update(studio_chat_sessions)
                    .where(studio_chat_sessions.c.id == session_id)
                    .values(
                        summary_through_message_id=summary_through_id,
                    )
                )
            connection.execute(
                update(studio_documents)
                .where(studio_documents.c.id == document_id)
                .values(
                    chat_index_revision=(
                        studio_documents.c.chat_index_revision + 1
                    )
                )
            )
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one()
            return (
                self._session_dto(
                    connection,
                    session,
                    self._message_rows(connection, session_id),
                ),
                session_id,
            )

        result, _replayed = self._execute_short_command(
            scope=f"POST:importStudioSession:{document_id}",
            key=idempotency_key,
            request=idempotency_request
            or {
                "documentId": document_id,
                "baseIndexRevision": base_index_revision,
                "session": dict(payload),
            },
            http_status=201,
            resource_type="studio_session",
            mutation=mutate,
        )
        return result

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one_or_none()
            if session is None:
                raise StudioNotFound("studio chat session not found")
            messages = self._message_rows(connection, session_id)
            return self._session_dto(connection, session, messages)

    def chat_state(self, document_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            document = self._assert_document(connection, document_id)
            message_count = (
                select(func.count(studio_messages.c.id))
                .where(
                    studio_messages.c.session_id
                    == studio_chat_sessions.c.id
                )
                .correlate(studio_chat_sessions)
                .scalar_subquery()
            )
            last_message_excerpt = (
                select(func.substr(studio_messages.c.content, 1, 160))
                .where(
                    studio_messages.c.session_id
                    == studio_chat_sessions.c.id
                )
                .order_by(studio_messages.c.ordinal.desc())
                .limit(1)
                .correlate(studio_chat_sessions)
                .scalar_subquery()
            )
            sessions = list(
                connection.execute(
                    select(
                        studio_chat_sessions,
                        message_count.label("message_count"),
                        last_message_excerpt.label(
                            "last_message_excerpt"
                        ),
                    )
                    .where(studio_chat_sessions.c.document_id == document_id)
                    .order_by(studio_chat_sessions.c.updated_at.desc())
                ).mappings()
            )
            active = next(
                (
                    session
                    for session in sessions
                    if session["archived_at"] is None
                ),
                None,
            )
            messages = (
                self._message_rows(connection, str(active["id"]))
                if active is not None
                else []
            )
            return {
                "documentId": document_id,
                "indexRevision": int(
                    document["chat_index_revision"]
                ),
                "sessions": [
                    {
                        "sessionId": str(row["id"]),
                        "title": str(row["title"]),
                        "revision": int(row["revision"]),
                        "generation": int(row["generation"]),
                        "archived": row["archived_at"] is not None,
                        "updatedAt": str(row["updated_at"]),
                        "messageCount": int(row["message_count"]),
                        "lastMessageExcerpt": str(
                            row["last_message_excerpt"] or ""
                        ),
                    }
                    for row in sessions
                ],
                "activeSession": (
                    self._session_dto(connection, active, messages)
                    if active is not None
                    else None
                ),
            }

    def create_generate_operation(
        self,
        *,
        document_id: str,
        base_revision: int,
        section: str,
        config: Mapping[str, Any],
        analysis_context: Mapping[str, Any] | None = None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if section not in {
            "identity",
            "greetings",
            "lorebook",
            "regex",
            "state-tasks",
            "translate",
            "full",
            "review",
        }:
            raise ValueError("unsupported Studio generation section")
        now = utcnow()
        scope = f"studio-generate:{document_id}"
        request_hash = hashlib.sha256(
            _json(
                {
                    "documentId": document_id,
                    "baseRevision": base_revision,
                    "section": section,
                    "analysisContext": (
                        dict(analysis_context)
                        if analysis_context is not None
                        else None
                    ),
                }
            ).encode("utf-8")
        ).hexdigest()
        with immediate_transaction(self.engine) as connection:
            replay = self._idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay
            row = self._assert_document(connection, document_id)
            if int(row["revision"]) != base_revision:
                raise StudioConflict("studio document revision changed")
            document = from_storage(row)
            request_payload = {
                "section": section,
                "document": document,
                "config": dict(config),
                "analysisContext": (
                    dict(analysis_context)
                    if analysis_context is not None
                    else None
                ),
            }
            response = self._insert_operation(
                connection,
                kind="studio_generate",
                document_id=document_id,
                session_id=None,
                base_revision=base_revision,
                base_generation=0,
                request_payload=request_payload,
                idempotency_scope=None,
                idempotency_key=None,
                now=now,
            )
            self._store_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                resource_id=str(response["operationId"]),
                now=now,
            )
            return response

    def send_message(
        self,
        *,
        session_id: str,
        base_revision: int,
        content: str,
        asset_ids: Sequence[str],
        config: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        content = content.strip()
        if not content and not asset_ids:
            raise ValueError("message content is required")
        now = utcnow()
        canonical_request = {
            "baseRevision": base_revision,
            "content": content,
            "assetIds": list(asset_ids),
        }
        request_hash = hashlib.sha256(
            _json(canonical_request).encode("utf-8")
        ).hexdigest()
        scope = f"studio-chat-send:{session_id}"
        with immediate_transaction(self.engine) as connection:
            replay = self._idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay
            session = self._assert_session_writable(
                connection,
                session_id,
                base_revision=base_revision,
            )
            document_row = self._assert_document(
                connection,
                str(session["document_id"]),
            )
            self._assert_assets(connection, asset_ids)
            ordinal = int(
                connection.execute(
                    select(func.coalesce(func.max(studio_messages.c.ordinal), 0))
                    .where(studio_messages.c.session_id == session_id)
                ).scalar_one()
            ) + 1
            message_id = str(uuid.uuid4())
            connection.execute(
                insert(studio_messages).values(
                    id=message_id,
                    session_id=session_id,
                    ordinal=ordinal,
                    role="user",
                    content=content,
                    runtime_log="",
                    variables_snapshot_json=session["variables_json"],
                    generation_meta_json=_json(
                        {
                            "runtimeState": _load(
                                session["runtime_state_json"],
                                _default_chat_runtime_state(),
                            )
                        }
                    ),
                    metadata_json="{}",
                    created_at=now,
                    updated_at=now,
                )
            )
            if asset_ids:
                connection.execute(
                    insert(studio_message_assets),
                    [
                        {
                            "message_id": message_id,
                            "asset_id": asset_id,
                            "ordinal": index,
                        }
                        for index, asset_id in enumerate(asset_ids, start=1)
                    ],
                )
            committed_revision = base_revision + 1
            committed_generation = int(session["generation"]) + 1
            connection.execute(
                update(studio_chat_sessions)
                .where(
                    studio_chat_sessions.c.id == session_id,
                    studio_chat_sessions.c.revision == base_revision,
                )
                .values(
                    revision=committed_revision,
                    generation=committed_generation,
                    updated_at=now,
                )
            )
            messages = self._message_rows(connection, session_id)
            message_dtos = self._messages_dto(connection, messages)
            document = from_storage(document_row)
            operation = self._insert_operation(
                connection,
                kind="studio_chat",
                document_id=None,
                session_id=session_id,
                base_revision=committed_revision,
                base_generation=committed_generation,
                request_payload={
                    "document": document,
                    "messages": message_dtos,
                    "variables": _load(session["variables_json"], {}),
                    "runtimeState": _load(
                        session["runtime_state_json"],
                        {},
                    ),
                    "summaryBlocks": _load(
                        session["summary_blocks_json"],
                        [],
                    ),
                    "summaryThroughMessageId": session[
                        "summary_through_message_id"
                    ],
                    "config": dict(config),
                },
                idempotency_scope=None,
                idempotency_key=None,
                now=now,
            )
            self._bind_operation_message_assets(
                connection,
                operation_id=str(operation["operationId"]),
                messages=message_dtos,
            )
            response = {
                **operation,
                "sessionRevision": committed_revision,
                "sessionGeneration": committed_generation,
                "userMessageId": message_id,
            }
            self._store_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                resource_id=str(operation["operationId"]),
                now=now,
            )
            return response

    def create_summary_operation(
        self,
        *,
        session_id: str,
        base_revision: int,
        config: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        now = utcnow()
        scope = f"studio-summary:{session_id}"
        request_hash = hashlib.sha256(
            _json(
                {
                    "sessionId": session_id,
                    "baseRevision": base_revision,
                }
            ).encode("utf-8")
        ).hexdigest()
        with immediate_transaction(self.engine) as connection:
            replay = self._idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay
            session = self._assert_session_writable(
                connection,
                session_id,
                base_revision=base_revision,
            )
            messages = self._message_rows(connection, session_id)
            message_dtos = self._messages_dto(connection, messages)
            through_id = session["summary_through_message_id"]
            if through_id is not None:
                through_index = next(
                    (
                        index
                        for index, message in enumerate(message_dtos)
                        if message["messageId"] == through_id
                    ),
                    None,
                )
                if through_index is not None:
                    message_dtos = message_dtos[through_index + 1 :]
            if not message_dtos:
                raise StudioConflict("当前会话没有待总结的新消息")
            existing_summaries = _load(
                session["summary_blocks_json"],
                [],
            )
            if existing_summaries:
                message_dtos.insert(
                    0,
                    {
                        "messageId": "summary-context",
                        "ordinal": 0,
                        "role": "system",
                        "content": (
                            "已有会话摘要："
                            + _json(existing_summaries)
                        ),
                        "attachments": [],
                        "runtimeLog": [],
                        "variablesSnapshot": {},
                        "generationMeta": {},
                    },
                )
            response = self._insert_operation(
                connection,
                kind="studio_summary",
                document_id=None,
                session_id=session_id,
                base_revision=base_revision,
                base_generation=int(session["generation"]),
                request_payload={
                    "messages": message_dtos,
                    "config": dict(config),
                },
                idempotency_scope=None,
                idempotency_key=None,
                now=now,
            )
            self._store_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                resource_id=str(response["operationId"]),
                now=now,
            )
            return response

    def abort(
        self,
        *,
        session_id: str,
        operation_id: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one_or_none()
            if session is None:
                raise StudioNotFound("studio chat session not found")
            changed = connection.execute(
                update(operations)
                .where(
                    operations.c.id == operation_id,
                    operations.c.studio_session_id == session_id,
                    operations.c.status.in_(ACTIVE_OPERATION_STATUSES),
                )
                .values(
                    status="cancelled",
                    executor_epoch_id=None,
                    attempt_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    finished_at=now,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise StudioConflict("operation is no longer active")
            connection.execute(
                insert(operation_events).values(
                    operation_id=operation_id,
                    type="operation_cancelled",
                    payload_json=_json({"status": "cancelled"}),
                    created_at=now,
                )
            )
            revision = int(session["revision"]) + 1
            generation = int(session["generation"]) + 1
            connection.execute(
                update(studio_chat_sessions)
                .where(studio_chat_sessions.c.id == session_id)
                .values(
                    revision=revision,
                    generation=generation,
                    updated_at=now,
                )
            )
            return (
                {
                    "operationId": operation_id,
                    "status": "cancelled",
                    "sessionRevision": revision,
                    "sessionGeneration": generation,
                },
                operation_id,
            )

        result, _replayed = self._execute_short_command(
            scope=f"POST:abortStudioSession:{session_id}",
            key=idempotency_key,
            request={
                "sessionId": session_id,
                "operationId": operation_id,
            },
            http_status=200,
            resource_type="operation",
            mutation=mutate,
        )
        return result

    def activate_session(
        self,
        session_id: str,
        *,
        base_index_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            target = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one_or_none()
            if target is None:
                raise StudioNotFound("studio chat session not found")
            document = self._assert_document(
                connection,
                str(target["document_id"]),
            )
            if (
                base_index_revision is not None
                and int(document["chat_index_revision"])
                != base_index_revision
            ):
                raise StudioConflict(
                    "studio chat session index revision changed"
                )
            if self._active_operation(connection, session_id=session_id):
                raise StudioBusy("studio session has an active operation")
            if target["archived_at"] is None:
                return (
                    self._session_dto(
                        connection,
                        target,
                        self._message_rows(connection, session_id),
                    ),
                    session_id,
                )
            current = connection.execute(
                select(studio_chat_sessions.c.id).where(
                    studio_chat_sessions.c.document_id
                    == target["document_id"],
                    studio_chat_sessions.c.archived_at.is_(None),
                    studio_chat_sessions.c.id != session_id,
                )
            ).scalar_one_or_none()
            if current is not None:
                if self._active_operation(
                    connection,
                    session_id=str(current),
                ):
                    raise StudioBusy("active studio session is busy")
                connection.execute(
                    update(studio_chat_sessions)
                    .where(studio_chat_sessions.c.id == current)
                    .values(archived_at=now, updated_at=now)
                )
            connection.execute(
                update(studio_chat_sessions)
                .where(studio_chat_sessions.c.id == session_id)
                .values(archived_at=None, updated_at=now)
            )
            connection.execute(
                update(studio_documents)
                .where(
                    studio_documents.c.id == target["document_id"]
                )
                .values(
                    chat_index_revision=(
                        studio_documents.c.chat_index_revision + 1
                    )
                )
            )
            refreshed = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one()
            return (
                self._session_dto(
                    connection,
                    refreshed,
                    self._message_rows(connection, session_id),
                ),
                session_id,
            )

        result, _replayed = self._execute_short_command(
            scope=f"POST:activateStudioSession:{session_id}",
            key=idempotency_key,
            request={
                "sessionId": session_id,
                "baseIndexRevision": base_index_revision,
            },
            http_status=200,
            resource_type="studio_session",
            mutation=mutate,
        )
        return result

    def delete_session(
        self,
        *,
        session_id: str,
        base_revision: int,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        def mutate(
            connection: Connection,
            _now,
        ) -> tuple[dict[str, Any], str]:
            row = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one_or_none()
            if row is None:
                raise StudioNotFound("studio chat session not found")
            if int(row["revision"]) != base_revision:
                raise StudioConflict("studio session revision changed")
            if row["archived_at"] is None:
                raise StudioConflict(
                    "only archived studio sessions may be deleted"
                )
            if self._active_operation(connection, session_id=session_id):
                raise StudioBusy("studio session has an active operation")
            connection.execute(
                delete(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            )
            return {"deleted": True, "sessionId": session_id}, session_id

        result, _replayed = self._execute_short_command(
            scope=f"DELETE:deleteStudioSession:{session_id}",
            key=idempotency_key,
            request={
                "sessionId": session_id,
                "baseRevision": base_revision,
            },
            http_status=200,
            resource_type="studio_session",
            mutation=mutate,
        )
        return result

    def delete_message_chain(
        self,
        *,
        message_id: str,
        base_revision: int,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        def mutate(
            connection: Connection,
            now,
        ) -> tuple[dict[str, Any], str]:
            message = connection.execute(
                select(studio_messages).where(
                    studio_messages.c.id == message_id
                )
            ).mappings().one_or_none()
            if message is None:
                raise StudioNotFound("studio message not found")
            session = self._assert_session_writable(
                connection,
                str(message["session_id"]),
                base_revision=base_revision,
            )
            summary_ordinal = self._summary_ordinal(
                connection,
                session,
            )
            clear_summary = (
                summary_ordinal is not None
                and int(message["ordinal"]) <= summary_ordinal
            )
            connection.execute(
                delete(studio_messages).where(
                    studio_messages.c.session_id == message["session_id"],
                    studio_messages.c.ordinal >= message["ordinal"],
                )
            )
            restored_variables, restored_runtime = (
                self._chat_state_from_messages(
                    connection,
                    str(message["session_id"]),
                )
            )
            revision = base_revision + 1
            generation = int(session["generation"]) + 1
            session_values: dict[str, Any] = {
                "revision": revision,
                "generation": generation,
                "variables_json": _json(restored_variables),
                "runtime_state_json": _json(restored_runtime),
                "updated_at": now,
            }
            if clear_summary:
                session_values.update(
                    summary_blocks_json="[]",
                    summary_through_message_id=None,
                )
            connection.execute(
                update(studio_chat_sessions)
                .where(
                    studio_chat_sessions.c.id == message["session_id"],
                    studio_chat_sessions.c.revision == base_revision,
                )
                .values(**session_values)
            )
            return (
                {
                    "sessionId": str(message["session_id"]),
                    "sessionRevision": revision,
                    "sessionGeneration": generation,
                },
                message_id,
            )

        result, _replayed = self._execute_short_command(
            scope=f"DELETE:deleteStudioMessage:{message_id}",
            key=idempotency_key,
            request={
                "messageId": message_id,
                "baseRevision": base_revision,
            },
            http_status=200,
            resource_type="studio_message",
            mutation=mutate,
        )
        return result

    def edit_or_regenerate_message(
        self,
        *,
        message_id: str,
        base_revision: int,
        content: str | None,
        config: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        now = utcnow()
        scope = f"studio-message-regenerate:{message_id}"
        request_hash = hashlib.sha256(
            _json(
                {
                    "messageId": message_id,
                    "baseRevision": base_revision,
                    "content": content,
                }
            ).encode("utf-8")
        ).hexdigest()
        with immediate_transaction(self.engine) as connection:
            replay = self._idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay
            message = connection.execute(
                select(studio_messages).where(
                    studio_messages.c.id == message_id
                )
            ).mappings().one_or_none()
            if message is None:
                raise StudioNotFound("studio message not found")
            session_id = str(message["session_id"])
            if content is not None and str(message["role"]) != "user":
                raise StudioConflict(
                    "only user messages may be edited"
                )
            session = self._assert_session_writable(
                connection,
                session_id,
                base_revision=base_revision,
            )
            target = message
            if str(message["role"]) == "assistant":
                target = connection.execute(
                    select(studio_messages)
                    .where(
                        studio_messages.c.session_id == session_id,
                        studio_messages.c.role == "user",
                        studio_messages.c.ordinal < message["ordinal"],
                    )
                    .order_by(studio_messages.c.ordinal.desc())
                    .limit(1)
                ).mappings().one_or_none()
                if target is None:
                    raise StudioConflict(
                        "assistant message has no preceding user message"
                    )
            summary_ordinal = self._summary_ordinal(
                connection,
                session,
            )
            clear_summary = (
                summary_ordinal is not None
                and int(target["ordinal"]) <= summary_ordinal
            )
            if content is not None:
                if str(target["role"]) != "user":
                    raise StudioConflict("only user messages may be edited")
                normalized = content.strip()
                if not normalized:
                    raise ValueError("message content is required")
                connection.execute(
                    update(studio_messages)
                    .where(studio_messages.c.id == target["id"])
                    .values(content=normalized, updated_at=now)
                )
            connection.execute(
                delete(studio_messages).where(
                    studio_messages.c.session_id == session_id,
                    studio_messages.c.ordinal > target["ordinal"],
                )
            )
            restored_variables, restored_runtime = (
                self._chat_state_from_messages(connection, session_id)
            )
            committed_revision = base_revision + 1
            committed_generation = int(session["generation"]) + 1
            session_values = {
                "revision": committed_revision,
                "generation": committed_generation,
                "variables_json": _json(restored_variables),
                "runtime_state_json": _json(restored_runtime),
                "updated_at": now,
            }
            if clear_summary:
                session_values.update(
                    summary_blocks_json="[]",
                    summary_through_message_id=None,
                )
            connection.execute(
                update(studio_chat_sessions)
                .where(
                    studio_chat_sessions.c.id == session_id,
                    studio_chat_sessions.c.revision == base_revision,
                )
                .values(**session_values)
            )
            document_row = self._assert_document(
                connection,
                str(session["document_id"]),
            )
            message_dtos = self._messages_dto(
                connection,
                self._message_rows(connection, session_id),
            )
            operation = self._insert_operation(
                connection,
                kind="studio_chat",
                document_id=None,
                session_id=session_id,
                base_revision=committed_revision,
                base_generation=committed_generation,
                request_payload={
                    "document": from_storage(document_row),
                    "messages": message_dtos,
                    "variables": restored_variables,
                    "runtimeState": restored_runtime,
                    "summaryBlocks": (
                        []
                        if clear_summary
                        else _load(
                            session["summary_blocks_json"],
                            [],
                        )
                    ),
                    "summaryThroughMessageId": (
                        None
                        if clear_summary
                        else session["summary_through_message_id"]
                    ),
                    "config": dict(config),
                },
                idempotency_scope=None,
                idempotency_key=None,
                now=now,
            )
            self._bind_operation_message_assets(
                connection,
                operation_id=str(operation["operationId"]),
                messages=message_dtos,
            )
            response = {
                **operation,
                "sessionRevision": committed_revision,
                "sessionGeneration": committed_generation,
            }
            self._store_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                resource_id=str(operation["operationId"]),
                now=now,
            )
        return response

    def publish_generate(
        self,
        fence: OperationFence,
        *,
        generated_document: Mapping[str, Any],
        review: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def publisher(
            connection: Connection,
            operation: Mapping[str, Any],
        ) -> None:
            document_id = str(operation["studio_document_id"])
            row = self._assert_document(connection, document_id)
            if int(row["revision"]) != int(operation["base_revision"]):
                raise OperationFenced(
                    "studio document changed before generation publish"
                )
            if review is not None:
                title = str(row["title"])
                storage_values = {
                    "last_review_json": _json(dict(review)),
                }
            else:
                canonical = normalize_document(
                    book_id=str(row["book_id"]),
                    title=None,
                    document=generated_document,
                )
                title, storage_values = to_storage(canonical)
                generated_values = {
                    "title": title,
                    **storage_values,
                }
                content_fields = (
                    "title",
                    "identity_json",
                    "core_messages_json",
                    "lorebook_json",
                    "regex_scripts_json",
                    "state_tasks_json",
                )
                if all(
                    row[field] == generated_values[field]
                    for field in content_fields
                ):
                    raise ValueError(
                        "Studio generation returned no document changes"
                    )
                storage_values["last_diagnostics_json"] = None
                storage_values["last_validated_at"] = None
            revision = int(row["revision"]) + 1
            changed = connection.execute(
                update(studio_documents)
                .where(
                    studio_documents.c.id == document_id,
                    studio_documents.c.revision == row["revision"],
                )
                .values(
                    title=title,
                    **storage_values,
                    revision=revision,
                    updated_at=utcnow(),
                )
            )
            if changed.rowcount != 1:
                raise OperationFenced(
                    "studio document changed before generation publish"
                )
            result.update(
                {
                    "documentId": document_id,
                    "documentRevision": revision,
                }
            )

        self.operations.complete(
            fence,
            result=result,
            publisher=publisher,
        )
        return {**result, "__already_published__": True}

    def publish_chat(
        self,
        fence: OperationFence,
        *,
        content: str,
        runtime_log: Sequence[Mapping[str, Any]],
        variables: Mapping[str, Any],
        runtime_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def publisher(
            connection: Connection,
            operation: Mapping[str, Any],
        ) -> None:
            session_id = str(operation["studio_session_id"])
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id,
                    studio_chat_sessions.c.revision
                    == operation["base_revision"],
                    studio_chat_sessions.c.generation
                    == operation["base_generation"],
                )
            ).mappings().one_or_none()
            if session is None:
                raise OperationFenced(
                    "studio session changed before assistant publish"
                )
            ordinal = int(
                connection.execute(
                    select(func.coalesce(func.max(studio_messages.c.ordinal), 0))
                    .where(studio_messages.c.session_id == session_id)
                ).scalar_one()
            ) + 1
            message_id = str(uuid.uuid4())
            connection.execute(
                insert(studio_messages).values(
                    id=message_id,
                    session_id=session_id,
                    ordinal=ordinal,
                    role="assistant",
                    content=content,
                    runtime_log=_json(list(runtime_log)),
                    variables_snapshot_json=_json(dict(variables)),
                    generation_meta_json=_json(
                        {
                            "generation": int(session["generation"]),
                            "runtimeState": dict(runtime_state),
                        }
                    ),
                    metadata_json="{}",
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
            revision = int(session["revision"]) + 1
            connection.execute(
                update(studio_chat_sessions)
                .where(
                    studio_chat_sessions.c.id == session_id,
                    studio_chat_sessions.c.revision == session["revision"],
                    studio_chat_sessions.c.generation == session["generation"],
                )
                .values(
                    revision=revision,
                    variables_json=_json(dict(variables)),
                    runtime_state_json=_json(dict(runtime_state)),
                    updated_at=utcnow(),
                )
            )
            result.update(
                {
                    "sessionId": session_id,
                    "sessionRevision": revision,
                    "sessionGeneration": int(session["generation"]),
                    "assistantMessageId": message_id,
                }
            )

        self.operations.complete(
            fence,
            result=result,
            publisher=publisher,
        )
        return {**result, "__already_published__": True}

    def publish_summary(
        self,
        fence: OperationFence,
        *,
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def publisher(
            connection: Connection,
            operation: Mapping[str, Any],
        ) -> None:
            session_id = str(operation["studio_session_id"])
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id,
                    studio_chat_sessions.c.revision
                    == operation["base_revision"],
                    studio_chat_sessions.c.generation
                    == operation["base_generation"],
                )
            ).mappings().one_or_none()
            if session is None:
                raise OperationFenced(
                    "studio session changed before summary publish"
                )
            last_message_id = connection.execute(
                select(studio_messages.c.id)
                .where(studio_messages.c.session_id == session_id)
                .order_by(studio_messages.c.ordinal.desc())
                .limit(1)
            ).scalar_one_or_none()
            revision = int(session["revision"]) + 1
            summary_generation = int(session["summary_generation"]) + 1
            connection.execute(
                update(studio_chat_sessions)
                .where(
                    studio_chat_sessions.c.id == session_id,
                    studio_chat_sessions.c.revision == session["revision"],
                    studio_chat_sessions.c.generation == session["generation"],
                )
                .values(
                    revision=revision,
                    summary_blocks_json=_json([dict(summary)]),
                    summary_through_message_id=last_message_id,
                    summary_generation=summary_generation,
                    updated_at=utcnow(),
                )
            )
            result.update(
                {
                    "sessionId": session_id,
                    "sessionRevision": revision,
                    "summaryGeneration": summary_generation,
                }
            )

        self.operations.complete(
            fence,
            result=result,
            publisher=publisher,
        )
        return {**result, "__already_published__": True}

    def _insert_operation(
        self,
        connection: Connection,
        *,
        kind: str,
        document_id: str | None,
        session_id: str | None,
        base_revision: int,
        base_generation: int,
        request_payload: Mapping[str, Any],
        idempotency_scope: str | None,
        idempotency_key: str | None,
        now,
    ) -> dict[str, Any]:
        request_hash = hashlib.sha256(
            _json(request_payload).encode("utf-8")
        ).hexdigest()
        if idempotency_scope and idempotency_key:
            replay = self._idempotency_replay(
                connection,
                scope=idempotency_scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay
        operation_id = str(uuid.uuid4())
        try:
            connection.execute(
                insert(operations).values(
                    id=operation_id,
                    kind=kind,
                    executor_role="api",
                    status="pending",
                    studio_document_id=document_id,
                    studio_session_id=session_id,
                    base_revision=base_revision,
                    base_generation=base_generation,
                    request_json=_json(dict(request_payload)),
                    request_schema_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
        except IntegrityError as exc:
            raise StudioBusy("studio target already has an active operation") from exc
        for role, credential_id in _credential_references(request_payload).items():
            connection.execute(
                insert(operation_credential_snapshots).values(
                    operation_id=operation_id,
                    credential_version_id=credential_id,
                    role=role,
                )
            )
        response = {
            "operationId": operation_id,
            "kind": kind,
            "status": "pending",
            "executorRole": "api",
            "baseRevision": base_revision,
            "baseGeneration": base_generation,
        }
        if idempotency_scope and idempotency_key:
            self._store_idempotency(
                connection,
                scope=idempotency_scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                resource_id=operation_id,
                now=now,
            )
        return response

    def replay_short_command(
        self,
        *,
        scope: str,
        key: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        now = utcnow()
        request_hash = hashlib.sha256(
            _json(dict(request)).encode("utf-8")
        ).hexdigest()
        with immediate_transaction(self.engine) as connection:
            return self._idempotency_replay(
                connection,
                scope=scope,
                key=key,
                request_hash=request_hash,
                now=now,
            )

    def execute_bound_short_command(
        self,
        connection: Connection,
        *,
        scope: str,
        key: str,
        request: Mapping[str, Any],
        http_status: int,
        resource_type: str,
        mutation: Callable[[], tuple[dict[str, Any], str | None]],
    ) -> tuple[dict[str, Any], bool]:
        now = utcnow()
        request_hash = hashlib.sha256(
            _json(dict(request)).encode("utf-8")
        ).hexdigest()
        replay = self._idempotency_replay(
            connection,
            scope=scope,
            key=key,
            request_hash=request_hash,
            now=now,
        )
        if replay is not None:
            return replay, True
        result, resource_id = mutation()
        self._store_idempotency(
            connection,
            scope=scope,
            key=key,
            request_hash=request_hash,
            response=result,
            resource_id=resource_id,
            now=now,
            http_status=http_status,
            resource_type=resource_type,
        )
        return result, False

    def _execute_short_command(
        self,
        *,
        scope: str,
        key: str | None,
        request: Mapping[str, Any],
        http_status: int,
        resource_type: str,
        mutation: Callable[
            [Connection, Any],
            tuple[dict[str, Any], str | None],
        ],
    ) -> tuple[dict[str, Any], bool]:
        now = utcnow()
        request_hash = hashlib.sha256(
            _json(dict(request)).encode("utf-8")
        ).hexdigest()
        with immediate_transaction(self.engine) as connection:
            if key is not None:
                replay = self._idempotency_replay(
                    connection,
                    scope=scope,
                    key=key,
                    request_hash=request_hash,
                    now=now,
                )
                if replay is not None:
                    return replay, True
            result, resource_id = mutation(connection, now)
            if key is not None:
                self._store_idempotency(
                    connection,
                    scope=scope,
                    key=key,
                    request_hash=request_hash,
                    response=result,
                    resource_id=resource_id,
                    now=now,
                    http_status=http_status,
                    resource_type=resource_type,
                )
            return result, False

    @staticmethod
    def _document_from_connection(
        connection: Connection,
        document_id: str,
    ) -> dict[str, Any]:
        row = connection.execute(
            select(studio_documents).where(
                studio_documents.c.id == document_id
            )
        ).mappings().one_or_none()
        if row is None:
            raise StudioNotFound("studio document not found")
        return from_storage(row)

    def _set_avatar_on_connection(
        self,
        connection: Connection,
        *,
        document_id: str,
        base_revision: int,
        asset_id: str | None,
        now,
    ) -> dict[str, Any]:
        row = self._assert_document(connection, document_id)
        if int(row["revision"]) != base_revision:
            raise StudioConflict("studio document revision changed")
        if asset_id is not None:
            self._assert_assets(connection, [asset_id])
        changed = connection.execute(
            update(studio_documents)
            .where(
                studio_documents.c.id == document_id,
                studio_documents.c.revision == base_revision,
            )
            .values(
                avatar_asset_id=asset_id,
                revision=base_revision + 1,
                updated_at=now,
            )
        )
        if changed.rowcount != 1:
            raise StudioConflict("studio document revision changed")
        return self._document_from_connection(connection, document_id)

    @staticmethod
    def _assert_book(connection: Connection, book_id: str) -> None:
        if connection.execute(
            select(books.c.id).where(
                books.c.id == book_id,
                books.c.kind == "library",
            )
        ).scalar_one_or_none() is None:
            raise StudioNotFound("book not found")

    @staticmethod
    def _assert_document(
        connection: Connection,
        document_id: str,
    ) -> Mapping[str, Any]:
        row = connection.execute(
            select(studio_documents).where(
                studio_documents.c.id == document_id
            )
        ).mappings().one_or_none()
        if row is None:
            raise StudioNotFound("studio document not found")
        return row

    def _assert_session_writable(
        self,
        connection: Connection,
        session_id: str,
        *,
        base_revision: int,
    ) -> Mapping[str, Any]:
        row = connection.execute(
            select(studio_chat_sessions).where(
                studio_chat_sessions.c.id == session_id
            )
        ).mappings().one_or_none()
        if row is None:
            raise StudioNotFound("studio chat session not found")
        if row["archived_at"] is not None:
            raise StudioConflict("studio chat session is archived")
        if int(row["revision"]) != base_revision:
            raise StudioConflict("studio session revision changed")
        if self._active_operation(connection, session_id=session_id):
            raise StudioBusy("studio session has an active operation")
        return row

    def _align_active_draft(
        self,
        document: Mapping[str, Any],
    ) -> None:
        document_id = str(document["id"])
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            session = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.document_id == document_id,
                    studio_chat_sessions.c.archived_at.is_(None),
                )
            ).mappings().one_or_none()
            if session is None or self._active_operation(
                connection,
                session_id=str(session["id"]),
            ):
                return
            messages = self._message_rows(connection, str(session["id"]))
            if _load(session["summary_blocks_json"], []):
                return
            if any(str(message["role"]) == "user" for message in messages):
                return
            if len(messages) > 1 or (
                messages and str(messages[0]["role"]) != "assistant"
            ):
                return
            source = _mapping(
                _load(session["greeting_source_json"], {})
            )
            core = _mapping(document.get("coreMessages"))
            if source.get("type") == "alternate_greeting":
                alternatives = core.get("alternate_greetings", [])
                index = int(source.get("index", 0) or 0)
                desired = (
                    str(alternatives[index])
                    if isinstance(alternatives, list)
                    and 0 <= index < len(alternatives)
                    else ""
                )
            else:
                desired = str(core.get("first_message", ""))
                source = {"type": "first_message", "index": 0}
            desired = desired.strip()
            changed = False
            if not messages and desired:
                connection.execute(
                    insert(studio_messages).values(
                        id=str(uuid.uuid4()),
                        session_id=session["id"],
                        ordinal=1,
                        role="assistant",
                        content=desired,
                        runtime_log="",
                        variables_snapshot_json=session["variables_json"],
                        generation_meta_json=_json({"source": "greeting"}),
                        metadata_json="{}",
                        created_at=now,
                        updated_at=now,
                    )
                )
                changed = True
            elif messages and str(messages[0]["content"]) != desired:
                if desired:
                    connection.execute(
                        update(studio_messages)
                        .where(studio_messages.c.id == messages[0]["id"])
                        .values(content=desired, updated_at=now)
                    )
                else:
                    connection.execute(
                        delete(studio_messages).where(
                            studio_messages.c.id == messages[0]["id"]
                        )
                    )
                changed = True
            if changed:
                connection.execute(
                    update(studio_chat_sessions)
                    .where(studio_chat_sessions.c.id == session["id"])
                    .values(
                        revision=int(session["revision"]) + 1,
                        generation=int(session["generation"]) + 1,
                        greeting_source_json=_json(source),
                        updated_at=now,
                    )
                )

    @staticmethod
    def _active_operation(
        connection: Connection,
        *,
        document_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        query = select(operations.c.id).where(
            operations.c.status.in_(ACTIVE_OPERATION_STATUSES)
        )
        if document_id is not None:
            query = query.where(
                operations.c.studio_document_id == document_id
            )
        if session_id is not None:
            query = query.where(
                operations.c.studio_session_id == session_id
            )
        return connection.execute(query.limit(1)).scalar_one_or_none() is not None

    @staticmethod
    def _summary_ordinal(
        connection: Connection,
        session: Mapping[str, Any],
    ) -> int | None:
        through_id = session["summary_through_message_id"]
        if through_id is None:
            return None
        ordinal = connection.execute(
            select(studio_messages.c.ordinal).where(
                studio_messages.c.id == through_id,
                studio_messages.c.session_id == session["id"],
            )
        ).scalar_one_or_none()
        return int(ordinal) if ordinal is not None else None

    @staticmethod
    def _assert_assets(
        connection: Connection,
        asset_ids: Sequence[str],
    ) -> None:
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("assetIds must be unique")
        if not asset_ids:
            return
        found = set(
            str(value)
            for value in connection.execute(
                select(assets.c.id).where(assets.c.id.in_(tuple(asset_ids)))
            ).scalars()
        )
        if found != set(asset_ids):
            raise StudioNotFound("one or more message assets do not exist")

    @staticmethod
    def _message_rows(
        connection: Connection,
        session_id: str,
    ) -> list[Mapping[str, Any]]:
        return list(
            connection.execute(
                select(studio_messages)
                .where(studio_messages.c.session_id == session_id)
                .order_by(studio_messages.c.ordinal)
            ).mappings()
        )

    @staticmethod
    def _chat_state_from_messages(
        connection: Connection,
        session_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore the session state represented by the retained linear chain."""
        variables: dict[str, Any] = {}
        runtime = _default_chat_runtime_state()
        pending_user = False

        for row in StudioRepository._message_rows(connection, session_id):
            snapshot = _load(row["variables_snapshot_json"], {})
            if isinstance(snapshot, Mapping):
                variables = dict(snapshot)

            generation_meta = _load(row["generation_meta_json"], {})
            meta = (
                dict(generation_meta)
                if isinstance(generation_meta, Mapping)
                else {}
            )
            runtime_snapshot = meta.get("runtimeState")
            if isinstance(runtime_snapshot, Mapping):
                runtime = json.loads(_json(dict(runtime_snapshot)))

            role = str(row["role"])
            if role == "user":
                pending_user = True
                continue
            if role != "assistant":
                continue
            if isinstance(runtime_snapshot, Mapping):
                pending_user = False
                continue
            if meta.get("source") == "greeting" or not pending_user:
                pending_user = False
                continue

            counts = runtime.setdefault("event_counts", {})
            if not isinstance(counts, dict):
                counts = {}
                runtime["event_counts"] = counts
            counts["message_received"] = int(
                counts.get("message_received", 0)
            ) + 1
            counts["message_sent"] = int(
                counts.get("message_sent", 0)
            ) + 1

            matched = runtime.setdefault("matched_lorebook_ids", [])
            if not isinstance(matched, list):
                matched = []
                runtime["matched_lorebook_ids"] = matched
            logs = _load(row["runtime_log"], [])
            for item in logs if isinstance(logs, list) else []:
                if not isinstance(item, Mapping):
                    continue
                entry_id = item.get("id")
                if (
                    item.get("type") == "lorebook"
                    and isinstance(entry_id, str)
                    and entry_id
                    and entry_id not in matched
                ):
                    matched.append(entry_id)
            pending_user = False

        return variables, runtime

    @staticmethod
    def _message_dto(
        row: Mapping[str, Any],
        *,
        attachments: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return {
            "messageId": str(row["id"]),
            "ordinal": int(row["ordinal"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "attachments": [dict(item) for item in attachments],
            "runtimeLog": _load(row["runtime_log"], []),
            "variablesSnapshot": _load(
                row["variables_snapshot_json"],
                {},
            ),
            "generationMeta": _load(row["generation_meta_json"], {}),
        }

    def _messages_dto(
        self,
        connection: Connection,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        message_ids = [str(row["id"]) for row in messages]
        attachments: dict[str, list[dict[str, Any]]] = {
            message_id: [] for message_id in message_ids
        }
        if message_ids:
            rows = connection.execute(
                select(
                    studio_message_assets.c.message_id,
                    studio_message_assets.c.asset_id,
                    studio_message_assets.c.ordinal,
                    assets.c.mime_type,
                    assets.c.byte_size,
                    assets.c.width,
                    assets.c.height,
                    assets.c.integrity_status,
                )
                .join(
                    assets,
                    assets.c.id == studio_message_assets.c.asset_id,
                )
                .where(
                    studio_message_assets.c.message_id.in_(
                        tuple(message_ids)
                    )
                )
                .order_by(
                    studio_message_assets.c.message_id,
                    studio_message_assets.c.ordinal,
                )
            ).mappings()
            for asset in rows:
                asset_id = str(asset["asset_id"])
                attachments[str(asset["message_id"])].append(
                    {
                        "assetId": asset_id,
                        "assetUrl": f"/api/v2/assets/{asset_id}",
                        "mimeType": str(asset["mime_type"]),
                        "byteSize": int(asset["byte_size"]),
                        "width": asset["width"],
                        "height": asset["height"],
                        "available": asset["integrity_status"] == "ok",
                    }
                )
        return [
            self._message_dto(
                message,
                attachments=attachments[str(message["id"])],
            )
            for message in messages
        ]

    @staticmethod
    def _bind_operation_message_assets(
        connection: Connection,
        *,
        operation_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> None:
        asset_ids: list[str] = []
        for message in messages:
            attachments = message.get("attachments", [])
            if not isinstance(attachments, list):
                continue
            for attachment in attachments:
                if not isinstance(attachment, Mapping):
                    continue
                asset_id = attachment.get("assetId")
                if isinstance(asset_id, str) and asset_id not in asset_ids:
                    asset_ids.append(asset_id)
        if asset_ids:
            connection.execute(
                insert(operation_asset_inputs),
                [
                    {
                        "operation_id": operation_id,
                        "role": f"attachment:{index}",
                        "asset_id": asset_id,
                    }
                    for index, asset_id in enumerate(asset_ids, start=1)
                ],
            )

    def _session_dto(
        self,
        connection: Connection,
        row: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        index_revision = connection.execute(
            select(studio_documents.c.chat_index_revision).where(
                studio_documents.c.id == row["document_id"]
            )
        ).scalar_one()
        return {
            "sessionId": str(row["id"]),
            "documentId": str(row["document_id"]),
            "indexRevision": int(index_revision),
            "title": str(row["title"]),
            "revision": int(row["revision"]),
            "generation": int(row["generation"]),
            "greetingSource": _load(row["greeting_source_json"], {}),
            "variables": _load(row["variables_json"], {}),
            "summaryBlocks": _load(row["summary_blocks_json"], []),
            "summaryThroughMessageId": row["summary_through_message_id"],
            "summaryGeneration": int(row["summary_generation"]),
            "runtimeState": _load(row["runtime_state_json"], {}),
            "archived": row["archived_at"] is not None,
            "messages": self._messages_dto(connection, messages),
        }

    @staticmethod
    def _idempotency_replay(
        connection: Connection,
        *,
        scope: str,
        key: str,
        request_hash: str,
        now,
    ) -> dict[str, Any] | None:
        if not key or len(key) > 200:
            raise ValueError("Idempotency-Key is required")
        row = connection.execute(
            select(
                idempotency_records.c.request_hash,
                idempotency_records.c.response_json,
                idempotency_records.c.expires_at,
            ).where(
                idempotency_records.c.scope == scope,
                idempotency_records.c.key == key,
            )
        ).mappings().one_or_none()
        if row is None:
            return None
        if row["expires_at"] <= now:
            connection.execute(
                delete(idempotency_records).where(
                    idempotency_records.c.scope == scope,
                    idempotency_records.c.key == key,
                )
            )
            return None
        if str(row["request_hash"]) != request_hash:
            raise StudioConflict(
                "Idempotency-Key was reused for a different request"
            )
        return dict(_load(row["response_json"], {}))

    @staticmethod
    def _store_idempotency(
        connection: Connection,
        *,
        scope: str,
        key: str,
        request_hash: str,
        response: Mapping[str, Any],
        resource_id: str | None,
        now,
        http_status: int = 202,
        resource_type: str = "operation",
    ) -> None:
        connection.execute(
            insert(idempotency_records).values(
                scope=scope,
                key=key,
                request_hash=request_hash,
                http_status=http_status,
                response_json=_json(dict(response)),
                resource_type=resource_type,
                resource_id=resource_id,
                created_at=now,
                expires_at=now + timedelta(days=7),
            )
        )


def _credential_references(value: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(current: object, path: tuple[str, ...]) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                if key == "credentialVersionId" and isinstance(child, str):
                    result[".".join(path) or "default"] = child
                else:
                    visit(child, (*path, str(key)))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                visit(child, (*path, str(index)))

    visit(value, ())
    return result


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
