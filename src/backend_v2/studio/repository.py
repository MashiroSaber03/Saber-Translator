"""Transactional Character Studio documents, sessions, and saved operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    operation_credential_snapshots,
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
                    "kind": str(row["kind"]),
                    "revision": int(row["revision"]),
                    "generation": int(row["generation"]),
                    "avatarAssetId": row["avatar_asset_id"],
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
    ) -> dict[str, Any]:
        canonical = normalize_document(
            book_id=book_id,
            title=title,
            document=document or new_document(book_id, title=title),
        )
        canonical_title, payload = to_storage(canonical)
        now = utcnow()
        document_id = str(uuid.uuid4())
        with immediate_transaction(self.engine) as connection:
            self._assert_book(connection, book_id)
            connection.execute(
                insert(studio_documents).values(
                    id=document_id,
                    book_id=book_id,
                    kind=kind,
                    title=canonical_title,
                    revision=1,
                    generation=1,
                    payload_json=_json(payload),
                    schema_version=2,
                    created_at=now,
                    updated_at=now,
                )
            )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(studio_documents).where(
                    studio_documents.c.id == document_id
                )
            ).mappings().one_or_none()
        if row is None:
            raise StudioNotFound("studio document not found")
        return from_storage(row, _load(row["payload_json"], {}))

    def update_document(
        self,
        *,
        document_id: str,
        base_revision: int,
        title: str | None,
        document: Mapping[str, Any],
    ) -> dict[str, Any]:
        current = self.get_document(document_id)
        canonical = normalize_document(
            book_id=str(current["bookId"]),
            title=title,
            document=document,
        )
        canonical_title, payload = to_storage(canonical)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(studio_documents)
                .where(
                    studio_documents.c.id == document_id,
                    studio_documents.c.revision == base_revision,
                )
                .values(
                    title=canonical_title,
                    payload_json=_json(payload),
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise StudioConflict("studio document revision changed")
        return self.get_document(document_id)

    def delete_document(self, document_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            self._assert_document(connection, document_id)
            if self._active_operation(
                connection,
                document_id=document_id,
            ):
                raise StudioBusy("studio document has an active operation")
            connection.execute(
                delete(studio_documents).where(
                    studio_documents.c.id == document_id
                )
            )

    def create_session(
        self,
        *,
        document_id: str,
        title: str,
        greeting: str | None = None,
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            document = self._assert_document(connection, document_id)
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
        return self.get_session(session_id)

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
        return self._session_dto(session, messages)

    def chat_state(self, document_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            self._assert_document(connection, document_id)
            sessions = list(
                connection.execute(
                    select(studio_chat_sessions)
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
            "sessions": [
                {
                    "sessionId": str(row["id"]),
                    "title": str(row["title"]),
                    "revision": int(row["revision"]),
                    "generation": int(row["generation"]),
                    "archived": row["archived_at"] is not None,
                    "updatedAt": str(row["updated_at"]),
                }
                for row in sessions
            ],
            "activeSession": (
                self._session_dto(active, messages)
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
        with immediate_transaction(self.engine) as connection:
            row = self._assert_document(connection, document_id)
            if int(row["revision"]) != base_revision:
                raise StudioConflict("studio document revision changed")
            document = from_storage(row, _load(row["payload_json"], {}))
            request_payload = {
                "section": section,
                "document": document,
                "config": dict(config),
            }
            return self._insert_operation(
                connection,
                kind="studio_generate",
                document_id=document_id,
                session_id=None,
                base_revision=base_revision,
                base_generation=int(row["generation"]),
                request_payload=request_payload,
                idempotency_scope=f"studio-generate:{document_id}",
                idempotency_key=idempotency_key,
                now=now,
            )

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
        if not content:
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
                    generation_meta_json="{}",
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
            document = from_storage(
                document_row,
                _load(document_row["payload_json"], {}),
            )
            operation = self._insert_operation(
                connection,
                kind="studio_chat",
                document_id=None,
                session_id=session_id,
                base_revision=committed_revision,
                base_generation=committed_generation,
                request_payload={
                    "document": document,
                    "messages": [self._message_dto(row) for row in messages],
                    "variables": _load(session["variables_json"], {}),
                    "runtimeState": _load(
                        session["runtime_state_json"],
                        {},
                    ),
                    "summaryBlocks": _load(
                        session["summary_blocks_json"],
                        [],
                    ),
                    "config": dict(config),
                },
                idempotency_scope=None,
                idempotency_key=None,
                now=now,
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
        with immediate_transaction(self.engine) as connection:
            session = self._assert_session_writable(
                connection,
                session_id,
                base_revision=base_revision,
            )
            messages = self._message_rows(connection, session_id)
            return self._insert_operation(
                connection,
                kind="studio_summary",
                document_id=None,
                session_id=session_id,
                base_revision=base_revision,
                base_generation=int(session["generation"]),
                request_payload={
                    "messages": [self._message_dto(row) for row in messages],
                    "config": dict(config),
                },
                idempotency_scope=f"studio-summary:{session_id}",
                idempotency_key=idempotency_key,
                now=now,
            )

    def abort(self, *, session_id: str, operation_id: str) -> dict[str, Any]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
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
        return {
            "operationId": operation_id,
            "status": "cancelled",
            "sessionRevision": revision,
            "sessionGeneration": generation,
        }

    def activate_session(self, session_id: str) -> dict[str, Any]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            target = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one_or_none()
            if target is None:
                raise StudioNotFound("studio chat session not found")
            if self._active_operation(connection, session_id=session_id):
                raise StudioBusy("studio session has an active operation")
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
        return self.get_session(session_id)

    def delete_session(
        self,
        *,
        session_id: str,
        base_revision: int,
    ) -> None:
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            ).mappings().one_or_none()
            if row is None:
                raise StudioNotFound("studio chat session not found")
            if int(row["revision"]) != base_revision:
                raise StudioConflict("studio session revision changed")
            if self._active_operation(connection, session_id=session_id):
                raise StudioBusy("studio session has an active operation")
            connection.execute(
                delete(studio_chat_sessions).where(
                    studio_chat_sessions.c.id == session_id
                )
            )

    def delete_message_chain(
        self,
        *,
        message_id: str,
        base_revision: int,
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
            session = self._assert_session_writable(
                connection,
                str(message["session_id"]),
                base_revision=base_revision,
            )
            connection.execute(
                delete(studio_messages).where(
                    studio_messages.c.session_id == message["session_id"],
                    studio_messages.c.ordinal >= message["ordinal"],
                )
            )
            revision = base_revision + 1
            generation = int(session["generation"]) + 1
            connection.execute(
                update(studio_chat_sessions)
                .where(
                    studio_chat_sessions.c.id == message["session_id"],
                    studio_chat_sessions.c.revision == base_revision,
                )
                .values(
                    revision=revision,
                    generation=generation,
                    summary_blocks_json="[]",
                    summary_through_message_id=None,
                    updated_at=now,
                )
            )
        return {
            "sessionId": str(message["session_id"]),
            "sessionRevision": revision,
            "sessionGeneration": generation,
        }

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
        with immediate_transaction(self.engine) as connection:
            message = connection.execute(
                select(studio_messages).where(
                    studio_messages.c.id == message_id
                )
            ).mappings().one_or_none()
            if message is None:
                raise StudioNotFound("studio message not found")
            session_id = str(message["session_id"])
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
                    summary_blocks_json="[]",
                    summary_through_message_id=None,
                    updated_at=now,
                )
            )
            document_row = self._assert_document(
                connection,
                str(session["document_id"]),
            )
            operation = self._insert_operation(
                connection,
                kind="studio_chat",
                document_id=None,
                session_id=session_id,
                base_revision=committed_revision,
                base_generation=committed_generation,
                request_payload={
                    "document": from_storage(
                        document_row,
                        _load(document_row["payload_json"], {}),
                    ),
                    "messages": [
                        self._message_dto(row)
                        for row in self._message_rows(connection, session_id)
                    ],
                    "variables": _load(session["variables_json"], {}),
                    "runtimeState": _load(
                        session["runtime_state_json"],
                        {},
                    ),
                    "summaryBlocks": [],
                    "config": dict(config),
                },
                idempotency_scope=None,
                idempotency_key=None,
                now=now,
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
        return {
            **operation,
            "sessionRevision": committed_revision,
            "sessionGeneration": committed_generation,
        }

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
            if (
                int(row["revision"]) != int(operation["base_revision"])
                or int(row["generation"]) != int(operation["base_generation"])
            ):
                raise OperationFenced(
                    "studio document changed before generation publish"
                )
            if review is not None:
                payload = _load(row["payload_json"], {})
                payload["lastDiagnostics"] = dict(review)
                title = str(row["title"])
            else:
                canonical = normalize_document(
                    book_id=str(row["book_id"]),
                    title=None,
                    document=generated_document,
                )
                title, payload = to_storage(canonical)
            revision = int(row["revision"]) + 1
            connection.execute(
                update(studio_documents)
                .where(
                    studio_documents.c.id == document_id,
                    studio_documents.c.revision == row["revision"],
                    studio_documents.c.generation == row["generation"],
                )
                .values(
                    title=title,
                    payload_json=_json(payload),
                    revision=revision,
                    updated_at=utcnow(),
                )
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
                        {"generation": int(session["generation"])}
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
    def _message_dto(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "messageId": str(row["id"]),
            "ordinal": int(row["ordinal"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "runtimeLog": _load(row["runtime_log"], []),
            "variablesSnapshot": _load(
                row["variables_snapshot_json"],
                {},
            ),
            "generationMeta": _load(row["generation_meta_json"], {}),
        }

    def _session_dto(
        self,
        row: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "sessionId": str(row["id"]),
            "documentId": str(row["document_id"]),
            "title": str(row["title"]),
            "revision": int(row["revision"]),
            "generation": int(row["generation"]),
            "variables": _load(row["variables_json"], {}),
            "summaryBlocks": _load(row["summary_blocks_json"], []),
            "summaryThroughMessageId": row["summary_through_message_id"],
            "summaryGeneration": int(row["summary_generation"]),
            "runtimeState": _load(row["runtime_state_json"], {}),
            "archived": row["archived_at"] is not None,
            "messages": [self._message_dto(message) for message in messages],
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
            ).where(
                idempotency_records.c.scope == scope,
                idempotency_records.c.key == key,
                idempotency_records.c.expires_at > now,
            )
        ).mappings().one_or_none()
        if row is None:
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
        resource_id: str,
        now,
    ) -> None:
        connection.execute(
            insert(idempotency_records).values(
                scope=scope,
                key=key,
                request_hash=request_hash,
                http_status=202,
                response_json=_json(dict(response)),
                resource_type="operation",
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
