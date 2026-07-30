from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping
import uuid

import pytest
from PIL import Image
from sqlalchemy import inspect, insert, select

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.content.repository import ContentLocked, ContentRepository
from src.backend_v2.operations.repository import (
    OperationFenced,
    OperationRepository,
)
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    metadata,
    studio_documents,
    timeline_characters,
    timeline_versions,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.studio.repository import (
    StudioConflict,
    StudioRepository,
)
from src.backend_v2.studio.io import StudioIOService
from src.backend_v2.studio.media import read_card_png
from src.backend_v2.studio.service import StudioOperationService
from src.backend_v2.studio.service import _apply_generated_section


class FakeStudioAlgorithms:
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
        on_chunk=None,
    ) -> Mapping[str, Any]:
        if on_chunk:
            on_chunk('{"identity":', '{"identity":')
            on_chunk("{}", '{"identity":{}}')
        if section == "identity":
            return {
                "identity": {
                    "name": "生成角色",
                    "description": "后端生成",
                    "personality": "坚定",
                    "scenario": "测试",
                }
            }
        if section == "full":
            return {
                "identity": {
                    "name": "完整生成角色",
                    "description": "完整身份",
                    "personality": "果断",
                    "scenario": "完整场景",
                },
                "coreMessages": {
                    "first_message": "新的问候",
                    "message_example": "",
                    "alternate_greetings": [],
                    "system_prompt": "",
                    "post_history_instructions": "",
                    "creator_notes": "",
                    "character_version": "2.0.0",
                },
                "lorebook": {"name": "新世界书", "entries": []},
                "regexScripts": [{"id": "new-regex"}],
                "stateTasks": [{"id": "new-task"}],
            }
        return {"summary": "review"}

    def chat(
        self,
        *,
        messages,
        system: str,
        config: Mapping[str, Any],
        on_chunk=None,
    ) -> str:
        if on_chunk:
            on_chunk("持久化", "持久化")
            on_chunk("回复", "持久化回复")
        return "持久化回复"

    def summarize(self, messages, *, config, on_chunk=None):
        return {"summary": f"{len(messages)} messages"}


@pytest.fixture()
def studio_platform(tmp_path: Path):
    engine = create_sqlite_engine(tmp_path / "studio.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    book = ContentRepository(engine).create_book(title="Studio Book")
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "api", "api", 777)
    )
    try:
        yield {
            "data_root": tmp_path,
            "engine": engine,
            "book": book,
            "epoch_id": epoch_id,
        }
    finally:
        engine.dispose()


def test_document_is_canonical_and_revision_cas_is_enforced(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    created = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    assert created["identity"]["name"] == "Saber"
    assert "grounding" not in created
    assert "chatPreset" not in created
    with studio_platform["engine"].connect() as connection:
        identity_json = connection.execute(
            select(studio_documents.c.identity_json).where(
                studio_documents.c.id == created["id"]
            )
        ).scalar_one()
    assert '"name":"Saber"' not in identity_json
    assert "grounding" not in identity_json
    physical_columns = {
        column["name"]
        for column in inspect(studio_platform["engine"]).get_columns(
            "studio_documents"
        )
    }
    assert {
        "origin_type",
        "identity_json",
        "core_messages_json",
        "lorebook_json",
        "regex_scripts_json",
        "state_tasks_json",
    } <= physical_columns
    assert {"payload_json", "generation", "kind"}.isdisjoint(
        physical_columns
    )

    changed = dict(created)
    changed["identity"] = {
        **changed["identity"],
        "description": "updated",
    }
    updated = repository.update_document(
        document_id=str(created["id"]),
        base_revision=1,
        title="Saber",
        document=changed,
    )
    assert updated["revision"] == 2
    summary = repository.index(
        book_id=str(studio_platform["book"]["id"])
    )["documents"][0]
    assert summary == {
        "avatarAssetId": None,
        "documentId": str(created["id"]),
        "hasAvatar": False,
        "isFavorite": False,
        "kind": "manual",
        "revision": 2,
        "sourceCharacter": None,
        "tags": [],
        "title": "Saber",
        "updatedAt": summary["updatedAt"],
    }
    with pytest.raises(StudioConflict, match="revision"):
        repository.update_document(
            document_id=str(created["id"]),
            base_revision=1,
            title="Saber",
            document=changed,
        )


def test_book_delete_rejects_active_studio_work_then_preserves_terminal_history(
    studio_platform,
) -> None:
    content = ContentRepository(studio_platform["engine"])
    studio = StudioRepository(studio_platform["engine"])
    document = studio.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Protected",
    )
    accepted = studio.create_generate_operation(
        document_id=str(document["id"]),
        base_revision=1,
        section="identity",
        config={},
        idempotency_key="protected-generate",
    )

    with pytest.raises(ContentLocked):
        content.delete_book(str(studio_platform["book"]["id"]))

    operations = OperationRepository(studio_platform["engine"])
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_generate",),
    )
    assert claimed is not None
    operations.fail(
        claimed[0],
        code="TEST_TERMINAL",
        message="terminal",
    )
    content.delete_book(str(studio_platform["book"]["id"]))

    stored = operations.get(str(accepted["operationId"]))
    assert stored["status"] == "failed"
    assert stored["studioDocumentId"] is None
    assert stored["request"]["document"]["identity"]["name"] == "Protected"
    with studio_platform["engine"].connect() as connection:
        assert connection.execute(
            select(studio_documents.c.id).where(
                studio_documents.c.id == document["id"]
            )
        ).scalar_one_or_none() is None


def test_chat_operation_persists_reply_after_request_lifecycle(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    session = repository.create_session(
        document_id=str(document["id"]),
        title="预览",
    )
    accepted = repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(session["revision"]),
        content="你好",
        asset_ids=[],
        config={},
        idempotency_key="chat-1",
    )
    operations = OperationRepository(studio_platform["engine"])
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    fence, operation = claimed
    assert operation["operationId"] == accepted["operationId"]
    result = StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=FakeStudioAlgorithms(),
    ).handle(fence, operation)
    assert result["__already_published__"]
    restored = repository.get_session(str(session["sessionId"]))
    assert [message["role"] for message in restored["messages"]] == [
        "user",
        "assistant",
    ]
    assert restored["messages"][-1]["content"] == "持久化回复"


def test_abort_advances_generation_and_fences_late_reply(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    session = repository.create_session(
        document_id=str(document["id"]),
        title="预览",
    )
    accepted = repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(session["revision"]),
        content="不会得到回复",
        asset_ids=[],
        config={},
        idempotency_key="chat-abort",
    )
    claimed = OperationRepository(
        studio_platform["engine"]
    ).claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    fence, operation = claimed
    aborted = repository.abort(
        session_id=str(session["sessionId"]),
        operation_id=str(accepted["operationId"]),
    )
    assert aborted["status"] == "cancelled"
    with pytest.raises(OperationFenced):
        StudioOperationService(
            engine=studio_platform["engine"],
            repository=repository,
            algorithms=FakeStudioAlgorithms(),
        ).handle(fence, operation)
    restored = repository.get_session(str(session["sessionId"]))
    assert [message["role"] for message in restored["messages"]] == ["user"]


def test_generate_and_edit_idempotency_replay_precedes_revision_checks(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    accepted = repository.create_generate_operation(
        document_id=str(document["id"]),
        base_revision=1,
        section="identity",
        config={},
        idempotency_key="generate-once",
    )
    claimed = OperationRepository(studio_platform["engine"]).claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_generate",),
    )
    assert claimed is not None
    StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=FakeStudioAlgorithms(),
    ).handle(*claimed)

    replayed = repository.create_generate_operation(
        document_id=str(document["id"]),
        base_revision=1,
        section="identity",
        config={},
        idempotency_key="generate-once",
    )
    assert replayed["operationId"] == accepted["operationId"]
    with pytest.raises(StudioConflict, match="different request"):
        repository.create_generate_operation(
            document_id=str(document["id"]),
            base_revision=1,
            section="review",
            config={},
            idempotency_key="generate-once",
        )

    events = OperationRepository(
        studio_platform["engine"]
    ).events_after(str(accepted["operationId"]))
    assert [event["type"] for event in events] == [
        "operation_started",
        "chunk",
        "chunk",
        "operation_completed",
    ]

    session = repository.create_session(
        document_id=str(document["id"]),
        title="预览",
    )
    sent = repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(session["revision"]),
        content="原始内容",
        asset_ids=[],
        config={},
        idempotency_key="chat-for-edit",
    )
    claimed_chat = OperationRepository(
        studio_platform["engine"]
    ).claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed_chat is not None
    StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=FakeStudioAlgorithms(),
    ).handle(*claimed_chat)
    current = repository.get_session(str(session["sessionId"]))
    with pytest.raises(StudioConflict, match="only user"):
        repository.edit_or_regenerate_message(
            message_id=str(current["messages"][-1]["messageId"]),
            base_revision=int(current["revision"]),
            content="不能编辑助手消息",
            config={},
            idempotency_key="edit-assistant-rejected",
        )
    edited = repository.edit_or_regenerate_message(
        message_id=str(sent["userMessageId"]),
        base_revision=int(current["revision"]),
        content="修改内容",
        config={},
        idempotency_key="edit-once",
    )
    replayed_edit = repository.edit_or_regenerate_message(
        message_id=str(sent["userMessageId"]),
        base_revision=int(current["revision"]),
        content="修改内容",
        config={},
        idempotency_key="edit-once",
    )
    assert replayed_edit == edited


def test_message_delete_truncates_chain_without_creating_operation(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    session = repository.create_session(
        document_id=str(document["id"]),
        title="预览",
        greeting="开场",
    )
    result = repository.delete_message_chain(
        message_id=str(session["messages"][0]["messageId"]),
        base_revision=int(session["revision"]),
    )
    restored = repository.get_session(str(session["sessionId"]))
    assert result["sessionRevision"] == session["revision"] + 1
    assert restored["messages"] == []


def test_png_and_session_portable_roundtrip_uses_asset_ids(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    io_service = StudioIOService(
        data_root=studio_platform["data_root"],
        engine=studio_platform["engine"],
        repository=repository,
    )
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    image_buffer = BytesIO()
    Image.new("RGB", (24, 32), (10, 20, 30)).save(
        image_buffer,
        format="PNG",
    )
    image_bytes = image_buffer.getvalue()
    document = io_service.set_avatar(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        upload=BytesIO(image_bytes),
        idempotency_key="avatar-roundtrip",
    )
    card_png = io_service.export_png(document)
    assert read_card_png(card_png)["spec"] == "chara_card_v3"
    imported_document = io_service.import_document(
        book_id=str(studio_platform["book"]["id"]),
        upload=BytesIO(card_png),
        filename="saber.png",
        idempotency_key="document-import-roundtrip",
    )
    assert imported_document["title"] == "Saber"
    assert imported_document["avatarAssetId"]

    repository.create_session(
        document_id=str(document["id"]),
        title="旧会话",
    )
    imported_session = io_service.import_session(
        document_id=str(document["id"]),
        idempotency_key="session-import-roundtrip",
        payload={
            "title": "便携会话",
            "messages": [
                {
                    "messageId": "portable-message-1",
                    "role": "user",
                    "content": "看图",
                    "attachments": [
                        {
                            "filename": "image.png",
                            "mime_type": "image/png",
                            "blob_base64": base64.b64encode(
                                image_bytes
                            ).decode("ascii"),
                        }
                    ],
                }
            ],
            "summaryBlocks": [{"summary": "便携摘要"}],
            "summaryThroughMessageId": "portable-message-1",
        },
    )
    attachment = imported_session["messages"][0]["attachments"][0]
    assert attachment["assetId"]
    assert attachment["assetUrl"].startswith("/api/v2/assets/")
    assert (
        imported_session["summaryThroughMessageId"]
        == imported_session["messages"][0]["messageId"]
    )
    exported = io_service.export_session(
        str(imported_session["sessionId"])
    )
    assert exported["schema"] == "saber-studio-chat-v2"
    assert exported["messages"][0]["attachments"][0]["blob_base64"]


def test_summary_window_and_summary_invalidation_follow_message_ordinals(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    operations = OperationRepository(studio_platform["engine"])
    service = StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=FakeStudioAlgorithms(),
    )
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="摘要角色",
    )
    session = repository.create_session(
        document_id=str(document["id"]),
        title="摘要会话",
    )
    first = repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(session["revision"]),
        content="第一轮",
        asset_ids=[],
        config={},
        idempotency_key="summary-first-chat",
    )
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    service.handle(*claimed)
    current = repository.get_session(str(session["sessionId"]))
    repository.create_summary_operation(
        session_id=str(session["sessionId"]),
        base_revision=int(current["revision"]),
        config={},
        idempotency_key="summary-first",
    )
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_summary",),
    )
    assert claimed is not None
    service.handle(*claimed)
    summarized = repository.get_session(str(session["sessionId"]))
    first_summary = summarized["summaryBlocks"]
    assert (
        summarized["summaryThroughMessageId"]
        == summarized["messages"][1]["messageId"]
    )

    second = repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(summarized["revision"]),
        content="第二轮",
        asset_ids=[],
        config={},
        idempotency_key="summary-second-chat",
    )
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    service.handle(*claimed)
    current = repository.get_session(str(session["sessionId"]))
    summary_operation = repository.create_summary_operation(
        session_id=str(session["sessionId"]),
        base_revision=int(current["revision"]),
        config={},
        idempotency_key="summary-incremental",
    )
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_summary",),
    )
    assert claimed is not None
    _fence, operation = claimed
    summary_messages = operation["request"]["messages"]
    assert [message["content"] for message in summary_messages[1:]] == [
        "第二轮",
        "持久化回复",
    ]
    assert "已有会话摘要" in summary_messages[0]["content"]
    repository.abort(
        session_id=str(session["sessionId"]),
        operation_id=str(summary_operation["operationId"]),
    )

    current = repository.get_session(str(session["sessionId"]))
    edited = repository.edit_or_regenerate_message(
        message_id=str(second["userMessageId"]),
        base_revision=int(current["revision"]),
        content="第二轮修改",
        config={},
        idempotency_key="edit-after-summary",
    )
    preserved = repository.get_session(str(session["sessionId"]))
    assert preserved["summaryBlocks"] == first_summary
    repository.abort(
        session_id=str(session["sessionId"]),
        operation_id=str(edited["operationId"]),
    )
    current = repository.get_session(str(session["sessionId"]))
    repository.edit_or_regenerate_message(
        message_id=str(first["userMessageId"]),
        base_revision=int(current["revision"]),
        content="第一轮修改",
        config={},
        idempotency_key="edit-inside-summary",
    )
    invalidated = repository.get_session(str(session["sessionId"]))
    assert invalidated["summaryBlocks"] == []
    assert invalidated["summaryThroughMessageId"] is None


def test_draft_greeting_alignment_and_archived_only_deletion(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="问候角色",
    )
    first_version = dict(document)
    first_version["coreMessages"] = {
        **first_version["coreMessages"],
        "first_message": "初始问候",
    }
    document = repository.update_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        title=str(document["title"]),
        document=first_version,
    )
    first_session = repository.create_session(
        document_id=str(document["id"]),
        title="草稿",
        greeting="初始问候",
        greeting_source={"type": "first_message", "index": 0},
    )
    second_version = dict(document)
    second_version["coreMessages"] = {
        **second_version["coreMessages"],
        "first_message": "自动对齐后的问候",
    }
    repository.update_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        title=str(document["title"]),
        document=second_version,
    )
    aligned = repository.get_session(str(first_session["sessionId"]))
    assert aligned["messages"][0]["content"] == "自动对齐后的问候"

    active = repository.create_session(
        document_id=str(document["id"]),
        title="新会话",
    )
    with pytest.raises(StudioConflict, match="archived"):
        repository.delete_session(
            session_id=str(active["sessionId"]),
            base_revision=int(active["revision"]),
        )
    archived = repository.get_session(str(first_session["sessionId"]))
    result = repository.delete_session(
        session_id=str(archived["sessionId"]),
        base_revision=int(archived["revision"]),
    )
    assert result["deleted"]
    with pytest.raises(LookupError):
        repository.get_session(str(archived["sessionId"]))


def test_diagnostics_validate_state_tasks_and_replay_without_extra_revision(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="诊断角色",
    )
    changed = dict(document)
    changed["stateTasks"] = [
        {
            "name": "",
            "triggerTiming": "unknown",
            "interval": -1,
            "commands": "<<taskjs>>",
        }
    ]
    document = repository.update_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        title=str(document["title"]),
        document=changed,
    )
    first = repository.validate_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        idempotency_key="validate-once",
    )
    replay = repository.validate_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        idempotency_key="validate-once",
    )
    assert replay == first
    assert len(first["diagnostics"]["errors"]) == 4
    restored = repository.get_document(str(document["id"]))
    assert restored["revision"] == document["revision"] + 1
    assert restored["status"]["last_diagnostics"] == first["diagnostics"]


def test_full_generation_respects_all_frozen_section_names() -> None:
    document = {
        "status": {
            "frozen_sections": [
                "greetings",
                "regex",
                "state-tasks",
            ]
        },
        "identity": {"name": "旧角色"},
        "meta": {"title": "旧角色"},
        "coreMessages": {"first_message": "保留问候"},
        "lorebook": {"name": "旧世界书", "entries": []},
        "regexScripts": [{"id": "keep-regex"}],
        "stateTasks": [{"id": "keep-task"}],
    }
    generated = FakeStudioAlgorithms().generate(
        document,
        section="full",
        config={},
    )
    result = _apply_generated_section(
        document,
        generated,
        section="full",
    )
    assert result["identity"]["name"] == "完整生成角色"
    assert result["lorebook"]["name"] == "新世界书"
    assert result["coreMessages"]["first_message"] == "保留问候"
    assert result["regexScripts"] == [{"id": "keep-regex"}]
    assert result["stateTasks"] == [{"id": "keep-task"}]


def test_short_command_idempotency_is_atomic_under_concurrency(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    book_id = str(studio_platform["book"]["id"])

    def create_once() -> dict[str, Any]:
        return repository.create_document(
            book_id=book_id,
            title="并发幂等角色",
            idempotency_key="same-create",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        created = list(pool.map(lambda _value: create_once(), range(2)))
    assert created[0]["id"] == created[1]["id"]
    assert len(repository.index(book_id=book_id)["documents"]) == 1
    with pytest.raises(StudioConflict, match="different request"):
        repository.create_document(
            book_id=book_id,
            title="不同请求",
            idempotency_key="same-create",
        )

    first = repository.create_session(
        document_id=str(created[0]["id"]),
        title="幂等会话",
        idempotency_key="same-session",
    )
    replay = repository.create_session(
        document_id=str(created[0]["id"]),
        title="幂等会话",
        idempotency_key="same-session",
    )
    assert replay["sessionId"] == first["sessionId"]
    assert len(
        repository.chat_state(str(created[0]["id"]))["sessions"]
    ) == 1


def test_chat_bootstrap_is_atomic_under_concurrency(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="并发会话角色",
    )

    def ensure_once() -> dict[str, Any]:
        return repository.ensure_active_session(
            document_id=str(document["id"]),
            title="首次对话",
            greeting="你好",
            greeting_source={"type": "first_message", "index": 0},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        sessions = list(pool.map(lambda _value: ensure_once(), range(2)))
    assert sessions[0]["sessionId"] == sessions[1]["sessionId"]
    state = repository.chat_state(str(document["id"]))
    assert len(state["sessions"]) == 1
    assert state["indexRevision"] == 2
    assert state["activeSession"]["messages"][0]["content"] == "你好"
    created = repository.create_session(
        document_id=str(document["id"]),
        title="第二次对话",
        base_index_revision=2,
        idempotency_key="indexed-session",
    )
    assert created["indexRevision"] == 3
    refreshed = repository.chat_state(str(document["id"]))
    archived = next(
        item for item in refreshed["sessions"] if item["archived"]
    )
    assert archived["messageCount"] == 1
    assert archived["lastMessageExcerpt"] == "你好"
    with pytest.raises(StudioConflict, match="index revision"):
        repository.create_session(
            document_id=str(document["id"]),
            title="过期标签页",
            base_index_revision=2,
            idempotency_key="stale-indexed-session",
        )


def test_candidates_expose_timeline_page_counts_without_dialogue_scan(
    studio_platform,
) -> None:
    timeline_id = str(uuid.uuid4())
    character_id = str(uuid.uuid4())
    with studio_platform["engine"].begin() as connection:
        connection.execute(
            insert(timeline_versions).values(
                id=timeline_id,
                book_id=str(studio_platform["book"]["id"]),
                mode="enhanced",
                status="ready",
                content_json="{}",
                dependency_fingerprint="candidate-source",
                is_active=True,
            )
        )
        connection.execute(
            insert(timeline_characters).values(
                id=character_id,
                timeline_version_id=timeline_id,
                name="Saber",
                payload_json=json.dumps(
                    {
                        "name": "Saber",
                        "aliases": ["阿尔托莉雅"],
                        "first_appearance": 2,
                        "key_moments": [
                            {"page": 4, "event": "拔剑"},
                            {"page": 9, "event": "决战"},
                        ],
                        "related_page_numbers": [2, 4, 7, 9],
                        "dialogues": ["不应返回"],
                        "sample_pages": [99],
                    },
                    ensure_ascii=False,
                ),
            )
        )
    app = create_api_app(
        ApiSettings(
            data_root=studio_platform["data_root"],
            identity=RuntimeIdentity(
                epoch_id="studio-candidate-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=studio_platform["engine"],
        )
    )
    response = app.test_client().get(
        "/api/v2/studio/books/"
        f"{studio_platform['book']['id']}/candidates"
    )
    assert response.status_code == 200
    candidate = response.get_json()["items"][0]
    assert candidate["firstAppearancePage"] == 2
    assert candidate["keyMomentCount"] == 2
    assert candidate["relatedPageCount"] == 4
    assert candidate["relatedPageNumbers"] == [2, 4, 7, 9]
    assert "dialogues" not in candidate
    assert "sample_pages" not in candidate


def test_studio_http_short_commands_and_operation_event_catchup(
    studio_platform,
) -> None:
    app = create_api_app(
        ApiSettings(
            data_root=studio_platform["data_root"],
            identity=RuntimeIdentity(
                epoch_id="studio-http-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=studio_platform["engine"],
        )
    )
    client = app.test_client()
    book_id = str(studio_platform["book"]["id"])
    created_response = client.post(
        f"/api/v2/studio/books/{book_id}/documents",
        json={"title": "HTTP Saber"},
        headers={"Idempotency-Key": "studio-http-create"},
    )
    assert created_response.status_code == 201
    document = created_response.get_json()

    image_buffer = BytesIO()
    Image.new("RGB", (8, 8), (90, 40, 20)).save(
        image_buffer,
        format="PNG",
    )
    avatar = client.post(
        f"/api/v2/studio/documents/{document['id']}/avatar",
        data={
            "baseRevision": str(document["revision"]),
            "file": (BytesIO(image_buffer.getvalue()), "avatar.png"),
        },
        content_type="multipart/form-data",
        headers={"Idempotency-Key": "studio-http-avatar"},
    )
    assert avatar.status_code == 201
    document = avatar.get_json()
    assert document["avatarUrl"].startswith("/api/v2/assets/")

    session_response = client.post(
        f"/api/v2/studio/documents/{document['id']}/chat/sessions",
        json={
            "baseIndexRevision": 1,
            "title": "HTTP chat",
            "greeting": "你好",
        },
        headers={"Idempotency-Key": "studio-http-session"},
    )
    assert session_response.status_code == 201
    session = session_response.get_json()
    preview = client.get(
        f"/api/v2/studio/chat/sessions/{session['sessionId']}"
        "/prompt-preview"
    )
    assert preview.status_code == 200
    assert preview.get_json()["promptPreview"]["messages"][0][
        "content"
    ] == "你好"

    repository = StudioRepository(studio_platform["engine"])
    accepted = repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(session["revision"]),
        content="测试事件",
        asset_ids=[],
        config={},
        idempotency_key="http-events",
    )
    claimed = OperationRepository(studio_platform["engine"]).claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=FakeStudioAlgorithms(),
    ).handle(*claimed)
    events = client.get(
        f"/api/v2/operations/{accepted['operationId']}/events"
    )
    assert events.status_code == 200
    assert events.get_json()["items"][-1]["type"] == "operation_completed"


def test_studio_exports_unicode_titles_with_wsgi_safe_headers(
    studio_platform,
) -> None:
    app = create_api_app(
        ApiSettings(
            data_root=studio_platform["data_root"],
            identity=RuntimeIdentity(
                epoch_id="studio-export-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=studio_platform["engine"],
        )
    )
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="回归角色",
    )
    client = app.test_client()

    for output_format in ("v3", "v2", "worldbook", "png"):
        response = client.get(
            f"/api/v2/studio/documents/{document['id']}/export",
            query_string={"format": output_format},
        )
        assert response.status_code == 200
        disposition = response.headers["Content-Disposition"]
        disposition.encode("latin-1")
        assert "attachment" in disposition
        assert "filename*=UTF-8''" in disposition
        if output_format == "png":
            assert response.mimetype == "image/png"
            assert response.data.startswith(b"\x89PNG")
        else:
            assert response.mimetype == "application/json"
            assert json.loads(response.data.decode("utf-8"))
