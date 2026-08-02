from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import threading
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
    analysis_artifacts,
    assets,
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
from src.backend_v2.studio.service import (
    DefaultStudioAlgorithms,
    StudioOperationService,
    _provider_config,
    _validate_generated_payload,
)
from src.backend_v2.studio.service import _apply_generated_section


class FakeStudioAlgorithms:
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
        analysis_context: Mapping[str, Any] | None = None,
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


def test_studio_generation_prompt_consumes_analysis_context(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def chat_json(
        self,
        prompt: str,
        *,
        config: Mapping[str, Any],
        on_chunk=None,
    ) -> object:
        captured["prompt"] = prompt
        return {
            "identity": {
                "name": "Darmil",
                "description": "在陨石坑发现碎片的研究者",
            }
        }

    monkeypatch.setattr(
        DefaultStudioAlgorithms,
        "_chat_json",
        chat_json,
    )
    result = DefaultStudioAlgorithms().generate(
        {"title": "Darmil", "origin": {"source_character": "Darmil"}},
        section="identity",
        config={},
        analysis_context={
            "artifactId": "context-1",
            "payload": {
                "summary": "Darmil 在陨石坑发现了神秘碎片。",
            },
        },
    )
    assert result["identity"]["description"].startswith("在陨石坑")
    assert "Darmil 在陨石坑发现了神秘碎片" in captured["prompt"]
    assert (
        '顶层结构必须为：{"identity":{"name":"角色名"'
        in captured["prompt"]
    )
    assert "当前角色文档" in captured["prompt"]


def test_studio_complete_respects_saved_nonstream_setting(
    monkeypatch,
) -> None:
    from src.shared.ai_transport import OpenAICompatibleChatTransport

    captured: dict[str, Any] = {}

    def complete(self, request) -> str:
        captured["use_stream"] = (
            request.openai_options.execution.use_stream
        )
        captured["has_callback"] = (
            request.runtime_options.on_stream_chunk is not None
        )
        return "{}"

    monkeypatch.setattr(
        OpenAICompatibleChatTransport,
        "complete",
        complete,
    )
    result = DefaultStudioAlgorithms._complete(
        [{"role": "user", "content": "test"}],
        config={
            "chat": {
                "provider": "test",
                "model_name": "test-model",
                "openai_options": {
                    "execution": {"use_stream": False},
                },
            }
        },
        temperature=0.3,
        force_json=True,
        on_chunk=lambda _chunk, _full: None,
    )
    assert result == "{}"
    assert captured == {
        "use_stream": False,
        "has_callback": True,
    }


def test_studio_agent_emits_complete_nonstream_response(
    tmp_path: Path,
) -> None:
    class NonStreamingStudioAlgorithms(FakeStudioAlgorithms):
        def chat(
            self,
            *,
            messages,
            system: str,
            config: Mapping[str, Any],
            on_chunk=None,
        ) -> str:
            return "非流式卡片助手回复"

    service = StudioOperationService(
        engine=create_sqlite_engine(tmp_path / "agent.sqlite3"),
        algorithms=NonStreamingStudioAlgorithms(),
    )

    assert list(
        service.agent_chunks(
            document={"identity": {"name": "测试角色"}},
            messages=[{"role": "user", "content": "请审查"}],
            config={},
            cancelled=threading.Event(),
        )
    ) == ["非流式卡片助手回复"]


def test_full_generation_rejects_partial_top_level_payload() -> None:
    with pytest.raises(ValueError, match="coreMessages.*lorebook"):
        _validate_generated_payload(
            {"status": {"frozen_sections": []}},
            {"identity": {"name": "Darmil"}},
            section="full",
        )


def test_studio_chat_uses_vlm_for_image_attachments(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def complete(
        messages,
        *,
        config,
        temperature,
        force_json,
        on_chunk,
        prefer_vlm=False,
    ):
        captured["messages"] = messages
        captured["prefer_vlm"] = prefer_vlm
        return "图片已收到"

    monkeypatch.setattr(
        DefaultStudioAlgorithms,
        "_complete",
        staticmethod(complete),
    )
    result = DefaultStudioAlgorithms().chat(
        messages=[
            {
                "role": "user",
                "content": "看图回答",
                "attachmentDataUrls": ["data:image/png;base64,AAAA"],
            }
        ],
        system="系统提示",
        config={},
    )
    assert result == "图片已收到"
    assert captured["prefer_vlm"] is True
    assert captured["messages"][1]["content"][0]["type"] == "image_url"
    assert _provider_config(
        {
                "chat": {"provider": "text", "model_name": "text-model"},
                "vlm": {"provider": "vision", "model_name": "vision-model"},
        },
        prefer_vlm=True,
    )["model"] == "vision-model"


def test_studio_chat_merges_session_system_messages(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def complete(
        messages,
        *,
        config,
        temperature,
        force_json,
        on_chunk,
        prefer_vlm=False,
    ):
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr(
        DefaultStudioAlgorithms,
        "_complete",
        staticmethod(complete),
    )

    result = DefaultStudioAlgorithms().chat(
        messages=[
            {"role": "system", "content": "导入会话上下文"},
            {"role": "user", "content": "继续对话"},
        ],
        system="角色系统提示",
        config={},
    )

    assert result == "ok"
    assert [
        message["role"] for message in captured["messages"]
    ] == ["system", "user"]
    assert captured["messages"][0]["content"] == (
        "角色系统提示\n\n导入会话上下文"
    )


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


def test_generate_operation_freezes_analysis_context(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Context Character",
    )
    context = {
        "artifactId": "compressed-1",
        "revision": 3,
        "dependencyFingerprint": "context-fingerprint",
        "payload": {"summary": "角色在第六页发现碎片"},
    }
    chat_config = {"provider": "chat-provider", "model_name": "chat-model"}
    accepted = repository.create_generate_operation(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        section="identity",
        config={
            "chat": chat_config,
            "vlm": {"provider": "vlm-provider", "model_name": "vlm-model"},
            "embedding": {
                "provider": "unused-provider",
                "model_name": "unused-model",
            },
        },
        analysis_context=context,
        idempotency_key="context-generate",
    )
    stored = OperationRepository(studio_platform["engine"]).get(
        str(accepted["operationId"])
    )
    assert stored["request"]["analysisContext"] == context
    assert stored["request"]["config"] == {"chat": chat_config}


def test_generate_rejects_unchanged_document_without_revision_bump(
    studio_platform,
) -> None:
    class NoopStudioAlgorithms(FakeStudioAlgorithms):
        def generate(
            self,
            document: Mapping[str, Any],
            *,
            section: str,
            config: Mapping[str, Any],
            analysis_context: Mapping[str, Any] | None = None,
            on_chunk=None,
        ) -> Mapping[str, Any]:
            return dict(document)

    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="No-op Character",
    )
    repository.create_generate_operation(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        section="full",
        config={},
        analysis_context={"payload": {"summary": "source"}},
        idempotency_key="noop-generate",
    )
    operations = OperationRepository(studio_platform["engine"])
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_generate",),
    )
    assert claimed is not None
    with pytest.raises(ValueError, match="no document changes"):
        StudioOperationService(
            engine=studio_platform["engine"],
            repository=repository,
            algorithms=NoopStudioAlgorithms(),
        ).handle(*claimed)
    operations.fail(
        claimed[0],
        code="NO_DOCUMENT_CHANGES",
        message="no changes",
    )
    restored = repository.get_document(str(document["id"]))
    assert restored["revision"] == document["revision"]


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
        base_index_revision=1,
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
        base_index_revision=1,
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
        base_index_revision=1,
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
        base_index_revision=1,
        greeting="开场",
    )
    result = repository.delete_message_chain(
        message_id=str(session["messages"][0]["messageId"]),
        base_revision=int(session["revision"]),
    )
    restored = repository.get_session(str(session["sessionId"]))
    assert result["sessionRevision"] == session["revision"] + 1
    assert restored["messages"] == []


def test_new_chat_session_runs_initialization_state_tasks(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    changed = dict(document)
    changed["stateTasks"] = [
        {
            "id": "task-initialization",
            "name": "初始化信任值",
            "triggerTiming": "initialization",
            "interval": 0,
            "commands": (
                "<<taskjs>>\n"
                'await STscript("/setvar key=trust_score 20");\n'
                "<</taskjs>>"
            ),
            "disabled": False,
        }
    ]
    document = repository.update_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        title=str(document["title"]),
        document=changed,
    )

    session = repository.create_session(
        document_id=str(document["id"]),
        title="初始化回归",
        base_index_revision=1,
        greeting="开场",
    )
    assert session["variables"] == {"trust_score": "20"}
    assert session["runtimeState"] == {
        "event_counts": {
            "message_received": 0,
            "message_sent": 0,
        },
        "matched_lorebook_ids": [],
    }
    assert session["messages"][0]["variablesSnapshot"] == {
        "trust_score": "20"
    }
    assert session["messages"][0]["runtimeLog"] == [
        {
            "type": "task",
            "name": "初始化信任值",
            "event": "initialization",
            "interval": 0,
        }
    ]


def test_chat_chain_rewrite_restores_runtime_and_variables(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="Saber",
    )
    changed = dict(document)
    changed["lorebook"] = {
        "name": "测试世界书",
        "entries": [
            {
                "id": "lore-saber",
                "keys": ["Saber"],
                "secondary_keys": [],
                "comment": "Saber 设定",
                "content": "Saber 的任务由后端执行。",
                "constant": True,
                "selective": True,
                "enabled": True,
                "position": "before_char",
                "priority": 100,
                "probability": 100,
                "prevent_recursion": True,
            }
        ],
    }
    changed["stateTasks"] = [
        {
            "id": "task-received",
            "name": "收到消息",
            "triggerTiming": "message_received",
            "interval": 1,
            "commands": (
                "<<taskjs>>\n"
                "/setvar key=phase received\n"
                "<</taskjs>>"
            ),
            "disabled": False,
        },
        {
            "id": "task-sent",
            "name": "发送消息",
            "triggerTiming": "message_sent",
            "interval": 1,
            "commands": (
                "<<taskjs>>\n"
                "/setvar key=phase sent\n"
                "<</taskjs>>"
            ),
            "disabled": False,
        },
    ]
    document = repository.update_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        title=str(document["title"]),
        document=changed,
    )
    session = repository.create_session(
        document_id=str(document["id"]),
        title="运行态回归",
        base_index_revision=1,
        greeting="开场",
    )
    sent = repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(session["revision"]),
        content="Saber",
        asset_ids=[],
        config={},
        idempotency_key="runtime-first-turn",
    )
    operations = OperationRepository(studio_platform["engine"])
    service = StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=FakeStudioAlgorithms(),
    )
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    service.handle(*claimed)

    completed = repository.get_session(str(session["sessionId"]))
    assistant = completed["messages"][-1]
    assert [item["type"] for item in assistant["runtimeLog"]] == [
        "lorebook",
        "task",
        "task",
    ]
    assert assistant["variablesSnapshot"] == {"phase": "sent"}
    assert completed["variables"] == {"phase": "sent"}
    assert completed["runtimeState"] == {
        "event_counts": {
            "message_received": 1,
            "message_sent": 1,
        },
        "matched_lorebook_ids": ["lore-saber"],
    }

    edited = repository.edit_or_regenerate_message(
        message_id=str(sent["userMessageId"]),
        base_revision=int(completed["revision"]),
        content="Saber 修改后",
        config={},
        idempotency_key="runtime-edit-turn",
    )
    claimed = operations.claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    assert claimed[1]["operationId"] == edited["operationId"]
    assert claimed[1]["request"]["variables"] == {}
    assert claimed[1]["request"]["runtimeState"] == {
        "event_counts": {
            "message_received": 0,
            "message_sent": 0,
        },
        "matched_lorebook_ids": [],
    }
    service.handle(*claimed)

    regenerated = repository.get_session(str(session["sessionId"]))
    assert [item["type"] for item in regenerated["messages"][-1]["runtimeLog"]] == [
        "lorebook",
        "task",
        "task",
    ]
    assert regenerated["runtimeState"]["event_counts"] == {
        "message_received": 1,
        "message_sent": 1,
    }
    assert regenerated["runtimeState"]["matched_lorebook_ids"] == [
        "lore-saber"
    ]

    repository.delete_message_chain(
        message_id=str(regenerated["messages"][-1]["messageId"]),
        base_revision=int(regenerated["revision"]),
    )
    rolled_back = repository.get_session(str(session["sessionId"]))
    assert rolled_back["messages"][-1]["role"] == "user"
    assert rolled_back["variables"] == {}
    assert rolled_back["runtimeState"] == {
        "event_counts": {
            "message_received": 0,
            "message_sent": 0,
        },
        "matched_lorebook_ids": [],
    }


def test_chat_input_regex_uses_prompt_text_without_rewriting_user_message(
    studio_platform,
) -> None:
    captured: dict[str, Any] = {}

    class CapturingAlgorithms(FakeStudioAlgorithms):
        def chat(self, *, messages, system, config, on_chunk=None) -> str:
            captured["messages"] = messages
            return "完成"

    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="正则角色",
    )
    changed = dict(document)
    changed["regexScripts"] = [
        {
            "id": "prompt-only",
            "scriptName": "仅修改模型提示",
            "findRegex": "原文",
            "replaceString": "模型文本",
            "placement": [1],
            "promptOnly": True,
            "runOnEdit": True,
        }
    ]
    document = repository.update_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        title=str(document["title"]),
        document=changed,
    )
    session = repository.create_session(
        document_id=str(document["id"]),
        title="正则会话",
        base_index_revision=1,
    )
    repository.send_message(
        session_id=str(session["sessionId"]),
        base_revision=int(session["revision"]),
        content="原文",
        asset_ids=[],
        config={},
        idempotency_key="prompt-regex",
    )
    service = StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=CapturingAlgorithms(),
    )
    current = repository.get_session(str(session["sessionId"]))
    preview = service.prompt_preview(document=document, session=current)
    assert preview["messages"][-1]["content"] == "模型文本"

    claimed = OperationRepository(studio_platform["engine"]).claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_chat",),
    )
    assert claimed is not None
    service.handle(*claimed)
    assert captured["messages"][-1]["content"] == "模型文本"
    restored = repository.get_session(str(session["sessionId"]))
    assert restored["messages"][0]["content"] == "原文"


def test_png_and_session_portable_roundtrip_uses_asset_ids(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    io_service = StudioIOService(
        data_root=studio_platform["data_root"],
        engine=studio_platform["engine"],
        repository=repository,
    )
    image_buffer = BytesIO()
    Image.new("RGB", (24, 32), (10, 20, 30)).save(
        image_buffer,
        format="PNG",
    )
    image_bytes = image_buffer.getvalue()
    document = io_service.import_document(
        book_id=str(studio_platform["book"]["id"]),
        upload=BytesIO(image_bytes),
        filename="Saber.png",
        idempotency_key="plain-image-import",
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
        base_index_revision=1,
    )
    imported_session = io_service.import_session(
        document_id=str(document["id"]),
        base_index_revision=2,
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


def test_failed_session_attachment_import_marks_published_assets_for_gc(
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
        title="附件清理角色",
    )
    image_buffer = BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(
        image_buffer,
        format="PNG",
    )
    with studio_platform["engine"].connect() as connection:
        before = set(connection.execute(select(assets.c.id)).scalars())

    with pytest.raises(ValueError, match="base64"):
        io_service.import_session(
            document_id=str(document["id"]),
            base_index_revision=1,
            idempotency_key="failed-attachment-import",
            payload={
                "messages": [
                    {
                        "role": "user",
                        "content": "附件",
                        "attachments": [
                            {
                                "blob_base64": base64.b64encode(
                                    image_buffer.getvalue()
                                ).decode("ascii")
                            },
                            {"blob_base64": "not-base64"},
                        ],
                    }
                ]
            },
        )

    with studio_platform["engine"].connect() as connection:
        published = list(
            connection.execute(
                select(assets.c.id, assets.c.gc_marked_at).where(
                    assets.c.id.not_in(before)
                )
            ).mappings()
        )
    assert len(published) == 1
    assert published[0]["gc_marked_at"] is not None


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
        base_index_revision=1,
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
        base_index_revision=1,
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
        base_index_revision=2,
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
    assert repository.chat_state(str(document["id"]))["indexRevision"] == 4
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


def test_ai_review_persists_without_overwriting_structural_diagnostics(
    studio_platform,
) -> None:
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=str(studio_platform["book"]["id"]),
        title="审查角色",
    )
    validated = repository.validate_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        idempotency_key="review-preserves-diagnostics",
    )
    document = repository.get_document(str(document["id"]))
    accepted = repository.create_generate_operation(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        section="review",
        config={},
        idempotency_key="review-once",
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

    restored = repository.get_document(str(document["id"]))
    assert restored["revision"] == document["revision"] + 1
    assert (
        restored["status"]["last_diagnostics"]
        == validated["diagnostics"]
    )
    assert restored["exportArtifacts"]["last_review"] == {
        "summary": "review",
        "issues": [],
        "suggestions": [],
    }
    operation = OperationRepository(studio_platform["engine"]).get(
        str(accepted["operationId"])
    )
    assert operation["status"] == "completed"
    assert operation["result"] == {
        "documentId": str(document["id"]),
        "documentRevision": document["revision"] + 1,
    }

    accepted_identity = repository.create_generate_operation(
        document_id=str(document["id"]),
        base_revision=int(restored["revision"]),
        section="identity",
        config={},
        idempotency_key="identity-invalidates-diagnostics",
    )
    claimed_identity = OperationRepository(
        studio_platform["engine"]
    ).claim_next(
        executor_role="api",
        executor_epoch_id=studio_platform["epoch_id"],
        allowed_kinds=("studio_generate",),
    )
    assert claimed_identity is not None
    StudioOperationService(
        engine=studio_platform["engine"],
        repository=repository,
        algorithms=FakeStudioAlgorithms(),
    ).handle(*claimed_identity)
    regenerated = repository.get_document(str(document["id"]))
    assert regenerated["status"]["last_diagnostics"] is None
    assert regenerated["status"]["last_validated_at"] is None
    assert regenerated["exportArtifacts"]["last_review"] == {
        "summary": "review",
        "issues": [],
        "suggestions": [],
    }
    assert OperationRepository(studio_platform["engine"]).get(
        str(accepted_identity["operationId"])
    )["result"]["documentRevision"] == regenerated["revision"]


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
        base_index_revision=1,
        idempotency_key="same-session",
    )
    replay = repository.create_session(
        document_id=str(created[0]["id"]),
        title="幂等会话",
        base_index_revision=1,
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
    changed = dict(document)
    changed["stateTasks"] = [
        {
            "id": "bootstrap-task",
            "name": "初始化状态",
            "triggerTiming": "initialization",
            "interval": 0,
            "commands": "/setvar key=bootstrapped yes",
        }
    ]
    document = repository.update_document(
        document_id=str(document["id"]),
        base_revision=int(document["revision"]),
        title=str(document["title"]),
        document=changed,
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
    assert state["activeSession"]["variables"] == {"bootstrapped": "yes"}
    assert state["activeSession"]["messages"][0]["runtimeLog"][0][
        "type"
    ] == "task"
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
                        "first_page": 2,
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


def test_studio_generate_route_freezes_ready_compressed_context(
    studio_platform,
) -> None:
    book_id = str(studio_platform["book"]["id"])
    context_id = str(uuid.uuid4())
    context_payload = {
        "summary": "Darmil 在陨石坑发现了碎片。",
        "characters": [{"name": "Darmil"}],
    }
    with studio_platform["engine"].begin() as connection:
        connection.execute(
            insert(analysis_artifacts).values(
                id=context_id,
                book_id=book_id,
                run_id=None,
                kind="compressed_context",
                template="default",
                status="ready",
                revision=2,
                is_active=True,
                dependency_fingerprint="compressed-fingerprint",
                payload_json=json.dumps(
                    context_payload,
                    ensure_ascii=False,
                ),
            )
        )
    repository = StudioRepository(studio_platform["engine"])
    document = repository.create_document(
        book_id=book_id,
        title="Darmil",
    )
    app = create_api_app(
        ApiSettings(
            data_root=studio_platform["data_root"],
            identity=RuntimeIdentity(
                epoch_id="studio-generate-route-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=studio_platform["engine"],
        )
    )
    response = app.test_client().post(
        f"/api/v2/studio/documents/{document['id']}/generate",
        json={
            "baseRevision": document["revision"],
            "section": "full",
        },
        headers={"Idempotency-Key": "studio-route-context"},
    )
    assert response.status_code == 202
    operation = OperationRepository(studio_platform["engine"]).get(
        response.get_json()["operationId"]
    )
    frozen = operation["request"]["analysisContext"]
    assert frozen["artifactId"] == context_id
    assert frozen["revision"] == 2
    assert frozen["dependencyFingerprint"] == "compressed-fingerprint"
    assert frozen["payload"] == context_payload


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
