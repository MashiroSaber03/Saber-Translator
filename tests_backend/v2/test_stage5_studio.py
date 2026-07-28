from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import uuid

import pytest
from sqlalchemy import select

from src.backend_v2.content.repository import ContentRepository
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
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.studio.repository import (
    StudioConflict,
    StudioRepository,
)
from src.backend_v2.studio.service import StudioOperationService


class FakeStudioAlgorithms:
    def generate(
        self,
        document: Mapping[str, Any],
        *,
        section: str,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if section == "identity":
            return {
                "identity": {
                    "name": "生成角色",
                    "description": "后端生成",
                    "personality": "坚定",
                    "scenario": "测试",
                }
            }
        return {"summary": "review"}

    def chat(self, *, prompt: str, system: str, config: Mapping[str, Any]) -> str:
        return "持久化回复"

    def summarize(self, messages, *, config):
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
        payload = connection.execute(
            select(studio_documents.c.payload_json).where(
                studio_documents.c.id == created["id"]
            )
        ).scalar_one()
    assert '"name":"Saber"' not in payload
    assert "grounding" not in payload

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
    with pytest.raises(StudioConflict, match="revision"):
        repository.update_document(
            document_id=str(created["id"]),
            base_revision=1,
            title="Saber",
            document=changed,
        )


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
