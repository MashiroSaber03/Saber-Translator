from __future__ import annotations

from io import BytesIO
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import uuid

from PIL import Image
import pytest
from sqlalchemy import insert, select, update

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.operations.repository import (
    OperationConflict,
    OperationFence,
    OperationFenced,
    OperationLocked,
    OperationRepository,
    RenderRequestRepository,
)
from src.backend_v2.operations.repair import PageRepairService
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import (
    create_sqlite_engine,
    immediate_transaction,
)
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import (
    assets,
    bubbles,
    chapter_write_intents,
    credentials,
    credential_versions,
    metadata,
    operation_credential_snapshots,
    page_assets,
    pages,
    render_requests,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.translation.interactive_operations import (
    InteractivePageOperationService,
)


@pytest.fixture()
def operation_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(book_id=str(book["id"]), title="Chapter")
    storage = AssetStorageService(data_root, engine)
    importer = ImageImportService(
        data_root=data_root,
        repository=content,
        storage=storage,
    )
    image_bytes = BytesIO()
    with Image.new("RGB", (64, 64), (12, 34, 56)) as image:
        image.save(image_bytes, format="PNG")
    lease = content.create_import_lease(str(chapter["id"]))
    try:
        imported, _ = importer.import_page(
            chapter_id=str(chapter["id"]),
            logical_path="1.png",
            upload=BytesIO(image_bytes.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key="page-1",
        )
    finally:
        content.release_import_lease(
            chapter_id=str(chapter["id"]),
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )
    page_id = str(imported["page"]["id"])
    bubble_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            insert(bubbles).values(
                id=bubble_id,
                page_id=page_id,
                ordinal=1,
                payload_json=(
                    '{"coords":[4,4,20,20],"fillColor":"#ff0000",'
                    '"inpaintMethod":"solid","originalText":""}'
                ),
                updated_revision=1,
            )
        )
    epoch_repository = ProcessEpochRepository(engine)
    worker_epoch_id = str(uuid.uuid4())
    api_epoch_id = str(uuid.uuid4())
    epoch_repository.register(
        EpochRegistration(worker_epoch_id, "worker-token", "worker", 101)
    )
    epoch_repository.register(
        EpochRegistration(api_epoch_id, "api-token", "api", 102)
    )
    try:
        yield {
            "data_root": data_root,
            "engine": engine,
            "content": content,
            "book": book,
            "chapter": chapter,
            "page_id": page_id,
            "bubble_id": bubble_id,
            "worker_epoch_id": worker_epoch_id,
            "api_epoch_id": api_epoch_id,
        }
    finally:
        engine.dispose()


def test_page_operation_is_idempotent_fenced_and_persistent(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    accepted, replayed = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_ocr",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="ocr-1",
    )
    replay, was_replayed = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_ocr",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="ocr-1",
    )
    assert not replayed
    assert was_replayed
    assert replay == accepted

    claimed = repository.claim_next(
        executor_role="worker",
        executor_epoch_id=platform["worker_epoch_id"],
        allowed_kinds=("bubble_ocr",),
    )
    assert claimed is not None
    fence, operation = claimed
    assert operation["inputs"].keys() == {"source"}
    forged = OperationFence(
        operation_id=fence.operation_id,
        attempt_id=fence.attempt_id,
        lease_token="forged",
        executor_epoch_id=fence.executor_epoch_id,
        executor_role=fence.executor_role,
        lease_expires_at=fence.lease_expires_at,
    )
    with pytest.raises(OperationFenced):
        repository.complete(forged, result={"text": "forbidden"})
    repository.complete(fence, result={"text": "hello"})
    stored = repository.get(fence.operation_id)
    assert stored["status"] == "completed"
    assert stored["result"] == {"text": "hello"}


def test_operation_creation_obeys_revision_and_write_intent(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    with pytest.raises(OperationConflict):
        repository.create_page_operation(
            page_id=platform["page_id"],
            kind="page_detect",
            base_revision=99,
            bubble_id=None,
            payload={},
            idempotency_key="stale",
        )
    with platform["engine"].begin() as connection:
        from src.backend_v2.storage.schema import jobs

        job_id = str(uuid.uuid4())
        connection.execute(
            insert(jobs).values(
                id=job_id,
                kind="translation",
                status="queued",
                queue_rank=1,
                chapter_id=platform["chapter"]["id"],
                config_json="{}",
            )
        )
        connection.execute(
            insert(chapter_write_intents).values(
                chapter_id=platform["chapter"]["id"],
                job_id=job_id,
                intent_set_id=str(uuid.uuid4()),
                intent_generation=1,
                worker_epoch_id=platform["worker_epoch_id"],
                lease_token="intent",
                lease_expires_at=(
                    datetime.now(timezone.utc).replace(tzinfo=None)
                    + timedelta(minutes=1)
                ),
            )
        )
    with pytest.raises(OperationLocked, match="chapter_write_pending"):
        repository.create_page_operation(
            page_id=platform["page_id"],
            kind="page_detect",
            base_revision=1,
            bubble_id=None,
            payload={},
            idempotency_key="blocked",
        )


def test_bubble_translate_freezes_credential_and_publishes_document(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    credential_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    with platform["engine"].begin() as connection:
        connection.execute(
            insert(credentials).values(
                id=credential_id,
                domain="translation",
                provider="test",
            )
        )
        connection.execute(
            insert(credential_versions).values(
                id=version_id,
                credential_id=credential_id,
                version=1,
                secret_json='{"api_key":"server-secret"}',
                key_fingerprint="0" * 64,
            )
        )
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == platform["bubble_id"])
            .values(
                payload_json=(
                    '{"coords":[4,4,20,20],"fillColor":"#ff0000",'
                    '"inpaintMethod":"solid","originalText":"source"}'
                )
            )
        )

    accepted, replayed = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_translate",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={
            "provider": "test",
            "target_language": "zh",
            "credentialVersionId": version_id,
        },
        idempotency_key="translate-bubble",
    )
    assert not replayed
    with platform["engine"].connect() as connection:
        snapshot = connection.execute(
            select(operation_credential_snapshots).where(
                operation_credential_snapshots.c.operation_id
                == accepted["operationId"]
            )
        ).mappings().one()
    assert snapshot["credential_version_id"] == version_id

    class TranslateOnlyAlgorithms:
        def translate(self, texts, config, *, mode):
            assert texts == ["source"]
            assert config["api_key"] == "server-secret"
            assert "credentialVersionId" not in config
            assert mode == "single"
            return {"translated": ["译文"], "textbox": []}

    service = InteractivePageOperationService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
        algorithms=TranslateOnlyAlgorithms(),
    )
    claimed = repository.claim_next(
        executor_role="api",
        executor_epoch_id=platform["api_epoch_id"],
        allowed_kinds=("bubble_translate",),
    )
    assert claimed is not None
    fence, operation = claimed
    result = service.handle(fence, operation)
    assert result["translatedText"] == "译文"
    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.id == platform["bubble_id"]
                )
            ).scalar_one()
        )
        revision = connection.execute(
            select(pages.c.document_revision).where(
                pages.c.id == platform["page_id"]
            )
        ).scalar_one()
    assert payload["translatedText"] == "译文"
    assert revision == 2
    assert repository.get(str(accepted["operationId"]))["status"] == "completed"


def test_render_request_coalesces_and_old_revision_cannot_publish(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = RenderRequestRepository(platform["engine"])
    with immediate_transaction(platform["engine"]) as connection:
        request_id = repository.upsert(
            connection,
            page_id=platform["page_id"],
            requested_revision=1,
        )
    first = repository.claim_next(api_epoch_id=platform["api_epoch_id"])
    assert first is not None
    with immediate_transaction(platform["engine"]) as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(document_revision=2, render_status="stale")
        )
        assert repository.upsert(
            connection,
            page_id=platform["page_id"],
            requested_revision=2,
        ) == request_id

    published: list[int] = []
    assert not repository.complete(
        first,
        publisher=lambda _connection: published.append(1),
    )
    assert published == []
    with platform["engine"].connect() as connection:
        row = connection.execute(
            select(render_requests).where(render_requests.c.id == request_id)
        ).mappings().one()
        assert row["status"] == "pending"
        assert row["requested_revision"] == 2

    second = repository.claim_next(api_epoch_id=platform["api_epoch_id"])
    assert second is not None
    assert second.rendering_revision == 2
    assert repository.complete(
        second,
        publisher=lambda _connection: published.append(2),
    )
    assert published == [2]
    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(pages.c.rendered_revision, pages.c.render_status).where(
                pages.c.id == platform["page_id"]
            )
        ).one()
        assert page == (2, "ready")


def test_page_repair_advances_revision_replays_without_new_mask_and_renders(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    service = PageRepairService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
    )
    accepted, replayed = service.create_for_bubble(
        page_id=platform["page_id"],
        bubble_id=platform["bubble_id"],
        base_revision=1,
        idempotency_key="repair-1",
    )
    with platform["engine"].connect() as connection:
        asset_count = connection.execute(select(assets.c.id)).all()
        page = connection.execute(
            select(pages.c.document_revision, pages.c.render_status).where(
                pages.c.id == platform["page_id"]
            )
        ).one()
    assert not replayed
    assert accepted["documentRevision"] == 2
    assert page == (2, "awaiting_repair")

    replay, was_replayed = service.create_for_bubble(
        page_id=platform["page_id"],
        bubble_id=platform["bubble_id"],
        base_revision=1,
        idempotency_key="repair-1",
    )
    with platform["engine"].connect() as connection:
        assert len(connection.execute(select(assets.c.id)).all()) == len(asset_count)
    assert was_replayed
    assert replay == accepted

    claimed = repository.claim_next(
        executor_role="api",
        executor_epoch_id=platform["api_epoch_id"],
        allowed_kinds=("page_repair",),
    )
    assert claimed is not None
    fence, operation = claimed
    result = service.handle(fence, operation)
    assert result["documentRevision"] == 2
    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(pages.c.render_status).where(
                pages.c.id == platform["page_id"]
            )
        ).scalar_one()
        clean_path = connection.execute(
            select(assets.c.relative_path)
            .join(page_assets, page_assets.c.asset_id == assets.c.id)
            .where(
                page_assets.c.page_id == platform["page_id"],
                page_assets.c.role == "clean",
            )
        ).scalar_one()
        render = connection.execute(
            select(render_requests.c.requested_revision).where(
                render_requests.c.page_id == platform["page_id"]
            )
        ).scalar_one()
    assert page == "stale"
    assert render == 2
    with Image.open(platform["data_root"] / clean_path) as repaired:
        assert repaired.getpixel((10, 10)) == (255, 0, 0)
        assert repaired.getpixel((30, 30)) == (12, 34, 56)


def test_failed_page_repair_sets_explicit_page_state(operation_platform) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    service = PageRepairService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
    )
    service.create_for_bubble(
        page_id=platform["page_id"],
        bubble_id=platform["bubble_id"],
        base_revision=1,
        idempotency_key="repair-failure",
    )
    claimed = repository.claim_next(
        executor_role="api",
        executor_epoch_id=platform["api_epoch_id"],
        allowed_kinds=("page_repair",),
    )
    assert claimed is not None
    fence, _operation = claimed
    repository.fail(fence, code="TEST_FAILURE", message="failed")
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(pages.c.render_status).where(
                pages.c.id == platform["page_id"]
            )
        ).scalar_one() == "repair_failed"
