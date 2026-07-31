from __future__ import annotations

from io import BytesIO
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import uuid
import zipfile

from PIL import Image
import pytest
from sqlalchemy import insert, select, update

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.operations.executor import WorkerOperationRunner
from src.backend_v2.operations.repository import (
    OperationConflict,
    OperationFence,
    OperationFenced,
    OperationLocked,
    OperationRepository,
    RenderRequestRepository,
)
from src.backend_v2.operations.repair import PageRepairService
from src.backend_v2.plugins.repository import PluginRegistry
from src.backend_v2.plugins.runtime import PluginOperationRuntime
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
    process_epochs,
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


def test_page_detect_publishes_precise_mask_and_keeps_page_state_consistent(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == platform["bubble_id"])
            .values(
                payload_json=json.dumps(
                    {
                        "coords": [4, 4, 20, 20],
                        "fillColor": "#ff0000",
                        "inpaintMethod": "solid",
                        "originalText": "保留原文",
                        "translatedText": "",
                    }
                )
            )
        )
    accepted, _ = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="page_detect",
        base_revision=1,
        bubble_id=None,
        payload={},
        idempotency_key="detect-with-precise-mask",
    )

    class DetectAlgorithms:
        def detect(self, image, _config):
            mask = Image.new("L", image.size, 0)
            mask.putpixel((8, 9), 255)
            return {
                "coords": [[4, 4, 20, 20]],
                "polygons": [[]],
                "angles": [0],
                "auto_directions": ["h"],
                "textlines_per_bubble": [[]],
                "raw_mask": mask,
            }

    service = InteractivePageOperationService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
        algorithms=DetectAlgorithms(),
    )
    claimed = repository.claim_next(
        executor_role="worker",
        executor_epoch_id=platform["worker_epoch_id"],
        allowed_kinds=("page_detect",),
    )
    assert claimed is not None
    fence, operation = claimed
    result = service.handle(fence, operation)

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.detection_state,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        bubble_payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )
        mask_row = connection.execute(
            select(
                page_assets.c.asset_id,
                page_assets.c.input_document_revision,
                page_assets.c.producer_operation_id,
                assets.c.relative_path,
                assets.c.width,
                assets.c.height,
            )
            .join(assets, assets.c.id == page_assets.c.asset_id)
            .where(
                page_assets.c.page_id == platform["page_id"],
                page_assets.c.role == "text_mask",
            )
        ).mappings().one()
    assert page == (2, "processed", None, "not_rendered")
    assert bubble_payload["originalText"] == "保留原文"
    assert bubble_payload["autoTextDirection"] == "horizontal"
    assert result["textMaskAssetId"] == mask_row["asset_id"]
    assert mask_row["input_document_revision"] == 2
    assert mask_row["producer_operation_id"] == accepted["operationId"]
    assert (mask_row["width"], mask_row["height"]) == (64, 64)
    with Image.open(
        platform["data_root"] / Path(mask_row["relative_path"])
    ) as stored_mask:
        assert stored_mask.mode == "L"
        assert stored_mask.getpixel((8, 9)) == 255


def test_worker_operation_runner_delegates_claimed_operation_to_handler(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    accepted, _ = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="page_detect",
        base_revision=1,
        bubble_id=None,
        payload={},
        idempotency_key="worker-plugin-runtime",
    )
    calls: list[tuple[str, object]] = []

    def handle(fence, operation):
        calls.append(("handler", operation["kind"]))
        return {"operationId": fence.operation_id}

    runner = WorkerOperationRunner(
        repository,
        worker_epoch_id=platform["worker_epoch_id"],
        handlers={"page_detect": handle},
    )

    assert runner.run_one() is True
    assert calls == [("handler", "page_detect")]
    stored = repository.get(accepted["operationId"])
    assert stored["status"] == "completed"
    assert stored["result"] == {"operationId": accepted["operationId"]}


def test_worker_ocr_plugin_mutates_domain_result_before_publish(
    operation_platform,
) -> None:
    platform = operation_platform
    manifest = {
        "schema_version": 3,
        "plugin_id": "operation_ocr_mutation",
        "display_name": "Operation OCR mutation",
        "package_version": "1.0.0",
        "entrypoint": "plugin.py:Plugin",
        "hooks": ["after_ocr"],
        "supported_steps": ["ocr"],
        "supported_modes": ["standard"],
        "priority": 100,
        "failure_policy": "fail",
        "default_enabled": True,
        "config_schema": {},
    }
    archive_payload = BytesIO()
    with zipfile.ZipFile(
        archive_payload,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("plugin.json", json.dumps(manifest))
        archive.writestr(
            "plugin.py",
            (
                "class Plugin:\n"
                "    def after_ocr(self, context, data):\n"
                "        result = dict(data)\n"
                "        result['originalTexts'] = [\n"
                "            value + '【hook】' for value in data['originalTexts']\n"
                "        ]\n"
                "        return result\n"
            ),
        )
    PluginRegistry(
        data_root=platform["data_root"],
        engine=platform["engine"],
    ).import_archive(
        data=archive_payload.getvalue(),
        base_revision=0,
        idempotency_key="operation-ocr-mutation-v1",
    )
    repository = OperationRepository(platform["engine"])
    repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_ocr",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="operation-ocr-mutation",
    )

    class OcrAlgorithms:
        def ocr(self, _image, _bubbles, _config):
            return {
                "texts": ["こんにちは"],
                "results": [{"confidence": 1.0}],
            }

    runtime = PluginOperationRuntime(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
    )
    service = InteractivePageOperationService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
        algorithms=OcrAlgorithms(),
        plugin_runtime=runtime,
    )
    runner = WorkerOperationRunner(
        repository,
        worker_epoch_id=platform["worker_epoch_id"],
        handlers={"bubble_ocr": service.handle},
    )
    assert runner.run_one() is True

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.id == platform["bubble_id"]
                )
            ).scalar_one()
        )
    assert payload["originalText"] == "こんにちは【hook】"


def test_zero_row_operation_renewal_fences_all_late_writes(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    accepted, _ = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="page_detect",
        base_revision=1,
        bubble_id=None,
        payload={},
        idempotency_key="renewal-fence",
    )
    claimed = repository.claim_next(
        executor_role="worker",
        executor_epoch_id=platform["worker_epoch_id"],
        allowed_kinds=("page_detect",),
    )
    assert claimed is not None
    fence, _operation = claimed
    assert repository.renew(fence) is not None
    with platform["engine"].begin() as connection:
        connection.execute(
            update(process_epochs)
            .where(process_epochs.c.id == platform["worker_epoch_id"])
            .values(status="lost")
        )
    assert repository.renew(fence) is None
    with pytest.raises(OperationFenced):
        repository.complete(fence, result={"late": True})
    assert repository.get(str(accepted["operationId"]))["status"] == "running"


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


def test_operation_errors_and_events_never_expose_frozen_credentials(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    canary = "CANARY-OPERATION-API-KEY-92741"
    credential_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    with platform["engine"].begin() as connection:
        connection.execute(
            insert(credentials).values(
                id=credential_id,
                domain="translation",
                provider="canary",
            )
        )
        connection.execute(
            insert(credential_versions).values(
                id=version_id,
                credential_id=credential_id,
                version=1,
                secret_json=json.dumps({"apiKey": canary}),
                key_fingerprint="1" * 64,
            )
        )
    accepted, _ = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_translate",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={
            "provider": "canary",
            "credentialVersionId": version_id,
        },
        idempotency_key="credential-redaction",
    )
    claimed = repository.claim_next(
        executor_role="api",
        executor_epoch_id=platform["api_epoch_id"],
        allowed_kinds=("bubble_translate",),
    )
    assert claimed is not None
    fence, _operation = claimed
    repository.append_event(
        fence,
        event_type="provider_debug",
        payload={
            "apiKey": canary,
            "header": f"Authorization: Bearer {canary}",
        },
    )
    repository.fail(
        fence,
        code="PROVIDER_FAILED",
        message=(
            f"upstream rejected {canary}; "
            f"Authorization: Bearer {canary}; "
            r"C:\Users\developer\private\trace.txt"
        ),
    )

    exposed = json.dumps(
        {
            "operation": repository.get(str(accepted["operationId"])),
            "events": repository.events_after(
                str(accepted["operationId"])
            ),
        },
        ensure_ascii=False,
    )
    assert canary not in exposed
    assert r"C:\Users\developer" not in exposed
    assert "[REDACTED]" in exposed
    assert "[LOCAL_PATH]" in exposed


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
