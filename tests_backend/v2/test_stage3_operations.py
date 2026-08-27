from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import uuid
import zipfile

from PIL import Image
import pytest
from sqlalchemy import insert, select, update

from src.backend_v2.auth.ownership import owner_scope
from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentConflict, ContentRepository
from src.backend_v2.operations.executor import WorkerOperationRunner
from src.backend_v2.operations.repository import (
    OperationConflict,
    OperationDataInvalid,
    OperationFence,
    OperationFenced,
    OperationLocked,
    OperationNotFound,
    OperationRepository,
    RenderRequestRepository,
)
from src.backend_v2.operations.repair import PageRepairService
from src.backend_v2.rendering.service import AuthoritativeRenderService
from src.backend_v2.plugins.repository import PluginRegistry
from src.backend_v2.plugins.runtime import PluginOperationRuntime
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import (
    create_sqlite_engine,
    immediate_transaction,
)
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.defaults import DEFAULT_TEXT_STYLE
from src.backend_v2.storage.schema import (
    app_settings,
    assets,
    bubbles,
    chapter_write_locks,
    credentials,
    credential_versions,
    metadata,
    operations,
    operation_events,
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


def _stored_job_progress(status: str) -> str:
    return json.dumps(
        {
            "executionMode": "sequential",
            "jobStatus": status,
            "totalItems": 0,
            "completedItems": 0,
            "failedItems": 0,
            "skippedItems": 0,
            "cancelledItems": 0,
            "pools": [],
        },
        separators=(",", ":"),
    )


def _bubble_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "originalText": "",
        "translatedText": "",
        "textboxText": "",
        "coords": [4, 4, 20, 20],
        "polygon": [],
        "fontSize": 26,
        "textDirection": "horizontal",
        "autoTextDirection": "horizontal",
        "textColor": "#000000",
        "fillColor": "#ff0000",
        "rotationAngle": 0,
        "position": {"x": 0, "y": 0},
        "strokeEnabled": False,
        "strokeColor": "#FFFFFF",
        "strokeWidth": 0,
        "lineSpacing": 1.0,
        "inlineAlign": "center",
        "blockAlign": "end",
        "inpaintMethod": "solid",
        "autoFgColor": None,
        "autoBgColor": None,
        "colorConfidence": 0.0,
        "textlines": [],
        "ocrResult": None,
    }
    payload.update(updates)
    return payload


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
    imported, _ = importer.import_page(
        chapter_id=str(chapter["id"]),
        logical_path="1.png",
        text_style=dict(DEFAULT_TEXT_STYLE),
        upload=BytesIO(image_bytes.getvalue()),
        idempotency_key="page-1",
    )
    page_id = str(imported["page"]["id"])
    bubble_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            insert(bubbles).values(
                id=bubble_id,
                page_id=page_id,
                ordinal=1,
                payload_json=json.dumps(_bubble_payload()),
                updated_revision=1,
            )
        )
    # Fixture actors do not run heartbeat threads. Keep their synthetic epochs
    # alive for the whole test even on a slow, full-suite CI run.
    epoch_repository = ProcessEpochRepository(engine, lease_seconds=300)
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


def test_empty_operation_claim_does_not_compete_for_sqlite_write_lock(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])

    with immediate_transaction(platform["engine"]):
        claimed = repository.claim_next(
            executor_role="api",
            executor_epoch_id=platform["api_epoch_id"],
            allowed_kinds=("bubble_translate",),
        )

    assert claimed is None


def test_operation_reads_and_page_creation_are_scoped_to_owner(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    accepted, _replayed = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_color",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="owned-operation",
    )
    operation_id = str(accepted["operationId"])

    with owner_scope(str(uuid.uuid4())):
        with pytest.raises(OperationNotFound):
            repository.get(operation_id)
        with pytest.raises(OperationNotFound):
            repository.events_after(operation_id)
        with pytest.raises(OperationNotFound):
            repository.create_page_operation(
                page_id=platform["page_id"],
                kind="bubble_color",
                base_revision=1,
                bubble_id=platform["bubble_id"],
                payload={},
                idempotency_key="foreign-operation",
            )
        with platform["engine"].begin() as connection:
            with pytest.raises(OperationNotFound):
                RenderRequestRepository(platform["engine"]).upsert(
                    connection,
                    page_id=platform["page_id"],
                    requested_revision=1,
                )


def test_empty_operation_claim_still_fences_an_inactive_epoch(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    with platform["engine"].begin() as connection:
        connection.execute(
            update(process_epochs)
            .where(process_epochs.c.id == platform["api_epoch_id"])
            .values(status="lost")
        )

    with pytest.raises(OperationFenced):
        repository.claim_next(
            executor_role="api",
            executor_epoch_id=platform["api_epoch_id"],
            allowed_kinds=("bubble_translate",),
        )


def test_corrupt_pending_operation_is_failed_once_instead_of_repolled(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    accepted, _ = repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_ocr",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="corrupt-operation",
    )
    operation_id = str(accepted["operationId"])
    with platform["engine"].begin() as connection:
        connection.execute(
            update(operations)
            .where(operations.c.id == operation_id)
            .values(request_json="[]")
        )

    claimed = repository.claim_next(
        executor_role="worker",
        executor_epoch_id=platform["worker_epoch_id"],
        allowed_kinds=("bubble_ocr",),
    )

    assert claimed is None
    failed = repository.get(operation_id)
    assert failed["status"] == "failed"
    assert failed["request"] == {"discardedInvalidStoredRequest": True}
    assert failed["error"] == {
        "code": "OPERATION_DATA_INVALID",
        "message": "operations.request_json must contain a JSON object",
    }
    assert [
        event["type"] for event in repository.events_after(operation_id)
    ] == ["operation_failed"]

    with platform["engine"].begin() as connection:
        connection.execute(
            update(operation_events)
            .where(operation_events.c.operation_id == operation_id)
            .values(payload_json="[]")
        )
    with pytest.raises(OperationDataInvalid, match="JSON object"):
        repository.events_after(operation_id)


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
        attempt_id=str(uuid.uuid4()),
        executor_epoch_id=fence.executor_epoch_id,
        executor_role=fence.executor_role,
        owner_user_id=fence.owner_user_id,
    )
    with pytest.raises(OperationFenced):
        repository.complete(forged, result={"text": "forbidden"})
    repository.complete(fence, result={"text": "hello"})
    stored = repository.get(fence.operation_id)
    assert stored["status"] == "completed"
    assert stored["result"] == {"text": "hello"}


def test_active_page_operation_blocks_concurrent_document_mutation(
    operation_platform,
) -> None:
    platform = operation_platform
    OperationRepository(platform["engine"]).create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_ocr",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="active-operation-document-fence",
    )

    with pytest.raises(ContentConflict, match="active operation"):
        platform["content"].mutate_page_document(
            page_id=platform["page_id"],
            base_revision=1,
            mutations=[
                {
                    "op": "patch",
                    "clientMutationId": "concurrent-document-mutation",
                    "bubbleId": platform["bubble_id"],
                    "fields": {"translatedText": "must not publish"},
                }
            ],
            idempotency_key="concurrent-document-mutation",
        )


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
                        _bubble_payload(originalText="保留原文")
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
        "author": "tests",
        "description": "operation OCR mutation",
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
                "results": [
                    {
                        "text": "こんにちは",
                        "confidence": 1.0,
                        "confidenceSupported": True,
                        "engine": "test",
                        "primaryEngine": "test",
                        "fallbackUsed": False,
                    }
                ],
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


def test_lost_executor_epoch_fences_all_late_operation_writes(
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
    with platform["engine"].begin() as connection:
        connection.execute(
            update(process_epochs)
            .where(process_epochs.c.id == platform["worker_epoch_id"])
            .values(status="lost")
        )
    with pytest.raises(OperationFenced):
        repository.complete(fence, result={"late": True})
    assert repository.get(str(accepted["operationId"]))["status"] == "running"


def test_operation_creation_obeys_revision_and_chapter_write_lock(
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
                latest_progress_json=_stored_job_progress("queued"),
            )
        )
        connection.execute(
            insert(chapter_write_locks).values(
                chapter_id=platform["chapter"]["id"],
                job_id=job_id,
            )
        )
    with pytest.raises(OperationLocked, match="chapter_locked"):
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
                .values(payload_json=json.dumps(_bubble_payload(originalText="source")))
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
        render_revision = connection.execute(
            select(render_requests.c.requested_revision).where(
                render_requests.c.page_id == platform["page_id"]
            )
        ).scalar_one()
    assert payload["translatedText"] == "译文"
    assert revision == 2
    assert render_revision == 2
    assert repository.get(str(accepted["operationId"]))["status"] == "completed"


def test_single_bubble_operation_advances_every_bubble_revision(
    operation_platform,
) -> None:
    platform = operation_platform
    second_bubble_id = str(uuid.uuid4())
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
                .where(bubbles.c.id == platform["bubble_id"])
                .values(payload_json=json.dumps(_bubble_payload(originalText="source")))
        )
        connection.execute(
            insert(bubbles).values(
                id=second_bubble_id,
                page_id=platform["page_id"],
                ordinal=2,
                payload_json=json.dumps(
                    _bubble_payload(
                        coords=[24, 24, 40, 40],
                        fillColor="#ffffff",
                        originalText="untouched",
                    )
                ),
                updated_revision=1,
            )
        )

    repository = OperationRepository(platform["engine"])
    repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_translate",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="translate-one-of-two-bubbles",
    )

    class TranslateOnlyAlgorithms:
        def translate(self, texts, _config, *, mode):
            assert texts == ["source"]
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
    service.handle(*claimed)

    with platform["engine"].connect() as connection:
        revision = connection.execute(
            select(pages.c.document_revision).where(
                pages.c.id == platform["page_id"]
            )
        ).scalar_one()
        bubble_revisions = list(
            connection.execute(
                select(bubbles.c.updated_revision)
                .where(bubbles.c.page_id == platform["page_id"])
                .order_by(bubbles.c.ordinal)
            ).scalars()
        )
    assert revision == 2
    assert bubble_revisions == [2, 2]


def test_single_bubble_ocr_rejects_missing_detail_result(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    repository.create_page_operation(
        page_id=platform["page_id"],
        kind="bubble_ocr",
        base_revision=1,
        bubble_id=platform["bubble_id"],
        payload={},
        idempotency_key="ocr-missing-detail",
    )

    class MissingDetailAlgorithms:
        def ocr(self, _image, _bubbles, _config):
            return {"texts": ["recognized"], "results": []}

    service = InteractivePageOperationService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
        algorithms=MissingDetailAlgorithms(),
    )
    claimed = repository.claim_next(
        executor_role="worker",
        executor_epoch_id=platform["worker_epoch_id"],
        allowed_kinds=("bubble_ocr",),
    )
    assert claimed is not None
    with pytest.raises(RuntimeError, match="invalid result count"):
        service.handle(*claimed)


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
            "providerMessage": f"upstream echoed {canary}",
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


def test_superseded_render_prepare_defers_to_repository_coalescing(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = RenderRequestRepository(platform["engine"])
    service = AuthoritativeRenderService(
        data_root=platform["data_root"],
        engine=platform["engine"],
    )
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

    publisher = service.prepare(first)
    assert not repository.complete(first, publisher=publisher)
    second = repository.claim_next(api_epoch_id=platform["api_epoch_id"])
    assert second is not None
    assert second.rendering_revision == 2


def test_non_rendering_page_edit_advances_an_existing_render_chain(
    operation_platform,
) -> None:
    platform = operation_platform
    renders = RenderRequestRepository(platform["engine"])
    with immediate_transaction(platform["engine"]) as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(render_status="stale")
        )
        request_id = renders.upsert(
            connection,
            page_id=platform["page_id"],
            requested_revision=1,
        )

    platform["content"].mutate_page_document(
        page_id=platform["page_id"],
        base_revision=1,
        mutations=[
            {
                "op": "patch",
                "clientMutationId": "non-rendering-edit",
                "bubbleId": platform["bubble_id"],
                "fields": {"fillColor": "#112233"},
            }
        ],
        idempotency_key="non-rendering-edit",
    )

    with platform["engine"].connect() as connection:
        render = connection.execute(
            select(
                render_requests.c.status,
                render_requests.c.requested_revision,
            ).where(render_requests.c.id == request_id)
        ).one()
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
    assert render == ("pending", 2)
    assert page == (2, "stale")


def test_old_render_failure_preserves_the_newer_coalesced_request(
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
        repository.upsert(
            connection,
            page_id=platform["page_id"],
            requested_revision=2,
        )

    repository.fail(first, code="OLD_RENDER_FAILED", message="old attempt")

    with platform["engine"].connect() as connection:
        render = connection.execute(
            select(
                render_requests.c.status,
                render_requests.c.requested_revision,
                render_requests.c.error_json,
            ).where(render_requests.c.id == request_id)
        ).one()
        page_status = connection.execute(
            select(pages.c.render_status).where(
                pages.c.id == platform["page_id"]
            )
        ).scalar_one()
    assert render == ("pending", 2, None)
    assert page_status == "stale"
    second = repository.claim_next(api_epoch_id=platform["api_epoch_id"])
    assert second is not None
    assert second.rendering_revision == 2


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


def test_bubble_repair_rejects_incomplete_legacy_payload(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    service = PageRepairService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
    )
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == platform["bubble_id"])
            .values(payload_json=json.dumps({"coords": [4, 4, 20, 20]}))
        )

    with pytest.raises(ValueError, match="missing fields"):
        service.create_for_bubble(
            page_id=platform["page_id"],
            bubble_id=platform["bubble_id"],
            base_revision=1,
            idempotency_key="repair-rejects-legacy-payload",
        )
    with platform["engine"].connect() as connection:
        assert connection.execute(select(operations.c.id)).all() == []


def test_bubble_repair_rejects_empty_geometry_without_creating_operation(
    operation_platform,
) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    service = PageRepairService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
    )
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
                .where(bubbles.c.id == platform["bubble_id"])
                .values(
                    payload_json=json.dumps(
                        _bubble_payload(coords=[4, 4, 4, 20])
                    )
                )
        )

    with pytest.raises(ValueError, match="positive-area"):
        service.create_for_bubble(
            page_id=platform["page_id"],
            bubble_id=platform["bubble_id"],
            base_revision=1,
            idempotency_key="repair-empty-geometry",
        )
    with platform["engine"].connect() as connection:
        assert connection.execute(select(operations.c.id)).all() == []


def test_page_repair_mask_requires_and_accepts_the_browser_grayscale_contract(
    operation_platform,
) -> None:
    platform = operation_platform
    service = PageRepairService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=OperationRepository(platform["engine"]),
    )
    rgba_bytes = BytesIO()
    with Image.new("RGBA", (64, 64), (255, 255, 255, 255)) as rgba_mask:
        rgba_mask.save(rgba_bytes, format="PNG")

    with pytest.raises(
        ValueError,
        match="single-frame 8-bit grayscale PNG",
    ):
        service.create_for_mask(
            page_id=platform["page_id"],
            upload=BytesIO(rgba_bytes.getvalue()),
            base_revision=1,
            method="restore_source",
            fill_color=None,
            idempotency_key="rgba-mask",
        )
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(pages.c.document_revision).where(
                pages.c.id == platform["page_id"]
            )
        ).scalar_one() == 1

    grayscale_bytes = BytesIO()
    with Image.new("L", (64, 64), 255) as grayscale_mask:
        grayscale_mask.save(grayscale_bytes, format="PNG")
    accepted, replayed = service.create_for_mask(
        page_id=platform["page_id"],
        upload=BytesIO(grayscale_bytes.getvalue()),
        base_revision=1,
        method="restore_source",
        fill_color=None,
        idempotency_key="grayscale-mask",
    )

    assert not replayed
    assert accepted["documentRevision"] == 2


def test_page_repair_rejects_non_contract_fill_color_before_image_work(
    operation_platform,
) -> None:
    platform = operation_platform
    service = PageRepairService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=OperationRepository(platform["engine"]),
    )

    with pytest.raises(ValueError, match="#RRGGBB"):
        service.create_for_mask(
            page_id=platform["page_id"],
            upload=BytesIO(b"not-read"),
            base_revision=1,
            method="solid",
            fill_color="white",
            idempotency_key="invalid-fill-color",
        )


@pytest.mark.parametrize("method", ["lama_mpe", "litelama", "restore_source"])
def test_non_solid_page_repair_rejects_unused_fill_color(
    operation_platform,
    method: str,
) -> None:
    platform = operation_platform
    service = PageRepairService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=OperationRepository(platform["engine"]),
    )

    with pytest.raises(ValueError, match="does not accept fillColor"):
        service.create_for_mask(
            page_id=platform["page_id"],
            upload=BytesIO(b"not-read"),
            base_revision=1,
            method=method,
            fill_color="#112233",
            idempotency_key=f"unused-fill-color-{method}",
        )


def test_lama_page_repair_freezes_and_consumes_disable_resize(
    operation_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = operation_platform
    with platform["engine"].begin() as connection:
        settings_payload = json.loads(
            connection.execute(
                select(app_settings.c.payload_json).where(
                    app_settings.c.domain == "translation"
                )
            ).scalar_one()
        )
        settings_payload["lamaDisableResize"] = True
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "translation")
            .values(payload_json=json.dumps(settings_payload))
        )
        bubble_payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.id == platform["bubble_id"]
                )
            ).scalar_one()
        )
        bubble_payload["inpaintMethod"] = "lama_mpe"
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == platform["bubble_id"])
            .values(payload_json=json.dumps(bubble_payload))
        )

    captured: dict[str, object] = {}

    def fake_inpaint(image, _coords, **kwargs):
        captured.update(kwargs)
        return image.copy()

    from src.core import inpainting

    monkeypatch.setattr(inpainting, "inpaint_bubbles", fake_inpaint)
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
        idempotency_key="repair-lama-disable-resize",
    )
    assert not replayed

    claimed = repository.claim_next(
        executor_role="worker",
        executor_epoch_id=platform["worker_epoch_id"],
        allowed_kinds=("page_repair",),
    )
    assert claimed is not None
    fence, operation = claimed
    assert operation["request"]["disableResize"] is True
    assert operation["request"]["settingsSnapshot"]["appRevision"] == 1
    assert "fillColor" not in operation["request"]
    result = service.handle(fence, operation)

    assert result["documentRevision"] == accepted["documentRevision"]
    assert captured["disable_resize"] is True


def test_failed_page_repair_sets_explicit_page_state(operation_platform) -> None:
    platform = operation_platform
    repository = OperationRepository(platform["engine"])
    with platform["engine"].begin() as connection:
        RenderRequestRepository(platform["engine"]).upsert(
            connection,
            page_id=platform["page_id"],
            requested_revision=1,
        )
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
        render = connection.execute(
            select(
                render_requests.c.status,
                render_requests.c.rendering_revision,
                render_requests.c.executor_epoch_id,
                render_requests.c.attempt_id,
                render_requests.c.error_json,
            ).where(render_requests.c.page_id == platform["page_id"])
        ).one()
    assert render[:4] == ("failed", None, None, None)
    assert json.loads(render.error_json)["code"] == "PAGE_REPAIR_FAILED"


def test_page_repair_closes_result_when_asset_publication_fails(
    operation_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        idempotency_key="repair-publication-failure",
    )
    claimed = repository.claim_next(
        executor_role="api",
        executor_epoch_id=platform["api_epoch_id"],
        allowed_kinds=("page_repair",),
    )
    assert claimed is not None
    fence, operation = claimed
    captured: dict[str, Image.Image] = {}

    def fail_publication(image: Image.Image):
        captured["image"] = image
        raise RuntimeError("asset publication failed")

    monkeypatch.setattr(service, "_publish_png", fail_publication)
    with pytest.raises(RuntimeError, match="asset publication failed"):
        service.handle(fence, operation)

    with pytest.raises(ValueError, match="closed image"):
        captured["image"].getpixel((0, 0))
