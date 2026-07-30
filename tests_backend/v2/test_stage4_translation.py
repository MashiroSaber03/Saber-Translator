from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import threading
from typing import Any, Mapping
import uuid

from PIL import Image
import pytest
from sqlalchemy import select, update

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.storage.platform_repositories import (
    CredentialEdit,
    ProviderSettingMutation,
    SettingMutation,
    SettingsRepository,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import default_translation_settings
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import (
    app_settings,
    assets,
    bubbles,
    job_config_snapshots,
    job_credential_snapshots,
    job_items,
    job_step_asset_outputs,
    job_steps,
    jobs,
    metadata,
    page_assets,
    pages,
    translation_constraints,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.translation.commands import (
    TranslationJobCommandService,
    normalize_translation_command,
)
from src.backend_v2.translation.pipeline import (
    LegacyTranslationAlgorithms,
    TranslationPipelineService,
    _validate_stable_batch_result,
)
from src.backend_v2.testing.fake_provider import (
    DETERMINISTIC_FAKE_PROVIDER_ID,
    DeterministicFakeProvider,
    registered_deterministic_fake_provider,
)


class FakeAlgorithms(DeterministicFakeProvider):
    """Compatibility alias for failure-injection tests in this module."""


class PageStyleRecordingAlgorithms(FakeAlgorithms):
    def __init__(self) -> None:
        super().__init__()
        self.repair_configs: list[dict[str, Any]] = []
        self.repair_masks: list[tuple[str, tuple[int, int], int] | None] = []

    def repair(self, image, payloads, config, *, precise_mask=None):
        self.repair_configs.append(dict(config))
        self.repair_masks.append(
            (
                precise_mask.mode,
                precise_mask.size,
                int(precise_mask.getpixel((0, 0))),
            )
            if precise_mask is not None
            else None
        )
        return super().repair(
            image,
            payloads,
            config,
            precise_mask=precise_mask,
        )


class ConstraintAwareFakeAlgorithms(FakeAlgorithms):
    def __init__(self) -> None:
        super().__init__()
        self.extract_calls: list[dict[str, Any]] = []
        self.translation_prompts: list[str] = []
        self.translation_inputs: list[list[str]] = []

    def extract_terms(self, texts, config, *, prompt):
        self.extract_calls.append(
            {
                "texts": list(texts),
                "credential": config.get("api_key"),
                "prompt": prompt,
            }
        )
        return {
            "rawContent": '[{"source":"勇者","target":"勇者"}]',
            "candidates": [{"source": "勇者", "target": "勇者"}],
        }

    def translate(self, texts, config, *, mode):
        self.translation_prompts.append(str(config.get("prompt_content", "")))
        self.translation_inputs.append(list(texts))
        if texts and "SABER_NT" in texts[0]:
            return {
                "translated": list(texts),
                "textbox": ["" for _text in texts],
                "mode": mode,
            }
        return super().translate(texts, config, mode=mode)


def test_legacy_color_adapter_accepts_serialized_dictionary_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import color_extractor

    extracted = [
        {
            "fg_color": [1, 2, 3],
            "bg_color": [250, 251, 252],
            "confidence": 0.875,
        }
    ]
    calls: list[tuple[list[list[int]], list[list[dict[str, Any]]]]] = []

    def fake_extract(_image, coords, textlines):
        calls.append((coords, textlines))
        return extracted

    monkeypatch.setattr(
        color_extractor,
        "extract_bubble_colors",
        fake_extract,
    )
    payloads = [
        {
            "coords": [3, 4, 20, 30],
            "textlines": [{"polygon": [[3, 4], [20, 4]], "direction": "h"}],
        }
    ]

    with Image.new("RGB", (32, 32), "white") as image:
        result = LegacyTranslationAlgorithms().colors(image, payloads)

    assert result == extracted
    assert result is not extracted
    assert result[0] is not extracted[0]
    assert calls == [
        (
            [[3, 4, 20, 30]],
            [[{"polygon": [[3, 4], [20, 4]], "direction": "h"}]],
        )
    ]


def test_legacy_repair_adapter_passes_precise_text_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import inpainting

    captured: dict[str, Any] = {}

    def fake_inpaint(image, _coords, **kwargs):
        captured.update(kwargs)
        return image.copy(), None

    monkeypatch.setattr(inpainting, "inpaint_bubbles", fake_inpaint)
    image = Image.new("RGB", (3, 2), "white")
    precise_mask = Image.new("L", (3, 2), 0)
    precise_mask.putpixel((1, 0), 255)

    repaired = LegacyTranslationAlgorithms().repair(
        image,
        [{"coords": [0, 0, 3, 2], "polygon": []}],
        {"method": "solid"},
        precise_mask=precise_mask,
    )

    assert captured["precise_mask"].tolist() == [
        [0, 255, 0],
        [0, 0, 0],
    ]
    repaired.close()
    precise_mask.close()
    image.close()


@pytest.fixture(autouse=True)
def deterministic_provider_registration():
    with registered_deterministic_fake_provider():
        yield


@pytest.fixture()
def translation_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    settings_payload = default_translation_settings()
    settings_payload["translation"] = {
        **settings_payload["translation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
    }
    with engine.begin() as connection:
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "translation")
            .values(
                payload_json=json.dumps(
                    settings_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
    SettingsRepository(engine).save_transaction(
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": "fixture-secret"},
                base_revision=0,
                client_ref="fixture-translation",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={"modelName": "fixture-model"},
                base_revision=0,
                credential_edit_ref="fixture-translation",
            ),
        ),
    )
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(book_id=str(book["id"]), title="Chapter")
    storage = AssetStorageService(data_root, engine)
    importer = ImageImportService(
        data_root=data_root,
        repository=content,
        storage=storage,
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (255, 255, 255)) as image:
        image.save(payload, format="PNG")
    lease = content.create_import_lease(str(chapter["id"]))
    try:
        imported, _ = importer.import_page(
            chapter_id=str(chapter["id"]),
            logical_path="page.png",
            upload=BytesIO(payload.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key="page",
        )
    finally:
        content.release_import_lease(
            chapter_id=str(chapter["id"]),
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 555)
    )
    try:
        yield {
            "data_root": data_root,
            "engine": engine,
            "book": book,
            "chapter": chapter,
            "page_id": str(imported["page"]["id"]),
            "epoch_id": epoch_id,
        }
    finally:
        engine.dispose()


def test_translation_job_executes_all_steps_and_publishes_each_page(
    translation_platform,
) -> None:
    platform = translation_platform
    command = TranslationJobCommandService(platform["engine"])
    accepted = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="translation-1",
    )
    replay = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="translation-1",
    )
    assert replay == accepted

    repository = JobQueueRepository(platform["engine"])
    bootstrap = ContentRepository(platform["engine"]).translation_bootstrap(
        book_id=str(platform["book"]["id"]),
        chapter_id=str(platform["chapter"]["id"]),
    )
    assert bootstrap["activeJobs"] == [
        {
            "id": accepted["jobIds"][0],
            "kind": "translation",
            "status": "queued",
            "queueRank": 1,
            "pageIds": [platform["page_id"]],
            "progress": {
                "executionMode": "parallel",
                "jobStatus": "queued",
                "totalItems": 1,
                "completedItems": 0,
                "failedItems": 0,
                "skippedItems": 0,
                "cancelledItems": 0,
                "pools": [
                    {
                        "kind": kind,
                        "total": 1,
                        "completed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "waiting": 1,
                        "processing": 0,
                        "lockWaiting": False,
                        "current": [],
                    }
                    for kind in (
                        "detect",
                        "ocr",
                        "color",
                        "auto_terms",
                        "translate",
                        "repair",
                        "render",
                        "save",
                    )
                ],
            },
        }
    ]
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=FakeAlgorithms(),
    )
    assert repository.claim_next(worker_epoch_id=platform["epoch_id"]) is None
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := repository.next_step(fence)) is not None:
        result = service.handler(fence, step)
        assert result["__already_published__"]
        if step["stepKind"] == "render":
            with platform["engine"].connect() as connection:
                assert connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == platform["page_id"],
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none() is None
                assert set(
                    connection.execute(
                        select(job_step_asset_outputs.c.role).where(
                            job_step_asset_outputs.c.job_step_id
                            == step["stepId"]
                        )
                    ).scalars()
                ) == {"translated", "thumbnail_translated"}
        if step["stepKind"] == "save":
            with platform["engine"].connect() as connection:
                assert connection.execute(
                    select(page_assets.c.producer_job_step_id).where(
                        page_assets.c.page_id == platform["page_id"],
                        page_assets.c.role == "translated",
                    )
                ).scalar_one() == step["stepId"]
    assert repository.finish_if_complete(fence) == "completed"

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        payload = connection.execute(
            select(bubbles.c.payload_json).where(
                bubbles.c.page_id == platform["page_id"]
            )
        ).scalar_one()
        roles = set(
            connection.execute(
                select(page_assets.c.role).where(
                    page_assets.c.page_id == platform["page_id"]
                )
            ).scalars()
        )
    assert page == (5, 5, "ready")
    assert '"originalText":"こんにちは"' in payload
    assert '"translatedText":"你好"' in payload
    assert {
        "source",
        "thumbnail_source",
        "text_mask",
        "clean",
        "translated",
        "thumbnail_translated",
    }.issubset(roles)


def test_translation_uses_current_page_layout_and_inpainting_defaults(
    translation_platform,
) -> None:
    platform = translation_platform
    page_style = {
        "fontSize": 26,
        "autoFontSize": True,
        "fontFamily": "00000000-0000-0000-0000-000000000010",
        "layoutDirection": "horizontal",
        "textColor": "#000000",
        "fillColor": "#123456",
        "inpaintMethod": "litelama",
        "useAutoTextColor": False,
        "strokeEnabled": True,
        "strokeColor": "#FFFFFF",
        "strokeWidth": 3,
        "lineSpacing": 1.0,
        "textAlign": "start",
    }
    with platform["engine"].begin() as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json=json.dumps(page_style))
        )

    TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=[platform["page_id"]],
        idempotency_key="page-style-translation",
    )
    algorithms = PageStyleRecordingAlgorithms()
    _run_translation_job(platform, algorithms)

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )

    assert payload["textDirection"] == "horizontal"
    assert payload["autoTextDirection"] == "vertical"
    assert algorithms.repair_configs == [
        {
            "method": "lama",
            "lama_model": "litelama",
            "fill_color": "#123456",
            "mask_dilate_size": 10,
            "mask_box_expand_ratio": 20,
        }
    ]
    assert algorithms.repair_masks == [("L", (64, 64), 255)]


def test_translation_constraints_are_frozen_extracted_and_consumed(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    saved = content.update_constraints(
        book_id=str(platform["book"]["id"]),
        base_revision=1,
        payload={
            "glossary": {
                "enabled": True,
                "autoExtractEnabled": True,
                "autoExtractPrompt": "从 {ocr_text} 提取术语",
                "entries": [
                    {
                        "source": "騎士",
                        "target": "骑士",
                        "note": "固定译名",
                        "matchMode": "text",
                    },
                    {
                        "source": "こんにちは",
                        "target": "固定问候",
                        "note": "用于检查告警",
                        "matchMode": "text",
                    },
                ],
            },
            "nonTranslate": {
                "enabled": True,
                "entries": [
                    {
                        "pattern": "Excalibur",
                        "note": "保留英文",
                        "matchMode": "text",
                    },
                    {
                        "pattern": "こんにちは",
                        "note": "保护 OCR 原文",
                        "matchMode": "text",
                    },
                ],
            },
        },
    )
    assert saved["revision"] == 2
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="constraints-job",
    )
    job_id = str(accepted["jobIds"][0])
    algorithms = ConstraintAwareFakeAlgorithms()
    assert _run_translation_job(platform, algorithms) == job_id

    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(job_config_snapshots.c.payload_json).where(
                    job_config_snapshots.c.job_id == job_id
                )
            ).scalar_one()
        )
        auto_checkpoint = json.loads(
            connection.execute(
                select(job_steps.c.checkpoint_json)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == job_id,
                    job_steps.c.kind == "auto_terms",
                )
            ).scalar_one()
        )
        translate_checkpoint = json.loads(
            connection.execute(
                select(job_steps.c.checkpoint_json)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == job_id,
                    job_steps.c.kind == "translate",
                )
            ).scalar_one()
        )
        bubble_payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )
        current_constraints = connection.execute(
            select(
                translation_constraints.c.revision,
                translation_constraints.c.payload_json,
            ).where(
                translation_constraints.c.book_id == str(platform["book"]["id"])
            )
        ).mappings().one()

    assert frozen["translationConstraintRevision"] == 2
    assert frozen["translationConstraints"]["glossary"]["entries"] == [
        {
            "source": "騎士",
            "target": "骑士",
            "note": "固定译名",
            "matchMode": "text",
        },
        {
            "source": "こんにちは",
            "target": "固定问候",
            "note": "用于检查告警",
            "matchMode": "text",
        },
    ]
    assert auto_checkpoint["candidateCount"] == 1
    assert auto_checkpoint["duplicateCount"] == 0
    assert auto_checkpoint["addedCount"] == 1
    assert auto_checkpoint["delta"] == [
        {
            "source": "勇者",
            "target": "勇者",
            "note": "",
            "matchMode": "text",
        }
    ]
    assert algorithms.extract_calls == [
        {
            "texts": ["こんにちは"],
            "credential": "fixture-secret",
            "prompt": "从 {ocr_text} 提取术语",
        }
    ]
    assert len(algorithms.translation_prompts) == 1
    assert len(algorithms.translation_inputs) == 1
    assert "SABER_NT" in algorithms.translation_inputs[0][0]
    assert "こんにちは" not in algorithms.translation_inputs[0][0]
    translated_prompt = algorithms.translation_prompts[0]
    assert '"source":"騎士"' in translated_prompt
    assert '"source":"勇者"' in translated_prompt
    assert '"pattern":"Excalibur"' in translated_prompt
    assert bubble_payload["translatedText"] == "こんにちは"
    assert bubble_payload["translationWarnings"] == [
        {
            "bubbleIndex": 0,
            "source": "こんにちは",
            "expectedTarget": "固定问候",
            "actualTranslation": "こんにちは",
        }
    ]
    assert translate_checkpoint["constraintWarnings"] == [
        {
            "bubbleIndex": 0,
            "source": "こんにちは",
            "expectedTarget": "固定问候",
            "actualTranslation": "こんにちは",
        }
    ]
    assert current_constraints["revision"] == 3
    assert [
        entry["source"]
        for entry in json.loads(current_constraints["payload_json"])["glossary"][
            "entries"
        ]
    ] == ["騎士", "こんにちは", "勇者"]


def test_multi_chapter_batch_creates_eligible_jobs_and_reports_skips(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    eligible = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Eligible",
    )
    empty = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Empty",
    )
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(
            platform["data_root"],
            platform["engine"],
        ),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (240, 240, 240)) as image:
        image.save(payload, format="PNG")
    lease = content.create_import_lease(str(eligible["id"]))
    try:
        importer.import_page(
            chapter_id=str(eligible["id"]),
            logical_path="eligible.png",
            upload=BytesIO(payload.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key="eligible-page",
        )
    finally:
        content.release_import_lease(
            chapter_id=str(eligible["id"]),
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )

    commands = TranslationJobCommandService(platform["engine"])
    occupied = commands.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard"},
        page_ids=None,
        idempotency_key="occupied",
    )
    accepted = commands.create_batch(
        chapter_ids=[
            str(platform["chapter"]["id"]),
            str(eligible["id"]),
            str(empty["id"]),
        ],
        config={"mode": "standard"},
        idempotency_key="partial-batch",
    )
    replay = commands.create_batch(
        chapter_ids=[
            str(platform["chapter"]["id"]),
            str(eligible["id"]),
            str(empty["id"]),
        ],
        config={"mode": "standard"},
        idempotency_key="partial-batch",
    )

    assert replay == accepted
    assert len(accepted["jobIds"]) == 1
    assert accepted["jobIds"][0] != occupied["jobIds"][0]
    assert accepted["skipped"] == [
        {
            "chapterId": str(platform["chapter"]["id"]),
            "reason": "active_job",
            "message": "章节已有未结束的同类任务",
        },
        {
            "chapterId": str(empty["id"]),
            "reason": "empty_chapter",
            "message": "translation task requires at least one page",
        },
    ]


def test_translation_command_rejects_browser_supplied_provider_config() -> None:
    with pytest.raises(ValueError, match="unknown translation config fields"):
        normalize_translation_command(
            {
                "translation": {
                    "provider": "openai",
                    "apiKey": "must-not-enter-job-json",
                }
            }
        )


def test_translation_job_rejects_missing_backend_credential_before_admission(
    translation_platform,
) -> None:
    platform = translation_platform
    payload = default_translation_settings()
    payload["translation"] = {
        **payload["translation"],
        "provider": "custom",
        "modelName": "must-not-be-trusted-from-app-payload",
        "customBaseUrl": "https://custom.example/v1",
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "custom-model",
                    "customBaseUrl": "https://custom.example/v1",
                },
                base_revision=0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="缺少已保存的 API Key"):
        TranslationJobCommandService(platform["engine"]).create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={"mode": "standard"},
            page_ids=None,
            idempotency_key="missing-credential",
        )

    assert JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"] == []


def test_failed_item_retry_refreezes_current_backend_settings(
    translation_platform,
) -> None:
    platform = translation_platform
    source = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="retry-source",
    )
    source_id = str(source["jobIds"][0])
    with platform["engine"].begin() as connection:
        connection.execute(
            update(job_items)
            .where(job_items.c.job_id == source_id)
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        connection.execute(
            update(job_steps)
            .where(
                job_steps.c.job_item_id.in_(
                    select(job_items.c.id).where(job_items.c.job_id == source_id)
                )
            )
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.id == source_id)
            .values(status="completed_with_errors", queue_rank=None)
        )

    settings_payload = default_translation_settings()
    settings_payload["translation"] = {
        **settings_payload["translation"],
        "provider": "custom",
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=settings_payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "new-backend-only-key"},
                base_revision=0,
                client_ref="retry-custom",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "retry-current-model",
                    "customBaseUrl": "https://retry.example/v1",
                },
                base_revision=0,
                credential_edit_ref="retry-custom",
            ),
        ),
    )

    retried = JobRetryService(platform["engine"]).retry(
        job_id=source_id,
        failed_only=True,
        strategy="current",
        idempotency_key="retry-current",
    )
    replacement_id = str(retried["jobIds"][0])
    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(job_config_snapshots.c.payload_json).where(
                    job_config_snapshots.c.job_id == replacement_id
                )
            ).scalar_one()
        )
    detail = JobQueueRepository(platform["engine"]).get_job(replacement_id)
    assert frozen["translation"]["provider"] == "custom"
    assert frozen["translation"]["model_name"] == "retry-current-model"
    assert "new-backend-only-key" not in json.dumps(frozen)
    assert detail["retryOfJobId"] == source_id
    assert detail["retryMode"] == "current"
    assert detail["configSummary"]["translation"]["model"] == "retry-current-model"


def test_translation_job_resolves_backend_settings_and_reuses_manual_bubbles(
    translation_platform,
) -> None:
    platform = translation_platform
    settings = SettingsRepository(platform["engine"])
    settings.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload={
                    "settingsSchemaVersion": 3,
                    "sourceLanguage": "japanese",
                    "targetLanguage": "zh",
                    "parallel": {
                        "enabled": True,
                        "deepLearningLockSize": 3,
                    },
                    "translation": {
                        "provider": "custom",
                        "translationMode": "batch",
                        "batchNormalPrompt": "backend prompt",
                        "batchJsonPrompt": "backend json prompt",
                        "singleNormalPrompt": "single",
                        "singleJsonPrompt": "single json",
                        "openaiOptions": {
                            "request": {"forceJsonOutput": False},
                            "execution": {},
                        },
                    },
                },
                base_revision=1,
                schema_version=3,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "backend-only-secret"},
                base_revision=0,
                client_ref="translation",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "backend-model",
                    "customBaseUrl": "https://backend.example/v1",
                },
                base_revision=0,
                credential_edit_ref="translation",
            ),
        ),
    )
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={
            "mode": "standard",
            "executionMode": "sequential",
            "reuseExistingBubbles": True,
        },
        page_ids=[platform["page_id"]],
        idempotency_key="manual-bubbles",
    )
    job_id = str(accepted["jobIds"][0])
    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(job_config_snapshots.c.payload_json).where(
                    job_config_snapshots.c.job_id == job_id
                )
            ).scalar_one()
        )
        steps = list(
            connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_steps.c.ordinal)
            ).scalars()
        )
        credential_count = len(
            connection.execute(
                select(job_credential_snapshots.c.credential_version_id).where(
                    job_credential_snapshots.c.job_id == job_id
                )
            ).scalars().all()
        )
    assert frozen["translation"]["model_name"] == "backend-model"
    assert frozen["translation"]["prompt_content"] == "backend prompt"
    assert frozen["deepLearningConcurrency"] == 3
    assert "backend-only-secret" not in json.dumps(frozen)
    assert steps[0] == "ocr"
    assert "detect" not in steps
    assert credential_count == 1


def _import_extra_page(
    platform: Mapping[str, Any],
    name: str,
    *,
    chapter_id: str | None = None,
) -> str:
    content = ContentRepository(platform["engine"])
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(platform["data_root"], platform["engine"]),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (255, 255, 255)) as image:
        image.save(payload, format="PNG")
    target_chapter_id = chapter_id or str(platform["chapter"]["id"])
    lease = content.create_import_lease(target_chapter_id)
    try:
        imported, _ = importer.import_page(
            chapter_id=target_chapter_id,
            logical_path=name,
            upload=BytesIO(payload.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key=f"import-{name}",
        )
    finally:
        content.release_import_lease(
            chapter_id=target_chapter_id,
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )
    return str(imported["page"]["id"])


def _configure_hq_and_proofreading(platform: Mapping[str, Any]) -> None:
    payload = default_translation_settings()
    payload["hqTranslation"] = {
        **payload["hqTranslation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
        "modelName": "hq-model",
        "batchSize": 2,
    }
    payload["proofreading"] = {
        "enabled": True,
        "maxRetries": 4,
        "rounds": [
            {
                **payload["hqTranslation"],
                "name": "准确性",
                "modelName": "proof-model-1",
                "batchSize": 2,
            },
            {
                **payload["hqTranslation"],
                "name": "润色",
                "modelName": "proof-model-2",
                "batchSize": 1,
            },
        ],
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        credentials_edits=tuple(
            CredentialEdit(
                domain=domain,
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": f"{domain}-secret"},
                base_revision=0,
                client_ref=domain,
            )
            for domain in ("hq", "proofreading_0", "proofreading_1")
        ),
        providers=tuple(
            ProviderSettingMutation(
                domain=domain,
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={"modelName": model},
                base_revision=0,
                credential_edit_ref=domain,
            )
            for domain, model in (
                ("hq", "hq-model"),
                ("proofreading_0", "proof-model-1"),
                ("proofreading_1", "proof-model-2"),
            )
        ),
    )


def _run_translation_job(
    platform: Mapping[str, Any],
    algorithms: FakeAlgorithms,
) -> str:
    repository = JobQueueRepository(platform["engine"])
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=algorithms,
    )
    handlers = {
        kind: service.handler
        for kind in (
            "detect",
            "ocr",
            "color",
            "auto_terms",
            "translate",
            "hq_translate",
            "proofread",
            "repair",
            "render",
            "save",
            "publish_clean",
        )
    }
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    JobWorkerLoop(
        repository,
        worker_epoch_id=platform["epoch_id"],
        handlers=handlers,
        batch_handlers={
            "hq_translate": service.batch_handler,
            "proofread": service.batch_handler,
        },
    )._run_attempt(fence, threading.Event())
    return fence.job_id


def _translation_result_snapshot(
    platform: Mapping[str, Any],
    *,
    page_id: str,
    job_id: str,
) -> dict[str, Any]:
    with platform["engine"].connect() as connection:
        document = connection.execute(
            select(
                pages.c.source_revision,
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == page_id)
        ).one()
        bubble_payloads = [
            json.loads(payload)
            for payload in connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).scalars()
        ]
        asset_structure = [
            tuple(row)
            for row in connection.execute(
                select(
                    page_assets.c.role,
                    assets.c.mime_type,
                    assets.c.width,
                    assets.c.height,
                )
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(page_assets.c.page_id == page_id)
                .order_by(page_assets.c.role)
            )
        ]
        step_sequence = list(
            connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_items.c.ordinal, job_steps.c.ordinal)
            ).scalars()
        )
    return {
        "document": tuple(document),
        "bubbles": bubble_payloads,
        "assets": asset_structure,
        "steps": step_sequence,
    }


def test_sequential_and_parallel_pipeline_results_are_equivalent(
    translation_platform,
) -> None:
    platform = translation_platform
    commands = TranslationJobCommandService(platform["engine"])
    sequential = commands.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="equivalence-sequential",
    )
    sequential_job_id = _run_translation_job(platform, FakeAlgorithms())
    assert sequential_job_id == sequential["jobIds"][0]
    sequential_result = _translation_result_snapshot(
        platform,
        page_id=platform["page_id"],
        job_id=sequential_job_id,
    )

    content = ContentRepository(platform["engine"])
    parallel_chapter = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Parallel equivalent",
    )
    parallel_page_id = _import_extra_page(
        platform,
        "parallel.png",
        chapter_id=str(parallel_chapter["id"]),
    )
    parallel = commands.create_chapter_job(
        chapter_id=str(parallel_chapter["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="equivalence-parallel",
    )
    parallel_job_id = _run_translation_job(platform, FakeAlgorithms())
    assert parallel_job_id == parallel["jobIds"][0]
    parallel_result = _translation_result_snapshot(
        platform,
        page_id=parallel_page_id,
        job_id=parallel_job_id,
    )

    assert parallel_result == sequential_result


def test_hq_and_multiround_proofreading_use_durable_stable_id_batches(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    command = TranslationJobCommandService(platform["engine"])

    hq = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-batches",
    )
    algorithms = FakeAlgorithms()
    hq_job_id = _run_translation_job(platform, algorithms)
    assert hq_job_id == hq["jobIds"][0]
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
    assert all(call["mode"] == "hq_translate" for call in algorithms.batch_calls)
    assert len(
        {
            bubble_id
            for call in algorithms.batch_calls
            for bubble_id in call["bubbleIds"]
        }
    ) == 3

    proofread = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "proofread", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="proofread-rounds",
    )
    proof_job_id = _run_translation_job(platform, algorithms)
    assert proof_job_id == proofread["jobIds"][0]
    proof_calls = [
        call for call in algorithms.batch_calls if call["mode"] == "proofread"
    ]
    assert [
        (call["model"], len(call["pageIds"]))
        for call in proof_calls
    ] == [
        ("proof-model-1", 2),
        ("proof-model-1", 1),
        ("proof-model-2", 1),
        ("proof-model-2", 1),
        ("proof-model-2", 1),
    ]

    repository = JobQueueRepository(platform["engine"])
    detail = repository.get_job(proof_job_id)
    assert detail["status"] == "completed"
    assert all(
        [step["kind"] for step in item["steps"]]
        == ["proofread", "proofread", "render", "save"]
        for item in detail["items"]
    )
    assert all(
        item["steps"][0]["checkpoint"]["roundIndex"] == 0
        and item["steps"][1]["checkpoint"]["roundIndex"] == 1
        for item in detail["items"]
    )
    with platform["engine"].connect() as connection:
        raw_assets = list(
            connection.execute(
                select(job_step_asset_outputs.c.asset_id)
                .join(job_steps, job_steps.c.id == job_step_asset_outputs.c.job_step_id)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id.in_((hq_job_id, proof_job_id)),
                    job_step_asset_outputs.c.role == "model_raw",
                )
            ).scalars()
        )
    assert len(raw_assets) == 7


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"pages": []},
        {
            "pages": [
                {"pageId": "page-1", "bubbles": []},
                {"pageId": "page-1", "bubbles": []},
            ]
        },
        {
            "pages": [
                {
                    "pageId": "page-1",
                    "bubbles": [
                        {"bubbleId": "bubble-unknown", "translatedText": "译文"}
                    ],
                }
            ]
        },
        {
            "pages": [
                {
                    "pageId": "page-1",
                    "bubbles": [
                        {"bubbleId": "bubble-1", "translatedText": "译文"},
                        {"bubbleId": "bubble-1", "translatedText": "重复"},
                    ],
                }
            ]
        },
    ],
)
def test_hq_stable_id_contract_rejects_invalid_model_results(payload) -> None:
    expected = [
        {
            "pageId": "page-1",
            "bubbles": [
                {
                    "bubbleId": "bubble-1",
                    "originalText": "原文",
                    "translatedText": "",
                }
            ],
        }
    ]
    with pytest.raises(ValueError):
        _validate_stable_batch_result(payload, expected_pages=expected)


def test_proofreading_skips_pages_without_existing_translation(
    translation_platform,
) -> None:
    platform = translation_platform
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "proofread", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="proofread-empty",
    )
    algorithms = FakeAlgorithms()
    job_id = _run_translation_job(platform, algorithms)
    assert job_id == accepted["jobIds"][0]
    assert algorithms.batch_calls == []
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed"
    assert detail["items"][0]["status"] == "skipped"
    assert all(
        step["status"] == "skipped"
        for step in detail["items"][0]["steps"]
    )


def test_hq_batch_failure_isolated_and_later_batch_continues(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-partial-failure",
    )
    algorithms = FakeAlgorithms(fail_batch_calls={1})
    job_id = _run_translation_job(platform, algorithms)
    assert job_id == accepted["jobIds"][0]
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed_with_errors"
    assert [item["status"] for item in detail["items"]] == [
        "failed",
        "failed",
        "completed",
    ]


def test_failed_hq_redetect_preserves_previous_published_text(
    translation_platform,
) -> None:
    platform = translation_platform
    command = TranslationJobCommandService(platform["engine"])
    standard = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="published-before-hq-failure",
    )
    standard_job_id = _run_translation_job(platform, FakeAlgorithms())
    assert standard_job_id == standard["jobIds"][0]

    _configure_hq_and_proofreading(platform)
    hq = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-failure-keeps-text",
    )
    hq_job_id = _run_translation_job(
        platform,
        FakeAlgorithms(fail_batch_calls={1}),
    )
    assert hq_job_id == hq["jobIds"][0]
    assert (
        JobQueueRepository(platform["engine"]).get_job(hq_job_id)["status"]
        == "completed_with_errors"
    )

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == platform["page_id"])
            ).scalar_one()
        )
    assert payload["originalText"] == "こんにちは"
    assert payload["translatedText"] == "你好"


def test_hq_pause_resume_keeps_completed_batch_checkpoint(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-pause-resume",
    )
    repository = JobQueueRepository(platform["engine"])
    job_id = str(accepted["jobIds"][0])

    def pause_after_first_batch(call_number: int) -> None:
        if call_number == 1:
            repository.request_pause(job_id)

    algorithms = FakeAlgorithms(on_batch=pause_after_first_batch)
    assert _run_translation_job(platform, algorithms) == job_id
    paused = repository.get_job(job_id)
    assert paused["status"] == "paused"
    completed_hq = [
        step
        for item in paused["items"]
        for step in item["steps"]
        if step["kind"] == "hq_translate" and step["status"] == "completed"
    ]
    assert len(completed_hq) == 2

    algorithms.on_batch = None
    repository.resume(job_id)
    assert _run_translation_job(platform, algorithms) == job_id
    assert repository.get_job(job_id)["status"] == "completed"
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
