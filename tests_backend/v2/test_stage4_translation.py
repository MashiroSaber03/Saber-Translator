from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from PIL import Image, ImageDraw
import pytest
from sqlalchemy import select

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
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
    bubbles,
    job_config_snapshots,
    job_credential_snapshots,
    job_items,
    job_steps,
    metadata,
    page_assets,
    pages,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.translation.commands import (
    TranslationJobCommandService,
    normalize_translation_command,
)
from src.backend_v2.translation.pipeline import TranslationPipelineService


class FakeAlgorithms:
    def detect(self, _image: Image.Image, _config: Mapping[str, Any]):
        return {
            "coords": [[5, 5, 40, 50]],
            "polygons": [[[5, 5], [40, 5], [40, 50], [5, 50]]],
            "angles": [0],
            "auto_directions": ["v"],
            "textlines_per_bubble": [[]],
            "raw_mask": Image.new("L", (64, 64), 255),
        }

    def ocr(self, _image, _payloads, _config):
        return {"texts": ["こんにちは"], "results": [{"confidence": 0.99}]}

    def colors(self, _image, _payloads):
        return [
            {
                "fg_color": [10, 20, 30],
                "bg_color": [245, 246, 247],
                "confidence": 0.9,
            }
        ]

    def translate(self, texts, _config, *, mode):
        assert texts == ["こんにちは"]
        return {"translated": ["你好"], "textbox": ["你好"], "mode": mode}

    def repair(self, image, _payloads, _config):
        return image.copy()

    def render(self, clean_image, _payloads, _config):
        rendered = clean_image.copy()
        ImageDraw.Draw(rendered).rectangle((5, 5, 10, 10), fill=(0, 0, 0))
        return rendered


@pytest.fixture()
def translation_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    SettingsRepository(engine).save_transaction(
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="siliconflow",
                secret={"api_key": "fixture-secret"},
                base_revision=0,
                client_ref="fixture-translation",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="siliconflow",
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
