from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
import uuid

from PIL import Image, ImageDraw
import pytest
from sqlalchemy import select

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import (
    bubbles,
    metadata,
    page_assets,
    pages,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.translation.commands import (
    TranslationJobCommandService,
    normalize_translation_config,
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


def test_translation_snapshot_rejects_plaintext_secrets() -> None:
    with pytest.raises(ValueError, match="credentialVersionId"):
        normalize_translation_config(
            {
                "translation": {
                    "provider": "openai",
                    "apiKey": "must-not-enter-job-json",
                }
            }
        )
