from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import uuid

from PIL import Image, ImageFont
import pytest
from sqlalchemy import insert, select, update

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.content.image_import import ImageImportService, ImportSafetyLimits
from src.backend_v2.content.repository import (
    ContentConflict,
    ContentLocked,
    ContentRepository,
    IdempotencyConflict,
)
from src.backend_v2.content.translation_constraints import (
    empty_translation_constraints,
)
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.rendering.fonts import (
    materialize_render_payloads,
    resolve_font_path,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.builtin_fonts import (
    discover_bundled_fonts,
    resolve_bundled_font_path,
)
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import DEFAULT_FONT_ID
from src.backend_v2.storage.schema import (
    app_settings,
    assets,
    chapter_write_locks,
    chapters,
    fonts,
    jobs,
    metadata,
    operations,
    page_assets,
    pages,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.storage.seeding import QUICK_WORKSPACE_BOOK_ID


@pytest.fixture()
def content_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    repository = ContentRepository(engine)
    book = repository.create_book(title="Book")
    chapter = repository.create_chapter(book_id=str(book["id"]), title="Chapter")
    storage = AssetStorageService(data_root, engine)
    importer = ImageImportService(
        data_root=data_root,
        repository=repository,
        storage=storage,
    )
    try:
        yield data_root, engine, repository, storage, importer, book, chapter
    finally:
        engine.dispose()


def _image_bytes(
    size: tuple[int, int],
    *,
    image_format: str = "PNG",
    color: tuple[int, int, int] = (20, 40, 60),
) -> bytes:
    output = BytesIO()
    with Image.new("RGB", size, color) as image:
        image.save(output, format=image_format)
    return output.getvalue()


def test_large_image_policy_has_no_byte_or_pixel_dimension_gate() -> None:
    limits = ImportSafetyLimits()

    assert Image.MAX_IMAGE_PIXELS is None
    assert not hasattr(limits, "max_image_bytes")
    assert not hasattr(limits, "max_container_bytes")
    assert not hasattr(limits, "max_expanded_bytes")


def test_font_resolution_rejects_an_unknown_v2_font(content_platform) -> None:
    _data_root, engine, _repository, storage, _importer, _book, _chapter = (
        content_platform
    )
    with engine.connect() as connection:
        with pytest.raises(LookupError, match="font not found"):
            resolve_font_path(connection, storage, str(uuid.uuid4()))


def test_bundled_font_id_resolves_to_a_renderable_resource(content_platform) -> None:
    _data_root, engine, _repository, storage, _importer, _book, _chapter = (
        content_platform
    )
    bundled = next(
        font for font in discover_bundled_fonts() if font.file_name == "ALGER.TTF"
    )
    with engine.connect() as connection:
        resolved = resolve_font_path(connection, storage, bundled.id)

    assert Path(resolved) == bundled.path
    assert ImageFont.truetype(resolved, 16).getbbox("A") is not None


def test_bundled_font_resolver_rejects_non_catalog_paths() -> None:
    with pytest.raises(RuntimeError, match="unsupported builtin font"):
        resolve_bundled_font_path("resource:../outside.ttf")


def _import(
    repository: ContentRepository,
    importer: ImageImportService,
    *,
    chapter_id: str,
    payload: bytes,
    logical_path: str,
    key: str,
):
    lease = repository.create_import_lease(chapter_id)
    try:
        return importer.import_page(
            chapter_id=chapter_id,
            logical_path=logical_path,
            upload=BytesIO(payload),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key=key,
        )
    finally:
        repository.release_import_lease(
            chapter_id=chapter_id,
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )


def test_server_info_uses_the_configured_v2_api_port(content_platform) -> None:
    data_root, engine, *_rest = content_platform
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-api",
                epoch_token="test-token",
                test_mode=True,
            ),
            engine=engine,
            host="0.0.0.0",
            port=5123,
        )
    )

    response = app.test_client().get("/api/v2/system/server-info")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["host"] == "0.0.0.0"
    assert payload["port"] == 5123
    assert payload["lanUrl"].endswith(":5123")


def test_book_creation_rejects_retired_request_fields(content_platform) -> None:
    data_root, engine, *_rest = content_platform
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-strict-book-api",
                epoch_token="test-token",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    try:
        client = app.test_client()
        response = client.post(
            "/api/v2/books",
            headers={"Idempotency-Key": "book-with-retired-field"},
            json={"title": "Book", "description": "retired"},
        )
        multipart_response = client.post(
            "/api/v2/books",
            headers={"Idempotency-Key": "book-with-retired-form-field"},
            data={"title": "Book", "description": "retired"},
            content_type="multipart/form-data",
        )
    finally:
        app.extensions["saber_v2_runtime"].close()

    assert response.status_code == 422
    assert "description" in response.get_data(as_text=True)
    assert multipart_response.status_code == 422
    assert "form.description" in multipart_response.get_data(as_text=True)


def test_book_detail_includes_the_list_projection_timestamps(content_platform) -> None:
    _data_root, _engine, repository, _storage, _importer, book, _chapter = (
        content_platform
    )

    summary = next(
        item for item in repository.list_books() if item["id"] == book["id"]
    )
    detail = repository.get_book(str(book["id"]))

    assert detail["createdAt"] == summary["createdAt"]
    assert detail["updatedAt"] == summary["updatedAt"]


def test_page_operation_rejects_client_payload(content_platform) -> None:
    data_root, engine, repository, _storage, importer, _book, chapter = (
        content_platform
    )
    imported, _replayed = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((24, 24)),
        logical_path="strict-operation.png",
        key="strict-operation-page",
    )
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-strict-operation-api",
                epoch_token="test-token",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    try:
        response = app.test_client().post(
            f"/api/v2/pages/{imported['page']['id']}/operations",
            headers={"Idempotency-Key": "operation-with-client-payload"},
            json={
                "kind": "page_ocr",
                "baseRevision": 1,
                "payload": {},
            },
        )
    finally:
        app.extensions["saber_v2_runtime"].close()

    assert response.status_code == 422
    assert "payload" in response.get_data(as_text=True)


def test_translation_bootstrap_includes_backend_owned_runtime_configuration(
    content_platform,
) -> None:
    data_root, engine, _repository, _storage, _importer, book, chapter = (
        content_platform
    )
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-bootstrap-api",
                epoch_token="test-token",
                test_mode=True,
            ),
            engine=engine,
        )
    )

    response = app.test_client().get(
        "/api/v2/translation/bootstrap",
        query_string={
            "bookId": str(book["id"]),
            "chapterId": str(chapter["id"]),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["book"]["id"] == str(book["id"])
    assert payload["chapter"]["id"] == str(chapter["id"])

    settings_by_domain = {
        item["domain"]: item for item in payload["settings"]["settings"]
    }
    assert {
        "translation",
        "text_style_defaults",
        "workflow_preferences",
    } <= settings_by_domain.keys()
    translation = settings_by_domain["translation"]
    assert translation["schemaVersion"] == 3
    assert translation["revision"] == 1
    assert translation["payload"]["settingsSchemaVersion"] == 3
    assert translation["payload"]["translation"]["provider"]
    assert "textStyle" not in translation["payload"]
    assert translation["payload"]["pluginAgent"]["provider"]
    assert settings_by_domain["text_style_defaults"]["payload"]["fontFamily"]

    workflow = settings_by_domain["workflow_preferences"]["payload"]
    assert workflow == {
        "rememberWorkflowModeEnabled": False,
        "lastWorkflowMode": "translate-current",
    }
    assert payload["fonts"] == [
        {
            "assetUrl": None,
            "builtinKey": font.builtin_key,
            "displayName": font.display_name,
            "id": font.id,
            "kind": "builtin",
        }
        for font in discover_bundled_fonts()
    ]
    assert {item["type"] for item in payload["prompts"]} == {
        "translate",
        "textbox",
    }
    assert all(item["isFactoryDefault"] for item in payload["prompts"])


def test_chapter_settings_memory_is_cas_scoped_and_rejects_style_or_secrets(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, _importer, book, chapter = (
        content_platform
    )
    chapter_id = str(chapter["id"])
    payload = {
        "sourceLanguage": "english",
        "targetLanguage": "zh",
        "parallel": {"enabled": True, "deepLearningLockSize": 2},
        "translation": {
            "provider": "siliconflow",
            "modelName": "chapter-model",
        },
    }
    updated = repository.update_chapter_settings_memory(
        chapter_id=chapter_id,
        base_revision=1,
        payload=payload,
    )
    assert updated == {
        "chapterId": chapter_id,
        "revision": 2,
        "payload": payload,
    }
    bootstrap = repository.translation_bootstrap(
        book_id=str(book["id"]),
        chapter_id=chapter_id,
    )
    assert bootstrap["chapter"]["settingsMemory"] == payload
    assert bootstrap["chapter"]["settingsMemoryRevision"] == 2

    with pytest.raises(ContentConflict):
        repository.update_chapter_settings_memory(
            chapter_id=chapter_id,
            base_revision=1,
            payload={"sourceLanguage": "korean"},
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        repository.update_chapter_settings_memory(
            chapter_id=chapter_id,
            base_revision=2,
            payload={"textStyle": {"fontSize": 42}},
        )
    with pytest.raises(ValueError, match="apiKey"):
        repository.update_chapter_settings_memory(
            chapter_id=chapter_id,
            base_revision=2,
            payload={
                "translation": {
                    "provider": "custom",
                    "apiKey": "must-not-be-stored",
                }
            },
        )
    with pytest.raises(ValueError, match="translation.textDetector is invalid"):
        repository.update_chapter_settings_memory(
            chapter_id=chapter_id,
            base_revision=2,
            payload={"textDetector": "browser-only-fallback"},
        )
def test_page_import_publishes_source_and_webp_thumbnail_without_base64(
    content_platform,
) -> None:
    data_root, engine, repository, storage, importer, _book, chapter = content_platform
    result, replayed = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((1200, 800)),
        logical_path="folder/page 1.png",
        key="page-1",
    )
    assert not replayed
    page = result["page"]
    assert page["width"] == 1200 and page["height"] == 800
    assert page["sourceUrl"].startswith("/api/v2/assets/")
    assert page["thumbnailSourceUrl"].startswith("/api/v2/assets/")
    assert "base64" not in str(result).lower()

    with engine.connect() as connection:
        asset_rows = list(
            connection.execute(
                select(
                    page_assets.c.role,
                    assets.c.relative_path,
                    assets.c.width,
                    assets.c.height,
                    assets.c.mime_type,
                )
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(page_assets.c.page_id == page["id"])
                .order_by(page_assets.c.role)
            )
        )
    assert {row.role for row in asset_rows} == {"source", "thumbnail_source"}
    thumbnail = next(row for row in asset_rows if row.role == "thumbnail_source")
    assert thumbnail.mime_type == "image/webp"
    assert (thumbnail.width, thumbnail.height) == (320, 213)
    with Image.open(storage.resolve_relative_path(thumbnail.relative_path)) as decoded:
        assert decoded.format == "WEBP"
        assert decoded.size == (320, 213)


def test_page_import_materializes_the_saved_default_font_as_a_foreign_key(
    content_platform,
) -> None:
    _root, engine, repository, _storage, importer, _book, chapter = content_platform
    font_id = "10000000-0000-0000-0000-000000000001"
    with engine.begin() as connection:
        connection.execute(
            insert(fonts).values(
                id=font_id,
                display_name="测试字体",
                kind="builtin",
                builtin_key="test-font",
            )
        )
        style = json.loads(
            connection.execute(
                select(app_settings.c.payload_json).where(
                    app_settings.c.domain == "text_style_defaults"
                )
            ).scalar_one()
        )
        style["fontFamily"] = font_id
        style["layoutDirection"] = "horizontal"
        style["inpaintMethod"] = "litelama"
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "text_style_defaults")
            .values(payload_json=json.dumps(style))
        )

    result, _replayed = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((64, 96)),
        logical_path="font-default.png",
        key="font-default-import",
    )
    page_id = str(result["page"]["id"])

    with engine.connect() as connection:
        stored = connection.execute(
            select(
                pages.c.default_font_id,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == page_id)
        ).mappings().one()
    page_style = json.loads(stored["page_style_defaults_json"])
    assert stored["default_font_id"] == font_id
    assert "fontFamily" not in page_style
    assert page_style["layoutDirection"] == "horizontal"
    assert page_style["inpaintMethod"] == "litelama"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"fontSize": 0}, "fontSize"),
        ({"textColor": "black"}, "textColor"),
        ({"layoutDirection": "diagonal"}, "layoutDirection"),
        ({"lineSpacing": float("inf")}, "lineSpacing"),
        ({"inpaintMethod": "auto"}, "inpaintMethod"),
        ({"fontFamily": DEFAULT_FONT_ID}, "unknown page style fields"),
    ],
)
def test_page_style_patch_is_validated_by_the_backend_domain(
    content_platform,
    patch,
    message,
) -> None:
    _root, _engine, repository, _storage, importer, _book, chapter = content_platform
    result, _replayed = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((64, 96)),
        logical_path="style-validation.png",
        key=f"style-validation-{message}",
    )

    with pytest.raises(ValueError, match=message):
        repository.mutate_page_document(
            page_id=str(result["page"]["id"]),
            base_revision=1,
            mutations=[],
            idempotency_key=f"style-validation-document-{message}",
            page_style_defaults_patch=patch,
        )


def test_long_strip_thumbnail_uses_width_cap_and_top_crop(content_platform) -> None:
    _root, engine, repository, _storage, importer, _book, chapter = content_platform
    result, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((500, 5000)),
        logical_path="webtoon.png",
        key="webtoon",
    )
    with engine.connect() as connection:
        thumbnail_id = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == result["page"]["id"],
                page_assets.c.role == "thumbnail_source",
            )
        ).scalar_one()
        dimensions = connection.execute(
            select(assets.c.width, assets.c.height).where(assets.c.id == thumbnail_id)
        ).one()
    assert dimensions == (320, 1280)


def test_import_is_idempotent_and_duplicate_names_are_server_deduplicated(
    content_platform,
) -> None:
    _root, engine, repository, _storage, importer, _book, chapter = content_platform
    chapter_id = str(chapter["id"])
    payload = _image_bytes((30, 40))
    first, _ = _import(
        repository,
        importer,
        chapter_id=chapter_id,
        payload=payload,
        logical_path="page.png",
        key="same",
    )
    lease = repository.create_import_lease(chapter_id)
    try:
        replay, replayed = importer.import_page(
            chapter_id=chapter_id,
            logical_path="page.png",
            upload=BytesIO(payload),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key="same",
        )
        assert replayed and replay == first
        with pytest.raises(IdempotencyConflict):
            importer.import_page(
                chapter_id=chapter_id,
                logical_path="page.png",
                upload=BytesIO(_image_bytes((31, 40))),
                lease_id=lease.id,
                owner_token=lease.owner_token,
                idempotency_key="same",
            )
    finally:
        repository.release_import_lease(
            chapter_id=chapter_id,
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )

    second, _ = _import(
        repository,
        importer,
        chapter_id=chapter_id,
        payload=payload,
        logical_path="page.png",
        key="different",
    )
    assert second["page"]["logicalSourcePath"] == "page (2).png"
    with engine.connect() as connection:
        assert connection.execute(
            select(pages.c.id).where(pages.c.chapter_id == chapter_id)
        ).all().__len__() == 2


def test_page_listing_is_cursor_paginated_metadata_only(content_platform) -> None:
    _root, _engine, repository, _storage, importer, _book, chapter = content_platform
    chapter_id = str(chapter["id"])
    for index in range(3):
        _import(
            repository,
            importer,
            chapter_id=chapter_id,
            payload=_image_bytes((10 + index, 20)),
            logical_path=f"{index}.png",
            key=f"key-{index}",
        )
    first = repository.list_pages(chapter_id=chapter_id, limit=2)
    second = repository.list_pages(
        chapter_id=chapter_id,
        after_ordinal=int(first["nextCursor"]),
        limit=2,
    )
    assert [item["ordinal"] for item in first["items"]] == [1, 2]
    assert [item["ordinal"] for item in second["items"]] == [3]
    assert "base64" not in str(first).lower()

    page_id = str(first["items"][0]["id"])
    summary = repository.get_page_summary(page_id)
    assert summary == first["items"][0]
    assert summary["sourceUrl"].startswith("/api/v2/assets/")


def test_clear_chapter_pages_is_one_guarded_atomic_backend_command(
    content_platform,
) -> None:
    root, engine, repository, _storage, importer, book, chapter = content_platform
    chapter_id = str(chapter["id"])
    for index in range(2):
        _import(
            repository,
            importer,
            chapter_id=chapter_id,
            payload=_image_bytes((16 + index, 24)),
            logical_path=f"clear-{index}.png",
            key=f"clear-{index}",
        )
    with engine.connect() as connection:
        revision_before = connection.execute(
            select(chapters.c.page_order_revision).where(
                chapters.c.id == chapter_id
            )
        ).scalar_one()
    with engine.begin() as connection:
        connection.execute(
            insert(jobs).values(
                id="clear-pages-blocker",
                kind="translation",
                status="queued",
                book_id=str(book["id"]),
                chapter_id=chapter_id,
                config_json="{}",
            )
        )

    app = create_api_app(
        ApiSettings(
            data_root=root,
            identity=RuntimeIdentity(
                epoch_id="test-clear-pages-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    client = app.test_client()
    blocked = client.delete(
        f"/api/v2/chapters/{chapter_id}/pages",
        headers={"Idempotency-Key": "clear-pages-blocked"},
    )
    assert blocked.status_code == 423
    assert len(repository.list_pages(chapter_id=chapter_id, all_pages=True)["items"]) == 2

    with engine.begin() as connection:
        connection.execute(
            jobs.delete().where(jobs.c.id == "clear-pages-blocker")
        )
    cleared = client.delete(
        f"/api/v2/chapters/{chapter_id}/pages",
        headers={"Idempotency-Key": "clear-pages-success"},
    )
    assert cleared.status_code == 200
    assert cleared.get_json() == {"deletedCount": 2}
    assert repository.list_pages(chapter_id=chapter_id, all_pages=True)["items"] == []
    with engine.connect() as connection:
        revision_after = connection.execute(
            select(chapters.c.page_order_revision).where(
                chapters.c.id == chapter_id
            )
        ).scalar_one()
    assert revision_after == revision_before + 1


def test_single_page_summary_route_returns_only_requested_page(content_platform) -> None:
    root, engine, repository, _storage, importer, _book, chapter = content_platform
    imported, _replayed = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((14, 21)),
        logical_path="single.png",
        key="single-summary",
    )
    app = create_api_app(
        ApiSettings(
            data_root=root,
            identity=RuntimeIdentity(
                epoch_id="test-page-summary-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    response = app.test_client().get(f"/api/v2/pages/{imported['page']['id']}")
    assert response.status_code == 200
    summary = response.get_json()
    assert summary["id"] == imported["page"]["id"]
    assert summary["sourceUrl"] == imported["page"]["sourceUrl"]
    assert summary["thumbnailSourceUrl"] == imported["page"]["thumbnailSourceUrl"]
    assert summary["renderStatus"] == "not_rendered"
    assert "base64" not in response.get_data(as_text=True).lower()


def test_last_visited_page_is_independent_last_write_wins(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, importer, book, chapter = (
        content_platform
    )
    first, _replayed = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((120, 180)),
        logical_path="001.png",
        key="navigation-first",
    )
    second, _replayed = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((120, 180), color=(60, 40, 20)),
        logical_path="002.png",
        key="navigation-second",
    )

    initial = repository.update_last_visited_page(
        chapter_id=str(chapter["id"]),
        page_id=str(first["page"]["id"]),
    )
    stale_tab = repository.update_last_visited_page(
        chapter_id=str(chapter["id"]),
        page_id=str(second["page"]["id"]),
    )
    bootstrap = repository.translation_bootstrap(
        book_id=str(book["id"]),
        chapter_id=str(chapter["id"]),
    )

    assert initial["revision"] == 1
    assert stale_tab["revision"] == 2
    assert bootstrap["navigation"] == {
        "lastVisitedPageId": str(second["page"]["id"]),
        "revision": 2,
    }


def test_import_lease_and_chapter_order_cas_enforce_backend_ownership(
    content_platform,
) -> None:
    _root, engine, repository, _storage, _importer, book, chapter = content_platform
    chapter_id = str(chapter["id"])
    lease = repository.create_import_lease(chapter_id)
    with pytest.raises(ContentLocked):
        repository.create_import_lease(chapter_id)
    with pytest.raises(ContentLocked):
        repository.release_import_lease(
            chapter_id=chapter_id,
            lease_id=lease.id,
            owner_token="wrong",
        )
    repository.release_import_lease(
        chapter_id=chapter_id,
        lease_id=lease.id,
        owner_token=lease.owner_token,
    )

    second = repository.create_chapter(book_id=str(book["id"]), title="Second")
    revision = repository.list_chapters(str(book["id"]))["book"][
        "chapter_order_revision"
    ]
    updated = repository.reorder_chapters(
        book_id=str(book["id"]),
        ordered_ids=[str(second["id"]), chapter_id],
        base_revision=int(revision),
    )
    assert updated == int(revision) + 1
    with pytest.raises(ContentConflict):
        repository.reorder_chapters(
            book_id=str(book["id"]),
            ordered_ids=[chapter_id, str(second["id"])],
            base_revision=int(revision),
        )

    with engine.begin() as connection:
        connection.execute(
            insert(jobs).values(
                id="lock-job",
                kind="translation",
                status="queued",
                chapter_id=chapter_id,
                config_json="{}",
            )
        )
        connection.execute(
            insert(chapter_write_locks).values(
                chapter_id=chapter_id,
                job_id="lock-job",
                lock_generation=1,
            )
        )
    with pytest.raises(ContentLocked):
        repository.create_import_lease(chapter_id)


def test_media_api_streams_immutable_asset_and_honors_conditional_get(
    content_platform,
) -> None:
    data_root, engine, repository, _storage, importer, _book, chapter = content_platform
    result, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((20, 20)),
        logical_path="media.png",
        key="media",
    )
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    client = app.test_client()
    first = client.get(result["page"]["thumbnailSourceUrl"])
    assert first.status_code == 200
    assert first.content_type == "image/webp"
    assert "immutable" in first.headers["Cache-Control"]
    assert first.headers["ETag"]
    conditional = client.get(
        result["page"]["thumbnailSourceUrl"],
        headers={"If-None-Match": first.headers["ETag"]},
    )
    assert conditional.status_code == 304
    download = client.get(
        f'{result["page"]["thumbnailSourceUrl"]}?download=1'
        "&filename=original_media.png",
    )
    assert download.status_code == 200
    assert "original_media.png" in download.headers["Content-Disposition"]
    assert "original_media.png.webp" not in download.headers["Content-Disposition"]
    extensionless_download = client.get(
        f'{result["page"]["thumbnailSourceUrl"]}?download=1'
        "&filename=task-artifact",
    )
    assert extensionless_download.status_code == 200
    assert (
        "task-artifact.webp"
        in extensionless_download.headers["Content-Disposition"]
    )


def test_media_api_marks_missing_object_as_failed_integrity(
    content_platform,
) -> None:
    data_root, engine, repository, storage, importer, _book, chapter = content_platform
    result, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((20, 20)),
        logical_path="missing.png",
        key="missing-media",
    )
    asset_id = result["page"]["thumbnailSourceUrl"].rsplit("/", 1)[-1]
    record = storage.get_record(asset_id)
    assert record is not None
    storage.resolve_relative_path(record.relative_path).unlink()

    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    response = app.test_client().get(result["page"]["thumbnailSourceUrl"])

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "asset_missing"
    with engine.connect() as connection:
        status = connection.execute(
            select(assets.c.integrity_status).where(assets.c.id == asset_id)
        ).scalar_one()
    assert status == "missing"


def test_page_document_uses_stable_bubble_ids_and_revision_cas(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, importer, _book, chapter = content_platform
    imported, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((50, 50)),
        logical_path="document.png",
        key="document",
    )
    page_id = str(imported["page"]["id"])
    created, _ = repository.mutate_page_document(
        page_id=page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "document-create-bubble",
                "fields": {"text": "hello", "fontSize": 24},
            }
        ],
        idempotency_key="document-create",
    )
    created_document = created["document"]
    bubble_id = created["mutationResults"][0]["bubbleId"]
    assert str(uuid.UUID(bubble_id)) == bubble_id
    assert created_document["documentRevision"] == 2
    assert created_document["renderStatus"] == "not_rendered"
    assert created_document["bubbles"][0]["bubbleId"] == bubble_id
    with pytest.raises(ValueError, match="cannot provide bubbleId"):
        repository.mutate_page_document(
            page_id=page_id,
            base_revision=2,
            mutations=[
                {
                    "op": "create",
                    "clientMutationId": "client-cannot-assign-fact-id",
                    "bubbleId": "00000000-0000-0000-0000-000000000111",
                    "fields": {},
                }
            ],
            idempotency_key="document-client-fact-id",
        )
    with pytest.raises(ValueError, match="non-payload"):
        repository.mutate_page_document(
            page_id=page_id,
            base_revision=2,
            mutations=[
                {
                    "op": "create",
                    "clientMutationId": "document-payload-font-path",
                    "fields": {"fontFamily": "fonts/client-path.ttf"},
                }
            ],
            idempotency_key="document-payload-font-path",
        )
    patched, _ = repository.mutate_page_document(
        page_id=page_id,
        base_revision=2,
        mutations=[
            {
                "op": "patch",
                "clientMutationId": "document-patch-bubble",
                "bubbleId": bubble_id,
                "fields": {"text": "translated"},
            }
        ],
        idempotency_key="document-patch",
    )
    assert patched["document"]["bubbles"][0]["payload"]["text"] == "translated"
    with pytest.raises(ContentConflict):
        repository.mutate_page_document(
            page_id=page_id,
            base_revision=2,
            mutations=[
                {
                    "op": "delete",
                    "clientMutationId": "document-delete-bubble",
                    "bubbleId": bubble_id,
                }
            ],
            idempotency_key="document-stale-delete",
        )


def test_page_document_route_persists_editor_mutations_and_optional_font(
    content_platform,
) -> None:
    data_root, engine, repository, _storage, importer, _book, chapter = (
        content_platform
    )
    imported, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((80, 120)),
        logical_path="editor-route.png",
        key="editor-route",
    )
    page_id = str(imported["page"]["id"])
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-editor-route-api",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    client = app.test_client()
    mutation_response = client.patch(
        f"/api/v2/pages/{page_id}/document",
        headers={"Idempotency-Key": "editor-route-mutation"},
        json={
            "baseRevision": 1,
            "mutations": [
                {
                    "op": "create",
                    "clientMutationId": "editor-route-create",
                    "fields": {
                        "translatedText": "编辑写入",
                        "coords": [5, 6, 50, 60],
                        "fontSize": 20,
                    },
                }
            ],
        },
    )

    assert mutation_response.status_code == 200
    mutated = mutation_response.get_json()
    bubble_id = mutated["mutationResults"][0]["bubbleId"]
    assert mutated["document"]["documentRevision"] == 2
    assert mutated["document"]["bubbles"][0]["bubbleId"] == bubble_id
    assert (
        mutated["document"]["bubbles"][0]["payload"]["translatedText"]
        == "编辑写入"
    )

    font_response = client.patch(
        f"/api/v2/pages/{page_id}/document",
        headers={"Idempotency-Key": "editor-route-default-font"},
        json={
            "baseRevision": 2,
            "mutations": [],
            "defaultFontId": DEFAULT_FONT_ID,
        },
    )

    assert font_response.status_code == 200
    updated = font_response.get_json()["document"]
    assert updated["documentRevision"] == 3
    assert updated["defaultFontId"] == DEFAULT_FONT_ID

    invalid_font_response = client.patch(
        f"/api/v2/pages/{page_id}/document",
        headers={"Idempotency-Key": "editor-route-invalid-font"},
        json={
            "baseRevision": 3,
            "mutations": [
                {
                    "op": "patch",
                    "clientMutationId": "editor-route-invalid-font",
                    "bubbleId": bubble_id,
                    "fields": {
                        "fontId": "",
                        "translatedText": "不能写入无效字体",
                    },
                }
            ],
        },
    )

    assert invalid_font_response.status_code == 422
    unchanged = repository.get_page_document(page_id)
    assert unchanged["documentRevision"] == 3
    assert unchanged["bubbles"][0]["payload"]["translatedText"] == "编辑写入"


def test_plan_page_source_bubble_and_render_status_routes(
    content_platform,
) -> None:
    data_root, engine, repository, _storage, importer, _book, chapter = (
        content_platform
    )
    imported, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((80, 120)),
        logical_path="plan-routes.png",
        key="plan-routes",
    )
    page_id = str(imported["page"]["id"])
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="test-plan-page-routes",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    client = app.test_client()
    try:
        replaced = client.put(
            f"/api/v2/pages/{page_id}/source",
            headers={"Idempotency-Key": "plan-source-replace"},
            data={
                "baseSourceRevision": "1",
                "file": (BytesIO(_image_bytes((96, 144))), "replacement.png"),
            },
            content_type="multipart/form-data",
        )
        assert replaced.status_code == 200
        assert "/api/v2/pages/<page_id>/replace-source" not in {
            rule.rule for rule in app.url_map.iter_rules()
        }

        base_revision = replaced.get_json()["documentRevision"]
        created = client.post(
            f"/api/v2/pages/{page_id}/bubbles",
            headers={"Idempotency-Key": "plan-bubble-create"},
            json={
                "baseRevision": base_revision,
                "fields": {
                    "translatedText": "后端气泡",
                    "coords": [5, 6, 50, 60],
                    "fontSize": 20,
                },
            },
        )
        assert created.status_code == 200
        created_payload = created.get_json()
        bubble_id = created_payload["mutationResults"][0]["bubbleId"]

        render_status = client.get(
            f"/api/v2/pages/{page_id}/render-status"
        )
        assert render_status.status_code == 200
        assert render_status.get_json()["pageId"] == page_id
        assert render_status.get_json()["documentRevision"] == (
            created_payload["document"]["documentRevision"]
        )

        patched = client.patch(
            f"/api/v2/pages/{page_id}/bubbles/{bubble_id}",
            headers={"Idempotency-Key": "plan-bubble-patch"},
            json={
                "baseRevision": created_payload["document"]["documentRevision"],
                "fields": {"translatedText": "修改后"},
            },
        )
        assert patched.status_code == 200
        patched_payload = patched.get_json()
        assert patched_payload["document"]["bubbles"][0]["payload"][
            "translatedText"
        ] == "修改后"

        deleted = client.delete(
            f"/api/v2/pages/{page_id}/bubbles/{bubble_id}",
            headers={"Idempotency-Key": "plan-bubble-delete"},
            json={
                "baseRevision": patched_payload["document"][
                    "documentRevision"
                ]
            },
        )
        assert deleted.status_code == 200
        assert deleted.get_json()["document"]["bubbles"] == []
    finally:
        app.extensions["saber_v2_runtime"].close()


def test_page_document_command_is_idempotent_and_propagates_style(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, importer, _book, chapter = content_platform
    imported, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((50, 50)),
        logical_path="idempotent-document.png",
        key="idempotent-document",
    )
    page_id = str(imported["page"]["id"])
    command = {
        "page_id": page_id,
        "base_revision": 1,
        "mutations": [
            {
                "op": "create",
                "clientMutationId": "document-command-create",
                "fields": {"translatedText": "hello"},
            }
        ],
        "idempotency_key": "document-command",
        "page_style_defaults_patch": {
            "autoFontSize": False,
            "fontSize": 30,
        },
        "propagate_style_fields": ["fontSize"],
    }
    first, replayed = repository.mutate_page_document(**command)
    assert replayed is False
    first_document = first["document"]
    assert first_document["documentRevision"] == 2
    assert first_document["pageStyleDefaults"]["fontSize"] == 30
    assert first_document["bubbles"][0]["payload"]["fontSize"] == 30
    assert first["mutationResults"][0]["clientMutationId"] == (
        "document-command-create"
    )

    replay, replayed = repository.mutate_page_document(**command)
    assert replayed is True
    assert replay == first

    with pytest.raises(IdempotencyConflict):
        repository.mutate_page_document(
            **{
                **command,
                "page_style_defaults_patch": {"fontSize": 31},
            }
        )


def test_document_mutation_materializes_auto_style_before_render_projection(
    content_platform,
    monkeypatch,
) -> None:
    _root, engine, repository, storage, importer, _book, chapter = (
        content_platform
    )
    imported, _ = _import(
        repository,
        importer,
        chapter_id=str(chapter["id"]),
        payload=_image_bytes((160, 120)),
        logical_path="auto-style.png",
        key="auto-style",
    )
    page_id = str(imported["page"]["id"])
    document, _ = repository.mutate_page_document(
        page_id=page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "auto-style-create-bubble",
                "fields": {
                    "translatedText": "自动样式",
                    "coords": [0, 0, 120, 80],
                    "autoTextDirection": "horizontal",
                    "autoFgColor": [1, 2, 3],
                    "autoBgColor": [10, 11, 12],
                },
            }
        ],
        idempotency_key="auto-style-create",
        page_style_defaults_patch={
            "autoFontSize": True,
            "layoutDirection": "auto",
            "useAutoTextColor": True,
        },
        propagate_style_fields=[
            "fontSize",
            "layoutDirection",
            "textColor",
            "fillColor",
        ],
    )
    document = document["document"]
    persisted_payload = document["bubbles"][0]["payload"]
    assert persisted_payload["textDirection"] == "horizontal"
    assert persisted_payload["textColor"] == "#010203"
    assert persisted_payload["fillColor"] == "#0A0B0C"
    assert persisted_payload["fontSize"] > 12

    from src.core import rendering

    def fail_if_recalculated(*_args, **_kwargs):
        raise AssertionError("ordinary render projection recalculated auto font size")

    monkeypatch.setattr(
        rendering,
        "calculate_auto_font_size",
        fail_if_recalculated,
    )
    with engine.connect() as connection:
        projected = materialize_render_payloads(
            connection,
            storage,
            page_id,
        )

    _bubble_id, persisted, render_payload = projected[0]
    assert persisted["textDirection"] == "horizontal"
    assert persisted["textColor"] == "#010203"
    assert persisted["fillColor"] == "#0A0B0C"
    assert persisted["fontSize"] == persisted_payload["fontSize"]
    assert render_payload["fontFamily"]
    assert "00000000-0000-0000-0000-000000000010" not in str(
        render_payload["fontFamily"]
    )

    manual, _ = repository.mutate_page_document(
        page_id=page_id,
        base_revision=2,
        mutations=[],
        idempotency_key="auto-style-manual",
        page_style_defaults_patch={
            "useAutoTextColor": False,
            "textColor": "#123456",
            "fillColor": "#ABCDEF",
        },
        propagate_style_fields=["textColor", "fillColor"],
    )
    manual = manual["document"]
    manual_payload = manual["bubbles"][0]["payload"]
    assert manual["pageStyleDefaults"]["useAutoTextColor"] is False
    assert manual_payload["textColor"] == "#123456"
    assert manual_payload["fillColor"] == "#ABCDEF"
    assert manual_payload["autoFgColor"] == [1, 2, 3]
    assert manual_payload["autoBgColor"] == [10, 11, 12]


def test_quick_workspace_promote_moves_relations_without_moving_assets(
    content_platform,
) -> None:
    _root, engine, repository, _storage, importer, _book, _chapter = content_platform
    quick = repository.list_chapters(QUICK_WORKSPACE_BOOK_ID)
    quick_chapter_id = str(quick["chapters"][0]["id"])
    imported, _ = _import(
        repository,
        importer,
        chapter_id=quick_chapter_id,
        payload=_image_bytes((40, 60)),
        logical_path="quick.png",
        key="quick",
    )
    with engine.connect() as connection:
        source_path_before = connection.execute(
            select(assets.c.relative_path)
            .join(page_assets, page_assets.c.asset_id == assets.c.id)
            .where(
                page_assets.c.page_id == imported["page"]["id"],
                page_assets.c.role == "source",
            )
        ).scalar_one()

    promoted = repository.promote_quick_workspace(
        chapter_title="Saved Chapter",
        new_book_title="Saved Book",
    )
    assert promoted["chapterId"] == quick_chapter_id
    with engine.connect() as connection:
        moved = connection.execute(
            select(chapters.c.book_id, chapters.c.title).where(
                chapters.c.id == quick_chapter_id
            )
        ).one()
        source_path_after = connection.execute(
            select(assets.c.relative_path)
            .join(page_assets, page_assets.c.asset_id == assets.c.id)
            .where(
                page_assets.c.page_id == imported["page"]["id"],
                page_assets.c.role == "source",
            )
        ).scalar_one()
    assert moved == (promoted["bookId"], "Saved Chapter")
    assert source_path_after == source_path_before
    assert repository.list_chapters(QUICK_WORKSPACE_BOOK_ID)["chapters"] == [
        {
            "id": promoted["quickChapterId"],
            "ordinal": 1,
            "title": "快速翻译",
            "pageCount": 0,
            "pageOrderRevision": 1,
        }
    ]


def test_quick_workspace_promote_route_uses_explicit_mode_contract(
    content_platform,
) -> None:
    data_root, engine, repository, _storage, importer, _book, _chapter = (
        content_platform
    )
    quick_chapter_id = str(
        repository.list_chapters(QUICK_WORKSPACE_BOOK_ID)["chapters"][0][
            "id"
        ]
    )
    _import(
        repository,
        importer,
        chapter_id=quick_chapter_id,
        payload=_image_bytes((40, 60)),
        logical_path="route-quick.png",
        key="route-quick",
    )
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="quick-promote-api",
                epoch_token="test-token",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    try:
        response = app.test_client().post(
            "/api/v2/quick-workspace/promote",
            headers={"Idempotency-Key": "promote-route"},
            json={
                "mode": "new_book",
                "title": "Route Book",
                "chapterTitle": "Route Chapter",
            },
        )
    finally:
        app.extensions["saber_v2_runtime"].close()

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chapterId"] == quick_chapter_id
    assert repository.get_book(str(payload["bookId"]))["title"] == "Route Book"


def test_quick_workspace_promote_rejects_duplicate_destinations(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, importer, book, _chapter = (
        content_platform
    )
    quick_chapter_id = str(
        repository.list_chapters(QUICK_WORKSPACE_BOOK_ID)["chapters"][0][
            "id"
        ]
    )
    _import(
        repository,
        importer,
        chapter_id=quick_chapter_id,
        payload=_image_bytes((40, 60)),
        logical_path="duplicate-quick.png",
        key="duplicate-quick",
    )

    with pytest.raises(ValueError, match="new book title already exists"):
        repository.promote_quick_workspace(
            chapter_title="Saved Chapter",
            new_book_title=str(book["title"]),
        )

    with pytest.raises(ValueError, match="chapter title already exists"):
        repository.promote_quick_workspace(
            chapter_title="Chapter",
            target_book_id=str(book["id"]),
        )


@pytest.mark.parametrize("blocker", ("job", "operation", "import_lease"))
def test_quick_workspace_reset_and_promote_reject_every_active_work_kind(
    content_platform,
    blocker: str,
) -> None:
    _root, engine, repository, _storage, importer, _book, _chapter = (
        content_platform
    )
    quick_chapter_id = str(
        repository.list_chapters(QUICK_WORKSPACE_BOOK_ID)["chapters"][0][
            "id"
        ]
    )
    imported, _ = _import(
        repository,
        importer,
        chapter_id=quick_chapter_id,
        payload=_image_bytes((32, 48)),
        logical_path=f"{blocker}.png",
        key=f"quick-{blocker}",
    )
    if blocker == "job":
        with engine.begin() as connection:
            connection.execute(
                insert(jobs).values(
                    id="quick-active-job",
                    kind="export",
                    status="queued",
                    book_id=QUICK_WORKSPACE_BOOK_ID,
                    chapter_id=quick_chapter_id,
                    config_json="{}",
                )
            )
    elif blocker == "operation":
        with engine.begin() as connection:
            connection.execute(
                insert(operations).values(
                    id="quick-active-operation",
                    kind="page_detect",
                    executor_role="worker",
                    status="pending",
                    page_id=imported["page"]["id"],
                    base_revision=1,
                    request_json="{}",
                )
            )
    else:
        repository.create_import_lease(quick_chapter_id)

    with pytest.raises(ContentLocked):
        repository.reset_quick_workspace()
    with pytest.raises(ContentLocked):
        repository.promote_quick_workspace(
            chapter_title="Blocked",
            new_book_title="Blocked Book",
        )


def test_quick_workspace_reset_clears_pages_and_constraints(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, importer, _book, _chapter = (
        content_platform
    )
    quick_chapter_id = str(
        repository.list_chapters(QUICK_WORKSPACE_BOOK_ID)["chapters"][0][
            "id"
        ]
    )
    _import(
        repository,
        importer,
        chapter_id=quick_chapter_id,
        payload=_image_bytes((32, 48)),
        logical_path="reset.png",
        key="quick-reset",
    )
    repository.update_constraints(
        book_id=QUICK_WORKSPACE_BOOK_ID,
        payload={
            "glossary": {
                "enabled": True,
                "autoExtractEnabled": False,
                "autoExtractPrompt": "提取术语：{ocr_text}",
                "entries": [
                    {
                        "source": "Saber",
                        "target": "阿尔托莉雅",
                        "note": "",
                        "matchMode": "text",
                    }
                ],
            },
            "nonTranslate": {
                "enabled": True,
                "entries": [
                    {
                        "pattern": "Excalibur",
                        "note": "",
                        "matchMode": "text",
                    }
                ],
            },
        },
        base_revision=1,
    )

    reset = repository.reset_quick_workspace()

    assert reset["chapterId"] != quick_chapter_id
    assert repository.list_pages(chapter_id=reset["chapterId"])["items"] == []
    assert (
        repository.get_constraints(QUICK_WORKSPACE_BOOK_ID)["payload"]
        == empty_translation_constraints()
    )


def test_translation_constraints_validate_regex_and_deduplicate_identical_rows(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, _importer, book, _chapter = (
        content_platform
    )
    payload = empty_translation_constraints()
    payload["glossary"]["enabled"] = True
    payload["glossary"]["entries"] = [
        {
            "source": "Saber",
            "target": "阿尔托莉雅",
            "note": "",
            "matchMode": "text",
        },
        {
            "source": "Saber",
            "target": "阿尔托莉雅",
            "note": "",
            "matchMode": "text",
        },
    ]
    saved = repository.update_constraints(
        book_id=str(book["id"]),
        base_revision=1,
        payload=payload,
    )
    assert len(saved["payload"]["glossary"]["entries"]) == 1
    assert saved["schemaVersion"] == 2

    invalid = empty_translation_constraints()
    invalid["nonTranslate"]["enabled"] = True
    invalid["nonTranslate"]["entries"] = [
        {
            "pattern": "[unterminated",
            "note": "",
            "matchMode": "regex",
        }
    ]
    with pytest.raises(ValueError, match="invalid regular expression"):
        repository.update_constraints(
            book_id=str(book["id"]),
            base_revision=2,
            payload=invalid,
        )


def test_promote_to_existing_book_keeps_destination_constraints_and_resets_quick(
    content_platform,
) -> None:
    _root, _engine, repository, _storage, importer, book, _chapter = (
        content_platform
    )
    quick_chapter_id = str(
        repository.list_chapters(QUICK_WORKSPACE_BOOK_ID)["chapters"][0][
            "id"
        ]
    )
    _import(
        repository,
        importer,
        chapter_id=quick_chapter_id,
        payload=_image_bytes((32, 48)),
        logical_path="existing.png",
        key="quick-existing",
    )
    repository.update_constraints(
        book_id=QUICK_WORKSPACE_BOOK_ID,
        payload={
            "glossary": {
                "enabled": True,
                "autoExtractEnabled": False,
                "autoExtractPrompt": "提取术语：{ocr_text}",
                "entries": [
                    {
                        "source": "Quick",
                        "target": "快速",
                        "note": "",
                        "matchMode": "text",
                    }
                ],
            },
            "nonTranslate": {"enabled": False, "entries": []},
        },
        base_revision=1,
    )
    repository.update_constraints(
        book_id=str(book["id"]),
        payload={
            "glossary": {
                "enabled": True,
                "autoExtractEnabled": False,
                "autoExtractPrompt": "提取术语：{ocr_text}",
                "entries": [
                    {
                        "source": "Library",
                        "target": "书架",
                        "note": "",
                        "matchMode": "text",
                    }
                ],
            },
            "nonTranslate": {
                "enabled": True,
                "entries": [
                    {
                        "pattern": "Keep",
                        "note": "",
                        "matchMode": "text",
                    }
                ],
            },
        },
        base_revision=1,
    )

    promoted = repository.promote_quick_workspace(
        chapter_title="Imported",
        target_book_id=str(book["id"]),
    )

    assert promoted["chapterId"] == quick_chapter_id
    assert repository.get_constraints(str(book["id"]))["payload"] == {
        "glossary": {
            "enabled": True,
            "autoExtractEnabled": False,
            "autoExtractPrompt": "提取术语：{ocr_text}",
            "entries": [
                {
                    "source": "Library",
                    "target": "书架",
                    "note": "",
                    "matchMode": "text",
                }
            ],
        },
        "nonTranslate": {
            "enabled": True,
            "entries": [
                {
                    "pattern": "Keep",
                    "note": "",
                    "matchMode": "text",
                }
            ],
        },
    }
    assert (
        repository.get_constraints(QUICK_WORKSPACE_BOOK_ID)["payload"]
        == empty_translation_constraints()
    )
