"""HTTP routes for the stage-2 content vertical slice."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    json_body as _json_body,
    require_idempotency_key as _require_idempotency_key,
    validate_multipart_fields as _validate_multipart_fields,
)
from src.backend_v2.content.image_import import (
    ImageImportService,
    UnsupportedImage,
)
from src.backend_v2.content.media import AssetMediaService
from src.backend_v2.content.repository import (
    ContentConflict,
    ContentLocked,
    ContentNotFound,
    ContentRepository,
    IdempotencyConflict,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.platform_repositories import (
    FontRepository,
    PromptRepository,
    SettingsRepository,
)


def _asset_download_name(
    requested_name: str | None,
    *,
    asset_id: str,
    suffix: str,
) -> str:
    if requested_name:
        # This value is only a Content-Disposition filename, but still discard
        # caller-supplied path components and header control characters.
        candidate = (
            requested_name.replace("\\", "/")
            .rsplit("/", 1)[-1]
            .replace("\r", "")
            .replace("\n", "")
            .strip()
        )
        if candidate:
            return candidate if Path(candidate).suffix else f"{candidate}{suffix}"
    return f"{asset_id}{suffix}"


def create_content_blueprint(*, data_root, engine: Engine) -> Blueprint:
    blueprint = Blueprint("content_v2", __name__, url_prefix="/api/v2")
    repository = ContentRepository(engine)
    settings_repository = SettingsRepository(engine)
    prompt_repository = PromptRepository(engine)
    font_repository = FontRepository(engine)
    storage = AssetStorageService(data_root, engine)
    importer = ImageImportService(
        data_root=data_root,
        repository=repository,
        storage=storage,
    )
    media = AssetMediaService(engine=engine, storage=storage)

    @blueprint.errorhandler(ContentNotFound)
    def not_found(error: ContentNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(IdempotencyConflict)
    def idempotency_conflict(error: IdempotencyConflict):
        return _error("idempotency_conflict", str(error), 409)

    @blueprint.errorhandler(ContentConflict)
    def conflict(error: ContentConflict):
        return _error("revision_conflict", str(error), 409)

    @blueprint.errorhandler(ContentLocked)
    def locked(error: ContentLocked):
        return _error("chapter_locked", str(error), 423)

    @blueprint.errorhandler(UnsupportedImage)
    def unsupported_image(error: UnsupportedImage):
        return _error("unsupported_image", str(error), 422)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.get("/books")
    def list_books() -> Response:
        tag_ids = tuple(
            value for value in request.args.get("tagIds", "").split(",") if value
        )
        return jsonify(
            {
                "items": repository.list_books(
                    search=request.args.get("search", ""),
                    tag_ids=tag_ids,
                    sort_by=request.args.get("sort_by", "updated_at"),
                    sort_order=request.args.get("sort_order", "desc"),
                )
            }
        )

    @blueprint.post("/books")
    def create_book() -> tuple[Response, int]:
        _require_idempotency_key()
        body = _book_body(update=False)
        tag_ids = body.get("tagIds", [])
        if not isinstance(tag_ids, list) or not all(
            isinstance(value, str) for value in tag_ids
        ):
            raise ValueError("tagIds must be a string array")
        cover = request.files.get("cover")
        cover_asset = importer.publish_cover(cover.stream) if cover else None
        created = repository.create_book(
            title=str(body.get("title", "")),
            tag_ids=tag_ids,
            cover_asset_id=cover_asset.id if cover_asset else None,
        )
        return jsonify(created), 201

    @blueprint.get("/books/<book_id>")
    def get_book(book_id: str) -> Response:
        return jsonify(repository.get_book(book_id))

    @blueprint.put("/books/<book_id>")
    def update_book(book_id: str) -> Response:
        _require_idempotency_key()
        body = _book_body(update=True)
        tag_ids = body.get("tagIds")
        if tag_ids is not None and (
            not isinstance(tag_ids, list)
            or not all(isinstance(value, str) for value in tag_ids)
        ):
            raise ValueError("tagIds must be a string array")
        cover = request.files.get("cover")
        cover_asset = importer.publish_cover(cover.stream) if cover else None
        clear_cover = _as_bool(body.get("clearCover", False))
        result = repository.update_book(
            book_id=book_id,
            title=str(body.get("title", "")),
            tag_ids=tag_ids,
            cover_asset_id=cover_asset.id if cover_asset else None,
            replace_cover=bool(cover_asset) or clear_cover,
        )
        return jsonify(result)

    @blueprint.delete("/books/<book_id>")
    def delete_book(book_id: str) -> Response:
        _require_idempotency_key()
        repository.delete_book(book_id)
        return jsonify({"deleted": True})

    @blueprint.post("/books/batch-delete")
    def batch_delete_books() -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"bookIds"})
        book_ids = body.get("bookIds")
        if not isinstance(book_ids, list) or not all(
            isinstance(value, str) for value in book_ids
        ):
            raise ValueError("bookIds must be a string array")
        result = repository.batch_delete_books(book_ids)
        return jsonify(result)

    @blueprint.post("/books/batch-tags")
    def batch_tags() -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"bookIds", "tagIds", "action"})
        book_ids = body.get("bookIds")
        tag_ids = body.get("tagIds")
        if (
            not isinstance(book_ids, list)
            or not all(isinstance(value, str) for value in book_ids)
            or not isinstance(tag_ids, list)
            or not all(isinstance(value, str) for value in tag_ids)
        ):
            raise ValueError("bookIds and tagIds must be string arrays")
        repository.batch_update_tags(
            book_ids=book_ids,
            tag_ids=tag_ids,
            action=str(body.get("action", "")),
        )
        return jsonify({"updated": len(book_ids)})

    @blueprint.get("/books/<book_id>/chapters")
    def list_chapters(book_id: str) -> Response:
        return jsonify(repository.list_chapters(book_id))

    @blueprint.post("/books/<book_id>/chapters")
    def create_chapter(book_id: str) -> tuple[Response, int]:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"title"})
        created = repository.create_chapter(
            book_id=book_id,
            title=str(body.get("title", "")),
        )
        return jsonify(created), 201

    @blueprint.put("/chapters/<chapter_id>")
    def update_chapter(chapter_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"title"})
        return jsonify(
            repository.update_chapter(
                chapter_id=chapter_id,
                title=str(body.get("title", "")),
            )
        )

    @blueprint.delete("/chapters/<chapter_id>")
    def delete_chapter(chapter_id: str) -> Response:
        _require_idempotency_key()
        repository.delete_chapter(chapter_id)
        return jsonify({"deleted": True})

    @blueprint.put("/books/<book_id>/chapters/order")
    def reorder_chapters(book_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"orderedIds", "baseRevision"})
        ordered_ids = body.get("orderedIds")
        if not isinstance(ordered_ids, list) or not all(
            isinstance(item, str) for item in ordered_ids
        ):
            raise ValueError("orderedIds must be a string array")
        revision = repository.reorder_chapters(
            book_id=book_id,
            ordered_ids=ordered_ids,
            base_revision=int(body.get("baseRevision", 0)),
        )
        return jsonify({"chapterOrderRevision": revision})

    @blueprint.get("/chapters/<chapter_id>/pages")
    def list_pages(chapter_id: str) -> Response:
        all_pages = request.args.get("all") == "1"
        cursor = int(request.args.get("cursor", "0"))
        limit = int(request.args.get("limit", "50"))
        return jsonify(
            repository.list_pages(
                chapter_id=chapter_id,
                after_ordinal=cursor,
                limit=limit,
                all_pages=all_pages,
            )
        )

    @blueprint.delete("/chapters/<chapter_id>/pages")
    def clear_chapter_pages(chapter_id: str) -> Response:
        _require_idempotency_key()
        deleted_count = repository.clear_chapter_pages(chapter_id)
        return jsonify({"deletedCount": deleted_count})

    @blueprint.get("/pages/<page_id>")
    def get_page(page_id: str) -> Response:
        return jsonify(repository.get_page_summary(page_id))

    @blueprint.put("/chapters/<chapter_id>/pages/order")
    def reorder_pages(chapter_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"orderedIds", "baseRevision"})
        ordered_ids = body.get("orderedIds")
        if not isinstance(ordered_ids, list) or not all(
            isinstance(value, str) for value in ordered_ids
        ):
            raise ValueError("orderedIds must be a string array")
        revision = repository.reorder_pages(
            chapter_id=chapter_id,
            ordered_ids=ordered_ids,
            base_revision=int(body.get("baseRevision", 0)),
        )
        return jsonify({"pageOrderRevision": revision})

    @blueprint.delete("/pages/<page_id>")
    def delete_page(page_id: str) -> Response:
        _require_idempotency_key()
        repository.delete_page(page_id)
        return jsonify({"deleted": True})

    @blueprint.post("/pages/<page_id>/replace-source")
    def replace_page_source(page_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        _validate_multipart_fields(
            allowed_form_keys={"baseSourceRevision"},
            allowed_file_keys={"file"},
        )
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        result, replayed = importer.replace_page_source(
            page_id=page_id,
            base_source_revision=int(
                request.form.get("baseSourceRevision", "0")
            ),
            upload=upload.stream,
            idempotency_key=idempotency_key,
        )
        response = jsonify(result)
        response.headers["Idempotency-Replayed"] = (
            "true" if replayed else "false"
        )
        return response

    @blueprint.get("/pages/<page_id>/document")
    def get_page_document(page_id: str) -> Response:
        return jsonify(repository.get_page_document(page_id))

    @blueprint.patch("/chapters/<chapter_id>/settings-memory")
    def update_chapter_settings_memory(chapter_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"payload", "baseRevision"})
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return jsonify(
            repository.update_chapter_settings_memory(
                chapter_id=chapter_id,
                base_revision=int(body.get("baseRevision", 0)),
                payload=payload,
            )
        )

    @blueprint.patch("/chapters/<chapter_id>/last-visited-page")
    def update_last_visited_page(chapter_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"pageId"})
        return jsonify(
            repository.update_last_visited_page(
                chapter_id=chapter_id,
                page_id=str(body.get("pageId", "")),
            )
        )

    @blueprint.get("/books/<book_id>/translation-constraints")
    def get_translation_constraints(book_id: str) -> Response:
        return jsonify(repository.get_constraints(book_id))

    @blueprint.put("/books/<book_id>/translation-constraints")
    def update_translation_constraints(book_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"payload", "baseRevision"})
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return jsonify(
            repository.update_constraints(
                book_id=book_id,
                base_revision=int(body.get("baseRevision", 0)),
                payload=payload,
            )
        )

    @blueprint.patch("/pages/<page_id>/document")
    def mutate_page_document(page_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={
                "baseRevision",
                "mutations",
                "defaultFontId",
                "pageStyleDefaultsPatch",
                "propagateStyleFields",
            }
        )
        mutations = body.get("mutations")
        if not isinstance(mutations, list):
            raise ValueError("mutations must be an array")
        optional_arguments: dict[str, object] = {}
        if "defaultFontId" in body:
            optional_arguments["default_font_id"] = body["defaultFontId"]
        result, replayed = repository.mutate_page_document(
            page_id=page_id,
            base_revision=int(body.get("baseRevision", 0)),
            mutations=mutations,
            idempotency_key=idempotency_key,
            page_style_defaults_patch=_optional_object(
                body,
                "pageStyleDefaultsPatch",
            ),
            propagate_style_fields=_optional_string_array(
                body,
                "propagateStyleFields",
            ),
            **optional_arguments,
        )
        response = jsonify(result)
        response.headers["Idempotency-Replayed"] = (
            "true" if replayed else "false"
        )
        return response

    @blueprint.post("/chapters/<chapter_id>/import-leases")
    def create_import_lease(chapter_id: str) -> tuple[Response, int]:
        _require_idempotency_key()
        lease = repository.create_import_lease(chapter_id)
        return (
            jsonify(
                {
                    "leaseId": lease.id,
                    "ownerToken": lease.owner_token,
                    "expiresAt": lease.expires_at.isoformat(),
                }
            ),
            201,
        )

    @blueprint.delete("/chapters/<chapter_id>/import-leases/<lease_id>")
    def release_import_lease(chapter_id: str, lease_id: str) -> Response:
        _require_idempotency_key()
        repository.release_import_lease(
            chapter_id=chapter_id,
            lease_id=lease_id,
            owner_token=request.headers.get("Import-Lease-Token", ""),
        )
        return jsonify({"released": True})

    @blueprint.post("/chapters/<chapter_id>/pages")
    def import_page(chapter_id: str):
        idempotency_key = _require_idempotency_key()
        _validate_multipart_fields(
            allowed_form_keys={"logicalPath"},
            allowed_file_keys={"file"},
        )
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        logical_path = str(request.form.get("logicalPath", "")).strip()
        if not logical_path:
            raise ValueError("multipart field 'logicalPath' is required")
        result, replayed = importer.import_page(
            chapter_id=chapter_id,
            logical_path=logical_path,
            upload=upload.stream,
            lease_id=request.headers.get("Import-Lease-Id", ""),
            owner_token=request.headers.get("Import-Lease-Token", ""),
            idempotency_key=idempotency_key,
        )
        response = jsonify(result)
        response.headers["Idempotency-Replayed"] = "true" if replayed else "false"
        return response, 200 if replayed else 201

    @blueprint.get("/assets/<asset_id>")
    def get_asset(asset_id: str):
        asset = media.locate(asset_id)
        if asset is None:
            return _error("asset_missing", "asset does not exist or failed integrity", 404)
        response = send_file(
            asset.path,
            mimetype=asset.mime_type,
            conditional=True,
            etag=asset.checksum,
            last_modified=asset.created_at,
            max_age=31536000,
            as_attachment=request.args.get("download") == "1",
            download_name=(
                _asset_download_name(
                    request.args.get("filename"),
                    asset_id=asset_id,
                    suffix=asset.path.suffix,
                )
                if request.args.get("download") == "1"
                else None
            ),
        )
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @blueprint.post("/quick-workspace/reset")
    def reset_quick_workspace() -> Response:
        _require_idempotency_key()
        return jsonify(repository.reset_quick_workspace())

    @blueprint.get("/translation/bootstrap")
    def translation_bootstrap() -> Response:
        result = repository.translation_bootstrap(
            book_id=request.args.get("bookId"),
            chapter_id=request.args.get("chapterId"),
        )
        result["settings"] = settings_repository.load()
        result["fonts"] = font_repository.list()
        result["prompts"] = [
            prompt
            for prompt in prompt_repository.list()
            if prompt["type"] in {"translate", "textbox"}
        ]
        return jsonify(result)

    @blueprint.post("/quick-workspace/promote")
    def promote_quick_workspace() -> Response:
        _require_idempotency_key()
        body = _json_body(
            allowed_keys={"mode", "chapterTitle", "title", "bookId"}
        )
        mode = str(body.get("mode", ""))
        if mode not in {"new_book", "existing_book"}:
            raise ValueError("mode must be new_book or existing_book")
        return jsonify(
            repository.promote_quick_workspace(
                chapter_title=str(body.get("chapterTitle", "")),
                new_book_title=(
                    str(body["title"])
                    if mode == "new_book" and body.get("title") is not None
                    else None
                ),
                target_book_id=(
                    str(body["bookId"])
                    if mode == "existing_book" and body.get("bookId") is not None
                    else None
                ),
            )
        )

    @blueprint.get("/tags")
    def list_tags() -> Response:
        return jsonify({"items": repository.list_tags()})

    @blueprint.post("/tags")
    def create_tag() -> tuple[Response, int]:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"name", "color"})
        return (
            jsonify(
                repository.create_tag(
                    name=str(body.get("name", "")),
                    color=str(body.get("color", "")),
                )
            ),
            201,
        )

    @blueprint.put("/tags/<tag_id>")
    def update_tag(tag_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"name", "color"})
        return jsonify(
            repository.update_tag(
                tag_id=tag_id,
                name=str(body.get("name", "")),
                color=str(body.get("color", "")),
            )
        )

    @blueprint.delete("/tags/<tag_id>")
    def delete_tag(tag_id: str) -> Response:
        _require_idempotency_key()
        repository.delete_tag(tag_id)
        return jsonify({"deleted": True})

    return blueprint


def _book_body(*, update: bool) -> dict[str, object]:
    allowed_keys = {"title", "tagIds"}
    if update:
        allowed_keys.add("clearCover")
    if request.is_json:
        return _json_body(allowed_keys=allowed_keys)
    _validate_multipart_fields(
        allowed_form_keys=allowed_keys,
        allowed_file_keys={"cover"},
    )
    raw_tags = request.form.get("tagIds", "[]")
    try:
        tag_ids = json.loads(raw_tags)
    except json.JSONDecodeError as exc:
        raise ValueError("tagIds must be a JSON string array") from exc
    body: dict[str, object] = {
        "title": request.form.get("title", ""),
        "tagIds": tag_ids,
    }
    if update:
        body["clearCover"] = request.form.get("clearCover", "false")
    return body


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_object(
    body: dict[str, object],
    key: str,
) -> dict[str, object] | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_string_array(
    body: dict[str, object],
    key: str,
) -> list[str] | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{key} must be a string array")
    return value
