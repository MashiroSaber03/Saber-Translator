"""HTTP routes for the stage-2 content vertical slice."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, send_file
from sqlalchemy import Engine

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


def create_content_blueprint(*, data_root, engine: Engine) -> Blueprint:
    blueprint = Blueprint("content_v2", __name__, url_prefix="/api/v2")
    repository = ContentRepository(engine)
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
                )
            }
        )

    @blueprint.post("/books")
    def create_book() -> tuple[Response, int]:
        _require_idempotency_key()
        body = _json_body()
        created = repository.create_book(title=str(body.get("title", "")))
        return jsonify(created), 201

    @blueprint.get("/books/<book_id>/chapters")
    def list_chapters(book_id: str) -> Response:
        return jsonify(repository.list_chapters(book_id))

    @blueprint.post("/books/<book_id>/chapters")
    def create_chapter(book_id: str) -> tuple[Response, int]:
        _require_idempotency_key()
        body = _json_body()
        created = repository.create_chapter(
            book_id=book_id,
            title=str(body.get("title", "")),
        )
        return jsonify(created), 201

    @blueprint.put("/books/<book_id>/chapters/order")
    def reorder_chapters(book_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body()
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

    @blueprint.get("/pages/<page_id>/document")
    def get_page_document(page_id: str) -> Response:
        return jsonify(repository.get_page_document(page_id))

    @blueprint.patch("/pages/<page_id>/document/batch")
    def mutate_page_document(page_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body()
        mutations = body.get("mutations")
        if not isinstance(mutations, list):
            raise ValueError("mutations must be an array")
        return jsonify(
            repository.mutate_page_document(
                page_id=page_id,
                base_revision=int(body.get("baseRevision", 0)),
                mutations=mutations,
            )
        )

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
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        logical_path = request.form.get("logicalPath") or upload.filename or ""
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
        )
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @blueprint.post("/quick-workspace/reset")
    def reset_quick_workspace() -> Response:
        _require_idempotency_key()
        return jsonify(repository.reset_quick_workspace())

    @blueprint.post("/quick-workspace/promote")
    def promote_quick_workspace() -> Response:
        _require_idempotency_key()
        body = _json_body()
        return jsonify(
            repository.promote_quick_workspace(
                chapter_title=str(body.get("chapterTitle", "")),
                new_book_title=(
                    str(body["newBookTitle"])
                    if body.get("newBookTitle") is not None
                    else None
                ),
                target_book_id=(
                    str(body["targetBookId"])
                    if body.get("targetBookId") is not None
                    else None
                ),
            )
        )

    return blueprint


def _json_body() -> dict[str, object]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _require_idempotency_key() -> str:
    value = request.headers.get("Idempotency-Key", "")
    if not value or len(value) > 200:
        raise ValueError("Idempotency-Key is required and must be at most 200 characters")
    return value


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status
