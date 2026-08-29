"""HTTP surface for resumable webpage-import drafts."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    integer_value as _integer_value,
    json_body as _json_body,
    require_idempotency_key as _require_idempotency_key,
    required_string as _required_string,
)
from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.web_import.commands import (
    DraftLocked,
    WebImportCommandService,
    _validated_url,
)


def create_web_import_blueprint(
    *,
    data_root: Path,
    engine: Engine,
) -> Blueprint:
    blueprint = Blueprint(
        "web_import_v2",
        __name__,
        url_prefix="/api/v2/web-import",
    )
    service = WebImportCommandService(data_root=data_root, engine=engine)

    @blueprint.errorhandler(DraftLocked)
    def locked(error: DraftLocked):
        return _error("draft_locked", str(error), 423)

    @blueprint.errorhandler(JobConflict)
    def conflict(error: JobConflict):
        return _error("draft_conflict", str(error), 409)

    @blueprint.errorhandler(LookupError)
    def not_found(error: LookupError):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.post("/support-checks")
    def support_check() -> Response:
        body = _json_body(allowed_keys={"sourceUrl"})
        source_url = _required_string(body, "sourceUrl")
        _validated_url(source_url)
        try:
            from gallery_dl import extractor

            gallery_dl_available = True
            gallery_dl_supported = extractor.find(source_url) is not None
        except (ImportError, RuntimeError):
            gallery_dl_available = False
            gallery_dl_supported = False
        # This endpoint is intentionally side-effect free.  The definitive
        # engine choice remains a Worker fact recorded on the draft.
        return jsonify(
            {
                "sourceUrl": source_url,
                "galleryDlAvailable": gallery_dl_available,
                "galleryDlSupported": gallery_dl_supported,
                "recommendedEngine": (
                    "gallery-dl" if gallery_dl_supported else "auto"
                ),
            }
        )

    @blueprint.post("/drafts")
    def create_draft():
        body = _json_body(
            allowed_keys={"chapterId", "sourceUrl", "engine", "textStyle"}
        )
        requested_engine = body.get("engine", "auto")
        if not isinstance(requested_engine, str):
            raise ValueError("engine must be a string")
        text_style = body.get("textStyle")
        if not isinstance(text_style, dict):
            raise ValueError("textStyle must be an object")
        result = service.create_draft(
            chapter_id=_required_string(body, "chapterId"),
            source_url=_required_string(body, "sourceUrl"),
            requested_engine=requested_engine,
            idempotency_key=_require_idempotency_key(),
            text_style=text_style,
        )
        return jsonify(result), 202

    @blueprint.get("/drafts/<draft_id>")
    def get_draft(draft_id: str) -> Response:
        return jsonify(service.get_draft(draft_id))

    @blueprint.get("/drafts/<draft_id>/pages")
    def list_pages(draft_id: str) -> Response:
        return jsonify(
            service.list_draft_pages(
                draft_id=draft_id,
                after_ordinal=_integer_value(
                    request.args.get("cursor", "0"),
                    "cursor",
                    minimum=0,
                ),
                limit=_integer_value(
                    request.args.get("limit", "50"),
                    "limit",
                    minimum=1,
                    maximum=200,
                ),
            )
        )

    @blueprint.put("/drafts/<draft_id>/selection")
    def update_selection(draft_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={"selectedPageIds", "baseRevision"}
        )
        selected = body.get("selectedPageIds")
        if not isinstance(selected, list) or not all(
            isinstance(value, str) for value in selected
        ):
            raise ValueError("selectedPageIds must be a string array")
        return jsonify(
            service.update_selection(
                draft_id=draft_id,
                selected_page_ids=selected,
                base_revision=_integer_value(
                    body.get("baseRevision"),
                    "baseRevision",
                    minimum=1,
                ),
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.post("/drafts/<draft_id>/commit")
    def commit(draft_id: str):
        body = _json_body(allowed_keys={"baseRevision"})
        result = service.commit(
            draft_id=draft_id,
            base_revision=_integer_value(
                body.get("baseRevision"),
                "baseRevision",
                minimum=1,
            ),
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.delete("/drafts/<draft_id>")
    def delete_draft(draft_id: str) -> Response:
        return jsonify(
            service.delete_draft(
                draft_id,
                idempotency_key=_require_idempotency_key(),
            )
        )

    @blueprint.get("/drafts/<draft_id>/media/<draft_page_id>")
    def draft_media(draft_id: str, draft_page_id: str):
        path, mime = service.media(
            draft_id=draft_id,
            page_id=draft_page_id,
            variant=request.args.get("variant", ""),
        )
        response = send_file(
            path,
            mimetype=mime,
            conditional=True,
            max_age=3600,
        )
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    return blueprint
