"""HTTP surface for resumable webpage-import drafts."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file
from sqlalchemy import Engine

from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.web_import.commands import WebImportCommandService


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
        body = _json_body()
        source_url = str(body.get("sourceUrl", "")).strip()
        from src.backend_v2.web_import.commands import _validated_url

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
        body = _json_body()
        config = body.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        result = service.create_draft(
            chapter_id=_required_string(body, "chapterId"),
            source_url=_required_string(body, "sourceUrl"),
            requested_engine=str(body.get("engine", "auto")),
            config=config,
            idempotency_key=_require_idempotency_key(),
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
                after_ordinal=int(request.args.get("cursor", "0")),
                limit=int(request.args.get("limit", "50")),
            )
        )

    @blueprint.put("/drafts/<draft_id>/selection")
    def update_selection(draft_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body()
        selected = body.get("selectedPageIds")
        if not isinstance(selected, list) or not all(
            isinstance(value, str) for value in selected
        ):
            raise ValueError("selectedPageIds must be a string array")
        return jsonify(
            service.update_selection(
                draft_id=draft_id,
                selected_page_ids=selected,
                base_revision=int(body.get("baseRevision", 0)),
            )
        )

    @blueprint.post("/drafts/<draft_id>/commit")
    def commit(draft_id: str):
        body = _json_body()
        result = service.commit(
            draft_id=draft_id,
            base_revision=int(body.get("baseRevision", 0)),
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.delete("/drafts/<draft_id>")
    def delete_draft(draft_id: str) -> Response:
        _require_idempotency_key()
        service.delete_draft(draft_id)
        return jsonify({"deleted": True})

    @blueprint.get("/drafts/<draft_id>/pages/<page_id>/media")
    def draft_media(draft_id: str, page_id: str):
        path, mime = service.media(
            draft_id=draft_id,
            page_id=page_id,
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


def _required_string(body: dict[str, object], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _json_body() -> dict[str, object]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _require_idempotency_key() -> str:
    value = request.headers.get("Idempotency-Key", "")
    if not value or len(value) > 200:
        raise ValueError(
            "Idempotency-Key is required and must be at most 200 characters"
        )
    return value


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status
