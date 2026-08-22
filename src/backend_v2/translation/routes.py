"""Specific translation task creation endpoints."""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    json_body as _json_body,
    required_integer as _required_integer,
    required_string as _required_string,
    require_idempotency_key as _require_idempotency_key,
    validate_multipart_fields as _validate_multipart_fields,
)
from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.translation.commands import TranslationJobCommandService
from src.backend_v2.translation.auxiliary import AuxiliaryTranslationCommands
from src.backend_v2.public_policy import PublicUserPolicyAccess
from src.backend_v2.runtime_profile import RuntimeProfile


def create_translation_blueprint(
    *,
    engine: Engine,
    profile: RuntimeProfile,
) -> Blueprint:
    blueprint = Blueprint("translation_v2", __name__, url_prefix="/api/v2")
    public_access = PublicUserPolicyAccess(engine, profile)
    service = TranslationJobCommandService(engine, profile=profile)
    auxiliary = AuxiliaryTranslationCommands(engine, profile=profile)

    @blueprint.before_request
    def require_translation_access() -> None:
        public_access.require_feature("translation")

    @blueprint.errorhandler(JobConflict)
    def conflict(error: JobConflict):
        return _error("job_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.post("/chapters/<chapter_id>/translation-jobs")
    def create_chapter_translation(chapter_id: str):
        body = _json_body(allowed_keys={"config", "pageIds"})
        config = body.get("config")
        page_ids = body.get("pageIds")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        if page_ids is not None and (
            not isinstance(page_ids, list)
            or not all(isinstance(page_id, str) for page_id in page_ids)
        ):
            raise ValueError("pageIds must be a string array")
        result = service.create_chapter_job(
            chapter_id=chapter_id,
            config=config,
            page_ids=page_ids,
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.post("/translation-batches")
    def create_translation_batch():
        body = _json_body(allowed_keys={"bookIds", "chapterIds", "config"})
        chapter_ids = body.get("chapterIds")
        book_ids = body.get("bookIds")
        config = body.get("config")
        if (chapter_ids is None) == (book_ids is None):
            raise ValueError("provide exactly one of chapterIds or bookIds")
        if chapter_ids is not None and (
            not isinstance(chapter_ids, list)
            or not all(isinstance(chapter_id, str) for chapter_id in chapter_ids)
        ):
            raise ValueError("chapterIds must be a string array")
        if book_ids is not None and (
            not isinstance(book_ids, list)
            or not all(isinstance(book_id, str) for book_id in book_ids)
        ):
            raise ValueError("bookIds must be a string array")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        result = service.create_batch(
            chapter_ids=chapter_ids,
            book_ids=book_ids,
            config=config,
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.post("/chapters/<chapter_id>/remove-text-jobs")
    def create_remove_text_job(chapter_id: str):
        body = _json_body(
            allowed_keys={
                "executionMode",
                "pageIds",
                "styleSourcePageId",
                "styleSourceDocumentRevision",
            }
        )
        page_ids = body.get("pageIds")
        if page_ids is not None and (
            not isinstance(page_ids, list)
            or not all(isinstance(value, str) for value in page_ids)
        ):
            raise ValueError("pageIds must be a string array")
        config = {
            "mode": "remove_text",
            "executionMode": body.get("executionMode", "sequential"),
        }
        for key in ("styleSourcePageId", "styleSourceDocumentRevision"):
            if key in body:
                config[key] = body[key]
        result = service.create_chapter_job(
            chapter_id=chapter_id,
            config=config,
            page_ids=page_ids,
            idempotency_key=_require_idempotency_key(),
            idempotency_scope=f"chapter-remove-text:{chapter_id}",
        )
        return jsonify(result), 202

    @blueprint.post("/chapters/<chapter_id>/detect-jobs")
    def create_detect_job(chapter_id: str):
        body = _json_body(allowed_keys={"pageIds"})
        page_ids = body.get("pageIds")
        if page_ids is not None and (
            not isinstance(page_ids, list)
            or not all(isinstance(value, str) for value in page_ids)
        ):
            raise ValueError("pageIds must be a string array")
        return (
            jsonify(
                auxiliary.create_detect_job(
                    chapter_id=chapter_id,
                    page_ids=page_ids,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/chapters/<chapter_id>/style-apply-jobs")
    def create_style_apply_job(chapter_id: str):
        public_access.require_feature("editMode")
        body = _json_body(
            allowed_keys={
                "sourcePageId",
                "sourceDocumentRevision",
                "selectedFields",
            }
        )
        selected = body.get("selectedFields")
        if not isinstance(selected, list) or not all(
            isinstance(value, str) for value in selected
        ):
            raise ValueError("selectedFields must be a string array")
        return (
            jsonify(
                auxiliary.create_style_apply_job(
                    chapter_id=chapter_id,
                    source_page_id=_required_string(body, "sourcePageId"),
                    source_document_revision=_required_integer(
                        body,
                        "sourceDocumentRevision",
                        minimum=1,
                    ),
                    selected_fields=selected,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.get("/chapters/<chapter_id>/text-export")
    def export_text(chapter_id: str):
        payload = json.dumps(
            auxiliary.export_text(chapter_id),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response = Response(
            payload,
            content_type="application/json; charset=utf-8",
        )
        response.headers["Content-Disposition"] = (
            f'attachment; filename="chapter-{chapter_id}-text.json"'
        )
        return response

    @blueprint.post("/chapters/<chapter_id>/text-import/preview")
    def preview_text_import(chapter_id: str) -> Response:
        _validate_multipart_fields(allowed_file_keys={"file"})
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        payload = upload.stream.read()
        if not payload:
            raise ValueError("text import is empty")
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("text import must be UTF-8 JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("text import root must be an object")
        return jsonify(
            auxiliary.preview_text_import(
                chapter_id=chapter_id,
                document=document,
            )
        )

    @blueprint.post("/chapters/<chapter_id>/text-import/commit")
    def commit_text_import(chapter_id: str):
        body = _json_body(allowed_keys={"confirmedPages"})
        confirmed = body.get("confirmedPages")
        if not isinstance(confirmed, list) or not all(
            isinstance(value, dict) for value in confirmed
        ):
            raise ValueError("confirmedPages must be an object array")
        result = auxiliary.create_text_import_job(
            chapter_id=chapter_id,
            confirmed_pages=confirmed,
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    return blueprint
