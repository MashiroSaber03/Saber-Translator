"""Specific translation task creation endpoints."""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.translation.commands import TranslationJobCommandService
from src.backend_v2.translation.auxiliary import AuxiliaryTranslationCommands


def create_translation_blueprint(*, engine: Engine) -> Blueprint:
    blueprint = Blueprint("translation_v2", __name__, url_prefix="/api/v2")
    service = TranslationJobCommandService(engine)
    auxiliary = AuxiliaryTranslationCommands(engine)

    @blueprint.errorhandler(JobConflict)
    def conflict(error: JobConflict):
        return _error("job_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.post("/chapters/<chapter_id>/translation-jobs")
    def create_chapter_translation(chapter_id: str):
        body = _json_body()
        config = body.get("config", {})
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

    @blueprint.post("/job-batches/translation")
    def create_translation_batch():
        body = _json_body()
        chapter_ids = body.get("chapterIds")
        config = body.get("config", {})
        if not isinstance(chapter_ids, list) or not all(
            isinstance(chapter_id, str) for chapter_id in chapter_ids
        ):
            raise ValueError("chapterIds must be a string array")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        result = service.create_batch(
            chapter_ids=chapter_ids,
            config=config,
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.post("/chapters/<chapter_id>/detect-jobs")
    def create_detect_job(chapter_id: str):
        body = _json_body()
        detector = body.get("detector", {})
        page_ids = body.get("pageIds")
        if not isinstance(detector, dict):
            raise ValueError("detector must be an object")
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
                    detector=detector,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/chapters/<chapter_id>/style-apply-jobs")
    def create_style_apply_job(chapter_id: str):
        body = _json_body()
        selected = body.get("selectedFields")
        if not isinstance(selected, list) or not all(
            isinstance(value, str) for value in selected
        ):
            raise ValueError("selectedFields must be a string array")
        return (
            jsonify(
                auxiliary.create_style_apply_job(
                    chapter_id=chapter_id,
                    source_page_id=str(body.get("sourcePageId", "")),
                    source_document_revision=int(
                        body.get("sourceDocumentRevision", 0)
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
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        payload = upload.stream.read(32 * 1024 * 1024 + 1)
        if not payload or len(payload) > 32 * 1024 * 1024:
            raise ValueError("text import is empty or exceeds 32 MiB")
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
        body = _json_body()
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
