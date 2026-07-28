"""Specific translation task creation endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.translation.commands import TranslationJobCommandService


def create_translation_blueprint(*, engine: Engine) -> Blueprint:
    blueprint = Blueprint("translation_v2", __name__, url_prefix="/api/v2")
    service = TranslationJobCommandService(engine)

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
