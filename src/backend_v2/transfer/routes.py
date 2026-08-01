"""HTTP commands for backend container imports and durable exports."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    require_idempotency_key as _require_idempotency_key,
)
from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.transfer.commands import TransferCommandService


def create_transfer_blueprint(*, data_root: Path, engine: Engine) -> Blueprint:
    blueprint = Blueprint("transfer_v2", __name__, url_prefix="/api/v2")
    service = TransferCommandService(data_root=data_root, engine=engine)

    @blueprint.errorhandler(JobConflict)
    def conflict(error: JobConflict):
        return _error("job_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.post("/chapters/<chapter_id>/container-import-jobs")
    def create_container_import(chapter_id: str):
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        result = service.create_container_import(
            chapter_id=chapter_id,
            upload=upload.stream,
            filename=upload.filename or "container",
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.post("/chapters/<chapter_id>/export-jobs")
    def create_export(chapter_id: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        page_ids = body.get("pageIds")
        if page_ids is not None and (
            not isinstance(page_ids, list)
            or not all(isinstance(value, str) for value in page_ids)
        ):
            raise ValueError("pageIds must be a string array")
        result = service.create_export(
            chapter_id=chapter_id,
            export_format=str(body.get("format", "")),
            page_ids=page_ids,
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    return blueprint
