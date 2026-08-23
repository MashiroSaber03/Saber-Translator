"""HTTP commands for backend container imports and durable exports."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    json_body as _json_body,
    required_boolean as _required_boolean,
    require_idempotency_key as _require_idempotency_key,
    required_string as _required_string,
    validate_multipart_fields as _validate_multipart_fields,
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
        _validate_multipart_fields(allowed_file_keys={"file"})
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
        body = _json_body(
            allowed_keys={"format", "pageIds", "preserveOriginalFilenames"}
        )
        page_ids = body.get("pageIds")
        if page_ids is not None and (
            not isinstance(page_ids, list)
            or not all(isinstance(value, str) for value in page_ids)
        ):
            raise ValueError("pageIds must be a string array")
        result = service.create_export(
            chapter_id=chapter_id,
            export_format=_required_string(body, "format"),
            page_ids=page_ids,
            preserve_original_filenames=_required_boolean(
                body,
                "preserveOriginalFilenames",
            ),
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.post("/books/export-jobs")
    def create_books_export():
        body = _json_body(
            allowed_keys={"bookIds", "preserveOriginalFilenames"}
        )
        book_ids = body.get("bookIds")
        if (
            not isinstance(book_ids, list)
            or not book_ids
            or not all(isinstance(value, str) and value for value in book_ids)
        ):
            raise ValueError("bookIds must be a non-empty string array")
        result = service.create_books_export(
            book_ids=book_ids,
            preserve_original_filenames=_required_boolean(
                body,
                "preserveOriginalFilenames",
            ),
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    return blueprint
