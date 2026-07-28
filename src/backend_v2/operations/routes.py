"""Public operation creation and durable status routes."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.operations.repository import (
    OperationConflict,
    OperationLocked,
    OperationNotFound,
    OperationRepository,
)


def create_operations_blueprint(*, data_root, engine: Engine) -> Blueprint:
    blueprint = Blueprint("operations_v2", __name__, url_prefix="/api/v2")
    repository = OperationRepository(engine)
    from src.backend_v2.operations.repair import PageRepairService

    repairs = PageRepairService(
        data_root=data_root,
        engine=engine,
        repository=repository,
    )

    @blueprint.errorhandler(OperationNotFound)
    def not_found(error: OperationNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(OperationLocked)
    def locked(error: OperationLocked):
        return _error(str(error), "chapter is reserved by backend work", 423)

    @blueprint.errorhandler(OperationConflict)
    def conflict(error: OperationConflict):
        return _error("operation_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.post("/pages/<page_id>/operations")
    def create_page_operation(page_id: str):
        body = _json_body()
        payload = body.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        response, replayed = repository.create_page_operation(
            page_id=page_id,
            kind=str(body.get("kind", "")),
            base_revision=int(body.get("baseRevision", 0)),
            bubble_id=(
                str(body["bubbleId"])
                if body.get("bubbleId") is not None
                else None
            ),
            payload=payload,
            idempotency_key=_require_idempotency_key(),
        )
        result = jsonify(response)
        result.headers["Idempotency-Replayed"] = (
            "true" if replayed else "false"
        )
        return result, 200 if replayed else 202

    @blueprint.get("/operations/<operation_id>")
    def get_operation(operation_id: str) -> Response:
        return jsonify(repository.get(operation_id))

    @blueprint.post("/pages/<page_id>/repairs")
    def create_repair(page_id: str):
        target = request.form.get("target", "")
        common = {
            "page_id": page_id,
            "base_revision": int(request.form.get("base_revision", "0")),
            "idempotency_key": _require_idempotency_key(),
        }
        if target == "bubble":
            bubble_id = request.form.get("bubble_id", "")
            if not bubble_id:
                raise ValueError("bubble_id is required")
            response, replayed = repairs.create_for_bubble(
                bubble_id=bubble_id,
                **common,
            )
        elif target == "mask":
            upload = request.files.get("mask")
            if upload is None:
                raise ValueError("mask file is required")
            response, replayed = repairs.create_for_mask(
                upload=upload.stream,
                method=request.form.get("method", ""),
                fill_color=request.form.get("fill_color"),
                **common,
            )
        else:
            raise ValueError("repair target must be bubble or mask")
        result = jsonify(response)
        result.headers["Idempotency-Replayed"] = (
            "true" if replayed else "false"
        )
        return result, 200 if replayed else 202

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
