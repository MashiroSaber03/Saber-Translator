"""Public operation creation and durable status routes."""

from __future__ import annotations

import json
import time
from typing import Iterator

from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    stream_with_context,
)
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    integer_value as _integer_value,
    json_body as _json_body,
    require_idempotency_key as _require_idempotency_key,
    required_integer as _required_integer,
    required_string as _required_string,
    validate_multipart_fields as _validate_multipart_fields,
)
from src.backend_v2.operations.repository import (
    OperationConflict,
    OperationLocked,
    OperationNotFound,
    OperationRepository,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.public_policy import PublicUserPolicyAccess
from src.backend_v2.runtime_profile import RuntimeProfile, resolve_runtime_profile


def create_operations_blueprint(
    *,
    data_root,
    engine: Engine,
    profile: RuntimeProfile | None = None,
) -> Blueprint:
    profile = profile or resolve_runtime_profile("local")
    blueprint = Blueprint("operations_v2", __name__, url_prefix="/api/v2")
    repository = OperationRepository(engine)
    settings = SettingsResolver(engine)
    public_access = PublicUserPolicyAccess(engine, profile)
    from src.backend_v2.operations.repair import PageRepairService

    repairs = PageRepairService(
        data_root=data_root,
        engine=engine,
        repository=repository,
        method_validator=public_access.require_inpaint_method,
        settings_transformer=public_access.apply_page_repair_settings,
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
        body = _json_body(
            allowed_keys={"kind", "baseRevision", "bubbleId"}
        )
        kind = _required_string(body, "kind")
        payload = settings.resolve_page_operation(
            page_id=page_id,
            kind=kind,
        )
        public_access.require_page_operation(kind, payload)
        bubble_id: str | None = None
        if "bubbleId" in body:
            bubble_id = _required_string(body, "bubbleId")
        response, replayed = repository.create_page_operation(
            page_id=page_id,
            kind=kind,
            base_revision=_required_integer(
                body,
                "baseRevision",
                minimum=1,
            ),
            bubble_id=bubble_id,
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

    @blueprint.get("/operations/<operation_id>/events")
    def get_operation_events(operation_id: str) -> Response:
        after = _integer_value(
            request.headers.get(
                "Last-Event-ID",
                request.args.get("after", "0"),
            ),
            "after",
            minimum=0,
        )
        stream = _integer_value(
            request.args.get("stream", "0"),
            "stream",
            minimum=0,
            maximum=1,
        )
        wants_stream = (
            stream == 1
            or "text/event-stream" in request.headers.get("Accept", "")
        )
        if not wants_stream:
            return jsonify(
                {
                    "items": repository.events_after(
                        operation_id,
                        after=after,
                        limit=_integer_value(
                            request.args.get("limit", "500"),
                            "limit",
                            minimum=1,
                            maximum=2000,
                        ),
                    )
                }
            )

        # Validate before the streaming response starts, so unknown operations
        # retain normal JSON 404 semantics.
        repository.get(operation_id)

        @stream_with_context
        def generate() -> Iterator[str]:
            cursor = after
            last_heartbeat = time.monotonic()
            yield "retry: 1000\n\n"
            while True:
                events = repository.events_after(
                    operation_id,
                    after=cursor,
                    limit=500,
                )
                for event in events:
                    cursor = int(event["eventId"])
                    yield (
                        f"id: {cursor}\n"
                        f"event: {event['type']}\n"
                        "data: "
                        + json.dumps(
                            event,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    )
                operation = repository.get(operation_id)
                if operation["status"] in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    return
                if time.monotonic() - last_heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                time.sleep(0.25)

        response = Response(
            generate(),
            content_type="text/event-stream; charset=utf-8",
        )
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @blueprint.post("/pages/<page_id>/repairs")
    def create_repair(page_id: str):
        public_access.require_feature("editMode")
        target = request.form.get("target", "")
        if target == "bubble":
            _validate_multipart_fields(
                allowed_form_keys={"target", "base_revision", "bubble_id"},
            )
            bubble_id = request.form.get("bubble_id", "")
            if not bubble_id:
                raise ValueError("bubble_id is required")
            response, replayed = repairs.create_for_bubble(
                bubble_id=bubble_id,
                page_id=page_id,
                base_revision=_integer_value(
                    request.form.get("base_revision"),
                    "base_revision",
                    minimum=1,
                ),
                idempotency_key=_require_idempotency_key(),
            )
        elif target == "mask":
            _validate_multipart_fields(
                allowed_form_keys={
                    "target",
                    "base_revision",
                    "method",
                    "fill_color",
                },
                allowed_file_keys={"mask"},
            )
            upload = request.files.get("mask")
            if upload is None:
                raise ValueError("mask file is required")
            response, replayed = repairs.create_for_mask(
                upload=upload.stream,
                method=request.form.get("method", ""),
                fill_color=request.form.get("fill_color"),
                page_id=page_id,
                base_revision=_integer_value(
                    request.form.get("base_revision"),
                    "base_revision",
                    minimum=1,
                ),
                idempotency_key=_require_idempotency_key(),
            )
        else:
            raise ValueError("repair target must be bubble or mask")
        result = jsonify(response)
        result.headers["Idempotency-Replayed"] = (
            "true" if replayed else "false"
        )
        return result, 200 if replayed else 202

    return blueprint
