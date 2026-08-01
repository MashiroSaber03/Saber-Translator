"""Metadata-only plugin v3 management API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    json_body as _json_body,
    require_idempotency_key as _idempotency_key,
    validate_multipart_fields as _validate_multipart_fields,
)
from src.backend_v2.plugins.package import MAX_ARCHIVE_BYTES
from src.backend_v2.plugins.repository import (
    PluginConflict,
    PluginLocked,
    PluginNotFound,
    PluginRegistry,
)


def create_plugins_blueprint(
    *,
    data_root: Path,
    engine: Engine,
) -> Blueprint:
    blueprint = Blueprint(
        "plugins_v2",
        __name__,
        url_prefix="/api/v2/plugins",
    )
    registry = PluginRegistry(data_root=data_root, engine=engine)

    @blueprint.errorhandler(PluginNotFound)
    def not_found(error: PluginNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(PluginLocked)
    def locked(error: PluginLocked):
        return _error("plugin_version_referenced", str(error), 423)

    @blueprint.errorhandler(PluginConflict)
    def conflict(error: PluginConflict):
        return _error(
            "revision_conflict",
            str(error),
            409,
            details=error.details,
        )

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.get("")
    def list_plugins() -> Response:
        return jsonify(registry.list_plugins())

    @blueprint.post("/refresh")
    def refresh() -> Response:
        return jsonify(registry.refresh())

    @blueprint.put("/<plugin_id>/runtime-enabled")
    def runtime_enabled(plugin_id: str) -> Response:
        body = _json_body(allowed_keys={"enabled"})
        return jsonify(
            registry.set_runtime_enabled(
                plugin_id=plugin_id,
                enabled=_required_bool(body, "enabled"),
            )
        )

    @blueprint.put("/<plugin_id>/default-enabled")
    def default_enabled(plugin_id: str) -> Response:
        body = _json_body(allowed_keys={"enabled"})
        return jsonify(
            registry.set_default_enabled(
                plugin_id=plugin_id,
                enabled=_required_bool(body, "enabled"),
            )
        )

    @blueprint.get("/<plugin_id>/config")
    def get_config(plugin_id: str) -> Response:
        return jsonify(registry.get_config(plugin_id))

    @blueprint.put("/<plugin_id>/config")
    def update_config(plugin_id: str) -> Response:
        body = _json_body(allowed_keys={"baseRevision", "config"})
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        return jsonify(
            registry.update_config(
                plugin_id=plugin_id,
                base_revision=_positive_int(
                    body.get("baseRevision"),
                    "baseRevision",
                ),
                config=config,
            )
        )

    @blueprint.post("/import")
    def import_plugin():
        _validate_multipart_fields(
            allowed_form_keys={"baseRevision"},
            allowed_file_keys={"file"},
        )
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("file is required")
        data = upload.stream.read(MAX_ARCHIVE_BYTES + 1)
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError("plugin archive is too large")
        base_revision = _positive_int(
            request.form.get("baseRevision"),
            "baseRevision",
            allow_zero=True,
        )
        result = registry.import_archive(
            data=data,
            base_revision=base_revision,
            idempotency_key=_idempotency_key(),
        )
        return jsonify(result), 201

    @blueprint.get("/<plugin_id>/export")
    def export_plugin(plugin_id: str) -> Response:
        data, filename = registry.export_current(plugin_id)
        response = Response(data, content_type="application/zip")
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{filename}"'
        )
        return response

    @blueprint.delete("/<plugin_id>")
    def delete_plugin(plugin_id: str) -> Response:
        return jsonify(
            registry.delete_plugin(
                plugin_id=plugin_id,
                base_revision=_positive_int(
                    request.headers.get("If-Match"),
                    "If-Match",
                ),
            )
        )

    return blueprint


def _required_bool(body: dict[str, Any], field: str) -> bool:
    value = body.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _positive_int(
    value: object,
    field: str,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized
