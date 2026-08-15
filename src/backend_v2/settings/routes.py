"""Atomic settings transactions and immediately persisted resource libraries."""

from __future__ import annotations

import hashlib
from io import BytesIO
import logging
from pathlib import Path

from flask import Blueprint, Response, jsonify, request
from fontTools.ttLib import TTCollection, TTFont
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    json_body as _json_body,
    require_idempotency_key as _require_idempotency_key,
    required_integer as _required_integer,
    required_string as _required_string,
    validate_multipart_fields as _validate_multipart_fields,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.builtin_fonts import SUPPORTED_FONT_SUFFIXES
from src.backend_v2.settings.diagnostics import (
    CONNECTION_TEST_KINDS,
    ProviderDiagnostics,
)
from src.backend_v2.storage.platform_repositories import (
    BookSettingMutation,
    CredentialEdit,
    FontRepository,
    PromptRepository,
    PromptMutation,
    ProviderSettingMutation,
    RevisionConflict,
    SettingMutation,
    SettingsRepository,
)
from src.shared.memory_errors import is_memory_allocation_error

LOGGER = logging.getLogger("saber.api.settings")

_DIAGNOSTIC_FIELDS = frozenset(
    {
        "provider",
        "domain",
        "baseUrl",
        "model",
        "prompt",
        "secret",
    }
)


def create_settings_blueprint(*, data_root: Path, engine: Engine) -> Blueprint:
    blueprint = Blueprint("settings_v2", __name__, url_prefix="/api/v2")
    settings = SettingsRepository(engine)
    prompt_repository = PromptRepository(engine)
    font_repository = FontRepository(engine)
    storage = AssetStorageService(data_root, engine)
    diagnostics = ProviderDiagnostics(settings)

    @blueprint.errorhandler(RevisionConflict)
    def conflict(error: RevisionConflict):
        LOGGER.warning("设置保存发生版本冲突：%s", error)
        return _error("revision_conflict", str(error), 409)

    @blueprint.errorhandler(LookupError)
    def not_found(error: LookupError):
        LOGGER.warning("设置资源不存在：%s", error)
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        LOGGER.warning("设置请求校验失败：%s", error)
        return _error("validation_error", str(error), 422)

    @blueprint.get("/settings")
    def get_settings() -> Response:
        domains = tuple(
            value
            for value in request.args.get("domains", "").split(",")
            if value
        )
        return jsonify(
            settings.load(
                domains=domains,
                book_id=request.args.get("book_id"),
            )
        )

    @blueprint.delete("/credentials/<credential_id>")
    def delete_credential(credential_id: str) -> Response:
        result, replayed = settings.delete_credential_idempotent(
            idempotency_key=_require_idempotency_key(),
            credential_id=credential_id,
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.put("/settings/transactions")
    def save_settings_transaction() -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={
                "settings",
                "bookSettings",
                "providerSettings",
                "credentialEdits",
                "promptEdits",
            }
        )
        setting_rows = _object_array(
            body,
            "settings",
            allowed_keys={"domain", "payload", "baseRevision", "schemaVersion"},
        )
        book_setting_rows = _object_array(
            body,
            "bookSettings",
            allowed_keys={
                "bookId",
                "domain",
                "payload",
                "baseRevision",
                "schemaVersion",
            },
        )
        provider_rows = _object_array(
            body,
            "providerSettings",
            allowed_keys={
                "domain",
                "provider",
                "payload",
                "baseRevision",
                "credentialVersionId",
                "credentialEditRef",
                "schemaVersion",
            },
        )
        credential_rows = _object_array(
            body,
            "credentialEdits",
            allowed_keys={
                "domain",
                "provider",
                "secret",
                "baseRevision",
                "credentialId",
                "clientRef",
            },
        )
        prompt_rows = _object_array(
            body,
            "promptEdits",
            allowed_keys={"id", "name", "content", "baseRevision"},
        )
        if not any(
            (
                setting_rows,
                book_setting_rows,
                provider_rows,
                credential_rows,
                prompt_rows,
            )
        ):
            raise ValueError("settings transaction must contain at least one mutation")
        result, replayed = settings.save_transaction_idempotent(
            idempotency_key=idempotency_key,
            request_body=body,
            settings=tuple(
                SettingMutation(
                    domain=_required_string(row, "domain"),
                    payload=_required_object(row, "payload"),
                    base_revision=_required_integer(row, "baseRevision", minimum=0),
                    schema_version=_required_integer(row, "schemaVersion", minimum=1),
                )
                for row in setting_rows
            ),
            book_settings_edits=tuple(
                BookSettingMutation(
                    book_id=_required_string(row, "bookId"),
                    domain=_required_string(row, "domain"),
                    payload=_required_object(row, "payload"),
                    base_revision=_required_integer(row, "baseRevision", minimum=0),
                    schema_version=_required_integer(row, "schemaVersion", minimum=1),
                )
                for row in book_setting_rows
            ),
            providers=tuple(
                ProviderSettingMutation(
                    domain=_required_string(row, "domain"),
                    provider=_required_string(row, "provider"),
                    payload=_required_object(row, "payload"),
                    base_revision=_required_integer(row, "baseRevision", minimum=0),
                    credential_version_id=(
                        _required_string(row, "credentialVersionId")
                        if row.get("credentialVersionId") is not None
                        else None
                    ),
                    credential_edit_ref=(
                        _required_string(row, "credentialEditRef")
                        if row.get("credentialEditRef") is not None
                        else None
                    ),
                    schema_version=_required_integer(row, "schemaVersion", minimum=1),
                )
                for row in provider_rows
            ),
            credentials_edits=tuple(
                CredentialEdit(
                    domain=_required_string(row, "domain"),
                    provider=_required_string(row, "provider"),
                    secret=_required_object(row, "secret"),
                    base_revision=_required_integer(row, "baseRevision", minimum=0),
                    credential_id=(
                        _required_string(row, "credentialId")
                        if row.get("credentialId") is not None
                        else None
                    ),
                    client_ref=_required_string(row, "clientRef"),
                )
                for row in credential_rows
            ),
            prompt_edits=tuple(
                PromptMutation(
                    prompt_id=_required_string(row, "id"),
                    name=_required_string(row, "name"),
                    content=_required_text(row, "content"),
                    base_revision=_required_integer(
                        row,
                        "baseRevision",
                        minimum=1,
                    ),
                )
                for row in prompt_rows
            ),
        )
        LOGGER.info(
            "设置事务已保存：domains=%s book_settings=%s providers=%s "
            "credentials=%s prompts=%s replayed=%s",
            ",".join(str(row.get("domain", "?")) for row in setting_rows) or "-",
            len(book_setting_rows),
            ",".join(
                f"{row.get('domain', '?')}:{row.get('provider', '?')}"
                for row in provider_rows
            )
            or "-",
            ",".join(
                f"{row.get('domain', '?')}:{row.get('provider', '?')}"
                for row in credential_rows
            )
            or "-",
            len(prompt_rows),
            replayed,
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.patch("/settings/workflow-preferences")
    def update_workflow_preferences() -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"payload", "baseRevision"})
        result, replayed = settings.save_transaction_idempotent(
            idempotency_key=idempotency_key,
            request_body=body,
            settings=(
                SettingMutation(
                    domain="workflow_preferences",
                    payload=_required_object(body, "payload"),
                    base_revision=_required_integer(body, "baseRevision", minimum=0),
                    schema_version=1,
                ),
            )
        )
        response = jsonify(result["settings"][0])
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.post("/model-catalog")
    def model_catalog() -> Response:
        return jsonify(
            diagnostics.model_catalog(
                _json_body(allowed_keys=_DIAGNOSTIC_FIELDS)
            )
        )

    @blueprint.post("/connection-tests/<kind>")
    def connection_test(kind: str) -> Response:
        if kind not in CONNECTION_TEST_KINDS:
            raise ValueError("unsupported connection test kind")
        return jsonify(
            diagnostics.connection_test(
                kind,
                _json_body(allowed_keys=_DIAGNOSTIC_FIELDS),
            )
        )

    @blueprint.get("/prompts")
    def list_prompts() -> Response:
        return jsonify(
            {"items": prompt_repository.list(request.args.get("type"))}
        )

    @blueprint.post("/prompts")
    def create_prompt() -> tuple[Response, int]:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"type", "name", "content"})
        result, replayed = prompt_repository.create_idempotent(
            idempotency_key=idempotency_key,
            prompt_type=_required_string(body, "type"),
            name=_required_string(body, "name"),
            content=_required_text(body, "content"),
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response, 201

    @blueprint.put("/prompts/<prompt_id>")
    def update_prompt(prompt_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={"name", "content", "baseRevision"}
        )
        result, replayed = prompt_repository.update_idempotent(
            idempotency_key=idempotency_key,
            prompt_id=prompt_id,
            name=_required_string(body, "name"),
            content=_required_text(body, "content"),
            base_revision=_required_integer(
                body,
                "baseRevision",
                minimum=1,
            ),
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.delete("/prompts/<prompt_id>")
    def delete_prompt(prompt_id: str) -> Response:
        result, replayed = prompt_repository.delete_idempotent(
            idempotency_key=_require_idempotency_key(),
            prompt_id=prompt_id,
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.post("/prompts/<prompt_id>/reset")
    def reset_prompt(prompt_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision"})
        result, replayed = prompt_repository.reset_idempotent(
            idempotency_key=idempotency_key,
            prompt_id=prompt_id,
            base_revision=_required_integer(
                body,
                "baseRevision",
                minimum=1,
            ),
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.get("/fonts")
    def list_fonts() -> Response:
        return jsonify({"items": font_repository.list()})

    @blueprint.post("/fonts")
    def upload_font() -> tuple[Response, int]:
        idempotency_key = _require_idempotency_key()
        _validate_multipart_fields(
            allowed_form_keys={"displayName"},
            allowed_file_keys={"file"},
        )
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        payload = upload.stream.read()
        if not payload:
            raise ValueError("font is empty")
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_FONT_SUFFIXES:
            raise ValueError("font extension must be ttf, ttc, otf, woff, or woff2")
        raw_display_name = request.form.get("displayName")
        display_name = (
            Path(upload.filename or "font").stem
            if raw_display_name is None
            else raw_display_name.strip()
        )
        if not display_name:
            raise ValueError("font displayName must not be empty")
        idempotency_body = {
            "checksum": hashlib.sha256(payload).hexdigest(),
            "byteSize": len(payload),
            "extension": suffix,
            "displayName": display_name,
        }
        replay = font_repository.replay_upload(
            idempotency_key=idempotency_key,
            request_body=idempotency_body,
        )
        if replay is not None:
            response = jsonify(replay)
            response.headers["Idempotency-Replayed"] = "true"
            return response, 201
        try:
            font = (
                TTCollection(BytesIO(payload), lazy=True)
                if suffix == ".ttc"
                else TTFont(BytesIO(payload), lazy=True)
            )
            font.close()
        except Exception as exc:
            if is_memory_allocation_error(exc):
                raise
            raise ValueError("uploaded file is not a valid font") from exc
        mime_types = {
            ".ttf": "font/ttf",
            ".ttc": "font/collection",
            ".otf": "font/otf",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
        }
        asset = storage.publish_bytes(
            payload,
            extension=suffix[1:],
            mime_type=mime_types[suffix],
        )
        result, replayed = font_repository.register_uploaded_idempotent(
            idempotency_key=idempotency_key,
            request_body=idempotency_body,
            asset_id=asset.id,
            display_name=display_name,
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response, 201

    @blueprint.delete("/fonts/<font_id>")
    def delete_font(font_id: str) -> Response:
        result, replayed = font_repository.delete_uploaded_idempotent(
            idempotency_key=_require_idempotency_key(),
            font_id=font_id,
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.post("/maintenance/clean-temp")
    def clean_temp() -> Response:
        idempotency_key = _require_idempotency_key()
        scope = "POST:cleanTemporaryAssets"
        replay = settings.replay_idempotent_command(
            scope=scope,
            idempotency_key=idempotency_key,
            request_body={},
        )
        if replay is not None:
            response = jsonify(replay)
            response.headers["Idempotency-Replayed"] = "true"
            return response
        recovered = storage.recover_journal()
        result, replayed = settings.record_idempotent_command(
            scope=scope,
            idempotency_key=idempotency_key,
            request_body={},
            response={"recovered": recovered},
            resource_type="asset_maintenance",
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    return blueprint


def _object_array(
    body: dict[str, object],
    key: str,
    *,
    allowed_keys: set[str],
) -> list[dict[str, object]]:
    value = body.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise ValueError(f"{key} must be an object array")
    for index, row in enumerate(value):
        unknown = set(row) - allowed_keys
        if unknown:
            raise ValueError(
                f"{key}[{index}] contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
    return value


def _required_object(
    body: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = body.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _required_text(body: dict[str, object], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value
