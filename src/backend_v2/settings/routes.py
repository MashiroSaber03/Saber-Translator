"""Atomic settings transactions and immediately persisted resource libraries."""

from __future__ import annotations

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
    ProviderSettingMutation,
    RevisionConflict,
    SettingMutation,
    SettingsRepository,
)

LOGGER = logging.getLogger("saber.api.settings")

_DIAGNOSTIC_FIELDS = frozenset(
    {
        "provider",
        "domain",
        "baseUrl",
        "model",
        "prompt",
        "secret",
        "credentialId",
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

    @blueprint.get("/provider-settings")
    def get_provider_settings() -> Response:
        domains = tuple(
            value
            for value in request.args.get("domains", "").split(",")
            if value
        )
        loaded = settings.load(domains=domains)
        return jsonify({"items": loaded["providerSettings"]})

    @blueprint.get("/credentials")
    def list_credentials() -> Response:
        return jsonify({"items": settings.credential_summaries()})

    @blueprint.delete("/credentials/<credential_id>")
    def delete_credential(credential_id: str) -> Response:
        _require_idempotency_key()
        settings.delete_credential(credential_id)
        return jsonify({"deleted": True})

    @blueprint.put("/settings/transactions")
    def save_settings_transaction() -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={
                "settings",
                "bookSettings",
                "providerSettings",
                "credentialEdits",
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
                        str(row["credentialVersionId"])
                        if row.get("credentialVersionId") is not None
                        else None
                    ),
                    credential_edit_ref=(
                        str(row["credentialEditRef"])
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
                        str(row["credentialId"])
                        if row.get("credentialId") is not None
                        else None
                    ),
                    client_ref=(
                        str(row["clientRef"])
                        if row.get("clientRef") is not None
                        else None
                    ),
                )
                for row in credential_rows
            ),
        )
        LOGGER.info(
            "设置事务已保存：domains=%s book_settings=%s providers=%s "
            "credentials=%s replayed=%s",
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
            replayed,
        )
        response = jsonify(result)
        if replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return response

    @blueprint.patch("/settings/workflow-preferences")
    def update_workflow_preferences() -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"payload", "baseRevision"})
        result = settings.save_transaction(
            settings=(
                SettingMutation(
                    domain="workflow_preferences",
                    payload=_required_object(body, "payload"),
                    base_revision=_required_integer(body, "baseRevision", minimum=0),
                    schema_version=1,
                ),
            )
        )
        return jsonify(result["settings"][0])

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
        _require_idempotency_key()
        body = _json_body(allowed_keys={"type", "name", "content"})
        return (
            jsonify(
                prompt_repository.create(
                    prompt_type=_required_string(body, "type"),
                    name=_required_string(body, "name"),
                    content=str(body.get("content", "")),
                )
            ),
            201,
        )

    @blueprint.put("/prompts/<prompt_id>")
    def update_prompt(prompt_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(
            allowed_keys={"name", "content", "baseRevision"}
        )
        return jsonify(
            prompt_repository.update(
                prompt_id=prompt_id,
                name=_required_string(body, "name"),
                content=str(body.get("content", "")),
                base_revision=int(body.get("baseRevision", 0)),
            )
        )

    @blueprint.delete("/prompts/<prompt_id>")
    def delete_prompt(prompt_id: str) -> Response:
        _require_idempotency_key()
        prompt_repository.delete(prompt_id)
        return jsonify({"deleted": True})

    @blueprint.post("/prompts/<prompt_id>/reset")
    def reset_prompt(prompt_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision"})
        return jsonify(
            prompt_repository.reset(
                prompt_id,
                base_revision=int(body.get("baseRevision", 0)),
            )
        )

    @blueprint.get("/fonts")
    def list_fonts() -> Response:
        return jsonify({"items": font_repository.list()})

    @blueprint.post("/fonts")
    def upload_font() -> tuple[Response, int]:
        _require_idempotency_key()
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
        try:
            font = (
                TTCollection(BytesIO(payload), lazy=True)
                if suffix == ".ttc"
                else TTFont(BytesIO(payload), lazy=True)
            )
            font.close()
        except Exception as exc:
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
        font_id = font_repository.register_uploaded(
            asset_id=asset.id,
            display_name=(
                request.form.get("displayName")
                or Path(upload.filename or "font").stem
            ),
        )
        return jsonify({"id": font_id, "assetUrl": f"/api/v2/assets/{asset.id}"}), 201

    @blueprint.delete("/fonts/<font_id>")
    def delete_font(font_id: str) -> Response:
        _require_idempotency_key()
        font_repository.delete_uploaded(font_id)
        return jsonify({"deleted": True})

    @blueprint.post("/maintenance/clean-temp")
    def clean_temp() -> Response:
        _require_idempotency_key()
        recovered = storage.recover_journal()
        return jsonify({"recovered": recovered})

    @blueprint.post("/maintenance/clean-debug")
    def clean_debug() -> Response:
        _require_idempotency_key()
        # Debug cleanup is intentionally owned by the backend.  Debug outputs
        # are not business facts, and physical removal is added when v2 debug
        # producers are introduced.
        return jsonify({"removed": 0})

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
