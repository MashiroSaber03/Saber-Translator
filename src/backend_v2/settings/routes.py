"""Atomic settings transactions and immediately persisted resource libraries."""

from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path

from flask import Blueprint, Response, jsonify, request
from fontTools.ttLib import TTFont
from sqlalchemy import Engine

from src.backend_v2.storage.assets import AssetStorageService
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
        body = _json_body()
        setting_rows = _object_array(body, "settings")
        book_setting_rows = _object_array(body, "bookSettings")
        provider_rows = _object_array(body, "providerSettings")
        credential_rows = _object_array(body, "credentialEdits")
        result, replayed = settings.save_transaction_idempotent(
            idempotency_key=idempotency_key,
            request_body=body,
            settings=tuple(
                SettingMutation(
                    domain=_required_string(row, "domain"),
                    payload=_required_object(row, "payload"),
                    base_revision=int(row.get("baseRevision", 0)),
                    schema_version=int(row.get("schemaVersion", 1)),
                )
                for row in setting_rows
            ),
            book_settings_edits=tuple(
                BookSettingMutation(
                    book_id=_required_string(row, "bookId"),
                    domain=_required_string(row, "domain"),
                    payload=_required_object(row, "payload"),
                    base_revision=int(row.get("baseRevision", 0)),
                    schema_version=int(row.get("schemaVersion", 1)),
                )
                for row in book_setting_rows
            ),
            providers=tuple(
                ProviderSettingMutation(
                    domain=_required_string(row, "domain"),
                    provider=_required_string(row, "provider"),
                    payload=_required_object(row, "payload"),
                    base_revision=int(row.get("baseRevision", 0)),
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
                    schema_version=int(row.get("schemaVersion", 1)),
                )
                for row in provider_rows
            ),
            credentials_edits=tuple(
                CredentialEdit(
                    domain=_required_string(row, "domain"),
                    provider=_required_string(row, "provider"),
                    secret=_required_object(row, "secret"),
                    base_revision=int(row.get("baseRevision", 0)),
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
        body = _json_body()
        result = settings.save_transaction(
            settings=(
                SettingMutation(
                    domain="workflow_preferences",
                    payload=_required_object(body, "payload"),
                    base_revision=int(body.get("baseRevision", 0)),
                ),
            )
        )
        return jsonify(result["settings"][0])

    @blueprint.post("/model-catalog")
    def model_catalog() -> Response:
        return jsonify(diagnostics.model_catalog(_json_body()))

    @blueprint.post("/connection-tests/<kind>")
    def connection_test(kind: str) -> Response:
        if kind not in CONNECTION_TEST_KINDS:
            raise ValueError("unsupported connection test kind")
        return jsonify(diagnostics.connection_test(kind, _json_body()))

    @blueprint.get("/prompts")
    def list_prompts() -> Response:
        return jsonify(
            {"items": prompt_repository.list(request.args.get("type"))}
        )

    @blueprint.post("/prompts")
    def create_prompt() -> tuple[Response, int]:
        _require_idempotency_key()
        body = _json_body()
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
        body = _json_body()
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
        body = _json_body()
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
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        payload = upload.stream.read(32 * 1024 * 1024 + 1)
        if not payload or len(payload) > 32 * 1024 * 1024:
            raise ValueError("font is empty or exceeds 32 MiB")
        try:
            font = TTFont(BytesIO(payload), lazy=True)
            font.close()
        except Exception as exc:
            raise ValueError("uploaded file is not a valid font") from exc
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in {".ttf", ".otf", ".woff", ".woff2"}:
            raise ValueError("font extension must be ttf, otf, woff, or woff2")
        mime_types = {
            ".ttf": "font/ttf",
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
        storage.collect_garbage()
        return jsonify({"deleted": True})

    @blueprint.post("/maintenance/clean-temp")
    def clean_temp() -> Response:
        _require_idempotency_key()
        recovered = storage.recover_journal(orphan_grace_seconds=0)
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
) -> list[dict[str, object]]:
    value = body.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise ValueError(f"{key} must be an object array")
    return value


def _required_object(
    body: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = body.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _required_string(body: dict[str, object], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _json_body() -> dict[str, object]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _require_idempotency_key() -> str:
    value = request.headers.get("Idempotency-Key", "")
    if not value or len(value) > 200:
        raise ValueError(
            "Idempotency-Key is required and must be at most 200 characters"
        )
    return value


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status
