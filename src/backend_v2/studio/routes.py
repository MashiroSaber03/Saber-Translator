"""Character Studio v2 HTTP commands and queries."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.insight.derived import InsightDerivedRepository
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.studio.repository import (
    StudioBusy,
    StudioConflict,
    StudioNotFound,
    StudioRepository,
)
from src.backend_v2.studio.pure import (
    build_export_bundle,
    import_document_payload,
    build_diagnostics_report,
)


def create_studio_blueprint(*, engine: Engine) -> Blueprint:
    blueprint = Blueprint("studio_v2", __name__, url_prefix="/api/v2/studio")
    repository = StudioRepository(engine)
    settings = SettingsResolver(engine)
    derived = InsightDerivedRepository(engine)

    @blueprint.errorhandler(StudioNotFound)
    def not_found(error: StudioNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(StudioBusy)
    def busy(error: StudioBusy):
        return _error("session_busy", str(error), 423)

    @blueprint.errorhandler(StudioConflict)
    def conflict(error: StudioConflict):
        return _error("revision_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.get("/books/<book_id>/index")
    def index(book_id: str) -> Response:
        payload = repository.index(book_id=book_id)
        timeline = derived.get_timeline(book_id=book_id)
        payload["candidateStatus"] = {
            "available": bool(
                timeline
                and timeline.get("status") in {"ready", "degraded"}
                and timeline.get("mode") == "enhanced"
            ),
            "reason": (
                None
                if timeline
                and timeline.get("status") in {"ready", "degraded"}
                and timeline.get("mode") == "enhanced"
                else "enhanced_timeline_missing_or_stale"
            ),
        }
        return jsonify(payload)

    @blueprint.get("/books/<book_id>/candidates")
    def candidates(book_id: str) -> Response:
        timeline = derived.get_timeline(book_id=book_id)
        if timeline is None or timeline.get("status") not in {
            "ready",
            "degraded",
        }:
            return jsonify(
                {
                    "available": False,
                    "reason": "timeline_missing_or_stale",
                    "items": [],
                }
            )
        return jsonify(
            {
                "available": timeline.get("mode") == "enhanced",
                "reason": (
                    None
                    if timeline.get("mode") == "enhanced"
                    else "enhanced_timeline_required"
                ),
                "items": timeline.get("characters", []),
            }
        )

    @blueprint.post("/books/<book_id>/documents")
    def create_document(book_id: str):
        body = _json_body()
        title = _required_string(body, "title")
        document = body.get("document")
        if document is not None and not isinstance(document, dict):
            raise ValueError("document must be an object")
        return (
            jsonify(
                repository.create_document(
                    book_id=book_id,
                    title=title,
                    document=document,
                    kind=str(body.get("kind", "manual")),
                )
            ),
            201,
        )

    @blueprint.get("/documents/<document_id>")
    def get_document(document_id: str) -> Response:
        return jsonify(repository.get_document(document_id))

    @blueprint.put("/documents/<document_id>")
    def update_document(document_id: str) -> Response:
        body = _json_body()
        document = body.get("document")
        if not isinstance(document, dict):
            raise ValueError("document must be an object")
        return jsonify(
            repository.update_document(
                document_id=document_id,
                base_revision=int(body.get("baseRevision", 0)),
                title=(
                    str(body["title"])
                    if body.get("title") is not None
                    else None
                ),
                document=document,
            )
        )

    @blueprint.delete("/documents/<document_id>")
    def delete_document(document_id: str) -> Response:
        repository.delete_document(document_id)
        return jsonify({"deleted": True})

    @blueprint.post("/documents/<document_id>/generate")
    def generate(document_id: str):
        body = _json_body()
        document = repository.get_document(document_id)
        response = repository.create_generate_operation(
            document_id=document_id,
            base_revision=int(body.get("baseRevision", 0)),
            section=_required_string(body, "section"),
            config=settings.resolve_insight(
                book_id=str(document["bookId"]),
                command={"scope": "full", "force": False},
            ),
            idempotency_key=_idempotency_key(),
        )
        return jsonify(response), 202

    @blueprint.post("/documents/<document_id>/validate")
    def validate(document_id: str) -> Response:
        body = _json_body()
        document = repository.get_document(document_id)
        base_revision = int(body.get("baseRevision", 0))
        if int(document["revision"]) != base_revision:
            raise StudioConflict("studio document revision changed")
        report = build_diagnostics_report(document)
        updated = dict(document)
        updated["lastDiagnostics"] = report
        saved = repository.update_document(
            document_id=document_id,
            base_revision=base_revision,
            title=str(document["title"]),
            document=updated,
        )
        return jsonify(
            {
                "documentRevision": saved["revision"],
                "diagnostics": report,
            }
        )

    @blueprint.get("/documents/<document_id>/chat")
    def chat_state(document_id: str) -> Response:
        return jsonify(repository.chat_state(document_id))

    @blueprint.post("/documents/<document_id>/chat/sessions")
    def create_session(document_id: str):
        body = _json_body()
        return (
            jsonify(
                repository.create_session(
                    document_id=document_id,
                    title=str(body.get("title", "")).strip(),
                    greeting=(
                        str(body["greeting"])
                        if body.get("greeting") is not None
                        else None
                    ),
                )
            ),
            201,
        )

    @blueprint.get("/chat/sessions/<session_id>")
    def get_session(session_id: str) -> Response:
        return jsonify(repository.get_session(session_id))

    @blueprint.post("/chat/sessions/<session_id>/activate")
    def activate_session(session_id: str) -> Response:
        _json_body()
        return jsonify(repository.activate_session(session_id))

    @blueprint.delete("/chat/sessions/<session_id>")
    def delete_session(session_id: str) -> Response:
        revision = int(
            request.headers.get(
                "If-Match",
                request.args.get("baseRevision", "0"),
            )
        )
        repository.delete_session(
            session_id=session_id,
            base_revision=revision,
        )
        return jsonify({"deleted": True})

    @blueprint.post("/chat/sessions/<session_id>/messages")
    def send_message(session_id: str):
        body = _json_body()
        session = repository.get_session(session_id)
        document = repository.get_document(str(session["documentId"]))
        asset_ids = body.get("assetIds", [])
        if not isinstance(asset_ids, list) or not all(
            isinstance(value, str) for value in asset_ids
        ):
            raise ValueError("assetIds must be a string array")
        response = repository.send_message(
            session_id=session_id,
            base_revision=int(body.get("baseSessionRevision", 0)),
            content=_required_string(body, "content"),
            asset_ids=asset_ids,
            config=settings.resolve_insight(
                book_id=str(document["bookId"]),
                command={"scope": "full", "force": False},
            ),
            idempotency_key=_idempotency_key(),
        )
        return jsonify(response), 202

    @blueprint.post("/chat/sessions/<session_id>/summarize")
    def summarize(session_id: str):
        body = _json_body()
        session = repository.get_session(session_id)
        document = repository.get_document(str(session["documentId"]))
        response = repository.create_summary_operation(
            session_id=session_id,
            base_revision=int(body.get("baseSessionRevision", 0)),
            config=settings.resolve_insight(
                book_id=str(document["bookId"]),
                command={"scope": "full", "force": False},
            ),
            idempotency_key=_idempotency_key(),
        )
        return jsonify(response), 202

    @blueprint.post("/chat/sessions/<session_id>/abort")
    def abort(session_id: str) -> Response:
        body = _json_body()
        return jsonify(
            repository.abort(
                session_id=session_id,
                operation_id=_required_string(body, "operationId"),
            )
        )

    @blueprint.put("/chat/messages/<message_id>")
    def edit_message(message_id: str):
        body = _json_body()
        session_id = str(
            repository.get_session(
                _session_id_for_message(repository, message_id)
            )["sessionId"]
        )
        session = repository.get_session(session_id)
        document = repository.get_document(str(session["documentId"]))
        return (
            jsonify(
                repository.edit_or_regenerate_message(
                    message_id=message_id,
                    base_revision=int(
                        body.get("baseSessionRevision", 0)
                    ),
                    content=_required_string(body, "content"),
                    config=settings.resolve_insight(
                        book_id=str(document["bookId"]),
                        command={"scope": "full", "force": False},
                    ),
                    idempotency_key=_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/chat/messages/<message_id>/regenerate")
    def regenerate_message(message_id: str):
        body = _json_body()
        session_id = _session_id_for_message(repository, message_id)
        session = repository.get_session(session_id)
        document = repository.get_document(str(session["documentId"]))
        return (
            jsonify(
                repository.edit_or_regenerate_message(
                    message_id=message_id,
                    base_revision=int(
                        body.get("baseSessionRevision", 0)
                    ),
                    content=None,
                    config=settings.resolve_insight(
                        book_id=str(document["bookId"]),
                        command={"scope": "full", "force": False},
                    ),
                    idempotency_key=_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.delete("/chat/messages/<message_id>")
    def delete_message(message_id: str) -> Response:
        body = _json_body()
        return jsonify(
            repository.delete_message_chain(
                message_id=message_id,
                base_revision=int(
                    body.get("baseSessionRevision", 0)
                ),
            )
        )

    @blueprint.get("/documents/<document_id>/export")
    def export_document(document_id: str) -> Response:
        document = repository.get_document(document_id)
        output_format = request.args.get("format", "v3")
        bundle = build_export_bundle(document)
        if output_format not in {"v2", "v3", "worldbook"}:
            raise ValueError("format must be v2, v3, or worldbook")
        response = Response(
            json.dumps(
                bundle[output_format],
                ensure_ascii=False,
                indent=2,
            ),
            content_type="application/json; charset=utf-8",
        )
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{document["title"]}-{output_format}.json"'
        )
        return response

    @blueprint.post("/books/<book_id>/imports")
    def import_document(book_id: str):
        body = _json_body()
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        imported = import_document_payload(book_id, payload)
        return (
            jsonify(
                repository.create_document(
                    book_id=book_id,
                    title=str(imported["identity"]["name"]),
                    document=imported,
                    kind="imported",
                )
            ),
            201,
        )

    return blueprint


def _json_body() -> dict[str, Any]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _required_string(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _idempotency_key() -> str:
    value = request.headers.get("Idempotency-Key", "")
    if not value or len(value) > 200:
        raise ValueError("Idempotency-Key is required")
    return value


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def _session_id_for_message(
    repository: StudioRepository,
    message_id: str,
) -> str:
    with repository.engine.connect() as connection:
        from src.backend_v2.storage.schema import studio_messages
        from sqlalchemy import select

        value = connection.execute(
            select(studio_messages.c.session_id).where(
                studio_messages.c.id == message_id
            )
        ).scalar_one_or_none()
    if value is None:
        raise StudioNotFound("studio message not found")
    return str(value)
