"""Character Studio v2 HTTP commands and queries."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Mapping

from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    stream_with_context,
)
from sqlalchemy import Engine

from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.insight.derived import InsightDerivedRepository
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.studio.repository import (
    StudioBusy,
    StudioConflict,
    StudioNotFound,
    StudioRepository,
)
from src.backend_v2.studio.io import StudioIOService
from src.backend_v2.studio.service import StudioOperationService
from src.backend_v2.studio.pure import (
    build_export_bundle,
    import_document_payload,
)


def create_studio_blueprint(
    *,
    engine: Engine,
    data_root: Path,
) -> Blueprint:
    blueprint = Blueprint("studio_v2", __name__, url_prefix="/api/v2/studio")
    repository = StudioRepository(engine)
    settings = SettingsResolver(engine)
    derived = InsightDerivedRepository(engine)
    io_service = StudioIOService(
        data_root=data_root,
        engine=engine,
        repository=repository,
    )
    runtime = StudioOperationService(
        engine=engine,
        data_root=data_root,
        repository=repository,
    )

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
                "items": [
                    _candidate_item(character)
                    for character in timeline.get("characters", [])
                    if isinstance(character, Mapping)
                    and str(character.get("name", "")).strip()
                ],
            }
        )

    @blueprint.post("/books/<book_id>/documents")
    def create_document(book_id: str):
        idempotency_key = _idempotency_key()
        body = _json_body()
        candidate = body.get("candidate")
        title = str(
            body.get("title")
            or (
                candidate.get("name")
                if isinstance(candidate, dict)
                else ""
            )
            or ""
        ).strip()
        if not title:
            raise ValueError("title is required")
        document = body.get("document")
        if document is not None and not isinstance(document, dict):
            raise ValueError("document must be an object")
        if document is None and isinstance(candidate, dict):
            document = {
                "origin": {
                    "type": "analysis",
                    "source_character": candidate.get("name"),
                },
                "identity": {
                    "name": title,
                    "aliases": candidate.get("aliases", []),
                    "description": candidate.get(
                        "description",
                        candidate.get("summary", ""),
                    ),
                    "personality": candidate.get("personality", ""),
                    "scenario": "",
                },
            }
        return (
            jsonify(
                repository.create_document(
                    book_id=book_id,
                    title=title,
                    document=document,
                    kind=str(
                        body.get(
                            "kind",
                            "analysis"
                            if isinstance(candidate, dict)
                            else "manual",
                        )
                    ),
                    idempotency_key=idempotency_key,
                )
            ),
            201,
        )

    @blueprint.get("/documents/<document_id>")
    def get_document(document_id: str) -> Response:
        return jsonify(repository.get_document(document_id))

    @blueprint.put("/documents/<document_id>")
    def update_document(document_id: str) -> Response:
        idempotency_key = _idempotency_key()
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
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.post("/documents/<document_id>/avatar")
    def upload_avatar(document_id: str):
        idempotency_key = _idempotency_key()
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        return (
            jsonify(
                io_service.set_avatar(
                    document_id=document_id,
                    base_revision=int(
                        request.form.get("baseRevision", "0")
                    ),
                    upload=upload.stream,
                    idempotency_key=idempotency_key,
                )
            ),
            201,
        )

    @blueprint.delete("/documents/<document_id>/avatar")
    def clear_avatar(document_id: str) -> Response:
        idempotency_key = _idempotency_key()
        return jsonify(
            repository.set_avatar(
                document_id=document_id,
                base_revision=int(
                    request.args.get(
                        "baseRevision",
                        request.headers.get("If-Match", "0"),
                    )
                ),
                asset_id=None,
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.post("/assets")
    def upload_asset():
        idempotency_key = _idempotency_key()
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
        return (
            jsonify(
                io_service.publish_image(
                    upload.stream,
                    idempotency_key=idempotency_key,
                )
            ),
            201,
        )

    @blueprint.delete("/documents/<document_id>")
    def delete_document(document_id: str) -> Response:
        return jsonify(
            repository.delete_document(
                document_id,
                idempotency_key=_idempotency_key(),
            )
        )

    @blueprint.post("/documents/<document_id>/generate")
    def generate(document_id: str):
        body = _json_body()
        document = repository.get_document(document_id)
        compressed = derived.get_artifact(
            book_id=str(document["bookId"]),
            kind="compressed_context",
            template="default",
        )
        if compressed is None or compressed.get("status") not in {
            "ready",
            "degraded",
        }:
            raise ValueError(
                "compressed Insight context is unavailable; "
                "finish manga analysis and build the compressed overview first"
            )
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
        idempotency_key = _idempotency_key()
        body = _json_body()
        return jsonify(
            repository.validate_document(
                document_id=document_id,
                base_revision=int(body.get("baseRevision", 0)),
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.get("/documents/<document_id>/chat")
    def chat_state(document_id: str) -> Response:
        document = repository.get_document(document_id)
        payload = repository.chat_state(document_id)
        greetings = _greetings(document)
        if payload["activeSession"] is None:
            selected = greetings[0] if greetings else None
            repository.ensure_active_session(
                document_id=document_id,
                title=f"{document['title']} 对话",
                greeting=(
                    str(selected["content"]) if selected else None
                ),
                greeting_source=(
                    dict(selected["source"]) if selected else None
                ),
            )
            payload = repository.chat_state(document_id)
        payload["availableGreetings"] = greetings
        return jsonify(payload)

    @blueprint.post("/documents/<document_id>/chat/sessions")
    def create_session(document_id: str):
        idempotency_key = _idempotency_key()
        body = _json_body()
        document = repository.get_document(document_id)
        selected = next(
            (
                greeting
                for greeting in _greetings(document)
                if greeting["greetingId"] == body.get("greetingId")
            ),
            None,
        )
        if selected is None:
            choices = _greetings(document)
            selected = choices[0] if choices else None
        return (
            jsonify(
                repository.create_session(
                    document_id=document_id,
                    title=str(body.get("title", "")).strip(),
                    base_index_revision=_positive_revision(
                        body.get("baseIndexRevision"),
                        "baseIndexRevision",
                    ),
                    greeting=(
                        str(body["greeting"])
                        if body.get("greeting") is not None
                        else (
                            str(selected["content"])
                            if selected is not None
                            else None
                        )
                    ),
                    greeting_source=(
                        body["greetingSource"]
                        if isinstance(
                            body.get("greetingSource"),
                            dict,
                        )
                        else (
                            dict(selected["source"])
                            if selected is not None
                            else None
                        )
                    ),
                    idempotency_key=idempotency_key,
                    idempotency_request={
                        "documentId": document_id,
                        "request": body,
                    },
                )
            ),
            201,
        )

    @blueprint.get("/chat/sessions/<session_id>")
    def get_session(session_id: str) -> Response:
        return jsonify(repository.get_session(session_id))

    @blueprint.post("/chat/sessions/<session_id>/activate")
    def activate_session(session_id: str) -> Response:
        idempotency_key = _idempotency_key()
        body = _json_body()
        return jsonify(
            repository.activate_session(
                session_id,
                base_index_revision=_positive_revision(
                    body.get("baseIndexRevision"),
                    "baseIndexRevision",
                ),
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.delete("/chat/sessions/<session_id>")
    def delete_session(session_id: str) -> Response:
        idempotency_key = _idempotency_key()
        revision = int(
            request.headers.get(
                "If-Match",
                request.args.get("baseRevision", "0"),
            )
        )
        return jsonify(
            repository.delete_session(
                session_id=session_id,
                base_revision=revision,
                idempotency_key=idempotency_key,
            )
        )

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
            content=str(body.get("content", "")),
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
        idempotency_key = _idempotency_key()
        body = _json_body()
        return jsonify(
            repository.abort(
                session_id=session_id,
                operation_id=_required_string(body, "operationId"),
                idempotency_key=idempotency_key,
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
        idempotency_key = _idempotency_key()
        body = _json_body()
        return jsonify(
            repository.delete_message_chain(
                message_id=message_id,
                base_revision=int(
                    body.get("baseSessionRevision", 0)
                ),
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.get("/documents/<document_id>/export")
    def export_document(document_id: str) -> Response:
        document = repository.get_document(document_id)
        output_format = request.args.get("format", "v3")
        bundle = build_export_bundle(document)
        if output_format not in {"v2", "v3", "png", "worldbook"}:
            raise ValueError("format must be v2, v3, png, or worldbook")
        if output_format == "png":
            response = Response(
                io_service.export_png(document),
                content_type="image/png",
            )
            response.headers["Content-Disposition"] = (
                f'attachment; filename="{_safe_filename(document["title"])}.png"'
            )
            return response
        response = Response(
            json.dumps(
                bundle[output_format],
                ensure_ascii=False,
                indent=2,
            ),
            content_type="application/json; charset=utf-8",
        )
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{_safe_filename(document["title"])}'
            f'-{output_format}.json"'
        )
        return response

    @blueprint.post("/books/<book_id>/imports")
    def import_document(book_id: str):
        idempotency_key = _idempotency_key()
        if request.files:
            upload = request.files.get("file") or next(
                iter(request.files.values())
            )
            return (
                jsonify(
                    io_service.import_document(
                        book_id=book_id,
                        upload=upload.stream,
                        filename=upload.filename or "",
                        idempotency_key=idempotency_key,
                    )
                ),
                201,
            )
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
                    idempotency_key=idempotency_key,
                    idempotency_request={
                        "bookId": book_id,
                        "payload": payload,
                    },
                    idempotency_scope=(
                        f"POST:importStudioDocument:{book_id}"
                    ),
                )
            ),
            201,
        )

    @blueprint.post("/documents/<document_id>/worldbook/import")
    def import_worldbook(document_id: str) -> Response:
        idempotency_key = _idempotency_key()
        body = _file_or_json_object()
        document = repository.get_document(document_id)
        entries = body.get("entries")
        if not isinstance(entries, (list, dict)):
            raise ValueError("worldbook entries must be an array or object")
        imported = import_document_payload(
            str(document["bookId"]),
            body,
        )
        changed = dict(document)
        changed["lorebook"] = imported["lorebook"]
        return jsonify(
            repository.update_document(
                document_id=document_id,
                base_revision=int(
                    request.form.get(
                        "baseRevision",
                        request.args.get(
                            "baseRevision",
                            request.headers.get("If-Match", "0"),
                        ),
                    )
                ),
                title=str(document["title"]),
                document=changed,
                idempotency_key=idempotency_key,
                idempotency_request={
                    "documentId": document_id,
                    "baseRevision": int(
                        request.form.get(
                            "baseRevision",
                            request.args.get(
                                "baseRevision",
                                request.headers.get("If-Match", "0"),
                            ),
                        )
                    ),
                    "worldbook": body,
                },
                idempotency_scope=(
                    f"POST:importStudioWorldbook:{document_id}"
                ),
            )
        )

    @blueprint.get("/chat/sessions/<session_id>/export")
    def export_session(session_id: str) -> Response:
        payload = io_service.export_session(session_id)
        response = Response(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            content_type="application/json; charset=utf-8",
        )
        response.headers["Content-Disposition"] = (
            f'attachment; filename="studio-chat-{session_id}.json"'
        )
        return response

    @blueprint.post("/documents/<document_id>/chat/import")
    def import_session(document_id: str):
        idempotency_key = _idempotency_key()
        return (
            jsonify(
                io_service.import_session(
                    document_id=document_id,
                    payload=_file_or_json_object(),
                    base_index_revision=_positive_revision(
                        request.headers.get("If-Match"),
                        "If-Match",
                    ),
                    idempotency_key=idempotency_key,
                )
            ),
            201,
        )

    @blueprint.get("/chat/sessions/<session_id>/prompt-preview")
    def prompt_preview(session_id: str) -> Response:
        session = repository.get_session(session_id)
        document = repository.get_document(
            str(session["documentId"])
        )
        return jsonify(
            {
                "sessionId": session_id,
                "promptPreview": runtime.prompt_preview(
                    document=document,
                    session=session,
                ),
            }
        )

    @blueprint.post("/documents/<document_id>/agent")
    def agent(document_id: str) -> Response:
        body = _json_body()
        document = repository.get_document(document_id)
        messages = body.get("messages")
        if messages is None:
            messages = [
                {
                    "role": "user",
                    "content": _required_string(body, "content"),
                }
            ]
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValueError("messages must be an object array")
        config = settings.resolve_insight(
            book_id=str(document["bookId"]),
            command={"scope": "full", "force": False},
        )
        cancelled = threading.Event()

        @stream_with_context
        def generate():
            try:
                yield "event: ready\ndata: {}\n\n"
                for chunk in runtime.agent_chunks(
                    document=document,
                    messages=messages,
                    config=config,
                    cancelled=cancelled,
                ):
                    yield (
                        "event: chunk\ndata: "
                        + json.dumps(
                            {"text": chunk},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n\n"
                    )
                yield "event: done\ndata: {}\n\n"
            except GeneratorExit:
                raise
            except Exception as exc:
                yield (
                    "event: error\ndata: "
                    + json.dumps(
                        {"message": redact_sensitive_text(exc)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n\n"
                )
            finally:
                cancelled.set()

        response = Response(
            generate(),
            content_type="text/event-stream; charset=utf-8",
        )
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

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


def _file_or_json_object() -> dict[str, Any]:
    if request.files:
        upload = request.files.get("file") or next(
            iter(request.files.values())
        )
        payload = json.loads(upload.read().decode("utf-8-sig"))
    else:
        payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def _safe_filename(value: object) -> str:
    name = str(value or "character")
    for character in '<>:"/\\|?*':
        name = name.replace(character, "_")
    return name.strip().strip(".")[:120] or "character"


def _greetings(document: dict[str, Any]) -> list[dict[str, Any]]:
    core = document.get("coreMessages", {})
    if not isinstance(core, dict):
        return []
    result: list[dict[str, Any]] = []
    first = str(core.get("first_message", "")).strip()
    if first:
        result.append(
            {
                "greetingId": "first",
                "label": "默认问候语",
                "content": first,
                "source": {"type": "first_message", "index": 0},
            }
        )
    alternates = core.get("alternate_greetings", [])
    if isinstance(alternates, list):
        for index, value in enumerate(alternates):
            content = str(value).strip()
            if content:
                result.append(
                    {
                        "greetingId": f"alternate-{index}",
                        "label": f"备选问候语 {index + 1}",
                        "content": content,
                        "source": {
                            "type": "alternate_greeting",
                            "index": index,
                        },
                    }
                )
    return result


def _idempotency_key() -> str:
    value = request.headers.get("Idempotency-Key", "")
    if not value or len(value) > 200:
        raise ValueError("Idempotency-Key is required")
    return value


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def _candidate_item(character: Mapping[str, Any]) -> dict[str, Any]:
    name = str(character.get("name", "")).strip()
    raw_aliases = character.get("aliases", [])
    if not isinstance(raw_aliases, list):
        raw_aliases = []
    aliases = [
        str(value).strip()
        for value in raw_aliases
        if isinstance(value, str) and value.strip()
    ]
    moments = character.get(
        "keyMoments",
        character.get("key_moments", []),
    )
    if not isinstance(moments, list):
        moments = []
    first_page = _positive_page_number(
        character.get(
            "firstAppearancePage",
            character.get(
                "first_appearance",
                character.get("first_page"),
            ),
        )
    )
    related_pages: set[int] = set()
    for field in ("relatedPageNumbers", "related_page_numbers"):
        values = character.get(field, [])
        if isinstance(values, list):
            related_pages.update(
                page
                for page in (
                    _positive_page_number(value) for value in values
                )
                if page is not None
            )
    for moment in moments:
        if not isinstance(moment, Mapping):
            continue
        page = _positive_page_number(
            moment.get(
                "page",
                moment.get("pageNumber", moment.get("page_number")),
            )
        )
        if page is not None:
            related_pages.add(page)
    if first_page is not None:
        related_pages.add(first_page)
    ordered_pages = sorted(related_pages)
    return {
        "characterId": str(character.get("characterId", "")),
        "name": name,
        "aliases": aliases,
        "description": str(
            character.get("description", character.get("summary", ""))
        ),
        "personality": str(character.get("personality", "")),
        "arc": str(character.get("arc", "")),
        "firstAppearancePage": first_page,
        "keyMomentCount": len(moments),
        "relatedPageCount": len(ordered_pages),
        "relatedPageNumbers": ordered_pages,
    }


def _positive_page_number(value: object) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _positive_revision(value: object, field: str) -> int:
    try:
        revision = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if revision < 1:
        raise ValueError(f"{field} must be a positive integer")
    return revision


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
