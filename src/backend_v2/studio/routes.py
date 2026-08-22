"""Character Studio v2 HTTP commands and queries."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import threading
from typing import Any, Mapping

from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    send_file,
    stream_with_context,
)
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    integer_value as _integer_value,
    json_body as _json_body,
    require_idempotency_key as _idempotency_key,
    required_string as _required_string,
    validate_multipart_fields as _validate_multipart_fields,
)
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
    create_empty_document,
    import_document_payload,
)
from src.backend_v2.public_policy import PublicUserPolicyAccess
from src.backend_v2.runtime_profile import RuntimeProfile


def create_studio_blueprint(
    *,
    engine: Engine,
    data_root: Path,
    profile: RuntimeProfile,
) -> Blueprint:
    blueprint = Blueprint("studio_v2", __name__, url_prefix="/api/v2/studio")
    public_access = PublicUserPolicyAccess(engine, profile)
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

    @blueprint.before_request
    def require_character_studio_access() -> None:
        public_access.require_feature("characterStudio")

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
        timeline = derived.get_timeline_status(book_id=book_id)
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
        timeline = derived.list_timeline_characters(book_id=book_id)
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
        body = _json_body(allowed_keys={"candidateId", "title"})
        candidate: Mapping[str, Any] | None = None
        candidate_id = body.get("candidateId")
        if candidate_id is not None:
            source = derived.get_timeline_character(
                book_id=book_id,
                character_id=_required_string(body, "candidateId"),
            )
            if source is None:
                raise ValueError("candidateId does not identify an active candidate")
            if (
                source.get("status") not in {"ready", "degraded"}
                or source.get("mode") != "enhanced"
            ):
                raise ValueError("candidateId requires an enhanced active timeline")
            raw_candidate = source.get("character")
            if not isinstance(raw_candidate, Mapping):
                raise ValueError("candidateId does not identify an active candidate")
            candidate = raw_candidate
        raw_title = body.get("title")
        if raw_title is not None and not isinstance(raw_title, str):
            raise ValueError("title must be a string")
        title = (
            (raw_title or "")
            or (str(candidate.get("name", "")) if candidate else "")
        ).strip()
        if not title:
            raise ValueError("title is required")
        document = None
        if candidate is not None:
            document = create_empty_document(book_id, title=title)
            document["origin"] = {
                "type": "analysis",
                "source_character": candidate["name"],
            }
        return (
            jsonify(
                repository.create_document(
                    book_id=book_id,
                    title=title,
                    document=document,
                    kind="analysis" if candidate is not None else "manual",
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
        body = _json_body(
            allowed_keys={"baseRevision", "title", "document"}
        )
        document = body.get("document")
        if not isinstance(document, dict):
            raise ValueError("document must be an object")
        raw_title = body.get("title")
        if raw_title is not None and not isinstance(raw_title, str):
            raise ValueError("title must be a string")
        return jsonify(
            repository.update_document(
                document_id=document_id,
                base_revision=_revision(body, "baseRevision"),
                title=raw_title,
                document=document,
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.post("/assets")
    def upload_asset():
        idempotency_key = _idempotency_key()
        _validate_multipart_fields(allowed_file_keys={"file"})
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
        body = _json_body(allowed_keys={"baseRevision", "section"})
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
                "漫画分析的压缩上下文尚不可用；请先完成漫画分析并生成压缩概览"
            )
        response = repository.create_generate_operation(
            document_id=document_id,
            base_revision=_revision(body, "baseRevision"),
            section=_required_string(body, "section"),
            config=settings.resolve_insight(
                book_id=str(document["bookId"]),
                scope="full",
            ),
            analysis_context=compressed,
            idempotency_key=_idempotency_key(),
        )
        return jsonify(response), 202

    @blueprint.post("/documents/<document_id>/validate")
    def validate(document_id: str) -> Response:
        idempotency_key = _idempotency_key()
        body = _json_body(allowed_keys={"baseRevision"})
        return jsonify(
            repository.validate_document(
                document_id=document_id,
                base_revision=_revision(body, "baseRevision"),
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
        body = _json_body(
            allowed_keys={
                "title",
                "baseIndexRevision",
                "greetingId",
            }
        )
        document = repository.get_document(document_id)
        raw_title = body.get("title")
        if raw_title is not None and not isinstance(raw_title, str):
            raise ValueError("title must be a string")
        raw_greeting_id = body.get("greetingId")
        if raw_greeting_id is not None and not isinstance(
            raw_greeting_id,
            str,
        ):
            raise ValueError("greetingId must be a string")
        choices = _greetings(document)
        selected = next(
            (
                greeting
                for greeting in choices
                if greeting["greetingId"] == raw_greeting_id
            ),
            None,
        )
        if raw_greeting_id is not None and selected is None:
            raise ValueError("greetingId does not identify an available greeting")
        if raw_greeting_id is None:
            selected = choices[0] if choices else None
        return (
            jsonify(
                repository.create_session(
                    document_id=document_id,
                    title=(raw_title or "").strip(),
                    base_index_revision=_positive_revision(
                        body.get("baseIndexRevision"),
                        "baseIndexRevision",
                    ),
                    greeting=(
                        str(selected["content"])
                        if selected is not None
                        else None
                    ),
                    greeting_source=(
                        dict(selected["source"])
                        if selected is not None
                        else None
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
        body = _json_body(allowed_keys={"baseIndexRevision"})
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
        revision = _positive_revision(
            request.headers.get("If-Match"),
            "If-Match",
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
        body = _json_body(
            allowed_keys={"baseSessionRevision", "content", "assetIds"}
        )
        asset_ids = body.get("assetIds", [])
        if not isinstance(asset_ids, list) or not all(
            isinstance(value, str) for value in asset_ids
        ):
            raise ValueError("assetIds must be a string array")
        content = body.get("content", "")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        response = repository.send_message(
            session_id=session_id,
            base_revision=_revision(body, "baseSessionRevision"),
            content=content,
            asset_ids=asset_ids,
            config=settings.resolve_insight(
                book_id=repository.session_book_id(session_id),
                scope="full",
            ),
            idempotency_key=_idempotency_key(),
        )
        return jsonify(response), 202

    @blueprint.post("/chat/sessions/<session_id>/summarize")
    def summarize(session_id: str):
        body = _json_body(allowed_keys={"baseSessionRevision"})
        response = repository.create_summary_operation(
            session_id=session_id,
            base_revision=_revision(body, "baseSessionRevision"),
            config=settings.resolve_insight(
                book_id=repository.session_book_id(session_id),
                scope="full",
            ),
            idempotency_key=_idempotency_key(),
        )
        return jsonify(response), 202

    @blueprint.post("/chat/sessions/<session_id>/abort")
    def abort(session_id: str) -> Response:
        idempotency_key = _idempotency_key()
        body = _json_body(allowed_keys={"operationId"})
        return jsonify(
            repository.abort(
                session_id=session_id,
                operation_id=_required_string(body, "operationId"),
                idempotency_key=idempotency_key,
            )
        )

    @blueprint.put("/chat/messages/<message_id>")
    def edit_message(message_id: str):
        body = _json_body(
            allowed_keys={"baseSessionRevision", "content"}
        )
        return (
            jsonify(
                repository.edit_or_regenerate_message(
                    message_id=message_id,
                    base_revision=_revision(
                        body,
                        "baseSessionRevision",
                    ),
                    content=_required_string(body, "content"),
                    config=settings.resolve_insight(
                        book_id=repository.message_book_id(message_id),
                        scope="full",
                    ),
                    idempotency_key=_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/chat/messages/<message_id>/regenerate")
    def regenerate_message(message_id: str):
        body = _json_body(allowed_keys={"baseSessionRevision"})
        return (
            jsonify(
                repository.edit_or_regenerate_message(
                    message_id=message_id,
                    base_revision=_revision(
                        body,
                        "baseSessionRevision",
                    ),
                    content=None,
                    config=settings.resolve_insight(
                        book_id=repository.message_book_id(message_id),
                        scope="full",
                    ),
                    idempotency_key=_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.delete("/chat/messages/<message_id>")
    def delete_message(message_id: str) -> Response:
        idempotency_key = _idempotency_key()
        body = _json_body(allowed_keys={"baseSessionRevision"})
        return jsonify(
            repository.delete_message_chain(
                message_id=message_id,
                base_revision=_revision(
                    body,
                    "baseSessionRevision",
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
            return send_file(
                BytesIO(io_service.export_png(document)),
                mimetype="image/png",
                as_attachment=True,
                download_name=(
                    f"{_safe_filename(document['title'])}.png"
                ),
            )
        return send_file(
            BytesIO(
                json.dumps(
                    bundle[output_format],
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8")
            ),
            mimetype="application/json",
            as_attachment=True,
            download_name=(
                f"{_safe_filename(document['title'])}"
                f"-{output_format}.json"
            ),
        )

    @blueprint.post("/books/<book_id>/imports")
    def import_document(book_id: str):
        idempotency_key = _idempotency_key()
        if request.mimetype == "multipart/form-data":
            _validate_multipart_fields(allowed_file_keys={"file"})
            upload = request.files.get("file")
            if upload is None:
                raise ValueError("multipart field 'file' is required")
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
        body = _json_body(allowed_keys={"payload"})
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
        base_revision = _positive_revision(
            request.headers.get("If-Match"),
            "If-Match",
        )
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
                base_revision=base_revision,
                title=str(document["title"]),
                document=changed,
                idempotency_key=idempotency_key,
                idempotency_request={
                    "documentId": document_id,
                    "baseRevision": base_revision,
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
        body = _json_body(allowed_keys={"content"})
        document = repository.get_document(document_id)
        messages = [
            {
                "role": "user",
                "content": _required_string(body, "content"),
            }
        ]
        config = settings.resolve_insight(
            book_id=str(document["bookId"]),
            scope="full",
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


def _file_or_json_object() -> dict[str, Any]:
    if request.mimetype == "multipart/form-data":
        _validate_multipart_fields(allowed_file_keys={"file"})
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("multipart field 'file' is required")
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
    moments = character.get("key_moments", [])
    if not isinstance(moments, list):
        moments = []
    first_page = _positive_page_number(character.get("first_page"))
    related_pages: set[int] = set()
    values = character.get("related_page_numbers", [])
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
        page = _positive_page_number(moment.get("page"))
        if page is not None:
            related_pages.add(page)
    if first_page is not None:
        related_pages.add(first_page)
    ordered_pages = sorted(related_pages)
    return {
        "characterId": str(character.get("characterId", "")),
        "name": name,
        "aliases": aliases,
        "description": str(character.get("description", "")),
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
    return _integer_value(value, field, minimum=1)


def _revision(body: Mapping[str, object], field: str) -> int:
    return _positive_revision(body.get(field), field)
