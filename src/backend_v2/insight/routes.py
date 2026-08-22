"""Manga Insight v2 query surface and durable task commands."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import time
from collections.abc import Mapping
from typing import Any

from flask import Blueprint, Response, jsonify, request, stream_with_context
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    integer_value as _integer_value,
    json_body as _json_body,
    required_boolean as _required_boolean,
    required_integer as _required_integer,
    require_idempotency_key as _require_idempotency_key,
    required_string as _required_string,
    validate_multipart_fields as _validate_multipart_fields,
)
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.insight.commands import InsightAnalysisCommandService
from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.insight.continuation import (
    ContinuationCommandService,
    ContinuationRepository,
)
from src.backend_v2.insight.derived import (
    InsightDerivedCommandService,
    InsightDerivedRepository,
)
from src.backend_v2.insight.exports import (
    InsightExportCommandService,
    build_current_export,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightLocked,
    InsightNotFound,
    InsightRepository,
)
from src.backend_v2.insight.qa import (
    DefaultQAApiAlgorithms,
    InsightQACommandService,
    QAApiAlgorithms,
    QAConflict,
    TransientRequestRepository,
    citations_for,
    select_candidates,
    validate_retrieval_candidates,
)
from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.public_policy import PublicUserPolicyAccess
from src.backend_v2.runtime_profile import RuntimeProfile


def create_insight_blueprint(
    *,
    engine: Engine,
    data_root: Path,
    qa_algorithms: QAApiAlgorithms | None = None,
    profile: RuntimeProfile,
) -> Blueprint:
    blueprint = Blueprint("insight_v2", __name__, url_prefix="/api/v2/insight")
    public_access = PublicUserPolicyAccess(engine, profile)
    repository = InsightRepository(engine)
    commands = InsightAnalysisCommandService(engine)
    derived = InsightDerivedRepository(engine)
    derived_commands = InsightDerivedCommandService(engine)
    export_commands = InsightExportCommandService(engine)
    continuation = ContinuationRepository(engine)
    continuation_commands = ContinuationCommandService(engine)
    qa_requests = TransientRequestRepository(engine)
    qa_commands = InsightQACommandService(
        engine,
        repository=qa_requests,
    )
    qa_api = qa_algorithms or DefaultQAApiAlgorithms()
    image_import = ImageImportService(
        data_root=data_root,
        repository=ContentRepository(engine),
        storage=AssetStorageService(data_root, engine),
    )

    @blueprint.before_request
    def require_insight_access() -> None:
        public_access.require_feature("insight")

    @blueprint.errorhandler(InsightNotFound)
    def not_found(error: InsightNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(InsightLocked)
    def locked(error: InsightLocked):
        return _error("resource_locked", str(error), 423)

    @blueprint.errorhandler(InsightConflict)
    @blueprint.errorhandler(JobConflict)
    def conflict(error: Exception):
        return _error("revision_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.get("/bootstrap")
    def bootstrap() -> Response:
        return jsonify(repository.bootstrap())

    @blueprint.get("/books/<book_id>/chapters")
    def chapters(book_id: str) -> Response:
        return jsonify(repository.list_chapters(book_id))

    @blueprint.get("/books/<book_id>/pages")
    def pages(book_id: str) -> Response:
        return jsonify(
            repository.list_pages(
                book_id=book_id,
                chapter_id=_optional_query_string("chapterId"),
                after=_integer_value(
                    request.args.get("cursor", "0"),
                    "cursor",
                    minimum=0,
                ),
                limit=_integer_value(
                    request.args.get("limit", "100"),
                    "limit",
                    minimum=1,
                    maximum=200,
                ),
            )
        )

    @blueprint.get("/books/<book_id>/recent-page-analyses")
    def recent_page_analyses(book_id: str) -> Response:
        return jsonify(
            repository.list_recent_page_analyses(
                book_id=book_id,
                limit=_integer_value(
                    request.args.get("limit", "5"),
                    "limit",
                    minimum=1,
                    maximum=20,
                ),
            )
        )

    @blueprint.get("/pages/<page_id>")
    def page_detail(page_id: str) -> Response:
        return jsonify(
            repository.page_detail(
                page_id=page_id,
                run_id=_optional_query_string("runId"),
            )
        )

    @blueprint.get("/runs/<run_id>")
    def run_detail(run_id: str) -> Response:
        return jsonify(repository.get_run(run_id))

    @blueprint.post("/analysis-jobs")
    def create_analysis_job():
        result = commands.create_analysis_job(
            command=_json_body(
                allowed_keys={
                    "bookId",
                    "scope",
                    "chapterIds",
                    "pageIds",
                }
            ),
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.get("/artifacts/overviews")
    def list_overviews() -> Response:
        book_id = _required_query_string("bookId")
        return jsonify(repository.list_overview_templates(book_id))

    @blueprint.get("/artifacts/overviews/<template>")
    def get_overview(template: str) -> Response:
        book_id = _required_query_string("bookId")
        artifact = derived.get_artifact(
            book_id=book_id,
            kind="overview",
            template=template,
        )
        if artifact is None:
            raise InsightNotFound("overview not found")
        return jsonify(artifact)

    @blueprint.post("/artifacts/overviews/<template>")
    def rebuild_overview(template: str):
        body = _json_body(allowed_keys={"bookId"})
        return (
            jsonify(
                derived_commands.create_job(
                    book_id=_required_string(body, "bookId"),
                    kind="overview",
                    template=template,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.get("/timeline")
    def get_timeline() -> Response:
        book_id = _required_query_string("bookId")
        timeline = derived.get_timeline(
            book_id=book_id,
            event_after=_integer_value(
                request.args.get("eventCursor", "0"),
                "eventCursor",
                minimum=0,
            ),
            event_limit=_integer_value(
                request.args.get("eventLimit", "100"),
                "eventLimit",
                minimum=1,
                maximum=200,
            ),
            character_after=_optional_query_string("characterCursor"),
            character_limit=_integer_value(
                request.args.get("characterLimit", "100"),
                "characterLimit",
                minimum=1,
                maximum=200,
            ),
        )
        if timeline is None:
            raise InsightNotFound("timeline not found")
        return jsonify(timeline)

    @blueprint.post("/timeline")
    def rebuild_timeline():
        body = _json_body(allowed_keys={"bookId"})
        return (
            jsonify(
                derived_commands.create_job(
                    book_id=_required_string(body, "bookId"),
                    kind="timeline",
                    template="default",
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/books/<book_id>/compressed-context/rebuild")
    def rebuild_compressed_context(book_id: str):
        _json_body(allowed_keys=set())
        return (
            jsonify(
                derived_commands.create_job(
                    book_id=book_id,
                    kind="compressed_context",
                    template="default",
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/books/<book_id>/vector-rebuild")
    def rebuild_vectors(book_id: str):
        _json_body(allowed_keys=set())
        return (
            jsonify(
                derived_commands.create_job(
                    book_id=book_id,
                    kind="vector",
                    template="default",
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.get("/qa/status")
    def qa_status() -> Response:
        book_id = _required_query_string("bookId")
        return jsonify(
            derived.qa_status(
                book_id=book_id,
                mode=_query_choice(
                    "mode",
                    allowed={"exact", "global"},
                    default="exact",
                ),
            )
        )

    @blueprint.post("/books/<book_id>/qa")
    def ask_question(book_id: str) -> Response:
        handle = qa_commands.create(
            book_id=book_id,
            command=_json_body(
                allowed_keys={
                    "question",
                    "mode",
                    "topK",
                    "threshold",
                    "useParentChild",
                    "useReasoning",
                    "useReranker",
                }
            ),
        )
        cancelled = threading.Event()

        @stream_with_context
        def generate():
            try:
                yield _qa_sse(
                    "status",
                    {
                        "requestId": handle.request_id,
                        "status": "retrieving",
                    },
                )
                heartbeat_at = time.monotonic() + 15
                while True:
                    state = qa_requests.poll(
                        request_id=handle.request_id,
                        connection_token=handle.connection_token,
                    )
                    status = state["status"]
                    if status == "completed":
                        break
                    if status in {"failed", "cancelled"}:
                        result = state["result"]
                        if status == "failed":
                            if (
                                not isinstance(result, Mapping)
                                or set(result) != {"error"}
                                or not isinstance(result["error"], Mapping)
                                or set(result["error"]) != {"code", "message"}
                                or not all(
                                    isinstance(result["error"][field], str)
                                    and result["error"][field]
                                    for field in ("code", "message")
                                )
                            ):
                                raise QAConflict(
                                    "stored QA failure result is invalid"
                                )
                            error = dict(result["error"])
                        else:
                            error = {
                                "code": "QA_RETRIEVAL_CANCELLED",
                                "message": "问答检索已取消",
                            }
                        yield _qa_sse(
                            "error",
                            error,
                        )
                        return
                    now = time.monotonic()
                    if now >= heartbeat_at:
                        qa_requests.touch_connection(
                            request_id=handle.request_id,
                            connection_token=handle.connection_token,
                        )
                        yield ": heartbeat\n\n"
                        heartbeat_at = now + 15
                    time.sleep(0.1)
                result = qa_requests.consume(
                    request_id=handle.request_id,
                    connection_token=handle.connection_token,
                )
                candidates = select_candidates(
                    validate_retrieval_candidates(result),
                    threshold=handle.options["threshold"],
                    top_k=handle.options["topK"] * 4,
                )
                config: Mapping[str, Any] = {}
                if candidates:
                    should_rerank = (
                        handle.options["useReranker"]
                        and len(candidates) > 1
                    )
                    config = qa_commands.materialize_api_config(
                        handle.config,
                        include_reranker=should_rerank,
                    )
                    if should_rerank:
                        candidates = [
                            dict(value)
                            for value in qa_api.rerank(
                                question=handle.question,
                                candidates=candidates,
                                top_k=handle.options["topK"],
                                config=config,
                            )
                        ]
                    else:
                        candidates = candidates[: handle.options["topK"]]
                citations = citations_for(candidates)
                yield _qa_sse(
                    "context",
                    {
                        "mode": handle.options["mode"],
                        "citations": citations,
                    },
                )
                if not candidates:
                    yield _qa_sse(
                        "chunk",
                        {
                            "text": (
                                "没有找到满足当前相关性阈值的漫画内容，"
                                "请降低阈值或换一种问法。"
                            )
                        },
                    )
                else:
                    answer_emitted = False
                    for chunk in qa_api.stream_answer(
                        question=handle.question,
                        candidates=candidates,
                        config=config,
                        cancelled=cancelled,
                    ):
                        if chunk:
                            answer_emitted = True
                            yield _qa_sse("chunk", {"text": chunk})
                        else:
                            yield ": heartbeat\n\n"
                    if not answer_emitted:
                        raise QAConflict("QA provider returned an empty answer")
                yield _qa_sse("done", {})
            except GeneratorExit:
                raise
            except Exception as exc:
                yield _qa_sse(
                    "error",
                    {
                        "code": "QA_FAILED",
                        "message": redact_sensitive_text(exc),
                    },
                )
            finally:
                cancelled.set()
                qa_requests.close(
                    request_id=handle.request_id,
                    connection_token=handle.connection_token,
                )

        response = Response(
            generate(),
            content_type="text/event-stream; charset=utf-8",
        )
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @blueprint.get("/pages/<page_id>/export")
    def export_page(page_id: str) -> Response:
        detail = repository.page_detail(page_id=page_id)
        if detail["analysis"] is None:
            raise InsightNotFound("page has no published analysis")
        output_format = request.args.get("format", "markdown")
        if output_format == "json":
            body = json.dumps(
                detail["analysis"],
                ensure_ascii=False,
                indent=2,
            )
            mime = "application/json; charset=utf-8"
            extension = "json"
        elif output_format == "markdown":
            analysis = detail["analysis"]
            events = "\n".join(
                f"- [{event['importance']}] {event['summary']}"
                for event in analysis["key_events"]
            )
            warnings = "\n".join(
                f"- {warning['code']}: {warning['message']}"
                for warning in analysis["warnings"]
            )
            body = (
                f"# 第 {detail['displayPageNumber']} 页\n\n"
                f"## 页面摘要\n\n{analysis['page_summary']}\n\n"
                f"## 关键事件\n\n{events or '- 无'}\n\n"
                f"## 连续性说明\n\n"
                f"{analysis['continuity_notes'] or '无'}\n\n"
                f"## 警告\n\n{warnings or '- 无'}\n"
            )
            mime = "text/markdown; charset=utf-8"
            extension = "md"
        else:
            raise ValueError("format must be markdown or json")
        response = Response(body, content_type=mime)
        response.headers["Content-Disposition"] = (
            f'attachment; filename="page-{detail["displayPageNumber"]}.{extension}"'
        )
        return response

    @blueprint.get("/books/<book_id>/export/current")
    def export_current(book_id: str) -> Response:
        template = _required_query_string("template")
        artifact = derived.get_artifact(
            book_id=book_id,
            kind="overview",
            template=template,
        )
        if artifact is None:
            raise InsightNotFound("overview not found")
        body, mime, filename = build_current_export(
            artifact,
            output_format=_query_choice(
                "format",
                allowed={"markdown", "json"},
                default="markdown",
            ),
        )
        response = Response(body, content_type=mime)
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{filename}"'
        )
        return response

    @blueprint.post("/books/<book_id>/exports")
    def export_all(book_id: str):
        _json_body(allowed_keys=set())
        return (
            jsonify(
                export_commands.create_export_job(
                    book_id=book_id,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.get("/notes")
    def list_notes() -> Response:
        book_id = _required_query_string("bookId")
        return jsonify(
            repository.list_notes(
                book_id=book_id,
                cursor=_optional_query_string("cursor"),
                limit=_integer_value(
                    request.args.get("limit", "50"),
                    "limit",
                    minimum=1,
                    maximum=200,
                ),
                kind=_query_choice(
                    "kind",
                    allowed={"text", "qa"},
                    default=None,
                ),
                include_content=_detail_requested(),
            )
        )

    @blueprint.get("/notes/<note_id>")
    def get_note(note_id: str) -> Response:
        return jsonify(repository.get_note(note_id=note_id))

    @blueprint.post("/notes")
    def create_note():
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={
                "bookId",
                "title",
                "content",
                "citations",
                "kind",
                "tags",
                "question",
                "comment",
            }
        )
        return (
            jsonify(
                repository.create_note(
                    idempotency_key=idempotency_key,
                    book_id=_required_string(body, "bookId"),
                    title=_required_string(body, "title"),
                    content=_required_text(body, "content"),
                    citations=_citations(body, required=True),
                    kind=_body_choice(
                        body,
                        "kind",
                        allowed={"text", "qa"},
                    ),
                    tags=_string_list(body, "tags", required=True),
                    question=_nullable_text(body, "question"),
                    comment=_nullable_text(body, "comment"),
                )
            ),
            201,
        )

    @blueprint.patch("/notes/<note_id>")
    def update_note(note_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={
                "baseRevision",
                "title",
                "content",
                "citations",
                "kind",
                "tags",
                "question",
                "comment",
            }
        )
        return jsonify(
            repository.update_note(
                idempotency_key=idempotency_key,
                note_id=note_id,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=1,
                ),
                title=_required_string(body, "title"),
                content=_required_text(body, "content"),
                citations=_citations(body, required=True),
                kind=_body_choice(
                    body,
                    "kind",
                    allowed={"text", "qa"},
                ),
                tags=_string_list(body, "tags", required=True),
                question=_nullable_text(body, "question"),
                comment=_nullable_text(body, "comment"),
            )
        )

    @blueprint.delete("/notes/<note_id>")
    def delete_note(note_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        base_revision = _base_revision()
        repository.delete_note(
            idempotency_key=idempotency_key,
            note_id=note_id,
            base_revision=base_revision,
        )
        return jsonify({"deleted": True})

    @blueprint.get("/books/<book_id>/continuation")
    def continuation_state(book_id: str) -> Response:
        return jsonify(continuation.bootstrap(book_id=book_id))

    @blueprint.post("/books/<book_id>/continuation/sync-analysis")
    def sync_continuation_analysis(book_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        _json_body(allowed_keys=set())
        return jsonify(
            continuation.sync_latest(
                idempotency_key=idempotency_key,
                book_id=book_id,
            )
        )

    @blueprint.patch("/continuation/projects/<project_id>")
    def update_continuation_project(project_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision", "config"})
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        return jsonify(
            continuation.update_project(
                idempotency_key=idempotency_key,
                project_id=project_id,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=1,
                ),
                config=config,
            )
        )

    @blueprint.put("/continuation/projects/<project_id>/references")
    def set_continuation_references(project_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision", "assetIds"})
        asset_ids = _string_list(body, "assetIds", required=True)
        return jsonify(
            continuation.set_project_references(
                idempotency_key=idempotency_key,
                project_id=project_id,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=1,
                ),
                asset_ids=asset_ids,
            )
        )

    @blueprint.post("/continuation/projects/<project_id>/characters")
    def create_continuation_character(project_id: str):
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={"name", "aliases", "enabled", "payload"}
        )
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return (
            jsonify(
                continuation.create_character(
                    idempotency_key=idempotency_key,
                    project_id=project_id,
                    name=_required_string(body, "name"),
                    aliases=_string_list(body, "aliases", required=True),
                    enabled=_required_boolean(body, "enabled"),
                    payload=payload,
                )
            ),
            201,
        )

    @blueprint.patch("/continuation/characters/<character_id>")
    def update_continuation_character(character_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={
                "baseRevision",
                "name",
                "aliases",
                "enabled",
                "payload",
            }
        )
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return jsonify(
            continuation.update_character(
                idempotency_key=idempotency_key,
                character_id=character_id,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=1,
                ),
                name=_required_string(body, "name"),
                aliases=_string_list(body, "aliases", required=True),
                enabled=_required_boolean(body, "enabled"),
                payload=payload,
            )
        )

    @blueprint.delete("/continuation/characters/<character_id>")
    def delete_continuation_character(character_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        continuation.delete_character(
            idempotency_key=idempotency_key,
            character_id=character_id,
            base_revision=_base_revision(),
        )
        return jsonify({"deleted": True})

    @blueprint.post("/continuation/characters/<character_id>/forms")
    def create_continuation_form(character_id: str):
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"name", "payload"})
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return (
            jsonify(
                continuation.create_form(
                    idempotency_key=idempotency_key,
                    character_id=character_id,
                    name=_required_string(body, "name"),
                    payload=payload,
                )
            ),
            201,
        )

    @blueprint.get("/continuation/projects/<project_id>/forms")
    def list_continuation_forms(project_id: str) -> Response:
        return jsonify(
            continuation.list_forms(
                project_id=project_id,
                cursor=_integer_value(
                    request.args.get("cursor", "0"),
                    "cursor",
                    minimum=0,
                ),
                limit=_integer_value(
                    request.args.get("limit", "50"),
                    "limit",
                    minimum=1,
                    maximum=200,
                ),
            )
        )

    @blueprint.patch("/continuation/forms/<form_id>")
    def update_continuation_form(form_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(
            allowed_keys={"baseRevision", "name", "payload"}
        )
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return jsonify(
            continuation.update_form(
                idempotency_key=idempotency_key,
                form_id=form_id,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=1,
                ),
                name=_required_string(body, "name"),
                payload=payload,
            )
        )

    @blueprint.delete("/continuation/forms/<form_id>")
    def delete_continuation_form(form_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        continuation.delete_form(
            idempotency_key=idempotency_key,
            form_id=form_id,
            base_revision=_base_revision(),
        )
        return jsonify({"deleted": True})

    @blueprint.post("/continuation/forms/<form_id>/reference")
    def upload_continuation_reference(form_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        _validate_multipart_fields(
            allowed_form_keys={"baseRevision"},
            allowed_file_keys={"file"},
        )
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("file is required")
        base_revision = _multipart_revision()
        content_checksum = _stream_sha256(upload.stream)
        replay = continuation.replay_form_reference_upload(
            idempotency_key=idempotency_key,
            form_id=form_id,
            base_revision=base_revision,
            content_checksum=content_checksum,
        )
        if replay is not None:
            return jsonify(replay)
        source, thumbnail = image_import.publish_standalone_image(
            upload.stream
        )
        return jsonify(
            continuation.bind_form_reference(
                idempotency_key=idempotency_key,
                form_id=form_id,
                base_revision=base_revision,
                asset_id=source.id,
                thumbnail_asset_id=thumbnail.id,
                content_checksum=content_checksum,
            )
        )

    @blueprint.delete("/continuation/forms/<form_id>/reference")
    def delete_continuation_reference(form_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        return jsonify(
            continuation.bind_form_reference(
                idempotency_key=idempotency_key,
                form_id=form_id,
                base_revision=_base_revision(),
                asset_id=None,
                thumbnail_asset_id=None,
            )
        )

    @blueprint.post(
        "/continuation/forms/<form_id>/image-versions/<int:version>/adopt"
    )
    def adopt_continuation_character_sheet(
        form_id: str,
        version: int,
    ) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision"})
        return jsonify(
            continuation.adopt_form_image(
                idempotency_key=idempotency_key,
                form_id=form_id,
                version=version,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=1,
                ),
            )
        )

    @blueprint.post("/books/<book_id>/continuation/jobs")
    def create_continuation_job(book_id: str):
        body = _json_body(
            allowed_keys={"kind", "ordinals", "format", "formId"}
        )
        kind = _required_string(body, "kind")
        idempotency_key = _require_idempotency_key()
        ordinals = _optional_ordinals(body)
        if kind == "script":
            _require_exact_keys(body, {"kind"})
            result = continuation_commands.create_script_job(
                book_id=book_id,
                idempotency_key=idempotency_key,
            )
        elif kind == "pages":
            _require_exact_keys(body, {"kind", "ordinals"})
            result = continuation_commands.create_pages_job(
                book_id=book_id,
                ordinals=ordinals,
                idempotency_key=idempotency_key,
            )
        elif kind == "images":
            _require_exact_keys(body, {"kind", "ordinals"})
            result = continuation_commands.create_images_job(
                book_id=book_id,
                ordinals=ordinals,
                idempotency_key=idempotency_key,
            )
        elif kind == "export":
            _require_exact_keys(body, {"kind", "format"})
            result = continuation_commands.create_export_job(
                book_id=book_id,
                output_format=_required_string(body, "format"),
                idempotency_key=idempotency_key,
            )
        elif kind == "character_sheet":
            _require_exact_keys(body, {"kind", "formId"})
            result = continuation_commands.create_character_sheet_job(
                book_id=book_id,
                form_id=_required_string(body, "formId"),
                idempotency_key=idempotency_key,
            )
        else:
            raise ValueError("unsupported continuation job kind")
        return jsonify(result), 202

    @blueprint.patch("/continuation/projects/<project_id>/script")
    def update_continuation_script(project_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision", "content"})
        return jsonify(
            continuation.update_script(
                idempotency_key=idempotency_key,
                project_id=project_id,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=0,
                ),
                content=_required_string(body, "content"),
            )
        )

    @blueprint.patch("/continuation/pages/<page_id>")
    def update_continuation_page(page_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision", "payload"})
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return jsonify(
            continuation.update_page(
                idempotency_key=idempotency_key,
                page_id=page_id,
                base_revision=_required_integer(
                    body,
                    "baseRevision",
                    minimum=1,
                ),
                payload=payload,
            )
        )

    @blueprint.post(
        "/continuation/pages/<page_id>/image-versions/<int:version>/activate"
    )
    def activate_continuation_image(
        page_id: str,
        version: int,
    ) -> Response:
        idempotency_key = _require_idempotency_key()
        _json_body(allowed_keys=set())
        return jsonify(
            continuation.switch_image_version(
                idempotency_key=idempotency_key,
                continuation_page_id=page_id,
                version=version,
            )
        )

    @blueprint.delete("/books/<book_id>/continuation")
    def clear_continuation(book_id: str) -> Response:
        idempotency_key = _require_idempotency_key()
        continuation.clear(
            idempotency_key=idempotency_key,
            book_id=book_id,
        )
        return jsonify({"deleted": True})

    return blueprint


def _base_revision() -> int:
    value = request.args.get("baseRevision")
    if value is None:
        raise ValueError("baseRevision is required")
    return _integer_value(value, "baseRevision", minimum=1)


def _required_query_string(name: str) -> str:
    value = request.args.get(name)
    if (
        value is None
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{name} is required")
    return value


def _optional_query_string(name: str) -> str | None:
    value = request.args.get(name)
    if value is None:
        return None
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _query_choice(
    name: str,
    *,
    allowed: set[str],
    default: str | None,
) -> str | None:
    value = request.args.get(name)
    if value is None:
        return default
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of: {', '.join(sorted(allowed))}"
        )
    return value


def _detail_requested() -> bool:
    value = request.args.get("detail")
    if value is None:
        return False
    if value != "1":
        raise ValueError("detail must be 1")
    return True


def _required_text(body: Mapping[str, Any], key: str) -> str:
    if key not in body or not isinstance(body[key], str):
        raise ValueError(f"{key} must be a string")
    return body[key]


def _body_choice(
    body: Mapping[str, Any],
    key: str,
    *,
    allowed: set[str],
) -> str:
    value = body.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{key} must be one of: {', '.join(sorted(allowed))}"
        )
    return value


def _citations(
    body: dict[str, Any],
    *,
    required: bool = False,
) -> list[dict[str, Any]]:
    if required and "citations" not in body:
        raise ValueError("citations is required")
    citations = body.get("citations", [])
    if not isinstance(citations, list):
        raise ValueError("citations must be an array")
    result: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("every citation must be an object")
        unknown = set(citation) - {"pageId", "excerpt", "score"}
        if unknown:
            raise ValueError(
                "citation has unknown fields: " + ", ".join(sorted(unknown))
            )
        if not isinstance(citation.get("pageId"), str) or not citation["pageId"]:
            raise ValueError("every citation requires pageId")
        if "excerpt" in citation and not isinstance(citation["excerpt"], str):
            raise ValueError("citation excerpt must be a string")
        score = citation.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("citation score must be a finite number or null")
        result.append(dict(citation))
    return result


def _string_list(
    body: dict[str, Any],
    key: str,
    *,
    required: bool = False,
) -> list[str]:
    if required and key not in body:
        raise ValueError(f"{key} is required")
    value = body.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{key} must be a string array")
    return value


def _optional_ordinals(body: Mapping[str, Any]) -> list[int] | None:
    if "ordinals" not in body:
        return None
    value = body["ordinals"]
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 1 for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            "ordinals must be a non-empty array of unique positive integers"
        )
    return value


def _require_exact_keys(body: Mapping[str, Any], allowed: set[str]) -> None:
    unexpected = set(body) - allowed
    if unexpected:
        raise ValueError(
            "fields are not valid for this continuation job kind: "
            + ", ".join(sorted(unexpected))
        )


def _multipart_revision() -> int:
    value = request.form.get("baseRevision")
    if value is None:
        raise ValueError("baseRevision is required")
    return _integer_value(value, "baseRevision", minimum=1)


def _stream_sha256(stream: Any) -> str:
    try:
        stream.seek(0)
    except (AttributeError, OSError) as exc:
        raise ValueError("uploaded file stream must be seekable") from exc
    digest = hashlib.sha256()
    try:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise ValueError("uploaded file stream must return bytes")
            digest.update(chunk)
    finally:
        try:
            stream.seek(0)
        except (AttributeError, OSError) as exc:
            raise ValueError("uploaded file stream must be seekable") from exc
    return digest.hexdigest()


def _nullable_text(body: Mapping[str, Any], key: str) -> str | None:
    if key not in body:
        raise ValueError(f"{key} is required")
    value = body[key]
    if value is not None and (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{key} must be null or a trimmed non-empty string")
    return value


def _qa_sse(event: str, payload: Mapping[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(dict(payload), ensure_ascii=False)}\n\n"
    )
