"""Manga Insight v2 query surface and durable task commands."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from typing import Any

from flask import Blueprint, Response, jsonify, request, stream_with_context
from sqlalchemy import Engine

from src.backend_v2.insight.commands import InsightAnalysisCommandService
from src.backend_v2.insight.continuation import (
    ContinuationCommandService,
    ContinuationRepository,
)
from src.backend_v2.insight.derived import (
    InsightDerivedCommandService,
    InsightDerivedRepository,
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
    TransientRequestRepository,
    citations_for,
    select_candidates,
    suggested_questions,
)
from src.backend_v2.jobs.repository import JobConflict


def create_insight_blueprint(
    *,
    engine: Engine,
    qa_algorithms: QAApiAlgorithms | None = None,
) -> Blueprint:
    blueprint = Blueprint("insight_v2", __name__, url_prefix="/api/v2/insight")
    repository = InsightRepository(engine)
    commands = InsightAnalysisCommandService(engine)
    derived = InsightDerivedRepository(engine)
    derived_commands = InsightDerivedCommandService(engine)
    continuation = ContinuationRepository(engine)
    continuation_commands = ContinuationCommandService(engine)
    qa_requests = TransientRequestRepository(engine)
    qa_commands = InsightQACommandService(
        engine,
        repository=qa_requests,
    )
    qa_api = qa_algorithms or DefaultQAApiAlgorithms()

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
                chapter_id=request.args.get("chapterId"),
                after=int(request.args.get("cursor", "0")),
                limit=int(request.args.get("limit", "50")),
            )
        )

    @blueprint.get("/pages/<page_id>")
    def page_detail(page_id: str) -> Response:
        return jsonify(
            repository.page_detail(
                page_id=page_id,
                run_id=request.args.get("run_id"),
            )
        )

    @blueprint.get("/runs/<run_id>")
    def run_detail(run_id: str) -> Response:
        return jsonify(repository.get_run(run_id))

    @blueprint.post("/analysis-jobs")
    def create_analysis_job():
        result = commands.create_analysis_job(
            command=_json_body(),
            idempotency_key=_require_idempotency_key(),
        )
        return jsonify(result), 202

    @blueprint.get("/artifacts/overviews/<template>")
    def get_overview(template: str) -> Response:
        book_id = request.args.get("bookId", "")
        if not book_id:
            raise ValueError("bookId is required")
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
        body = _json_body()
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
        book_id = request.args.get("bookId", "")
        if not book_id:
            raise ValueError("bookId is required")
        timeline = derived.get_timeline(book_id=book_id)
        if timeline is None:
            raise InsightNotFound("timeline not found")
        return jsonify(timeline)

    @blueprint.post("/timeline")
    def rebuild_timeline():
        body = _json_body()
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
        _json_body()
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
        _json_body()
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
        book_id = request.args.get("bookId", "")
        if not book_id:
            raise ValueError("bookId is required")
        return jsonify(derived.qa_status(book_id=book_id))

    @blueprint.post("/books/<book_id>/qa")
    def ask_question(book_id: str) -> Response:
        handle = qa_commands.create(
            book_id=book_id,
            command=_json_body(),
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
                    status = str(state["status"])
                    if status == "completed":
                        break
                    if status in {"failed", "cancelled"}:
                        error = (
                            dict(state["result"].get("error", {}))
                            if isinstance(state["result"], dict)
                            else {}
                        )
                        yield _qa_sse(
                            "error",
                            {
                                "code": error.get(
                                    "code",
                                    "QA_RETRIEVAL_FAILED",
                                ),
                                "message": error.get(
                                    "message",
                                    "问答检索已取消或失败",
                                ),
                            },
                        )
                        return
                    now = time.monotonic()
                    if now >= heartbeat_at:
                        yield ": heartbeat\n\n"
                        heartbeat_at = now + 15
                    time.sleep(0.1)
                result = qa_requests.consume(
                    request_id=handle.request_id,
                    connection_token=handle.connection_token,
                )
                raw_candidates = result.get("candidates", [])
                candidates = select_candidates(
                    (
                        raw_candidates
                        if isinstance(raw_candidates, list)
                        else []
                    ),
                    threshold=float(handle.options["threshold"]),
                    top_k=max(
                        int(handle.options["topK"]) * 4,
                        int(handle.options["topK"]),
                    ),
                )
                config = qa_commands.materialize_api_config(handle.config)
                if bool(handle.options["useReranker"]) and candidates:
                    candidates = [
                        dict(value)
                        for value in qa_api.rerank(
                            question=handle.question,
                            candidates=candidates,
                            top_k=int(handle.options["topK"]),
                            config=config,
                        )
                    ]
                else:
                    candidates = candidates[: int(handle.options["topK"])]
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
                    for chunk in qa_api.stream_answer(
                        question=handle.question,
                        candidates=candidates,
                        config=config,
                        cancelled=cancelled,
                    ):
                        if chunk:
                            yield _qa_sse("chunk", {"text": chunk})
                        else:
                            yield ": heartbeat\n\n"
                yield _qa_sse(
                    "done",
                    {
                        "citations": citations,
                        "suggestedQuestions": suggested_questions(
                            handle.question,
                            candidates,
                        ),
                    },
                )
            except GeneratorExit:
                raise
            except Exception as exc:
                yield _qa_sse(
                    "error",
                    {"code": "QA_FAILED", "message": str(exc)},
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
                for event in analysis.get("key_events", [])
            )
            warnings = "\n".join(
                f"- {warning['code']}: {warning['message']}"
                for warning in analysis.get("warnings", [])
            )
            body = (
                f"# 第 {detail['displayPageNumber']} 页\n\n"
                f"## 页面摘要\n\n{analysis.get('page_summary', '')}\n\n"
                f"## 关键事件\n\n{events or '- 无'}\n\n"
                f"## 连续性说明\n\n"
                f"{analysis.get('continuity_notes', '') or '无'}\n\n"
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

    @blueprint.get("/notes")
    def list_notes() -> Response:
        book_id = request.args.get("bookId", "")
        if not book_id:
            raise ValueError("bookId is required")
        return jsonify({"items": repository.list_notes(book_id=book_id)})

    @blueprint.post("/notes")
    def create_note():
        _require_idempotency_key()
        body = _json_body()
        return (
            jsonify(
                repository.create_note(
                    book_id=_required_string(body, "bookId"),
                    title=_required_string(body, "title"),
                    content=str(body.get("content", "")),
                    page_ids=_page_ids(body),
                )
            ),
            201,
        )

    @blueprint.patch("/notes/<note_id>")
    def update_note(note_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body()
        return jsonify(
            repository.update_note(
                note_id=note_id,
                base_revision=int(body.get("baseRevision", 0)),
                title=_required_string(body, "title"),
                content=str(body.get("content", "")),
                page_ids=_page_ids(body),
            )
        )

    @blueprint.delete("/notes/<note_id>")
    def delete_note(note_id: str) -> Response:
        _require_idempotency_key()
        base_revision = int(
            request.args.get(
                "baseRevision",
                request.headers.get("If-Match", "0"),
            )
        )
        repository.delete_note(
            note_id=note_id,
            base_revision=base_revision,
        )
        return jsonify({"deleted": True})

    @blueprint.get("/books/<book_id>/continuation")
    def continuation_state(book_id: str) -> Response:
        return jsonify(continuation.bootstrap(book_id=book_id))

    @blueprint.post("/books/<book_id>/continuation/sync")
    def sync_continuation(book_id: str) -> Response:
        _json_body()
        return jsonify(continuation.sync_latest(book_id=book_id))

    @blueprint.patch("/continuation/projects/<project_id>")
    def update_continuation_project(project_id: str) -> Response:
        body = _json_body()
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        return jsonify(
            continuation.update_project(
                project_id=project_id,
                base_revision=int(body.get("baseRevision", 0)),
                config=config,
            )
        )

    @blueprint.post("/books/<book_id>/continuation/script-jobs")
    def continuation_script_job(book_id: str):
        _json_body()
        return (
            jsonify(
                continuation_commands.create_script_job(
                    book_id=book_id,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.patch("/continuation/projects/<project_id>/script")
    def update_continuation_script(project_id: str) -> Response:
        body = _json_body()
        return jsonify(
            continuation.update_script(
                project_id=project_id,
                base_revision=int(body.get("baseRevision", 0)),
                content=_required_string(body, "content"),
            )
        )

    @blueprint.post("/books/<book_id>/continuation/page-jobs")
    def continuation_page_job(book_id: str):
        body = _json_body()
        ordinals = body.get("ordinals")
        if ordinals is not None and (
            not isinstance(ordinals, list)
            or not all(isinstance(value, int) for value in ordinals)
        ):
            raise ValueError("ordinals must be an integer array")
        return (
            jsonify(
                continuation_commands.create_pages_job(
                    book_id=book_id,
                    ordinals=ordinals,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.patch("/continuation/pages/<page_id>")
    def update_continuation_page(page_id: str) -> Response:
        body = _json_body()
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return jsonify(
            continuation.update_page(
                page_id=page_id,
                base_revision=int(body.get("baseRevision", 0)),
                payload=payload,
            )
        )

    @blueprint.post("/books/<book_id>/continuation/image-jobs")
    def continuation_image_job(book_id: str):
        body = _json_body()
        ordinals = body.get("ordinals")
        if ordinals is not None and (
            not isinstance(ordinals, list)
            or not all(isinstance(value, int) for value in ordinals)
        ):
            raise ValueError("ordinals must be an integer array")
        return (
            jsonify(
                continuation_commands.create_images_job(
                    book_id=book_id,
                    ordinals=ordinals,
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/books/<book_id>/continuation/export-jobs")
    def continuation_export_job(book_id: str):
        body = _json_body()
        return (
            jsonify(
                continuation_commands.create_export_job(
                    book_id=book_id,
                    output_format=_required_string(body, "format"),
                    idempotency_key=_require_idempotency_key(),
                )
            ),
            202,
        )

    @blueprint.post("/continuation/pages/<page_id>/image-versions/<int:version>/activate")
    def activate_continuation_image(
        page_id: str,
        version: int,
    ) -> Response:
        _json_body()
        return jsonify(
            continuation.switch_image_version(
                continuation_page_id=page_id,
                version=version,
            )
        )

    @blueprint.delete("/books/<book_id>/continuation")
    def clear_continuation(book_id: str) -> Response:
        continuation.clear(book_id=book_id)
        return jsonify({"deleted": True})

    return blueprint


def _json_body() -> dict[str, Any]:
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


def _required_string(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _page_ids(body: dict[str, Any]) -> list[str]:
    citations = body.get("citations", [])
    if not isinstance(citations, list):
        raise ValueError("citations must be an array")
    result: list[str] = []
    for citation in citations:
        if isinstance(citation, str):
            page_id = citation
        elif isinstance(citation, dict):
            page_id = citation.get("pageId")
        else:
            page_id = None
        if not isinstance(page_id, str) or not page_id:
            raise ValueError("every citation requires pageId")
        result.append(page_id)
    return result


def _qa_sse(event: str, payload: Mapping[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(dict(payload), ensure_ascii=False)}\n\n"
    )


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status
