"""Manga Insight v2 query surface and durable task commands."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, Response, jsonify, request
from sqlalchemy import Engine

from src.backend_v2.insight.commands import InsightAnalysisCommandService
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
from src.backend_v2.jobs.repository import JobConflict


def create_insight_blueprint(*, engine: Engine) -> Blueprint:
    blueprint = Blueprint("insight_v2", __name__, url_prefix="/api/v2/insight")
    repository = InsightRepository(engine)
    commands = InsightAnalysisCommandService(engine)
    derived = InsightDerivedRepository(engine)
    derived_commands = InsightDerivedCommandService(engine)

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


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status
