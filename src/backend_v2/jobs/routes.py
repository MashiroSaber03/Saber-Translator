"""REST control plane and the single global job SSE endpoint."""

from __future__ import annotations

import json
import queue
from typing import Iterator

from flask import Blueprint, Response, jsonify, redirect, request, stream_with_context
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    json_body as _json_body,
    require_idempotency_key as _require_idempotency_key,
)
from src.backend_v2.jobs.events import JobEventBroadcaster
from src.backend_v2.jobs.repository import (
    InvalidJobTransition,
    JobConflict,
    JobNotFound,
    JobQueueRepository,
)
from src.backend_v2.jobs.retry import JobRetryService


def create_jobs_blueprint(
    *,
    engine: Engine,
    broadcaster: JobEventBroadcaster,
) -> Blueprint:
    blueprint = Blueprint("jobs_v2", __name__, url_prefix="/api/v2")
    repository = JobQueueRepository(engine)
    retry_service = JobRetryService(engine)

    @blueprint.errorhandler(JobNotFound)
    def not_found(error: JobNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(InvalidJobTransition)
    def invalid_transition(error: InvalidJobTransition):
        return _error("invalid_job_transition", str(error), 409)

    @blueprint.errorhandler(JobConflict)
    def conflict(error: JobConflict):
        return _error("job_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.get("/jobs")
    def list_jobs() -> Response:
        return jsonify(
            repository.list_jobs(
                scope=request.args.get("scope", "queue"),
                status=request.args.get("status"),
                kind=request.args.get("type"),
                book_id=request.args.get("book_id"),
                limit=int(request.args.get("limit", "200")),
            )
        )

    @blueprint.get("/jobs/events")
    def stream_events() -> Response:
        header_cursor = request.headers.get("Last-Event-ID")
        query_cursor = request.args.get("after")
        after = int(header_cursor or query_cursor or "0")
        subscription = broadcaster.subscribe()

        @stream_with_context
        def generate() -> Iterator[str]:
            cursor = after
            try:
                # Durable catch-up first; the subscription was installed before
                # this query, so events created during catch-up remain queued.
                while True:
                    backlog = repository.events_after(
                        after=cursor,
                        limit=1000,
                    )
                    if not backlog:
                        break
                    for event in backlog:
                        event_id = int(event["eventId"])
                        if event_id <= cursor:
                            continue
                        cursor = event_id
                        yield _sse(event)
                    if len(backlog) < 1000:
                        break
                yield "retry: 2000\n\n"
                while not subscription.closed.is_set():
                    try:
                        event = subscription.queue.get(timeout=15)
                    except queue.Empty:
                        yield ": heartbeat\n\n"
                        continue
                    if event is None:
                        return
                    event_id = int(event["eventId"])
                    if event_id <= cursor:
                        continue
                    cursor = event_id
                    yield _sse(event)
            finally:
                broadcaster.unsubscribe(subscription)

        response = Response(
            generate(),
            content_type="text/event-stream; charset=utf-8",
        )
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @blueprint.get("/jobs/snapshot")
    def get_job_snapshot() -> Response:
        return jsonify(
            repository.job_snapshot(
                job_ids=request.args.getlist("job_id"),
            )
        )

    @blueprint.get("/jobs/<job_id>")
    def get_job(job_id: str) -> Response:
        return jsonify(repository.get_job(job_id))

    @blueprint.get("/jobs/<job_id>/events")
    def get_job_events(job_id: str) -> Response:
        # Validate the target so an unknown ID is a true 404.
        repository.get_job(job_id)
        before = request.args.get("before")
        after = request.args.get("after")
        if before is not None and after is not None:
            raise ValueError("before and after cannot be used together")
        items = (
            repository.events_before(
                job_id=job_id,
                before=int(before),
                limit=int(request.args.get("limit", "200")),
            )
            if before is not None
            else repository.events_after(
                job_id=job_id,
                after=int(after or "0"),
                limit=int(request.args.get("limit", "200")),
            )
        )
        return jsonify({"items": items})

    @blueprint.get("/jobs/<job_id>/download")
    def download_job_artifact(job_id: str) -> Response:
        job = repository.get_job(job_id)
        artifacts = job.get("artifacts", [])
        if not artifacts:
            raise JobNotFound("job has no downloadable artifact")
        selected_kind = request.args.get("kind")
        artifact = next(
            (
                item
                for item in artifacts
                if selected_kind is None or item["kind"] == selected_kind
            ),
            None,
        )
        if artifact is None:
            raise JobNotFound("requested job artifact not found")
        return redirect(
            f"{artifact['url']}?download=1&filename={job_id}",
            code=302,
        )

    @blueprint.post("/jobs/<job_id>/pause")
    def pause_job(job_id: str) -> Response:
        _require_idempotency_key()
        return jsonify(repository.request_pause(job_id))

    @blueprint.post("/jobs/<job_id>/resume")
    def resume_job(job_id: str) -> Response:
        _require_idempotency_key()
        return jsonify(repository.resume(job_id))

    @blueprint.post("/jobs/<job_id>/continue")
    def continue_job(job_id: str) -> Response:
        _require_idempotency_key()
        return jsonify(repository.continue_interrupted(job_id))

    @blueprint.post("/jobs/<job_id>/cancel")
    def cancel_job(job_id: str) -> Response:
        _require_idempotency_key()
        return jsonify(repository.request_cancel(job_id))

    @blueprint.post("/jobs/<job_id>/retry")
    def retry_job(job_id: str) -> Response:
        body = _json_body(allowed_keys={"strategy"}, optional=True)
        return jsonify(
            retry_service.retry(
                job_id=job_id,
                failed_only=False,
                strategy=str(body.get("strategy", "current")),
                idempotency_key=_require_idempotency_key(),
            )
        ), 202

    @blueprint.post("/jobs/<job_id>/retry-failed")
    def retry_failed_job(job_id: str) -> Response:
        body = _json_body(allowed_keys={"strategy"}, optional=True)
        return jsonify(
            retry_service.retry(
                job_id=job_id,
                failed_only=True,
                strategy=str(body.get("strategy", "current")),
                idempotency_key=_require_idempotency_key(),
            )
        ), 202

    @blueprint.post("/jobs/reorder")
    def reorder_jobs() -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"orderedJobIds", "baseRevision"})
        ordered = body.get("orderedJobIds")
        if not isinstance(ordered, list) or not all(
            isinstance(value, str) for value in ordered
        ):
            raise ValueError("orderedJobIds must be a string array")
        revision = repository.reorder(
            ordered_job_ids=ordered,
            base_revision=int(body.get("baseRevision", 0)),
        )
        return jsonify({"queueRevision": revision})

    @blueprint.post("/jobs/cancel-queued")
    def cancel_queued() -> Response:
        _require_idempotency_key()
        return jsonify({"cancelled": repository.cancel_all_queued()})

    @blueprint.post("/jobs/history/clear")
    def clear_history() -> Response:
        _require_idempotency_key()
        return jsonify({"removed": repository.clear_history()})

    @blueprint.get("/job-batches/<batch_id>")
    def get_batch(batch_id: str) -> Response:
        return jsonify(repository.get_batch(batch_id))

    @blueprint.post("/job-batches/<batch_id>/cancel")
    def cancel_batch(batch_id: str) -> Response:
        _require_idempotency_key()
        return jsonify({"cancelled": repository.cancel_batch_queued(batch_id)})

    @blueprint.post("/job-batches/<batch_id>/prioritize")
    def prioritize_batch(batch_id: str) -> Response:
        _require_idempotency_key()
        body = _json_body(allowed_keys={"baseRevision"})
        return jsonify(
            {
                "queueRevision": repository.prioritize_batch(
                    batch_id=batch_id,
                    base_revision=int(body.get("baseRevision", 0)),
                )
            }
        )

    @blueprint.post("/job-batches/<batch_id>/continue")
    def continue_batch(batch_id: str) -> Response:
        _require_idempotency_key()
        return jsonify(repository.continue_batch(batch_id))

    return blueprint


def _sse(event: dict[str, object]) -> str:
    return (
        f"id: {event['eventId']}\n"
        f"event: {event['type']}\n"
        f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )
