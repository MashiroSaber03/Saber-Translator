"""REST control plane and the single global job SSE endpoint."""

from __future__ import annotations

import json
import queue
from typing import Iterator

from flask import Blueprint, Response, jsonify, request, stream_with_context
from sqlalchemy import Engine

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.auth.context import require_admin
from src.backend_v2.api.request_helpers import (
    error_response as _error,
    integer_value as _integer_value,
    json_body as _json_body,
    require_idempotency_key as _require_idempotency_key,
    required_string as _required_string,
)
from src.backend_v2.jobs.events import JobEventBroadcaster
from src.backend_v2.jobs.repository import (
    InvalidJobTransition,
    JobConflict,
    JobNotFound,
    JobQueueRepository,
)
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.runtime_profile import RuntimeProfile
from src.backend_v2.scheduling_policy import (
    SchedulingPolicyRepository,
    available_memory_mib,
)
from src.shared.user_logging import job_label, user_log, user_log_context


def create_jobs_blueprint(
    *,
    engine: Engine,
    broadcaster: JobEventBroadcaster,
    profile: RuntimeProfile,
) -> Blueprint:
    blueprint = Blueprint("jobs_v2", __name__, url_prefix="/api/v2")
    repository = JobQueueRepository(engine)
    retry_service = JobRetryService(engine, profile=profile)
    scheduling_policy = (
        SchedulingPolicyRepository(engine)
        if profile.name == "public"
        else None
    )

    def low_memory() -> bool:
        if scheduling_policy is None:
            return False
        threshold = int(scheduling_policy.load()["minAvailableMemoryMiB"])
        return bool(threshold and available_memory_mib() < threshold)

    def log_job_command(result: dict[str, object], action: str) -> None:
        job_id = str(result.get("jobId") or "")
        kind = str(result.get("kind") or "")
        with user_log_context(job_id=job_id or None):
            user_log(
                "task",
                f"{job_label(kind)}{action}",
            )

    def replacement_job_id(result: dict[str, object]) -> str:
        job_ids = result.get("jobIds")
        if (
            isinstance(job_ids, list)
            and job_ids
            and isinstance(job_ids[0], str)
        ):
            return job_ids[0]
        return ""

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
                limit=_integer_value(
                    request.args.get("limit", "200"),
                    "limit",
                    minimum=1,
                    maximum=200,
                ),
                low_memory=low_memory(),
            )
        )

    @blueprint.get("/jobs/events")
    def stream_events() -> Response:
        header_cursor = request.headers.get("Last-Event-ID")
        query_cursor = request.args.get("after")
        after = _integer_value(
            header_cursor or query_cursor or "0",
            "event cursor",
            minimum=0,
        )
        owner_user_id = effective_owner_id()
        subscription = broadcaster.subscribe(owner_user_id=owner_user_id)

        @stream_with_context
        def generate() -> Iterator[str]:
            cursor = after
            try:
                # Durable catch-up first; the subscription was installed before
                # this query, so events created during catch-up remain queued.
                while True:
                    backlog = repository.events_after(
                        after=cursor,
                        owner_user_id=owner_user_id,
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
                low_memory=low_memory(),
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
        limit = _integer_value(
            request.args.get("limit", "200"),
            "limit",
            minimum=1,
            maximum=1000,
        )
        items = (
            repository.events_before(
                job_id=job_id,
                before=_integer_value(before, "before", minimum=1),
                limit=limit,
            )
            if before is not None
            else repository.events_after(
                job_id=job_id,
                after=_integer_value(after or "0", "after", minimum=0),
                limit=limit,
            )
        )
        return jsonify({"items": items})

    @blueprint.post("/jobs/<job_id>/pause")
    def pause_job(job_id: str) -> Response:
        result = repository.request_pause(job_id)
        log_job_command(result, "已暂停")
        return jsonify({"jobId": result["jobId"], "status": result["status"]})

    @blueprint.post("/jobs/<job_id>/resume")
    def resume_job(job_id: str) -> Response:
        result = repository.resume(job_id)
        log_job_command(result, "已恢复并重新排队")
        return jsonify({"jobId": result["jobId"], "status": result["status"]})

    @blueprint.post("/jobs/<job_id>/continue")
    def continue_job(job_id: str) -> Response:
        result = repository.continue_interrupted(job_id)
        log_job_command(result, "已继续并重新排队")
        return jsonify({"jobId": result["jobId"], "status": result["status"]})

    @blueprint.post("/jobs/<job_id>/cancel")
    def cancel_job(job_id: str) -> Response:
        result = repository.request_cancel(job_id)
        log_job_command(result, "已取消")
        return jsonify({"jobId": result["jobId"], "status": result["status"]})

    @blueprint.post("/jobs/<job_id>/retry")
    def retry_job(job_id: str) -> Response:
        body = _json_body(allowed_keys={"strategy"}, optional=True)
        strategy = (
            _required_string(body, "strategy")
            if "strategy" in body
            else "current"
        )
        result = retry_service.retry(
            job_id=job_id,
            failed_only=False,
            strategy=strategy,
            idempotency_key=_require_idempotency_key(),
        )
        new_job_id = replacement_job_id(result)
        with user_log_context(job_id=new_job_id or None):
            user_log(
                "task",
                f"已从任务 {job_id[:8]} 创建完整重试",
            )
        return jsonify(result), 202

    @blueprint.post("/jobs/<job_id>/retry-failed")
    def retry_failed_job(job_id: str) -> Response:
        body = _json_body(allowed_keys={"strategy"}, optional=True)
        strategy = (
            _required_string(body, "strategy")
            if "strategy" in body
            else "current"
        )
        result = retry_service.retry(
            job_id=job_id,
            failed_only=True,
            strategy=strategy,
            idempotency_key=_require_idempotency_key(),
        )
        new_job_id = replacement_job_id(result)
        with user_log_context(job_id=new_job_id or None):
            user_log(
                "task",
                f"已从任务 {job_id[:8]} 创建失败页面重试",
            )
        return jsonify(result), 202

    @blueprint.post("/jobs/reorder")
    def reorder_jobs() -> Response:
        body = _json_body(allowed_keys={"orderedJobIds"})
        ordered = body.get("orderedJobIds")
        if not isinstance(ordered, list) or not ordered or not all(
            isinstance(value, str) and bool(value.strip()) for value in ordered
        ):
            raise ValueError("orderedJobIds must be a non-empty string array")
        repository.reorder(ordered_job_ids=ordered)
        user_log("task", f"任务队列顺序已更新｜共 {len(ordered)} 个任务")
        return jsonify({"status": "reordered"})

    @blueprint.post("/jobs/queue/pause")
    def pause_queue() -> Response:
        if profile.requires_auth:
            require_admin()
        result = repository.set_queue_paused(True)
        user_log("task", "已暂停任务队列｜当前任务继续运行")
        return jsonify(result)

    @blueprint.post("/jobs/queue/resume")
    def resume_queue() -> Response:
        if profile.requires_auth:
            require_admin()
        result = repository.set_queue_paused(False)
        user_log("task", "已恢复任务队列")
        return jsonify(result)

    @blueprint.post("/jobs/cancel-queued")
    def cancel_queued() -> Response:
        cancelled = repository.cancel_all_queued()
        user_log("task", f"已取消全部排队任务｜共 {cancelled} 个")
        return jsonify({"cancelled": cancelled})

    @blueprint.post("/jobs/history/clear")
    def clear_history() -> Response:
        removed = repository.clear_history()
        user_log("task", f"已清理任务历史｜共 {removed} 个")
        return jsonify({"removed": removed})

    @blueprint.post("/job-batches/<batch_id>/cancel")
    def cancel_batch(batch_id: str) -> Response:
        cancelled = repository.cancel_batch_queued(batch_id)
        user_log("task", f"已取消任务批次中的排队任务｜共 {cancelled} 个")
        return jsonify({"cancelled": cancelled})

    @blueprint.post("/job-batches/<batch_id>/prioritize")
    def prioritize_batch(batch_id: str) -> Response:
        _json_body(allowed_keys=set(), optional=True)
        repository.prioritize_batch(batch_id=batch_id)
        user_log("task", f"任务批次 {batch_id[:8]} 已移到队列前方")
        return jsonify({"status": "prioritized"})

    @blueprint.post("/job-batches/<batch_id>/continue")
    def continue_batch(batch_id: str) -> Response:
        result = repository.continue_batch(batch_id)
        user_log("task", f"已继续任务批次｜共 {result['continued']} 个任务")
        return jsonify(result)

    return blueprint


def _sse(event: dict[str, object]) -> str:
    public_event = {
        key: value for key, value in event.items() if key != "_ownerUserId"
    }
    return (
        f"id: {event['eventId']}\n"
        f"event: {event['type']}\n"
        f"data: {json.dumps(public_event, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )
