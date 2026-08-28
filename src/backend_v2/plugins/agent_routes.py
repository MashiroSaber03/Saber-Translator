"""Plugin Agent planning APIs; execution is handed to the global job queue."""

from __future__ import annotations

from pathlib import Path
from flask import Blueprint, Response, jsonify
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import (
    error_response as _error,
    json_body as _json_body,
    require_idempotency_key as _idempotency_key,
    required_string as _required_text,
)
from src.backend_v2.jobs.repository import JobConflict
from src.backend_v2.plugins.agent import (
    PluginAgentSessionNotFound,
    PluginAgentSessionService,
)
from src.backend_v2.plugins.repository import PluginNotFound


def create_plugin_agent_blueprint(
    *,
    data_root: Path,
    engine: Engine,
) -> Blueprint:
    blueprint = Blueprint(
        "plugin_agent_v2",
        __name__,
        url_prefix="/api/v2/plugin-agent",
    )
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
    )

    @blueprint.errorhandler(PluginAgentSessionNotFound)
    def not_found(error: PluginAgentSessionNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(PluginNotFound)
    def plugin_not_found(error: PluginNotFound):
        return _error("not_found", str(error), 404)

    @blueprint.errorhandler(JobConflict)
    def conflict(error: JobConflict):
        return _error("job_conflict", str(error), 409)

    @blueprint.errorhandler(ValueError)
    def validation(error: ValueError):
        return _error("validation_error", str(error), 422)

    @blueprint.post("/sessions")
    def create_session() -> tuple[Response, int]:
        body = _json_body(allowed_keys={"mode", "pluginId"})
        plugin_id = body.get("pluginId")
        if plugin_id is not None and not isinstance(plugin_id, str):
            raise ValueError("pluginId must be a string")
        return (
            jsonify(
                {
                    "session": sessions.create(
                        mode=_required_text(body, "mode"),
                        plugin_id=plugin_id,
                    )
                }
            ),
            201,
        )

    @blueprint.get("/sessions/<session_id>")
    def get_session(session_id: str) -> Response:
        return jsonify({"session": sessions.get(session_id)})

    @blueprint.delete("/sessions/<session_id>")
    def delete_session(session_id: str) -> Response:
        return jsonify(sessions.delete(session_id))

    @blueprint.post("/sessions/<session_id>/messages")
    def send_message(session_id: str) -> Response:
        body = _json_body(allowed_keys={"content"})
        return jsonify(
            {
                "session": sessions.send_message(
                    session_id=session_id,
                    content=_required_text(body, "content"),
                )
            }
        )

    @blueprint.post("/sessions/<session_id>/lock-target")
    def lock_target(session_id: str) -> Response:
        body = _json_body(allowed_keys={"proposal"})
        proposal = body.get("proposal")
        if not isinstance(proposal, dict):
            raise ValueError("proposal must be an object")
        return jsonify(
            {
                "session": sessions.lock_target(
                    session_id=session_id,
                    proposal=proposal,
                )
            }
        )

    @blueprint.post("/sessions/<session_id>/start")
    def start(session_id: str) -> tuple[Response, int]:
        _json_body(allowed_keys=set())
        return (
            jsonify(
                sessions.start(
                    session_id=session_id,
                    idempotency_key=_idempotency_key(),
                )
            ),
            202,
        )

    return blueprint
