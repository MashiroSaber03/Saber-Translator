"""System control endpoints whose effects are executed by the Worker."""

from __future__ import annotations

from flask import Blueprint, jsonify
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import error_response
from src.backend_v2.auth.context import require_admin
from src.backend_v2.runtime_profile import RuntimeProfile, resolve_runtime_profile
from src.backend_v2.worker.model_lifecycle import (
    ModelInferenceBusy,
    WorkerCommandFenced,
    WorkerModelControlRepository,
)


def create_system_blueprint(
    *, engine: Engine, profile: RuntimeProfile | None = None
) -> Blueprint:
    profile = profile or resolve_runtime_profile("local")
    blueprint = Blueprint(
        "system_controls_v2",
        __name__,
        url_prefix="/api/v2/system",
    )
    repository = WorkerModelControlRepository(engine)

    @blueprint.errorhandler(ModelInferenceBusy)
    def inference_busy(error: ModelInferenceBusy):
        return error_response("model_inference_busy", str(error), 409)

    @blueprint.errorhandler(WorkerCommandFenced)
    def worker_unavailable(error: WorkerCommandFenced):
        return error_response("worker_unavailable", str(error), 503)

    @blueprint.post("/release-models")
    def release_model_cache():
        if profile.requires_auth:
            require_admin()
        return jsonify(repository.request_release()), 202

    return blueprint
