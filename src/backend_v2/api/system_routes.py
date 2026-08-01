"""System control endpoints whose effects are executed by the Worker."""

from __future__ import annotations

from flask import Blueprint, jsonify
from sqlalchemy import Engine

from src.backend_v2.api.request_helpers import error_response
from src.backend_v2.worker.model_lifecycle import (
    ModelInferenceBusy,
    WorkerModelControlRepository,
)


def create_system_blueprint(*, engine: Engine) -> Blueprint:
    blueprint = Blueprint(
        "system_controls_v2",
        __name__,
        url_prefix="/api/v2/system",
    )
    repository = WorkerModelControlRepository(engine)

    @blueprint.errorhandler(ModelInferenceBusy)
    def inference_busy(error: ModelInferenceBusy):
        return error_response("model_inference_busy", str(error), 409)

    @blueprint.post("/release-models")
    def release_model_cache():
        return jsonify(repository.request_release()), 202

    return blueprint
