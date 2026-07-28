"""Flask application factory for v2-only API routes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from flask import Blueprint, Flask, Response, jsonify
from flask_cors import CORS
from sqlalchemy import Engine
import yaml

from src.backend_v2.import_guard import assert_api_import_boundary
from src.backend_v2.paths import data_root_fingerprint, project_root
from src.backend_v2.runtime_identity import RuntimeIdentity


@dataclass(frozen=True, slots=True)
class ApiSettings:
    data_root: Path
    identity: RuntimeIdentity
    epoch_healthy: Callable[[], bool] = lambda: True
    engine: Engine | None = None


def _load_openapi_document() -> dict[str, Any]:
    spec_path = project_root() / "openapi" / "v2.yaml"
    with spec_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise RuntimeError("OpenAPI document must be an object")
    return document


def _create_v2_blueprint(settings: ApiSettings) -> Blueprint:
    blueprint = Blueprint("api_v2", __name__, url_prefix="/api/v2")

    @blueprint.get("/health")
    def health() -> Response:
        healthy = settings.epoch_healthy()
        response = jsonify(
            {
                "status": "ok" if healthy else "fenced",
                "role": "api",
                "schemaVersion": "v2",
                "epochId": settings.identity.epoch_id,
                "dataRootFingerprint": data_root_fingerprint(settings.data_root),
            }
        )
        response.status_code = 200 if healthy else 503
        return response

    @blueprint.get("/openapi.json")
    def openapi_document() -> Response:
        return Response(
            json.dumps(_load_openapi_document(), ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )

    return blueprint


def create_api_app(settings: ApiSettings) -> Flask:
    app = Flask("saber_translator_v2", static_folder=None)
    app.config.update(
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=512 * 1024 * 1024,
        SABER_V2_DATA_ROOT=str(settings.data_root),
        SABER_V2_API_EPOCH_ID=settings.identity.epoch_id,
    )
    CORS(app)
    app.register_blueprint(_create_v2_blueprint(settings))
    from src.backend_v2.content.routes import create_content_blueprint
    from src.backend_v2.storage.database import create_sqlite_engine, database_path_for

    engine = settings.engine or create_sqlite_engine(database_path_for(settings.data_root))
    app.register_blueprint(
        create_content_blueprint(data_root=settings.data_root, engine=engine)
    )

    assert_api_import_boundary()
    return app
