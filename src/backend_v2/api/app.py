"""Flask application factory for v2-only API routes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket
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
    host: str = "0.0.0.0"
    port: int = 5000


@dataclass(slots=True)
class ApiRuntimeServices:
    job_events: Any
    executors: tuple[Any, ...] = ()

    def start(self) -> None:
        for executor in self.executors:
            executor.start()

    def close(self) -> None:
        for executor in reversed(self.executors):
            executor.close()
        self.job_events.close()


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

    @blueprint.get("/system/server-info")
    def server_info() -> Response:
        hostname = socket.gethostname()
        try:
            lan_address = socket.gethostbyname(hostname)
        except OSError:
            lan_address = "127.0.0.1"
        return jsonify(
            {
                "hostname": hostname,
                "host": settings.host,
                "port": settings.port,
                "lanUrl": f"http://{lan_address}:{settings.port}",
            }
        )

    return blueprint


def create_api_app(settings: ApiSettings) -> Flask:
    app = Flask("saber_translator_v2", static_folder=None)
    app.config.update(
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=1024 * 1024 * 1024,
        SABER_V2_DATA_ROOT=str(settings.data_root),
        SABER_V2_API_EPOCH_ID=settings.identity.epoch_id,
    )
    CORS(app)
    app.register_blueprint(_create_v2_blueprint(settings))
    from src.backend_v2.content.routes import create_content_blueprint
    from src.backend_v2.jobs.events import JobEventBroadcaster
    from src.backend_v2.jobs.repository import JobQueueRepository
    from src.backend_v2.jobs.routes import create_jobs_blueprint
    from src.backend_v2.insight.routes import create_insight_blueprint
    from src.backend_v2.operations.routes import create_operations_blueprint
    from src.backend_v2.plugins.routes import create_plugins_blueprint
    from src.backend_v2.plugins.agent_routes import (
        create_plugin_agent_blueprint,
    )
    from src.backend_v2.operations.executor import (
        DurableOperationExecutor,
        DurableRenderExecutor,
    )
    from src.backend_v2.operations.repair import PageRepairService
    from src.backend_v2.operations.repository import (
        OperationRepository,
        RenderRequestRepository,
    )
    from src.backend_v2.rendering.service import AuthoritativeRenderService
    from src.backend_v2.settings.routes import create_settings_blueprint
    from src.backend_v2.studio.routes import create_studio_blueprint
    from src.backend_v2.studio.repository import StudioRepository
    from src.backend_v2.studio.service import StudioOperationService
    from src.backend_v2.translation.routes import create_translation_blueprint
    from src.backend_v2.transfer.routes import create_transfer_blueprint
    from src.backend_v2.web_import.routes import (
        create_web_import_blueprint,
    )
    from src.backend_v2.storage.database import create_sqlite_engine, database_path_for

    engine = settings.engine or create_sqlite_engine(database_path_for(settings.data_root))
    broadcaster = JobEventBroadcaster(JobQueueRepository(engine))
    render_service = AuthoritativeRenderService(
        data_root=settings.data_root,
        engine=engine,
    )
    render_executor = DurableRenderExecutor(
        RenderRequestRepository(engine),
        api_epoch_id=settings.identity.epoch_id,
        handler=render_service.prepare,
    )
    repair_service = PageRepairService(
        data_root=settings.data_root,
        engine=engine,
        repository=OperationRepository(engine),
    )
    from src.backend_v2.translation.interactive_operations import (
        InteractivePageOperationService,
    )

    remote_page_operations = InteractivePageOperationService(
        data_root=settings.data_root,
        engine=engine,
        repository=repair_service.repository,
    )
    studio_operations = StudioOperationService(
        engine=engine,
        data_root=settings.data_root,
        repository=StudioRepository(engine),
    )
    cpu_operation_executor = DurableOperationExecutor(
        repair_service.repository,
        executor_role="api",
        executor_epoch_id=settings.identity.epoch_id,
        handlers={
            "page_repair": repair_service.handle,
            "bubble_translate": remote_page_operations.handle,
            "studio_generate": studio_operations.handle,
            "studio_chat": studio_operations.handle,
            "studio_summary": studio_operations.handle,
        },
        max_workers=4,
    )
    app.extensions["saber_v2_runtime"] = ApiRuntimeServices(
        job_events=broadcaster,
        executors=(cpu_operation_executor, render_executor),
    )
    app.register_blueprint(
        create_content_blueprint(data_root=settings.data_root, engine=engine)
    )
    app.register_blueprint(
        create_jobs_blueprint(engine=engine, broadcaster=broadcaster)
    )
    app.register_blueprint(
        create_insight_blueprint(
            engine=engine,
            data_root=settings.data_root,
        )
    )
    app.register_blueprint(
        create_studio_blueprint(
            engine=engine,
            data_root=settings.data_root,
        )
    )
    app.register_blueprint(
        create_operations_blueprint(data_root=settings.data_root, engine=engine)
    )
    app.register_blueprint(
        create_plugins_blueprint(
            data_root=settings.data_root,
            engine=engine,
        )
    )
    app.register_blueprint(
        create_plugin_agent_blueprint(
            data_root=settings.data_root,
            engine=engine,
        )
    )
    app.register_blueprint(create_translation_blueprint(engine=engine))
    app.register_blueprint(
        create_settings_blueprint(data_root=settings.data_root, engine=engine)
    )
    app.register_blueprint(
        create_transfer_blueprint(data_root=settings.data_root, engine=engine)
    )
    app.register_blueprint(
        create_web_import_blueprint(
            data_root=settings.data_root,
            engine=engine,
        )
    )

    assert_api_import_boundary()
    return app
