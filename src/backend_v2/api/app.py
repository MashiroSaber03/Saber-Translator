"""Flask application factory for v2-only API routes."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
import logging
from pathlib import Path
import socket
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from flask import Blueprint, Flask, Response, g, jsonify, request
from sqlalchemy import Engine
import yaml

from src.backend_v2.import_guard import assert_api_import_boundary
from src.backend_v2.paths import data_root_fingerprint, project_root
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.runtime_profile import (
    RuntimeProfile,
    resolve_runtime_profile,
    validate_profile_bind_host,
)
from src.shared.user_logging import user_log


LOGGER = logging.getLogger("saber.api.http")


@dataclass(frozen=True, slots=True)
class ApiSettings:
    data_root: Path
    identity: RuntimeIdentity
    engine: Engine
    epoch_healthy: Callable[[], bool] = lambda: True
    host: str = "0.0.0.0"
    port: int = 5000
    profile: RuntimeProfile = resolve_runtime_profile("local")
    public_host: str | None = None


@dataclass(slots=True)
class ApiRuntimeServices:
    job_events: Any
    executors: tuple[Any, ...] = ()

    def start(self) -> None:
        self.job_events.start()
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
    from src.backend_v2.public_policy import (
        DEFAULT_PUBLIC_USER_POLICY,
        PublicUserPolicyRepository,
    )
    from src.backend_v2.scheduling_policy import SchedulingPolicyRepository

    blueprint = Blueprint("api_v2", __name__, url_prefix="/api/v2")
    if settings.profile.requires_auth:
        from src.backend_v2.auth.repository import AuthRepository

        auth_repository = AuthRepository(settings.engine)
        public_policy = PublicUserPolicyRepository(settings.engine)
        scheduling_policy = SchedulingPolicyRepository(settings.engine)
        registration_requires_invite = auth_repository.registration_requires_invite
        load_public_policy = public_policy.load
        load_scheduling_limit = lambda: scheduling_policy.load()[
            "maxDeepLearningConcurrency"
        ]
    else:
        local_policy = deepcopy(DEFAULT_PUBLIC_USER_POLICY)
        local_policy["settings"]["lamaDisableResize"]["editable"] = True
        local_policy["settings"]["parallel"]["allowed"] = True
        registration_requires_invite = lambda: False
        load_public_policy = lambda: deepcopy(local_policy)
        load_scheduling_limit = lambda: None

    @blueprint.get("/health")
    def health() -> Response:
        healthy = settings.epoch_healthy()
        payload: dict[str, object] = {
            "status": "ok" if healthy else "fenced",
        }
        if settings.profile.name != "public" or not request.headers.get(
            "X-Forwarded-For"
        ):
            payload.update(
                {
                    "role": "api",
                    "schemaVersion": "v2",
                    "epochId": settings.identity.epoch_id,
                    "dataRootFingerprint": data_root_fingerprint(settings.data_root),
                }
            )
        response = jsonify(payload)
        response.status_code = 200 if healthy else 503
        return response

    @blueprint.get("/openapi.json")
    def openapi_document() -> Response:
        return Response(
            json.dumps(_load_openapi_document(), ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )

    @blueprint.get("/system/capabilities")
    def capabilities() -> Response:
        return jsonify(
            {
                "profile": settings.profile.name,
                "requiresAuth": settings.profile.requires_auth,
                "browserCredentials": settings.profile.browser_credentials,
                "registrationRequiresInvite": (
                    registration_requires_invite()
                ),
                "publicUserPolicy": load_public_policy(),
                "scheduling": {
                    "maxDeepLearningConcurrency": load_scheduling_limit(),
                },
                "features": {
                    "plugins": settings.profile.allow_plugins,
                    "webImport": settings.profile.allow_web_import,
                    "localProviders": settings.profile.allow_local_providers,
                },
            }
        )

    @blueprint.get("/system/server-info")
    def server_info() -> Response:
        if settings.profile.name == "public":
            return jsonify({"error": {"code": "not_found", "message": "not found"}}), 404
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


def _install_request_logging(app: Flask) -> None:
    """Record useful API timings without logging bodies, queries, or secrets."""

    @app.before_request
    def start_request_timer() -> None:
        g.saber_request_started_at = time.perf_counter()

    @app.after_request
    def log_request(response: Response) -> Response:
        started_at = getattr(g, "saber_request_started_at", None)
        duration_ms = (
            (time.perf_counter() - started_at) * 1000
            if isinstance(started_at, float)
            else 0.0
        )
        path = request.path
        successful_health_check = (
            request.method == "GET"
            and path == "/api/v2/health"
            and response.status_code < 400
            and duration_ms < 1000
        )
        if successful_health_check:
            response.headers["X-Response-Time"] = f"{duration_ms / 1000:.3f}s"
            return response
        if response.status_code >= 500 and path != "/api/v2/health":
            user_log(
                "error",
                f"接口请求失败｜{request.method} {path}｜"
                f"状态码 {response.status_code}｜耗时 {duration_ms:.1f} 毫秒",
            )
        LOGGER.debug(
            "HTTP %s %s -> %s (%.1f ms, client=%s)",
            request.method,
            path,
            response.status_code,
            duration_ms,
            request.remote_addr or "-",
        )
        response.headers["X-Response-Time"] = f"{duration_ms / 1000:.3f}s"
        return response

    @app.teardown_request
    def log_unhandled_request_error(error: BaseException | None) -> None:
        # Flask closes streaming response generators with GeneratorExit when an
        # SSE client reloads, navigates away, or otherwise disconnects.  That is
        # normal transport lifecycle, not an application failure.
        if isinstance(error, GeneratorExit):
            return
        if error is not None:
            LOGGER.error(
                "HTTP %s %s raised an unhandled exception",
                request.method,
                request.path,
                exc_info=(type(error), error, error.__traceback__),
            )


def create_api_app(settings: ApiSettings) -> Flask:
    app = Flask("saber_translator_v2", static_folder=None)
    app.config.update(
        JSON_SORT_KEYS=False,
        SABER_V2_DATA_ROOT=str(settings.data_root),
        SABER_V2_API_EPOCH_ID=settings.identity.epoch_id,
        SABER_V2_PROFILE=settings.profile.name,
    )
    from src.backend_v2.public_policy import PublicPolicyDenied
    from src.backend_v2.storage.assets import AssetQuotaExceeded

    @app.errorhandler(PublicPolicyDenied)
    def public_policy_denied(error: PublicPolicyDenied):
        return jsonify(
            {"error": {"code": error.code, "message": str(error)}}
        ), 403

    @app.errorhandler(AssetQuotaExceeded)
    def asset_quota_exceeded(error: AssetQuotaExceeded):
        return jsonify(
            {
                "error": {
                    "code": "asset_quota_exceeded",
                    "message": str(error),
                    "usedBytes": error.used_bytes,
                    "quotaBytes": error.quota_bytes,
                    "incomingBytes": error.incoming_bytes,
                }
            }
        ), 413

    @app.errorhandler(PermissionError)
    def permission_denied(error: PermissionError):
        return jsonify(
            {"error": {"code": "forbidden", "message": str(error)}}
        ), 403

    if settings.profile.name == "public":
        validate_profile_bind_host(settings.profile, settings.host)
        if settings.public_host is None:
            raise ValueError("public_host is required for the public profile")
        allowed_hosts = {
            "127.0.0.1",
            "localhost",
            "::1",
            settings.public_host,
        }

        @app.before_request
        def validate_public_host():
            host = (urlsplit(request.host_url).hostname or "").lower().rstrip(".")
            if host not in allowed_hosts:
                return jsonify(
                    {"error": {"code": "invalid_host", "message": "invalid host"}}
                ), 400
            return None

        @app.after_request
        def add_public_security_headers(response: Response) -> Response:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "font-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = (
                "camera=(), microphone=(), geolocation=(), payment=()"
            )
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            if request.path.startswith("/api/") or request.path == "/":
                response.headers["Cache-Control"] = "no-store"
            elif request.path.startswith(("/js/", "/assets/")):
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            return response

    _install_request_logging(app)
    if settings.profile.requires_auth:
        from src.backend_v2.auth.authorization import install_route_ownership
        from src.backend_v2.auth.http import install_authentication
        from src.backend_v2.auth.repository import AuthRepository

        auth_repository = AuthRepository(settings.engine)
        install_authentication(
            app,
            repository=auth_repository,
            profile=settings.profile,
        )
        install_route_ownership(
            app,
            engine=settings.engine,
            profile=settings.profile,
        )
    app.register_blueprint(_create_v2_blueprint(settings))
    if settings.profile.requires_auth:
        from src.backend_v2.auth.routes import create_auth_blueprint

        app.register_blueprint(
            create_auth_blueprint(engine=settings.engine, profile=settings.profile)
        )
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
    from src.backend_v2.api.web import create_web_blueprint
    from src.backend_v2.api.system_routes import create_system_blueprint
    from src.backend_v2.storage.epochs import ProcessEpochRepository
    from src.backend_v2.storage.platform_repositories import ProviderRateLimiter
    from src.backend_v2.scheduling_policy import (
        SchedulingPolicyCache,
        SchedulingPolicyRepository,
    )
    from src.shared.openai_rate_limits import configure_provider_rate_limit_store

    engine = settings.engine
    configure_provider_rate_limit_store(ProviderRateLimiter(engine))
    broadcaster = JobEventBroadcaster(
        JobQueueRepository(engine),
        epoch_repository=ProcessEpochRepository(engine),
    )
    render_service = AuthoritativeRenderService(
        data_root=settings.data_root,
        engine=engine,
    )
    render_executor = DurableRenderExecutor(
        RenderRequestRepository(engine),
        api_epoch_id=settings.identity.epoch_id,
        handler=render_service.prepare,
        poll_seconds=0.05,
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
    scheduling_policy = (
        SchedulingPolicyCache(SchedulingPolicyRepository(engine))
        if settings.profile.name == "public"
        else None
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
        max_workers=8 if scheduling_policy is not None else 4,
        concurrency_limit=(
            (lambda: int(scheduling_policy.load()["apiOperationConcurrency"]))
            if scheduling_policy is not None
            else None
        ),
    )
    app.extensions["saber_v2_runtime"] = ApiRuntimeServices(
        job_events=broadcaster,
        executors=(cpu_operation_executor, render_executor),
    )
    app.register_blueprint(
        create_content_blueprint(
            data_root=settings.data_root,
            engine=engine,
            profile=settings.profile,
        )
    )
    app.register_blueprint(
        create_system_blueprint(engine=engine, profile=settings.profile)
    )
    app.register_blueprint(
        create_jobs_blueprint(
            engine=engine,
            broadcaster=broadcaster,
            profile=settings.profile,
        )
    )
    app.register_blueprint(
        create_insight_blueprint(
            engine=engine,
            data_root=settings.data_root,
            profile=settings.profile,
        )
    )
    app.register_blueprint(
        create_studio_blueprint(
            engine=engine,
            data_root=settings.data_root,
            profile=settings.profile,
        )
    )
    app.register_blueprint(
        create_operations_blueprint(
            data_root=settings.data_root,
            engine=engine,
            profile=settings.profile,
        )
    )
    if settings.profile.allow_plugins:
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
    app.register_blueprint(
        create_translation_blueprint(engine=engine, profile=settings.profile)
    )
    app.register_blueprint(
        create_settings_blueprint(
            data_root=settings.data_root,
            engine=engine,
            profile=settings.profile,
        )
    )
    app.register_blueprint(
        create_transfer_blueprint(data_root=settings.data_root, engine=engine)
    )
    if settings.profile.allow_web_import:
        app.register_blueprint(
            create_web_import_blueprint(
                data_root=settings.data_root,
                engine=engine,
            )
        )
    app.register_blueprint(create_web_blueprint())

    # Unit tests may share a pytest process with suites that import Torch and
    # model interfaces. Production roles still enforce the
    # process-wide guard; the isolated API probe verifies test-mode import graphs.
    if not settings.identity.test_mode:
        assert_api_import_boundary()
    return app
