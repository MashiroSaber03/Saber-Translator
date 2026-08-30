"""v2 API process entrypoint."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.browser_extension.auth import (
    BROWSER_EXTENSION_ENABLED_ENV,
    BROWSER_EXTENSION_TOKEN_ENV,
)
from src.backend_v2.import_guard import loaded_forbidden_api_modules
from src.backend_v2.logging_config import configure_backend_logging
from src.backend_v2.paths import data_root_fingerprint, ensure_data_root, resolve_data_root
from src.backend_v2.runtime_heartbeat import EpochHeartbeat
from src.backend_v2.runtime_identity import (
    CHILD_LEASE_LOST_EXIT_CODE,
    LauncherParentMonitor,
    RuntimeIdentity,
    start_launcher_parent_monitor,
)
from src.backend_v2.runtime_profile import (
    PROFILE_ENV,
    RuntimeProfile,
    resolve_public_host,
    resolve_runtime_profile,
)
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import ProcessEpochRepository
from src.shared.user_logging import user_log


LOGGER = logging.getLogger("saber.api")


def _waitress_server_options(profile: RuntimeProfile) -> dict[str, object]:
    options: dict[str, object] = {"threads": 24}
    if profile.name == "public":
        options.update(
            trusted_proxy="*",
            trusted_proxy_count=1,
            trusted_proxy_headers={"x-forwarded-for"},
        )
    return options


def run_api(args: object) -> int:
    profile = resolve_runtime_profile(getattr(args, "profile", "local"))
    if profile.name == "public" and not getattr(args, "data_dir", None):
        raise ValueError("--data-dir is required for the public profile")
    public_host = resolve_public_host(profile)
    data_root = ensure_data_root(resolve_data_root(args.data_dir))
    os.environ[PROFILE_ENV] = profile.name
    if not args.probe:
        log_path = configure_backend_logging(
            role="api",
            data_root=data_root,
            console_level=args.log_level,
        )
        LOGGER.debug(
            "API 进程启动：pid=%s，data_root=%s，日志=%s",
            os.getpid(),
            data_root,
            log_path,
        )
    identity = RuntimeIdentity.for_api(test_mode=args.test_mode)
    heartbeat: EpochHeartbeat | None = None
    repository: ProcessEpochRepository | None = None
    engine = create_sqlite_engine(database_path_for(data_root))
    fenced = threading.Event()
    close_server: Callable[[], None] | None = None
    parent_monitor: LauncherParentMonitor | None = None

    def stop_fenced_server() -> None:
        LOGGER.error("API 进程租约失效，正在停止服务")
        fenced.set()
        if close_server is not None:
            close_server()

    def stop_orphaned_server() -> None:
        LOGGER.critical("Launcher 进程已退出，API 立即终止")
        os._exit(CHILD_LEASE_LOST_EXIT_CODE)

    if not identity.test_mode:
        repository = ProcessEpochRepository(engine)
        if not repository.validate(
            role="api",
            epoch_id=identity.epoch_id,
            token=identity.epoch_token,
        ):
            engine.dispose()
            raise RuntimeError("Launcher-issued API epoch is missing, expired, or invalid")
        heartbeat = EpochHeartbeat(
            repository,
            role="api",
            identity=identity,
            on_fenced=stop_fenced_server,
        )
        # API route/runtime construction can take longer than one lease on a
        # busy machine.  The process owns the epoch as soon as validation
        # succeeds, so renewal must cover initialization as well as serving.
        heartbeat.start()
        try:
            parent_monitor = start_launcher_parent_monitor(
                stop_orphaned_server,
                test_mode=identity.test_mode,
            )
        except BaseException:
            heartbeat.stop()
            engine.dispose()
            raise

    app = None
    server = None
    try:
        app = create_api_app(
            ApiSettings(
                data_root=data_root,
                identity=identity,
                epoch_healthy=lambda: not fenced.is_set(),
                engine=engine,
                host=args.host,
                port=args.port,
                profile=profile,
                public_host=public_host,
                browser_extension_enabled=(
                    os.environ.get(BROWSER_EXTENSION_ENABLED_ENV, "") == "1"
                ),
                browser_extension_token=os.environ.get(
                    BROWSER_EXTENSION_TOKEN_ENV,
                    "",
                ),
            )
        )
        if not args.probe:
            LOGGER.debug(
                "API 应用初始化完成：已注册 %s 条路由",
                sum(1 for _rule in app.url_map.iter_rules()),
            )
        if args.probe:
            print(
                json.dumps(
                    {
                        "role": "api",
                        "status": "ready",
                        "epochId": identity.epoch_id,
                        "dataRootFingerprint": data_root_fingerprint(data_root),
                        "forbiddenModules": loaded_forbidden_api_modules(),
                        "routes": sorted(rule.rule for rule in app.url_map.iter_rules()),
                    },
                    sort_keys=True,
                )
            )
            return 0

        from waitress.server import create_server

        server = create_server(
            app,
            host=args.host,
            port=args.port,
            **_waitress_server_options(profile),
        )
        close_server = server.close
        if fenced.is_set():
            return CHILD_LEASE_LOST_EXIT_CODE
        app.extensions["saber_v2_runtime"].start()
        user_log(
            "system",
            f"API 服务已就绪｜{args.host}:{args.port}｜24 个请求线程",
        )
        server.run()
    finally:
        if server is not None:
            LOGGER.debug("API 服务正在关闭")
        if heartbeat is not None:
            heartbeat.stop()
        if parent_monitor is not None:
            parent_monitor.stop()
        if server is not None:
            server.close()
            server.task_dispatcher.shutdown(cancel_pending=True, timeout=5)
        if app is not None:
            app.extensions["saber_v2_runtime"].close()
        engine.dispose()
        if server is not None:
            LOGGER.debug("API 服务已关闭")
    if fenced.is_set():
        return CHILD_LEASE_LOST_EXIT_CODE
    return 0
