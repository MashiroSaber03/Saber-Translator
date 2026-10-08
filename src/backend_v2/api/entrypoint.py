"""v2 API process entrypoint."""

from __future__ import annotations

from src.backend_v2.storage.startup import current_storage_process
from src.storage_migrator.control import business_ready

import json
import logging
import os
import socket
import threading

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.browser_extension.auth import (
    BROWSER_EXTENSION_ENABLED_ENV,
    BROWSER_EXTENSION_TOKEN_ENV,
)
from src.backend_v2.import_guard import loaded_forbidden_api_modules
from src.backend_v2.logging_config import configure_backend_logging
from src.backend_v2.paths import data_root_fingerprint, ensure_data_root, resolve_data_root
from src.backend_v2.runtime_identity import (
    API_BIND_FAILED_EXIT_CODE,
    LAUNCHER_PARENT_LOST_EXIT_CODE,
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


def _create_http_server(app, *, host: str, port: int, profile: RuntimeProfile):
    from waitress.adjustments import Adjustments
    from waitress.server import create_server

    options = _waitress_server_options(profile)
    if os.name != "nt":
        return create_server(app, host=host, port=port, **options)

    # Windows port reuse can bind successfully while requests reach another API.
    # Give Waitress the actual, exclusively bound listeners instead of probing.
    listeners = []
    try:
        for family, socktype, proto, address in Adjustments(host=host, port=port).listen:
            listener = socket.socket(family, socktype, proto)
            listeners.append(listener)
            if family == socket.AF_INET6:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(address)
        return create_server(app, sockets=listeners, **options)
    except BaseException:
        for listener in listeners:
            listener.close()
        raise


@current_storage_process("api")
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
    engine = create_sqlite_engine(database_path_for(data_root))
    parent_monitor: LauncherParentMonitor | None = None

    def stop_orphaned_server() -> None:
        LOGGER.critical("Launcher 进程已退出，API 立即终止")
        os._exit(LAUNCHER_PARENT_LOST_EXIT_CODE)

    if not identity.test_mode:
        repository = ProcessEpochRepository(engine)
        if not repository.validate(
            role="api",
            epoch_id=identity.epoch_id,
            token=identity.epoch_token,
        ):
            engine.dispose()
            raise RuntimeError("Launcher-issued API epoch is missing or invalid")
        try:
            parent_monitor = start_launcher_parent_monitor(
                stop_orphaned_server,
                test_mode=identity.test_mode,
            )
        except BaseException:
            engine.dispose()
            raise

    app = None
    server = None
    runtime_stop = threading.Event()
    runtime_start_thread = None
    try:
        app = create_api_app(
            ApiSettings(
                data_root=data_root,
                identity=identity,
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

        try:
            server = _create_http_server(
                app, host=args.host, port=args.port, profile=profile,
            )
        except OSError as error:
            message = f"无法监听 {args.host}:{args.port}：{error}"
            LOGGER.exception(message)
            user_log("error", message, level=logging.ERROR)
            return API_BIND_FAILED_EXIT_CODE
        def start_when_committed():
            while not business_ready(data_root):
                if runtime_stop.wait(0.1):
                    return
            if not runtime_stop.is_set():
                app.extensions["saber_v2_runtime"].start()

        if business_ready(data_root):
            app.extensions["saber_v2_runtime"].start()
        else:
            runtime_start_thread = threading.Thread(target=start_when_committed, name="storage-admission", daemon=True)
            runtime_start_thread.start()
        user_log(
            "system",
            f"API 服务已就绪｜{args.host}:{args.port}｜24 个请求线程",
        )
        server.run()
    finally:
        runtime_stop.set()
        if runtime_start_thread is not None:
            runtime_start_thread.join()
        if server is not None:
            LOGGER.debug("API 服务正在关闭")
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
    return 0
