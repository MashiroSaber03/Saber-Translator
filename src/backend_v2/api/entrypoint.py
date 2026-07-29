"""v2 API process entrypoint."""

from __future__ import annotations

import json
import logging
import os
import threading

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.import_guard import loaded_forbidden_api_modules
from src.backend_v2.logging_config import configure_backend_logging
from src.backend_v2.paths import data_root_fingerprint, ensure_data_root, resolve_data_root
from src.backend_v2.runtime_heartbeat import EpochHeartbeat
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import ProcessEpochRepository


LOGGER = logging.getLogger("saber.api")


def run_api(args: object) -> int:
    data_root = ensure_data_root(resolve_data_root(getattr(args, "data_dir", None)))
    if not getattr(args, "probe", False):
        log_path = configure_backend_logging(
            role="api",
            data_root=data_root,
            console_level=getattr(args, "log_level", None),
        )
        LOGGER.info(
            "API 进程启动：pid=%s，data_root=%s，日志=%s",
            os.getpid(),
            data_root,
            log_path,
        )
    identity = RuntimeIdentity.for_api(test_mode=bool(getattr(args, "test_mode", False)))
    heartbeat: EpochHeartbeat | None = None
    repository: ProcessEpochRepository | None = None
    engine = None
    fenced = threading.Event()
    if not identity.test_mode:
        engine = create_sqlite_engine(database_path_for(data_root))
        repository = ProcessEpochRepository(engine)
        if not repository.validate(
            role="api",
            epoch_id=identity.epoch_id,
            token=identity.epoch_token,
        ):
            engine.dispose()
            raise RuntimeError("Launcher-issued API epoch is missing, expired, or invalid")
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=identity,
            epoch_healthy=lambda: not fenced.is_set(),
            engine=engine,
            host=str(getattr(args, "host", "0.0.0.0")),
            port=int(getattr(args, "port", 5000)),
        )
    )
    if not getattr(args, "probe", False):
        LOGGER.info(
            "API 应用初始化完成：已注册 %s 条路由",
            sum(1 for _rule in app.url_map.iter_rules()),
        )

    if getattr(args, "probe", False):
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
        app.extensions["saber_v2_runtime"].close()
        if engine is not None:
            engine.dispose()
        return 0

    from waitress.server import create_server

    server = create_server(
        app,
        host=str(getattr(args, "host", "0.0.0.0")),
        port=int(getattr(args, "port", 5000)),
        threads=24,
    )
    app.extensions["saber_v2_runtime"].start()
    LOGGER.info(
        "API 服务就绪：http://127.0.0.1:%s/（监听 %s:%s，线程数=24）",
        getattr(args, "port", 5000),
        getattr(args, "host", "0.0.0.0"),
        getattr(args, "port", 5000),
    )

    def stop_fenced_server() -> None:
        LOGGER.error("API 进程租约失效，正在停止服务")
        fenced.set()
        server.close()

    if repository is not None:
        heartbeat = EpochHeartbeat(
            repository,
            role="api",
            identity=identity,
            on_fenced=stop_fenced_server,
        )
        heartbeat.start()
    try:
        server.run()
    finally:
        LOGGER.info("API 服务正在关闭")
        if heartbeat is not None:
            heartbeat.stop()
        server.close()
        server.task_dispatcher.shutdown(cancel_pending=True, timeout=5)
        app.extensions["saber_v2_runtime"].close()
        if engine is not None:
            engine.dispose()
        LOGGER.info("API 服务已关闭")
    if fenced.is_set():
        return 75
    return 0
