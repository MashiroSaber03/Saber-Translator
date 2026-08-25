"""Role heartbeat loop that turns zero-row renewal into immediate self-fencing."""

from __future__ import annotations

from collections.abc import Callable
import logging
import threading

from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.database import (
    SQLITE_HEARTBEAT_BUSY_RETRY_DELAY_SECONDS,
    SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT,
    is_sqlite_busy_error,
)
from src.backend_v2.storage.epochs import ProcessEpochRepository
from src.shared.user_logging import inline_log_text, user_log


LOGGER = logging.getLogger("saber.runtime.heartbeat")


class EpochHeartbeat:
    def __init__(
        self,
        repository: ProcessEpochRepository,
        *,
        role: str,
        identity: RuntimeIdentity,
        interval_seconds: float = 2.0,
        on_fenced: Callable[[], None] | None = None,
    ) -> None:
        if role not in {"api", "worker"}:
            raise ValueError("only API and Worker have runtime heartbeat loops")
        self._repository = repository
        self._role = role
        self._identity = identity
        self._interval_seconds = interval_seconds
        self._on_fenced = on_fenced
        self._stop = threading.Event()
        self._fenced = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def healthy(self) -> bool:
        return not self._fenced.is_set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("heartbeat already started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"v2-{self._role}-epoch-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        role_label = "接口进程" if self._role == "api" else "工作进程"
        while not self._stop.wait(self._interval_seconds):
            busy_failures = 0
            while True:
                try:
                    renewed = self._repository.renew(
                        role=self._role,  # type: ignore[arg-type]
                        epoch_id=self._identity.epoch_id,
                        token=self._identity.epoch_token,
                    )
                except Exception as exc:
                    if (
                        is_sqlite_busy_error(exc)
                        and busy_failures < SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT
                    ):
                        busy_failures += 1
                        LOGGER.debug(
                            "%s epoch 心跳遇到 SQLite 写锁竞争，将有限重试："
                            "attempt=%s/%s",
                            self._role.upper(),
                            busy_failures,
                            SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT,
                        )
                        if self._stop.wait(
                            SQLITE_HEARTBEAT_BUSY_RETRY_DELAY_SECONDS
                        ):
                            return
                        continue
                    LOGGER.exception(
                        "%s epoch 心跳执行失败，执行器立即自我隔离",
                        self._role.upper(),
                    )
                    user_log(
                        "error",
                        f"{role_label}心跳失败，已停止领取新任务｜"
                        f"{inline_log_text(exc)}",
                    )
                    self._fence()
                    return
                break
            if renewed:
                continue
            LOGGER.error("%s epoch 心跳失租，执行器立即自我隔离", self._role.upper())
            user_log(
                "error",
                f"{role_label}运行权已失效，已停止领取新任务",
            )
            self._fence()
            return

    def _fence(self) -> None:
        self._fenced.set()
        self._stop.set()
        if self._on_fenced is not None:
            self._on_fenced()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self._interval_seconds * 2))
