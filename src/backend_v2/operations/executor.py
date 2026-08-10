"""Bounded background executors for persisted operations and renders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from typing import Any

from src.backend_v2.operations.repository import (
    OperationFence,
    OperationFenced,
    OperationRepository,
    RenderFence,
    RenderRequestRepository,
)
from src.backend_v2.storage.database import (
    SQLITE_HEARTBEAT_BUSY_RETRY_DELAY_SECONDS,
    SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT,
    is_sqlite_busy_error,
)


OperationHandler = Callable[
    [OperationFence, Mapping[str, Any]],
    Mapping[str, Any],
]
RenderHandler = Callable[[RenderFence], Callable[[Any], None]]
LOGGER = logging.getLogger("saber.operations")


def _short(value: object) -> str:
    return str(value)[:8]


class _LeaseHeartbeat:
    def __init__(
        self,
        renew: Callable[[], object | None],
        *,
        interval_seconds: float = 2,
    ) -> None:
        self.renew = renew
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.fenced = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="operation-attempt-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.interval_seconds * 2))

    def _run(self) -> None:
        busy_failures = 0
        while not self.stop_event.wait(self.interval_seconds):
            try:
                renewed = self.renew()
            except Exception as exc:
                if (
                    is_sqlite_busy_error(exc)
                    and busy_failures < SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT
                ):
                    busy_failures += 1
                    LOGGER.warning(
                        "operation/render attempt 心跳遇到 SQLite 写锁竞争，"
                        "将重试（%s/%s）",
                        busy_failures,
                        SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT,
                    )
                    if self.stop_event.wait(
                        SQLITE_HEARTBEAT_BUSY_RETRY_DELAY_SECONDS
                    ):
                        return
                    continue
                LOGGER.exception("operation/render attempt 心跳执行失败，放弃本次发布")
                self.fenced.set()
                return
            busy_failures = 0
            if renewed is None:
                self.fenced.set()
                return


class DurableOperationExecutor:
    """Fixed-size pool; requests are claimed from SQLite, never HTTP threads."""

    def __init__(
        self,
        repository: OperationRepository,
        *,
        executor_role: str,
        executor_epoch_id: str,
        handlers: Mapping[str, OperationHandler],
        max_workers: int,
        poll_seconds: float = 0.25,
    ) -> None:
        self.repository = repository
        self.executor_role = executor_role
        self.executor_epoch_id = executor_epoch_id
        self.handlers = dict(handlers)
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._admission = threading.Semaphore(max_workers)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{executor_role}-operation",
        )
        self._scheduler = threading.Thread(
            target=self._run,
            name=f"{executor_role}-operation-scheduler",
            daemon=True,
        )

    def start(self) -> None:
        if self.handlers:
            self._scheduler.start()

    def close(self) -> None:
        self._stop.set()
        if self._scheduler.is_alive():
            self._scheduler.join(timeout=3)
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            if not self._admission.acquire(blocking=False):
                continue
            try:
                claimed = self.repository.claim_next(
                    executor_role=self.executor_role,
                    executor_epoch_id=self.executor_epoch_id,
                    allowed_kinds=tuple(self.handlers),
                )
            except OperationFenced:
                self._admission.release()
                self._stop.set()
                return
            except Exception as exc:
                self._admission.release()
                if is_sqlite_busy_error(exc):
                    LOGGER.warning(
                        "%s operation 调度器遇到 SQLite 写锁竞争，将继续轮询",
                        self.executor_role,
                    )
                else:
                    LOGGER.exception(
                        "%s operation 调度器领取失败，将继续轮询",
                        self.executor_role,
                    )
                self._stop.wait(max(1.0, self.poll_seconds))
                continue
            if claimed is None:
                self._admission.release()
                continue
            self._pool.submit(self._execute, *claimed)

    def _execute(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> None:
        started_at = time.monotonic()
        kind = str(operation.get("kind", "unknown"))
        LOGGER.info(
            "操作开始：role=%s operation=%s kind=%s",
            self.executor_role,
            _short(fence.operation_id),
            kind,
        )
        heartbeat = _LeaseHeartbeat(lambda: self.repository.renew(fence))
        heartbeat.start()
        try:
            handler = self.handlers[str(operation["kind"])]
            result = handler(fence, operation)
            if (
                not heartbeat.fenced.is_set()
                and not result.get("__already_published__")
            ):
                self.repository.complete(fence, result=result)
        except OperationFenced:
            LOGGER.warning(
                "操作被 fencing 中断：operation=%s kind=%s",
                _short(fence.operation_id),
                kind,
            )
        except Exception as exc:
            LOGGER.exception(
                "操作失败：operation=%s kind=%s duration=%.2fs",
                _short(fence.operation_id),
                kind,
                time.monotonic() - started_at,
            )
            if not heartbeat.fenced.is_set():
                try:
                    self.repository.fail(
                        fence,
                        code="OPERATION_FAILED",
                        message=str(exc),
                    )
                except OperationFenced:
                    pass
        finally:
            heartbeat.stop()
            self._admission.release()
        if not heartbeat.fenced.is_set():
            LOGGER.info(
                "操作结束：operation=%s kind=%s duration=%.2fs",
                _short(fence.operation_id),
                kind,
                time.monotonic() - started_at,
            )


class WorkerOperationRunner:
    """Run at most one local-model operation at a Worker safe point."""

    def __init__(
        self,
        repository: OperationRepository,
        *,
        worker_epoch_id: str,
        handlers: Mapping[str, OperationHandler],
    ) -> None:
        self.repository = repository
        self.worker_epoch_id = worker_epoch_id
        self.handlers = dict(handlers)

    def run_one(self) -> bool:
        claimed = self.repository.claim_next(
            executor_role="worker",
            executor_epoch_id=self.worker_epoch_id,
            allowed_kinds=tuple(self.handlers),
        )
        if claimed is None:
            return False
        fence, operation = claimed
        started_at = time.monotonic()
        kind = str(operation.get("kind", "unknown"))
        LOGGER.info(
            "Worker 操作开始：operation=%s kind=%s",
            _short(fence.operation_id),
            kind,
        )
        heartbeat = _LeaseHeartbeat(lambda: self.repository.renew(fence))
        heartbeat.start()
        try:
            result = self.handlers[str(operation["kind"])](
                fence,
                operation,
            )
            if (
                not heartbeat.fenced.is_set()
                and not result.get("__already_published__")
            ):
                self.repository.complete(fence, result=result)
        except OperationFenced:
            LOGGER.warning(
                "Worker 操作被 fencing 中断：operation=%s kind=%s",
                _short(fence.operation_id),
                kind,
            )
        except Exception as exc:
            LOGGER.exception(
                "Worker 操作失败：operation=%s kind=%s duration=%.2fs",
                _short(fence.operation_id),
                kind,
                time.monotonic() - started_at,
            )
            if not heartbeat.fenced.is_set():
                try:
                    self.repository.fail(
                        fence,
                        code="OPERATION_FAILED",
                        message=str(exc),
                    )
                except OperationFenced:
                    pass
        finally:
            heartbeat.stop()
        if not heartbeat.fenced.is_set():
            LOGGER.info(
                "Worker 操作结束：operation=%s kind=%s duration=%.2fs",
                _short(fence.operation_id),
                kind,
                time.monotonic() - started_at,
            )
        return True


class DurableRenderExecutor:
    """One render worker coalesces repeated revisions of the same page."""

    def __init__(
        self,
        repository: RenderRequestRepository,
        *,
        api_epoch_id: str,
        handler: RenderHandler | None,
        poll_seconds: float = 0.25,
    ) -> None:
        self.repository = repository
        self.api_epoch_id = api_epoch_id
        self.handler = handler
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="api-render-executor",
            daemon=True,
        )

    def start(self) -> None:
        if self.handler is not None:
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        assert self.handler is not None
        while not self._stop.wait(self.poll_seconds):
            try:
                fence = self.repository.claim_next(api_epoch_id=self.api_epoch_id)
            except OperationFenced:
                self._stop.set()
                return
            except Exception as exc:
                if is_sqlite_busy_error(exc):
                    LOGGER.warning(
                        "render 调度器遇到 SQLite 写锁竞争，将继续轮询"
                    )
                else:
                    LOGGER.exception("render 调度器领取失败，将继续轮询")
                self._stop.wait(max(1.0, self.poll_seconds))
                continue
            if fence is None:
                continue
            started_at = time.monotonic()
            LOGGER.debug(
                "渲染开始：request=%s page=%s revision=%s",
                _short(fence.render_request_id),
                _short(fence.page_id),
                fence.rendering_revision,
            )
            heartbeat = _LeaseHeartbeat(lambda: self.repository.renew(fence))
            heartbeat.start()
            try:
                publisher = self.handler(fence)
                if not heartbeat.fenced.is_set():
                    self.repository.complete(fence, publisher=publisher)
            except OperationFenced:
                LOGGER.warning(
                    "渲染被 fencing 中断：request=%s page=%s",
                    _short(fence.render_request_id),
                    _short(fence.page_id),
                )
            except Exception as exc:
                LOGGER.exception(
                    "渲染失败：request=%s page=%s duration=%.2fs",
                    _short(fence.render_request_id),
                    _short(fence.page_id),
                    time.monotonic() - started_at,
                )
                if not heartbeat.fenced.is_set():
                    try:
                        self.repository.fail(
                            fence,
                            code="RENDER_FAILED",
                            message=str(exc),
                        )
                    except OperationFenced:
                        pass
            finally:
                heartbeat.stop()
            if not heartbeat.fenced.is_set():
                LOGGER.debug(
                    "渲染结束：request=%s page=%s duration=%.2fs",
                    _short(fence.render_request_id),
                    _short(fence.page_id),
                    time.monotonic() - started_at,
                )
