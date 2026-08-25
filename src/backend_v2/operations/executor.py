"""Bounded background executors for persisted operations and renders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from typing import Any

from src.backend_v2.auth.ownership import owner_scope
from src.backend_v2.operations.repository import (
    OperationFence,
    OperationFenced,
    OperationRepository,
    RenderFence,
    RenderRequestRepository,
)
from src.backend_v2.storage.database import is_sqlite_busy_error
from src.shared.user_logging import (
    RetryLogEpisode,
    log_step_failed,
    log_step_finished,
    log_step_started,
    user_log,
    user_log_context,
)


OperationHandler = Callable[
    [OperationFence, Mapping[str, Any]],
    Mapping[str, Any],
]
RenderHandler = Callable[[RenderFence], Callable[[Any], None]]
LOGGER = logging.getLogger("saber.operations")


def _short(value: object) -> str:
    return str(value)[:8]


def _execute_operation(
    repository: OperationRepository,
    handlers: Mapping[str, OperationHandler],
    fence: OperationFence,
    operation: Mapping[str, Any],
) -> None:
    """Run the shared persisted-operation lifecycle for either executor role."""

    started_at = time.monotonic()
    kind = str(operation.get("kind", "unknown"))
    with user_log_context(
        operation_id=fence.operation_id,
        step_kind=kind,
    ):
        log_step_started()
        try:
            handler = handlers.get(kind)
            if handler is None:
                raise RuntimeError(f"没有即时操作处理器：{kind}")
            with owner_scope(fence.owner_user_id):
                result = handler(fence, operation)
            if not isinstance(result, Mapping):
                raise RuntimeError("operation handler result must be an object")
            already_published = result.get("__already_published__", False)
            if not isinstance(already_published, bool):
                raise RuntimeError("operation publication marker must be boolean")
            if not already_published:
                repository.complete(fence, result=result)
            log_step_finished(duration=time.monotonic() - started_at)
        except OperationFenced as error:
            LOGGER.debug(
                "即时操作结果已被 fencing：operation=%s kind=%s reason=%s",
                _short(fence.operation_id),
                kind,
                error,
            )
            user_log(
                "system",
                "目标数据或执行状态已变化，本次操作结果不再应用",
            )
        except Exception as error:
            log_step_failed(error, duration=time.monotonic() - started_at)
            try:
                repository.fail(
                    fence,
                    code="OPERATION_FAILED",
                    message=str(error),
                )
            except OperationFenced:
                pass


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
        concurrency_limit: Callable[[], int] | None = None,
        poll_seconds: float = 0.25,
    ) -> None:
        if max_workers < 1:
            raise ValueError("operation executor size must be positive")
        self.repository = repository
        self.executor_role = executor_role
        self.executor_epoch_id = executor_epoch_id
        self.handlers = dict(handlers)
        self.poll_seconds = poll_seconds
        self.max_workers = max_workers
        self.concurrency_limit = concurrency_limit
        self._stop = threading.Event()
        self._admission_lock = threading.Lock()
        self._active = 0
        self._scheduler_log = RetryLogEpisode("即时操作调度")
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
            try:
                reserved = self._reserve_slot()
            except Exception as exc:
                if self._scheduler_log.record_failure(exc):
                    LOGGER.exception(
                        "%s operation 调度器读取并发上限失败，将继续轮询",
                        self.executor_role,
                    )
                self._stop.wait(max(1.0, self.poll_seconds))
                continue
            if not reserved:
                continue
            try:
                claimed = self.repository.claim_next(
                    executor_role=self.executor_role,
                    executor_epoch_id=self.executor_epoch_id,
                    allowed_kinds=tuple(self.handlers),
                )
            except OperationFenced:
                self._release_slot()
                self._stop.set()
                return
            except Exception as exc:
                self._release_slot()
                if is_sqlite_busy_error(exc):
                    LOGGER.debug(
                        "%s operation 调度器遇到 SQLite 写锁竞争，将继续轮询",
                        self.executor_role,
                    )
                else:
                    if self._scheduler_log.record_failure(exc):
                        LOGGER.exception(
                            "%s operation 调度器领取失败，将继续轮询",
                            self.executor_role,
                        )
                self._stop.wait(max(1.0, self.poll_seconds))
                continue
            self._scheduler_log.report_recovery()
            if claimed is None:
                self._release_slot()
                continue
            self._pool.submit(self._execute, *claimed)

    def _reserve_slot(self) -> bool:
        limit = self.max_workers
        if self.concurrency_limit is not None:
            value = self.concurrency_limit()
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RuntimeError("operation concurrency limit must be positive")
            limit = min(limit, value)
        with self._admission_lock:
            if self._active >= limit:
                return False
            self._active += 1
            return True

    def _release_slot(self) -> None:
        with self._admission_lock:
            self._active -= 1

    def _execute(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> None:
        try:
            _execute_operation(self.repository, self.handlers, fence, operation)
        finally:
            self._release_slot()


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
        _execute_operation(self.repository, self.handlers, fence, operation)
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
        self._scheduler_log = RetryLogEpisode("编辑结果调度")
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
                    LOGGER.debug(
                        "render 调度器遇到 SQLite 写锁竞争，将继续轮询"
                    )
                else:
                    if self._scheduler_log.record_failure(exc):
                        LOGGER.exception("render 调度器领取失败，将继续轮询")
                self._stop.wait(max(1.0, self.poll_seconds))
                continue
            self._scheduler_log.report_recovery()
            if fence is None:
                continue
            self._execute(fence)

    def _execute(self, fence: RenderFence) -> None:
        assert self.handler is not None
        started_at = time.monotonic()
        with user_log_context(
            operation_id=fence.render_request_id,
            step_kind="live_render",
        ):
            user_log(
                "step",
                f"开始｜文档版本 {fence.rendering_revision}",
                level=logging.DEBUG,
            )
            try:
                with owner_scope(fence.owner_user_id):
                    publisher = self.handler(fence)
                    published = self.repository.complete(
                        fence,
                        publisher=publisher,
                    )
            except OperationFenced:
                LOGGER.debug(
                    "渲染被 fencing 中断：request=%s page=%s",
                    _short(fence.render_request_id),
                    _short(fence.page_id),
                )
            except Exception as exc:
                log_step_failed(
                    exc,
                    duration=time.monotonic() - started_at,
                )
                try:
                    self.repository.fail(
                        fence,
                        code="RENDER_FAILED",
                        message=str(exc),
                    )
                except OperationFenced:
                    pass
            else:
                outcome = "完成" if published else "已由新版本接管"
                user_log(
                    "step",
                    f"{outcome}｜文档版本 {fence.rendering_revision}｜"
                    f"耗时 {time.monotonic() - started_at:.2f} 秒",
                    level=logging.DEBUG,
                )
