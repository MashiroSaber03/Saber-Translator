"""Worker scheduler that reconstructs all durable work from SQLite."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import logging
import sqlite3
import threading
import time
from typing import Any

from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobQueueRepository,
)


StepHandler = Callable[[AttemptFence, Mapping[str, Any]], Mapping[str, Any]]
BatchStepHandler = Callable[
    [AttemptFence, Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]
DEEP_LEARNING_STEP_KINDS = frozenset({"detect", "ocr", "color", "repair"})
PIPELINE_BUSY_RETRY_LIMIT = 3
PIPELINE_BUSY_RETRY_BASE_SECONDS = 0.05
MIN_SCHEDULER_POLL_SECONDS = 0.1
MAX_SCHEDULER_POLL_SECONDS = 0.5
LOGGER = logging.getLogger("saber.worker.jobs")


def _short(value: object) -> str:
    return str(value)[:8]


def _is_sqlite_busy_error(exc: BaseException) -> bool:
    """Recognize SQLite writer-contention errors through SQLAlchemy wrappers."""

    candidates: list[BaseException] = [exc]
    seen: set[int] = set()
    while candidates:
        current = candidates.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, sqlite3.OperationalError):
            code = getattr(current, "sqlite_errorcode", None)
            if isinstance(code, int) and (code & 0xFF) in {
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            }:
                return True
            message = str(current).lower()
            if (
                "database is locked" in message
                or "database table is locked" in message
            ):
                return True
        for nested in (
            getattr(current, "orig", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(nested, BaseException):
                candidates.append(nested)
    return False


def _step_log_fields(step: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(step.get("stepKind", "unknown")),
        _short(step.get("stepId", "-")),
        _short(step.get("pageId") or step.get("jobItemId") or "-"),
    )


class AttemptHeartbeat:
    """Renew a job attempt independently from potentially long model calls."""

    def __init__(
        self,
        repository: JobQueueRepository,
        fence: AttemptFence,
        *,
        interval_seconds: float = 2.0,
    ) -> None:
        self.repository = repository
        self.fence = fence
        self.interval_seconds = interval_seconds
        self.fenced = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"job-heartbeat-{fence.job_id[:8]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 2))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                renewed = self.repository.renew_attempt(self.fence)
            except Exception as exc:
                if _is_sqlite_busy_error(exc):
                    LOGGER.warning(
                        "任务 attempt 心跳遇到 SQLite 写锁竞争，将继续重试："
                        "job=%s attempt=%s",
                        _short(self.fence.job_id),
                        _short(self.fence.attempt_id),
                    )
                    continue
                LOGGER.exception(
                    "任务 attempt 心跳执行失败，立即放弃本次 attempt："
                    "job=%s attempt=%s",
                    _short(self.fence.job_id),
                    _short(self.fence.attempt_id),
                )
                self.fenced.set()
                return
            if renewed is None:
                LOGGER.warning(
                    "任务租约失效：job=%s attempt=%s",
                    _short(self.fence.job_id),
                    _short(self.fence.attempt_id),
                )
                self.fenced.set()
                return
            self.fence = renewed


class JobWorkerLoop:
    """Single global FIFO scheduler with persisted step checkpoints."""

    def __init__(
        self,
        repository: JobQueueRepository,
        *,
        worker_epoch_id: str,
        handlers: Mapping[str, StepHandler],
        batch_handlers: Mapping[str, BatchStepHandler] | None = None,
        plugin_runtime: Any | None = None,
        safe_point: Callable[[], bool] | None = None,
        on_activity: Callable[[], None] | None = None,
        idle_poll_seconds: float = 0.25,
    ) -> None:
        self.repository = repository
        self.worker_epoch_id = worker_epoch_id
        self.handlers = dict(handlers)
        self.batch_handlers = dict(batch_handlers or {})
        if not set(self.batch_handlers).issubset(self.handlers):
            raise ValueError("every batch handler requires a matching step handler")
        self.plugin_runtime = plugin_runtime
        self.safe_point = safe_point
        self.on_activity = on_activity
        if idle_poll_seconds <= 0:
            raise ValueError("idle poll interval must be positive")
        self.idle_poll_seconds = idle_poll_seconds

    def run(self, stop_event: threading.Event) -> None:
        LOGGER.info("持久任务调度器开始运行")
        while not stop_event.is_set():
            if self.safe_point is not None and self.safe_point():
                continue
            try:
                fence = self.repository.claim_next(
                    worker_epoch_id=self.worker_epoch_id
                )
            except AttemptFenced:
                LOGGER.error("Worker epoch 已失效，停止领取任务")
                stop_event.set()
                return
            if fence is None:
                stop_event.wait(self.idle_poll_seconds)
                continue
            self._note_activity()
            self._run_attempt(fence, stop_event)
        LOGGER.info("持久任务调度器已停止")

    def _run_attempt(
        self,
        fence: AttemptFence,
        stop_event: threading.Event,
    ) -> None:
        started_at = time.monotonic()
        heartbeat = AttemptHeartbeat(self.repository, fence)
        heartbeat.start()
        try:
            config = self.repository.attempt_config(fence)
            if self.plugin_runtime is not None:
                config = self.plugin_runtime.before_job(fence, config)
            execution_mode = str(config.get("executionMode", "sequential"))
            if execution_mode not in {"sequential", "parallel"}:
                raise ValueError(f"unsupported execution mode: {execution_mode}")
            LOGGER.info(
                "任务开始：job=%s attempt=%s mode=%s",
                _short(fence.job_id),
                _short(fence.attempt_id),
                execution_mode,
            )
            if execution_mode == "parallel":
                self._run_parallel_attempt(heartbeat, stop_event, config)
            else:
                self._run_sequential_attempt(heartbeat, stop_event)
        except AttemptFenced:
            LOGGER.warning(
                "任务执行被 fencing 中断：job=%s attempt=%s",
                _short(fence.job_id),
                _short(fence.attempt_id),
            )
            return
        except Exception as exc:
            LOGGER.exception(
                "任务生命周期失败：job=%s attempt=%s",
                _short(fence.job_id),
                _short(fence.attempt_id),
            )
            if not heartbeat.fenced.is_set():
                try:
                    self.repository.fail_job(
                        heartbeat.fence,
                        code="JOB_LIFECYCLE_FAILED",
                        message=str(exc),
                    )
                except AttemptFenced:
                    pass
        finally:
            heartbeat.stop()
            if self.plugin_runtime is not None:
                self.plugin_runtime.release_job_state(fence.job_id)
            self._note_activity()
            LOGGER.info(
                "任务轮次结束：job=%s attempt=%s duration=%.2fs fenced=%s",
                _short(fence.job_id),
                _short(fence.attempt_id),
                time.monotonic() - started_at,
                heartbeat.fenced.is_set(),
            )

    def _note_activity(self) -> None:
        if self.on_activity is not None:
            self.on_activity()

    def _run_sequential_attempt(
        self,
        heartbeat: AttemptHeartbeat,
        stop_event: threading.Event,
    ) -> None:
        last_step_id: str | None = None
        try:
            while not stop_event.is_set() and not heartbeat.fenced.is_set():
                fence = heartbeat.fence
                status = self.repository.control_status(fence)
                if status in {"pausing", "cancelling"}:
                    LOGGER.info(
                        "任务进入安全排空：job=%s status=%s last_step=%s",
                        _short(fence.job_id),
                        status,
                        _short(last_step_id or "-"),
                    )
                    self.repository.acknowledge_drain(
                        fence,
                        pool_id="main",
                        worker_slot=0,
                        last_step_id=last_step_id,
                    )
                    self.repository.finalize_drain(
                        fence,
                        expected_slots={("main", 0)},
                    )
                    return
                if self.safe_point is not None and self.safe_point():
                    continue
                handled_batch = False
                config = self.repository.attempt_config(fence)
                for batch_kind, batch_handler in self.batch_handlers.items():
                    step_ordinal = self.repository.ready_step_ordinal(
                        fence,
                        step_kind=batch_kind,
                    )
                    if step_ordinal is None:
                        continue
                    batch = self.repository.next_step_batch(
                        fence,
                        step_kind=batch_kind,
                        limit=self._batch_size(
                            batch_kind,
                            config,
                            step_ordinal=step_ordinal,
                        ),
                    )
                    if not batch:
                        continue
                    last_step_id = str(batch[-1]["stepId"])
                    self._execute_batch(
                        heartbeat,
                        batch_handler,
                        batch,
                    )
                    handled_batch = True
                    break
                if handled_batch:
                    continue
                ordinary_kinds = tuple(
                    kind
                    for kind in self.handlers
                    if kind not in self.batch_handlers
                )
                step = self.repository.next_step(
                    fence,
                    allowed_kinds=ordinary_kinds,
                )
                if step is None:
                    pending, running = self.repository.active_step_counts(fence)
                    if pending or running:
                        if pending and not running:
                            unsupported = tuple(
                                kind
                                for kind in self.repository.pending_step_kinds(fence)
                                if kind not in self.handlers
                            )
                            if unsupported:
                                kinds = ", ".join(unsupported)
                                LOGGER.error(
                                    "任务步骤无处理器：job=%s kinds=%s",
                                    _short(fence.job_id),
                                    kinds,
                                )
                                self.repository.fail_job(
                                    fence,
                                    code="UNSUPPORTED_STEP_KIND",
                                    message=f"Worker 没有以下步骤的处理器：{kinds}",
                                )
                                return
                        stop_event.wait(
                            min(
                                max(
                                    self.idle_poll_seconds,
                                    MIN_SCHEDULER_POLL_SECONDS,
                                ),
                                MAX_SCHEDULER_POLL_SECONDS,
                            )
                        )
                        continue
                    final_status = self._finish_job(fence)
                    LOGGER.info(
                        "任务全部步骤处理结束：job=%s status=%s",
                        _short(fence.job_id),
                        final_status or "running",
                    )
                    return
                last_step_id = str(step["stepId"])
                handler = self.handlers.get(str(step["stepKind"]))
                if handler is None:
                    LOGGER.error(
                        "任务步骤无处理器：job=%s kind=%s step=%s",
                        _short(fence.job_id),
                        step["stepKind"],
                        _short(last_step_id),
                    )
                    self.repository.fail_job(
                        fence,
                        code="UNSUPPORTED_STEP_KIND",
                        message=f"Worker 没有步骤处理器：{step['stepKind']}",
                    )
                    return
                step_kind, step_id, page_id = _step_log_fields(step)
                step_started_at = time.monotonic()
                LOGGER.info(
                    "步骤开始：job=%s kind=%s step=%s page=%s",
                    _short(fence.job_id),
                    step_kind,
                    step_id,
                    page_id,
                )
                if not self._before_pipeline(heartbeat.fence, step):
                    continue
                try:
                    checkpoint = handler(
                        heartbeat.fence,
                        step,
                    )
                except Exception as exc:
                    LOGGER.exception(
                        "步骤失败：job=%s kind=%s step=%s page=%s duration=%.2fs",
                        _short(fence.job_id),
                        step_kind,
                        step_id,
                        page_id,
                        time.monotonic() - step_started_at,
                    )
                    if heartbeat.fenced.is_set():
                        return
                    self._after_pipeline(
                        heartbeat.fence,
                        item_id=str(step["itemId"]),
                        page_id=step.get("pageId"),
                        status="failed",
                    )
                    self.repository.fail_step(
                        heartbeat.fence,
                        step_id=last_step_id,
                        code="STEP_FAILED",
                        message=str(exc),
                    )
                else:
                    if checkpoint.get("__control_drained__"):
                        continue
                    LOGGER.info(
                        "步骤完成：job=%s kind=%s step=%s page=%s duration=%.2fs",
                        _short(fence.job_id),
                        step_kind,
                        step_id,
                        page_id,
                        time.monotonic() - step_started_at,
                    )
                    if heartbeat.fenced.is_set():
                        return
                    if not checkpoint.get("__already_published__"):
                        self.repository.complete_step(
                            heartbeat.fence,
                            step_id=last_step_id,
                            checkpoint=checkpoint,
                        )
                    self._after_completed_step(heartbeat.fence, step)
        except AttemptFenced:
            return

    def _run_parallel_attempt(
        self,
        heartbeat: AttemptHeartbeat,
        stop_event: threading.Event,
        config: Mapping[str, Any],
    ) -> None:
        """Run one serial worker per step kind so different stages can overlap."""

        pool_kinds = self.repository.step_kinds(heartbeat.fence)
        if not pool_kinds:
            self.repository.fail_job(
                heartbeat.fence,
                code="NO_STEP_HANDLERS",
                message="parallel job has no registered step handlers",
            )
            return
        admission_closed = threading.Event()
        worker_errors: list[BaseException] = []
        error_lock = threading.Lock()
        pipeline_condition = threading.Condition()
        pipeline_state_version = 0
        pipeline_wait_seconds = min(
            max(self.idle_poll_seconds, MIN_SCHEDULER_POLL_SECONDS),
            MAX_SCHEDULER_POLL_SECONDS,
        )
        has_deep_learning_pool = bool(
            set(pool_kinds).intersection(DEEP_LEARNING_STEP_KINDS)
        )
        deep_learning_concurrency = (
            int(config["deepLearningConcurrency"])
            if has_deep_learning_pool
            else 1
        )
        if not 1 <= deep_learning_concurrency <= 4:
            raise ValueError("deepLearningConcurrency must be between 1 and 4")
        deep_learning_admission = threading.Semaphore(
            deep_learning_concurrency
        )
        LOGGER.info(
            "并行流水线启动：job=%s pools=%s deep_learning_concurrency=%s",
            _short(heartbeat.fence.job_id),
            ",".join(pool_kinds),
            deep_learning_concurrency,
        )
        lock_waiting_states = {
            kind: False
            for kind in pool_kinds
            if kind in DEEP_LEARNING_STEP_KINDS
        }
        lock_waiting_state_lock = threading.Lock()

        def pipeline_version() -> int:
            with pipeline_condition:
                return pipeline_state_version

        def signal_pipeline_changed() -> None:
            nonlocal pipeline_state_version
            with pipeline_condition:
                pipeline_state_version += 1
                pipeline_condition.notify_all()

        def close_admission() -> None:
            admission_closed.set()
            signal_pipeline_changed()

        def wait_for_pipeline_change(observed_version: int, timeout: float) -> None:
            with pipeline_condition:
                pipeline_condition.wait_for(
                    lambda: (
                        pipeline_state_version != observed_version
                        or admission_closed.is_set()
                        or stop_event.is_set()
                        or heartbeat.fenced.is_set()
                    ),
                    timeout=timeout,
                )

        def set_lock_waiting(pool_kind: str, waiting: bool) -> None:
            with lock_waiting_state_lock:
                lock_waiting_states[pool_kind] = waiting
                snapshot = dict(lock_waiting_states)
            self.repository.write_pipeline_progress(
                heartbeat.fence,
                lock_waiting=snapshot,
            )
            LOGGER.info(
                "深度学习并发锁状态：job=%s pool=%s waiting=%s",
                _short(heartbeat.fence.job_id),
                pool_kind,
                waiting,
            )

        def run_pool(pool_kind: str) -> None:
            busy_failures = 0
            while (
                not stop_event.is_set()
                and not heartbeat.fenced.is_set()
                and not admission_closed.is_set()
            ):
                observed_version = pipeline_version()
                admission_phase = True
                try:
                    status = self.repository.control_status(heartbeat.fence)
                    if status in {"pausing", "cancelling"}:
                        close_admission()
                        return
                    step_ordinal = self.repository.ready_step_ordinal(
                        heartbeat.fence,
                        step_kind=pool_kind,
                    )
                    steps = (
                        self.repository.next_step_batch(
                            heartbeat.fence,
                            step_kind=pool_kind,
                            limit=self._batch_size(
                                pool_kind,
                                config,
                                step_ordinal=step_ordinal,
                            ),
                        )
                        if (
                            pool_kind in self.batch_handlers
                            and step_ordinal is not None
                        )
                        else []
                    )
                    step = (
                        None
                        if pool_kind in self.batch_handlers
                        else (
                            self.repository.next_step(
                                heartbeat.fence,
                                allowed_kinds=(pool_kind,),
                            )
                            if step_ordinal is not None
                            else None
                        )
                    )
                    if not steps and step is None:
                        pool_pending, pool_running = (
                            self.repository.active_step_counts(
                                heartbeat.fence,
                                step_kind=pool_kind,
                            )
                        )
                        busy_failures = 0
                        if pool_pending == 0 and pool_running == 0:
                            pending, running = self.repository.active_step_counts(
                                heartbeat.fence
                            )
                            LOGGER.debug(
                                "并行阶段池已完成并退出：job=%s pool=%s",
                                _short(heartbeat.fence.job_id),
                                pool_kind,
                            )
                            if pending == 0 and running == 0:
                                close_admission()
                            return
                        wait_for_pipeline_change(
                            observed_version,
                            pipeline_wait_seconds,
                        )
                        continue
                    busy_failures = 0
                    admission_phase = False
                    if steps:
                        self._execute_batch(
                            heartbeat,
                            self.batch_handlers[pool_kind],
                            steps,
                        )
                        signal_pipeline_changed()
                        continue
                    assert step is not None
                    handler = self.handlers.get(str(step["stepKind"]))
                    if handler is None:
                        LOGGER.error(
                            "并行步骤无处理器：job=%s kind=%s step=%s",
                            _short(heartbeat.fence.job_id),
                            step["stepKind"],
                            _short(step["stepId"]),
                        )
                        self.repository.fail_step(
                            heartbeat.fence,
                            step_id=str(step["stepId"]),
                            code="UNSUPPORTED_STEP_KIND",
                            message=f"no Worker handler for {step['stepKind']}",
                        )
                        signal_pipeline_changed()
                        continue
                    step_kind, step_id, page_id = _step_log_fields(step)
                    step_started_at = time.monotonic()
                    LOGGER.info(
                        "步骤开始：job=%s kind=%s step=%s page=%s pool=%s",
                        _short(heartbeat.fence.job_id),
                        step_kind,
                        step_id,
                        page_id,
                        pool_kind,
                    )
                    if not self._before_pipeline(heartbeat.fence, step):
                        signal_pipeline_changed()
                        continue
                    try:
                        def execute_step() -> Mapping[str, Any]:
                            return handler(
                                heartbeat.fence,
                                step,
                            )

                        if pool_kind in DEEP_LEARNING_STEP_KINDS:
                            acquired = deep_learning_admission.acquire(blocking=False)
                            if not acquired:
                                set_lock_waiting(pool_kind, True)
                                deep_learning_admission.acquire()
                            try:
                                if not acquired:
                                    set_lock_waiting(pool_kind, False)
                                checkpoint = execute_step()
                            finally:
                                deep_learning_admission.release()
                        else:
                            checkpoint = execute_step()
                    except Exception as exc:
                        LOGGER.exception(
                            "步骤失败：job=%s kind=%s step=%s page=%s "
                            "pool=%s duration=%.2fs",
                            _short(heartbeat.fence.job_id),
                            step_kind,
                            step_id,
                            page_id,
                            pool_kind,
                            time.monotonic() - step_started_at,
                        )
                        if heartbeat.fenced.is_set():
                            return
                        self._after_pipeline(
                            heartbeat.fence,
                            item_id=str(step["itemId"]),
                            page_id=step.get("pageId"),
                            status="failed",
                        )
                        self.repository.fail_step(
                            heartbeat.fence,
                            step_id=str(step["stepId"]),
                            code="STEP_FAILED",
                            message=str(exc),
                        )
                        signal_pipeline_changed()
                    else:
                        if checkpoint.get("__control_drained__"):
                            close_admission()
                            return
                        LOGGER.info(
                            "步骤完成：job=%s kind=%s step=%s page=%s "
                            "pool=%s duration=%.2fs",
                            _short(heartbeat.fence.job_id),
                            step_kind,
                            step_id,
                            page_id,
                            pool_kind,
                            time.monotonic() - step_started_at,
                        )
                        if (
                            not heartbeat.fenced.is_set()
                            and not checkpoint.get("__already_published__")
                        ):
                            self.repository.complete_step(
                                heartbeat.fence,
                                step_id=str(step["stepId"]),
                                checkpoint=checkpoint,
                            )
                        if not heartbeat.fenced.is_set():
                            self._after_completed_step(
                                heartbeat.fence,
                                step,
                            )
                        signal_pipeline_changed()
                except AttemptFenced:
                    return
                except BaseException as exc:
                    if (
                        admission_phase
                        and _is_sqlite_busy_error(exc)
                        and busy_failures < PIPELINE_BUSY_RETRY_LIMIT
                        and not admission_closed.is_set()
                        and not stop_event.is_set()
                        and not heartbeat.fenced.is_set()
                    ):
                        busy_failures += 1
                        retry_delay = min(
                            PIPELINE_BUSY_RETRY_BASE_SECONDS
                            * (2 ** (busy_failures - 1)),
                            pipeline_wait_seconds,
                        )
                        LOGGER.warning(
                            "并行阶段领取遇到 SQLite 写锁竞争，将重试："
                            "job=%s pool=%s attempt=%s/%s delay=%.2fs",
                            _short(heartbeat.fence.job_id),
                            pool_kind,
                            busy_failures,
                            PIPELINE_BUSY_RETRY_LIMIT,
                            retry_delay,
                        )
                        wait_for_pipeline_change(observed_version, retry_delay)
                        continue
                    LOGGER.exception(
                        "并行流水线线程失败：job=%s pool=%s",
                        _short(heartbeat.fence.job_id),
                        pool_kind,
                    )
                    with error_lock:
                        worker_errors.append(exc)
                    close_admission()
                    return

        with ThreadPoolExecutor(
            max_workers=len(pool_kinds),
            thread_name_prefix="job-pipeline",
        ) as executor:
            futures = [
                executor.submit(run_pool, pool_kind)
                for pool_kind in pool_kinds
            ]
            while (
                not admission_closed.is_set()
                and not stop_event.is_set()
                and not heartbeat.fenced.is_set()
            ):
                observed_version = pipeline_version()
                if all(future.done() for future in futures):
                    break
                if self.safe_point is not None:
                    _pending, running = self.repository.active_step_counts(
                        heartbeat.fence
                    )
                    if running == 0:
                        if self.safe_point():
                            signal_pipeline_changed()
                            continue
                wait_for_pipeline_change(
                    observed_version,
                    pipeline_wait_seconds,
                )
            close_admission()
            for future in futures:
                future.result()

        if heartbeat.fenced.is_set() or stop_event.is_set():
            return
        if worker_errors:
            LOGGER.error(
                "并行流水线失败：job=%s error=%s",
                _short(heartbeat.fence.job_id),
                worker_errors[0],
            )
            self.repository.fail_job(
                heartbeat.fence,
                code="PIPELINE_POOL_FAILED",
                message=str(worker_errors[0]),
            )
            return
        status = self.repository.control_status(heartbeat.fence)
        if status in {"pausing", "cancelling"}:
            LOGGER.info(
                "并行任务进入安全排空：job=%s status=%s",
                _short(heartbeat.fence.job_id),
                status,
            )
            expected = {(kind, 0) for kind in pool_kinds}
            for kind, slot in expected:
                self.repository.acknowledge_drain(
                    heartbeat.fence,
                    pool_id=kind,
                    worker_slot=slot,
                    last_step_id=None,
                )
            self.repository.finalize_drain(
                heartbeat.fence,
                expected_slots=expected,
            )
            return
        final_status = self._finish_job(heartbeat.fence)
        LOGGER.info(
            "并行任务全部步骤处理结束：job=%s status=%s",
            _short(heartbeat.fence.job_id),
            final_status or "running",
        )

    def _before_pipeline(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> bool:
        page_id = step.get("pageId")
        if (
            self.plugin_runtime is None
            or not bool(step.get("isFirstStep"))
            or not isinstance(page_id, str)
        ):
            return True
        try:
            self.plugin_runtime.before_pipeline(
                fence,
                item_id=str(step["itemId"]),
                page_id=page_id,
                data={"pageId": page_id},
            )
        except Exception as exc:
            LOGGER.exception(
                "页面流水线 before 插件失败：job=%s item=%s page=%s",
                _short(fence.job_id),
                _short(step["itemId"]),
                _short(page_id),
            )
            self.repository.fail_step(
                fence,
                step_id=str(step["stepId"]),
                code="PLUGIN_PIPELINE_FAILED",
                message=str(exc),
            )
            return False
        return True

    def _after_pipeline(
        self,
        fence: AttemptFence,
        *,
        item_id: str,
        page_id: object,
        status: str,
    ) -> None:
        if self.plugin_runtime is None or not isinstance(page_id, str):
            return
        try:
            self.plugin_runtime.after_pipeline(
                fence,
                item_id=item_id,
                page_id=page_id,
                data={"pageId": page_id, "status": status},
            )
        except Exception as exc:
            LOGGER.exception(
                "页面流水线 after 插件失败：job=%s item=%s page=%s",
                _short(fence.job_id),
                _short(item_id),
                _short(page_id),
            )
            if status in {"completed", "skipped", "cancelled"}:
                self.repository.fail_terminal_item(
                    fence,
                    item_id=item_id,
                    code="PLUGIN_PIPELINE_FAILED",
                    message=str(exc),
                )

    def _after_completed_step(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> None:
        if not bool(step.get("isLastStep")):
            return
        item_id = str(step["itemId"])
        status = self.repository.item_statuses(fence, (item_id,)).get(item_id)
        if status in {"completed", "failed", "skipped", "cancelled"}:
            self._after_pipeline(
                fence,
                item_id=item_id,
                page_id=step.get("pageId"),
                status=status,
            )

    def _finish_job(self, fence: AttemptFence) -> str | None:
        if self.plugin_runtime is not None:
            for item in self.repository.terminal_page_items(fence):
                self._after_pipeline(
                    fence,
                    item_id=item["itemId"],
                    page_id=item["pageId"],
                    status=item["status"],
                )
            status = self.repository.completion_status(fence)
            if status is None:
                return None
            self.plugin_runtime.after_job(fence, {"status": status})
        return self.repository.finish_if_complete(fence)

    def _execute_batch(
        self,
        heartbeat: AttemptHeartbeat,
        handler: BatchStepHandler,
        steps: Sequence[Mapping[str, Any]],
    ) -> None:
        active_steps = [
            step
            for step in steps
            if self._before_pipeline(heartbeat.fence, step)
        ]
        if not active_steps:
            return
        step_kind = str(active_steps[0].get("stepKind", "unknown"))
        step_ids = ",".join(
            _short(step.get("stepId", "-")) for step in active_steps
        )
        started_at = time.monotonic()
        LOGGER.info(
            "批处理开始：job=%s kind=%s count=%s steps=%s",
            _short(heartbeat.fence.job_id),
            step_kind,
            len(active_steps),
            step_ids,
        )
        try:
            checkpoint = handler(heartbeat.fence, active_steps)
        except Exception as exc:
            LOGGER.exception(
                "批处理失败：job=%s kind=%s count=%s duration=%.2fs",
                _short(heartbeat.fence.job_id),
                step_kind,
                len(active_steps),
                time.monotonic() - started_at,
            )
            if heartbeat.fenced.is_set():
                return
            statuses = self.repository.item_statuses(
                heartbeat.fence,
                tuple(str(step["itemId"]) for step in active_steps),
            )
            for step in active_steps:
                item_id = str(step["itemId"])
                status = statuses.get(item_id, "running")
                if status in {"completed", "failed", "skipped", "cancelled"}:
                    self._after_pipeline(
                        heartbeat.fence,
                        item_id=item_id,
                        page_id=step.get("pageId"),
                        status=status,
                    )
                    continue
                self._after_pipeline(
                    heartbeat.fence,
                    item_id=item_id,
                    page_id=step.get("pageId"),
                    status="failed",
                )
                try:
                    self.repository.fail_step(
                        heartbeat.fence,
                        step_id=str(step["stepId"]),
                        code="BATCH_STEP_FAILED",
                        message=str(exc),
                    )
                except AttemptFenced:
                    continue
            return
        LOGGER.info(
            "批处理完成：job=%s kind=%s count=%s duration=%.2fs",
            _short(heartbeat.fence.job_id),
            step_kind,
            len(active_steps),
            time.monotonic() - started_at,
        )
        if not checkpoint.get("__already_published__"):
            per_step = checkpoint.get("steps")
            for step in active_steps:
                value = (
                    per_step.get(str(step["stepId"]), {})
                    if isinstance(per_step, Mapping)
                    else checkpoint
                )
                self.repository.complete_step(
                    heartbeat.fence,
                    step_id=str(step["stepId"]),
                    checkpoint=value if isinstance(value, Mapping) else {},
                )
        statuses = self.repository.item_statuses(
            heartbeat.fence,
            tuple(str(step["itemId"]) for step in active_steps),
        )
        for step in active_steps:
            item_id = str(step["itemId"])
            status = statuses.get(item_id)
            if status in {"completed", "failed", "skipped", "cancelled"}:
                self._after_pipeline(
                    heartbeat.fence,
                    item_id=item_id,
                    page_id=step.get("pageId"),
                    status=status,
                )

    @staticmethod
    def _batch_size(
        step_kind: str,
        config: Mapping[str, Any],
        *,
        step_ordinal: int | None,
    ) -> int:
        if step_kind == "hq_translate":
            section = config["translation"]
            if not isinstance(section, Mapping):
                raise ValueError("translation config must be an object")
            value = section["batchSize"]
        elif step_kind == "proofread":
            rounds = config["proofreadingRounds"]
            if not isinstance(rounds, list):
                raise ValueError("proofreadingRounds must be an array")
            index = int(step_ordinal or 1) - 1
            if index < 0 or index >= len(rounds):
                raise ValueError("proofreading round does not match the step ordinal")
            section = rounds[index]
            if not isinstance(section, Mapping):
                raise ValueError("proofreading round config must be an object")
            value = section["batchSize"]
        elif step_kind == "web_extract_page":
            section = config["options"]
            if not isinstance(section, Mapping):
                raise ValueError("web import options must be an object")
            value = section["concurrency"]
        else:
            value = 1
        parsed = int(value)
        maximum = 32 if step_kind == "web_extract_page" else 10
        if not 1 <= parsed <= maximum:
            raise ValueError(f"batch size must be between 1 and {maximum}")
        return parsed
