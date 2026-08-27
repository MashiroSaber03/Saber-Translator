"""Worker scheduler that reconstructs all durable work from SQLite."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import logging
import threading
import time
from typing import Any

from src.backend_v2.auth.ownership import owner_scope
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobQueueRepository,
)
from src.backend_v2.storage.database import is_sqlite_busy_error
from src.backend_v2.timestamps import utcnow
from src.shared.user_logging import (
    inline_log_text,
    log_step_failed,
    log_step_finished,
    log_step_started,
    log_task_failed,
    log_task_finished,
    log_task_started,
    user_log,
    user_log_context,
)


StepHandler = Callable[[AttemptFence, Mapping[str, Any]], Mapping[str, Any]]
BatchStepHandler = Callable[
    [AttemptFence, Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]
ControlTimeoutHandler = Callable[[AttemptFence, str], None]
DEEP_LEARNING_STEP_KINDS = frozenset({"detect", "ocr", "color", "repair"})
PIPELINE_BUSY_RETRY_LIMIT = 3
PIPELINE_BUSY_RETRY_BASE_SECONDS = 0.05
PARALLEL_PIPELINE_LEAD_WINDOW = 50
MAX_DEEP_LEARNING_THREADS = 8
MIN_SCHEDULER_POLL_SECONDS = 0.1
MAX_SCHEDULER_POLL_SECONDS = 0.5
DEFAULT_CONTROL_TIMEOUT_SECONDS = 1.5
DEFAULT_CONTROL_POLL_SECONDS = 0.25
LOGGER = logging.getLogger("saber.worker.jobs")


def _short(value: object) -> str:
    return str(value)[:8]


def _task_duration(fence: AttemptFence) -> float:
    return max(0.0, (utcnow() - fence.started_at).total_seconds())


def _step_context_values(
    fence: AttemptFence,
    step: Mapping[str, Any],
) -> dict[str, Any]:
    raw_page_number = step.get("itemOrdinal")
    page_number = (
        int(raw_page_number)
        if isinstance(raw_page_number, int) and not isinstance(raw_page_number, bool)
        else None
    )
    raw_step_ordinal = step.get("stepOrdinal")
    step_ordinal = (
        int(raw_step_ordinal)
        if isinstance(raw_step_ordinal, int) and not isinstance(raw_step_ordinal, bool)
        else None
    )
    page_id = step.get("pageId")
    return {
        "job_id": fence.job_id,
        "page_number": page_number if isinstance(page_id, str) else None,
        "step_kind": str(step.get("stepKind") or ""),
        "step_ordinal": step_ordinal,
    }


def _checkpoint_log_status(checkpoint: Mapping[str, Any]) -> str:
    if checkpoint.get("failed") is True:
        return "failed"
    skipped = checkpoint.get("skipped")
    if skipped is True or isinstance(skipped, str):
        return "skipped"
    return "completed"


class JobWorkerLoop:
    """Single compute-slot scheduler with persisted step checkpoints."""

    def __init__(
        self,
        repository: JobQueueRepository,
        *,
        worker_epoch_id: str,
        handlers: Mapping[str, StepHandler],
        batch_handlers: Mapping[str, BatchStepHandler] | None = None,
        handler_resolver: Callable[[str], StepHandler | None] | None = None,
        plugin_runtime: Any | None = None,
        safe_point: Callable[[], bool] | None = None,
        idle_work: Callable[[], bool] | None = None,
        scheduling_policy: Callable[[], Mapping[str, Any]] | None = None,
        admission_check: Callable[[], bool] | None = None,
        on_activity: Callable[[], None] | None = None,
        on_control_timeout: ControlTimeoutHandler | None = None,
        control_timeout_seconds: float = DEFAULT_CONTROL_TIMEOUT_SECONDS,
        control_poll_seconds: float = DEFAULT_CONTROL_POLL_SECONDS,
        idle_poll_seconds: float = 0.25,
    ) -> None:
        self.repository = repository
        self.worker_epoch_id = worker_epoch_id
        self.handlers = dict(handlers)
        self.batch_handlers = dict(batch_handlers or {})
        self.handler_resolver = handler_resolver
        if not set(self.batch_handlers).issubset(self.handlers):
            raise ValueError("every batch handler requires a matching step handler")
        self.plugin_runtime = plugin_runtime
        self.safe_point = safe_point
        self.idle_work = idle_work
        self.scheduling_policy = scheduling_policy
        self.admission_check = admission_check
        self.on_activity = on_activity
        self.on_control_timeout = on_control_timeout
        if control_timeout_seconds <= 0:
            raise ValueError("control timeout must be positive")
        if control_poll_seconds <= 0:
            raise ValueError("control poll interval must be positive")
        self.control_timeout_seconds = control_timeout_seconds
        self.control_poll_seconds = control_poll_seconds
        if idle_poll_seconds <= 0:
            raise ValueError("idle poll interval must be positive")
        self.idle_poll_seconds = idle_poll_seconds

    def run(self, stop_event: threading.Event) -> None:
        LOGGER.debug("持久任务调度器开始运行")
        while not stop_event.is_set():
            try:
                policy = self._policy()
                admitted = (
                    self.admission_check is None or self.admission_check()
                )
            except Exception as exc:
                if not is_sqlite_busy_error(exc):
                    raise
                LOGGER.debug(
                    "持久任务调度器读取调度条件时遇到 SQLite 写锁竞争"
                )
                stop_event.wait(self.idle_poll_seconds)
                continue
            if not admitted:
                if self.safe_point is not None and self.safe_point():
                    continue
                stop_event.wait(self.idle_poll_seconds)
                continue
            try:
                claim_options = (
                    {
                        "queue_discipline": str(policy["queueDiscipline"]),
                    }
                    if policy is not None
                    else {}
                )
                fence = self.repository.claim_next(
                    worker_epoch_id=self.worker_epoch_id,
                    **claim_options,
                )
            except AttemptFenced:
                LOGGER.error("Worker epoch 已失效，停止领取任务")
                stop_event.set()
                return
            except Exception as exc:
                if not is_sqlite_busy_error(exc):
                    raise
                LOGGER.debug(
                    "持久任务调度器遇到 SQLite 写锁竞争，将继续轮询"
                )
                stop_event.wait(self.idle_poll_seconds)
                continue
            if fence is None:
                if self.safe_point is not None and self.safe_point():
                    continue
                if self.idle_work is not None and self.idle_work():
                    continue
                stop_event.wait(self.idle_poll_seconds)
                continue
            self._note_activity()
            with owner_scope(fence.owner_user_id):
                self._run_attempt(fence, stop_event)
        LOGGER.debug("持久任务调度器已停止")

    def _run_attempt(
        self,
        fence: AttemptFence,
        stop_event: threading.Event,
    ) -> None:
        started_at = time.monotonic()
        execution_mode = "sequential"
        watchdog_stop = threading.Event()
        watchdog: threading.Thread | None = None
        if self.on_control_timeout is not None:
            watchdog = threading.Thread(
                target=self._watch_attempt_control,
                args=(fence, stop_event, watchdog_stop),
                name=f"job-control-{_short(fence.job_id)}",
                daemon=True,
            )
            watchdog.start()
        try:
            config = self.repository.attempt_config(fence)
            if self.plugin_runtime is not None:
                config = self.plugin_runtime.before_job(fence, config)
            if not isinstance(config, Mapping):
                raise ValueError("job config must be an object")
            self._resolve_attempt_handlers(fence)
            execution_mode = config.get("executionMode", "sequential")
            if not isinstance(execution_mode, str):
                raise ValueError("executionMode must be a string")
            if execution_mode not in {"sequential", "parallel"}:
                raise ValueError(f"unsupported execution mode: {execution_mode}")
            if fence.first_claim:
                log_task_started(
                    job_id=fence.job_id,
                    kind=fence.kind,
                    execution_mode=execution_mode,
                )
            if execution_mode == "parallel":
                self._run_parallel_attempt(fence, stop_event, config)
            else:
                self._run_sequential_attempt(fence, stop_event)
        except AttemptFenced:
            LOGGER.debug(
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
            try:
                self.repository.fail_job(
                    fence,
                    code="JOB_LIFECYCLE_FAILED",
                    message=str(exc),
                )
            except AttemptFenced:
                pass
            log_task_failed(
                job_id=fence.job_id,
                kind=fence.kind,
                duration=_task_duration(fence),
                error=exc,
            )
        finally:
            watchdog_stop.set()
            if watchdog is not None and watchdog is not threading.current_thread():
                watchdog.join(timeout=max(1.0, self.control_poll_seconds * 2))
            if self.plugin_runtime is not None:
                self.plugin_runtime.release_job_state(fence.job_id)
            self._note_activity()
            LOGGER.debug(
                "任务轮次结束：job=%s attempt=%s duration=%.2fs",
                _short(fence.job_id),
                _short(fence.attempt_id),
                time.monotonic() - started_at,
            )

    def _watch_attempt_control(
        self,
        fence: AttemptFence,
        stop_event: threading.Event,
        watchdog_stop: threading.Event,
    ) -> None:
        controlled_since: float | None = None
        reason = ""
        while not watchdog_stop.wait(self.control_poll_seconds):
            if stop_event.is_set():
                return
            try:
                self.repository.assert_attempt_active(fence)
            except AttemptFenced:
                current_reason = "execution_rights_revoked"
            except Exception as exc:
                if not is_sqlite_busy_error(exc):
                    LOGGER.warning(
                        "任务控制看门狗读取失败，将继续重试：job=%s attempt=%s",
                        _short(fence.job_id),
                        _short(fence.attempt_id),
                        exc_info=exc,
                    )
                continue
            else:
                controlled_since = None
                reason = ""
                continue
            if controlled_since is None or current_reason != reason:
                controlled_since = time.monotonic()
                reason = current_reason
                LOGGER.debug(
                    "任务控制看门狗开始等待处理器退出：job=%s attempt=%s reason=%s",
                    _short(fence.job_id),
                    _short(fence.attempt_id),
                    reason,
                )
                continue
            if time.monotonic() - controlled_since < self.control_timeout_seconds:
                continue
            LOGGER.critical(
                "任务处理器未在控制宽限期内退出：job=%s attempt=%s "
                "reason=%s timeout=%.1fs",
                _short(fence.job_id),
                _short(fence.attempt_id),
                reason,
                self.control_timeout_seconds,
            )
            handler = self.on_control_timeout
            if handler is not None:
                handler(fence, reason)
            return

    def _note_activity(self) -> None:
        if self.on_activity is not None:
            self.on_activity()

    def _policy(self) -> Mapping[str, Any] | None:
        if self.scheduling_policy is None:
            return None
        value = self.scheduling_policy()
        if not isinstance(value, Mapping):
            raise RuntimeError("scheduling policy provider must return an object")
        return value

    def _slice_boundary(
        self,
        fence: AttemptFence,
        *,
        terminal_count: int,
    ) -> tuple[bool, int]:
        """Serve bounded interactive work and decide whether to yield."""

        self.repository.assert_attempt_active(fence)
        if self.admission_check is not None and not self.admission_check():
            self.repository.yield_attempt(fence, reason="memory_pressure")
            return True, terminal_count
        policy = self._policy()
        if policy is not None and self.safe_point is not None:
            for _index in range(int(policy["interactiveBurst"])):
                if not self.safe_point():
                    break
        if (
            policy is not None
            and policy["queueDiscipline"] == "owner_round_robin"
            and self.repository.has_ready_queued_competitor(
                owner_user_id=fence.owner_user_id
            )
        ):
            self.repository.yield_attempt(fence, reason="fairness")
            return True, terminal_count
        quantum = (
            int(policy["pageQuantum"])
            if policy is not None
            else PARALLEL_PIPELINE_LEAD_WINDOW
        )
        return False, terminal_count + quantum

    def _resolve_attempt_handlers(self, fence: AttemptFence) -> None:
        if self.handler_resolver is None:
            return
        for step_kind in self.repository.step_kinds(fence):
            if step_kind in self.handlers:
                continue
            handler = self.handler_resolver(step_kind)
            if handler is not None:
                self.handlers[step_kind] = handler

    def _run_sequential_attempt(
        self,
        fence: AttemptFence,
        stop_event: threading.Event,
    ) -> None:
        last_step_id: str | None = None
        policy = self._policy()
        slice_target = (
            self.repository.terminal_item_count(fence) + int(policy["pageQuantum"])
            if policy is not None
            else None
        )

        def finish_slice_if_needed() -> bool:
            nonlocal slice_target
            if slice_target is None:
                return False
            terminal_count = self.repository.terminal_item_count(fence)
            if terminal_count < slice_target:
                return False
            pending, running = self.repository.active_step_counts(fence)
            if pending == 0 and running == 0:
                return False
            should_stop, slice_target = self._slice_boundary(
                fence,
                terminal_count=terminal_count,
            )
            return should_stop

        try:
            while not stop_event.is_set():
                self.repository.assert_attempt_active(fence)
                handled_batch = False
                config = self.repository.attempt_config(fence)
                ordinal_limit = (
                    {"max_item_ordinal": slice_target}
                    if slice_target is not None
                    else {}
                )
                for batch_kind, batch_handler in self.batch_handlers.items():
                    step_ordinal = self.repository.ready_step_ordinal(
                        fence,
                        step_kind=batch_kind,
                        **ordinal_limit,
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
                        **ordinal_limit,
                    )
                    if not batch:
                        continue
                    last_step_id = str(batch[-1]["stepId"])
                    self._execute_batch(
                        fence,
                        batch_handler,
                        batch,
                    )
                    handled_batch = True
                    break
                if handled_batch:
                    if finish_slice_if_needed():
                        return
                    continue
                ordinary_kinds = tuple(
                    kind
                    for kind in self.handlers
                    if kind not in self.batch_handlers
                )
                step = self.repository.next_step(
                    fence,
                    allowed_kinds=ordinary_kinds,
                    **ordinal_limit,
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
                                error = RuntimeError(
                                    f"没有以下步骤的处理器：{kinds}"
                                )
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
                                log_task_failed(
                                    job_id=fence.job_id,
                                    kind=fence.kind,
                                    duration=_task_duration(fence),
                                    error=error,
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
                    log_task_finished(
                        job_id=fence.job_id,
                        kind=fence.kind,
                        duration=_task_duration(fence),
                        status=final_status,
                    )
                    return
                last_step_id = str(step["stepId"])
                handler = self.handlers.get(str(step["stepKind"]))
                if handler is None:
                    error = RuntimeError(
                        f"没有步骤处理器：{step['stepKind']}"
                    )
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
                    log_task_failed(
                        job_id=fence.job_id,
                        kind=fence.kind,
                        duration=_task_duration(fence),
                        error=error,
                    )
                    return
                step_started_at = time.monotonic()
                with user_log_context(**_step_context_values(fence, step)):
                    log_step_started()
                    if not self._before_pipeline(fence, step):
                        if finish_slice_if_needed():
                            return
                        continue
                    try:
                        with owner_scope(fence.owner_user_id):
                            checkpoint = handler(
                                fence,
                                step,
                            )
                        if not isinstance(checkpoint, Mapping):
                            raise TypeError("step handler must return an object")
                    except AttemptFenced:
                        raise
                    except Exception as exc:
                        duration = time.monotonic() - step_started_at
                        log_step_failed(exc, duration=duration)
                        self._after_pipeline(
                            fence,
                            item_id=str(step["itemId"]),
                            page_id=step.get("pageId"),
                            status="failed",
                        )
                        self.repository.fail_step(
                            fence,
                            step_id=last_step_id,
                            code="STEP_FAILED",
                            message=str(exc),
                        )
                    else:
                        log_step_finished(
                            duration=time.monotonic() - step_started_at,
                            status=_checkpoint_log_status(checkpoint),
                        )
                        if not checkpoint.get("__already_published__"):
                            self.repository.complete_step(
                                fence,
                                step_id=last_step_id,
                                checkpoint=checkpoint,
                            )
                        self._after_completed_step(fence, step)
                if finish_slice_if_needed():
                    return

        except AttemptFenced:
            raise

    def _run_parallel_attempt(
        self,
        fence: AttemptFence,
        stop_event: threading.Event,
        config: Mapping[str, Any],
    ) -> None:
        """Admit SQLite work centrally while stage handlers overlap."""

        pool_kinds = self.repository.step_kinds(fence)
        if not pool_kinds:
            error = RuntimeError("并行任务没有可执行的步骤")
            LOGGER.error("并行任务没有可执行步骤：job=%s", _short(fence.job_id))
            self.repository.fail_job(
                fence,
                code="NO_STEP_HANDLERS",
                message="parallel job has no registered step handlers",
            )
            log_task_failed(
                job_id=fence.job_id,
                kind=fence.kind,
                duration=_task_duration(fence),
                error=error,
            )
            return
        admission_closed = threading.Event()
        worker_errors: list[Exception] = []
        pipeline_wait_seconds = min(
            max(self.idle_poll_seconds, MIN_SCHEDULER_POLL_SECONDS),
            MAX_SCHEDULER_POLL_SECONDS,
        )
        has_deep_learning_pool = bool(
            set(pool_kinds).intersection(DEEP_LEARNING_STEP_KINDS)
        )
        has_local_pipeline_window = bool(
            {"save", "publish_clean"}.intersection(pool_kinds)
        )
        policy = self._policy()
        slice_target = (
            self.repository.terminal_item_count(fence)
            + (
                int(policy["pageQuantum"])
                if policy is not None
                else PARALLEL_PIPELINE_LEAD_WINDOW
            )
            if policy is not None or has_local_pipeline_window
            else None
        )
        attempt_released = False
        deep_learning_concurrency = 1
        if has_deep_learning_pool:
            value = config.get("deepLearningConcurrency")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("deepLearningConcurrency must be an integer")
            deep_learning_concurrency = value
        if deep_learning_concurrency < 1:
            raise ValueError("deepLearningConcurrency must be a positive integer")

        def current_deep_learning_limit() -> int:
            current_policy = self._policy()
            if current_policy is None:
                return deep_learning_concurrency
            return min(
                deep_learning_concurrency,
                int(current_policy["maxDeepLearningConcurrency"]),
            )

        LOGGER.debug(
            "并行流水线参数：job=%s pools=%s deep_learning_concurrency=%s "
            "lead_window=%s",
            _short(fence.job_id),
            ",".join(pool_kinds),
            current_deep_learning_limit(),
            (
                int(policy["pageQuantum"])
                if policy is not None
                else PARALLEL_PIPELINE_LEAD_WINDOW
                if has_local_pipeline_window
                else "disabled"
            ),
        )
        lock_waiting_states = {
            kind: False
            for kind in pool_kinds
            if kind in DEEP_LEARNING_STEP_KINDS
        }
        lock_waiting_state_lock = threading.Lock()
        model_gate = threading.Condition()
        active_model_calls = 0

        def set_lock_waiting(pool_kind: str, waiting: bool) -> None:
            with lock_waiting_state_lock:
                lock_waiting_states[pool_kind] = waiting
                snapshot = dict(lock_waiting_states)
            busy_failures = 0
            while True:
                try:
                    self.repository.write_pipeline_progress(
                        fence,
                        lock_waiting=snapshot,
                    )
                    break
                except AttemptFenced:
                    raise
                except Exception as exc:
                    if not is_sqlite_busy_error(exc):
                        raise
                    if busy_failures >= PIPELINE_BUSY_RETRY_LIMIT:
                        LOGGER.debug(
                            "深度学习并发锁遥测持续遇到 SQLite 写锁竞争，"
                            "跳过本次遥测：job=%s pool=%s waiting=%s",
                            _short(fence.job_id),
                            pool_kind,
                            waiting,
                        )
                        break
                    busy_failures += 1
                    delay = min(
                        PIPELINE_BUSY_RETRY_BASE_SECONDS
                        * (2 ** (busy_failures - 1)),
                        pipeline_wait_seconds,
                    )
                    LOGGER.debug(
                        "深度学习并发锁遥测遇到 SQLite 写锁竞争，将有限重试："
                        "job=%s pool=%s attempt=%s/%s",
                        _short(fence.job_id),
                        pool_kind,
                        busy_failures,
                        PIPELINE_BUSY_RETRY_LIMIT,
                    )
                    if stop_event.wait(delay):
                        break
            LOGGER.debug(
                "深度学习并发锁状态：job=%s pool=%s waiting=%s",
                _short(fence.job_id),
                pool_kind,
                waiting,
            )

        def acquire_model_slot(pool_kind: str) -> None:
            nonlocal active_model_calls
            waiting_recorded = False
            while True:
                if admission_closed.is_set():
                    raise AttemptFenced("parallel attempt admission closed")
                with model_gate:
                    if active_model_calls < current_deep_learning_limit():
                        active_model_calls += 1
                        break
                    model_gate.wait(timeout=pipeline_wait_seconds)
                if not waiting_recorded:
                    set_lock_waiting(pool_kind, True)
                    waiting_recorded = True
            if waiting_recorded:
                set_lock_waiting(pool_kind, False)

        def release_model_slot() -> None:
            nonlocal active_model_calls
            with model_gate:
                active_model_calls -= 1
                model_gate.notify_all()

        def execute_step(
            pool_kind: str,
            step: Mapping[str, Any],
        ) -> None:
            step_started_at = time.monotonic()
            with user_log_context(**_step_context_values(fence, step)):
                log_step_started()
                handler = self.handlers.get(str(step["stepKind"]))
                if handler is None:
                    error = RuntimeError(
                        f"没有步骤处理器：{step['stepKind']}"
                    )
                    LOGGER.error(
                        "并行步骤无处理器：job=%s kind=%s step=%s",
                        _short(fence.job_id),
                        step["stepKind"],
                        _short(step["stepId"]),
                    )
                    self.repository.fail_step(
                        fence,
                        step_id=str(step["stepId"]),
                        code="UNSUPPORTED_STEP_KIND",
                        message=f"no Worker handler for {step['stepKind']}",
                    )
                    log_step_failed(
                        error,
                        duration=time.monotonic() - step_started_at,
                    )
                    return
                if not self._before_pipeline(fence, step):
                    return
                try:
                    with owner_scope(fence.owner_user_id):
                        checkpoint = handler(fence, step)
                    if not isinstance(checkpoint, Mapping):
                        raise TypeError("step handler must return an object")
                except AttemptFenced:
                    raise
                except Exception as exc:
                    log_step_failed(
                        exc,
                        duration=time.monotonic() - step_started_at,
                    )
                    self._after_pipeline(
                        fence,
                        item_id=str(step["itemId"]),
                        page_id=step.get("pageId"),
                        status="failed",
                    )
                    self.repository.fail_step(
                        fence,
                        step_id=str(step["stepId"]),
                        code="STEP_FAILED",
                        message=str(exc),
                    )
                    return

                log_step_finished(
                    duration=time.monotonic() - step_started_at,
                    status=_checkpoint_log_status(checkpoint),
                )
                if not checkpoint.get("__already_published__"):
                    self.repository.complete_step(
                        fence,
                        step_id=str(step["stepId"]),
                        checkpoint=checkpoint,
                    )
                self._after_completed_step(fence, step)

        def execute_claimed(
            pool_kind: str,
            steps: Sequence[Mapping[str, Any]],
            step: Mapping[str, Any] | None,
            model_waiting: bool = False,
        ) -> None:
            if steps:
                self._execute_batch(
                    fence,
                    self.batch_handlers[pool_kind],
                    steps,
                )
            elif step is not None:
                if pool_kind in DEEP_LEARNING_STEP_KINDS:
                    acquire_model_slot(pool_kind)
                    try:
                        if model_waiting:
                            set_lock_waiting(pool_kind, False)
                        execute_step(pool_kind, step)
                    finally:
                        release_model_slot()
                else:
                    execute_step(pool_kind, step)

        def claim_with_retry(
            pool_kind: str,
            claim: Callable[
                [],
                tuple[list[Mapping[str, Any]], Mapping[str, Any] | None],
            ],
        ) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None]:
            busy_failures = 0
            while (
                not stop_event.is_set()
                and not admission_closed.is_set()
            ):
                try:
                    return claim()
                except AttemptFenced:
                    raise
                except Exception as exc:
                    if is_sqlite_busy_error(exc):
                        if busy_failures >= PIPELINE_BUSY_RETRY_LIMIT:
                            LOGGER.debug(
                                "并行阶段领取持续遇到 SQLite 写锁竞争，本轮延后："
                                "job=%s pool=%s retries=%s",
                                _short(fence.job_id),
                                pool_kind,
                                busy_failures,
                            )
                            return [], None
                        busy_failures += 1
                        retry_delay = min(
                            PIPELINE_BUSY_RETRY_BASE_SECONDS
                            * (2 ** (busy_failures - 1)),
                            pipeline_wait_seconds,
                        )
                        LOGGER.debug(
                            "并行阶段领取遇到 SQLite 写锁竞争，将重试："
                            "job=%s pool=%s attempt=%s/%s delay=%.2fs",
                            _short(fence.job_id),
                            pool_kind,
                            busy_failures,
                            PIPELINE_BUSY_RETRY_LIMIT,
                            retry_delay,
                        )
                        stop_event.wait(retry_delay)
                        continue
                    raise
            return [], None

        def record_worker_error(
            pool_kind: str,
            exc: Exception,
        ) -> None:
            LOGGER.exception(
                "并行流水线执行失败：job=%s pool=%s",
                _short(fence.job_id),
                pool_kind,
                exc_info=exc,
            )
            if worker_errors:
                return
            worker_errors.append(exc)
            admission_closed.set()
            try:
                self.repository.fail_job(
                    fence,
                    code="PIPELINE_POOL_FAILED",
                    message=str(exc),
                )
            except AttemptFenced:
                pass
            except Exception:
                LOGGER.exception(
                    "并行流水线失败状态持久化失败：job=%s",
                    _short(fence.job_id),
                )

        # cuDNN keeps host-side execution plans per calling thread.  A dedicated
        # pool preserves the configured concurrency without migrating model calls
        # across every pipeline thread during long jobs.
        deep_learning_executor = ThreadPoolExecutor(
            max_workers=(
                deep_learning_concurrency
                if self.scheduling_policy is None
                else min(
                    MAX_DEEP_LEARNING_THREADS,
                    deep_learning_concurrency,
                )
            ),
            thread_name_prefix="job-model",
        )
        executor = ThreadPoolExecutor(
            max_workers=len(pool_kinds),
            thread_name_prefix="job-pipeline",
        )
        active_futures: dict[Future[None], str] = {}
        try:
            while not stop_event.is_set():
                finished = [
                    future for future in active_futures if future.done()
                ]
                for future in finished:
                    pool_kind = active_futures.pop(future)
                    try:
                        future.result()
                    except AttemptFenced:
                        admission_closed.set()
                    except Exception as exc:
                        record_worker_error(pool_kind, exc)
                if admission_closed.is_set():
                    break

                try:
                    self.repository.assert_attempt_active(fence)
                    if not active_futures:
                        pending, running = self.repository.active_step_counts(
                            fence
                        )
                        if pending == 0 and running == 0:
                            break
                except AttemptFenced:
                    admission_closed.set()
                    break
                except Exception as exc:
                    record_worker_error("admission", exc)
                    break

                if (
                    not active_futures
                    and slice_target is not None
                    and self.repository.terminal_item_count(fence) >= slice_target
                ):
                    terminal_count = self.repository.terminal_item_count(fence)
                    if self.scheduling_policy is not None:
                        should_stop, slice_target = self._slice_boundary(
                            fence,
                            terminal_count=terminal_count,
                        )
                        if should_stop:
                            attempt_released = True
                            admission_closed.set()
                            break
                    else:
                        slice_target = (
                            terminal_count + PARALLEL_PIPELINE_LEAD_WINDOW
                        )

                try:
                    should_admit = (
                        bool(finished)
                        or not active_futures
                    )
                    if should_admit:
                        max_item_ordinal = slice_target
                        ordinal_limit = (
                            {"max_item_ordinal": max_item_ordinal}
                            if max_item_ordinal is not None
                            else {}
                        )
                        active_pool_kinds = set(active_futures.values())
                        batch_pool_kinds = tuple(
                            kind
                            for kind in pool_kinds
                            if (
                                kind in self.batch_handlers
                                and kind not in active_pool_kinds
                            )
                        )
                        ready_batch_ordinals = (
                            self.repository.ready_step_ordinals(
                                fence,
                                step_kinds=batch_pool_kinds,
                                **ordinal_limit,
                            )
                            if batch_pool_kinds
                            else {}
                        )
                        for pool_kind, step_ordinal in ready_batch_ordinals.items():
                            steps, _step = claim_with_retry(
                                pool_kind,
                                lambda pool_kind=pool_kind, step_ordinal=step_ordinal: (
                                    self.repository.next_step_batch(
                                        fence,
                                        step_kind=pool_kind,
                                        limit=self._batch_size(
                                            pool_kind,
                                            config,
                                            step_ordinal=step_ordinal,
                                        ),
                                        **ordinal_limit,
                                    ),
                                    None,
                                ),
                            )
                            if admission_closed.is_set():
                                break
                            if not steps:
                                continue
                            future = executor.submit(
                                execute_claimed,
                                pool_kind,
                                tuple(steps),
                                None,
                            )
                            active_futures[future] = pool_kind
                            active_pool_kinds.add(pool_kind)

                        ordinary_pool_kinds = [
                            kind
                            for kind in pool_kinds
                            if (
                                kind not in self.batch_handlers
                                and kind not in active_pool_kinds
                            )
                        ]
                        while ordinary_pool_kinds and not admission_closed.is_set():
                            allowed_kinds = tuple(ordinary_pool_kinds)
                            _steps, step = claim_with_retry(
                                ",".join(allowed_kinds),
                                lambda allowed_kinds=allowed_kinds: (
                                    [],
                                    self.repository.next_step(
                                        fence,
                                        allowed_kinds=allowed_kinds,
                                        **ordinal_limit,
                                    ),
                                ),
                            )
                            if step is None:
                                break
                            pool_kind = str(step["stepKind"])
                            if pool_kind not in ordinary_pool_kinds:
                                raise RuntimeError(
                                    "claimed step does not belong to an idle pool"
                                )
                            target_executor = executor
                            model_waiting = False
                            if pool_kind in DEEP_LEARNING_STEP_KINDS:
                                active_model_steps = sum(
                                    1
                                    for active_kind in active_futures.values()
                                    if active_kind in DEEP_LEARNING_STEP_KINDS
                                )
                                model_waiting = (
                                    active_model_steps
                                    >= current_deep_learning_limit()
                                )
                                if model_waiting:
                                    set_lock_waiting(pool_kind, True)
                                target_executor = deep_learning_executor
                            future = target_executor.submit(
                                execute_claimed,
                                pool_kind,
                                (),
                                step,
                                model_waiting,
                            )
                            active_futures[future] = pool_kind
                            ordinary_pool_kinds.remove(pool_kind)
                except AttemptFenced:
                    admission_closed.set()
                    break
                except Exception as exc:
                    record_worker_error("admission", exc)
                    break

                if admission_closed.is_set():
                    break
                if active_futures:
                    wait(
                        tuple(active_futures),
                        timeout=pipeline_wait_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    continue
                stop_event.wait(pipeline_wait_seconds)

        finally:
            admission_closed.set()
            with model_gate:
                model_gate.notify_all()
            for future in active_futures:
                future.cancel()
            completed: set[Future[None]] = set()
            unfinished: set[Future[None]] = set()
            if active_futures:
                completed, unfinished = wait(
                    tuple(active_futures),
                    timeout=self.control_timeout_seconds,
                )
            for future in completed:
                pool_kind = active_futures[future]
                try:
                    future.result()
                except AttemptFenced:
                    pass
                except Exception as exc:
                    record_worker_error(pool_kind, exc)
            executor.shutdown(wait=False, cancel_futures=True)
            deep_learning_executor.shutdown(wait=False, cancel_futures=True)
            if unfinished:
                LOGGER.critical(
                    "并行流水线线程未在回收宽限期内退出：job=%s count=%s",
                    _short(fence.job_id),
                    len(unfinished),
                )
                stop_event.set()
                handler = self.on_control_timeout
                if handler is not None:
                    handler(fence, "pipeline_abort_timeout")

        if stop_event.is_set():
            return
        if attempt_released:
            return
        if worker_errors:
            LOGGER.error(
                "并行流水线失败：job=%s error=%s",
                _short(fence.job_id),
                worker_errors[0],
            )
            try:
                self.repository.fail_job(
                    fence,
                    code="PIPELINE_POOL_FAILED",
                    message=str(worker_errors[0]),
                )
            except AttemptFenced:
                pass
            log_task_failed(
                job_id=fence.job_id,
                kind=fence.kind,
                duration=_task_duration(fence),
                error=worker_errors[0],
            )
            return
        self.repository.assert_attempt_active(fence)
        final_status = self._finish_job(fence)
        log_task_finished(
            job_id=fence.job_id,
            kind=fence.kind,
            duration=_task_duration(fence),
            status=final_status,
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
        started_at = time.monotonic()
        try:
            self.plugin_runtime.before_pipeline(
                fence,
                item_id=str(step["itemId"]),
                page_id=page_id,
                data={"pageId": page_id},
            )
        except AttemptFenced:
            raise
        except Exception as exc:
            log_step_failed(
                exc,
                duration=time.monotonic() - started_at,
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
        except AttemptFenced:
            raise
        except Exception as exc:
            LOGGER.debug(
                "页面流水线 after 插件失败：job=%s item=%s page=%s",
                _short(fence.job_id),
                _short(item_id),
                _short(page_id),
                exc_info=True,
            )
            user_log(
                "error",
                f"页面完成后的插件处理失败｜{inline_log_text(exc)}",
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
        fence: AttemptFence,
        handler: BatchStepHandler,
        steps: Sequence[Mapping[str, Any]],
    ) -> None:
        active_steps: list[Mapping[str, Any]] = []
        for step in steps:
            with user_log_context(**_step_context_values(fence, step)):
                if self._before_pipeline(fence, step):
                    active_steps.append(step)
        if not active_steps:
            return
        started_at = time.monotonic()
        batch_context = _step_context_values(fence, active_steps[0])
        batch_context["page_number"] = None
        try:
            with user_log_context(**batch_context):
                user_log("step", f"批处理开始｜共 {len(active_steps)} 页")
                with owner_scope(fence.owner_user_id):
                    checkpoint = handler(fence, active_steps)
            if not isinstance(checkpoint, Mapping):
                raise TypeError("batch handler must return an object")
        except AttemptFenced:
            raise
        except Exception as exc:
            duration = time.monotonic() - started_at
            with user_log_context(**batch_context):
                log_step_failed(exc, duration=duration)
            step_statuses = self.repository.step_statuses(
                fence,
                tuple(str(step["stepId"]) for step in active_steps),
            )
            item_statuses = self.repository.item_statuses(
                fence,
                tuple(str(step["itemId"]) for step in active_steps),
            )
            for step in active_steps:
                step_id = str(step["stepId"])
                item_id = str(step["itemId"])
                step_status = step_statuses.get(step_id, "running")
                item_status = item_statuses.get(item_id, "running")
                with user_log_context(**_step_context_values(fence, step)):
                    if item_status in {
                        "completed",
                        "failed",
                        "skipped",
                        "cancelled",
                    }:
                        self._after_pipeline(
                            fence,
                            item_id=item_id,
                            page_id=step.get("pageId"),
                            status=item_status,
                        )
                        if step_status not in {
                            "completed",
                            "failed",
                            "skipped",
                            "cancelled",
                        }:
                            step_status = item_status
                    elif step_status not in {
                        "completed",
                        "failed",
                        "skipped",
                        "cancelled",
                    }:
                        self._after_pipeline(
                            fence,
                            item_id=item_id,
                            page_id=step.get("pageId"),
                            status="failed",
                        )
                        try:
                            self.repository.fail_step(
                                fence,
                                step_id=step_id,
                                code="BATCH_STEP_FAILED",
                                message=str(exc),
                            )
                        except AttemptFenced:
                            continue
                        step_status = "failed"
                    log_step_finished(duration=None, status=step_status)
            return
        if not checkpoint.get("__already_published__"):
            per_step = checkpoint.get("steps")
            for step in active_steps:
                value = (
                    per_step.get(str(step["stepId"]), {})
                    if isinstance(per_step, Mapping)
                    else checkpoint
                )
                self.repository.complete_step(
                    fence,
                    step_id=str(step["stepId"]),
                    checkpoint=value if isinstance(value, Mapping) else {},
                )
        duration = time.monotonic() - started_at
        step_statuses = self.repository.step_statuses(
            fence,
            tuple(str(step["stepId"]) for step in active_steps),
        )
        per_step = checkpoint.get("steps")
        for step in active_steps:
            value = (
                per_step.get(str(step["stepId"]), {})
                if isinstance(per_step, Mapping)
                else checkpoint
            )
            with user_log_context(**_step_context_values(fence, step)):
                status = step_statuses.get(str(step["stepId"]))
                if status not in {"completed", "failed", "skipped", "cancelled"}:
                    status = _checkpoint_log_status(
                        value if isinstance(value, Mapping) else {}
                    )
                log_step_finished(
                    duration=None,
                    status=status,
                )
        with user_log_context(**batch_context):
            user_log(
                "step",
                f"批处理完成｜共 {len(active_steps)} 页｜耗时 {duration:.2f} 秒",
            )
        statuses = self.repository.item_statuses(
            fence,
            tuple(str(step["itemId"]) for step in active_steps),
        )
        for step in active_steps:
            item_id = str(step["itemId"])
            status = statuses.get(item_id)
            if status in {"completed", "failed", "skipped", "cancelled"}:
                with user_log_context(**_step_context_values(fence, step)):
                    self._after_pipeline(
                        fence,
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
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("batch size must be an integer")
        parsed = value
        maximum: int | None = None
        if step_kind == "web_extract_page":
            maximum = 32
        if parsed < 1 or (maximum is not None and parsed > maximum):
            if maximum is None:
                raise ValueError("batch size must be at least 1")
            raise ValueError(f"batch size must be between 1 and {maximum}")
        return parsed
