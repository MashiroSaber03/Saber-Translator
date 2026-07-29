"""Worker scheduler that reconstructs all durable work from SQLite."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import logging
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
LOGGER = logging.getLogger("saber.worker.jobs")


def _short(value: object) -> str:
    return str(value)[:8]


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
            renewed = self.repository.renew_attempt(self.fence)
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
                config = self.plugin_runtime.before_pipeline(
                    fence,
                    config,
                )
            execution_mode = str(config.get("executionMode", "sequential"))
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
                        code="PLUGIN_LIFECYCLE_FAILED",
                        message=str(exc),
                    )
                except AttemptFenced:
                    pass
        finally:
            heartbeat.stop()
            LOGGER.info(
                "任务轮次结束：job=%s attempt=%s duration=%.2fs fenced=%s",
                _short(fence.job_id),
                _short(fence.attempt_id),
                time.monotonic() - started_at,
                heartbeat.fenced.is_set(),
            )

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
                        time.sleep(0.02)
                        continue
                    if self.plugin_runtime is not None:
                        terminal = {"status": "completed"}
                        terminal = self.plugin_runtime.after_pipeline(
                            fence,
                            terminal,
                        )
                        self.plugin_runtime.after_job(
                            fence,
                            terminal,
                        )
                    final_status = self.repository.finish_if_complete(fence)
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
                        message=f"no Worker handler for {step['stepKind']}",
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
                try:
                    effective_step = (
                        self.plugin_runtime.before_step(
                            heartbeat.fence,
                            step,
                        )
                        if self.plugin_runtime is not None
                        else step
                    )
                    checkpoint = handler(
                        heartbeat.fence,
                        effective_step,
                    )
                    if self.plugin_runtime is not None:
                        checkpoint = self.plugin_runtime.after_step(
                            heartbeat.fence,
                            effective_step,
                            checkpoint,
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
                    self.repository.fail_step(
                        heartbeat.fence,
                        step_id=last_step_id,
                        code="STEP_FAILED",
                        message=str(exc),
                    )
                else:
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
        try:
            deep_learning_concurrency = int(
                config.get("deepLearningConcurrency", 1)
            )
        except (TypeError, ValueError):
            deep_learning_concurrency = 1
        deep_learning_admission = threading.Semaphore(
            max(1, min(4, deep_learning_concurrency))
        )
        LOGGER.info(
            "并行流水线启动：job=%s pools=%s deep_learning_concurrency=%s",
            _short(heartbeat.fence.job_id),
            ",".join(pool_kinds),
            max(1, min(4, deep_learning_concurrency)),
        )
        lock_waiting_states = {
            kind: False
            for kind in pool_kinds
            if kind in DEEP_LEARNING_STEP_KINDS
        }
        lock_waiting_state_lock = threading.Lock()

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
            while (
                not stop_event.is_set()
                and not heartbeat.fenced.is_set()
                and not admission_closed.is_set()
            ):
                try:
                    status = self.repository.control_status(heartbeat.fence)
                    if status in {"pausing", "cancelling"}:
                        admission_closed.set()
                        return
                    step_ordinal = (
                        self.repository.ready_step_ordinal(
                            heartbeat.fence,
                            step_kind=pool_kind,
                        )
                        if pool_kind in self.batch_handlers
                        else None
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
                        if step_ordinal is not None
                        else []
                    )
                    step = (
                        None
                        if pool_kind in self.batch_handlers
                        else self.repository.next_step(
                            heartbeat.fence,
                            allowed_kinds=(pool_kind,),
                        )
                    )
                    if not steps and step is None:
                        pending, running = self.repository.active_step_counts(
                            heartbeat.fence
                        )
                        if pending == 0 and running == 0:
                            admission_closed.set()
                            return
                        time.sleep(0.02)
                        continue
                    if steps:
                        self._execute_batch(
                            heartbeat,
                            self.batch_handlers[pool_kind],
                            steps,
                        )
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
                    try:
                        def execute_step() -> Mapping[str, Any]:
                            effective_step = (
                                self.plugin_runtime.before_step(
                                    heartbeat.fence,
                                    step,
                                )
                                if self.plugin_runtime is not None
                                else step
                            )
                            checkpoint = handler(
                                heartbeat.fence,
                                effective_step,
                            )
                            if self.plugin_runtime is not None:
                                checkpoint = self.plugin_runtime.after_step(
                                    heartbeat.fence,
                                    effective_step,
                                    checkpoint,
                                )
                            return checkpoint

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
                        self.repository.fail_step(
                            heartbeat.fence,
                            step_id=str(step["stepId"]),
                            code="STEP_FAILED",
                            message=str(exc),
                        )
                    else:
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
                except AttemptFenced:
                    return
                except BaseException as exc:
                    LOGGER.exception(
                        "并行流水线线程失败：job=%s pool=%s",
                        _short(heartbeat.fence.job_id),
                        pool_kind,
                    )
                    with error_lock:
                        worker_errors.append(exc)
                    admission_closed.set()
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
                not admission_closed.wait(0.05)
                and not stop_event.is_set()
                and not heartbeat.fenced.is_set()
            ):
                if self.safe_point is not None:
                    _pending, running = self.repository.active_step_counts(
                        heartbeat.fence
                    )
                    if running == 0:
                        self.safe_point()
            admission_closed.set()
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
        if self.plugin_runtime is not None:
            terminal = {"status": "completed"}
            terminal = self.plugin_runtime.after_pipeline(
                heartbeat.fence,
                terminal,
            )
            self.plugin_runtime.after_job(
                heartbeat.fence,
                terminal,
            )
        final_status = self.repository.finish_if_complete(heartbeat.fence)
        LOGGER.info(
            "并行任务全部步骤处理结束：job=%s status=%s",
            _short(heartbeat.fence.job_id),
            final_status or "running",
        )

    def _execute_batch(
        self,
        heartbeat: AttemptHeartbeat,
        handler: BatchStepHandler,
        steps: Sequence[Mapping[str, Any]],
    ) -> None:
        step_kind = str(steps[0].get("stepKind", "unknown")) if steps else "unknown"
        step_ids = ",".join(_short(step.get("stepId", "-")) for step in steps)
        started_at = time.monotonic()
        LOGGER.info(
            "批处理开始：job=%s kind=%s count=%s steps=%s",
            _short(heartbeat.fence.job_id),
            step_kind,
            len(steps),
            step_ids,
        )
        effective_steps = [
            (
                self.plugin_runtime.before_step(heartbeat.fence, step)
                if self.plugin_runtime is not None
                else step
            )
            for step in steps
        ]
        try:
            checkpoint = handler(heartbeat.fence, effective_steps)
        except Exception as exc:
            LOGGER.exception(
                "批处理失败：job=%s kind=%s count=%s duration=%.2fs",
                _short(heartbeat.fence.job_id),
                step_kind,
                len(steps),
                time.monotonic() - started_at,
            )
            if heartbeat.fenced.is_set():
                return
            for step in steps:
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
            len(steps),
            time.monotonic() - started_at,
        )
        if self.plugin_runtime is not None:
            for step in effective_steps:
                self.plugin_runtime.after_step(
                    heartbeat.fence,
                    step,
                    checkpoint,
                )
        if checkpoint.get("__already_published__"):
            return
        per_step = checkpoint.get("steps")
        for step in steps:
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

    @staticmethod
    def _batch_size(
        step_kind: str,
        config: Mapping[str, Any],
        *,
        step_ordinal: int | None,
    ) -> int:
        if step_kind == "hq_translate":
            section = config.get("translation")
            value = section.get("batchSize", 3) if isinstance(section, Mapping) else 3
        elif step_kind == "proofread":
            rounds = config.get("proofreadingRounds")
            index = max(0, int(step_ordinal or 1) - 1)
            section = (
                rounds[index]
                if isinstance(rounds, list)
                and index < len(rounds)
                and isinstance(rounds[index], Mapping)
                else {}
            )
            value = section.get("batchSize", 3)
        else:
            value = 1
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 3
        return max(1, min(10, parsed))


def wait_for_stop_step(
    _fence: AttemptFence,
    step: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Small deterministic handler used only by integration probes/tests."""

    time.monotonic()
    return {"stepId": step["stepId"], "completed": True}
