"""Minimal v2 Worker process lifecycle."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import threading
from typing import Any

from src.backend_v2.logging_config import configure_backend_logging
from src.backend_v2.paths import data_root_fingerprint, ensure_data_root, resolve_data_root
from src.backend_v2.runtime_heartbeat import EpochHeartbeat
from src.backend_v2.runtime_identity import (
    LauncherParentMonitor,
    RuntimeIdentity,
    start_launcher_parent_monitor,
)
from src.backend_v2.runtime_profile import PROFILE_ENV, resolve_runtime_profile
from src.backend_v2.storage.database import (
    create_sqlite_engine,
    database_path_for,
    is_sqlite_busy_error,
)
from src.backend_v2.storage.epochs import ProcessEpochRepository


LOGGER = logging.getLogger("saber.worker")


def _insight_layer_handler(step_kind: str, service: Any):
    prefix = "insight_build_layer_"
    suffix = step_kind.removeprefix(prefix)
    if (
        step_kind.startswith(prefix)
        and suffix.isascii()
        and suffix.isdigit()
        and str(int(suffix)) == suffix
    ):
        return service.handle
    return None


def _write_ready_marker(data_root: Path, identity: RuntimeIdentity) -> None:
    runtime_dir = data_root / "runtime"
    marker = runtime_dir / "worker-ready.json"
    temporary = runtime_dir / f".worker-ready-{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "epochId": identity.epoch_id,
                "dataRootFingerprint": data_root_fingerprint(data_root),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(marker)


def run_worker(args: object) -> int:
    profile = resolve_runtime_profile(getattr(args, "profile", "local"))
    if profile.name == "public" and not getattr(args, "data_dir", None):
        raise ValueError("--data-dir is required for the public profile")
    os.environ[PROFILE_ENV] = profile.name
    data_root = ensure_data_root(resolve_data_root(args.data_dir))
    if not args.probe:
        log_path = configure_backend_logging(
            role="worker",
            data_root=data_root,
            console_level=args.log_level,
        )
        LOGGER.info(
            "Worker 进程启动：pid=%s，data_root=%s，日志=%s",
            os.getpid(),
            data_root,
            log_path,
        )
    identity = RuntimeIdentity.for_worker(test_mode=args.test_mode)
    engine = None
    repository = None
    if not identity.test_mode:
        engine = create_sqlite_engine(database_path_for(data_root))
        repository = ProcessEpochRepository(engine)
        if not repository.validate(
            role="worker",
            epoch_id=identity.epoch_id,
            token=identity.epoch_token,
        ):
            engine.dispose()
            raise RuntimeError("Launcher-issued Worker epoch is missing, expired, or invalid")

    if args.probe:
        print(
            json.dumps(
                {
                    "role": "worker",
                    "status": "ready",
                    "epochId": identity.epoch_id,
                    "dataRootFingerprint": data_root_fingerprint(data_root),
                },
                sort_keys=True,
            )
        )
        if engine is not None:
            engine.dispose()
        return 0

    _write_ready_marker(data_root, identity)
    LOGGER.info(
        "Worker 租约验证完成：epoch=%s，ready marker 已写入",
        identity.epoch_id[:8],
    )
    stop_event = threading.Event()
    parent_monitor: LauncherParentMonitor | None = None
    heartbeat = (
        EpochHeartbeat(
            repository,
            role="worker",
            identity=identity,
            on_fenced=stop_event.set,
        )
        if repository is not None
        else None
    )

    def request_stop(_signum: int, _frame: object) -> None:
        LOGGER.info("Worker 收到终止信号")
        stop_event.set()

    def stop_orphaned_worker() -> None:
        LOGGER.critical("Launcher 进程已退出，Worker 立即终止")
        os._exit(75)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if heartbeat is not None:
        heartbeat.start()
    try:
        parent_monitor = start_launcher_parent_monitor(
            stop_orphaned_worker,
            test_mode=identity.test_mode,
        )
    except BaseException:
        if heartbeat is not None:
            heartbeat.stop()
        if engine is not None:
            engine.dispose()
        raise
    try:
        if engine is None:
            while not stop_event.wait(timeout=0.5):
                pass
        else:
            from src.backend_v2.jobs.repository import JobQueueRepository
            from src.backend_v2.jobs.worker_loop import JobWorkerLoop
            from src.backend_v2.insight.worker import (
                InsightAnalysisWorkerService,
            )
            from src.backend_v2.insight.derived import (
                InsightDerivedWorkerService,
            )
            from src.backend_v2.insight.continuation import (
                ContinuationWorkerService,
            )
            from src.backend_v2.insight.qa import InsightQAWorkerService
            from src.backend_v2.insight.exports import (
                InsightExportWorkerService,
            )
            from src.backend_v2.operations.executor import WorkerOperationRunner
            from src.backend_v2.operations.repair import PageRepairService
            from src.backend_v2.operations.repository import OperationRepository
            from src.backend_v2.plugins.runtime import (
                PluginJobRuntime,
                PluginOperationRuntime,
            )
            from src.backend_v2.plugins.agent_worker import (
                PluginAgentWorkerService,
            )
            from src.backend_v2.translation.interactive_operations import (
                InteractivePageOperationService,
            )
            from src.backend_v2.translation.auxiliary import (
                StyleApplyWorkerService,
                TextImportWorkerService,
            )
            from src.backend_v2.translation.pipeline import (
                TranslationPipelineService,
            )
            from src.backend_v2.content.image_import import (
                ImportSafetyLimits,
                PUBLIC_IMPORT_SAFETY_LIMITS,
            )
            from src.backend_v2.transfer.worker import TransferWorkerService
            from src.backend_v2.web_import.worker import (
                WebImportWorkerService,
            )
            from src.backend_v2.worker.model_lifecycle import (
                WorkerModelControlRepository,
                WorkerModelLifecycle,
            )
            from src.backend_v2.worker.maintenance import WorkerMaintenance
            from src.backend_v2.storage.platform_repositories import (
                ProviderRateLimiter,
            )
            from src.backend_v2.scheduling_policy import (
                SchedulingPolicyCache,
                SchedulingPolicyRepository,
                available_memory_mib,
            )
            from src.shared.openai_rate_limits import (
                configure_provider_rate_limit_store,
            )

            configure_provider_rate_limit_store(ProviderRateLimiter(engine))
            scheduling_policy = (
                SchedulingPolicyCache(SchedulingPolicyRepository(engine))
                if profile.name == "public"
                else None
            )
            job_repository = JobQueueRepository(engine)
            plugin_job_runtime = PluginJobRuntime(
                data_root=data_root,
                engine=engine,
                repository=job_repository,
            )
            translation = TranslationPipelineService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
                plugin_runtime=plugin_job_runtime,
            )
            translation_steps = {
                "detect",
                "ocr",
                "color",
                "auto_terms",
                "translate",
                "hq_translate",
                "proofread",
                "repair",
                "render",
                "save",
                "publish_clean",
            }
            transfer = TransferWorkerService(
                data_root=data_root,
                engine=engine,
                jobs_repository=job_repository,
                limits=(
                    PUBLIC_IMPORT_SAFETY_LIMITS
                    if profile.name == "public"
                    else ImportSafetyLimits()
                ),
            )
            job_handlers = {
                step_kind: translation.handler
                for step_kind in translation_steps
            }
            job_handlers.update(
                {
                    step_kind: transfer.handler
                    for step_kind in (
                        "container_scan",
                        "container_import_page",
                        "export_package",
                    )
                }
            )
            web_import = WebImportWorkerService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
            )
            job_handlers.update(
                {
                    step_kind: web_import.handle
                    for step_kind in (
                        "web_extract_scan",
                        "web_extract_page",
                        "web_extract_finalize",
                        "web_extract_auto_commit",
                        "web_import_commit_page",
                        "web_import_commit_finalize",
                    )
                }
            )
            style_apply = StyleApplyWorkerService(
                engine=engine,
                jobs=job_repository,
            )
            job_handlers["style_apply_document"] = style_apply.handle
            text_import = TextImportWorkerService(
                engine=engine,
                jobs=job_repository,
            )
            job_handlers["text_import_apply"] = text_import.handle
            insight = InsightAnalysisWorkerService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
            )
            job_handlers.update(
                {
                    "insight_analyze_page": insight.handle,
                    "insight_validate_run": insight.handle,
                    "insight_publish_run": insight.handle,
                }
            )
            insight_derived = InsightDerivedWorkerService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
            )
            job_handlers.update(
                {
                    "insight_build_overview": insight_derived.handle,
                    "insight_build_compressed_context": insight_derived.handle,
                    "insight_build_timeline": insight_derived.handle,
                    "insight_build_vectors": insight_derived.handle,
                    "insight_stage_compressed_context": insight_derived.handle,
                    "insight_stage_overview_no_spoiler": insight_derived.handle,
                    "insight_stage_overview_story_summary": insight_derived.handle,
                    "insight_stage_timeline": insight_derived.handle,
                    "insight_stage_vectors": insight_derived.handle,
                }
            )
            continuation = ContinuationWorkerService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
            )
            job_handlers.update(
                {
                    "continuation_generate_script": continuation.handle,
                    "continuation_generate_page": continuation.handle,
                    "continuation_generate_image": continuation.handle,
                    "continuation_generate_character_sheet": (
                        continuation.handle
                    ),
                    "continuation_export": continuation.handle,
                }
            )
            insight_export = InsightExportWorkerService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
            )
            job_handlers["insight_export_report"] = insight_export.handle
            plugin_agent = PluginAgentWorkerService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
            )
            job_handlers["plugin_agent_execute"] = plugin_agent.handle
            operation_repository = OperationRepository(engine)
            plugin_operation_runtime = PluginOperationRuntime(
                data_root=data_root,
                engine=engine,
                repository=operation_repository,
            )
            interactive = InteractivePageOperationService(
                data_root=data_root,
                engine=engine,
                repository=operation_repository,
                plugin_runtime=plugin_operation_runtime,
            )
            repairs = PageRepairService(
                data_root=data_root,
                engine=engine,
                repository=operation_repository,
                plugin_runtime=plugin_operation_runtime,
            )
            operation_runner = WorkerOperationRunner(
                operation_repository,
                worker_epoch_id=identity.epoch_id,
                handlers={
                    "bubble_ocr": interactive.handle,
                    "bubble_color": interactive.handle,
                    "page_detect": interactive.handle,
                    "page_repair": repairs.handle,
                },
            )
            qa_runner = InsightQAWorkerService(
                data_root=data_root,
                engine=engine,
                worker_epoch_id=identity.epoch_id,
            )
            model_lifecycle = WorkerModelLifecycle(
                WorkerModelControlRepository(engine),
                worker_epoch_id=identity.epoch_id,
                idle_timeout_provider=(
                    (lambda: float(scheduling_policy.load()["modelIdleSeconds"]))
                    if scheduling_policy is not None
                    else None
                ),
                release_callbacks=(
                    plugin_job_runtime.release_cached_instances,
                    plugin_operation_runtime.release_cached_instances,
                ),
            )
            maintenance = WorkerMaintenance(
                data_root=data_root,
                engine=engine,
            )
            maintenance.run_if_due(force=True)
            LOGGER.info(
                "Worker 服务初始化完成：任务步骤处理器=%s，批处理器=3，操作处理器=4",
                len(job_handlers),
            )
            LOGGER.info("Worker 调度循环已就绪，开始从 SQLite 队列领取任务")

            def memory_admitted() -> bool:
                if scheduling_policy is None:
                    return True
                threshold = int(
                    scheduling_policy.load()["minAvailableMemoryMiB"]
                )
                if threshold == 0 or available_memory_mib() >= threshold:
                    return True
                try:
                    model_lifecycle.release_for_memory_pressure()
                except Exception as exc:
                    if not is_sqlite_busy_error(exc):
                        raise
                    return False
                return available_memory_mib() >= threshold

            def run_immediate_work() -> bool:
                try:
                    if maintenance.run_if_due():
                        return True
                    if model_lifecycle.run_pending_release():
                        return True
                    if not memory_admitted():
                        return False
                    if operation_runner.run_one() or qa_runner.run_one():
                        model_lifecycle.note_activity()
                        return True
                    return model_lifecycle.release_if_idle()
                except Exception as exc:
                    if not is_sqlite_busy_error(exc):
                        raise
                    LOGGER.warning(
                        "Worker 即时任务遇到 SQLite 写锁竞争，将在下一轮重试"
                    )
                    return False

            JobWorkerLoop(
                job_repository,
                worker_epoch_id=identity.epoch_id,
                handlers=job_handlers,
                batch_handlers={
                    "hq_translate": translation.batch_handler,
                    "proofread": translation.batch_handler,
                    "web_extract_page": web_import.handle_download_batch,
                },
                handler_resolver=lambda step_kind: _insight_layer_handler(
                    step_kind,
                    insight_derived,
                ),
                safe_point=run_immediate_work,
                scheduling_policy=(
                    scheduling_policy.load
                    if scheduling_policy is not None
                    else None
                ),
                admission_check=(
                    memory_admitted if scheduling_policy is not None else None
                ),
                on_activity=model_lifecycle.note_activity,
                plugin_runtime=plugin_job_runtime,
            ).run(stop_event)
    except BaseException:
        LOGGER.exception("Worker 运行失败")
        raise
    finally:
        LOGGER.info("Worker 正在关闭")
        if heartbeat is not None:
            heartbeat.stop()
        if parent_monitor is not None:
            parent_monitor.stop()
        if engine is not None:
            engine.dispose()
        LOGGER.info("Worker 已关闭")

    return 75 if heartbeat is not None and not heartbeat.healthy else 0
