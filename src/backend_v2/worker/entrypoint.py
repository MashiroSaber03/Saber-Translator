"""Minimal v2 Worker process lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import threading

from src.backend_v2.paths import data_root_fingerprint, ensure_data_root, resolve_data_root
from src.backend_v2.runtime_heartbeat import EpochHeartbeat
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.epochs import ProcessEpochRepository


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
    data_root = ensure_data_root(resolve_data_root(getattr(args, "data_dir", None)))
    identity = RuntimeIdentity.for_worker(test_mode=bool(getattr(args, "test_mode", False)))
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

    if getattr(args, "probe", False):
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
    stop_event = threading.Event()
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
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if heartbeat is not None:
        heartbeat.start()
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
            from src.backend_v2.operations.executor import WorkerOperationRunner
            from src.backend_v2.operations.repair import PageRepairService
            from src.backend_v2.operations.repository import OperationRepository
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
            from src.backend_v2.transfer.worker import TransferWorkerService
            from src.backend_v2.web_import.worker import (
                WebImportWorkerService,
            )

            job_repository = JobQueueRepository(engine)
            translation = TranslationPipelineService(
                data_root=data_root,
                engine=engine,
                jobs=job_repository,
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
                "publish_clean",
            }
            transfer = TransferWorkerService(
                data_root=data_root,
                engine=engine,
                jobs_repository=job_repository,
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
                        "container_cleanup",
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
            for layer_index in range(8):
                job_handlers[
                    f"insight_build_layer_{layer_index}"
                ] = insight_derived.handle
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
                    "continuation_export": continuation.handle,
                }
            )
            operation_repository = OperationRepository(engine)
            interactive = InteractivePageOperationService(
                data_root=data_root,
                engine=engine,
                repository=operation_repository,
            )
            repairs = PageRepairService(
                data_root=data_root,
                engine=engine,
                repository=operation_repository,
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

            def run_immediate_work() -> bool:
                return operation_runner.run_one() or qa_runner.run_one()

            JobWorkerLoop(
                job_repository,
                worker_epoch_id=identity.epoch_id,
                handlers=job_handlers,
                safe_point=run_immediate_work,
            ).run(stop_event)
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if engine is not None:
            engine.dispose()

    return 75 if heartbeat is not None and not heartbeat.healthy else 0
