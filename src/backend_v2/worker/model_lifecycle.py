"""Worker-owned local model unloading and its durable control channel."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import logging
import sys
import time
from typing import Any
import uuid

from sqlalchemy import Engine, exists, select, update

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import utcnow
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    jobs,
    job_items,
    job_steps,
    operations,
    process_epochs,
    transient_requests,
)
from src.shared.user_logging import inline_log_text, user_log

LOGGER = logging.getLogger("saber.worker.models")


LOCAL_MODEL_JOB_STEPS = frozenset({"detect", "ocr", "color", "repair"})
ACTIVE_JOB_RUNTIME_STATUSES = frozenset(
    {"running"}
)


class ModelInferenceBusy(RuntimeError):
    """A manual release cannot interrupt a local model call."""


class WorkerCommandFenced(RuntimeError):
    """The Worker command belongs to a different process epoch."""


def _log_list(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "-"
    return ",".join(str(item) for item in value)


class WorkerModelControlRepository:
    """Send the one supported process control through the Worker epoch row."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def request_release(self) -> dict[str, object]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            if self._model_inference_active(connection):
                raise ModelInferenceBusy(
                    "local model inference is running; try again after the "
                    "current model step finishes"
                )
            worker = connection.execute(
                select(
                    process_epochs.c.id,
                    process_epochs.c.model_release_request_id,
                    process_epochs.c.model_release_handled_id,
                )
                .where(
                    process_epochs.c.role == "worker",
                    process_epochs.c.status == "active",
                    process_epochs.c.lease_expires_at > now,
                )
                .order_by(process_epochs.c.created_at.desc())
                .limit(1)
            ).mappings().one_or_none()
            if worker is None:
                raise WorkerCommandFenced("no active Worker is available")
            active_request = worker["model_release_request_id"]
            if (
                active_request is not None
                and active_request != worker["model_release_handled_id"]
            ):
                return {
                    "commandId": str(active_request),
                    "kind": "release_models",
                    "status": "pending",
                }
            command_id = str(uuid.uuid4())
            connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == worker["id"],
                    process_epochs.c.status == "active",
                )
                .values(
                    model_release_request_id=command_id,
                    model_release_result_json=None,
                    model_release_error_json=None,
                    updated_at=now,
                )
            )
        return {
            "commandId": command_id,
            "kind": "release_models",
            "status": "pending",
        }

    def claim_release(
        self,
        *,
        worker_epoch_id: str,
    ) -> dict[str, object] | None:
        now = utcnow()
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    process_epochs.c.model_release_request_id,
                    process_epochs.c.model_release_handled_id,
                )
                .where(
                    process_epochs.c.id == worker_epoch_id,
                    process_epochs.c.role == "worker",
                    process_epochs.c.status == "active",
                    process_epochs.c.lease_expires_at > now,
                )
            ).mappings().one_or_none()
            if (
                row is None
                or row["model_release_request_id"] is None
                or row["model_release_request_id"]
                == row["model_release_handled_id"]
            ):
                return None
            return {
                "commandId": str(row["model_release_request_id"]),
                "kind": "release_models",
                "status": "running",
            }

    def complete(
        self,
        *,
        command_id: str,
        worker_epoch_id: str,
        result: Mapping[str, Any],
    ) -> None:
        self._finish(
            command_id=command_id,
            worker_epoch_id=worker_epoch_id,
            values={"result_json": _json(dict(result)), "error_json": None},
        )

    def fail(
        self,
        *,
        command_id: str,
        worker_epoch_id: str,
        message: str,
    ) -> None:
        self._finish(
            command_id=command_id,
            worker_epoch_id=worker_epoch_id,
            values={
                "result_json": None,
                "error_json": _json(
                    {"code": "MODEL_RELEASE_FAILED", "message": message}
                ),
            },
        )

    def runtime_busy(self) -> bool:
        with self.engine.connect() as connection:
            return bool(
                connection.execute(
                    select(
                        exists().where(
                            jobs.c.status.in_(ACTIVE_JOB_RUNTIME_STATUSES)
                        )
                    )
                ).scalar()
                or connection.execute(
                    select(
                        exists().where(
                            operations.c.executor_role == "worker",
                            operations.c.status == "running",
                        )
                    )
                ).scalar()
                or connection.execute(
                    select(
                        exists().where(
                            transient_requests.c.status == "running"
                        )
                    )
                ).scalar()
            )

    def model_inference_busy(self) -> bool:
        with self.engine.connect() as connection:
            return self._model_inference_active(connection)

    @staticmethod
    def _model_inference_active(connection: Any) -> bool:
        active_step = connection.execute(
            select(
                exists().where(
                    job_steps.c.status == "running",
                    job_steps.c.kind.in_(LOCAL_MODEL_JOB_STEPS),
                    job_steps.c.job_item_id == job_items.c.id,
                    job_items.c.job_id == jobs.c.id,
                    jobs.c.status.in_(ACTIVE_JOB_RUNTIME_STATUSES),
                )
            )
        ).scalar()
        active_operation = connection.execute(
            select(
                exists().where(
                    operations.c.executor_role == "worker",
                    operations.c.status == "running",
                )
            )
        ).scalar()
        active_transient = connection.execute(
            select(
                exists().where(
                    transient_requests.c.status == "running"
                )
            )
        ).scalar()
        return bool(active_step or active_operation or active_transient)

    def _finish(
        self,
        *,
        command_id: str,
        worker_epoch_id: str,
        values: Mapping[str, object],
    ) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            release_result = values.get("result_json")
            release_error = values.get("error_json")
            changed = connection.execute(
                update(process_epochs)
                .where(
                    process_epochs.c.id == worker_epoch_id,
                    process_epochs.c.role == "worker",
                    process_epochs.c.status == "active",
                    process_epochs.c.lease_expires_at > now,
                    process_epochs.c.model_release_request_id == command_id,
                )
                .values(
                    model_release_handled_id=command_id,
                    model_release_result_json=release_result,
                    model_release_error_json=release_error,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise WorkerCommandFenced(
                    "Worker model command is no longer owned by this epoch"
                )


class WorkerModelLifecycle:
    """Run manual releases and the ten-minute idle eviction policy."""

    def __init__(
        self,
        repository: WorkerModelControlRepository,
        *,
        worker_epoch_id: str,
        idle_timeout_seconds: float = 600,
        idle_timeout_provider: Callable[[], float] | None = None,
        release_callbacks: Sequence[Callable[[], object]] = (),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repository = repository
        self.worker_epoch_id = worker_epoch_id
        self.idle_timeout_seconds = max(0.0, idle_timeout_seconds)
        self.idle_timeout_provider = idle_timeout_provider
        self.release_callbacks = tuple(release_callbacks)
        self.monotonic = monotonic
        self.last_activity = monotonic()
        self.released_since_activity = False

    def note_activity(self) -> None:
        self.last_activity = self.monotonic()
        self.released_since_activity = False

    def run_pending_release(self) -> bool:
        if self.repository.model_inference_busy():
            return False
        command = self.repository.claim_release(
            worker_epoch_id=self.worker_epoch_id
        )
        if command is None:
            return False
        command_id = str(command["commandId"])
        LOGGER.debug("开始执行手动模型释放：command=%s", command_id[:8])
        user_log("system", "正在释放本地模型与运行时缓存")
        try:
            result = unload_loaded_models(
                release_callbacks=self.release_callbacks,
            )
        except Exception as exc:
            LOGGER.debug("手动模型释放失败：command=%s", command_id[:8], exc_info=True)
            user_log(
                "error",
                f"本地模型释放失败｜{inline_log_text(exc)}",
            )
            self.repository.fail(
                command_id=command_id,
                worker_epoch_id=self.worker_epoch_id,
                message=str(exc),
            )
        else:
            # A transient SQLite lock leaves the request pending, so the next
            # scheduler pass can safely repeat this idempotent cache release.
            self.repository.complete(
                command_id=command_id,
                worker_epoch_id=self.worker_epoch_id,
                result=result,
            )
            user_log(
                "system",
                f"本地模型与缓存已释放｜{_log_list(result.get('released'))}",
            )
        self.last_activity = self.monotonic()
        self.released_since_activity = True
        return True

    def release_if_idle(self) -> bool:
        if self.released_since_activity:
            return False
        if (
            self.monotonic() - self.last_activity
            < self._idle_timeout_seconds()
        ):
            return False
        if self.repository.runtime_busy():
            self.note_activity()
            return False
        unload_loaded_models(
            release_callbacks=self.release_callbacks,
        )
        user_log(
            "system",
            f"空闲 {self.monotonic() - self.last_activity:.0f} 秒，"
            "已自动释放本地模型与缓存",
        )
        self.released_since_activity = True
        return True

    def release_for_memory_pressure(self) -> bool:
        """Release caches once at a safe boundary before delaying new work."""

        if self.released_since_activity or self.repository.model_inference_busy():
            return False
        unload_loaded_models(release_callbacks=self.release_callbacks)
        user_log(
            "warning",
            "可用内存低于安全阈值，已释放本地模型与运行时缓存",
        )
        self.released_since_activity = True
        return True

    def _idle_timeout_seconds(self) -> float:
        if self.idle_timeout_provider is None:
            return self.idle_timeout_seconds
        value = self.idle_timeout_provider()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("model idle timeout must be numeric")
        return max(0.0, float(value))


def unload_loaded_models(
    *,
    release_callbacks: Sequence[Callable[[], object]] = (),
) -> dict[str, object]:
    """Unload only modules already imported by this Worker process."""

    released: list[str] = []
    failures: list[str] = []
    resetters = (
        (
            "src.core.detector.registry",
            "reset_detector",
            "detectors",
        ),
        (
            "src.interfaces.manga_ocr_interface",
            "reset_manga_ocr_instance",
            "manga_ocr",
        ),
        (
            "src.interfaces.ocr_48px.interface",
            "reset_48px_ocr_handler",
            "ocr_48px",
        ),
        (
            "src.interfaces.paddleocr_vl_interface",
            "reset_paddleocr_vl_handler",
            "paddleocr_vl",
        ),
        (
            "src.interfaces.paddle_ocr_onnx_interface",
            "reset_paddle_ocr_handler",
            "paddle_ocr",
        ),
        (
            "src.interfaces.lama_interface",
            "reset_litelama_inpainter",
            "litelama",
        ),
        (
            "src.interfaces.lama_mpe_interface",
            "reset_lama_mpe_inpainter",
            "lama_mpe",
        ),
    )
    for module_name, function_name, label in resetters:
        module = sys.modules.get(module_name)
        resetter = (
            getattr(module, function_name, None)
            if module is not None
            else None
        )
        if callable(resetter):
            try:
                resetter()
                released.append(label)
            except Exception as exc:
                failures.append(f"{label}: {exc}")
    for index, callback in enumerate(release_callbacks):
        label = f"runtime_cache_{index + 1}"
        try:
            callback()
            released.append(label)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    gc.collect()
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        if cuda.is_available():
            cuda.empty_cache()
    if failures:
        raise RuntimeError(
            "some model caches could not be released: "
            + "; ".join(failures)
        )
    return {
        "released": released,
        "releasedCount": len(released),
    }
