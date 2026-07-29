"""Worker-owned local model unloading and its durable control channel."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import gc
import json
import sys
import time
from typing import Any
import uuid

from sqlalchemy import Engine, exists, insert, select, update
from sqlalchemy.exc import IntegrityError

from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    jobs,
    job_items,
    job_steps,
    operations,
    process_epochs,
    transient_requests,
    worker_commands,
)


LOCAL_MODEL_JOB_STEPS = frozenset({"detect", "ocr", "color", "repair"})
ACTIVE_JOB_RUNTIME_STATUSES = frozenset(
    {"running", "pausing", "cancelling"}
)


class ModelInferenceBusy(RuntimeError):
    """A manual release cannot interrupt a local model call."""


class WorkerCommandFenced(RuntimeError):
    """The Worker command belongs to a different process epoch."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class WorkerModelControlRepository:
    """Persist and fence commands sent from API to the isolated Worker."""

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
            active = connection.execute(
                select(worker_commands)
                .where(
                    worker_commands.c.kind == "release_models",
                    worker_commands.c.status.in_(("pending", "running")),
                )
                .order_by(worker_commands.c.created_at)
                .limit(1)
            ).mappings().one_or_none()
            if active is not None:
                return self._dto(active)
            command_id = str(uuid.uuid4())
            try:
                connection.execute(
                    insert(worker_commands).values(
                        id=command_id,
                        kind="release_models",
                        status="pending",
                        created_at=now,
                        updated_at=now,
                    )
                )
            except IntegrityError:
                active = connection.execute(
                    select(worker_commands)
                    .where(
                        worker_commands.c.kind == "release_models",
                        worker_commands.c.status.in_(("pending", "running")),
                    )
                    .limit(1)
                ).mappings().one()
                return self._dto(active)
        return {
            "commandId": command_id,
            "kind": "release_models",
            "status": "pending",
        }

    def recover_for_worker(self, *, worker_epoch_id: str) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_worker_epoch(
                connection,
                worker_epoch_id=worker_epoch_id,
            )
            connection.execute(
                update(worker_commands)
                .where(
                    worker_commands.c.status == "running",
                    (
                        worker_commands.c.worker_epoch_id.is_(None)
                        | (
                            worker_commands.c.worker_epoch_id
                            != worker_epoch_id
                        )
                    ),
                )
                .values(
                    status="pending",
                    worker_epoch_id=None,
                    started_at=None,
                    updated_at=now,
                )
            )

    def claim_release(
        self,
        *,
        worker_epoch_id: str,
    ) -> dict[str, object] | None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._assert_worker_epoch(
                connection,
                worker_epoch_id=worker_epoch_id,
            )
            row = connection.execute(
                select(worker_commands)
                .where(
                    worker_commands.c.kind == "release_models",
                    worker_commands.c.status == "pending",
                )
                .order_by(worker_commands.c.created_at)
                .limit(1)
            ).mappings().one_or_none()
            if row is None:
                return None
            changed = connection.execute(
                update(worker_commands)
                .where(
                    worker_commands.c.id == row["id"],
                    worker_commands.c.status == "pending",
                )
                .values(
                    status="running",
                    worker_epoch_id=worker_epoch_id,
                    started_at=now,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                return None
            return {
                "commandId": str(row["id"]),
                "kind": str(row["kind"]),
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
            status="completed",
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
            status="failed",
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
        status: str,
        values: Mapping[str, object],
    ) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(worker_commands)
                .where(
                    worker_commands.c.id == command_id,
                    worker_commands.c.status == "running",
                    worker_commands.c.worker_epoch_id == worker_epoch_id,
                )
                .values(
                    status=status,
                    finished_at=now,
                    updated_at=now,
                    **dict(values),
                )
            )
            if changed.rowcount != 1:
                raise WorkerCommandFenced(
                    "Worker model command is no longer owned by this epoch"
                )

    @staticmethod
    def _assert_worker_epoch(
        connection: Any,
        *,
        worker_epoch_id: str,
    ) -> None:
        active = connection.execute(
            select(process_epochs.c.id).where(
                process_epochs.c.id == worker_epoch_id,
                process_epochs.c.role == "worker",
                process_epochs.c.status == "active",
                process_epochs.c.lease_expires_at > utcnow(),
            )
        ).scalar_one_or_none()
        if active is None:
            raise WorkerCommandFenced("Worker epoch is inactive or expired")

    @staticmethod
    def _dto(row: Mapping[str, Any]) -> dict[str, object]:
        return {
            "commandId": str(row["id"]),
            "kind": str(row["kind"]),
            "status": str(row["status"]),
        }


class WorkerModelLifecycle:
    """Run manual releases and the ten-minute idle eviction policy."""

    def __init__(
        self,
        repository: WorkerModelControlRepository,
        *,
        worker_epoch_id: str,
        idle_timeout_seconds: float = 600,
        release_callbacks: Sequence[Callable[[], object]] = (),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repository = repository
        self.worker_epoch_id = worker_epoch_id
        self.idle_timeout_seconds = max(0.0, idle_timeout_seconds)
        self.release_callbacks = tuple(release_callbacks)
        self.monotonic = monotonic
        self.last_activity = monotonic()
        self.released_since_activity = False
        self.repository.recover_for_worker(
            worker_epoch_id=worker_epoch_id
        )

    def note_activity(self) -> None:
        self.last_activity = self.monotonic()
        self.released_since_activity = False

    def run_pending_release(self) -> bool:
        command = self.repository.claim_release(
            worker_epoch_id=self.worker_epoch_id
        )
        if command is None:
            return False
        command_id = str(command["commandId"])
        try:
            result = unload_loaded_models(
                release_callbacks=self.release_callbacks
            )
            self.repository.complete(
                command_id=command_id,
                worker_epoch_id=self.worker_epoch_id,
                result=result,
            )
        except Exception as exc:
            self.repository.fail(
                command_id=command_id,
                worker_epoch_id=self.worker_epoch_id,
                message=str(exc),
            )
        self.last_activity = self.monotonic()
        self.released_since_activity = True
        return True

    def release_if_idle(self) -> bool:
        if self.repository.runtime_busy():
            self.note_activity()
            return False
        if self.released_since_activity:
            return False
        if (
            self.monotonic() - self.last_activity
            < self.idle_timeout_seconds
        ):
            return False
        unload_loaded_models(release_callbacks=self.release_callbacks)
        self.released_since_activity = True
        return True


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
