from __future__ import annotations

from pathlib import Path
import uuid

from flask import Flask
import pytest
from sqlalchemy import insert, select

from src.backend_v2.api.system_routes import create_system_blueprint
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    metadata,
    operations,
    pages,
    worker_commands,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.worker.model_lifecycle import (
    WorkerModelControlRepository,
    WorkerModelLifecycle,
)


@pytest.fixture()
def model_platform(tmp_path: Path):
    engine = create_sqlite_engine(tmp_path / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    page_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            insert(pages).values(
                id=page_id,
                chapter_id=str(chapter["id"]),
                ordinal=1,
                logical_source_path="page.png",
            )
        )
    worker_epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            epoch_id=worker_epoch_id,
            token="worker-token",
            role="worker",
            pid=771,
        )
    )
    try:
        yield engine, worker_epoch_id, page_id
    finally:
        engine.dispose()


def test_manual_model_release_is_durable_and_worker_fenced(
    model_platform,
    monkeypatch,
) -> None:
    engine, worker_epoch_id, _page_id = model_platform
    repository = WorkerModelControlRepository(engine)
    accepted = repository.request_release()
    assert accepted["status"] == "pending"
    assert repository.request_release()["commandId"] == accepted["commandId"]

    released: list[str] = []
    monkeypatch.setattr(
        "src.backend_v2.worker.model_lifecycle.unload_loaded_models",
        lambda *, release_callbacks: _run_release_callbacks(release_callbacks),
    )
    lifecycle = WorkerModelLifecycle(
        repository,
        worker_epoch_id=worker_epoch_id,
        release_callbacks=(lambda: released.append("plugins"),),
    )
    assert lifecycle.run_pending_release() is True
    assert released == ["plugins"]
    with engine.connect() as connection:
        command = connection.execute(
            select(worker_commands).where(
                worker_commands.c.id == accepted["commandId"]
            )
        ).mappings().one()
    assert command["status"] == "completed"
    assert command["worker_epoch_id"] == worker_epoch_id
    assert '"releasedCount":1' in str(command["result_json"])


def test_release_endpoint_returns_409_during_local_model_inference(
    model_platform,
) -> None:
    engine, worker_epoch_id, page_id = model_platform
    operation_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            insert(operations).values(
                id=operation_id,
                kind="page_detect",
                executor_role="worker",
                status="running",
                page_id=page_id,
                base_revision=1,
                request_json="{}",
                executor_epoch_id=worker_epoch_id,
                attempt_id=str(uuid.uuid4()),
                lease_token="lease",
            )
        )
    app = Flask("model-control-test")
    app.register_blueprint(create_system_blueprint(engine=engine))

    response = app.test_client().post(
        "/api/v2/system/release-models"
    )

    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "model_inference_busy"


def test_idle_model_cache_is_released_once_after_ten_minutes(
    model_platform,
    monkeypatch,
) -> None:
    engine, worker_epoch_id, _page_id = model_platform
    clock = [0.0]
    released: list[str] = []
    monkeypatch.setattr(
        "src.backend_v2.worker.model_lifecycle.unload_loaded_models",
        lambda *, release_callbacks: _run_release_callbacks(release_callbacks),
    )
    lifecycle = WorkerModelLifecycle(
        WorkerModelControlRepository(engine),
        worker_epoch_id=worker_epoch_id,
        idle_timeout_seconds=600,
        release_callbacks=(lambda: released.append("plugins"),),
        monotonic=lambda: clock[0],
    )
    runtime_checks = 0

    def runtime_busy() -> bool:
        nonlocal runtime_checks
        runtime_checks += 1
        return False

    monkeypatch.setattr(lifecycle.repository, "runtime_busy", runtime_busy)

    clock[0] = 599
    assert lifecycle.release_if_idle() is False
    assert runtime_checks == 0
    clock[0] = 600
    assert lifecycle.release_if_idle() is True
    assert runtime_checks == 1
    assert lifecycle.release_if_idle() is False
    assert runtime_checks == 1
    assert released == ["plugins"]


def _run_release_callbacks(release_callbacks) -> dict[str, object]:
    for callback in release_callbacks:
        callback()
    return {"released": ["runtime_cache_1"], "releasedCount": 1}
