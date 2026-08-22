from __future__ import annotations

import threading
import time

import pytest

from src.backend_v2.operations.executor import DurableOperationExecutor
from src.backend_v2.operations.repository import OperationFence
from src.backend_v2.scheduling_policy import (
    DEFAULT_SCHEDULING_POLICY,
    SchedulingPolicyRepository,
    validate_scheduling_policy,
)
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.lifecycle import initialize_database


def test_scheduling_policy_uses_the_small_machine_defaults_and_closed_schema(
    tmp_path,
) -> None:
    data_root = tmp_path / "data-v2"
    initialized = initialize_database(data_root)
    engine = create_sqlite_engine(initialized.database_path)
    try:
        repository = SchedulingPolicyRepository(engine)
        assert repository.load() == DEFAULT_SCHEDULING_POLICY
        changed = dict(DEFAULT_SCHEDULING_POLICY)
        changed["pageQuantum"] = 2
        changed["minAvailableMemoryMiB"] = 0
        assert repository.save(changed) == changed
        assert repository.load() == changed
    finally:
        engine.dispose()

    with pytest.raises(ValueError, match="字段无效"):
        validate_scheduling_policy({**changed, "extra": True})
    with pytest.raises(ValueError, match="最低可用内存"):
        validate_scheduling_policy({**changed, "minAvailableMemoryMiB": 256})


def test_api_operation_executor_applies_a_changed_limit_to_new_operations() -> None:
    class FakeRepository:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.pending = ["one", "two", "three"]
            self.completed: list[str] = []

        def claim_next(self, **_kwargs):
            with self.lock:
                if not self.pending:
                    return None
                operation_id = self.pending.pop(0)
            return (
                OperationFence(
                    operation_id=operation_id,
                    attempt_id=f"attempt-{operation_id}",
                    executor_epoch_id="api",
                    executor_role="api",
                ),
                {"kind": "light"},
            )

        def complete(self, fence, *, result):
            assert result == {"ok": True}
            with self.lock:
                self.completed.append(fence.operation_id)

        def fail(self, *_args, **_kwargs):
            raise AssertionError("operation should not fail")

    repository = FakeRepository()
    limit = 1
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def handler(_fence, _operation):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        assert release.wait(3)
        with state_lock:
            active -= 1
        return {"ok": True}

    executor = DurableOperationExecutor(
        repository,  # type: ignore[arg-type]
        executor_role="api",
        executor_epoch_id="api",
        handlers={"light": handler},
        max_workers=4,
        concurrency_limit=lambda: limit,
        poll_seconds=0.01,
    )
    executor.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with state_lock:
                if active == 1:
                    break
            time.sleep(0.01)
        with state_lock:
            assert active == 1
            assert maximum_active == 1

        limit = 2
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with state_lock:
                if active == 2:
                    break
            time.sleep(0.01)
        with state_lock:
            assert active == 2
            assert maximum_active == 2
    finally:
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with repository.lock:
                if len(repository.completed) == 3:
                    break
            time.sleep(0.01)
        executor.close()

    assert sorted(repository.completed) == ["one", "three", "two"]
