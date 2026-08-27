"""Small, administrator-owned scheduling policy for the public deployment."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import threading
import time
from typing import Any, Callable

import psutil
from sqlalchemy import Engine, func, select, update

from src.backend_v2.serialization import canonical_json
from src.backend_v2.storage.database import is_sqlite_busy_error
from src.backend_v2.storage.schema import (
    DEFAULT_SCHEDULING_POLICY_JSON,
    EXECUTING_JOB_STATUSES,
    jobs,
    platform_config,
    process_epochs,
    queue_state,
    users,
)
from src.backend_v2.timestamps import iso_utc, utcnow


POLICY_KEYS = {
    "queueDiscipline",
    "pageQuantum",
    "interactiveBurst",
    "maxDeepLearningConcurrency",
    "apiOperationConcurrency",
    "modelIdleSeconds",
    "minAvailableMemoryMiB",
}
QUEUE_DISCIPLINES = {"owner_round_robin", "fifo"}


def _integer(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{label} 必须是 {minimum} 到 {maximum} 之间的整数")
    return value


def validate_scheduling_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != POLICY_KEYS:
        raise ValueError("调度策略字段无效")
    discipline = value["queueDiscipline"]
    if discipline not in QUEUE_DISCIPLINES:
        raise ValueError("队列规则必须是按用户轮转或先进先出")
    minimum_memory = value["minAvailableMemoryMiB"]
    if isinstance(minimum_memory, bool) or not isinstance(minimum_memory, int):
        raise ValueError("最低可用内存必须是整数")
    if minimum_memory != 0 and not 512 <= minimum_memory <= 1_048_576:
        raise ValueError("最低可用内存必须为 0，或 512 到 1048576 MiB")
    return {
        "queueDiscipline": str(discipline),
        "pageQuantum": _integer(
            value["pageQuantum"], label="每轮页数", minimum=1, maximum=20
        ),
        "interactiveBurst": _integer(
            value["interactiveBurst"],
            label="交互操作插队数",
            minimum=0,
            maximum=3,
        ),
        "maxDeepLearningConcurrency": _integer(
            value["maxDeepLearningConcurrency"],
            label="深度学习并发数",
            minimum=1,
            maximum=8,
        ),
        "apiOperationConcurrency": _integer(
            value["apiOperationConcurrency"],
            label="轻量操作并发数",
            minimum=1,
            maximum=8,
        ),
        "modelIdleSeconds": _integer(
            value["modelIdleSeconds"],
            label="模型空闲释放时间",
            minimum=60,
            maximum=3600,
        ),
        "minAvailableMemoryMiB": minimum_memory,
    }


DEFAULT_SCHEDULING_POLICY = validate_scheduling_policy(
    json.loads(DEFAULT_SCHEDULING_POLICY_JSON)
)


def available_memory_mib() -> int:
    return int(psutil.virtual_memory().available // (1024 * 1024))


class SchedulingPolicyRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            payload = connection.execute(
                select(platform_config.c.scheduler_policy_json).where(
                    platform_config.c.singleton_id == 1
                )
            ).scalar_one()
        try:
            value = json.loads(str(payload))
        except json.JSONDecodeError as exc:
            raise RuntimeError("调度策略不是有效 JSON") from exc
        return validate_scheduling_policy(value)

    def save(self, value: object) -> dict[str, Any]:
        policy = validate_scheduling_policy(value)
        with self.engine.begin() as connection:
            connection.execute(
                update(platform_config)
                .where(platform_config.c.singleton_id == 1)
                .values(
                    scheduler_policy_json=canonical_json(policy),
                    updated_at=utcnow(),
                )
            )
        return policy

    def overview(self) -> dict[str, Any]:
        policy = self.load()
        memory = psutil.virtual_memory()
        now = utcnow()
        with self.engine.connect() as connection:
            worker_online = connection.execute(
                select(process_epochs.c.id)
                .where(
                    process_epochs.c.role == "worker",
                    process_epochs.c.status == "active",
                    process_epochs.c.lease_expires_at > now,
                )
                .limit(1)
            ).scalar_one_or_none() is not None
            current = connection.execute(
                select(
                    jobs.c.id,
                    jobs.c.kind,
                    jobs.c.status,
                    jobs.c.started_at,
                    jobs.c.owner_user_id,
                    users.c.username,
                )
                .join(users, users.c.id == jobs.c.owner_user_id)
                .where(jobs.c.status.in_(EXECUTING_JOB_STATUSES))
                .limit(1)
            ).mappings().one_or_none()
            queued_jobs = int(
                connection.execute(
                    select(func.count()).select_from(jobs).where(jobs.c.status == "queued")
                ).scalar_one()
            )
            ready_jobs = int(
                connection.execute(
                    select(func.count())
                    .select_from(jobs)
                    .where(
                        jobs.c.status == "queued",
                        jobs.c.blocked_by_job_id.is_(None),
                    )
                ).scalar_one()
            )
            queue_paused = bool(
                connection.execute(
                    select(queue_state.c.admission_paused).where(
                        queue_state.c.singleton_id == 1
                    )
                ).scalar_one()
            )
            queued_users = int(
                connection.execute(
                    select(func.count(func.distinct(jobs.c.owner_user_id))).where(
                        jobs.c.status == "queued"
                    )
                ).scalar_one()
            )
            paused_jobs = int(
                connection.execute(
                    select(func.count()).select_from(jobs).where(jobs.c.status == "paused")
                ).scalar_one()
            )

        available = int(memory.available // (1024 * 1024))
        threshold = int(policy["minAvailableMemoryMiB"])
        waiting_reason: str | None = None
        if queued_jobs and queue_paused:
            waiting_reason = "queue_paused"
        elif current is None and queued_jobs:
            if not worker_online:
                waiting_reason = "worker_offline"
            elif threshold and available < threshold:
                waiting_reason = "low_memory"
            elif ready_jobs == 0:
                waiting_reason = "queue_blocked"
        current_task = None
        if current is not None:
            current_task = {
                "jobId": str(current["id"]),
                "kind": str(current["kind"]),
                "status": str(current["status"]),
                "ownerUserId": str(current["owner_user_id"]),
                "ownerUsername": str(current["username"]),
                "startedAt": iso_utc(current["started_at"]),
            }
        return {
            "policy": policy,
            "status": {
                "workerOnline": worker_online,
                "currentTask": current_task,
                "queuedJobCount": queued_jobs,
                "queuedUserCount": queued_users,
                "pausedJobCount": paused_jobs,
                "availableMemoryMiB": available,
                "totalMemoryMiB": int(memory.total // (1024 * 1024)),
                "waitingReason": waiting_reason,
            },
        }


class SchedulingPolicyCache:
    """Tiny cross-request cache; SQLite remains the source of truth."""

    def __init__(
        self,
        repository: SchedulingPolicyRepository,
        *,
        ttl_seconds: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repository = repository
        self.ttl_seconds = max(0.0, ttl_seconds)
        self.monotonic = monotonic
        self._lock = threading.Lock()
        self._expires_at = 0.0
        self._value: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        now = self.monotonic()
        with self._lock:
            if self._value is None or now >= self._expires_at:
                try:
                    self._value = self.repository.load()
                except Exception as exc:
                    if self._value is None or not is_sqlite_busy_error(exc):
                        raise
                    self._expires_at = now + min(self.ttl_seconds, 0.5)
                    return deepcopy(self._value)
                self._expires_at = now + self.ttl_seconds
            return deepcopy(self._value)
