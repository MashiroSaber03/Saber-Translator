"""Provider RPM limiting shared by API and Worker processes."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import logging
import threading
import time
from typing import Protocol

from src.shared.ai_providers import normalize_provider_id


logger = logging.getLogger("SharedOpenAIRateLimits")


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: float


class ProviderRateLimitStore(Protocol):
    def acquire(
        self,
        *,
        provider: str,
        credential_version_id: str,
        rpm_limit: int,
    ) -> RateLimitDecision: ...


_store: ProviderRateLimitStore | None = None
_local_windows: dict[str, tuple[float, int, int]] = {}
_local_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def configure_provider_rate_limit_store(
    store: ProviderRateLimitStore | None,
) -> None:
    global _store
    _store = store


def build_openai_rpm_service_name(capability: str, provider: str) -> str:
    normalized_provider = normalize_provider_id(provider) or "unknown"
    return f"{capability} ({normalized_provider})"


class SharedRPMLimiter:
    def __init__(
        self,
        rpm_limit: int,
        *,
        provider: str,
        credential_version_id: str | None,
    ) -> None:
        self.rpm_limit = max(0, int(rpm_limit or 0))
        self.provider = normalize_provider_id(provider) or "unknown"
        self.credential_version_id = credential_version_id

    def wait_sync(self) -> None:
        while True:
            wait_seconds = self._acquire()
            if wait_seconds <= 0:
                return
            logger.info(
                "RPM 限制(%s): 等待 %.1f 秒",
                self.provider,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    async def wait(self) -> None:
        while True:
            wait_seconds = self._acquire()
            if wait_seconds <= 0:
                return
            logger.info(
                "RPM 限制(%s): 等待 %.1f 秒",
                self.provider,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)

    def _acquire(self) -> float:
        if self.rpm_limit <= 0:
            return 0.0
        if self.credential_version_id:
            if _store is None:
                raise RuntimeError("provider rate-limit store is not configured")
            decision = _store.acquire(
                provider=self.provider,
                credential_version_id=self.credential_version_id,
                rpm_limit=self.rpm_limit,
            )
            return 0.0 if decision.allowed else decision.retry_after_seconds
        return self._acquire_local()

    def _acquire_local(self) -> float:
        current_time = time.monotonic()
        with _local_locks[self.provider]:
            started_at, count, stored_limit = _local_windows.get(
                self.provider,
                (current_time, 0, self.rpm_limit),
            )
            elapsed = current_time - started_at
            if elapsed >= 60:
                started_at, count, stored_limit = (
                    current_time,
                    0,
                    self.rpm_limit,
                )
            effective_limit = min(stored_limit, self.rpm_limit)
            if count >= effective_limit:
                return max(0.0, 60 - elapsed)
            _local_windows[self.provider] = (
                started_at,
                count + 1,
                effective_limit,
            )
            return 0.0
