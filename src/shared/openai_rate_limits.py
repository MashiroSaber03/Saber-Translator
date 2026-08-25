"""Provider RPM limiting shared by API and Worker processes."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Protocol

from src.shared.ai_providers import normalize_provider_id
from src.shared.user_logging import user_log


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
_wait_log_lock = threading.Lock()
_wait_log_deadlines: dict[str, float] = {}


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
        if isinstance(rpm_limit, bool) or not isinstance(rpm_limit, int) or rpm_limit < 0:
            raise ValueError("rpm_limit 必须是非负整数")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider 必须是非空字符串")
        if credential_version_id is not None and (
            not isinstance(credential_version_id, str)
            or not credential_version_id
        ):
            raise ValueError("credential_version_id 必须是非空字符串或 null")
        self.rpm_limit = rpm_limit
        self.provider = normalize_provider_id(provider)
        self.credential_version_id = credential_version_id

    def wait_sync(self) -> None:
        while True:
            wait_seconds = self._acquire()
            if wait_seconds <= 0:
                return
            self._log_wait(wait_seconds)
            logger.debug(
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
            self._log_wait(wait_seconds)
            logger.debug(
                "RPM 限制(%s): 等待 %.1f 秒",
                self.provider,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)

    def _log_wait(self, wait_seconds: float) -> None:
        now = time.monotonic()
        with _wait_log_lock:
            if now < _wait_log_deadlines.get(self.provider, 0.0):
                return
            _wait_log_deadlines[self.provider] = (
                now + min(60.0, max(1.0, wait_seconds))
            )
        user_log(
            "system",
            f"服务商 {self.provider} 已达到每分钟请求上限，"
            f"等待 {wait_seconds:.1f} 秒后继续",
        )

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
            if not isinstance(decision, RateLimitDecision):
                raise RuntimeError("provider rate-limit store returned an invalid decision")
            if not isinstance(decision.allowed, bool):
                raise RuntimeError("provider rate-limit decision.allowed must be boolean")
            if (
                isinstance(decision.remaining, bool)
                or not isinstance(decision.remaining, int)
                or decision.remaining < 0
            ):
                raise RuntimeError("provider rate-limit decision.remaining is invalid")
            retry_after = decision.retry_after_seconds
            if (
                isinstance(retry_after, bool)
                or not isinstance(retry_after, (int, float))
                or not math.isfinite(float(retry_after))
                or retry_after < 0
            ):
                raise RuntimeError("provider rate-limit retry_after_seconds is invalid")
            if decision.allowed:
                return 0.0
            if retry_after <= 0:
                raise RuntimeError("denied provider rate-limit decision must include a positive retry delay")
            return float(retry_after)
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
