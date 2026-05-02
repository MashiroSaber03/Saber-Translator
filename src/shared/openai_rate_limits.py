"""
Shared provider-scoped RPM limiting helpers for OpenAI-compatible flows.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from typing import Callable

from src.shared.ai_providers import normalize_provider_id

logger = logging.getLogger("SharedOpenAIRateLimits")

_CAPABILITY_SERVICE_NAMES = {
    "translation": "Translation",
    "hq_translation": "HQTranslation",
    "vision_ocr": "AIVisionOCR",
    "chat": "MangaInsightChat",
    "vlm": "MangaInsightVLM",
}

_sync_last_reset_by_bucket = defaultdict(lambda: [0.0])
_sync_request_count_by_bucket = defaultdict(lambda: [0])
_sync_lock_by_bucket = defaultdict(threading.Lock)


def build_openai_rpm_bucket_key(capability: str, provider: str) -> str:
    normalized_provider = normalize_provider_id(provider) or "unknown"
    return f"{capability}:{normalized_provider}"


def build_openai_rpm_service_name(capability: str, provider: str) -> str:
    normalized_provider = normalize_provider_id(provider) or "unknown"
    capability_label = _CAPABILITY_SERVICE_NAMES.get(capability, capability)
    return f"{capability_label} ({normalized_provider})"


def enforce_sync_rpm_limit_window(
    rpm_limit: int,
    service_name: str,
    last_reset_time_ref: list,
    request_count_ref: list,
) -> None:
    if rpm_limit <= 0:
        return

    current_time = time.time()

    if current_time - last_reset_time_ref[0] >= 60:
        logger.info("rpm: %s - 1分钟窗口已过，重置计数器和时间。", service_name)
        last_reset_time_ref[0] = current_time
        request_count_ref[0] = 0

    if request_count_ref[0] >= rpm_limit:
        time_to_wait = 60 - (current_time - last_reset_time_ref[0])
        if time_to_wait > 0:
            logger.info(
                "rpm: %s - 已达到每分钟 %s 次请求上限。将等待 %.2f 秒...",
                service_name,
                rpm_limit,
                time_to_wait,
            )
            time.sleep(time_to_wait)
            last_reset_time_ref[0] = time.time()
            request_count_ref[0] = 0
        else:
            logger.info("rpm: %s - 窗口已过但计数未重置，立即重置。", service_name)
            last_reset_time_ref[0] = current_time
            request_count_ref[0] = 0

    if request_count_ref[0] == 0 and last_reset_time_ref[0] == 0:
        last_reset_time_ref[0] = current_time
        logger.info("rpm: %s - 启动新的1分钟请求窗口。", service_name)

    request_count_ref[0] += 1
    logger.debug(
        "rpm: %s - 当前窗口请求计数: %s/%s",
        service_name,
        request_count_ref[0],
        rpm_limit if rpm_limit > 0 else "无限制",
    )


def apply_sync_rpm_limit(
    bucket_key: str,
    rpm_limit: int,
    service_name: str,
    enforcer: Callable[[int, str, list, list], None],
) -> None:
    if rpm_limit <= 0:
        return

    with _sync_lock_by_bucket[bucket_key]:
        enforcer(
            rpm_limit,
            service_name,
            _sync_last_reset_by_bucket[bucket_key],
            _sync_request_count_by_bucket[bucket_key],
        )


class SharedRPMLimiter:
    _last_reset_by_bucket = defaultdict(float)
    _count_by_bucket = defaultdict(int)
    _lock_by_bucket = defaultdict(threading.Lock)

    def __init__(self, rpm_limit: int = 0, bucket_id: str | None = None):
        self.rpm_limit = max(0, int(rpm_limit or 0))
        self.bucket_id = bucket_id
        self._instance_last_reset = 0.0
        self._instance_count = 0
        self._instance_lock = threading.Lock()

    async def wait(self):
        if self.rpm_limit <= 0:
            return

        if not self.bucket_id:
            await self._wait_instance()
            return

        await self._wait_bucket()

    async def _wait_instance(self):
        while True:
            wait_time = 0.0
            with self._instance_lock:
                current_time = time.time()
                if current_time - self._instance_last_reset >= 60:
                    self._instance_last_reset = current_time
                    self._instance_count = 0

                if self._instance_count >= self.rpm_limit:
                    wait_time = max(0.0, 60 - (current_time - self._instance_last_reset))
                else:
                    if self._instance_count == 0 and self._instance_last_reset == 0:
                        self._instance_last_reset = current_time
                    self._instance_count += 1
                    return

            if wait_time > 0:
                logger.info("RPM 限制: 等待 %.1f 秒", wait_time)
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    async def _wait_bucket(self):
        while True:
            wait_time = 0.0
            lock = self._lock_by_bucket[self.bucket_id]
            with lock:
                current_time = time.time()
                last_reset = self._last_reset_by_bucket[self.bucket_id]
                request_count = self._count_by_bucket[self.bucket_id]

                if current_time - last_reset >= 60:
                    last_reset = current_time
                    request_count = 0

                if request_count >= self.rpm_limit:
                    wait_time = max(0.0, 60 - (current_time - last_reset))
                else:
                    if request_count == 0 and last_reset == 0:
                        last_reset = current_time
                    request_count += 1
                    self._last_reset_by_bucket[self.bucket_id] = last_reset
                    self._count_by_bucket[self.bucket_id] = request_count
                    return

                self._last_reset_by_bucket[self.bucket_id] = last_reset
                self._count_by_bucket[self.bucket_id] = request_count

            if wait_time > 0:
                logger.info("RPM 限制(%s): 等待 %.1f 秒", self.bucket_id, wait_time)
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    def reset(self):
        if self.bucket_id:
            lock = self._lock_by_bucket[self.bucket_id]
            with lock:
                self._last_reset_by_bucket[self.bucket_id] = 0.0
                self._count_by_bucket[self.bucket_id] = 0
            return

        self._instance_last_reset = 0.0
        self._instance_count = 0
