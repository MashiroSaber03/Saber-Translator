"""
HTTP 请求重试工具

统一 requests 的超时与指数退避重试策略。
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import requests

DEFAULT_TIMEOUT: Tuple[float, float] = (10.0, 120.0)
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_RETRY_STATUS = (429, 500, 502, 503, 504)


def _sleep_with_backoff(attempt: int, base: float) -> None:
    time.sleep(base * (2 ** attempt))


def post_with_retry(
    url: str,
    *,
    headers=None,
    params=None,
    data=None,
    json=None,
    timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    retry_status: Iterable[int] = DEFAULT_RETRY_STATUS,
):
    """带超时与指数退避的 POST 请求。"""
    retry_status = set(retry_status)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                data=data,
                json=json,
                timeout=timeout,
            )
            if response.status_code in retry_status and attempt < max_retries - 1:
                _sleep_with_backoff(attempt, backoff_base)
                continue
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                raise
            _sleep_with_backoff(attempt, backoff_base)

    if last_error:
        raise last_error
    raise RuntimeError("HTTP POST 重试失败")


def get_with_retry(
    url: str,
    *,
    headers=None,
    params=None,
    timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    retry_status: Iterable[int] = DEFAULT_RETRY_STATUS,
):
    """带超时与指数退避的 GET 请求。"""
    retry_status = set(retry_status)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
            if response.status_code in retry_status and attempt < max_retries - 1:
                _sleep_with_backoff(attempt, backoff_base)
                continue
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                raise
            _sleep_with_backoff(attempt, backoff_base)

    if last_error:
        raise last_error
    raise RuntimeError("HTTP GET 重试失败")

