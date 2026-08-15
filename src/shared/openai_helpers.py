"""
OpenAI 客户端辅助函数

提供创建 OpenAI 客户端的唯一工厂入口，统一处理：
- 本地服务的代理旁路（trust_env=False）
- 浏览器 UA 伪装（绕过套 CF 中转站的 WAF UA 黑名单）

网络策略细节集中在 src.shared.http_config 中维护。
"""
import logging
import math
from typing import Optional

import httpx
from openai import OpenAI

from src.shared.http_config import (
    BROWSER_HEADERS,
    build_httpx_kwargs,
    is_local_service,
)

logger = logging.getLogger(__name__)

LOCAL_OPENAI_COMPATIBLE_API_KEY_PLACEHOLDER = "ollama"

__all__ = ["create_openai_client", "resolve_openai_api_key"]


def resolve_openai_api_key(api_key: Optional[str], base_url: Optional[str] = None) -> str:
    if api_key is not None and not isinstance(api_key, str):
        raise TypeError("api_key 必须是字符串或 null")
    if base_url is not None and not isinstance(base_url, str):
        raise TypeError("base_url 必须是字符串或 null")
    normalized = api_key.strip() if api_key is not None else ""
    if normalized:
        return normalized
    if is_local_service(base_url):
        return LOCAL_OPENAI_COMPATIBLE_API_KEY_PLACEHOLDER
    return ""


def create_openai_client(
    api_key: Optional[str],
    base_url: Optional[str] = None,
    timeout: float = 30.0,
    bypass_proxy: bool = False,
) -> OpenAI:
    """
    创建 OpenAI 客户端（统一注入代理策略与浏览器伪装头）。

    - 本地服务自动禁用代理，避免系统代理干扰回环访问。
    - 远程服务保留系统代理，并注入 Chrome UA。
    - 通过 `default_headers` 二次注入，以覆盖 OpenAI SDK 默认的 `User-Agent`。
    """
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ValueError("base_url 必须是非空字符串或 null")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("timeout 必须是正有限数")
    if not isinstance(bypass_proxy, bool):
        raise ValueError("bypass_proxy 必须是布尔值")

    resolved_api_key = resolve_openai_api_key(api_key, base_url)
    http_options = build_httpx_kwargs(base_url, timeout)
    if bypass_proxy:
        http_options["trust_env"] = False
    http_client = httpx.Client(**http_options)

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=base_url,
        http_client=http_client,
        default_headers=dict(BROWSER_HEADERS),
        max_retries=0,
    )

    if is_local_service(base_url):
        logger.debug("已创建无代理 OpenAI 客户端: %s", base_url)
    else:
        logger.debug("已创建 OpenAI 客户端: %s", base_url or "默认")

    return client
