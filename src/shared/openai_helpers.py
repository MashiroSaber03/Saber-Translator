"""
OpenAI 客户端辅助函数

提供创建 OpenAI 客户端的唯一工厂入口，统一处理：
- 本地服务的代理旁路（trust_env=False）
- 浏览器 UA 伪装（绕过套 CF 中转站的 WAF UA 黑名单）

网络策略细节集中在 src.shared.http_config 中维护。
"""
import logging
from typing import Optional

import httpx
from openai import OpenAI

from src.shared.http_config import (
    BROWSER_HEADERS,
    build_httpx_kwargs,
    is_local_service,  # 保留 re-export，向后兼容旧 import 点
)

logger = logging.getLogger(__name__)

__all__ = ["create_openai_client", "is_local_service"]


def create_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
) -> OpenAI:
    """
    创建 OpenAI 客户端（统一注入代理策略与浏览器伪装头）。

    - 本地服务自动禁用代理，避免系统代理干扰回环访问。
    - 远程服务保留系统代理，并注入 Chrome UA。
    - 通过 `default_headers` 二次注入，以覆盖 OpenAI SDK 默认的 `User-Agent`。
    """
    http_client = httpx.Client(**build_httpx_kwargs(base_url, timeout))

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        default_headers=dict(BROWSER_HEADERS),
    )

    if is_local_service(base_url):
        logger.debug(f"已创建无代理 OpenAI 客户端: {base_url}")
    else:
        logger.debug(f"已创建 OpenAI 客户端: {base_url or '默认'}")

    return client
