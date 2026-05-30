"""
Manga Insight API 客户端基类

当前主要为仍未迁移到共享 transport 的客户端提供：
- HTTP 客户端管理
- 基础请求头
- 生命周期关闭
- 可复用的 RPM 限制器
"""

import logging
from typing import Dict, Optional

import httpx

from .provider_registry import get_base_url
from src.shared.openai_rate_limits import SharedRPMLimiter

logger = logging.getLogger("MangaInsight.BaseClient")

RPMLimiter = SharedRPMLimiter


class BaseAPIClient:
    """
    API 客户端基类

    提供统一的 HTTP 客户端管理与基础工具。
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        resolved_base_url: Optional[str] = None,
        rpm_limit: int = 0,
        timeout: float = 120.0,
    ):
        """
        初始化基础客户端

        Args:
            provider: 服务商名称
            api_key: API 密钥
            base_url: 自定义 base_url（仅 custom 服务商需要）
            resolved_base_url: 已解析的完整 base_url（优先级高于 base_url）
            rpm_limit: RPM 限制
            timeout: 请求超时时间（秒）
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self._base_url = resolved_base_url if resolved_base_url is not None else get_base_url(provider, base_url)
        self._rpm_limiter = RPMLimiter(rpm_limit)
        self._timeout = timeout

        # 创建 HTTP 客户端
        self.client = self._create_http_client()

    def _create_http_client(self) -> httpx.AsyncClient:
        """
        创建 HTTP 客户端

        统一经由 src.shared.http_config 注入：
        - 本地服务禁用代理
        - 远程服务保留系统代理
        - 全场景注入浏览器 UA（绕过套 CF 中转站的 WAF UA 黑名单）
        """
        from src.shared.http_config import build_httpx_kwargs, is_local_service

        if is_local_service(self._base_url):
            logger.info(f"检测到本地服务 ({self._base_url})，禁用代理")
        return httpx.AsyncClient(**build_httpx_kwargs(self._base_url, self._timeout))

    @property
    def base_url(self) -> str:
        """获取 base_url"""
        return self._base_url

    async def close(self):
        """关闭客户端。

        httpx.AsyncClient 绑定在创建它的 event loop 上。Flask 路由经由 `run_async`
        每次新建临时 loop 执行协程，build/查询阶段的 loop 销毁后再调用 close()，
        aclose() 会抛 "Event loop is closed"。此时底层 socket 已随旧 loop 被系统
        回收，无资源泄漏，吞掉该错误即可。其他错误仍原样抛出。
        """
        try:
            await self.client.aclose()
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                logger.debug("aclose 跨 loop（可忽略）: %s", exc)
                return
            raise

    async def __aenter__(self):
        """上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        await self.close()

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _enforce_rpm_limit(self):
        """执行 RPM 限制"""
        await self._rpm_limiter.wait()
