"""
OpenAI 客户端辅助函数

提供创建 OpenAI 客户端的工具函数，解决系统代理干扰本地服务访问的问题，
并添加了浏览器 User-Agent 伪装，以绕过 Cloudflare 等 WAF (Web 应用防火墙) 的拦截。
"""
import logging
from typing import Optional
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)


def is_local_service(base_url: Optional[str]) -> bool:
    """
    检测给定的 URL 是否为本地服务
    """
    if not base_url:
        return False
    
    base_url_lower = base_url.lower()
    local_indicators = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
    return any(indicator in base_url_lower for indicator in local_indicators)


def create_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    timeout: float = 30.0
) -> OpenAI:
    """
    创建 OpenAI 客户端（支持自动绕过本地服务的代理，并强力绕过 WAF 拦截）
    """
    # ✨ 核心伪装头
    fake_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    if is_local_service(base_url):
        logger.info(f"检测到本地服务 ({base_url})，禁用代理以避免连接失败")
        http_client = httpx.Client(
            timeout=timeout,
            trust_env=False,     # 禁用代理
            headers=fake_headers # 底层 HTTP 伪装
        )
        logger.debug(f"已创建无代理且带伪装的 OpenAI 客户端: {base_url}")
    else:
        http_client = httpx.Client(
            timeout=timeout,
            trust_env=True,      # 允许系统代理
            headers=fake_headers # 底层 HTTP 伪装
        )
        logger.debug(f"已创建带伪装头的 OpenAI 客户端: {base_url or '默认'}")

    # ✨ 关键修复：通过 default_headers 强行覆盖 OpenAI SDK 的默认 User-Agent
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        default_headers=fake_headers  
    )
    
    return client