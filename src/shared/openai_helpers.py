"""
OpenAI 客户端辅助函数

提供创建 OpenAI 客户端的工具函数，解决系统代理干扰本地服务访问的问题
"""
import logging
from typing import Optional
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)


def is_local_service(base_url: Optional[str]) -> bool:
    """
    检测给定的 URL 是否为本地服务
    
    Args:
        base_url: 要检测的 URL
    
    Returns:
        bool: 如果是本地服务返回 True，否则返回 False
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
    创建 OpenAI 客户端（支持自动绕过本地服务的代理）
    
    Args:
        api_key: API密钥
        base_url: 基础URL（可选）
        timeout: 超时时间（秒）
    
    Returns:
        配置好的 OpenAI 客户端实例
    
    Notes:
        - 自动检测是否为本地服务（localhost, 127.0.0.1等）
        - 对于本地服务，强制禁用代理，避免系统代理干扰
        - 对于远程服务，使用系统代理设置
    """
    # 为本地服务创建无代理的 HTTP 客户端
    if is_local_service(base_url):
        logger.info(f"检测到本地服务 ({base_url})，禁用代理以避免连接失败")
        
        # 创建自定义 HTTP 客户端，使用 trust_env=False 禁用代理
        http_client = httpx.Client(
            timeout=timeout,
            trust_env=False,  # 关键：禁用代理（不从环境变量读取）
        )
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        
        logger.debug(f"已创建无代理 OpenAI 客户端: {base_url}")
    else:
        # 远程服务使用默认配置（会使用系统代理）
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        logger.debug(f"已创建 OpenAI 客户端: {base_url or '默认'}")
    
    return client
