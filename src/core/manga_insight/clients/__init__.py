"""
Manga Insight 客户端模块

提供统一的 API 客户端基础设施。
"""

from .provider_registry import get_base_url, get_default_model, get_rerank_url, get_image_gen_base_url
from .base_client import BaseAPIClient, RPMLimiter
from .image_gen_client import ImageGenClient

__all__ = [
    "get_base_url",
    "get_default_model",
    "get_rerank_url",
    "get_image_gen_base_url",
    "BaseAPIClient",
    "RPMLimiter",
    "ImageGenClient",
]
