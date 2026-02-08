"""
Manga Insight 客户端模块

提供统一的 API 客户端基础设施。
"""

from .provider_registry import (
    PROVIDER_CONFIGS,
    get_provider_config,
    get_all_providers,
    get_providers_for_capability,
    get_base_url,
    get_default_model,
    get_vlm_models,
    get_image_gen_models,
    get_rerank_url,
    get_image_gen_base_url,
    get_image_gen_url,
)
from .base_client import BaseAPIClient, RPMLimiter

__all__ = [
    "PROVIDER_CONFIGS",
    "get_provider_config",
    "get_all_providers",
    "get_providers_for_capability",
    "get_base_url",
    "get_default_model",
    "get_vlm_models",
    "get_image_gen_models",
    "get_rerank_url",
    "get_image_gen_base_url",
    "get_image_gen_url",
    "BaseAPIClient",
    "RPMLimiter",
]
