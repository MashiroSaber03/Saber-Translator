"""
Manga Insight 服务商兼容 facade。

仅保留当前仓库内仍在使用的最小接口：
- chat/base_url 解析
- image_gen base_url 解析
- rerank URL 解析
- 默认模型读取
"""

from __future__ import annotations

from typing import Optional

from src.shared.ai_providers import (
    CHAT_CAPABILITY,
    IMAGE_GEN_CAPABILITY,
    RERANK_CAPABILITY,
    get_provider_default_model,
    resolve_provider_base_url_for_capability,
    resolve_provider_endpoint_for_capability,
)


def get_base_url(provider: str, custom_url: Optional[str] = None) -> str:
    """获取服务商 chat/base_url。"""
    return resolve_provider_base_url_for_capability(provider, CHAT_CAPABILITY, custom_url) or ""


def get_image_gen_base_url(provider: str, custom_url: Optional[str] = None) -> str:
    """获取生图服务商 base_url。"""
    return resolve_provider_base_url_for_capability(provider, IMAGE_GEN_CAPABILITY, custom_url) or ""


def get_default_model(provider: str, model_type: str = "vlm") -> str:
    """获取服务商默认模型。"""
    return get_provider_default_model(provider, model_type)


def get_rerank_url(provider: str, custom_url: Optional[str] = None) -> str:
    """获取重排序 API 的完整 URL。"""
    provider_lower = provider.lower()
    if provider_lower == "custom":
        return custom_url or ""

    base_url = resolve_provider_base_url_for_capability(provider, RERANK_CAPABILITY, custom_url) or ""
    endpoint = resolve_provider_endpoint_for_capability(provider, RERANK_CAPABILITY) or ""
    if not base_url or not endpoint:
        return ""
    return f"{base_url.rstrip('/')}{endpoint}"
