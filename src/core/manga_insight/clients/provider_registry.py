"""
Manga Insight 服务商注册表

统一管理所有 API 服务商的配置，消除重复定义。
所有模型类型（VLM、Embedding、Reranker、生图）共用相同的服务商列表。
"""

from typing import Dict, Optional, List

# 统一的服务商配置（单一定义点）
# 所有模型类型共用这套服务商，各服务商根据能力提供不同的 endpoint
PROVIDER_CONFIGS: Dict[str, Dict] = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "chat_endpoint": "/chat/completions",
        "embedding_endpoint": "/embeddings",
        "rerank_endpoint": "/rerank",
        "image_gen_endpoint": "/images/generations",
        "default_vlm_model": "gpt-4o",
        "default_chat_model": "gpt-4o",
        "default_embedding_model": "text-embedding-3-small",
        "default_image_gen_model": "dall-e-3",
        "vlm_models": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
        "image_gen_models": ["dall-e-3", "dall-e-2"],
        # 能力标记
        "supports_vlm": True,
        "supports_embedding": True,
        "supports_rerank": False,
        "supports_image_gen": True,
    },
    "gemini": {
        "name": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "chat_endpoint": "/chat/completions",
        "embedding_endpoint": "/embeddings",
        "rerank_endpoint": None,
        "image_gen_endpoint": None,  # Gemini 暂不支持生图
        "default_vlm_model": "gemini-2.0-flash",
        "default_chat_model": "gemini-2.0-flash",
        "default_embedding_model": "text-embedding-004",
        "default_image_gen_model": "",
        "vlm_models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "image_gen_models": [],
        "supports_vlm": True,
        "supports_embedding": True,
        "supports_rerank": False,
        "supports_image_gen": False,
    },
    "qwen": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "image_gen_base_url": "https://dashscope.aliyuncs.com/api/v1",  # 生图使用不同的 base_url
        "chat_endpoint": "/chat/completions",
        "embedding_endpoint": "/embeddings",
        "rerank_endpoint": "/rerank",
        "image_gen_endpoint": "/services/aigc/text2image/image-synthesis",
        "default_vlm_model": "qwen-vl-max",
        "default_chat_model": "qwen-plus",
        "default_embedding_model": "text-embedding-v3",
        "default_image_gen_model": "wanx-v1",
        "vlm_models": ["qwen-vl-max", "qwen-vl-plus"],
        "image_gen_models": ["wanx-v1", "wanx2.1-t2i-turbo", "wanx2.1-t2i-plus"],
        "supports_vlm": True,
        "supports_embedding": True,
        "supports_rerank": True,
        "supports_image_gen": True,
    },
    "siliconflow": {
        "name": "SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "chat_endpoint": "/chat/completions",
        "embedding_endpoint": "/embeddings",
        "rerank_endpoint": "/rerank",
        "image_gen_endpoint": "/images/generations",
        "default_vlm_model": "",
        "default_chat_model": "",
        "default_embedding_model": "BAAI/bge-m3",
        "default_reranker_model": "BAAI/bge-reranker-v2-m3",
        "default_image_gen_model": "stabilityai/stable-diffusion-3-5-large",
        "vlm_models": [],
        "image_gen_models": [
            "stabilityai/stable-diffusion-3-5-large",
            "stabilityai/stable-diffusion-3-medium",
            "black-forest-labs/FLUX.1-schnell"
        ],
        "supports_vlm": True,
        "supports_embedding": True,
        "supports_rerank": True,
        "supports_image_gen": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "chat_endpoint": "/chat/completions",
        "embedding_endpoint": "/embeddings",
        "rerank_endpoint": "/rerank",
        "image_gen_endpoint": None,  # DeepSeek 暂不支持生图
        "default_vlm_model": "deepseek-chat",
        "default_chat_model": "deepseek-chat",
        "default_embedding_model": "",
        "default_image_gen_model": "",
        "vlm_models": ["deepseek-chat"],
        "image_gen_models": [],
        "supports_vlm": True,
        "supports_embedding": True,
        "supports_rerank": True,
        "supports_image_gen": False,
    },
    "volcano": {
        "name": "火山引擎",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "image_gen_base_url": "https://visual.volcengineapi.com",  # 生图使用不同的 base_url
        "chat_endpoint": "/chat/completions",
        "embedding_endpoint": "/embeddings",
        "rerank_endpoint": "/rerank",
        "image_gen_endpoint": "/v1/images/generations",
        "default_vlm_model": "",
        "default_chat_model": "",
        "default_embedding_model": "",
        "default_image_gen_model": "",
        "vlm_models": [],
        "image_gen_models": [],
        "supports_vlm": True,
        "supports_embedding": True,
        "supports_rerank": True,
        "supports_image_gen": True,
    },
    "jina": {
        "name": "Jina AI",
        "base_url": "https://api.jina.ai/v1",
        "rerank_endpoint": "/rerank",
        "default_reranker_model": "jina-reranker-v2-base-multilingual",
        "supports_vlm": False,
        "supports_embedding": False,
        "supports_rerank": True,
        "supports_image_gen": False,
    },
    "cohere": {
        "name": "Cohere",
        "base_url": "https://api.cohere.ai/v1",
        "rerank_endpoint": "/rerank",
        "default_reranker_model": "rerank-multilingual-v3.0",
        "supports_vlm": False,
        "supports_embedding": False,
        "supports_rerank": True,
        "supports_image_gen": False,
    },
    "custom": {
        "name": "自定义 API",
        "base_url": "",
        "chat_endpoint": "/chat/completions",
        "embedding_endpoint": "/embeddings",
        "rerank_endpoint": "/rerank",
        "image_gen_endpoint": "/images/generations",
        "supports_vlm": True,
        "supports_embedding": True,
        "supports_rerank": True,
        "supports_image_gen": True,
    }
}


def get_provider_config(provider: str) -> Dict:
    """
    获取服务商配置

    Args:
        provider: 服务商名称（不区分大小写）

    Returns:
        Dict: 服务商配置，不存在则返回空字典
    """
    return PROVIDER_CONFIGS.get(provider.lower(), {})


def get_all_providers() -> List[str]:
    """
    获取所有服务商名称列表

    Returns:
        List[str]: 服务商名称列表
    """
    return list(PROVIDER_CONFIGS.keys())


def get_providers_for_capability(capability: str) -> List[str]:
    """
    获取支持指定能力的服务商列表

    Args:
        capability: 能力类型 (vlm, embedding, rerank, image_gen)

    Returns:
        List[str]: 支持该能力的服务商列表
    """
    key = f"supports_{capability}"
    return [
        provider for provider, config in PROVIDER_CONFIGS.items()
        if config.get(key, False)
    ]


def get_base_url(provider: str, custom_url: Optional[str] = None) -> str:
    """
    获取服务商 base_url

    Args:
        provider: 服务商名称
        custom_url: 自定义 URL（仅 custom 服务商使用）

    Returns:
        str: base_url
    """
    provider_lower = provider.lower()
    if provider_lower == "custom":
        return custom_url or ""
    return get_provider_config(provider_lower).get("base_url", "")


def get_image_gen_base_url(provider: str, custom_url: Optional[str] = None) -> str:
    """
    获取生图服务商 base_url（部分服务商生图使用不同的 base_url）

    Args:
        provider: 服务商名称
        custom_url: 自定义 URL

    Returns:
        str: base_url
    """
    provider_lower = provider.lower()
    if provider_lower == "custom":
        return custom_url or ""

    config = get_provider_config(provider_lower)
    # 优先使用专用的 image_gen_base_url，否则使用通用 base_url
    return config.get("image_gen_base_url", config.get("base_url", ""))


def get_default_model(provider: str, model_type: str = "vlm") -> str:
    """
    获取服务商默认模型

    Args:
        provider: 服务商名称
        model_type: 模型类型 (vlm, chat, embedding, reranker, image_gen)

    Returns:
        str: 默认模型名称
    """
    config = get_provider_config(provider)
    key = f"default_{model_type}_model"
    return config.get(key, "")


def get_vlm_models(provider: str) -> List[str]:
    """
    获取服务商支持的 VLM 模型列表

    Args:
        provider: 服务商名称

    Returns:
        List[str]: 模型列表
    """
    return get_provider_config(provider).get("vlm_models", [])


def get_image_gen_models(provider: str) -> List[str]:
    """
    获取服务商支持的生图模型列表

    Args:
        provider: 服务商名称

    Returns:
        List[str]: 模型列表
    """
    return get_provider_config(provider).get("image_gen_models", [])


def get_rerank_url(provider: str, custom_url: Optional[str] = None) -> str:
    """
    获取重排序 API 的完整 URL

    Args:
        provider: 服务商名称
        custom_url: 自定义 URL

    Returns:
        str: 完整的 rerank URL
    """
    provider_lower = provider.lower()
    if provider_lower == "custom":
        return custom_url or ""

    config = get_provider_config(provider_lower)
    base_url = config.get("base_url", "")
    endpoint = config.get("rerank_endpoint", "/rerank")

    if not base_url or not endpoint:
        return ""

    # 处理 URL 拼接
    base_url = base_url.rstrip("/")
    return f"{base_url}{endpoint}"


def get_image_gen_url(provider: str, custom_url: Optional[str] = None) -> str:
    """
    获取生图 API 的完整 URL

    Args:
        provider: 服务商名称
        custom_url: 自定义 URL

    Returns:
        str: 完整的生图 URL
    """
    provider_lower = provider.lower()
    if provider_lower == "custom":
        return custom_url or ""

    config = get_provider_config(provider_lower)
    base_url = get_image_gen_base_url(provider_lower)
    endpoint = config.get("image_gen_endpoint", "/images/generations")

    if not base_url or not endpoint:
        return ""

    base_url = base_url.rstrip("/")
    return f"{base_url}{endpoint}"
