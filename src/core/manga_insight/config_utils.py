"""
Manga Insight 配置工具

用于加载和保存分析配置。
"""

import logging
from typing import Dict, Any, List, TYPE_CHECKING

from src.shared.config_loader import load_json_config, save_json_config
from .config_models import MangaInsightConfig

if TYPE_CHECKING:
    from .embedding_client import ChatClient

logger = logging.getLogger("MangaInsight.Config")

# 配置文件名
CONFIG_FILENAME = "manga_insight_settings.json"


def validate_config(config: MangaInsightConfig, strict: bool = False) -> List[str]:
    """
    验证配置，返回错误列表

    Args:
        config: 配置对象
        strict: 是否严格模式（True 时会抛出异常）

    Returns:
        List[str]: 错误信息列表
    """
    errors = []
    warnings = []

    # VLM 配置验证（警告级别 - 用户可能还没配置）
    if config.vlm.provider and not config.vlm.api_key:
        warnings.append("VLM 已选择服务商但未配置 API Key")

    # base_url 格式验证（错误级别 - 格式错误会导致请求失败）
    if config.vlm.base_url:
        if not config.vlm.base_url.startswith(("http://", "https://")):
            errors.append("VLM base_url 格式无效，应以 http:// 或 https:// 开头")

    # Embedding 配置验证
    if config.embedding.api_key and not config.embedding.model:
        warnings.append("Embedding 已配置 API Key 但未选择模型")

    if config.embedding.base_url:
        if not config.embedding.base_url.startswith(("http://", "https://")):
            errors.append("Embedding base_url 格式无效，应以 http:// 或 https:// 开头")

    # 批量分析参数验证（错误级别 - 无效参数会导致分析失败）
    if config.analysis.batch.pages_per_batch < 1:
        errors.append("每批页数不能小于 1")
    if config.analysis.batch.pages_per_batch > 20:
        warnings.append("每批页数过大（建议不超过 20），可能导致 Token 超限")

    if config.analysis.batch.context_batch_count < 0:
        errors.append("上下文批次数不能为负数")
    if config.analysis.batch.context_batch_count > 10:
        warnings.append("上下文批次数过大（建议不超过 10）")

    # VLM 参数验证
    if config.vlm.temperature < 0 or config.vlm.temperature > 2:
        errors.append("VLM temperature 应在 0-2 之间")

    if config.vlm.rpm_limit < 0:
        errors.append("VLM rpm_limit 不能为负数")

    if config.vlm.max_images_per_request < 1:
        errors.append("VLM max_images_per_request 不能小于 1")

    # 记录警告
    for warning in warnings:
        logger.warning(f"配置警告: {warning}")

    # 严格模式：有错误时抛出异常
    if strict and errors:
        raise ValueError(f"配置验证失败: {'; '.join(errors)}")

    return errors + warnings


def load_insight_config(strict: bool = False) -> MangaInsightConfig:
    """
    加载 Manga Insight 配置

    Args:
        strict: 是否严格模式（True 时配置错误会抛出异常）

    Returns:
        MangaInsightConfig: 配置对象

    Raises:
        ValueError: 严格模式下配置验证失败时抛出
    """
    try:
        data = load_json_config(CONFIG_FILENAME, default_value={})
        config = MangaInsightConfig.from_dict(data)

        # 验证配置（严格模式会抛出异常）
        issues = validate_config(config, strict=strict)

        # 非严格模式下记录错误
        if not strict:
            for issue in issues:
                if "不能" in issue or "无效" in issue or "应在" in issue:
                    logger.error(f"配置错误: {issue}")

        return config
    except ValueError:
        # 严格模式的验证异常，直接抛出
        raise
    except Exception as e:
        logger.error(f"加载配置失败: {e}", exc_info=True)
        return MangaInsightConfig()


def save_insight_config(config: MangaInsightConfig) -> bool:
    """
    保存 Manga Insight 配置
    
    Args:
        config: 配置对象或字典
    
    Returns:
        bool: 是否保存成功
    """
    try:
        if isinstance(config, MangaInsightConfig):
            data = config.to_dict()
        elif isinstance(config, dict):
            data = config
        else:
            logger.error(f"无效的配置类型: {type(config)}")
            return False
        
        success = save_json_config(CONFIG_FILENAME, data)
        if success:
            logger.debug("成功保存 Manga Insight 配置")
        return success
    except Exception as e:
        logger.error(f"保存配置失败: {e}", exc_info=True)
        return False


def get_vlm_config_for_provider(provider: str, full_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取指定服务商的 VLM 配置
    
    Args:
        provider: 服务商名称
        full_config: 完整配置字典（如未提供则从文件加载）
    
    Returns:
        Dict: 服务商配置
    """
    if full_config is None:
        full_config = load_json_config(CONFIG_FILENAME, default_value={})
    
    vlm_config = full_config.get("vlm", {})
    providers = vlm_config.get("providers", {})
    return providers.get(provider, {})


def get_embedding_config_for_provider(provider: str, full_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取指定服务商的 Embedding 配置
    
    Args:
        provider: 服务商名称
        full_config: 完整配置字典
    
    Returns:
        Dict: 服务商配置
    """
    if full_config is None:
        full_config = load_json_config(CONFIG_FILENAME, default_value={})
    
    embedding_config = full_config.get("embedding", {})
    providers = embedding_config.get("providers", {})
    return providers.get(provider, {})


def get_current_vlm_provider(full_config: Dict[str, Any] = None) -> str:
    """获取当前选择的 VLM 服务商"""
    if full_config is None:
        full_config = load_json_config(CONFIG_FILENAME, default_value={})
    
    vlm_config = full_config.get("vlm", {})
    return vlm_config.get("current_provider", "gemini")


def get_current_embedding_provider(full_config: Dict[str, Any] = None) -> str:
    """获取当前选择的 Embedding 服务商"""
    if full_config is None:
        full_config = load_json_config(CONFIG_FILENAME, default_value={})
    
    embedding_config = full_config.get("embedding", {})
    return embedding_config.get("current_provider", "openai")


def update_provider_config(
    config_type: str,
    provider: str,
    provider_config: Dict[str, Any]
) -> bool:
    """
    更新指定服务商的配置
    
    Args:
        config_type: 配置类型 ("vlm", "embedding", "reranker")
        provider: 服务商名称
        provider_config: 服务商配置
    
    Returns:
        bool: 是否成功
    """
    try:
        full_config = load_json_config(CONFIG_FILENAME, default_value={})
        
        if config_type not in full_config:
            full_config[config_type] = {"providers": {}}
        
        if "providers" not in full_config[config_type]:
            full_config[config_type]["providers"] = {}
        
        full_config[config_type]["providers"][provider] = provider_config
        
        return save_json_config(CONFIG_FILENAME, full_config)
    except Exception as e:
        logger.error(f"更新服务商配置失败: {e}", exc_info=True)
        return False


def set_current_provider(config_type: str, provider: str) -> bool:
    """
    设置当前选择的服务商

    Args:
        config_type: 配置类型 ("vlm", "embedding", "reranker")
        provider: 服务商名称

    Returns:
        bool: 是否成功
    """
    try:
        full_config = load_json_config(CONFIG_FILENAME, default_value={})

        if config_type not in full_config:
            full_config[config_type] = {}

        full_config[config_type]["current_provider"] = provider

        return save_json_config(CONFIG_FILENAME, full_config)
    except Exception as e:
        logger.error(f"设置当前服务商失败: {e}", exc_info=True)
        return False


def create_chat_client(config: MangaInsightConfig) -> "ChatClient":
    """
    创建 ChatClient 实例的工厂函数

    根据配置决定使用 VLM 配置还是独立的 LLM 配置。
    消除多处重复的 use_same_as_vlm 判断逻辑。

    Args:
        config: MangaInsightConfig 配置对象

    Returns:
        ChatClient: 聊天客户端实例
    """
    from .embedding_client import ChatClient

    if config.chat_llm.use_same_as_vlm:
        return ChatClient(config.vlm)
    return ChatClient(config.chat_llm)
