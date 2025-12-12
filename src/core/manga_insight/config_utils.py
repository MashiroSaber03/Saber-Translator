"""
Manga Insight 配置工具

用于加载和保存分析配置。
"""

import logging
from typing import Dict, Any, List

from src.shared.config_loader import load_json_config, save_json_config
from .config_models import MangaInsightConfig

logger = logging.getLogger("MangaInsight.Config")

# 配置文件名
CONFIG_FILENAME = "manga_insight_settings.json"


def validate_config(config: MangaInsightConfig) -> List[str]:
    """
    验证配置，返回错误列表
    
    Args:
        config: 配置对象
    
    Returns:
        List[str]: 错误信息列表
    """
    errors = []
    
    # VLM 配置验证
    if config.vlm.provider and not config.vlm.api_key:
        errors.append("VLM 已选择服务商但未配置 API Key")
    
    if config.vlm.base_url:
        if not config.vlm.base_url.startswith(("http://", "https://")):
            errors.append("VLM base_url 格式无效，应以 http:// 或 https:// 开头")
    
    # Embedding 配置验证
    if config.embedding.api_key and not config.embedding.model:
        errors.append("Embedding 已配置 API Key 但未选择模型")
    
    if config.embedding.base_url:
        if not config.embedding.base_url.startswith(("http://", "https://")):
            errors.append("Embedding base_url 格式无效，应以 http:// 或 https:// 开头")
    
    # 批量分析参数验证
    if config.analysis.batch.pages_per_batch < 1:
        errors.append("每批页数不能小于 1")
    if config.analysis.batch.pages_per_batch > 20:
        errors.append("每批页数过大（建议不超过 20），可能导致 Token 超限")
    
    if config.analysis.batch.context_batch_count < 0:
        errors.append("上下文批次数不能为负数")
    if config.analysis.batch.context_batch_count > 10:
        errors.append("上下文批次数过大（建议不超过 10）")
    
    # VLM 参数验证
    if config.vlm.temperature < 0 or config.vlm.temperature > 2:
        errors.append("VLM temperature 应在 0-2 之间")
    
    if config.vlm.rpm_limit < 0:
        errors.append("VLM rpm_limit 不能为负数")
    
    if config.vlm.max_images_per_request < 1:
        errors.append("VLM max_images_per_request 不能小于 1")
    
    return errors


def load_insight_config() -> MangaInsightConfig:
    """
    加载 Manga Insight 配置
    
    Returns:
        MangaInsightConfig: 配置对象
    """
    try:
        data = load_json_config(CONFIG_FILENAME, default_value={})
        config = MangaInsightConfig.from_dict(data)
        
        # 验证配置
        errors = validate_config(config)
        for error in errors:
            logger.warning(f"配置警告: {error}")
        
        return config
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
