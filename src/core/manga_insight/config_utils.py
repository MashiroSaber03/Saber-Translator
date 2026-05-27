"""
Manga Insight 配置工具

用于加载和保存分析配置。
"""

import copy
import logging
from typing import Dict, Any, List, TYPE_CHECKING

from src.shared.ai_providers import (
    IMAGE_GEN_CAPABILITY,
    provider_requires_api_key,
    provider_supports_capability,
)
from src.shared.config_loader import load_json_config, save_json_config
from .config_models import MangaInsightConfig
from src.shared.openai_options import (
    DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
    OpenAICompatibleOptions,
)

if TYPE_CHECKING:
    from .embedding_client import ChatClient

logger = logging.getLogger("MangaInsight.Config")

# 配置文件名
CONFIG_FILENAME = "manga_insight_settings.json"
CURRENT_SCHEMA_VERSION = 2


def has_provider_credentials(provider: str, api_key: str = "") -> bool:
    return bool(api_key or not provider_requires_api_key(provider))


def has_provider_model_config(provider: str, model: str, api_key: str = "") -> bool:
    return bool(model and has_provider_credentials(provider, api_key))


def _mapping_value(data: Dict[str, Any], *keys: str):
    for key in keys:
        if key in data and data.get(key) is not None:
            return True, data.get(key)
    return False, None


def _migrate_openai_compatible_entry(
    payload: Dict[str, Any],
    *,
    default_options: OpenAICompatibleOptions,
) -> tuple[Dict[str, Any], bool]:
    if not isinstance(payload, dict):
        return payload, False

    migrated = copy.deepcopy(payload)
    options = OpenAICompatibleOptions.from_dict(default_options.to_dict())
    changed = False

    openai_options = migrated.get("openai_options")
    if not isinstance(openai_options, dict):
        openai_options = {}
        changed = "openai_options" in migrated

    request_payload = openai_options.get("request")
    if not isinstance(request_payload, dict):
        request_payload = {}
        if "request" in openai_options:
            changed = True

    execution_payload = openai_options.get("execution")
    if not isinstance(execution_payload, dict):
        execution_payload = {}
        if "execution" in openai_options:
            changed = True

    has_force_json, force_json_value = _mapping_value(
        request_payload,
        "force_json_output",
        "force_json",
        "forceJsonOutput",
        "forceJson",
    )
    if has_force_json:
        options.request.force_json_output = bool(force_json_value)
        changed = True
    else:
        has_force_json, force_json_value = _mapping_value(
            migrated,
            "force_json_output",
            "force_json",
            "forceJsonOutput",
            "forceJson",
        )
        if has_force_json:
            options.request.force_json_output = bool(force_json_value)
            changed = True

    has_temperature, temperature_value = _mapping_value(request_payload, "temperature")
    if has_temperature:
        options.request.temperature = float(temperature_value)
        changed = True
    else:
        has_temperature, temperature_value = _mapping_value(migrated, "temperature")
        if has_temperature:
            options.request.temperature = float(temperature_value)
            changed = True

    has_use_stream, use_stream_value = _mapping_value(
        execution_payload,
        "use_stream",
        "useStream",
    )
    if has_use_stream:
        options.execution.use_stream = bool(use_stream_value)
        changed = True
    else:
        has_use_stream, use_stream_value = _mapping_value(
            migrated,
            "use_stream",
            "useStream",
        )
        if has_use_stream:
            options.execution.use_stream = bool(use_stream_value)
            changed = True

    has_rpm_limit, rpm_limit_value = _mapping_value(
        execution_payload,
        "rpm_limit",
        "rpmLimit",
    )
    if has_rpm_limit:
        options.execution.rpm_limit = max(0, int(rpm_limit_value))
        changed = True
    else:
        has_rpm_limit, rpm_limit_value = _mapping_value(
            migrated,
            "rpm_limit",
            "rpmLimit",
        )
        if has_rpm_limit:
            options.execution.rpm_limit = max(0, int(rpm_limit_value))
            changed = True

    has_transport_retries, transport_retries_value = _mapping_value(
        execution_payload,
        "transport_retries",
        "transportRetries",
    )
    if has_transport_retries:
        options.execution.transport_retries = max(0, int(transport_retries_value))
        changed = True
    else:
        options.execution.transport_retries = default_options.execution.transport_retries

    has_business_retries, business_retries_value = _mapping_value(
        execution_payload,
        "business_retries",
        "businessRetries",
        "max_retries",
        "maxRetries",
    )
    if has_business_retries:
        options.execution.business_retries = max(0, int(business_retries_value))
        changed = True
    else:
        has_business_retries, business_retries_value = _mapping_value(
            migrated,
            "business_retries",
            "businessRetries",
            "max_retries",
            "maxRetries",
        )
        if has_business_retries:
            options.execution.business_retries = max(0, int(business_retries_value))
            changed = True

    migrated["openai_options"] = options.to_dict()

    for legacy_key in (
        "force_json_output",
        "force_json",
        "forceJsonOutput",
        "forceJson",
        "temperature",
        "use_stream",
        "useStream",
        "rpm_limit",
        "rpmLimit",
        "max_retries",
        "maxRetries",
        "transport_retries",
        "transportRetries",
        "business_retries",
        "businessRetries",
    ):
        if legacy_key in migrated:
            migrated.pop(legacy_key, None)
            changed = True

    return migrated, changed


def _migrate_legacy_config_payload(data: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    if not isinstance(data, dict):
        return {}, False

    migrated = copy.deepcopy(data)
    changed = False

    root_defaults = {
        "vlm": OpenAICompatibleOptions.from_dict(
            {
                "request": {"force_json_output": False, "temperature": 0.3},
                "execution": {
                    "use_stream": True,
                    "rpm_limit": 0,
                    "transport_retries": 10,
                    "business_retries": 10,
                },
            }
        ),
        "chat_llm": OpenAICompatibleOptions.from_dict(
            {
                "request": {"force_json_output": False},
                "execution": {
                    "use_stream": True,
                    "rpm_limit": 0,
                    "transport_retries": 10,
                    "business_retries": 10,
                },
            }
        ),
    }

    for field_name, default_options in root_defaults.items():
        section = migrated.get(field_name)
        if isinstance(section, dict):
            migrated_section, section_changed = _migrate_openai_compatible_entry(
                section,
                default_options=default_options,
            )
            migrated[field_name] = migrated_section
            changed = changed or section_changed

    provider_settings = migrated.get("provider_settings")
    if not isinstance(provider_settings, dict):
        provider_settings = migrated.get("providerSettings")

    if isinstance(provider_settings, dict):
        for group_name, default_options in (
            ("vlmProvider", root_defaults["vlm"]),
            ("llmProvider", root_defaults["chat_llm"]),
        ):
            provider_group = provider_settings.get(group_name)
            if not isinstance(provider_group, dict):
                continue
            for provider_name, provider_payload in list(provider_group.items()):
                if not isinstance(provider_payload, dict):
                    continue
                migrated_payload, payload_changed = _migrate_openai_compatible_entry(
                    provider_payload,
                    default_options=default_options,
                )
                provider_group[provider_name] = migrated_payload
                changed = changed or payload_changed

    if migrated.get("schema_version") != CURRENT_SCHEMA_VERSION:
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION
        changed = True

    return migrated, changed


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
    if config.vlm.provider and provider_requires_api_key(config.vlm.provider) and not config.vlm.api_key:
        warnings.append("VLM 已选择服务商但未配置 API Key")

    # base_url 格式验证（错误级别 - 格式错误会导致请求失败）
    if config.vlm.base_url:
        if not config.vlm.base_url.startswith(("http://", "https://")):
            errors.append("VLM base_url 格式无效，应以 http:// 或 https:// 开头")

    # Embedding 配置验证
    if has_provider_credentials(config.embedding.provider, config.embedding.api_key) and not config.embedding.model:
        warnings.append("Embedding 已选择服务商但未选择模型")

    if config.embedding.base_url:
        if not config.embedding.base_url.startswith(("http://", "https://")):
            errors.append("Embedding base_url 格式无效，应以 http:// 或 https:// 开头")

    # ImageGen 配置验证
    if not provider_supports_capability(config.image_gen.provider, IMAGE_GEN_CAPABILITY):
        errors.append(f"生图服务商 '{config.image_gen.provider}' 不支持 image_gen")
    if provider_requires_api_key(config.image_gen.provider) and not config.image_gen.api_key:
        warnings.append("ImageGen 已选择服务商但未配置 API Key")
    if config.image_gen.base_url:
        if not config.image_gen.base_url.startswith(("http://", "https://")):
            errors.append("ImageGen base_url 格式无效，应以 http:// 或 https:// 开头")
    else:
        warnings.append("ImageGen 已选择服务商但未配置 Base URL")

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
    vlm_temperature = config.vlm.openai_options.request.temperature
    if vlm_temperature is not None and (vlm_temperature < 0 or vlm_temperature > 2):
        errors.append("VLM temperature 应在 0-2 之间")

    if config.vlm.openai_options.execution.rpm_limit < 0:
        errors.append("VLM rpm_limit 不能为负数")

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
        migrated_data, migration_changed = _migrate_legacy_config_payload(data)
        config = MangaInsightConfig.from_dict(migrated_data)
        if migration_changed:
            save_insight_config(config)

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
