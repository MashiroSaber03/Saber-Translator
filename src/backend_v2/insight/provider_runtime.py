"""Strict readers for the current frozen Insight provider contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

from src.backend_v2.insight.repository import InsightConflict
from src.shared.ai_providers import get_provider_manifest, provider_requires_api_key

if TYPE_CHECKING:
    from src.core.manga_insight.config_models import (
        ChatLLMConfig,
        EmbeddingConfig,
        ImageGenConfig,
        RerankerConfig,
        VLMConfig,
    )

_CREDENTIAL_FIELDS = {"api_key", "credential_version_id"}


def _section(
    config: Mapping[str, Any],
    name: str,
    required_fields: set[str],
) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise InsightConflict(f"frozen Insight {name} config must be an object")
    actual = set(value)
    missing = required_fields - actual
    unknown = actual - required_fields - _CREDENTIAL_FIELDS
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise InsightConflict(
            f"frozen Insight {name} config fields are invalid "
            f"({'; '.join(details)})"
        )
    return dict(value)


def _credential_values(
    section: Mapping[str, Any],
    *,
    name: str,
) -> tuple[str, str | None]:
    provider = section.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise InsightConflict(f"frozen Insight {name} provider is invalid")
    try:
        get_provider_manifest(provider)
    except (TypeError, ValueError) as exc:
        raise InsightConflict(
            f"frozen Insight {name} provider is invalid"
        ) from exc

    api_key = section.get("api_key", "")
    if not isinstance(api_key, str):
        raise InsightConflict(f"frozen Insight {name} api_key must be a string")
    base_url = section.get("custom_base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise InsightConflict(f"frozen Insight {name} base URL is invalid")
    if provider_requires_api_key(provider, base_url) and not api_key.strip():
        raise InsightConflict(f"frozen Insight {name} credential is missing")

    credential_version_id = section.get("credential_version_id")
    if credential_version_id is not None and (
        not isinstance(credential_version_id, str)
        or not credential_version_id.strip()
    ):
        raise InsightConflict(
            f"frozen Insight {name} credential version is invalid"
        )
    return api_key, credential_version_id


def _parse_config(factory, payload: dict[str, Any], *, name: str):
    try:
        return factory.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise InsightConflict(f"frozen Insight {name} config is invalid") from exc


def frozen_vlm_config(config: Mapping[str, Any]) -> VLMConfig:
    from src.core.manga_insight.config_models import VLMConfig

    section = _section(
        config,
        "vlm",
        {
            "provider",
            "model_name",
            "custom_base_url",
            "openai_options",
            "image_max_size",
        },
    )
    api_key, credential_version_id = _credential_values(section, name="vlm")
    return _parse_config(
        VLMConfig,
        {
            "provider": section["provider"],
            "api_key": api_key,
            "model": section["model_name"],
            "base_url": section["custom_base_url"],
            "credential_version_id": credential_version_id,
            "openai_options": section["openai_options"],
            "image_max_size": section["image_max_size"],
        },
        name="vlm",
    )


def frozen_chat_config(config: Mapping[str, Any]) -> ChatLLMConfig:
    from src.core.manga_insight.config_models import ChatLLMConfig

    section = _section(
        config,
        "chat",
        {"provider", "model_name", "custom_base_url", "openai_options"},
    )
    api_key, credential_version_id = _credential_values(section, name="chat")
    return _parse_config(
        ChatLLMConfig,
        {
            "provider": section["provider"],
            "api_key": api_key,
            "model": section["model_name"],
            "base_url": section["custom_base_url"],
            "credential_version_id": credential_version_id,
            "openai_options": section["openai_options"],
        },
        name="chat",
    )


def frozen_embedding_config(config: Mapping[str, Any]) -> EmbeddingConfig:
    from src.core.manga_insight.config_models import EmbeddingConfig

    section = _section(
        config,
        "embedding",
        {
            "provider",
            "model_name",
            "custom_base_url",
            "rpm_limit",
            "transport_retries",
            "business_retries",
            "timeout_seconds",
        },
    )
    api_key, credential_version_id = _credential_values(
        section,
        name="embedding",
    )
    return _parse_config(
        EmbeddingConfig,
        {
            "provider": section["provider"],
            "api_key": api_key,
            "model": section["model_name"],
            "base_url": section["custom_base_url"],
            "credential_version_id": credential_version_id,
            "rpm_limit": section["rpm_limit"],
            "transport_retries": section["transport_retries"],
            "business_retries": section["business_retries"],
            "timeout_seconds": section["timeout_seconds"],
        },
        name="embedding",
    )


def frozen_reranker_config(config: Mapping[str, Any]) -> RerankerConfig:
    from src.core.manga_insight.config_models import RerankerConfig

    section = _section(
        config,
        "reranker",
        {
            "provider",
            "model_name",
            "custom_base_url",
            "transport_retries",
            "business_retries",
            "timeout_seconds",
        },
    )
    api_key, credential_version_id = _credential_values(
        section,
        name="reranker",
    )
    return _parse_config(
        RerankerConfig,
        {
            "provider": section["provider"],
            "api_key": api_key,
            "model": section["model_name"],
            "base_url": section["custom_base_url"],
            "credential_version_id": credential_version_id,
            "transport_retries": section["transport_retries"],
            "business_retries": section["business_retries"],
            "timeout_seconds": section["timeout_seconds"],
        },
        name="reranker",
    )


def frozen_image_gen_config(config: Mapping[str, Any]) -> ImageGenConfig:
    from src.core.manga_insight.config_models import ImageGenConfig

    section = _section(
        config,
        "imageGen",
        {
            "provider",
            "model_name",
            "custom_base_url",
            "transport_retries",
            "business_retries",
            "timeout_seconds",
        },
    )
    api_key, credential_version_id = _credential_values(
        section,
        name="imageGen",
    )
    return _parse_config(
        ImageGenConfig,
        {
            "provider": section["provider"],
            "api_key": api_key,
            "model": section["model_name"],
            "base_url": section["custom_base_url"],
            "credential_version_id": credential_version_id,
            "transport_retries": section["transport_retries"],
            "business_retries": section["business_retries"],
            "timeout_seconds": section["timeout_seconds"],
        },
        name="imageGen",
    )
