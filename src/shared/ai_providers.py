"""
全项目 AI 服务商注册表与能力映射。

该模块以 ai_provider_manifest.json 作为单一真相源，统一维护：
- provider id 规范化
- 能力位
- 默认 / 分能力 base_url
- 分能力 endpoint
- 默认模型与模型清单
- 是否为 OpenAI 兼容 / 本地 / adapter
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from src.shared.http_config import is_local_service


TRANSLATION_CAPABILITY = "translation"
HQ_TRANSLATION_CAPABILITY = "hq_translation"
VISION_OCR_CAPABILITY = "vision_ocr"
MODEL_FETCH_CAPABILITY = "model_fetch"
CONNECTION_TEST_CAPABILITY = "connection_test"
WEB_IMPORT_AGENT_CAPABILITY = "web_import_agent"
PLUGIN_AGENT_CAPABILITY = "plugin_agent"

CHAT_CAPABILITY = "chat"
VLM_CAPABILITY = "vlm"
EMBEDDING_CAPABILITY = "embedding"
RERANK_CAPABILITY = "rerank"
IMAGE_GEN_CAPABILITY = "image_gen"

_CAPABILITY_NAME_MAP = {
    "hqTranslation": HQ_TRANSLATION_CAPABILITY,
    "visionOcr": VISION_OCR_CAPABILITY,
    "modelFetch": MODEL_FETCH_CAPABILITY,
    "connectionTest": CONNECTION_TEST_CAPABILITY,
    "imageGen": IMAGE_GEN_CAPABILITY,
    "webImportAgent": WEB_IMPORT_AGENT_CAPABILITY,
    "pluginAgent": PLUGIN_AGENT_CAPABILITY,
}

_MODEL_TYPE_NAME_MAP = {
    "imageGen": "image_gen",
}

_MANIFEST_FIELDS = frozenset(
    {
        "id",
        "label",
        "kind",
        "defaultBaseUrl",
        "capabilityBaseUrls",
        "capabilityEndpoints",
        "capabilities",
        "requiresApiKey",
        "requiresModel",
        "requiresBaseUrl",
        "isLocal",
        "supportsStream",
        "supportsJsonResponse",
        "defaultModels",
        "modelCatalogs",
    }
)
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "id",
        "label",
        "kind",
        "capabilities",
        "requiresApiKey",
        "requiresModel",
        "requiresBaseUrl",
        "isLocal",
        "supportsStream",
        "supportsJsonResponse",
    }
)
_MANIFEST_CAPABILITIES = frozenset(
    {
        "translation",
        "hqTranslation",
        "visionOcr",
        "modelFetch",
        "connectionTest",
        "webImportAgent",
        "pluginAgent",
        "chat",
        "vlm",
        "embedding",
        "rerank",
        "imageGen",
    }
)
_MANIFEST_MODEL_TYPES = frozenset(
    {"vlm", "chat", "embedding", "reranker", "imageGen"}
)


@dataclass(frozen=True)
class ProviderManifest:
    id: str
    display_name: str
    kind: str  # openai_compatible | local | adapter
    default_base_url: Optional[str] = None
    capabilities: FrozenSet[str] = field(default_factory=frozenset)
    requires_api_key: bool = True
    requires_model: bool = True
    requires_base_url: bool = False
    is_local: bool = False
    supports_stream: bool = False
    supports_json_response: bool = False
    capability_base_urls: Mapping[str, str] = field(default_factory=dict)
    capability_endpoints: Mapping[str, str] = field(default_factory=dict)
    default_models: Mapping[str, str] = field(default_factory=dict)
    model_catalogs: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)


_MANIFEST_PATH = Path(__file__).with_name("ai_provider_manifest.json")


def _load_provider_manifest_data() -> list[dict]:
    with _MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)
    if not isinstance(data, list) or not data:
        raise RuntimeError("ai_provider_manifest.json 必须是非空数组")
    if any(not isinstance(entry, dict) for entry in data):
        raise RuntimeError("ai_provider_manifest.json 每一项都必须是对象")
    ids = []
    for entry in data:
        provider_id = entry.get("id")
        if not isinstance(provider_id, str):
            raise RuntimeError("ai_provider_manifest.json provider id 必须是字符串")
        ids.append(provider_id)
    if len(set(ids)) != len(ids):
        raise RuntimeError("ai_provider_manifest.json provider id 必须唯一")
    return data


def _normalize_capability_name(name: str) -> str:
    return _CAPABILITY_NAME_MAP.get(name, name)


def _normalize_model_type_name(name: str) -> str:
    return _MODEL_TYPE_NAME_MAP.get(name, name)


def _default_capability_endpoints() -> Dict[str, str]:
    return {
        CHAT_CAPABILITY: "/chat/completions",
        EMBEDDING_CAPABILITY: "/embeddings",
        RERANK_CAPABILITY: "/rerank",
        IMAGE_GEN_CAPABILITY: "/images/generations",
    }


def _build_provider_manifest(entry: dict) -> ProviderManifest:
    fields = set(entry)
    missing = _REQUIRED_MANIFEST_FIELDS - fields
    unknown = fields - _MANIFEST_FIELDS
    if missing or unknown:
        raise RuntimeError(
            "ai_provider_manifest.json 字段不匹配: "
            f"缺少 {sorted(missing)}，多余 {sorted(unknown)}"
        )
    provider_id = entry["id"]
    if (
        not isinstance(provider_id, str)
        or not provider_id
        or provider_id != provider_id.strip().lower()
    ):
        raise RuntimeError("provider id 必须是非空 canonical 小写字符串")
    if not isinstance(entry["label"], str) or not entry["label"]:
        raise RuntimeError(f"provider {provider_id} label 必须是非空字符串")
    if (
        not isinstance(entry["kind"], str)
        or entry["kind"] not in {"openai_compatible", "local", "adapter"}
    ):
        raise RuntimeError(f"provider {provider_id} kind 无效")
    for field_name in (
        "requiresApiKey",
        "requiresModel",
        "requiresBaseUrl",
        "isLocal",
        "supportsStream",
        "supportsJsonResponse",
    ):
        if not isinstance(entry[field_name], bool):
            raise RuntimeError(f"provider {provider_id} {field_name} 必须是布尔值")
    default_base_url = entry.get("defaultBaseUrl")
    if default_base_url is not None and (
        not isinstance(default_base_url, str) or not default_base_url
    ):
        raise RuntimeError(f"provider {provider_id} defaultBaseUrl 无效")
    raw_capabilities = entry["capabilities"]
    if (
        not isinstance(raw_capabilities, list)
        or not raw_capabilities
        or any(
            not isinstance(capability, str)
            or capability not in _MANIFEST_CAPABILITIES
            for capability in raw_capabilities
        )
        or len(set(raw_capabilities)) != len(raw_capabilities)
    ):
        raise RuntimeError(f"provider {provider_id} capabilities 无效")
    capabilities = frozenset(
        _normalize_capability_name(capability)
        for capability in raw_capabilities
    )
    for field_name in (
        "capabilityBaseUrls",
        "capabilityEndpoints",
        "defaultModels",
        "modelCatalogs",
    ):
        if field_name in entry and not isinstance(entry[field_name], dict):
            raise RuntimeError(f"provider {provider_id} {field_name} 必须是对象")
    endpoints = {
        capability: endpoint
        for capability, endpoint in _default_capability_endpoints().items()
        if capability in capabilities
    }
    endpoints.update({
        _normalize_capability_name(capability): endpoint
        for capability, endpoint in entry.get("capabilityEndpoints", {}).items()
        if _normalize_capability_name(capability) in capabilities
    })
    for field_name in ("capabilityBaseUrls", "capabilityEndpoints"):
        values = entry.get(field_name, {})
        if any(
            capability not in _MANIFEST_CAPABILITIES
            or _normalize_capability_name(capability) not in capabilities
            or not isinstance(value, str)
            or not value
            for capability, value in values.items()
        ):
            raise RuntimeError(f"provider {provider_id} {field_name} 无效")
    default_models = entry.get("defaultModels", {})
    if any(
        model_type not in _MANIFEST_MODEL_TYPES
        or not isinstance(model, str)
        or not model
        for model_type, model in default_models.items()
    ):
        raise RuntimeError(f"provider {provider_id} defaultModels 无效")
    raw_catalogs = entry.get("modelCatalogs", {})
    if any(
        model_type not in _MANIFEST_MODEL_TYPES
        or not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not model for model in models)
        or len(set(models)) != len(models)
        for model_type, models in raw_catalogs.items()
    ):
        raise RuntimeError(f"provider {provider_id} modelCatalogs 无效")
    model_catalogs = {
        _normalize_model_type_name(model_type): tuple(models)
        for model_type, models in raw_catalogs.items()
    }
    return ProviderManifest(
        id=provider_id,
        display_name=entry["label"],
        kind=entry["kind"],
        default_base_url=default_base_url,
        capabilities=capabilities,
        requires_api_key=entry.get("requiresApiKey", True),
        requires_model=entry.get("requiresModel", True),
        requires_base_url=entry.get("requiresBaseUrl", False),
        is_local=entry.get("isLocal", False),
        supports_stream=entry.get("supportsStream", False),
        supports_json_response=entry.get("supportsJsonResponse", False),
        capability_base_urls={
            _normalize_capability_name(capability): base_url
            for capability, base_url in entry.get("capabilityBaseUrls", {}).items()
            if _normalize_capability_name(capability) in capabilities
        },
        capability_endpoints=endpoints,
        default_models={
            _normalize_model_type_name(model_type): model
            for model_type, model in default_models.items()
        },
        model_catalogs=model_catalogs,
    )


_PROVIDERS: Dict[str, ProviderManifest] = {
    entry["id"]: _build_provider_manifest(entry)
    for entry in _load_provider_manifest_data()
}


def normalize_provider_id(provider: Optional[str]) -> str:
    if provider is None:
        return ""
    if not isinstance(provider, str):
        raise TypeError("provider 必须是字符串或 null")
    return provider.strip().lower()


def get_provider_manifest(provider: Optional[str]) -> ProviderManifest:
    canonical = normalize_provider_id(provider)
    if canonical not in _PROVIDERS:
        raise ValueError(f"未知的 AI 服务商: {provider}")
    return _PROVIDERS[canonical]


def provider_supports_capability(provider: Optional[str], capability: str) -> bool:
    canonical = normalize_provider_id(provider)
    manifest = _PROVIDERS.get(canonical)
    return capability in manifest.capabilities if manifest else False


def provider_requires_model(provider: Optional[str]) -> bool:
    return get_provider_manifest(provider).requires_model


def provider_requires_api_key(
    provider: Optional[str],
    custom_base_url: Optional[str] = None,
) -> bool:
    manifest = get_provider_manifest(provider)
    resolved_base_url = resolve_provider_base_url(provider, custom_base_url)
    return manifest.requires_api_key and not is_local_service(resolved_base_url)


def resolve_provider_base_url(
    provider: Optional[str],
    custom_base_url: Optional[str] = None,
) -> Optional[str]:
    return resolve_provider_base_url_for_capability(provider, CHAT_CAPABILITY, custom_base_url)


def resolve_provider_base_url_for_capability(
    provider: Optional[str],
    capability: str,
    custom_base_url: Optional[str] = None,
) -> Optional[str]:
    manifest = get_provider_manifest(provider)
    if manifest.id == "custom":
        return custom_base_url or None
    return manifest.capability_base_urls.get(capability) or manifest.default_base_url


def resolve_provider_endpoint_for_capability(provider: Optional[str], capability: str) -> Optional[str]:
    manifest = get_provider_manifest(provider)
    return manifest.capability_endpoints.get(capability)
