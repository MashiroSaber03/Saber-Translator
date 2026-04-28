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


TRANSLATION_CAPABILITY = "translation"
HQ_TRANSLATION_CAPABILITY = "hq_translation"
VISION_OCR_CAPABILITY = "vision_ocr"
MODEL_FETCH_CAPABILITY = "model_fetch"
CONNECTION_TEST_CAPABILITY = "connection_test"
WEB_IMPORT_AGENT_CAPABILITY = "web_import_agent"

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
}

_MODEL_TYPE_NAME_MAP = {
    "imageGen": "image_gen",
    "image_gen": "image_gen",
}


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
    supports_reasoning_control: bool = False
    legacy_ids: FrozenSet[str] = field(default_factory=frozenset)
    capability_base_urls: Mapping[str, str] = field(default_factory=dict)
    capability_endpoints: Mapping[str, str] = field(default_factory=dict)
    default_models: Mapping[str, str] = field(default_factory=dict)
    model_catalogs: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)


_MANIFEST_PATH = Path(__file__).with_name("ai_provider_manifest.json")


def _load_provider_manifest_data() -> list[dict]:
    with _MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


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
    provider_id = entry["id"]
    endpoints = _default_capability_endpoints()
    endpoints.update({
        _normalize_capability_name(capability): endpoint
        for capability, endpoint in entry.get("capabilityEndpoints", {}).items()
    })
    model_catalogs = {
        _normalize_model_type_name(model_type): tuple(models)
        for model_type, models in entry.get("modelCatalogs", {}).items()
    }
    return ProviderManifest(
        id=provider_id,
        display_name=entry["label"],
        kind=entry["kind"],
        default_base_url=entry.get("defaultBaseUrl"),
        capabilities=frozenset(_normalize_capability_name(capability) for capability in entry.get("capabilities", [])),
        requires_api_key=entry.get("requiresApiKey", True),
        requires_model=entry.get("requiresModel", True),
        requires_base_url=entry.get("requiresBaseUrl", False),
        is_local=entry.get("isLocal", False),
        supports_stream=entry.get("supportsStream", False),
        supports_json_response=entry.get("supportsJsonResponse", False),
        supports_reasoning_control=entry.get("supportsReasoningControl", False),
        legacy_ids=frozenset(entry.get("legacyIds", [])),
        capability_base_urls={
            _normalize_capability_name(capability): base_url
            for capability, base_url in entry.get("capabilityBaseUrls", {}).items()
        },
        capability_endpoints=endpoints,
        default_models={
            _normalize_model_type_name(model_type): model
            for model_type, model in entry.get("defaultModels", {}).items()
        },
        model_catalogs=model_catalogs,
    )


_PROVIDERS: Dict[str, ProviderManifest] = {
    entry["id"]: _build_provider_manifest(entry)
    for entry in _load_provider_manifest_data()
}

_LEGACY_ID_MAP = {
    legacy_id: manifest.id
    for manifest in _PROVIDERS.values()
    for legacy_id in manifest.legacy_ids
}


def normalize_provider_id(provider: Optional[str]) -> str:
    if not provider:
        return ""
    lowered = str(provider).strip().lower()
    return _LEGACY_ID_MAP.get(lowered, lowered)


def get_provider_manifest(provider: Optional[str]) -> ProviderManifest:
    canonical = normalize_provider_id(provider)
    if canonical not in _PROVIDERS:
        raise ValueError(f"未知的 AI 服务商: {provider}")
    return _PROVIDERS[canonical]


def get_all_provider_manifests() -> Dict[str, ProviderManifest]:
    return dict(_PROVIDERS)


def provider_supports_capability(provider: Optional[str], capability: str) -> bool:
    canonical = normalize_provider_id(provider)
    manifest = _PROVIDERS.get(canonical)
    return capability in manifest.capabilities if manifest else False


def resolve_provider_base_url(provider: Optional[str], custom_base_url: Optional[str] = None) -> Optional[str]:
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


def get_provider_default_model(provider: Optional[str], model_type: str) -> str:
    manifest = get_provider_manifest(provider)
    return manifest.default_models.get(_normalize_model_type_name(model_type), "")


def get_provider_model_catalog(provider: Optional[str], model_type: str) -> Tuple[str, ...]:
    manifest = get_provider_manifest(provider)
    return tuple(manifest.model_catalogs.get(_normalize_model_type_name(model_type), ()))


def is_openai_compatible_provider(provider: Optional[str]) -> bool:
    try:
        return get_provider_manifest(provider).kind == "openai_compatible"
    except ValueError:
        return False
