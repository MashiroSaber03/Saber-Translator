"""
翻译页在线 AI 服务商清单与能力映射。

该模块作为翻译页相关 AI 调用的单一真相源，统一维护：
- 规范化 provider id
- 默认 base_url
- 能力位
- 是否为 OpenAI 兼容 / 本地 / adapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional


TRANSLATION_CAPABILITY = "translation"
HQ_TRANSLATION_CAPABILITY = "hq_translation"
VISION_OCR_CAPABILITY = "vision_ocr"
MODEL_FETCH_CAPABILITY = "model_fetch"
CONNECTION_TEST_CAPABILITY = "connection_test"


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


_PROVIDERS: Dict[str, ProviderManifest] = {
    "siliconflow": ProviderManifest(
        id="siliconflow",
        display_name="SiliconFlow",
        kind="openai_compatible",
        default_base_url="https://api.siliconflow.cn/v1",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            HQ_TRANSLATION_CAPABILITY,
            VISION_OCR_CAPABILITY,
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        supports_stream=True,
        supports_json_response=True,
        supports_reasoning_control=True,
    ),
    "deepseek": ProviderManifest(
        id="deepseek",
        display_name="DeepSeek",
        kind="openai_compatible",
        default_base_url="https://api.deepseek.com/v1",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            HQ_TRANSLATION_CAPABILITY,
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        supports_stream=True,
        supports_json_response=True,
        supports_reasoning_control=True,
    ),
    "volcano": ProviderManifest(
        id="volcano",
        display_name="火山引擎",
        kind="openai_compatible",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            HQ_TRANSLATION_CAPABILITY,
            VISION_OCR_CAPABILITY,
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        supports_stream=True,
        supports_json_response=True,
        supports_reasoning_control=True,
    ),
    "gemini": ProviderManifest(
        id="gemini",
        display_name="Google Gemini",
        kind="openai_compatible",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            HQ_TRANSLATION_CAPABILITY,
            VISION_OCR_CAPABILITY,
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        supports_stream=True,
        supports_json_response=True,
        supports_reasoning_control=True,
    ),
    "custom": ProviderManifest(
        id="custom",
        display_name="自定义 OpenAI 兼容服务",
        kind="openai_compatible",
        default_base_url=None,
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            HQ_TRANSLATION_CAPABILITY,
            VISION_OCR_CAPABILITY,
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        requires_base_url=True,
        supports_stream=True,
        supports_json_response=True,
        supports_reasoning_control=True,
        legacy_ids=frozenset({"custom_openai", "custom_openai_vision"}),
    ),
    "ollama": ProviderManifest(
        id="ollama",
        display_name="Ollama",
        kind="local",
        default_base_url="http://localhost:11434",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        requires_api_key=False,
        is_local=True,
    ),
    "sakura": ProviderManifest(
        id="sakura",
        display_name="Sakura",
        kind="local",
        default_base_url="http://localhost:8080/v1",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        requires_api_key=False,
        is_local=True,
    ),
    "caiyun": ProviderManifest(
        id="caiyun",
        display_name="彩云小译",
        kind="adapter",
        default_base_url="http://api.interpreter.caiyunai.com/v1/translator",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        requires_model=False,
    ),
    "baidu_translate": ProviderManifest(
        id="baidu_translate",
        display_name="百度翻译",
        kind="adapter",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
    ),
    "youdao_translate": ProviderManifest(
        id="youdao_translate",
        display_name="有道翻译",
        kind="adapter",
        capabilities=frozenset({
            TRANSLATION_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
    ),
    "openai": ProviderManifest(
        id="openai",
        display_name="OpenAI",
        kind="openai_compatible",
        default_base_url="https://api.openai.com/v1",
        capabilities=frozenset({
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        supports_stream=True,
        supports_json_response=True,
        supports_reasoning_control=True,
    ),
    "qwen": ProviderManifest(
        id="qwen",
        display_name="通义千问",
        kind="openai_compatible",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        capabilities=frozenset({
            MODEL_FETCH_CAPABILITY,
            CONNECTION_TEST_CAPABILITY,
        }),
        supports_stream=True,
        supports_json_response=True,
        supports_reasoning_control=True,
    ),
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


def provider_supports_capability(provider: Optional[str], capability: str) -> bool:
    canonical = normalize_provider_id(provider)
    manifest = _PROVIDERS.get(canonical)
    return capability in manifest.capabilities if manifest else False


def get_providers_for_capability(capability: str) -> List[str]:
    return [
        provider_id
        for provider_id, manifest in _PROVIDERS.items()
        if capability in manifest.capabilities
    ]


def resolve_provider_base_url(provider: Optional[str], custom_base_url: Optional[str] = None) -> Optional[str]:
    manifest = get_provider_manifest(provider)
    if manifest.id == "custom":
        return custom_base_url or None
    return manifest.default_base_url


def is_openai_compatible_provider(provider: Optional[str]) -> bool:
    try:
        return get_provider_manifest(provider).kind == "openai_compatible"
    except ValueError:
        return False


def iter_provider_manifests() -> Iterable[ProviderManifest]:
    return _PROVIDERS.values()
