"""Non-persistent provider diagnostics used by the v2 settings UI."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from io import BytesIO
import secrets
import time
from typing import Any, Mapping
import uuid

import httpx
from PIL import Image, ImageDraw

from src.backend_v2.paths import project_root
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.shared.ai_providers import (
    CHAT_CAPABILITY,
    CONNECTION_TEST_CAPABILITY,
    EMBEDDING_CAPABILITY,
    HQ_TRANSLATION_CAPABILITY,
    IMAGE_GEN_CAPABILITY,
    MODEL_FETCH_CAPABILITY,
    PLUGIN_AGENT_CAPABILITY,
    RERANK_CAPABILITY,
    TRANSLATION_CAPABILITY,
    VLM_CAPABILITY,
    VISION_OCR_CAPABILITY,
    WEB_IMPORT_AGENT_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_requires_api_key,
    provider_supports_capability,
)
from src.backend_v2.settings.validation import is_proofreading_provider_domain
from src.shared.ai_transport import (
    AsyncOpenAICompatibleTransport,
    OpenAICompatibleChatTransport,
    ProviderConnectionTestRequest,
    ProviderModelListRequest,
    UnifiedEmbeddingRequest,
    UnifiedRerankRequest,
    UnifiedVisionRequest,
)
from src.shared.memory_errors import is_memory_allocation_error


CONNECTION_TEST_KINDS = frozenset(
    {
        "ollama",
        "sakura",
        "lama_repair",
        "baidu_ocr",
        "ai_vision_ocr",
        "baidu_translate",
        "youdao_translate",
        "ai_translate",
        "firecrawl",
        "web_import_agent",
        "vlm",
        "llm",
        "embedding",
        "reranker",
    }
)

_MODEL_CATALOG_FIELDS = frozenset(
    {"provider", "domain", "baseUrl", "secret"}
)
_CONNECTION_TEST_FIELDS = {
    "ollama": frozenset({"domain", "baseUrl", "model"}),
    "sakura": frozenset({"domain", "baseUrl"}),
    "lama_repair": frozenset(),
    "baidu_ocr": frozenset({"domain", "secret"}),
    "ai_vision_ocr": frozenset(
        {"provider", "domain", "baseUrl", "model", "prompt", "secret"}
    ),
    "baidu_translate": frozenset({"domain", "secret"}),
    "youdao_translate": frozenset({"domain", "secret"}),
    "ai_translate": frozenset(
        {"provider", "domain", "baseUrl", "model", "secret"}
    ),
    "firecrawl": frozenset({"domain", "secret"}),
    "web_import_agent": frozenset(
        {"provider", "domain", "baseUrl", "model", "secret"}
    ),
    "vlm": frozenset({"provider", "domain", "baseUrl", "model", "secret"}),
    "llm": frozenset({"provider", "domain", "baseUrl", "model", "secret"}),
    "embedding": frozenset(
        {"provider", "domain", "baseUrl", "model", "secret"}
    ),
    "reranker": frozenset(
        {"provider", "domain", "baseUrl", "model", "secret"}
    ),
}


class DiagnosticRequestError(ValueError):
    """The diagnostic request or selected provider configuration is invalid."""


class ProviderDiagnostics:
    """Runs short, connection-bound checks without persisting their results."""

    def __init__(self, settings: SettingsRepository) -> None:
        self.settings = settings
        self.chat = OpenAICompatibleChatTransport()

    def model_catalog(self, body: Mapping[str, object]) -> dict[str, object]:
        self._validate_fields(body, _MODEL_CATALOG_FIELDS)
        provider = self._provider(body)
        if not provider_supports_capability(provider, MODEL_FETCH_CAPABILITY):
            raise DiagnosticRequestError(f"provider does not support model discovery: {provider}")
        domain = self._optional_string(body, "domain")
        required_capability = self._domain_capability(domain)
        if required_capability and not provider_supports_capability(
            provider,
            required_capability,
        ):
            raise DiagnosticRequestError(f"provider does not support {domain}: {provider}")
        secret = self._secret(body, provider=provider)
        manifest = get_provider_manifest(provider)
        api_key = self._api_key(
            secret,
            field=(
                "ai_vision_api_key"
                if self._optional_string(body, "domain") == "ai_vision_ocr"
                else "api_key"
            ),
        )
        base_url = self._optional_string(body, "baseUrl")
        if provider_requires_api_key(provider, base_url) and not api_key:
            raise DiagnosticRequestError("API key is required")
        if manifest.requires_base_url and not base_url:
            raise DiagnosticRequestError("baseUrl is required")
        models = self.chat.list_models(
            ProviderModelListRequest(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
            )
        )
        return {"models": models}

    def connection_test(
        self,
        kind: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        if kind not in CONNECTION_TEST_KINDS:
            raise DiagnosticRequestError("unsupported connection test kind")
        self._validate_fields(body, _CONNECTION_TEST_FIELDS[kind])
        try:
            message = self._run_test(kind, body)
            return {"success": True, "message": message}
        except (DiagnosticRequestError, LookupError):
            raise
        except Exception as exc:
            if is_memory_allocation_error(exc):
                raise
            return {"success": False, "message": self._friendly_error(exc)}

    def _run_test(self, kind: str, body: Mapping[str, object]) -> str:
        if kind == "lama_repair":
            return self._test_lama_assets()
        if kind == "firecrawl":
            return self._test_firecrawl(body)
        if kind == "baidu_ocr":
            return self._test_baidu_ocr(body)
        if kind == "baidu_translate":
            return self._test_baidu_translate(body)
        if kind == "youdao_translate":
            return self._test_youdao_translate(body)
        if kind == "embedding":
            return self._test_embedding(body)
        if kind == "reranker":
            return self._test_reranker(body)
        if kind == "ai_vision_ocr":
            return self._test_vision(
                body,
                api_key_field="ai_vision_api_key",
                capability=VISION_OCR_CAPABILITY,
            )
        if kind == "vlm":
            return self._test_vision(
                body,
                api_key_field="api_key",
                capability=VLM_CAPABILITY,
            )

        provider = (
            kind
            if kind in {"ollama", "sakura"}
            else self._provider(body)
        )
        if kind == "ai_translate" and provider == "caiyun":
            secret = self._secret(body, provider=provider)
            translated = self._translate_with_caiyun(self._api_key(secret))
            return f"连接成功：{translated}"
        required_capability = {
            "ollama": TRANSLATION_CAPABILITY,
            "sakura": TRANSLATION_CAPABILITY,
            "web_import_agent": WEB_IMPORT_AGENT_CAPABILITY,
            "llm": CHAT_CAPABILITY,
        }.get(kind)
        if kind == "ai_translate":
            required_capability = self._domain_capability(
                self._optional_string(body, "domain") or "translation"
            )
        if required_capability and not provider_supports_capability(
            provider,
            required_capability,
        ):
            raise DiagnosticRequestError(
                f"provider does not support {required_capability}: {provider}"
            )
        if not provider_supports_capability(provider, CONNECTION_TEST_CAPABILITY):
            raise DiagnosticRequestError(f"provider does not support connection tests: {provider}")
        secret = self._secret(body, provider=provider)
        manifest = get_provider_manifest(provider)
        api_key = self._api_key(secret)
        if manifest.requires_api_key and not api_key:
            raise DiagnosticRequestError("API key is required")
        base_url = self._optional_string(body, "baseUrl")
        if manifest.requires_base_url and not base_url:
            raise DiagnosticRequestError("baseUrl is required")
        model = self._optional_string(body, "model")
        if not model:
            catalog = self.model_catalog(
                {
                    **dict(body),
                    "provider": provider,
                    "secret": secret,
                }
            )
            models = catalog["models"]
            if not isinstance(models, list) or not models:
                raise RuntimeError("no model is available for the connection test")
            first_model = models[0]
            if not isinstance(first_model, Mapping):
                raise RuntimeError("model catalog returned an invalid model")
            model = self._required_string(first_model, "id")
        success, result = self.chat.test_connection(
            ProviderConnectionTestRequest(
                provider=provider,
                api_key=api_key,
                model=model,
                base_url=base_url,
                system_prompt=(
                    "You are a web import assistant."
                    if kind == "web_import_agent"
                    else "Reply briefly."
                ),
            )
        )
        if not success:
            raise RuntimeError(result)
        return f"连接成功：{result}"

    def _test_vision(
        self,
        body: Mapping[str, object],
        *,
        api_key_field: str,
        capability: str,
    ) -> str:
        provider = self._provider(body)
        if not provider_supports_capability(provider, capability):
            raise DiagnosticRequestError(f"provider does not support vision: {provider}")
        secret = self._secret(body, provider=provider)
        api_key = self._api_key(secret, field=api_key_field)
        manifest = get_provider_manifest(provider)
        model = self._required_string(body, "model")
        base_url = self._optional_string(body, "baseUrl")
        if provider_requires_api_key(provider, base_url) and not api_key:
            raise DiagnosticRequestError("API key is required")
        if manifest.requires_base_url and not base_url:
            raise DiagnosticRequestError("baseUrl is required")
        with Image.new("RGB", (320, 96), "white") as image:
            ImageDraw.Draw(image).text((16, 32), "Saber OCR test 123", fill="black")
            with BytesIO() as buffer:
                image.save(buffer, "PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        result = self.chat.complete_vision(
            UnifiedVisionRequest(
                provider=provider,
                api_key=api_key,
                model=model,
                prompt=self._optional_string(body, "prompt")
                or "Read the text in this image.",
                image_base64=image_base64,
                base_url=base_url,
            )
        )
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("vision provider returned no result")
        return f"连接成功：{result}"

    def _test_embedding(self, body: Mapping[str, object]) -> str:
        provider = self._provider(body)
        if not provider_supports_capability(provider, EMBEDDING_CAPABILITY):
            raise DiagnosticRequestError(f"provider does not support embedding: {provider}")
        secret = self._secret(body, provider=provider)
        manifest = get_provider_manifest(provider)
        api_key = self._api_key(secret)
        base_url = self._optional_string(body, "baseUrl")
        if manifest.requires_api_key and not api_key:
            raise DiagnosticRequestError("API key is required")
        if manifest.requires_base_url and not base_url:
            raise DiagnosticRequestError("baseUrl is required")
        result = asyncio.run(
            AsyncOpenAICompatibleTransport().embed(
                UnifiedEmbeddingRequest(
                    provider=provider,
                    api_key=api_key,
                    model=self._required_string(body, "model"),
                    inputs=["Saber connection test"],
                    base_url=base_url,
                    timeout=30,
                )
            )
        )
        if not result or not result[0]:
            raise RuntimeError("embedding provider returned no vector")
        return f"连接成功：向量维度 {len(result[0])}"

    def _test_reranker(self, body: Mapping[str, object]) -> str:
        provider = self._provider(body)
        if not provider_supports_capability(provider, RERANK_CAPABILITY):
            raise DiagnosticRequestError(f"provider does not support reranking: {provider}")
        secret = self._secret(body, provider=provider)
        manifest = get_provider_manifest(provider)
        api_key = self._api_key(secret)
        base_url = self._optional_string(body, "baseUrl")
        if manifest.requires_api_key and not api_key:
            raise DiagnosticRequestError("API key is required")
        if manifest.requires_base_url and not base_url:
            raise DiagnosticRequestError("baseUrl is required")
        result = asyncio.run(
            AsyncOpenAICompatibleTransport().rerank(
                UnifiedRerankRequest(
                    provider=provider,
                    api_key=api_key,
                    model=self._required_string(body, "model"),
                    query="manga",
                    documents=["manga translation", "weather report"],
                    top_n=1,
                    base_url=base_url,
                    timeout=30,
                )
            )
        )
        if not result.get("results"):
            raise RuntimeError("reranker returned no results")
        return "连接成功"

    def _test_firecrawl(self, body: Mapping[str, object]) -> str:
        secret = self._secret(body, provider="firecrawl")
        api_key = self._api_key(secret)
        if not api_key:
            raise DiagnosticRequestError("API key is required")
        response = httpx.get(
            "https://api.firecrawl.dev/v2/team/credit-usage",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = self._response_object(response, "Firecrawl")
        if payload.get("success") is not True:
            raise RuntimeError("Firecrawl returned an invalid response")
        return "连接成功"

    def _test_baidu_ocr(self, body: Mapping[str, object]) -> str:
        secret = self._secret(body, provider="baidu")
        self._require_secret_fields(
            secret,
            {"baidu_api_key", "baidu_secret_key"},
        )
        api_key = self._secret_string(secret, "baidu_api_key")
        secret_key = self._secret_string(secret, "baidu_secret_key")
        if not api_key or not secret_key:
            raise DiagnosticRequestError("Baidu OCR API key and secret key are required")
        response = httpx.post(
            "https://aip.baidubce.com/oauth/2.0/token",
            params={
                "grant_type": "client_credentials",
                "client_id": api_key,
                "client_secret": secret_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = self._response_object(response, "Baidu OCR")
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            description = payload.get("error_description")
            raise RuntimeError(
                description
                if isinstance(description, str) and description
                else "authentication failed"
            )
        return "连接成功"

    def _test_baidu_translate(self, body: Mapping[str, object]) -> str:
        secret = self._secret(body, provider="baidu_translate")
        self._require_secret_fields(secret, {"app_id", "app_key"})
        app_id = self._secret_string(secret, "app_id")
        app_key = self._secret_string(secret, "app_key")
        if not app_id or not app_key:
            raise DiagnosticRequestError("Baidu Translate app ID and app key are required")
        salt = str(secrets.randbelow(32769) + 32768)
        signature = hashlib.md5(
            f"{app_id}Hello{salt}{app_key}".encode("utf-8")
        ).hexdigest()
        response = httpx.post(
            "https://fanyi-api.baidu.com/api/trans/vip/translate",
            params={
                "appid": app_id,
                "q": "Hello",
                "from": "auto",
                "to": "zh",
                "salt": salt,
                "sign": signature,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = self._response_object(response, "Baidu Translate")
        if payload.get("error_code"):
            error_message = payload.get("error_msg")
            raise RuntimeError(
                error_message
                if isinstance(error_message, str) and error_message
                else "Baidu Translate rejected the request"
            )
        raw_results = payload.get("trans_result")
        if not isinstance(raw_results, list):
            raise RuntimeError("Baidu Translate returned an invalid response")
        translated_parts = []
        for item in raw_results:
            if not isinstance(item, Mapping):
                raise RuntimeError("Baidu Translate returned an invalid response")
            value = item.get("dst")
            if not isinstance(value, str):
                raise RuntimeError("Baidu Translate returned an invalid response")
            if value:
                translated_parts.append(value)
        translated = "\n".join(translated_parts)
        if not translated:
            raise RuntimeError("translation provider returned no result")
        return f"连接成功：{translated}"

    def _test_youdao_translate(self, body: Mapping[str, object]) -> str:
        secret = self._secret(body, provider="youdao_translate")
        self._require_secret_fields(secret, {"app_key", "app_secret"})
        app_key = self._secret_string(secret, "app_key")
        app_secret = self._secret_string(secret, "app_secret")
        if not app_key or not app_secret:
            raise DiagnosticRequestError("Youdao app key and app secret are required")
        source = "Hello, this is a test."
        salt = str(uuid.uuid4())
        current_time = str(int(time.time()))
        signature = hashlib.sha256(
            (
                app_key
                + self._youdao_signature_input(source)
                + salt
                + current_time
                + app_secret
            ).encode("utf-8")
        ).hexdigest()
        response = httpx.post(
            "https://openapi.youdao.com/api",
            data={
                "q": source,
                "from": "auto",
                "to": "zh-CHS",
                "appKey": app_key,
                "salt": salt,
                "sign": signature,
                "signType": "v3",
                "curtime": current_time,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = self._response_object(response, "Youdao Translate")
        error_code = payload.get("errorCode")
        if error_code is not None and error_code != "0":
            raise RuntimeError("Youdao Translate rejected the request")
        translations = payload.get("translation")
        if not isinstance(translations, list):
            raise RuntimeError("Youdao Translate returned an invalid response")
        translated = translations[0] if translations else ""
        if not isinstance(translated, str):
            raise RuntimeError("Youdao Translate returned an invalid response")
        if not translated or translated == source:
            raise RuntimeError("translation provider returned no result")
        return f"连接成功：{translated}"

    @staticmethod
    def _translate_with_caiyun(api_key: str) -> str:
        if not api_key:
            raise DiagnosticRequestError("API key is required")
        response = httpx.post(
            "https://api.interpreter.caiyunai.com/v1/translator",
            headers={
                "Content-Type": "application/json",
                "X-Authorization": f"token {api_key}",
            },
            json={
                "source": ["Hello"],
                "trans_type": "en2zh",
                "request_id": "saber-connection-test",
                "detect": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = ProviderDiagnostics._response_object(response, "Caiyun")
        translations = payload.get("target")
        if (
            not isinstance(translations, list)
            or not translations
            or not isinstance(translations[0], str)
            or not translations[0]
        ):
            raise RuntimeError("translation provider returned no result")
        return translations[0]

    @staticmethod
    def _youdao_signature_input(value: str) -> str:
        if len(value) <= 20:
            return value
        return f"{value[:10]}{len(value)}{value[-10:]}"

    @staticmethod
    def _test_lama_assets() -> str:
        model_root = project_root() / "models" / "lama"
        candidates = (
            model_root / "inpainting_lama_mpe.ckpt",
            model_root / "big-lama.safetensors",
        )
        available = [path.name for path in candidates if path.is_file()]
        if not available:
            raise RuntimeError("LaMA model files are not installed")
        return f"Worker 模型资源可用：{', '.join(available)}"

    def _secret(
        self,
        body: Mapping[str, object],
        *,
        provider: str,
    ) -> dict[str, Any]:
        raw = body.get("secret")
        if raw is not None:
            if not isinstance(raw, dict):
                raise DiagnosticRequestError("secret must be an object")
            return dict(raw)
        domain = self._optional_string(body, "domain")
        if domain:
            try:
                return self.settings.resolve_provider_secret(
                    domain=domain,
                    provider=provider,
                )
            except LookupError:
                if not provider_requires_api_key(
                    provider,
                    self._optional_string(body, "baseUrl"),
                ):
                    return {}
                raise
        return {}

    @staticmethod
    def _api_key(
        secret: Mapping[str, object],
        *,
        field: str = "api_key",
    ) -> str:
        ProviderDiagnostics._require_secret_fields(secret, {field})
        return ProviderDiagnostics._secret_string(secret, field)

    @staticmethod
    def _require_secret_fields(
        secret: Mapping[str, object],
        expected: set[str],
    ) -> None:
        if secret and set(secret) != expected:
            raise DiagnosticRequestError(
                "diagnostic secret fields must be exactly: "
                + ", ".join(sorted(expected))
            )

    @staticmethod
    def _secret_string(secret: Mapping[str, object], key: str) -> str:
        value = secret.get(key)
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _provider(body: Mapping[str, object]) -> str:
        provider = normalize_provider_id(
            ProviderDiagnostics._required_string(body, "provider")
        )
        try:
            get_provider_manifest(provider)
        except ValueError as exc:
            raise DiagnosticRequestError(str(exc)) from exc
        return provider

    @staticmethod
    def _required_string(body: Mapping[str, object], key: str) -> str:
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DiagnosticRequestError(f"{key} is required")
        return value.strip()

    @staticmethod
    def _optional_string(body: Mapping[str, object], key: str) -> str | None:
        value = body.get(key)
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise DiagnosticRequestError(f"{key} must be a string")
        return value.strip() or None

    @staticmethod
    def _validate_fields(
        body: Mapping[str, object],
        allowed: frozenset[str],
    ) -> None:
        unknown = set(body) - allowed
        if unknown:
            raise DiagnosticRequestError(
                "diagnostic request contains unknown fields: "
                + ", ".join(sorted(unknown))
            )

    @staticmethod
    def _response_object(
        response: httpx.Response,
        provider: str,
    ) -> Mapping[str, object]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{provider} returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"{provider} returned an invalid response")
        return payload

    @staticmethod
    def _domain_capability(domain: str | None) -> str | None:
        if domain is None:
            return None
        if domain == "translation":
            return TRANSLATION_CAPABILITY
        if domain == "hq" or is_proofreading_provider_domain(domain):
            return HQ_TRANSLATION_CAPABILITY
        capability = {
            "ai_vision_ocr": VISION_OCR_CAPABILITY,
            "plugin_agent": PLUGIN_AGENT_CAPABILITY,
            "web_import_agent": WEB_IMPORT_AGENT_CAPABILITY,
            "insight_vlm": VLM_CAPABILITY,
            "insight_chat": CHAT_CAPABILITY,
            "insight_embedding": EMBEDDING_CAPABILITY,
            "insight_reranker": RERANK_CAPABILITY,
            "insight_image_gen": IMAGE_GEN_CAPABILITY,
        }.get(domain)
        if capability is None:
            raise DiagnosticRequestError(f"unsupported diagnostic domain: {domain}")
        return capability

    @staticmethod
    def _friendly_error(error: Exception) -> str:
        message = redact_sensitive_text(error)
        lowered = message.lower()
        if "429" in lowered or "rate limit" in lowered or "too many requests" in lowered:
            return "服务请求过于频繁，请稍后重试"
        if (
            "401" in lowered
            or "403" in lowered
            or "authentication" in lowered
            or "unauthorized" in lowered
            or "invalid api key" in lowered
        ):
            return "API Key 无效或已过期"
        if "timeout" in lowered:
            return "连接超时，请检查网络"
        if "connect" in lowered or "connection reset" in lowered:
            return "无法连接到服务"
        if any(code in lowered for code in ("500", "502", "503", "504")):
            return "服务暂时不可用，请稍后重试"
        return message
