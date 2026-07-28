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
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.shared.ai_providers import (
    CONNECTION_TEST_CAPABILITY,
    MODEL_FETCH_CAPABILITY,
    RERANK_CAPABILITY,
    VISION_OCR_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
)
from src.shared.ai_transport import (
    AsyncOpenAICompatibleTransport,
    OpenAICompatibleChatTransport,
    ProviderConnectionTestRequest,
    ProviderModelListRequest,
    UnifiedEmbeddingRequest,
    UnifiedRerankRequest,
    UnifiedVisionRequest,
)


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


class ProviderDiagnostics:
    """Runs short, connection-bound checks without persisting their results."""

    def __init__(self, settings: SettingsRepository) -> None:
        self.settings = settings
        self.chat = OpenAICompatibleChatTransport()

    def model_catalog(self, body: Mapping[str, object]) -> dict[str, object]:
        provider = self._provider(body)
        if not provider_supports_capability(provider, MODEL_FETCH_CAPABILITY):
            raise ValueError(f"provider does not support model discovery: {provider}")
        secret = self._secret(body, provider=provider)
        manifest = get_provider_manifest(provider)
        api_key = self._api_key(secret)
        if manifest.requires_api_key and not api_key:
            raise ValueError("API key is required")
        base_url = self._optional_string(body, "baseUrl")
        if manifest.requires_base_url and not base_url:
            raise ValueError("baseUrl is required")
        models = self.chat.list_models(
            ProviderModelListRequest(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
            )
        )
        return {"success": True, "models": models}

    def connection_test(
        self,
        kind: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        if kind not in CONNECTION_TEST_KINDS:
            raise ValueError("unsupported connection test kind")
        try:
            message = self._run_test(kind, body)
            return {"success": True, "message": message}
        except (ValueError, LookupError):
            raise
        except Exception as exc:
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
        if kind in {"ai_vision_ocr", "vlm"}:
            return self._test_vision(body)

        provider = (
            kind
            if kind in {"ollama", "sakura"}
            else self._provider(body)
        )
        if kind == "ai_translate" and provider == "caiyun":
            secret = self._secret(body, provider=provider)
            translated = self._translate_with_caiyun(self._api_key(secret))
            return f"连接成功：{translated}"
        if not provider_supports_capability(provider, CONNECTION_TEST_CAPABILITY):
            raise ValueError(f"provider does not support connection tests: {provider}")
        secret = self._secret(body, provider=provider)
        manifest = get_provider_manifest(provider)
        api_key = self._api_key(secret)
        if manifest.requires_api_key and not api_key:
            raise ValueError("API key is required")
        base_url = self._optional_string(body, "baseUrl")
        if manifest.requires_base_url and not base_url:
            raise ValueError("baseUrl is required")
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
                raise ValueError("no model is available for the connection test")
            model = str(models[0]["id"])
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

    def _test_vision(self, body: Mapping[str, object]) -> str:
        provider = self._provider(body)
        if not provider_supports_capability(provider, VISION_OCR_CAPABILITY):
            raise ValueError(f"provider does not support vision: {provider}")
        secret = self._secret(body, provider=provider)
        api_key = self._api_key(secret)
        manifest = get_provider_manifest(provider)
        model = self._required_string(body, "model")
        base_url = self._optional_string(body, "baseUrl")
        if manifest.requires_api_key and not api_key:
            raise ValueError("API key is required")
        if manifest.requires_base_url and not base_url:
            raise ValueError("baseUrl is required")
        image = Image.new("RGB", (320, 96), "white")
        ImageDraw.Draw(image).text((16, 32), "Saber OCR test 123", fill="black")
        buffer = BytesIO()
        image.save(buffer, "PNG")
        result = self.chat.complete_vision(
            UnifiedVisionRequest(
                provider=provider,
                api_key=api_key,
                model=model,
                prompt=self._optional_string(body, "prompt")
                or "Read the text in this image.",
                image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                base_url=base_url,
            )
        )
        return f"连接成功：{result}"

    def _test_embedding(self, body: Mapping[str, object]) -> str:
        provider = self._provider(body)
        secret = self._secret(body, provider=provider)
        result = asyncio.run(
            AsyncOpenAICompatibleTransport().embed(
                UnifiedEmbeddingRequest(
                    provider=provider,
                    api_key=self._api_key(secret),
                    model=self._required_string(body, "model"),
                    inputs=["Saber connection test"],
                    base_url=self._optional_string(body, "baseUrl"),
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
            raise ValueError(f"provider does not support reranking: {provider}")
        secret = self._secret(body, provider=provider)
        result = asyncio.run(
            AsyncOpenAICompatibleTransport().rerank(
                UnifiedRerankRequest(
                    provider=provider,
                    api_key=self._api_key(secret),
                    model=self._required_string(body, "model"),
                    query="manga",
                    documents=["manga translation", "weather report"],
                    top_n=1,
                    base_url=self._optional_string(body, "baseUrl"),
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
            raise ValueError("API key is required")
        response = httpx.get(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if response.status_code in {401, 403}:
            raise RuntimeError("API key is invalid")
        return "连接成功"

    def _test_baidu_ocr(self, body: Mapping[str, object]) -> str:
        secret = self._secret(body, provider="baidu")
        api_key = self._first_secret(secret, "apiKey", "api_key", "baidu_api_key")
        secret_key = self._first_secret(
            secret,
            "secretKey",
            "secret_key",
            "baidu_secret_key",
        )
        if not api_key or not secret_key:
            raise ValueError("Baidu OCR API key and secret key are required")
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
        payload = response.json()
        if not payload.get("access_token"):
            raise RuntimeError(str(payload.get("error_description") or "authentication failed"))
        return "连接成功"

    def _test_baidu_translate(self, body: Mapping[str, object]) -> str:
        secret = self._secret(body, provider="baidu_translate")
        app_id = self._first_secret(secret, "appId", "app_id", "apiKey", "api_key")
        app_key = self._first_secret(
            secret,
            "appKey",
            "app_key",
            "secretKey",
            "secret_key",
        )
        if not app_id or not app_key:
            raise ValueError("Baidu Translate app ID and app key are required")
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
        payload = response.json()
        if payload.get("error_code"):
            raise RuntimeError(
                str(payload.get("error_msg") or payload["error_code"])
            )
        translated = "\n".join(
            str(item.get("dst", ""))
            for item in payload.get("trans_result", [])
            if item.get("dst")
        )
        if not translated:
            raise RuntimeError("translation provider returned no result")
        return f"连接成功：{translated}"

    def _test_youdao_translate(self, body: Mapping[str, object]) -> str:
        secret = self._secret(body, provider="youdao_translate")
        app_key = self._first_secret(secret, "appKey", "app_key", "apiKey", "api_key")
        app_secret = self._first_secret(
            secret,
            "appSecret",
            "app_secret",
            "secretKey",
            "secret_key",
        )
        if not app_key or not app_secret:
            raise ValueError("Youdao app key and app secret are required")
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
        payload = response.json()
        translations = payload.get("translation", [])
        translated = str(translations[0]) if translations else ""
        if not translated or translated == source:
            raise RuntimeError("translation provider returned no result")
        return f"连接成功：{translated}"

    @staticmethod
    def _translate_with_caiyun(api_key: str) -> str:
        if not api_key:
            raise ValueError("API key is required")
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
        payload = response.json()
        translations = payload.get("target", [])
        if not translations:
            raise RuntimeError("translation provider returned no result")
        return str(translations[0])

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
                raise ValueError("secret must be an object")
            return dict(raw)
        credential_id = self._optional_string(body, "credentialId")
        if credential_id:
            return self.settings.resolve_current_secret(credential_id)
        domain = self._optional_string(body, "domain")
        if domain:
            return self.settings.resolve_provider_secret(
                domain=domain,
                provider=provider,
            )
        return {}

    @staticmethod
    def _api_key(secret: Mapping[str, object]) -> str:
        return ProviderDiagnostics._first_secret(
            secret,
            "apiKey",
            "api_key",
            "ai_vision_api_key",
            "token",
        )

    @staticmethod
    def _first_secret(secret: Mapping[str, object], *keys: str) -> str:
        for key in keys:
            value = secret.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _provider(body: Mapping[str, object]) -> str:
        provider = normalize_provider_id(
            ProviderDiagnostics._required_string(body, "provider")
        )
        get_provider_manifest(provider)
        return provider

    @staticmethod
    def _required_string(body: Mapping[str, object], key: str) -> str:
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} is required")
        return value.strip()

    @staticmethod
    def _optional_string(body: Mapping[str, object], key: str) -> str | None:
        value = body.get(key)
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value.strip() or None

    @staticmethod
    def _friendly_error(error: Exception) -> str:
        message = str(error)
        lowered = message.lower()
        if "401" in lowered or "authentication" in lowered or "api key" in lowered:
            return "API Key 无效或已过期"
        if "timeout" in lowered:
            return "连接超时，请检查网络"
        if "connect" in lowered:
            return "无法连接到服务"
        return message
