"""
OpenAI-compatible transport shared by translation and Manga Insight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from src.shared.ai_providers import (
    EMBEDDING_CAPABILITY,
    RERANK_CAPABILITY,
    normalize_provider_id,
    resolve_provider_base_url,
    resolve_provider_base_url_for_capability,
    resolve_provider_endpoint_for_capability,
)
from src.shared.http_config import build_httpx_kwargs
from src.shared.openai_helpers import create_openai_client

logger = logging.getLogger("SharedAITransport")

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectError,
    httpx.ReadError,
    ConnectionResetError,
)


@dataclass
class UnifiedChatRequest:
    provider: str
    api_key: str
    model: str
    messages: List[Dict[str, Any]]
    base_url: Optional[str] = None
    timeout: float = 120.0
    use_stream: bool = False
    print_stream_output: bool = False
    stream_output_label: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    request_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedVisionRequest:
    provider: str
    api_key: str
    model: str
    prompt: str
    image_base64: str
    base_url: Optional[str] = None
    timeout: float = 120.0
    use_json_format: bool = False
    temperature: Optional[float] = None
    request_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedEmbeddingRequest:
    provider: str
    api_key: str
    model: str
    inputs: List[str]
    base_url: Optional[str] = None
    timeout: float = 60.0
    request_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedRerankRequest:
    provider: str
    api_key: str
    model: str
    query: str
    documents: List[str]
    top_n: int
    base_url: Optional[str] = None
    timeout: float = 30.0
    endpoint: Optional[str] = None
    request_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConnectionTestRequest:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str] = None
    prompt: str = "Hello"
    system_prompt: Optional[str] = "You are a translator. Translate to Chinese."
    timeout: float = 30.0


@dataclass
class ProviderModelListRequest:
    provider: str
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 15.0


def _build_chat_body(request: UnifiedChatRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.response_format:
        body["response_format"] = request.response_format
    if request.request_overrides:
        body.update(request.request_overrides)
    return body


def _build_embedding_body(request: UnifiedEmbeddingRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": request.model,
        "input": request.inputs,
    }
    if request.request_overrides:
        body.update(request.request_overrides)
    return body


def _build_rerank_body(request: UnifiedRerankRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": request.model,
        "query": request.query,
        "documents": request.documents,
        "top_n": request.top_n,
    }
    if request.request_overrides:
        body.update(request.request_overrides)
    return body


def _extract_chat_content_from_sdk_response(response: Any) -> str:
    if not response or not response.choices:
        raise ValueError("AI 未返回有效内容")
    return (response.choices[0].message.content or "").strip()


def _extract_chat_content_from_payload(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("AI 未返回有效内容")
    return (choices[0].get("message", {}).get("content") or "").strip()


def _extract_stream_chunk(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    delta = choices[0].get("delta", {})
    return delta.get("content") or ""


def _resolve_capability_base_url(
    provider: str,
    base_url: Optional[str],
    capability: str,
) -> Optional[str]:
    return resolve_provider_base_url_for_capability(provider, capability, base_url)


def _calculate_backoff(
    attempt: int,
    response: Optional[httpx.Response] = None,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    base_delay = 2 ** attempt
    jitter = random.uniform(0, 0.5)
    return min(base_delay * (1 + jitter), 60.0)


class OpenAICompatibleChatTransport:
    def complete(self, request: UnifiedChatRequest) -> str:
        base_url = resolve_provider_base_url(request.provider, request.base_url)
        if request.use_stream:
            return self._complete_stream(request, base_url)

        client = create_openai_client(
            api_key=request.api_key,
            base_url=base_url,
            timeout=request.timeout,
        )
        kwargs = _build_chat_body(request)
        kwargs["timeout"] = request.timeout

        response = client.chat.completions.create(**kwargs)
        return _extract_chat_content_from_sdk_response(response)

    def complete_vision(self, request: UnifiedVisionRequest) -> str:
        response_format = {"type": "json_object"} if request.use_json_format else None
        chat_request = UnifiedChatRequest(
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
            base_url=request.base_url,
            timeout=request.timeout,
            response_format=response_format,
            temperature=request.temperature,
            request_overrides=dict(request.request_overrides),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{request.image_base64}"},
                        },
                    ],
                }
            ],
        )
        return self.complete(chat_request)

    def test_connection(self, request: ProviderConnectionTestRequest) -> tuple[bool, str]:
        messages: List[Dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        try:
            content = self.complete(
                UnifiedChatRequest(
                    provider=request.provider,
                    api_key=request.api_key,
                    model=request.model,
                    base_url=request.base_url,
                    timeout=request.timeout,
                    messages=messages,
                    request_overrides={"max_tokens": 50},
                )
            )
            return True, content
        except Exception as exc:  # pragma: no cover - exercised via callers
            return False, str(exc)

    def list_models(self, request: ProviderModelListRequest) -> List[Dict[str, str]]:
        provider = normalize_provider_id(request.provider)
        if provider == "gemini":
            return self._list_gemini_models(request)

        base_url = resolve_provider_base_url(request.provider, request.base_url)
        if not base_url:
            raise ValueError("该服务商需要提供 Base URL")
        models_url = self._build_models_url(base_url)

        with httpx.Client(**build_httpx_kwargs(base_url, request.timeout)) as client:
            response = client.get(
                models_url,
                headers={"Authorization": f"Bearer {request.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
        models = [
            {"id": model.get("id", ""), "name": model.get("id", "")}
            for model in data.get("data", [])
            if model.get("id")
        ]
        return self._filter_models_for_provider(provider, models)

    def _complete_stream(self, request: UnifiedChatRequest, base_url: Optional[str]) -> str:
        if not base_url:
            raise ValueError("缺少 Base URL")
        url = f"{base_url.rstrip('/')}/chat/completions"
        body = _build_chat_body(request)
        body["stream"] = True

        full_text = ""
        if request.print_stream_output:
            label = request.stream_output_label or request.model
            print(f"\n[{label}] 开始流式输出: ", end="", flush=True)
        with httpx.Client(**build_httpx_kwargs(base_url, request.timeout)) as client:
            with client.stream(
                "POST",
                url,
                headers={
                    "Authorization": f"Bearer {request.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            ) as response:
                if response.status_code != 200:
                    error_text = response.read().decode("utf-8", errors="ignore")[:500]
                    raise ValueError(f"API 错误 {response.status_code}: {error_text}")
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    chunk = _extract_stream_chunk(data)
                    if chunk:
                        full_text += chunk
                        if request.print_stream_output:
                            print(chunk, end="", flush=True)
        if request.print_stream_output:
            label = request.stream_output_label or request.model
            print(f"\n[{label}] 流式输出完成，共 {len(full_text)} 字符\n", flush=True)
        return full_text.strip()

    def _list_gemini_models(self, request: ProviderModelListRequest) -> List[Dict[str, str]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={request.api_key}"
        with httpx.Client(**build_httpx_kwargs(url, request.timeout)) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        models: List[Dict[str, str]] = []
        for model in data.get("models", []):
            supported_methods = model.get("supportedGenerationMethods", [])
            if "generateContent" not in supported_methods:
                continue
            model_name = model.get("name", "")
            model_id = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
            models.append({"id": model_id, "name": model.get("displayName", model_id)})
        return models

    @staticmethod
    def _filter_models_for_provider(provider: str, models: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if provider != "siliconflow":
            return sorted(models, key=lambda item: item["id"])

        filtered = []
        for model in models:
            model_id = model.get("id", "")
            lower = model_id.lower()
            if (
                "chat" in lower
                or "llm" in lower
                or "qwen" in lower
                or "deepseek" in lower
                or "glm" in lower
                or "yi-" in lower
                or "internlm" in lower
                or "gemma" in lower
            ):
                filtered.append(model)
        return sorted(filtered, key=lambda item: item["id"])

    @staticmethod
    def _build_models_url(base_url: str) -> str:
        has_version_path = bool(
            httpx.URL(base_url).path.rstrip("/").endswith("/v1")
            or "/api/v" in httpx.URL(base_url).path
            or httpx.URL(base_url).path.rstrip("/").endswith("/models")
        )
        if base_url.rstrip("/").endswith("/models"):
            return base_url
        normalized = base_url.rstrip("/")
        if not has_version_path:
            normalized = f"{normalized}/v1"
        return f"{normalized}/models"


class AsyncOpenAICompatibleTransport:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max(0, int(max_retries))

    async def complete(self, request: UnifiedChatRequest) -> str:
        base_url = resolve_provider_base_url(request.provider, request.base_url)
        if request.use_stream:
            return await self._complete_stream(request, base_url)

        if not base_url:
            raise ValueError("缺少 Base URL")
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = await self._request_json(
            base_url=base_url,
            timeout=request.timeout,
            method="POST",
            url=url,
            api_key=request.api_key,
            body=_build_chat_body(request),
        )
        return _extract_chat_content_from_payload(payload)

    async def complete_vision(self, request: UnifiedVisionRequest) -> str:
        response_format = {"type": "json_object"} if request.use_json_format else None
        chat_request = UnifiedChatRequest(
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
            base_url=request.base_url,
            timeout=request.timeout,
            response_format=response_format,
            temperature=request.temperature,
            request_overrides=dict(request.request_overrides),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{request.image_base64}"},
                        },
                    ],
                }
            ],
        )
        return await self.complete(chat_request)

    async def embed(self, request: UnifiedEmbeddingRequest) -> List[List[float]]:
        base_url = _resolve_capability_base_url(request.provider, request.base_url, EMBEDDING_CAPABILITY)
        if not base_url:
            raise ValueError("缺少 Base URL")
        url = f"{base_url.rstrip('/')}/embeddings"
        payload = await self._request_json(
            base_url=base_url,
            timeout=request.timeout,
            method="POST",
            url=url,
            api_key=request.api_key,
            body=_build_embedding_body(request),
        )
        return [item["embedding"] for item in payload.get("data", [])]

    async def rerank(self, request: UnifiedRerankRequest) -> Dict[str, Any]:
        base_url = _resolve_capability_base_url(request.provider, request.base_url, RERANK_CAPABILITY)
        if not base_url:
            raise ValueError("缺少 Base URL")
        endpoint = request.endpoint or resolve_provider_endpoint_for_capability(request.provider, RERANK_CAPABILITY) or "/rerank"
        url = f"{base_url.rstrip('/')}{endpoint}"
        return await self._request_json(
            base_url=base_url,
            timeout=request.timeout,
            method="POST",
            url=url,
            api_key=request.api_key,
            body=_build_rerank_body(request),
        )

    async def _request_json(
        self,
        *,
        base_url: str,
        timeout: float,
        method: str,
        url: str,
        api_key: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            client = httpx.AsyncClient(**build_httpx_kwargs(base_url, timeout))
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )

                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    wait_time = _calculate_backoff(attempt, response)
                    logger.warning(
                        "Async transport received %s, retrying in %.1fs (%s/%s)",
                        response.status_code,
                        wait_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    error_text = response.text[:500] if response.text else "无响应内容"
                    raise ValueError(f"API 错误 {response.status_code}: {error_text}")

                return response.json()
            except RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    wait_time = _calculate_backoff(attempt)
                    logger.warning(
                        "Async transport request failed (%s), retrying in %.1fs (%s/%s)",
                        type(exc).__name__,
                        wait_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise
            finally:
                await client.aclose()

        if last_exception:
            raise last_exception
        raise RuntimeError("重试耗尽")

    async def _complete_stream(self, request: UnifiedChatRequest, base_url: Optional[str]) -> str:
        if not base_url:
            raise ValueError("缺少 Base URL")

        url = f"{base_url.rstrip('/')}/chat/completions"
        body = _build_chat_body(request)
        body["stream"] = True

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            full_text = ""
            client = httpx.AsyncClient(**build_httpx_kwargs(base_url, request.timeout))
            try:
                if request.print_stream_output:
                    label = request.stream_output_label or request.model
                    print(f"\n[{label}] 开始流式输出: ", end="", flush=True)
                async with client.stream(
                    "POST",
                    url,
                    headers={
                        "Authorization": f"Bearer {request.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                ) as response:
                    if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                        wait_time = _calculate_backoff(attempt, response)
                        logger.warning(
                            "Async stream transport received %s, retrying in %.1fs (%s/%s)",
                            response.status_code,
                            wait_time,
                            attempt + 1,
                            self.max_retries,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    if response.status_code != 200:
                        error_bytes = await response.aread()
                        error_text = error_bytes.decode("utf-8", errors="ignore")[:500]
                        raise ValueError(f"API 错误 {response.status_code}: {error_text}")

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        chunk = _extract_stream_chunk(data)
                        if chunk:
                            full_text += chunk
                            if request.print_stream_output:
                                print(chunk, end="", flush=True)

                if request.print_stream_output:
                    label = request.stream_output_label or request.model
                    print(f"\n[{label}] 流式输出完成，共 {len(full_text)} 字符\n", flush=True)
                return full_text.strip()
            except RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    wait_time = _calculate_backoff(attempt)
                    logger.warning(
                        "Async stream transport failed (%s), retrying in %.1fs (%s/%s)",
                        type(exc).__name__,
                        wait_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise
            finally:
                await client.aclose()

        if last_exception:
            raise last_exception
        raise RuntimeError("重试耗尽")
