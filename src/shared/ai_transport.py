"""
OpenAI-compatible transport shared by translation and Manga Insight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from src.shared.ai_providers import (
    CHAT_CAPABILITY,
    CONNECTION_TEST_CAPABILITY,
    EMBEDDING_CAPABILITY,
    MODEL_FETCH_CAPABILITY,
    RERANK_CAPABILITY,
    VISION_OCR_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
    resolve_provider_base_url,
    resolve_provider_base_url_for_capability,
    resolve_provider_endpoint_for_capability,
)
from src.shared.http_config import build_httpx_kwargs
from src.shared.openai_execution import (
    OpenAICompatibleEmptyContentError,
    OpenAICompatibleRuntimeOptions,
    ResolvedOpenAICompatibleInvocation,
    build_openai_compatible_runtime_options,
    clone_openai_compatible_runtime_options,
    resolve_openai_compatible_invocation,
)
from src.shared.openai_helpers import resolve_openai_api_key
from src.shared.openai_options import (
    OpenAICompatibleOptions,
    clone_openai_compatible_options,
    validate_and_clone_openai_extra_body,
)
from src.shared.openai_rate_limits import SharedRPMLimiter

logger = logging.getLogger("SharedAITransport")

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    ConnectionResetError,
)


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _require_optional_string(value: Any, *, name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_nonempty_string(value, name=name)


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _require_timeout(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} 必须是正有限数")
    return float(value)


def _require_string_list(value: Any, *, name: str) -> List[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{name} 必须是非空字符串列表")
    return value


@dataclass
class UnifiedChatRequest:
    provider: str
    api_key: Optional[str]
    model: str
    messages: List[Dict[str, Any]]
    credential_version_id: Optional[str] = None
    base_url: Optional[str] = None
    openai_options: OpenAICompatibleOptions = field(
        default_factory=OpenAICompatibleOptions
    )
    runtime_options: OpenAICompatibleRuntimeOptions = field(
        default_factory=OpenAICompatibleRuntimeOptions
    )
    capability: str = CHAT_CAPABILITY

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, name="provider")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key 必须是字符串或 null")
        _require_nonempty_string(self.model, name="model")
        if not isinstance(self.messages, list) or not self.messages:
            raise ValueError("messages 必须是非空列表")
        for message in self.messages:
            if not isinstance(message, dict):
                raise ValueError("messages 中的每项必须是对象")
            _require_nonempty_string(message.get("role"), name="message.role")
            content = message.get("content")
            if isinstance(content, str):
                if not content.strip():
                    raise ValueError("message.content 不能为空")
            elif not isinstance(content, list) or not content or any(
                not isinstance(item, dict) or not item for item in content
            ):
                raise ValueError("message.content 必须是非空字符串或内容列表")
        self.credential_version_id = _require_optional_string(
            self.credential_version_id,
            name="credential_version_id",
        )
        self.base_url = _require_optional_string(self.base_url, name="base_url")
        if not isinstance(self.openai_options, OpenAICompatibleOptions):
            raise TypeError("openai_options 类型错误")
        if not isinstance(self.runtime_options, OpenAICompatibleRuntimeOptions):
            raise TypeError("runtime_options 类型错误")
        _require_nonempty_string(self.capability, name="capability")

    @property
    def timeout(self) -> float:
        return self.runtime_options.timeout_or(120.0)

    @property
    def use_stream(self) -> bool:
        return self.openai_options.execution.use_stream

    @property
    def print_stream_output(self) -> bool:
        return self.runtime_options.print_stream_output

    @property
    def stream_output_label(self) -> Optional[str]:
        return self.runtime_options.stream_output_label

    @property
    def temperature(self) -> Optional[float]:
        return self.openai_options.request.temperature

    @property
    def response_format(self) -> Optional[Dict[str, Any]]:
        if self.openai_options.request.force_json_output:
            return {"type": "json_object"}
        return None

@dataclass
class UnifiedVisionRequest:
    provider: str
    api_key: Optional[str]
    model: str
    prompt: str
    image_base64: str
    credential_version_id: Optional[str] = None
    base_url: Optional[str] = None
    openai_options: OpenAICompatibleOptions = field(
        default_factory=OpenAICompatibleOptions
    )
    runtime_options: OpenAICompatibleRuntimeOptions = field(
        default_factory=OpenAICompatibleRuntimeOptions
    )
    capability: str = VISION_OCR_CAPABILITY

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, name="provider")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key 必须是字符串或 null")
        _require_nonempty_string(self.model, name="model")
        _require_nonempty_string(self.prompt, name="prompt")
        _require_nonempty_string(self.image_base64, name="image_base64")
        self.credential_version_id = _require_optional_string(
            self.credential_version_id,
            name="credential_version_id",
        )
        self.base_url = _require_optional_string(self.base_url, name="base_url")
        if not isinstance(self.openai_options, OpenAICompatibleOptions):
            raise TypeError("openai_options 类型错误")
        if not isinstance(self.runtime_options, OpenAICompatibleRuntimeOptions):
            raise TypeError("runtime_options 类型错误")
        _require_nonempty_string(self.capability, name="capability")

    @property
    def timeout(self) -> float:
        return self.runtime_options.timeout_or(120.0)

    @property
    def use_json_format(self) -> bool:
        return self.openai_options.request.force_json_output

    @property
    def temperature(self) -> Optional[float]:
        return self.openai_options.request.temperature

@dataclass
class UnifiedEmbeddingRequest:
    provider: str
    api_key: Optional[str]
    model: str
    inputs: List[str]
    credential_version_id: Optional[str] = None
    rpm_limit: int = 0
    base_url: Optional[str] = None
    timeout: Optional[float] = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, name="provider")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key 必须是字符串或 null")
        _require_nonempty_string(self.model, name="model")
        self.inputs = _require_string_list(self.inputs, name="inputs")
        self.credential_version_id = _require_optional_string(
            self.credential_version_id,
            name="credential_version_id",
        )
        self.rpm_limit = _require_nonnegative_int(self.rpm_limit, name="rpm_limit")
        self.base_url = _require_optional_string(self.base_url, name="base_url")
        if self.timeout is not None:
            self.timeout = _require_timeout(self.timeout, name="timeout")


@dataclass
class UnifiedRerankRequest:
    provider: str
    api_key: Optional[str]
    model: str
    query: str
    documents: List[str]
    top_n: int
    credential_version_id: Optional[str] = None
    rpm_limit: int = 0
    base_url: Optional[str] = None
    timeout: Optional[float] = 30.0
    endpoint: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, name="provider")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key 必须是字符串或 null")
        _require_nonempty_string(self.model, name="model")
        _require_nonempty_string(self.query, name="query")
        self.documents = _require_string_list(self.documents, name="documents")
        self.top_n = _require_positive_int(self.top_n, name="top_n")
        if self.top_n > len(self.documents):
            raise ValueError("top_n 不能超过 documents 数量")
        self.credential_version_id = _require_optional_string(
            self.credential_version_id,
            name="credential_version_id",
        )
        self.rpm_limit = _require_nonnegative_int(self.rpm_limit, name="rpm_limit")
        self.base_url = _require_optional_string(self.base_url, name="base_url")
        if self.timeout is not None:
            self.timeout = _require_timeout(self.timeout, name="timeout")
        self.endpoint = _require_optional_string(self.endpoint, name="endpoint")
        if self.endpoint is not None and not self.endpoint.startswith("/"):
            raise ValueError("endpoint 必须是以 / 开头的路径")


@dataclass
class ProviderConnectionTestRequest:
    provider: str
    api_key: Optional[str]
    model: str
    base_url: Optional[str] = None
    prompt: str = "Hello"
    system_prompt: Optional[str] = "You are a translator. Translate to Chinese."
    timeout: float = 30.0

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, name="provider")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key 必须是字符串或 null")
        _require_nonempty_string(self.model, name="model")
        self.base_url = _require_optional_string(self.base_url, name="base_url")
        _require_nonempty_string(self.prompt, name="prompt")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt 必须是字符串或 null")
        self.timeout = _require_timeout(self.timeout, name="timeout")


@dataclass
class ProviderModelListRequest:
    provider: str
    api_key: Optional[str]
    base_url: Optional[str] = None
    timeout: float = 15.0

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, name="provider")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key 必须是字符串或 null")
        self.base_url = _require_optional_string(self.base_url, name="base_url")
        self.timeout = _require_timeout(self.timeout, name="timeout")


def _build_chat_body(
    request: UnifiedChatRequest,
    invocation: Optional[ResolvedOpenAICompatibleInvocation] = None,
) -> Dict[str, Any]:
    effective_options = invocation.effective_options if invocation else request.openai_options
    body: Dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
    }
    if effective_options.request.temperature is not None:
        body["temperature"] = effective_options.request.temperature
    if effective_options.request.force_json_output:
        body["response_format"] = {"type": "json_object"}
    extra_body = validate_and_clone_openai_extra_body(
        effective_options.request.extra_body,
        prefix="openai_options.request.extra_body",
    )
    if extra_body:
        body.update(extra_body)
    return body


def _build_embedding_body(request: UnifiedEmbeddingRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": request.model,
        "input": request.inputs,
    }
    return body


def _build_rerank_body(request: UnifiedRerankRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": request.model,
        "query": request.query,
        "documents": request.documents,
        "top_n": request.top_n,
    }
    return body


def _extract_chat_content_from_payload(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("AI 响应必须是 JSON 对象")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAICompatibleEmptyContentError("AI 未返回有效内容")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("AI 响应 choices[0].message 格式错误")
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("AI 响应 message.content 必须是字符串")
    content = content.strip()
    if not content:
        raise OpenAICompatibleEmptyContentError("AI 未返回有效内容")
    return content


def _extract_stream_chunk(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("AI 流响应必须是 JSON 对象")
    choices = payload.get("choices")
    if choices == []:
        return ""
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("AI 流响应 choices 格式错误")
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        raise ValueError("AI 流响应 delta 格式错误")
    content = delta.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("AI 流响应 delta.content 必须是字符串")
    return content


def _extract_sse_data(line: str) -> Optional[str]:
    if not isinstance(line, str):
        raise ValueError("AI 流响应行必须是字符串")
    if not line.startswith("data:"):
        return None
    data = line[5:]
    if data.startswith(" "):
        data = data[1:]
    return data.strip()


def _extract_embeddings(
    payload: Dict[str, Any],
    *,
    expected_count: int,
) -> List[List[float]]:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise ValueError(
            f"嵌入向量数量不匹配: 期望 {expected_count}, 实际 "
            f"{len(data) if isinstance(data, list) else '非列表'}"
        )
    embeddings: List[List[float]] = []
    dimension: Optional[int] = None
    for expected_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {expected_index} 个嵌入条目必须是对象")
        item_index = item.get("index")
        if (
            isinstance(item_index, bool)
            or not isinstance(item_index, int)
            or item_index != expected_index
        ):
            raise ValueError(f"第 {expected_index} 个嵌入条目索引错误")
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"第 {expected_index} 个嵌入向量为空或格式错误")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in embedding
        ):
            raise ValueError(f"第 {expected_index} 个嵌入向量包含非有限数值")
        if dimension is None:
            dimension = len(embedding)
        elif len(embedding) != dimension:
            raise ValueError("嵌入向量维度不一致")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def _resolve_capability_base_url(
    provider: str,
    base_url: Optional[str],
    capability: str,
) -> Optional[str]:
    return resolve_provider_base_url_for_capability(provider, capability, base_url)


def _require_provider_api_key(provider: str, api_key: Optional[str]) -> None:
    manifest = get_provider_manifest(provider)
    if manifest.requires_api_key and (
        not isinstance(api_key, str) or not api_key.strip()
    ):
        raise ValueError(f"{manifest.display_name}需要 API Key")


def _calculate_backoff(
    attempt: int,
    response: Optional[httpx.Response] = None,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                parsed = float(retry_after)
                if math.isfinite(parsed) and parsed >= 0:
                    return parsed
            except ValueError:
                pass

    base_delay = 2 ** attempt
    jitter = random.uniform(0, 0.5)
    return min(base_delay * (1 + jitter), 60.0)


def _build_auth_headers(
    api_key: Optional[str],
    base_url: Optional[str],
    *,
    include_content_type: bool = True,
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    resolved_api_key = resolve_openai_api_key(api_key, base_url)
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def _build_models_url(base_url: str) -> str:
    if base_url.rstrip("/").endswith("/models"):
        return base_url
    return f"{base_url.rstrip('/')}/models"


def _resolve_chat_invocation(
    request: UnifiedChatRequest,
    invocation: Optional[ResolvedOpenAICompatibleInvocation],
) -> ResolvedOpenAICompatibleInvocation:
    return invocation or resolve_openai_compatible_invocation(
        request.provider,
        request.capability,
        request.openai_options,
        request.runtime_options,
    )


class OpenAICompatibleChatTransport:
    def complete(
        self,
        request: UnifiedChatRequest,
        *,
        resolved_invocation: Optional[ResolvedOpenAICompatibleInvocation] = None,
        before_request: Optional[Callable[[], None]] = None,
    ) -> str:
        invocation = _resolve_chat_invocation(request, resolved_invocation)
        _require_provider_api_key(invocation.provider, request.api_key)
        base_url = resolve_provider_base_url(invocation.provider, request.base_url)
        limiter = SharedRPMLimiter(
            invocation.effective_options.execution.rpm_limit,
            provider=invocation.provider,
            credential_version_id=request.credential_version_id,
        )

        def prepare_request() -> None:
            if before_request is not None:
                before_request()
            limiter.wait_sync()

        if invocation.use_stream:
            return self._complete_stream(request, base_url, invocation, prepare_request)

        if not base_url:
            raise ValueError("缺少 Base URL")

        payload = self._request_json(
            base_url=base_url,
            timeout=invocation.timeout,
            method="POST",
            url=f"{base_url.rstrip('/')}/chat/completions",
            api_key=request.api_key,
            body=_build_chat_body(request, invocation),
            max_retries=invocation.effective_options.execution.transport_retries,
            before_request=prepare_request,
        )
        return _extract_chat_content_from_payload(payload)

    def complete_vision(
        self,
        request: UnifiedVisionRequest,
        *,
        resolved_invocation: Optional[ResolvedOpenAICompatibleInvocation] = None,
        before_request: Optional[Callable[[], None]] = None,
    ) -> str:
        invocation = resolved_invocation or resolve_openai_compatible_invocation(
            request.provider,
            request.capability,
            request.openai_options,
            request.runtime_options,
        )
        chat_request = UnifiedChatRequest(
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
            credential_version_id=request.credential_version_id,
            base_url=request.base_url,
            capability=request.capability,
            openai_options=clone_openai_compatible_options(request.openai_options),
            runtime_options=clone_openai_compatible_runtime_options(request.runtime_options),
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
        return self.complete(
            chat_request,
            resolved_invocation=invocation,
            before_request=before_request,
        )

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
                    capability=CONNECTION_TEST_CAPABILITY,
                    openai_options=OpenAICompatibleOptions(),
                    runtime_options=build_openai_compatible_runtime_options(
                        timeout=request.timeout,
                    ),
                    messages=messages,
                )
            )
            return True, content
        except Exception as exc:  # pragma: no cover - exercised via callers
            return False, str(exc)

    def list_models(self, request: ProviderModelListRequest) -> List[Dict[str, str]]:
        provider = normalize_provider_id(request.provider)
        if not provider_supports_capability(provider, MODEL_FETCH_CAPABILITY):
            raise ValueError(f"{request.provider} 不支持模型列表")
        _require_provider_api_key(provider, request.api_key)
        if provider == "gemini":
            return self._list_gemini_models(request)

        base_url = resolve_provider_base_url(request.provider, request.base_url)
        if not base_url:
            raise ValueError("该服务商需要提供 Base URL")
        models_url = _build_models_url(base_url)

        with httpx.Client(**build_httpx_kwargs(base_url, request.timeout)) as client:
            response = client.get(
                models_url,
                headers=_build_auth_headers(request.api_key, base_url, include_content_type=False),
            )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("模型列表响应格式错误")
        models: List[Dict[str, str]] = []
        seen_ids: set[str] = set()
        for item in data["data"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not item["id"].strip()
            ):
                raise ValueError("模型列表条目缺少有效 id")
            model_id = item["id"].strip()
            if model_id in seen_ids:
                raise ValueError("模型列表包含重复 id")
            seen_ids.add(model_id)
            models.append({"id": model_id, "name": model_id})
        return sorted(models, key=lambda item: item["id"])

    def _request_json(
        self,
        *,
        base_url: str,
        timeout: float,
        method: str,
        url: str,
        api_key: Optional[str],
        body: Dict[str, Any],
        max_retries: int,
        before_request: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            with httpx.Client(**build_httpx_kwargs(base_url, timeout)) as client:
                try:
                    if before_request is not None:
                        before_request()
                    response = client.request(
                        method=method,
                        url=url,
                        headers=_build_auth_headers(api_key, base_url),
                        json=body,
                    )

                    if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                        wait_time = _calculate_backoff(attempt, response)
                        logger.warning(
                            "Sync transport received %s, retrying in %.1fs (%s/%s)",
                            response.status_code,
                            wait_time,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        error_text = response.text[:500] if response.text else "无响应内容"
                        raise ValueError(f"API 错误 {response.status_code}: {error_text}")

                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("AI API 响应必须是 JSON 对象")
                    return payload
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        wait_time = _calculate_backoff(attempt)
                        logger.warning(
                            "Sync transport request failed (%s), retrying in %.1fs (%s/%s)",
                            type(exc).__name__,
                            wait_time,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(wait_time)
                        continue
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("重试耗尽")

    def _complete_stream(
        self,
        request: UnifiedChatRequest,
        base_url: Optional[str],
        invocation: ResolvedOpenAICompatibleInvocation,
        before_request: Optional[Callable[[], None]] = None,
    ) -> str:
        if not base_url:
            raise ValueError("缺少 Base URL")

        url = f"{base_url.rstrip('/')}/chat/completions"
        body = _build_chat_body(request, invocation)
        body["stream"] = True
        max_retries = invocation.effective_options.execution.transport_retries

        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            full_text = ""
            with httpx.Client(**build_httpx_kwargs(base_url, invocation.timeout)) as client:
                try:
                    if before_request is not None:
                        before_request()
                    attempt_started_at = time.monotonic()
                    with client.stream(
                        "POST",
                        url,
                        headers=_build_auth_headers(request.api_key, base_url),
                        json=body,
                    ) as response:
                        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                            wait_time = _calculate_backoff(attempt, response)
                            logger.warning(
                                "Sync stream transport received %s, retrying in %.1fs (%s/%s)",
                                response.status_code,
                                wait_time,
                                attempt + 1,
                                max_retries,
                            )
                            time.sleep(wait_time)
                            continue

                        if response.status_code != 200:
                            error_text = response.read().decode("utf-8", errors="ignore")[:500]
                            raise ValueError(f"API 错误 {response.status_code}: {error_text}")

                        if invocation.runtime_options.print_stream_output:
                            label = invocation.runtime_options.stream_output_label or request.model
                            print(f"\n[{label}] 开始流式输出: ", end="", flush=True)

                        for line in response.iter_lines():
                            if time.monotonic() - attempt_started_at > invocation.timeout:
                                raise httpx.ReadTimeout(
                                    "AI stream attempt exceeded "
                                    f"{invocation.timeout:g} seconds"
                                )
                            data_str = _extract_sse_data(line)
                            if data_str is None or not data_str:
                                continue
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError as exc:
                                raise ValueError("AI 流响应包含无效 JSON") from exc
                            chunk = _extract_stream_chunk(data)
                            if chunk:
                                full_text += chunk
                                if invocation.runtime_options.on_stream_chunk:
                                    invocation.runtime_options.on_stream_chunk(chunk, full_text)
                                if invocation.runtime_options.print_stream_output:
                                    print(chunk, end="", flush=True)

                    if invocation.runtime_options.print_stream_output:
                        label = invocation.runtime_options.stream_output_label or request.model
                        print(f"\n[{label}] 流式输出完成，共 {len(full_text)} 字符\n", flush=True)
                    full_text = full_text.strip()
                    if not full_text:
                        raise OpenAICompatibleEmptyContentError("AI 未返回有效内容")
                    return full_text
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exception = exc
                    if attempt < max_retries and not full_text:
                        wait_time = _calculate_backoff(attempt)
                        logger.warning(
                            "Sync stream transport failed (%s), retrying in %.1fs (%s/%s)",
                            type(exc).__name__,
                            wait_time,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(wait_time)
                        continue
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("重试耗尽")

    def _list_gemini_models(self, request: ProviderModelListRequest) -> List[Dict[str, str]]:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        with httpx.Client(**build_httpx_kwargs(url, request.timeout)) as client:
            response = client.get(url, params={"key": request.api_key})
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Gemini 模型列表响应格式错误")
        models: List[Dict[str, str]] = []
        for model in data["models"]:
            if not isinstance(model, dict):
                raise ValueError("Gemini 模型列表条目格式错误")
            supported_methods = model.get("supportedGenerationMethods")
            if not isinstance(supported_methods, list) or any(
                not isinstance(method, str) for method in supported_methods
            ):
                raise ValueError("Gemini 模型支持方法格式错误")
            if "generateContent" not in supported_methods:
                continue
            model_name = model.get("name")
            if not isinstance(model_name, str) or not model_name:
                raise ValueError("Gemini 模型条目缺少有效 name")
            model_id = model_name.removeprefix("models/")
            display_name = model.get("displayName", model_id)
            if not isinstance(display_name, str) or not display_name:
                raise ValueError("Gemini 模型条目的 displayName 格式错误")
            models.append({"id": model_id, "name": display_name})
        return sorted(models, key=lambda item: item["id"])


class AsyncOpenAICompatibleTransport:
    def __init__(self, max_retries: int = 0):
        self.max_retries = _require_nonnegative_int(
            max_retries,
            name="max_retries",
        )

    async def complete(
        self,
        request: UnifiedChatRequest,
        *,
        resolved_invocation: Optional[ResolvedOpenAICompatibleInvocation] = None,
        before_request: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        invocation = _resolve_chat_invocation(request, resolved_invocation)
        _require_provider_api_key(invocation.provider, request.api_key)
        base_url = resolve_provider_base_url(invocation.provider, request.base_url)
        limiter = SharedRPMLimiter(
            invocation.effective_options.execution.rpm_limit,
            provider=invocation.provider,
            credential_version_id=request.credential_version_id,
        )

        async def prepare_request() -> None:
            if before_request is not None:
                await before_request()
            await limiter.wait()

        if invocation.use_stream:
            return await self._complete_stream(request, base_url, invocation, prepare_request)

        if not base_url:
            raise ValueError("缺少 Base URL")

        payload = await self._request_json(
            base_url=base_url,
            timeout=invocation.timeout,
            method="POST",
            url=f"{base_url.rstrip('/')}/chat/completions",
            api_key=request.api_key,
            body=_build_chat_body(request, invocation),
            max_retries=invocation.effective_options.execution.transport_retries,
            before_request=prepare_request,
        )
        return _extract_chat_content_from_payload(payload)

    async def complete_vision(
        self,
        request: UnifiedVisionRequest,
        *,
        resolved_invocation: Optional[ResolvedOpenAICompatibleInvocation] = None,
        before_request: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        invocation = resolved_invocation or resolve_openai_compatible_invocation(
            request.provider,
            request.capability,
            request.openai_options,
            request.runtime_options,
        )
        chat_request = UnifiedChatRequest(
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
            credential_version_id=request.credential_version_id,
            base_url=request.base_url,
            capability=request.capability,
            openai_options=clone_openai_compatible_options(request.openai_options),
            runtime_options=clone_openai_compatible_runtime_options(request.runtime_options),
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
        return await self.complete(
            chat_request,
            resolved_invocation=invocation,
            before_request=before_request,
        )

    async def embed(self, request: UnifiedEmbeddingRequest) -> List[List[float]]:
        if not provider_supports_capability(request.provider, EMBEDDING_CAPABILITY):
            raise ValueError(f"{request.provider} 不支持嵌入向量")
        _require_provider_api_key(request.provider, request.api_key)
        base_url = _resolve_capability_base_url(
            request.provider,
            request.base_url,
            EMBEDDING_CAPABILITY,
        )
        if not base_url:
            raise ValueError("缺少 Base URL")
        url = f"{base_url.rstrip('/')}/embeddings"
        limiter = SharedRPMLimiter(
            request.rpm_limit,
            provider=request.provider,
            credential_version_id=request.credential_version_id,
        )
        payload = await self._request_json(
            base_url=base_url,
            timeout=request.timeout,
            method="POST",
            url=url,
            api_key=request.api_key,
            body=_build_embedding_body(request),
            max_retries=self.max_retries,
            before_request=limiter.wait,
        )
        return _extract_embeddings(payload, expected_count=len(request.inputs))

    async def rerank(self, request: UnifiedRerankRequest) -> Dict[str, Any]:
        if not provider_supports_capability(request.provider, RERANK_CAPABILITY):
            raise ValueError(f"{request.provider} 不支持重排")
        _require_provider_api_key(request.provider, request.api_key)
        base_url = _resolve_capability_base_url(
            request.provider,
            request.base_url,
            RERANK_CAPABILITY,
        )
        if not base_url:
            raise ValueError("缺少 Base URL")
        endpoint = request.endpoint or resolve_provider_endpoint_for_capability(
            request.provider,
            RERANK_CAPABILITY,
        )
        if endpoint is None:
            raise ValueError("重排服务缺少 endpoint")
        url = f"{base_url.rstrip('/')}{endpoint}"
        limiter = SharedRPMLimiter(
            request.rpm_limit,
            provider=request.provider,
            credential_version_id=request.credential_version_id,
        )
        return await self._request_json(
            base_url=base_url,
            timeout=request.timeout,
            method="POST",
            url=url,
            api_key=request.api_key,
            body=_build_rerank_body(request),
            max_retries=self.max_retries,
            before_request=limiter.wait,
        )

    async def _request_json(
        self,
        *,
        base_url: str,
        timeout: Optional[float],
        method: str,
        url: str,
        api_key: Optional[str],
        body: Dict[str, Any],
        max_retries: int,
        before_request: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            client = httpx.AsyncClient(**build_httpx_kwargs(base_url, timeout))
            try:
                if before_request is not None:
                    await before_request()
                try:
                    async with asyncio.timeout(timeout):
                        response = await client.request(
                            method=method,
                            url=url,
                            headers=_build_auth_headers(api_key, base_url),
                            json=body,
                        )
                except TimeoutError as exc:
                    message = (
                        "AI request attempt timed out"
                        if timeout is None
                        else f"AI request attempt exceeded {timeout:g} seconds"
                    )
                    raise httpx.ReadTimeout(
                        message
                    ) from exc

                if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                    wait_time = _calculate_backoff(attempt, response)
                    logger.warning(
                        "Async transport received %s, retrying in %.1fs (%s/%s)",
                        response.status_code,
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    error_text = response.text[:500] if response.text else "无响应内容"
                    raise ValueError(f"API 错误 {response.status_code}: {error_text}")

                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("AI API 响应必须是 JSON 对象")
                return payload
            except RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if attempt < max_retries:
                    wait_time = _calculate_backoff(attempt)
                    logger.warning(
                        "Async transport request failed (%s), retrying in %.1fs (%s/%s)",
                        type(exc).__name__,
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise
            finally:
                await client.aclose()

        if last_exception:
            raise last_exception
        raise RuntimeError("重试耗尽")

    async def _complete_stream(
        self,
        request: UnifiedChatRequest,
        base_url: Optional[str],
        invocation: ResolvedOpenAICompatibleInvocation,
        before_request: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        if not base_url:
            raise ValueError("缺少 Base URL")

        url = f"{base_url.rstrip('/')}/chat/completions"
        body = _build_chat_body(request, invocation)
        body["stream"] = True
        max_retries = invocation.effective_options.execution.transport_retries

        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            full_text = ""
            client = httpx.AsyncClient(
                **build_httpx_kwargs(base_url, invocation.timeout)
            )
            try:
                if before_request is not None:
                    await before_request()
                try:
                    # httpx's timeout is an inactivity timeout. Bound the
                    # complete attempt as well so keep-alive/empty SSE frames
                    # cannot consume the caller's entire logical deadline and
                    # suppress configured transport retries.
                    async with asyncio.timeout(invocation.timeout):
                        async with client.stream(
                            "POST",
                            url,
                            headers=_build_auth_headers(request.api_key, base_url),
                            json=body,
                        ) as response:
                            if (
                                response.status_code in RETRYABLE_STATUS_CODES
                                and attempt < max_retries
                            ):
                                wait_time = _calculate_backoff(attempt, response)
                                logger.warning(
                                    "Async stream transport received %s, retrying in %.1fs (%s/%s)",
                                    response.status_code,
                                    wait_time,
                                    attempt + 1,
                                    max_retries,
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            if response.status_code != 200:
                                error_bytes = await response.aread()
                                error_text = error_bytes.decode(
                                    "utf-8",
                                    errors="ignore",
                                )[:500]
                                raise ValueError(
                                    f"API 错误 {response.status_code}: {error_text}"
                                )

                            if invocation.runtime_options.print_stream_output:
                                label = (
                                    invocation.runtime_options.stream_output_label
                                    or request.model
                                )
                                print(f"\n[{label}] 开始流式输出: ", end="", flush=True)

                            async for line in response.aiter_lines():
                                data_str = _extract_sse_data(line)
                                if data_str is None or not data_str:
                                    continue
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                except json.JSONDecodeError as exc:
                                    raise ValueError("AI 流响应包含无效 JSON") from exc
                                chunk = _extract_stream_chunk(data)
                                if chunk:
                                    full_text += chunk
                                    if invocation.runtime_options.on_stream_chunk:
                                        invocation.runtime_options.on_stream_chunk(chunk, full_text)
                                    if invocation.runtime_options.print_stream_output:
                                        print(chunk, end="", flush=True)
                except TimeoutError as exc:
                    raise httpx.ReadTimeout(
                        "AI stream attempt exceeded "
                        f"{invocation.timeout:g} seconds"
                    ) from exc

                if invocation.runtime_options.print_stream_output:
                    label = invocation.runtime_options.stream_output_label or request.model
                    print(f"\n[{label}] 流式输出完成，共 {len(full_text)} 字符\n", flush=True)
                full_text = full_text.strip()
                if not full_text:
                    raise OpenAICompatibleEmptyContentError("AI 未返回有效内容")
                return full_text
            except RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if attempt < max_retries and not full_text:
                    wait_time = _calculate_backoff(attempt)
                    logger.warning(
                        "Async stream transport failed (%s), retrying in %.1fs (%s/%s)",
                        type(exc).__name__,
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise
            finally:
                await client.aclose()

        if last_exception:
            raise last_exception
        raise RuntimeError("重试耗尽")
