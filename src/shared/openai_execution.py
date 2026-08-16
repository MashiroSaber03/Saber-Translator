"""
Runtime-only execution helpers for OpenAI-compatible request flows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

from src.shared.ai_providers import (
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
)
from src.shared.openai_options import (
    OpenAICompatibleOptions,
    clone_openai_compatible_options,
)
from src.shared.openai_rate_limits import build_openai_rpm_service_name
from src.shared.memory_errors import is_memory_allocation_error

if TYPE_CHECKING:
    from src.shared.ai_transport import (
        AsyncOpenAICompatibleTransport,
        OpenAICompatibleChatTransport,
        UnifiedChatRequest,
        UnifiedVisionRequest,
    )

logger = logging.getLogger("SharedOpenAIExecution")

T = TypeVar("T")


@dataclass
class OpenAICompatibleRuntimeOptions:
    timeout: Optional[float] = None
    print_stream_output: bool = False
    stream_output_label: Optional[str] = None
    on_stream_chunk: Optional[Callable[[str, str], None]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.timeout is not None:
            if (
                isinstance(self.timeout, bool)
                or not isinstance(self.timeout, (int, float))
                or not math.isfinite(float(self.timeout))
                or self.timeout <= 0
            ):
                raise ValueError("AI 请求超时必须是正有限数")
            self.timeout = float(self.timeout)
        if not isinstance(self.print_stream_output, bool):
            raise ValueError("print_stream_output 必须是布尔值")
        if self.stream_output_label is not None and not isinstance(
            self.stream_output_label,
            str,
        ):
            raise ValueError("stream_output_label 必须是字符串或 null")
        if self.on_stream_chunk is not None and not callable(self.on_stream_chunk):
            raise ValueError("on_stream_chunk 必须可调用")

    def timeout_or(self, default: float) -> float:
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            raise ValueError("默认 AI 请求超时必须是正数")
        resolved = self.timeout if self.timeout is not None else float(default)
        if not math.isfinite(resolved) or resolved <= 0:
            raise ValueError("默认 AI 请求超时必须是正有限数")
        return resolved


@dataclass
class ResolvedOpenAICompatibleInvocation:
    provider: str
    capability: str
    effective_options: OpenAICompatibleOptions
    runtime_options: OpenAICompatibleRuntimeOptions
    response_format: Optional[dict[str, Any]]
    use_stream: bool
    timeout: float
    service_name: str


@dataclass
class OpenAICompatibleExecutionResult(Generic[T]):
    raw_content: str
    parsed: T
    invocation: ResolvedOpenAICompatibleInvocation


class OpenAICompatibleBusinessRetryableError(RuntimeError):
    pass


class OpenAICompatibleEmptyContentError(OpenAICompatibleBusinessRetryableError):
    pass


class OpenAICompatibleBusinessRetriesExhaustedError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        last_raw_content: Optional[str] = None,
        last_error: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.last_raw_content = last_raw_content
        self.last_error = last_error


def build_openai_compatible_runtime_options(
    *,
    timeout: Optional[float] = None,
    print_stream_output: bool = False,
    stream_output_label: Optional[str] = None,
    on_stream_chunk: Optional[Callable[[str, str], None]] = None,
) -> OpenAICompatibleRuntimeOptions:
    return OpenAICompatibleRuntimeOptions(
        timeout=timeout,
        print_stream_output=print_stream_output,
        stream_output_label=stream_output_label,
        on_stream_chunk=on_stream_chunk,
    )


def clone_openai_compatible_runtime_options(
    options: OpenAICompatibleRuntimeOptions,
) -> OpenAICompatibleRuntimeOptions:
    if not isinstance(options, OpenAICompatibleRuntimeOptions):
        raise TypeError("options 必须是 OpenAICompatibleRuntimeOptions")
    return OpenAICompatibleRuntimeOptions(
        timeout=options.timeout,
        print_stream_output=options.print_stream_output,
        stream_output_label=options.stream_output_label,
        on_stream_chunk=options.on_stream_chunk,
    )


def resolve_openai_compatible_invocation(
    provider: str,
    capability: str,
    options: OpenAICompatibleOptions,
    runtime_options: Optional[OpenAICompatibleRuntimeOptions] = None,
) -> ResolvedOpenAICompatibleInvocation:
    canonical_provider = normalize_provider_id(provider)
    manifest = get_provider_manifest(canonical_provider)
    if not isinstance(capability, str) or not capability:
        raise ValueError("AI 能力必须是非空字符串")
    if not provider_supports_capability(canonical_provider, capability):
        raise ValueError(f"{manifest.display_name}不支持 {capability}")
    if not isinstance(options, OpenAICompatibleOptions):
        raise TypeError("options 必须是 OpenAICompatibleOptions")

    effective_options = clone_openai_compatible_options(options)
    runtime = clone_openai_compatible_runtime_options(
        runtime_options or OpenAICompatibleRuntimeOptions(),
    )

    if effective_options.execution.use_stream and not manifest.supports_stream:
        raise ValueError(f"{manifest.display_name}不支持流式调用")

    if effective_options.request.force_json_output and not manifest.supports_json_response:
        raise ValueError(f"{manifest.display_name}不支持强制 JSON 输出")

    return ResolvedOpenAICompatibleInvocation(
        provider=canonical_provider,
        capability=capability,
        effective_options=effective_options,
        runtime_options=runtime,
        response_format={"type": "json_object"} if effective_options.request.force_json_output else None,
        use_stream=effective_options.execution.use_stream,
        timeout=runtime.timeout_or(120.0),
        service_name=build_openai_rpm_service_name(capability, canonical_provider),
    )


def strip_markdown_code_fences(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    cleaned = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        cleaned,
        flags=re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()
    if cleaned.count("```") % 2:
        raise OpenAICompatibleBusinessRetryableError("JSON 代码块围栏不完整")
    return cleaned


def strip_reasoning_tags(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    cleaned = text
    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
        r"<thought>.*?</thought>",
        r"<reflection>.*?</reflection>",
        r"<内心独白>.*?</内心独白>",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned


def extract_json_block_from_text(text: str) -> str:
    cleaned = strip_markdown_code_fences(strip_reasoning_tags(text))
    start = next(
        (
            index
            for index, character in enumerate(cleaned)
            if character in {"{", "["}
        ),
        None,
    )
    if start is None:
        raise OpenAICompatibleBusinessRetryableError(
            "返回内容中未找到 JSON 对象或数组"
        )
    candidate = cleaned[start:]

    open_char = candidate[0]
    close_char = "}" if open_char == "{" else "]" if open_char == "[" else ""
    if not close_char:
        raise OpenAICompatibleBusinessRetryableError("返回内容中未找到有效的 JSON 起始符")

    depth = 0
    in_string = False
    escaping = False
    for index, character in enumerate(candidate):
        if escaping:
            escaping = False
            continue
        if character == "\\" and in_string:
            escaping = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == open_char:
            depth += 1
        elif character == close_char:
            depth -= 1
            if depth == 0:
                surrounding = cleaned[:start] + candidate[index + 1 :]
                if "{" in surrounding or "[" in surrounding:
                    raise OpenAICompatibleBusinessRetryableError(
                        "返回内容包含多个或不明确的 JSON 块"
                    )
                return candidate[: index + 1]

    raise OpenAICompatibleBusinessRetryableError("返回内容中未找到完整的 JSON 块")


def parse_json_block_from_text(text: str) -> Any:
    json_block = extract_json_block_from_text(text)
    try:
        return json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleBusinessRetryableError(f"JSON 解析失败: {exc}") from exc


class OpenAICompatibleSyncExecutor:
    def __init__(self, transport: Optional["OpenAICompatibleChatTransport"] = None):
        if transport is None:
            from src.shared.ai_transport import OpenAICompatibleChatTransport

            transport = OpenAICompatibleChatTransport()
        self.transport = transport

    def execute(
        self,
        request: "UnifiedChatRequest | UnifiedVisionRequest",
        *,
        capability: str,
        runtime_options: Optional[OpenAICompatibleRuntimeOptions] = None,
        parser: Optional[Callable[[str], T]] = None,
        logger_instance: Optional[logging.Logger] = None,
        before_request: Optional[Callable[[], None]] = None,
    ) -> OpenAICompatibleExecutionResult[T | str]:
        if before_request is not None and not callable(before_request):
            raise TypeError("before_request 必须可调用")
        effective_logger = logger_instance or logger
        invocation = resolve_openai_compatible_invocation(
            request.provider,
            capability,
            request.openai_options,
            runtime_options or request.runtime_options,
        )

        last_raw_content: Optional[str] = None
        last_error: Optional[BaseException] = None
        total_attempts = invocation.effective_options.execution.business_retries + 1
        for attempt in range(total_attempts):
            try:
                raw_content = self._complete(
                    request,
                    invocation,
                    before_request=before_request,
                )
                last_raw_content = raw_content
                parsed: T | str = parser(raw_content) if parser else raw_content
                return OpenAICompatibleExecutionResult(
                    raw_content=raw_content,
                    parsed=parsed,
                    invocation=invocation,
                )
            except OpenAICompatibleBusinessRetryableError as error:
                if is_memory_allocation_error(error):
                    raise
                last_error = error
                if attempt >= total_attempts - 1:
                    break
                self._log_business_retry(effective_logger, invocation, attempt, total_attempts, error)
                time.sleep(1)

        raise OpenAICompatibleBusinessRetriesExhaustedError(
            f"{invocation.service_name} 业务重试耗尽",
            last_raw_content=last_raw_content,
            last_error=last_error,
        )

    def _complete(
        self,
        request: "UnifiedChatRequest | UnifiedVisionRequest",
        invocation: ResolvedOpenAICompatibleInvocation,
        *,
        before_request: Optional[Callable[[], None]] = None,
    ) -> str:
        prepared_request = replace(
            request,
            provider=invocation.provider,
            openai_options=clone_openai_compatible_options(invocation.effective_options),
            runtime_options=clone_openai_compatible_runtime_options(invocation.runtime_options),
        )
        from src.shared.ai_transport import UnifiedVisionRequest

        if isinstance(prepared_request, UnifiedVisionRequest):
            return self.transport.complete_vision(
                prepared_request,
                resolved_invocation=invocation,
                before_request=before_request,
            )
        return self.transport.complete(
            prepared_request,
            resolved_invocation=invocation,
            before_request=before_request,
        )

    @staticmethod
    def _log_business_retry(
        logger_instance: logging.Logger,
        invocation: ResolvedOpenAICompatibleInvocation,
        attempt: int,
        total_attempts: int,
        error: BaseException,
    ) -> None:
        label = invocation.runtime_options.stream_output_label or invocation.service_name
        logger_instance.warning(
            "[%s] 业务重试 %s/%s: %s",
            label,
            attempt + 1,
            total_attempts,
            error,
        )


class OpenAICompatibleAsyncExecutor:
    def __init__(self, transport: Optional["AsyncOpenAICompatibleTransport"] = None):
        if transport is None:
            from src.shared.ai_transport import AsyncOpenAICompatibleTransport

            transport = AsyncOpenAICompatibleTransport()
        self.transport = transport

    async def execute(
        self,
        request: "UnifiedChatRequest | UnifiedVisionRequest",
        *,
        capability: str,
        runtime_options: Optional[OpenAICompatibleRuntimeOptions] = None,
        parser: Optional[Callable[[str], T]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> OpenAICompatibleExecutionResult[T | str]:
        effective_logger = logger_instance or logger
        invocation = resolve_openai_compatible_invocation(
            request.provider,
            capability,
            request.openai_options,
            runtime_options or request.runtime_options,
        )
        last_raw_content: Optional[str] = None
        last_error: Optional[BaseException] = None
        total_attempts = invocation.effective_options.execution.business_retries + 1
        for attempt in range(total_attempts):
            try:
                raw_content = await self._complete(request, invocation)
                last_raw_content = raw_content
                parsed: T | str = parser(raw_content) if parser else raw_content
                return OpenAICompatibleExecutionResult(
                    raw_content=raw_content,
                    parsed=parsed,
                    invocation=invocation,
                )
            except OpenAICompatibleBusinessRetryableError as error:
                if is_memory_allocation_error(error):
                    raise
                last_error = error
                if attempt >= total_attempts - 1:
                    break
                OpenAICompatibleSyncExecutor._log_business_retry(
                    effective_logger,
                    invocation,
                    attempt,
                    total_attempts,
                    error,
                )
                await asyncio.sleep(1)

        raise OpenAICompatibleBusinessRetriesExhaustedError(
            f"{invocation.service_name} 业务重试耗尽",
            last_raw_content=last_raw_content,
            last_error=last_error,
        )

    async def _complete(
        self,
        request: "UnifiedChatRequest | UnifiedVisionRequest",
        invocation: ResolvedOpenAICompatibleInvocation,
    ) -> str:
        prepared_request = replace(
            request,
            provider=invocation.provider,
            openai_options=clone_openai_compatible_options(invocation.effective_options),
            runtime_options=clone_openai_compatible_runtime_options(invocation.runtime_options),
        )
        from src.shared.ai_transport import UnifiedVisionRequest

        if isinstance(prepared_request, UnifiedVisionRequest):
            return await self.transport.complete_vision(
                prepared_request,
                resolved_invocation=invocation,
            )
        return await self.transport.complete(
            prepared_request,
            resolved_invocation=invocation,
        )
