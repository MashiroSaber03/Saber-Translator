"""
Runtime-only execution helpers for OpenAI-compatible request flows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generic, Optional, TypeVar

from src.shared.ai_providers import get_provider_manifest, normalize_provider_id
from src.shared.openai_options import (
    OpenAICompatibleOptions,
    clone_openai_compatible_options,
)
from src.shared.openai_rate_limits import (
    SharedRPMLimiter,
    apply_sync_rpm_limit,
    build_openai_rpm_bucket_key,
    build_openai_rpm_service_name,
    enforce_sync_rpm_limit_window,
)

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
    request_overrides: dict[str, Any] = field(default_factory=dict)
    on_stream_chunk: Optional[Callable[[str, str], None]] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout": self.timeout,
            "print_stream_output": self.print_stream_output,
            "stream_output_label": self.stream_output_label,
            "request_overrides": dict(self.request_overrides),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "OpenAICompatibleRuntimeOptions":
        data = data or {}
        return cls(
            timeout=float(data["timeout"]) if data.get("timeout") is not None else None,
            print_stream_output=bool(data.get("print_stream_output", False)),
            stream_output_label=data.get("stream_output_label"),
            request_overrides=dict(data.get("request_overrides") or {}),
        )

    def timeout_or(self, default: float) -> float:
        return float(self.timeout) if self.timeout is not None else float(default)


@dataclass
class ResolvedOpenAICompatibleInvocation:
    provider: str
    capability: str
    effective_options: OpenAICompatibleOptions
    runtime_options: OpenAICompatibleRuntimeOptions
    response_format: Optional[dict[str, Any]]
    use_stream: bool
    timeout: float
    bucket_key: str
    service_name: str


@dataclass
class OpenAICompatibleExecutionResult(Generic[T]):
    raw_content: str
    parsed: T
    invocation: ResolvedOpenAICompatibleInvocation


class OpenAICompatibleBusinessRetryableError(RuntimeError):
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
    request_overrides: Optional[dict[str, Any]] = None,
    on_stream_chunk: Optional[Callable[[str, str], None]] = None,
) -> OpenAICompatibleRuntimeOptions:
    return OpenAICompatibleRuntimeOptions(
        timeout=timeout,
        print_stream_output=print_stream_output,
        stream_output_label=stream_output_label,
        request_overrides=dict(request_overrides or {}),
        on_stream_chunk=on_stream_chunk,
    )


def clone_openai_compatible_runtime_options(
    options: OpenAICompatibleRuntimeOptions,
) -> OpenAICompatibleRuntimeOptions:
    return OpenAICompatibleRuntimeOptions(
        timeout=options.timeout,
        print_stream_output=options.print_stream_output,
        stream_output_label=options.stream_output_label,
        request_overrides=dict(options.request_overrides),
        on_stream_chunk=options.on_stream_chunk,
    )


def resolve_openai_compatible_invocation(
    provider: str,
    capability: str,
    options: OpenAICompatibleOptions,
    runtime_options: Optional[OpenAICompatibleRuntimeOptions] = None,
    *,
    logger_instance: Optional[logging.Logger] = None,
) -> ResolvedOpenAICompatibleInvocation:
    canonical_provider = normalize_provider_id(provider)
    manifest = get_provider_manifest(canonical_provider)
    effective_logger = logger_instance or logger

    effective_options = clone_openai_compatible_options(options)
    runtime = clone_openai_compatible_runtime_options(
        runtime_options or OpenAICompatibleRuntimeOptions(),
    )

    if effective_options.execution.use_stream and not manifest.supports_stream:
        effective_logger.info("%s 不支持流式调用，自动回退为非流式模式", manifest.display_name)
        effective_options.execution.use_stream = False
        runtime.print_stream_output = False

    if effective_options.request.force_json_output and not manifest.supports_json_response:
        effective_logger.info("%s 不支持强制 JSON 输出，自动关闭 JSON 模式", manifest.display_name)
        effective_options.request.force_json_output = False

    return ResolvedOpenAICompatibleInvocation(
        provider=canonical_provider,
        capability=capability,
        effective_options=effective_options,
        runtime_options=runtime,
        response_format={"type": "json_object"} if effective_options.request.force_json_output else None,
        use_stream=effective_options.execution.use_stream,
        timeout=runtime.timeout_or(120.0),
        bucket_key=build_openai_rpm_bucket_key(capability, canonical_provider),
        service_name=build_openai_rpm_service_name(capability, canonical_provider),
    )


def strip_markdown_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def extract_json_block_from_text(text: str) -> str:
    cleaned = strip_markdown_code_fences(text)
    if cleaned.startswith("{") or cleaned.startswith("["):
        candidate = cleaned
    else:
        start_index = -1
        for index, character in enumerate(cleaned):
            if character in "{[":
                start_index = index
                break
        if start_index < 0:
            raise OpenAICompatibleBusinessRetryableError("返回内容中未找到有效的 JSON 块")
        candidate = cleaned[start_index:]

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
        if character == "\\":
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
                return candidate[: index + 1]

    raise OpenAICompatibleBusinessRetryableError("返回内容中未找到完整的 JSON 块")


def parse_json_block_from_text(text: str) -> Any:
    json_block = extract_json_block_from_text(text)
    try:
        return json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleBusinessRetryableError(f"JSON 解析失败: {exc}") from exc


def _is_empty_content_error(error: BaseException) -> bool:
    message = str(error or "").lower()
    return "未返回有效内容" in message or "empty choices" in message


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
    ) -> OpenAICompatibleExecutionResult[T | str]:
        effective_logger = logger_instance or logger
        invocation = resolve_openai_compatible_invocation(
            request.provider,
            capability,
            request.openai_options,
            runtime_options or request.runtime_options,
            logger_instance=effective_logger,
        )

        def before_request() -> None:
            apply_sync_rpm_limit(
                invocation.bucket_key,
                invocation.effective_options.execution.rpm_limit,
                invocation.service_name,
                enforce_sync_rpm_limit_window,
            )

        last_raw_content: Optional[str] = None
        last_error: Optional[BaseException] = None
        total_attempts = invocation.effective_options.execution.business_retries + 1
        for attempt in range(total_attempts):
            try:
                raw_content = self._complete(request, invocation, before_request)
                last_raw_content = raw_content
                parsed: T | str = parser(raw_content) if parser else raw_content
                return OpenAICompatibleExecutionResult(
                    raw_content=raw_content,
                    parsed=parsed,
                    invocation=invocation,
                )
            except OpenAICompatibleBusinessRetryableError as error:
                last_error = error
                if attempt >= total_attempts - 1:
                    break
                self._log_business_retry(effective_logger, invocation, attempt, total_attempts, error)
                time.sleep(1)
            except Exception as error:
                if not _is_empty_content_error(error):
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
        before_request: Callable[[], None],
    ) -> str:
        prepared_request = replace(
            request,
            provider=invocation.provider,
            openai_options=clone_openai_compatible_options(invocation.effective_options),
            runtime_options=clone_openai_compatible_runtime_options(invocation.runtime_options),
        )
        if prepared_request.__class__.__name__ == "UnifiedVisionRequest":
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
            logger_instance=effective_logger,
        )
        limiter = SharedRPMLimiter(
            invocation.effective_options.execution.rpm_limit,
            bucket_id=invocation.bucket_key,
        )

        async def before_request() -> None:
            await limiter.wait()

        last_raw_content: Optional[str] = None
        last_error: Optional[BaseException] = None
        total_attempts = invocation.effective_options.execution.business_retries + 1
        for attempt in range(total_attempts):
            try:
                raw_content = await self._complete(request, invocation, before_request)
                last_raw_content = raw_content
                parsed: T | str = parser(raw_content) if parser else raw_content
                return OpenAICompatibleExecutionResult(
                    raw_content=raw_content,
                    parsed=parsed,
                    invocation=invocation,
                )
            except OpenAICompatibleBusinessRetryableError as error:
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
            except Exception as error:
                if not _is_empty_content_error(error):
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
        before_request: Callable[[], Awaitable[None]],
    ) -> str:
        prepared_request = replace(
            request,
            provider=invocation.provider,
            openai_options=clone_openai_compatible_options(invocation.effective_options),
            runtime_options=clone_openai_compatible_runtime_options(invocation.runtime_options),
        )
        if prepared_request.__class__.__name__ == "UnifiedVisionRequest":
            return await self.transport.complete_vision(
                prepared_request,
                resolved_invocation=invocation,
                before_request=before_request,
            )
        return await self.transport.complete(
            prepared_request,
            resolved_invocation=invocation,
            before_request=before_request,
        )
