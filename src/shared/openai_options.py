"""
Shared OpenAI-compatible request/execution options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Mapping, Optional, Sequence

from src.shared.ai_providers import get_provider_manifest

logger = logging.getLogger("SharedOpenAIOptions")


def _value_from_mapping(data: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _coerce_int(
    value: Any,
    default: int = 0,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class OpenAICompatibleRequestOptions:
    force_json_output: bool = False
    temperature: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_json_output": self.force_json_output,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "OpenAICompatibleRequestOptions":
        data = data or {}
        return cls(
            force_json_output=_coerce_bool(
                _value_from_mapping(
                    data,
                    "force_json_output",
                    "force_json",
                    "forceJsonOutput",
                    "forceJson",
                    default=False,
                ),
                False,
            ),
            temperature=_coerce_float(_value_from_mapping(data, "temperature", default=None), None),
        )


@dataclass
class OpenAICompatibleExecutionOptions:
    use_stream: bool = False
    rpm_limit: int = 0
    max_retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_stream": self.use_stream,
            "rpm_limit": self.rpm_limit,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "OpenAICompatibleExecutionOptions":
        data = data or {}
        return cls(
            use_stream=_coerce_bool(
                _value_from_mapping(data, "use_stream", "useStream", default=False),
                False,
            ),
            rpm_limit=_coerce_int(
                _value_from_mapping(data, "rpm_limit", "rpmLimit", default=0),
                0,
                minimum=0,
            ),
            max_retries=_coerce_int(
                _value_from_mapping(data, "max_retries", "maxRetries", default=0),
                0,
                minimum=0,
            ),
        )


@dataclass
class OpenAICompatibleOptions:
    request: OpenAICompatibleRequestOptions = field(default_factory=OpenAICompatibleRequestOptions)
    execution: OpenAICompatibleExecutionOptions = field(default_factory=OpenAICompatibleExecutionOptions)
    timeout: Optional[float] = None
    print_stream_output: bool = False
    stream_output_label: Optional[str] = None
    request_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "execution": self.execution.to_dict(),
            "timeout": self.timeout,
            "print_stream_output": self.print_stream_output,
            "stream_output_label": self.stream_output_label,
            "request_overrides": dict(self.request_overrides),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "OpenAICompatibleOptions":
        data = data or {}
        request_data = _value_from_mapping(data, "request", default={}) or {}
        execution_data = _value_from_mapping(data, "execution", default={}) or {}
        return cls(
            request=OpenAICompatibleRequestOptions.from_dict(request_data),
            execution=OpenAICompatibleExecutionOptions.from_dict(execution_data),
            timeout=_coerce_float(_value_from_mapping(data, "timeout", default=None), None),
            print_stream_output=_coerce_bool(
                _value_from_mapping(data, "print_stream_output", "printStreamOutput", default=False),
                False,
            ),
            stream_output_label=_value_from_mapping(
                data,
                "stream_output_label",
                "streamOutputLabel",
                default=None,
            ),
            request_overrides=dict(_value_from_mapping(data, "request_overrides", "requestOverrides", default={}) or {}),
        )

    def timeout_or(self, default: float) -> float:
        return float(self.timeout) if self.timeout is not None else float(default)


def build_openai_compatible_options(
    data: Mapping[str, Any],
    *,
    options_keys: Sequence[str] = ("openai_options", "openaiOptions"),
    force_json_keys: Sequence[str] = (),
    temperature_keys: Sequence[str] = (),
    use_stream_keys: Sequence[str] = (),
    rpm_limit_keys: Sequence[str] = (),
    max_retries_keys: Sequence[str] = (),
    default_force_json_output: bool = False,
    default_temperature: Optional[float] = None,
    default_use_stream: bool = False,
    default_rpm_limit: int = 0,
    default_max_retries: int = 0,
    timeout: Optional[float] = None,
    print_stream_output: bool = False,
    stream_output_label: Optional[str] = None,
    request_overrides: Optional[Mapping[str, Any]] = None,
    max_retries_maximum: Optional[int] = None,
) -> OpenAICompatibleOptions:
    nested_payload = None
    for key in options_keys:
        candidate = data.get(key)
        if isinstance(candidate, Mapping):
            nested_payload = candidate
            break

    if nested_payload is not None:
        options = OpenAICompatibleOptions.from_dict(nested_payload)
    else:
        options = OpenAICompatibleOptions(
            request=OpenAICompatibleRequestOptions(
                force_json_output=_coerce_bool(
                    _value_from_mapping(data, *force_json_keys, default=default_force_json_output),
                    default_force_json_output,
                ),
                temperature=_coerce_float(
                    _value_from_mapping(data, *temperature_keys, default=default_temperature),
                    default_temperature,
                ),
            ),
            execution=OpenAICompatibleExecutionOptions(
                use_stream=_coerce_bool(
                    _value_from_mapping(data, *use_stream_keys, default=default_use_stream),
                    default_use_stream,
                ),
                rpm_limit=_coerce_int(
                    _value_from_mapping(data, *rpm_limit_keys, default=default_rpm_limit),
                    default_rpm_limit,
                    minimum=0,
                ),
                max_retries=_coerce_int(
                    _value_from_mapping(data, *max_retries_keys, default=default_max_retries),
                    default_max_retries,
                    minimum=0,
                    maximum=max_retries_maximum,
                ),
            ),
        )

    if max_retries_maximum is not None:
        options.execution.max_retries = min(options.execution.max_retries, max_retries_maximum)

    if request_overrides:
        merged_overrides = dict(options.request_overrides)
        merged_overrides.update(request_overrides)
        options.request_overrides = merged_overrides

    options.timeout = timeout if timeout is not None else options.timeout
    options.print_stream_output = print_stream_output
    options.stream_output_label = stream_output_label
    return options


def normalize_openai_compatible_options_for_provider(
    provider: str,
    options: OpenAICompatibleOptions,
    *,
    logger_instance: Optional[logging.Logger] = None,
) -> OpenAICompatibleOptions:
    manifest = get_provider_manifest(provider)
    effective_logger = logger_instance or logger

    normalized = OpenAICompatibleOptions.from_dict(options.to_dict())

    if normalized.execution.use_stream and not manifest.supports_stream:
        effective_logger.info("%s 不支持流式调用，自动回退为非流式模式", manifest.display_name)
        normalized.execution.use_stream = False
        normalized.print_stream_output = False

    if normalized.request.force_json_output and not manifest.supports_json_response:
        effective_logger.info("%s 不支持强制 JSON 输出，自动关闭 JSON 模式", manifest.display_name)
        normalized.request.force_json_output = False

    return normalized
