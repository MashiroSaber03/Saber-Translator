"""
Shared persistent OpenAI-compatible request/execution options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES = 1


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
    transport_retries: int = DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES
    business_retries: int = 0

    @property
    def max_retries(self) -> int:
        return self.business_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self.business_retries = max(0, int(value or 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_stream": self.use_stream,
            "rpm_limit": self.rpm_limit,
            "transport_retries": self.transport_retries,
            "business_retries": self.business_retries,
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
            transport_retries=_coerce_int(
                _value_from_mapping(
                    data,
                    "transport_retries",
                    "transportRetries",
                    default=DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
                ),
                DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
                minimum=0,
            ),
            business_retries=_coerce_int(
                _value_from_mapping(
                    data,
                    "business_retries",
                    "businessRetries",
                    "max_retries",
                    "maxRetries",
                    default=0,
                ),
                0,
                minimum=0,
            ),
        )


@dataclass
class OpenAICompatibleOptions:
    request: OpenAICompatibleRequestOptions = field(default_factory=OpenAICompatibleRequestOptions)
    execution: OpenAICompatibleExecutionOptions = field(default_factory=OpenAICompatibleExecutionOptions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "execution": self.execution.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "OpenAICompatibleOptions":
        data = data or {}
        request_data = _value_from_mapping(data, "request", default={}) or {}
        execution_data = _value_from_mapping(data, "execution", default={}) or {}
        return cls(
            request=OpenAICompatibleRequestOptions.from_dict(request_data),
            execution=OpenAICompatibleExecutionOptions.from_dict(execution_data),
        )


def clone_openai_compatible_options(options: OpenAICompatibleOptions) -> OpenAICompatibleOptions:
    return OpenAICompatibleOptions.from_dict(options.to_dict())


def build_openai_compatible_options(
    data: Mapping[str, Any],
    *,
    options_keys: Sequence[str] = ("openai_options", "openaiOptions"),
    force_json_keys: Sequence[str] = (),
    temperature_keys: Sequence[str] = (),
    use_stream_keys: Sequence[str] = (),
    rpm_limit_keys: Sequence[str] = (),
    transport_retries_keys: Sequence[str] = (),
    business_retries_keys: Sequence[str] = (),
    max_retries_keys: Sequence[str] = (),
    default_force_json_output: bool = False,
    default_temperature: Optional[float] = None,
    default_use_stream: bool = False,
    default_rpm_limit: int = 0,
    default_transport_retries: int = DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
    default_business_retries: int = 0,
    business_retries_maximum: Optional[int] = None,
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
                transport_retries=_coerce_int(
                    _value_from_mapping(
                        data,
                        *transport_retries_keys,
                        default=default_transport_retries,
                    ),
                    default_transport_retries,
                    minimum=0,
                ),
                business_retries=_coerce_int(
                    _value_from_mapping(
                        data,
                        *business_retries_keys,
                        *max_retries_keys,
                        default=default_business_retries,
                    ),
                    default_business_retries,
                    minimum=0,
                    maximum=business_retries_maximum,
                ),
            ),
        )

    if business_retries_maximum is not None:
        options.execution.business_retries = min(
            options.execution.business_retries,
            business_retries_maximum,
        )

    return options
