"""
Shared persistent OpenAI-compatible request/execution options.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES = 1
_OPENAI_OPTIONS_TOP_LEVEL_KEYS = {"request", "execution"}
_OPENAI_OPTIONS_REQUEST_KEYS = {"force_json_output", "temperature", "extra_body"}
_OPENAI_OPTIONS_EXECUTION_KEYS = {
    "use_stream",
    "rpm_limit",
    "transport_retries",
    "business_retries",
}
_OPENAI_EXTRA_BODY_RESERVED_KEYS = {
    "model",
    "messages",
    "temperature",
    "response_format",
    "stream",
}


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


def _clone_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return {}


def _validate_extra_body_payload(value: Any, *, prefix: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [prefix]

    invalid_keys: list[str] = []
    for key in value.keys():
        if key in _OPENAI_EXTRA_BODY_RESERVED_KEYS:
            invalid_keys.append(f"{prefix}.{key}")
    return invalid_keys


def validate_and_clone_openai_extra_body(
    value: Any,
    *,
    prefix: str = "openai_options.request.extra_body",
) -> dict[str, Any]:
    invalid_keys = _validate_extra_body_payload(value, prefix=prefix)
    if invalid_keys:
        if invalid_keys == [prefix]:
            raise ValueError(f"{prefix} 必须是 JSON 对象")

        reserved_keys = ", ".join(key.split(".")[-1] for key in invalid_keys)
        raise ValueError(f"{prefix} 包含不允许覆盖的保留字段: {reserved_keys}")

    return _clone_mapping(value)


@dataclass
class OpenAICompatibleRequestOptions:
    force_json_output: bool = False
    temperature: Optional[float] = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_json_output": self.force_json_output,
            "temperature": self.temperature,
            "extra_body": _clone_mapping(self.extra_body),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "OpenAICompatibleRequestOptions":
        data = data or {}
        return cls(
            force_json_output=_coerce_bool(data.get("force_json_output"), False),
            temperature=_coerce_float(data.get("temperature"), None),
            extra_body=validate_and_clone_openai_extra_body(data.get("extra_body")),
        )


@dataclass
class OpenAICompatibleExecutionOptions:
    use_stream: bool = False
    rpm_limit: int = 0
    transport_retries: int = DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES
    business_retries: int = 0

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
            use_stream=_coerce_bool(data.get("use_stream"), False),
            rpm_limit=_coerce_int(data.get("rpm_limit"), 0, minimum=0),
            transport_retries=_coerce_int(
                data.get("transport_retries"),
                DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
                minimum=0,
            ),
            business_retries=_coerce_int(
                data.get("business_retries"),
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


def create_openai_compatible_options(
    *,
    force_json_output: bool = False,
    temperature: Optional[float] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
    use_stream: bool = False,
    rpm_limit: int = 0,
    transport_retries: int = DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
    business_retries: int = 0,
) -> OpenAICompatibleOptions:
    return OpenAICompatibleOptions(
        request=OpenAICompatibleRequestOptions(
            force_json_output=force_json_output,
            temperature=temperature,
            extra_body=validate_and_clone_openai_extra_body(extra_body),
        ),
        execution=OpenAICompatibleExecutionOptions(
            use_stream=use_stream,
            rpm_limit=max(0, int(rpm_limit or 0)),
            transport_retries=max(0, int(transport_retries or 0)),
            business_retries=max(0, int(business_retries or 0)),
        ),
    )


def merge_openai_compatible_options(
    payload: Optional[Mapping[str, Any]],
    *,
    defaults: Optional[OpenAICompatibleOptions] = None,
    business_retries_maximum: Optional[int] = None,
) -> OpenAICompatibleOptions:
    options = clone_openai_compatible_options(defaults or OpenAICompatibleOptions())
    if not isinstance(payload, Mapping):
        return options

    request_data = payload.get("request")
    if isinstance(request_data, Mapping):
        if "force_json_output" in request_data:
            options.request.force_json_output = _coerce_bool(
                request_data.get("force_json_output"),
                options.request.force_json_output,
            )
        if "temperature" in request_data:
            options.request.temperature = _coerce_float(
                request_data.get("temperature"),
                options.request.temperature,
            )
        if "extra_body" in request_data:
            options.request.extra_body = validate_and_clone_openai_extra_body(
                request_data.get("extra_body")
            )

    execution_data = payload.get("execution")
    if isinstance(execution_data, Mapping):
        if "use_stream" in execution_data:
            options.execution.use_stream = _coerce_bool(
                execution_data.get("use_stream"),
                options.execution.use_stream,
            )
        if "rpm_limit" in execution_data:
            options.execution.rpm_limit = _coerce_int(
                execution_data.get("rpm_limit"),
                options.execution.rpm_limit,
                minimum=0,
            )
        if "transport_retries" in execution_data:
            options.execution.transport_retries = _coerce_int(
                execution_data.get("transport_retries"),
                options.execution.transport_retries,
                minimum=0,
            )
        if "business_retries" in execution_data:
            options.execution.business_retries = _coerce_int(
                execution_data.get("business_retries"),
                options.execution.business_retries,
                minimum=0,
            )

    if business_retries_maximum is not None:
        options.execution.business_retries = min(
            options.execution.business_retries,
            business_retries_maximum,
        )

    return options


def validate_openai_options_payload(payload: Optional[Mapping[str, Any]]) -> list[str]:
    if payload is None:
        return []
    if not isinstance(payload, Mapping):
        return ["openai_options"]

    invalid_keys: list[str] = []
    for key in payload.keys():
        if key not in _OPENAI_OPTIONS_TOP_LEVEL_KEYS:
            invalid_keys.append(f"openai_options.{key}")

    request_data = payload.get("request")
    if request_data is not None:
        if not isinstance(request_data, Mapping):
            invalid_keys.append("openai_options.request")
        else:
            for key in request_data.keys():
                if key not in _OPENAI_OPTIONS_REQUEST_KEYS:
                    invalid_keys.append(f"openai_options.request.{key}")
            invalid_keys.extend(
                _validate_extra_body_payload(
                    request_data.get("extra_body"),
                    prefix="openai_options.request.extra_body",
                )
            )

    execution_data = payload.get("execution")
    if execution_data is not None:
        if not isinstance(execution_data, Mapping):
            invalid_keys.append("openai_options.execution")
        else:
            for key in execution_data.keys():
                if key not in _OPENAI_OPTIONS_EXECUTION_KEYS:
                    invalid_keys.append(f"openai_options.execution.{key}")

    return invalid_keys
