"""
Shared persistent OpenAI-compatible request/execution options.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES = 1
_OPENAI_EXTRA_BODY_RESERVED_KEYS = {
    "model",
    "messages",
    "temperature",
    "response_format",
    "stream",
}
_REQUEST_OPTION_FIELDS = {"force_json_output", "temperature", "extra_body"}
_EXECUTION_OPTION_FIELDS = {
    "use_stream",
    "rpm_limit",
    "transport_retries",
    "business_retries",
}
_OPTION_FIELDS = {"request", "execution"}


def _require_exact_mapping(
    value: Any,
    *,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} 必须是完整的当前结构")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return value


def _require_temperature(value: Any) -> Optional[float]:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 2
    ):
        raise ValueError("openai_options.request.temperature 必须是 0 到 2 的有限数值")
    return float(value)


def _clone_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("openai_options.request.extra_body 必须是 JSON 对象")
    return copy.deepcopy(dict(value))


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


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{path} 必须是有限数值")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} 的键必须是字符串")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} 必须是 JSON 可序列化值")


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

    _validate_json_value(value, path=prefix)
    return _clone_mapping(value)


@dataclass
class OpenAICompatibleRequestOptions:
    force_json_output: bool = False
    temperature: Optional[float] = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.force_json_output = _require_bool(
            self.force_json_output,
            name="openai_options.request.force_json_output",
        )
        self.temperature = _require_temperature(self.temperature)
        if not isinstance(self.extra_body, Mapping):
            raise ValueError("openai_options.request.extra_body 必须是 JSON 对象")
        self.extra_body = validate_and_clone_openai_extra_body(self.extra_body)

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_json_output": self.force_json_output,
            "temperature": self.temperature,
            "extra_body": _clone_mapping(self.extra_body),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpenAICompatibleRequestOptions":
        data = _require_exact_mapping(
            data,
            fields=_REQUEST_OPTION_FIELDS,
            name="openai_options.request",
        )
        if not isinstance(data["extra_body"], Mapping):
            raise ValueError("openai_options.request.extra_body 必须是 JSON 对象")
        return cls(
            force_json_output=data["force_json_output"],
            temperature=data["temperature"],
            extra_body=data["extra_body"],
        )


@dataclass
class OpenAICompatibleExecutionOptions:
    use_stream: bool = False
    rpm_limit: int = 0
    transport_retries: int = DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES
    business_retries: int = 0

    def __post_init__(self) -> None:
        self.use_stream = _require_bool(
            self.use_stream,
            name="openai_options.execution.use_stream",
        )
        self.rpm_limit = _require_nonnegative_int(
            self.rpm_limit,
            name="openai_options.execution.rpm_limit",
        )
        self.transport_retries = _require_nonnegative_int(
            self.transport_retries,
            name="openai_options.execution.transport_retries",
        )
        self.business_retries = _require_nonnegative_int(
            self.business_retries,
            name="openai_options.execution.business_retries",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_stream": self.use_stream,
            "rpm_limit": self.rpm_limit,
            "transport_retries": self.transport_retries,
            "business_retries": self.business_retries,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpenAICompatibleExecutionOptions":
        data = _require_exact_mapping(
            data,
            fields=_EXECUTION_OPTION_FIELDS,
            name="openai_options.execution",
        )
        return cls(
            use_stream=data["use_stream"],
            rpm_limit=data["rpm_limit"],
            transport_retries=data["transport_retries"],
            business_retries=data["business_retries"],
        )


@dataclass
class OpenAICompatibleOptions:
    request: OpenAICompatibleRequestOptions = field(default_factory=OpenAICompatibleRequestOptions)
    execution: OpenAICompatibleExecutionOptions = field(default_factory=OpenAICompatibleExecutionOptions)

    def __post_init__(self) -> None:
        if not isinstance(self.request, OpenAICompatibleRequestOptions):
            raise TypeError("openai_options.request 类型错误")
        if not isinstance(self.execution, OpenAICompatibleExecutionOptions):
            raise TypeError("openai_options.execution 类型错误")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "execution": self.execution.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpenAICompatibleOptions":
        data = _require_exact_mapping(
            data,
            fields=_OPTION_FIELDS,
            name="openai_options",
        )
        return cls(
            request=OpenAICompatibleRequestOptions.from_dict(data["request"]),
            execution=OpenAICompatibleExecutionOptions.from_dict(data["execution"]),
        )


def clone_openai_compatible_options(options: OpenAICompatibleOptions) -> OpenAICompatibleOptions:
    if not isinstance(options, OpenAICompatibleOptions):
        raise TypeError("options 必须是 OpenAICompatibleOptions")
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
            extra_body={} if extra_body is None else extra_body,
        ),
        execution=OpenAICompatibleExecutionOptions(
            use_stream=use_stream,
            rpm_limit=rpm_limit,
            transport_retries=transport_retries,
            business_retries=business_retries,
        ),
    )
