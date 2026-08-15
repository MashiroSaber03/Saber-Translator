"""Strict runtime configuration models for Manga Insight providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any

from src.shared.openai_options import (
    OpenAICompatibleExecutionOptions,
    OpenAICompatibleOptions,
    OpenAICompatibleRequestOptions,
)

_VLM_FIELDS = {
    "provider",
    "api_key",
    "model",
    "base_url",
    "credential_version_id",
    "openai_options",
    "image_max_size",
}
_CHAT_FIELDS = _VLM_FIELDS - {"image_max_size"}
_EMBEDDING_FIELDS = {
    "provider",
    "api_key",
    "model",
    "base_url",
    "credential_version_id",
    "rpm_limit",
    "transport_retries",
    "business_retries",
    "timeout_seconds",
}
_IMAGE_GEN_FIELDS = _EMBEDDING_FIELDS - {"rpm_limit"}
_RERANKER_FIELDS = _IMAGE_GEN_FIELDS
def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{name} fields are invalid ({'; '.join(details)})")
    return value


def _string(value: object, *, name: str, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _options(value: object, *, name: str) -> OpenAICompatibleOptions:
    if not isinstance(value, OpenAICompatibleOptions):
        raise TypeError(f"{name} must be OpenAICompatibleOptions")
    return OpenAICompatibleOptions.from_dict(value.to_dict())


@dataclass
class VLMConfig:
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    base_url: str | None = None
    credential_version_id: str | None = None
    openai_options: OpenAICompatibleOptions = field(
        default_factory=lambda: OpenAICompatibleOptions(
            request=OpenAICompatibleRequestOptions(
                force_json_output=False,
                temperature=0.3,
            ),
            execution=OpenAICompatibleExecutionOptions(
                use_stream=True,
                rpm_limit=0,
                transport_retries=1,
                business_retries=0,
            ),
        )
    )
    image_max_size: int = 0

    def __post_init__(self) -> None:
        self.provider = _string(self.provider, name="vlm.provider", allow_empty=False)
        self.api_key = _string(self.api_key, name="vlm.api_key")
        self.model = _string(self.model, name="vlm.model", allow_empty=False)
        self.base_url = _optional_string(self.base_url, name="vlm.base_url")
        self.credential_version_id = _optional_string(
            self.credential_version_id,
            name="vlm.credential_version_id",
        )
        self.openai_options = _options(
            self.openai_options,
            name="vlm.openai_options",
        )
        self.image_max_size = _nonnegative_integer(
            self.image_max_size,
            name="vlm.image_max_size",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "credential_version_id": self.credential_version_id,
            "openai_options": self.openai_options.to_dict(),
            "image_max_size": self.image_max_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> "VLMConfig":
        data = _mapping(value, name="vlm", keys=_VLM_FIELDS)
        options = data["openai_options"]
        if not isinstance(options, Mapping):
            raise TypeError("vlm.openai_options must be an object")
        return cls(
            provider=data["provider"],
            api_key=data["api_key"],
            model=data["model"],
            base_url=data["base_url"],
            credential_version_id=data["credential_version_id"],
            openai_options=OpenAICompatibleOptions.from_dict(options),
            image_max_size=data["image_max_size"],
        )


@dataclass
class ChatLLMConfig:
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    base_url: str | None = None
    credential_version_id: str | None = None
    openai_options: OpenAICompatibleOptions = field(
        default_factory=lambda: OpenAICompatibleOptions(
            request=OpenAICompatibleRequestOptions(),
            execution=OpenAICompatibleExecutionOptions(
                use_stream=True,
                rpm_limit=0,
                transport_retries=1,
                business_retries=0,
            ),
        )
    )

    def __post_init__(self) -> None:
        self.provider = _string(self.provider, name="chat.provider", allow_empty=False)
        self.api_key = _string(self.api_key, name="chat.api_key")
        self.model = _string(self.model, name="chat.model", allow_empty=False)
        self.base_url = _optional_string(self.base_url, name="chat.base_url")
        self.credential_version_id = _optional_string(
            self.credential_version_id,
            name="chat.credential_version_id",
        )
        self.openai_options = _options(
            self.openai_options,
            name="chat.openai_options",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "credential_version_id": self.credential_version_id,
            "openai_options": self.openai_options.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ChatLLMConfig":
        data = _mapping(value, name="chat", keys=_CHAT_FIELDS)
        options = data["openai_options"]
        if not isinstance(options, Mapping):
            raise TypeError("chat.openai_options must be an object")
        return cls(
            provider=data["provider"],
            api_key=data["api_key"],
            model=data["model"],
            base_url=data["base_url"],
            credential_version_id=data["credential_version_id"],
            openai_options=OpenAICompatibleOptions.from_dict(options),
        )


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    credential_version_id: str | None = None
    rpm_limit: int = 0
    transport_retries: int = 1
    business_retries: int = 0
    timeout_seconds: float = 0

    def __post_init__(self) -> None:
        self.provider = _string(
            self.provider,
            name="embedding.provider",
            allow_empty=False,
        )
        self.api_key = _string(self.api_key, name="embedding.api_key")
        self.model = _string(
            self.model,
            name="embedding.model",
            allow_empty=False,
        )
        self.base_url = _optional_string(self.base_url, name="embedding.base_url")
        self.credential_version_id = _optional_string(
            self.credential_version_id,
            name="embedding.credential_version_id",
        )
        self.rpm_limit = _nonnegative_integer(
            self.rpm_limit,
            name="embedding.rpm_limit",
        )
        self.transport_retries = _nonnegative_integer(
            self.transport_retries,
            name="embedding.transport_retries",
        )
        self.business_retries = _nonnegative_integer(
            self.business_retries,
            name="embedding.business_retries",
        )
        self.timeout_seconds = _nonnegative_number(
            self.timeout_seconds,
            name="embedding.timeout_seconds",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "credential_version_id": self.credential_version_id,
            "rpm_limit": self.rpm_limit,
            "transport_retries": self.transport_retries,
            "business_retries": self.business_retries,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EmbeddingConfig":
        data = _mapping(value, name="embedding", keys=_EMBEDDING_FIELDS)
        return cls(**data)


@dataclass
class RerankerConfig:
    provider: str = "jina"
    api_key: str = ""
    model: str = "jina-reranker-v2-base-multilingual"
    base_url: str | None = None
    credential_version_id: str | None = None
    transport_retries: int = 1
    business_retries: int = 0
    timeout_seconds: float = 0

    def __post_init__(self) -> None:
        self.provider = _string(
            self.provider,
            name="reranker.provider",
            allow_empty=False,
        )
        self.api_key = _string(self.api_key, name="reranker.api_key")
        self.model = _string(
            self.model,
            name="reranker.model",
            allow_empty=False,
        )
        self.base_url = _optional_string(self.base_url, name="reranker.base_url")
        self.credential_version_id = _optional_string(
            self.credential_version_id,
            name="reranker.credential_version_id",
        )
        self.transport_retries = _nonnegative_integer(
            self.transport_retries,
            name="reranker.transport_retries",
        )
        self.business_retries = _nonnegative_integer(
            self.business_retries,
            name="reranker.business_retries",
        )
        self.timeout_seconds = _nonnegative_number(
            self.timeout_seconds,
            name="reranker.timeout_seconds",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "credential_version_id": self.credential_version_id,
            "transport_retries": self.transport_retries,
            "business_retries": self.business_retries,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RerankerConfig":
        data = _mapping(value, name="reranker", keys=_RERANKER_FIELDS)
        return cls(**data)


@dataclass
class ImageGenConfig:
    provider: str = "gpt2api"
    api_key: str = ""
    model: str = "gpt-image-2"
    base_url: str | None = None
    credential_version_id: str | None = None
    transport_retries: int = 1
    business_retries: int = 0
    timeout_seconds: float = 0

    def __post_init__(self) -> None:
        self.provider = _string(
            self.provider,
            name="image_gen.provider",
            allow_empty=False,
        )
        self.api_key = _string(self.api_key, name="image_gen.api_key")
        self.model = _string(
            self.model,
            name="image_gen.model",
            allow_empty=False,
        )
        self.base_url = _optional_string(self.base_url, name="image_gen.base_url")
        self.credential_version_id = _optional_string(
            self.credential_version_id,
            name="image_gen.credential_version_id",
        )
        self.transport_retries = _nonnegative_integer(
            self.transport_retries,
            name="image_gen.transport_retries",
        )
        self.business_retries = _nonnegative_integer(
            self.business_retries,
            name="image_gen.business_retries",
        )
        self.timeout_seconds = _nonnegative_number(
            self.timeout_seconds,
            name="image_gen.timeout_seconds",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "credential_version_id": self.credential_version_id,
            "transport_retries": self.transport_retries,
            "business_retries": self.business_retries,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ImageGenConfig":
        data = _mapping(value, name="image_gen", keys=_IMAGE_GEN_FIELDS)
        return cls(**data)
