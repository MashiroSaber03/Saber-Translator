"""Static plugin v3 manifest and hook data contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import base64
import re
from typing import Any


PLUGIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
PACKAGE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
HOOK_STEPS = (
    "job",
    "pipeline",
    "detect",
    "ocr",
    "color",
    "translate",
    "ai_translate",
    "inpaint",
    "render",
)
HOOK_NAMES = frozenset(
    f"{phase}_{step}"
    for step in HOOK_STEPS
    for phase in ("before", "after")
)
ATOMIC_STEPS = frozenset(HOOK_STEPS[2:])
PLUGIN_MODES = frozenset(
    {"standard", "hq", "proofread", "remove_text"}
)
CONFIG_TYPES = frozenset({"text", "number", "boolean", "select"})


class PluginContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    display_name: str
    package_version: str
    entrypoint: str
    hooks: tuple[str, ...]
    supported_steps: tuple[str, ...]
    supported_modes: tuple[str, ...]
    priority: int
    failure_policy: str
    author: str
    description: str
    default_enabled: bool
    config_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "plugin_id": self.plugin_id,
            "display_name": self.display_name,
            "package_version": self.package_version,
            "entrypoint": self.entrypoint,
            "hooks": list(self.hooks),
            "supported_steps": list(self.supported_steps),
            "supported_modes": list(self.supported_modes),
            "priority": self.priority,
            "failure_policy": self.failure_policy,
            "author": self.author,
            "description": self.description,
            "default_enabled": self.default_enabled,
            "config_schema": self.config_schema,
        }


@dataclass(frozen=True, slots=True)
class PluginContext:
    job_id: str | None
    batch_id: str | None
    book_id: str | None
    chapter_id: str | None
    page_id: str | None
    mode: str
    step: str
    scope: str
    config: Mapping[str, Any]
    repository: object
    assets: object
    logger: object


def parse_manifest(raw: Mapping[str, Any]) -> PluginManifest:
    schema_version = raw.get("schema_version", raw.get("schemaVersion"))
    if schema_version != 3:
        raise PluginContractError("plugin manifest schema_version must be 3")
    plugin_id = str(raw.get("plugin_id", "")).strip()
    if not PLUGIN_ID_PATTERN.fullmatch(plugin_id):
        raise PluginContractError("plugin_id is invalid")
    display_name = str(raw.get("display_name", "")).strip()
    if not display_name or len(display_name) > 200:
        raise PluginContractError(
            "display_name must contain 1-200 characters"
        )
    package_version = str(raw.get("package_version", "")).strip()
    if not PACKAGE_VERSION_PATTERN.fullmatch(package_version):
        raise PluginContractError("package_version is invalid")
    entrypoint = str(raw.get("entrypoint", "")).strip()
    _validate_entrypoint(entrypoint)
    hooks = _string_tuple(raw.get("hooks"), "hooks")
    if not hooks or any(value not in HOOK_NAMES for value in hooks):
        raise PluginContractError("manifest hooks contain unsupported values")
    if len(set(hooks)) != len(hooks):
        raise PluginContractError("manifest hooks must be unique")
    supported_steps = _string_tuple(
        raw.get("supported_steps", []),
        "supported_steps",
    )
    if not supported_steps:
        supported_steps = tuple(
            dict.fromkeys(value.split("_", 1)[1] for value in hooks)
        )
    if any(value not in HOOK_STEPS for value in supported_steps):
        raise PluginContractError("supported_steps contains invalid values")
    supported_modes = _string_tuple(
        raw.get("supported_modes", list(PLUGIN_MODES)),
        "supported_modes",
    )
    if not supported_modes or any(
        value not in PLUGIN_MODES for value in supported_modes
    ):
        raise PluginContractError("supported_modes contains invalid values")
    priority = raw.get("priority", 100)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise PluginContractError("priority must be an integer")
    if not -10_000 <= priority <= 10_000:
        raise PluginContractError("priority is out of range")
    failure_policy = str(raw.get("failure_policy", "continue")).strip()
    if failure_policy not in {"continue", "fail"}:
        raise PluginContractError(
            "failure_policy must be continue or fail"
        )
    config_schema = raw.get("config_schema", {})
    if not isinstance(config_schema, Mapping):
        raise PluginContractError("config_schema must be an object")
    normalized_schema = normalize_config_schema(config_schema)
    author = str(raw.get("author", "")).strip()
    description = str(raw.get("description", "")).strip()
    if len(author) > 200:
        raise PluginContractError("author is too long")
    if len(description) > 20_000:
        raise PluginContractError("description is too long")
    return PluginManifest(
        plugin_id=plugin_id,
        display_name=display_name,
        package_version=package_version,
        entrypoint=entrypoint,
        hooks=hooks,
        supported_steps=supported_steps,
        supported_modes=supported_modes,
        priority=priority,
        failure_policy=failure_policy,
        author=author,
        description=description,
        default_enabled=bool(raw.get("default_enabled", False)),
        config_schema=normalized_schema,
    )


def normalize_config_schema(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, field in raw.items():
        name = str(key).strip()
        if not name or len(name) > 100 or not isinstance(field, Mapping):
            raise PluginContractError("config_schema field is invalid")
        field_type = str(field.get("type", "")).strip()
        if field_type not in CONFIG_TYPES:
            raise PluginContractError(
                f"config_schema.{name}.type is unsupported"
            )
        item = dict(field)
        item["type"] = field_type
        if field_type == "select":
            options = field.get("options")
            if not isinstance(options, list) or not options:
                raise PluginContractError(
                    f"config_schema.{name}.options is required"
                )
            item["options"] = list(options)
        normalized[name] = item
    return normalized


def default_config(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): field["default"]
        for key, field in schema.items()
        if isinstance(field, Mapping) and "default" in field
    }


def validate_config(
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = set(config) - set(schema)
    if unknown:
        raise PluginContractError(
            "plugin config contains unknown fields: "
            + ", ".join(sorted(str(value) for value in unknown))
        )
    result = default_config(schema)
    for key, value in config.items():
        field = schema[key]
        field_type = str(field["type"])
        if field_type == "text":
            if not isinstance(value, str):
                raise PluginContractError(f"config.{key} must be text")
        elif field_type == "number":
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise PluginContractError(f"config.{key} must be a number")
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            if minimum is not None and value < minimum:
                raise PluginContractError(f"config.{key} is below minimum")
            if maximum is not None and value > maximum:
                raise PluginContractError(f"config.{key} exceeds maximum")
        elif field_type == "boolean":
            if not isinstance(value, bool):
                raise PluginContractError(f"config.{key} must be boolean")
        elif field_type == "select":
            if value not in field["options"]:
                raise PluginContractError(
                    f"config.{key} is not one of the allowed options"
                )
        result[str(key)] = value
    return result


def validate_hook_data(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PluginContractError("plugin hook result must be an object")
    _assert_no_base64(value)
    return dict(value)


def _assert_no_base64(value: object, *, depth: int = 0) -> None:
    if depth > 20:
        raise PluginContractError("plugin hook data is nested too deeply")
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if "base64" in lowered or lowered in {
                "image_data",
                "imagebytes",
                "image_bytes",
            }:
                raise PluginContractError(
                    "plugin hook data must use asset references, not Base64"
                )
            _assert_no_base64(child, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_base64(child, depth=depth + 1)
    elif isinstance(value, bytes):
        raise PluginContractError(
            "plugin hook data must use asset references, not bytes"
        )
    elif isinstance(value, str) and len(value) > 1024:
        candidate = value.strip()
        if candidate.startswith("data:") and ";base64," in candidate[:100]:
            raise PluginContractError(
                "plugin hook data must use asset references, not Base64"
            )
        try:
            if len(candidate) % 4 == 0:
                base64.b64decode(candidate[:4096], validate=True)
            else:
                return
        except (ValueError, base64.binascii.Error):
            pass
        else:
            raise PluginContractError(
                "plugin hook data must use asset references, not Base64"
            )


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PluginContractError(f"{field} must be an array")
    return tuple(str(item).strip() for item in value)


def _validate_entrypoint(value: str) -> None:
    if ":" not in value:
        raise PluginContractError(
            "entrypoint must use relative/module.py:Class format"
        )
    module_path, class_name = value.rsplit(":", 1)
    normalized = module_path.replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in normalized.split("/")
        or not normalized.endswith(".py")
        or not class_name.isidentifier()
    ):
        raise PluginContractError("entrypoint is invalid")
