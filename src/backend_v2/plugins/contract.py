"""Static plugin v3 manifest and hook data contracts."""

from __future__ import annotations

import ast
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
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "plugin_id",
        "display_name",
        "package_version",
        "entrypoint",
        "hooks",
        "supported_steps",
        "supported_modes",
        "priority",
        "failure_policy",
        "author",
        "description",
        "default_enabled",
        "config_schema",
    }
)
ATOMIC_PAYLOAD_FIELDS = {
    ("detect", "before"): frozenset(
        {"pageId", "sourceAssetId", "detectorConfig"}
    ),
    ("detect", "after"): frozenset(
        {"pageId", "bubbles", "textMaskAssetId"}
    ),
    ("ocr", "before"): frozenset(
        {"pageId", "sourceAssetId", "bubbles", "ocrConfig"}
    ),
    ("ocr", "after"): frozenset(
        {"pageId", "originalTexts", "ocrResults"}
    ),
    ("color", "before"): frozenset(
        {"pageId", "sourceAssetId", "bubbles"}
    ),
    ("color", "after"): frozenset({"pageId", "colors"}),
    ("translate", "before"): frozenset(
        {"pageId", "originalTexts", "translationConfig"}
    ),
    ("translate", "after"): frozenset(
        {"pageId", "originalTexts", "translations", "textboxTexts"}
    ),
    ("ai_translate", "before"): frozenset(
        {"pageId", "originalTexts", "translations"}
    ),
    ("ai_translate", "after"): frozenset(
        {"pageId", "originalTexts", "translations"}
    ),
    ("inpaint", "before"): frozenset(
        {
            "pageId",
            "sourceAssetId",
            "inputAssetId",
            "textMaskAssetId",
            "bubbles",
            "method",
            "fillColor",
        }
    ),
    ("inpaint", "after"): frozenset(
        {"pageId", "cleanAssetId", "documentRevision"}
    ),
    ("render", "before"): frozenset(
        {"pageId", "inputAssetId", "bubbles", "renderConfig"}
    ),
    ("render", "after"): frozenset(
        {
            "pageId",
            "translatedAssetId",
            "documentRevision",
        }
    ),
}
ATOMIC_FIELD_KINDS = {
    "pageId": "text",
    "sourceAssetId": "text",
    "inputAssetId": "text",
    "textMaskAssetId": "nullable_text",
    "cleanAssetId": "text",
    "translatedAssetId": "text",
    "detectorConfig": "object",
    "ocrConfig": "object",
    "translationConfig": "object",
    "renderConfig": "object",
    "bubbles": "array",
    "originalTexts": "array",
    "ocrResults": "array",
    "colors": "array",
    "translations": "array",
    "textboxTexts": "array",
    "method": "text",
    "fillColor": "nullable_text",
    "documentRevision": "number",
}


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
    fields = set(raw)
    missing = MANIFEST_FIELDS - fields
    unknown = fields - MANIFEST_FIELDS
    if missing or unknown:
        raise PluginContractError(
            "plugin manifest field mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    schema_version = raw["schema_version"]
    if schema_version != 3:
        raise PluginContractError("plugin manifest schema_version must be 3")
    plugin_id = _required_string(raw["plugin_id"], "plugin_id")
    if not PLUGIN_ID_PATTERN.fullmatch(plugin_id):
        raise PluginContractError("plugin_id is invalid")
    display_name = _required_string(raw["display_name"], "display_name")
    if not display_name or len(display_name) > 200:
        raise PluginContractError(
            "display_name must contain 1-200 characters"
        )
    package_version = _required_string(
        raw["package_version"],
        "package_version",
    )
    if not PACKAGE_VERSION_PATTERN.fullmatch(package_version):
        raise PluginContractError("package_version is invalid")
    entrypoint = _required_string(raw["entrypoint"], "entrypoint")
    _validate_entrypoint(entrypoint)
    hooks = _string_tuple(raw["hooks"], "hooks")
    if not hooks or any(value not in HOOK_NAMES for value in hooks):
        raise PluginContractError("manifest hooks contain unsupported values")
    if len(set(hooks)) != len(hooks):
        raise PluginContractError("manifest hooks must be unique")
    supported_steps = _string_tuple(
        raw["supported_steps"],
        "supported_steps",
    )
    if not supported_steps or any(
        value not in HOOK_STEPS for value in supported_steps
    ):
        raise PluginContractError("supported_steps contains invalid values")
    supported_modes = _string_tuple(
        raw["supported_modes"],
        "supported_modes",
    )
    if not supported_modes or any(
        value not in PLUGIN_MODES for value in supported_modes
    ):
        raise PluginContractError("supported_modes contains invalid values")
    priority = raw["priority"]
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise PluginContractError("priority must be an integer")
    if not -10_000 <= priority <= 10_000:
        raise PluginContractError("priority is out of range")
    failure_policy = _required_string(
        raw["failure_policy"],
        "failure_policy",
    )
    if failure_policy not in {"continue", "fail"}:
        raise PluginContractError(
            "failure_policy must be continue or fail"
        )
    config_schema = raw["config_schema"]
    if not isinstance(config_schema, Mapping):
        raise PluginContractError("config_schema must be an object")
    normalized_schema = normalize_config_schema(config_schema)
    author = _string(raw["author"], "author")
    description = _string(raw["description"], "description")
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
        default_enabled=_boolean(raw["default_enabled"], "default_enabled"),
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


def validate_atomic_hook_data(
    step: str,
    phase: str,
    value: object,
) -> dict[str, Any]:
    """Validate the stable domain payload exposed by one atomic v3 hook."""

    if step not in ATOMIC_STEPS:
        raise PluginContractError(f"unsupported atomic plugin step: {step}")
    if phase not in {"before", "after"}:
        raise PluginContractError(f"unsupported plugin hook phase: {phase}")
    data = validate_hook_data(value)
    unknown = set(data) - ATOMIC_PAYLOAD_FIELDS[(step, phase)]
    if unknown:
        raise PluginContractError(
            f"{phase}_{step} returned unsupported fields: "
            + ", ".join(sorted(str(field) for field in unknown))
        )
    _require_text(data, "pageId")

    if step == "detect":
        if phase == "before":
            _require_text(data, "sourceAssetId")
            _require_mapping(data, "detectorConfig")
        else:
            _require_mapping_list(data, "bubbles")
            _require_optional_text(data, "textMaskAssetId")
    elif step == "ocr":
        if phase == "before":
            _require_text(data, "sourceAssetId")
            _require_mapping_list(data, "bubbles")
            _require_mapping(data, "ocrConfig")
        else:
            _require_text_list(data, "originalTexts")
            _require_list(data, "ocrResults")
    elif step == "color":
        if phase == "before":
            _require_text(data, "sourceAssetId")
            _require_mapping_list(data, "bubbles")
        else:
            colors = _require_mapping_list(data, "colors")
            for index, color in enumerate(colors):
                _require_rgb(color, "fgColor", index=index)
                _require_rgb(color, "bgColor", index=index)
                confidence = color.get("confidence", 0)
                if isinstance(confidence, bool) or not isinstance(
                    confidence, (int, float)
                ):
                    raise PluginContractError(
                        f"colors[{index}].confidence must be a number"
                    )
    elif step in {"translate", "ai_translate"}:
        _require_text_list(data, "originalTexts")
        if phase == "before":
            if "translations" in data:
                _require_text_list(data, "translations")
            if "translationConfig" in data:
                _require_mapping(data, "translationConfig")
        else:
            _require_text_list(data, "translations")
            if "textboxTexts" in data:
                _require_text_list(data, "textboxTexts")
    elif step == "inpaint":
        if phase == "before":
            _require_text(data, "sourceAssetId")
            _require_text(data, "inputAssetId")
            _require_optional_text(data, "textMaskAssetId")
            _require_mapping_list(data, "bubbles")
            _require_text(data, "method")
            _require_optional_text(data, "fillColor")
        else:
            _require_text(data, "cleanAssetId")
    elif step == "render":
        if phase == "before":
            _require_text(data, "inputAssetId")
            _require_mapping_list(data, "bubbles")
            _require_mapping(data, "renderConfig")
        else:
            _require_text(data, "translatedAssetId")
    return data


def validate_hook_source_contract(
    manifest: PluginManifest,
    source: str,
    *,
    filename: str,
) -> None:
    """Catch literal top-level hook fields that cannot exist at runtime."""

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise PluginContractError(
            f"plugin entrypoint has invalid Python syntax: {exc}"
        ) from exc
    class_name = manifest.entrypoint.rsplit(":", 1)[1]
    plugin_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if plugin_class is None:
        raise PluginContractError(
            f"plugin entrypoint class is missing: {class_name}"
        )
    methods = {
        node.name: node
        for node in plugin_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    constructor = methods.get("__init__")
    if constructor is not None:
        if isinstance(constructor, ast.AsyncFunctionDef):
            raise PluginContractError(
                "plugin __init__ must be synchronous and callable without arguments"
            )
        _validate_method_call_shape(
            constructor,
            required_bound_arguments=0,
            label="plugin __init__",
        )
    for hook in manifest.hooks:
        callback = methods.get(hook)
        if callback is None:
            raise PluginContractError(
                f"manifest declares missing hook {hook}"
            )
        if isinstance(callback, ast.AsyncFunctionDef):
            raise PluginContractError(
                f"{hook} must be synchronous"
            )
        if any(
            isinstance(decorator, ast.Name)
            and decorator.id in {"staticmethod", "classmethod"}
            for decorator in callback.decorator_list
        ):
            raise PluginContractError(
                f"{hook} must be a normal instance method"
            )
        _validate_method_call_shape(
            callback,
            required_bound_arguments=2,
            label=hook,
        )
        step = hook.split("_", 1)[1]
        if step not in ATOMIC_STEPS:
            continue
        phase = hook.split("_", 1)[0]
        aliases = {callback.args.args[2].arg}
        changed = True
        while changed:
            changed = False
            for node in ast.walk(callback):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if value is None or not _copies_hook_mapping(
                    value,
                    aliases,
                ):
                    continue
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id not in aliases
                    ):
                        aliases.add(target.id)
                        changed = True
        allowed = ATOMIC_PAYLOAD_FIELDS[(step, phase)]
        for node in ast.walk(callback):
            if (
                not isinstance(node, ast.Subscript)
                or not isinstance(node.value, ast.Name)
                or node.value.id not in aliases
            ):
                continue
            key = _literal_subscript_key(node.slice)
            if key is not None and key not in allowed:
                raise PluginContractError(
                    f"{hook} uses unsupported data field: {key}"
                )
        _validate_literal_hook_field_types(
            callback,
            hook=hook,
            aliases=aliases,
            allowed=allowed,
        )


def _validate_method_call_shape(
    method: ast.FunctionDef,
    *,
    required_bound_arguments: int,
    label: str,
) -> None:
    positional = [*method.args.posonlyargs, *method.args.args]
    defaults = method.args.defaults
    required_positional = len(positional) - len(defaults)
    expected_required = required_bound_arguments + 1
    if (
        len(positional) < expected_required
        or required_positional > expected_required
        or any(
            default is None
            for default in method.args.kw_defaults
        )
    ):
        if required_bound_arguments == 0:
            raise PluginContractError(
                "plugin __init__ must be callable without arguments; "
                "hook context is passed to each hook"
            )
        raise PluginContractError(
            f"{label} must be callable as {label}(context, data)"
        )


def _copies_hook_mapping(
    value: ast.expr,
    aliases: set[str],
) -> bool:
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"dict", "deepcopy"}
        and value.args
    ):
        argument = value.args[0]
        return (
            isinstance(argument, ast.Name)
            and argument.id in aliases
        ) or _copies_hook_mapping(argument, aliases)
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "copy"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id in aliases
    ):
        return True
    return False


def _literal_subscript_key(value: ast.expr) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _validate_literal_hook_field_types(
    callback: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    hook: str,
    aliases: set[str],
    allowed: frozenset[str],
) -> None:
    for node in ast.walk(callback):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in aliases
            and node.args
        ):
            key = _literal_subscript_key(node.args[0])
            if key in allowed and len(node.args) >= 2:
                _assert_literal_field_kind(
                    hook,
                    key,
                    node.args[1],
                )
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        for target in targets:
            if (
                not isinstance(target, ast.Subscript)
                or not isinstance(target.value, ast.Name)
                or target.value.id not in aliases
            ):
                continue
            key = _literal_subscript_key(target.slice)
            if key in allowed:
                _assert_literal_field_kind(hook, key, value)


def _assert_literal_field_kind(
    hook: str,
    field: str,
    value: ast.expr,
) -> None:
    actual = _literal_expression_kind(value)
    expected = ATOMIC_FIELD_KINDS[field]
    compatible = (
        actual is None
        or actual == expected
        or (
            expected == "nullable_text"
            and actual in {"text", "null"}
        )
    )
    if not compatible:
        raise PluginContractError(
            f"{hook} treats {field} as {actual}; expected {expected}"
        )


def _literal_expression_kind(value: ast.expr) -> str | None:
    if isinstance(value, (ast.List, ast.ListComp)):
        return "array"
    if isinstance(value, (ast.Dict, ast.DictComp)):
        return "object"
    if isinstance(value, (ast.Tuple, ast.Set, ast.SetComp)):
        return "non_json_collection"
    if isinstance(value, ast.Constant):
        if value.value is None:
            return "null"
        if isinstance(value.value, str):
            return "text"
        if isinstance(value.value, bool):
            return "boolean"
        if isinstance(value.value, (int, float)):
            return "number"
    if isinstance(value, ast.Call):
        function_name = (
            value.func.id
            if isinstance(value.func, ast.Name)
            else None
        )
        return {
            "list": "array",
            "dict": "object",
            "str": "text",
            "int": "number",
            "float": "number",
        }.get(function_name)
    return None


def _require_text(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise PluginContractError(f"{field} must be non-empty text")
    return value


def _require_optional_text(
    data: Mapping[str, Any],
    field: str,
) -> str | None:
    value = data.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise PluginContractError(f"{field} must be text or null")
    return value


def _require_mapping(
    data: Mapping[str, Any],
    field: str,
) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, Mapping):
        raise PluginContractError(f"{field} must be an object")
    return value


def _require_list(
    data: Mapping[str, Any],
    field: str,
) -> list[Any]:
    value = data.get(field)
    if not isinstance(value, list):
        raise PluginContractError(f"{field} must be an array")
    return value


def _require_mapping_list(
    data: Mapping[str, Any],
    field: str,
) -> list[Mapping[str, Any]]:
    values = _require_list(data, field)
    if any(not isinstance(value, Mapping) for value in values):
        raise PluginContractError(f"{field} must contain only objects")
    return values


def _require_text_list(
    data: Mapping[str, Any],
    field: str,
) -> list[str]:
    values = _require_list(data, field)
    if any(not isinstance(value, str) for value in values):
        raise PluginContractError(f"{field} must contain only text")
    return values


def _require_rgb(
    data: Mapping[str, Any],
    field: str,
    *,
    index: int,
) -> None:
    value = data.get(field)
    if value is None:
        return
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or channel < 0
            or channel > 255
            for channel in value
        )
    ):
        raise PluginContractError(
            f"colors[{index}].{field} must be RGB integers or null"
        )


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
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(item, str) for item in value)
    ):
        raise PluginContractError(f"{field} must be an array")
    return tuple(item.strip() for item in value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise PluginContractError(f"{field} must be a string")
    return value.strip()


def _required_string(value: object, field: str) -> str:
    result = _string(value, field)
    if not result:
        raise PluginContractError(f"{field} must not be empty")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise PluginContractError(f"{field} must be boolean")
    return value


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
