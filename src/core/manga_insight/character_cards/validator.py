"""
角色卡校验器。
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple


REQUIRED_DATA_FIELDS = [
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "creator_notes",
    "system_prompt",
    "post_history_instructions",
    "alternate_greetings",
    "tags",
    "creator",
    "character_version",
    "character_book",
    "extensions",
]


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_list(value: Any, split_pattern: str = r"[,;\n]+") -> List[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(split_pattern, value)
    elif value is None:
        items = []
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def normalize_card_v2(card: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    规范化角色卡结构，以提升 SillyTavern/Tavern Helper 兼容性。
    """
    warnings: List[str] = []
    normalized: Dict[str, Any] = copy.deepcopy(card if isinstance(card, dict) else {})

    if normalized.get("spec") != "chara_card_v2":
        warnings.append("spec 已自动规范化为 chara_card_v2")
        normalized["spec"] = "chara_card_v2"

    spec_version = str(normalized.get("spec_version", "")).strip()
    if not spec_version.startswith("2"):
        warnings.append("spec_version 已自动规范化为 2.0")
        normalized["spec_version"] = "2.0"

    data = normalized.get("data")
    if not isinstance(data, dict):
        warnings.append("data 结构异常，已重建为对象")
        data = {}
        normalized["data"] = data

    text_fields = [
        "name",
        "description",
        "personality",
        "scenario",
        "first_mes",
        "mes_example",
        "creator_notes",
        "system_prompt",
        "post_history_instructions",
        "creator",
        "character_version",
    ]
    for field in text_fields:
        data[field] = _as_text(data.get(field, ""))

    data["alternate_greetings"] = _to_list(data.get("alternate_greetings"), split_pattern=r"[\n]+")
    data["tags"] = _to_list(data.get("tags"))

    character_book = data.get("character_book")
    if not isinstance(character_book, dict):
        warnings.append("character_book 结构异常，已重建")
        character_book = {}

    character_book["name"] = _as_text(character_book.get("name"), "Character Lorebook")
    character_book["description"] = _as_text(character_book.get("description"), "")
    character_book["scan_depth"] = _to_int(character_book.get("scan_depth"), 2)
    character_book["token_budget"] = _to_int(character_book.get("token_budget"), 768)
    character_book["recursive_scanning"] = bool(character_book.get("recursive_scanning", True))
    if not isinstance(character_book.get("extensions"), dict):
        character_book["extensions"] = {}

    raw_entries = character_book.get("entries")
    if not isinstance(raw_entries, list):
        warnings.append("character_book.entries 非数组，已重置为空")
        raw_entries = []

    normalized_entries: List[Dict[str, Any]] = []
    for idx, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            warnings.append(f"character_book.entries[{idx}] 非对象，已忽略")
            continue

        entry = dict(raw_entry)
        uid = _to_int(entry.get("uid", entry.get("id", idx + 1)), idx + 1)
        keys = _to_list(entry.get("key", entry.get("keys")))
        secondary_keys = _to_list(entry.get("keysecondary", entry.get("secondary_keys")))
        if not keys:
            fallback_name = _as_text(entry.get("name"))
            if fallback_name:
                keys = [fallback_name]

        normalized_entry = {
            "uid": uid,
            "id": uid,  # 兼容某些扩展侧读取 id 的场景
            "name": _as_text(entry.get("name"), _as_text(entry.get("comment"), f"entry_{uid}")),
            "enabled": bool(entry.get("enabled", True)),
            "key": keys,
            "keys": keys,  # 兼容命名
            "keysecondary": secondary_keys,
            "secondary_keys": secondary_keys,  # 兼容命名
            "content": _as_text(entry.get("content")),
            "insertion_order": _to_int(entry.get("insertion_order"), 100),
            "position": _as_text(entry.get("position"), "before_char") or "before_char",
            "priority": _to_int(entry.get("priority", entry.get("insertion_order")), 100),
            "case_sensitive": bool(entry.get("case_sensitive", False)),
            "comment": _as_text(entry.get("comment")),
            "constant": bool(entry.get("constant", False)),
            "selective": bool(entry.get("selective", False)),
            "extensions": entry.get("extensions") if isinstance(entry.get("extensions"), dict) else {},
        }
        normalized_entries.append(normalized_entry)

    character_book["entries"] = normalized_entries
    data["character_book"] = character_book

    extensions = data.get("extensions")
    if not isinstance(extensions, dict):
        warnings.append("extensions 结构异常，已重建")
        extensions = {}

    saber_tavern = extensions.get("saber_tavern")
    if not isinstance(saber_tavern, dict):
        warnings.append("extensions.saber_tavern 缺失，已创建空结构")
        saber_tavern = {}

    raw_regex = saber_tavern.get("regex_profiles")
    if raw_regex is None:
        raw_regex = []
    if not isinstance(raw_regex, list):
        warnings.append("regex_profiles 非数组，已重置为空")
        raw_regex = []
    regex_profiles: List[Dict[str, Any]] = []
    for idx, raw_rule in enumerate(raw_regex):
        if not isinstance(raw_rule, dict):
            warnings.append(f"regex_profiles[{idx}] 非对象，已忽略")
            continue
        regex_profiles.append({
            "id": _as_text(raw_rule.get("id"), f"rule_{idx + 1}"),
            "name": _as_text(raw_rule.get("name"), f"规则{idx + 1}"),
            "enabled": bool(raw_rule.get("enabled", True)),
            "scope": _as_text(raw_rule.get("scope"), "character"),
            "source": _as_text(raw_rule.get("source"), "ai_output"),
            "pattern": _as_text(raw_rule.get("pattern")),
            "replacement": _as_text(raw_rule.get("replacement")),
            "flags": _as_text(raw_rule.get("flags"), "g"),
            "depth_min": _to_int(raw_rule.get("depth_min"), 0),
            "depth_max": _to_int(raw_rule.get("depth_max"), 99),
            "order": _to_int(raw_rule.get("order"), 100),
            "notes": _as_text(raw_rule.get("notes")),
        })
    saber_tavern["regex_profiles"] = regex_profiles

    raw_mvu = saber_tavern.get("mvu")
    if raw_mvu is None:
        raw_mvu = {}
    if not isinstance(raw_mvu, dict):
        warnings.append("mvu 非对象，已重建")
        raw_mvu = {}
    raw_variables = raw_mvu.get("variables")
    if raw_variables is None:
        raw_variables = []
    if not isinstance(raw_variables, list):
        warnings.append("mvu.variables 非数组，已重置为空")
        raw_variables = []
    variables: List[Dict[str, Any]] = []
    for idx, raw_var in enumerate(raw_variables):
        if not isinstance(raw_var, dict):
            warnings.append(f"mvu.variables[{idx}] 非对象，已忽略")
            continue
        variables.append({
            "name": _as_text(raw_var.get("name"), f"var_{idx + 1}"),
            "type": _as_text(raw_var.get("type"), "string"),
            "scope": _as_text(raw_var.get("scope"), "chat"),
            "default": raw_var.get("default"),
            "value": raw_var.get("value"),
            "validator": raw_var.get("validator") if isinstance(raw_var.get("validator"), dict) else {},
            "description": _as_text(raw_var.get("description")),
        })
    saber_tavern["mvu"] = {
        "version": _as_text(raw_mvu.get("version"), "1.0.0"),
        "variables": variables,
    }

    raw_ui_manifest = saber_tavern.get("ui_manifest")
    if raw_ui_manifest is None:
        raw_ui_manifest = {}
    if not isinstance(raw_ui_manifest, dict):
        warnings.append("ui_manifest 非对象，已重建")
        raw_ui_manifest = {}
    ui_manifest = {
        "layout": _as_text(raw_ui_manifest.get("layout"), "split-dashboard"),
        "theme": _as_text(raw_ui_manifest.get("theme"), "manga-insight-light"),
        "panels": raw_ui_manifest.get("panels") if isinstance(raw_ui_manifest.get("panels"), list) else [],
        "widgets": raw_ui_manifest.get("widgets") if isinstance(raw_ui_manifest.get("widgets"), list) else [],
        "actions": raw_ui_manifest.get("actions") if isinstance(raw_ui_manifest.get("actions"), list) else [],
        "events": raw_ui_manifest.get("events") if isinstance(raw_ui_manifest.get("events"), list) else [],
        "bindings": raw_ui_manifest.get("bindings") if isinstance(raw_ui_manifest.get("bindings"), list) else [],
    }
    saber_tavern["ui_manifest"] = ui_manifest

    raw_import_manifest = saber_tavern.get("import_manifest")
    if raw_import_manifest is None:
        raw_import_manifest = {}
    if not isinstance(raw_import_manifest, dict):
        warnings.append("import_manifest 非对象，已重建")
        raw_import_manifest = {}
    import_manifest = dict(raw_import_manifest)
    import_manifest.setdefault("version", "1.0.0")
    import_manifest.setdefault("requires", {})
    import_manifest.setdefault("activate_steps", [])
    import_manifest.setdefault("fallback_behavior", "extensions ignored")
    saber_tavern["import_manifest"] = import_manifest

    extensions["saber_tavern"] = saber_tavern
    data["extensions"] = extensions
    normalized["data"] = data

    return normalized, warnings


def validate_character_book(book: Dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(book, dict):
        return ["character_book 必须为对象"], warnings

    if "entries" not in book or not isinstance(book.get("entries"), list):
        errors.append("character_book.entries 缺失或类型错误")
    else:
        for idx, entry in enumerate(book.get("entries", [])):
            if not isinstance(entry, dict):
                errors.append(f"character_book.entries[{idx}] 不是对象")
                continue
            keys = entry.get("key", entry.get("keys", []))
            if not isinstance(keys, list) or not keys:
                errors.append(f"character_book.entries[{idx}].key 必须为非空数组")
            if not isinstance(entry.get("content", ""), str) or not entry.get("content", "").strip():
                warnings.append(f"character_book.entries[{idx}] content 为空")

    return errors, warnings


def validate_extensions(extensions: Dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(extensions, dict):
        return ["extensions 必须为对象"], warnings

    saber_ext = extensions.get("saber_tavern")
    if not isinstance(saber_ext, dict):
        warnings.append("extensions.saber_tavern 不存在，将影响 Tavern Helper 增强能力。")
        return errors, warnings

    regex_profiles = saber_ext.get("regex_profiles")
    if regex_profiles is not None and not isinstance(regex_profiles, list):
        errors.append("extensions.saber_tavern.regex_profiles 必须为数组")
    elif isinstance(regex_profiles, list):
        for idx, rule in enumerate(regex_profiles):
            if not isinstance(rule, dict):
                errors.append(f"extensions.saber_tavern.regex_profiles[{idx}] 必须为对象")
                continue
            if not str(rule.get("id", "")).strip():
                errors.append(f"extensions.saber_tavern.regex_profiles[{idx}].id 缺失")
            if "pattern" in rule and not isinstance(rule.get("pattern"), str):
                errors.append(f"extensions.saber_tavern.regex_profiles[{idx}].pattern 必须为字符串")

    mvu = saber_ext.get("mvu")
    if mvu is not None:
        if not isinstance(mvu, dict):
            errors.append("extensions.saber_tavern.mvu 必须为对象")
        elif not isinstance(mvu.get("variables", []), list):
            errors.append("extensions.saber_tavern.mvu.variables 必须为数组")
        else:
            for idx, var in enumerate(mvu.get("variables", [])):
                if not isinstance(var, dict):
                    errors.append(f"extensions.saber_tavern.mvu.variables[{idx}] 必须为对象")
                    continue
                if not str(var.get("name", "")).strip():
                    errors.append(f"extensions.saber_tavern.mvu.variables[{idx}].name 缺失")

    ui_manifest = saber_ext.get("ui_manifest")
    if ui_manifest is not None and not isinstance(ui_manifest, dict):
        errors.append("extensions.saber_tavern.ui_manifest 必须为对象")
    elif isinstance(ui_manifest, dict):
        for field in ["panels", "widgets", "actions", "events", "bindings"]:
            if field in ui_manifest and not isinstance(ui_manifest.get(field), list):
                errors.append(f"extensions.saber_tavern.ui_manifest.{field} 必须为数组")

    import_manifest = saber_ext.get("import_manifest")
    if import_manifest is not None and not isinstance(import_manifest, dict):
        errors.append("extensions.saber_tavern.import_manifest 必须为对象")

    return errors, warnings


def validate_card_v2(card: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(card, dict):
        return {"valid": False, "errors": ["角色卡必须为 JSON 对象"], "warnings": warnings}

    spec = card.get("spec")
    spec_version = str(card.get("spec_version", ""))
    if spec != "chara_card_v2":
        errors.append("spec 必须为 chara_card_v2")
    if not spec_version.startswith("2"):
        errors.append("spec_version 必须为 2.x")

    data = card.get("data")
    if not isinstance(data, dict):
        errors.append("data 字段缺失或类型错误")
        return {"valid": False, "errors": errors, "warnings": warnings}

    for field in REQUIRED_DATA_FIELDS:
        if field not in data:
            errors.append(f"data.{field} 缺失")

    for text_field in [
        "name",
        "description",
        "personality",
        "scenario",
        "first_mes",
        "mes_example",
        "creator_notes",
        "system_prompt",
        "post_history_instructions",
    ]:
        value = data.get(text_field)
        if not isinstance(value, str):
            errors.append(f"data.{text_field} 必须为字符串")
        elif not value.strip():
            warnings.append(f"data.{text_field} 为空")

    if not isinstance(data.get("alternate_greetings"), list):
        errors.append("data.alternate_greetings 必须为数组")
    elif len(data.get("alternate_greetings", [])) == 0:
        warnings.append("data.alternate_greetings 为空")

    if not isinstance(data.get("tags"), list):
        errors.append("data.tags 必须为数组")

    book_errors, book_warnings = validate_character_book(data.get("character_book", {}))
    errors.extend(book_errors)
    warnings.extend(book_warnings)

    ext_errors, ext_warnings = validate_extensions(data.get("extensions", {}))
    errors.extend(ext_errors)
    warnings.extend(ext_warnings)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def build_compatibility_report(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    构建兼容性报告，面向 SillyTavern 核心与 Tavern Helper 扩展能力。
    """
    validation = validate_card_v2(card)
    data = card.get("data", {}) if isinstance(card, dict) else {}
    book = data.get("character_book", {}) if isinstance(data, dict) else {}
    ext = data.get("extensions", {}) if isinstance(data, dict) else {}
    saber_ext = ext.get("saber_tavern", {}) if isinstance(ext, dict) else {}
    regex_profiles = saber_ext.get("regex_profiles", []) if isinstance(saber_ext, dict) else []
    mvu = saber_ext.get("mvu", {}) if isinstance(saber_ext, dict) else {}
    ui_manifest = saber_ext.get("ui_manifest", {}) if isinstance(saber_ext, dict) else {}
    import_manifest = saber_ext.get("import_manifest", {}) if isinstance(saber_ext, dict) else {}

    checks = {
        "v2_core": validation.get("valid", False),
        "embedded_worldbook": isinstance(book, dict) and isinstance(book.get("entries"), list),
        "helper_extension_namespace": isinstance(saber_ext, dict),
        "regex_profiles": isinstance(regex_profiles, list) and len(regex_profiles) > 0,
        "mvu_variables": isinstance(mvu, dict) and isinstance(mvu.get("variables"), list),
        "ui_manifest": isinstance(ui_manifest, dict),
        "import_manifest": isinstance(import_manifest, dict),
    }

    helper_ready = all([
        checks["helper_extension_namespace"],
        checks["regex_profiles"],
        checks["mvu_variables"],
        checks["ui_manifest"],
    ])

    return {
        "compatible": checks["v2_core"],
        "core_ready": checks["v2_core"],
        "helper_ready": helper_ready,
        "checks": checks,
        "errors": validation.get("errors", []),
        "warnings": validation.get("warnings", []),
    }
