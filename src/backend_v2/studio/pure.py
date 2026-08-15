"""Dependency-light Character Studio document and runtime functions.

This module is independent from Worker-only Manga Insight implementations so
the API import graph stays dependency-light.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import re
from typing import Any, Mapping


def create_empty_document(book_id: str, *, title: str = "新角色") -> dict[str, Any]:
    return {
        "bookId": book_id,
        "origin": {"type": "manual", "source_character": None},
        "status": {
            "is_favorite": False,
            "frozen_sections": [],
            "last_diagnostics": None,
            "last_validated_at": None,
        },
        "meta": {"title": title, "tags": []},
        "identity": {
            "name": title,
            "aliases": [],
            "description": "",
            "personality": "",
            "scenario": "",
        },
        "coreMessages": {
            "first_message": "",
            "message_example": "",
            "alternate_greetings": [],
            "system_prompt": "",
            "post_history_instructions": "",
            "creator_notes": "",
            "character_version": "2.0.0",
        },
        "lorebook": {"name": f"{title} 世界书", "entries": []},
        "regexScripts": [],
        "stateTasks": [],
        "exportArtifacts": {},
    }


def select_provider_section(
    config: Mapping[str, Any],
    *,
    prefer_vlm: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Return the single provider section a Studio request will use."""

    primary_name = "vlm" if prefer_vlm else "chat"
    fallback_name = "chat" if prefer_vlm else "vlm"
    primary = _provider_section(config, primary_name)
    fallback = _provider_section(config, fallback_name)
    if primary.get("provider") or not fallback.get("provider"):
        return primary_name, primary
    return fallback_name, fallback


def validate_current_document(
    document: Mapping[str, Any],
    *,
    book_id: str,
    title: str | None = None,
) -> dict[str, Any]:
    raw = _exact_object(
        document,
        allowed={
            "bookId",
            "title",
            "origin",
            "status",
            "meta",
            "identity",
            "coreMessages",
            "lorebook",
            "regexScripts",
            "stateTasks",
            "exportArtifacts",
            "id",
            "revision",
            "avatarAssetId",
            "avatarUrl",
            "createdAt",
            "updatedAt",
        },
        required={
            "origin",
            "status",
            "meta",
            "identity",
            "coreMessages",
            "lorebook",
            "regexScripts",
            "stateTasks",
            "exportArtifacts",
        },
        label="Studio document",
    )
    if "bookId" in raw and _required_string(
        raw["bookId"],
        "Studio document bookId",
    ) != book_id:
        raise ValueError("Studio document bookId does not match its book")
    if "id" in raw:
        _required_string(raw["id"], "Studio document id")
    if "revision" in raw:
        _integer(raw["revision"], "Studio document revision", minimum=1)
    if "avatarAssetId" in raw:
        _nullable_string(raw["avatarAssetId"], "Studio document avatarAssetId")
    if "avatarUrl" in raw:
        _nullable_string(raw["avatarUrl"], "Studio document avatarUrl")
    for field in ("createdAt", "updatedAt"):
        if field in raw:
            _date_string(raw[field], f"Studio document {field}")

    origin = _exact_object(
        raw["origin"],
        allowed={"type", "source_character"},
        required={"type", "source_character"},
        label="Studio document origin",
    )
    origin_type = _required_string(
        origin["type"],
        "Studio document origin.type",
    )
    if origin_type not in {"analysis", "manual", "imported"}:
        raise ValueError("Studio document origin.type is invalid")

    status = _exact_object(
        raw["status"],
        allowed={
            "is_favorite",
            "frozen_sections",
            "last_diagnostics",
            "last_validated_at",
        },
        required={
            "is_favorite",
            "frozen_sections",
            "last_diagnostics",
            "last_validated_at",
        },
        label="Studio document status",
    )
    frozen_sections = _string_array(
        status["frozen_sections"],
        "Studio document status.frozen_sections",
    )
    allowed_frozen = {
        "identity",
        "greetings",
        "lorebook",
        "regex",
        "state-tasks",
    }
    if any(value not in allowed_frozen for value in frozen_sections):
        raise ValueError("Studio document frozen_sections contains an invalid value")
    diagnostics = (
        None
        if status["last_diagnostics"] is None
        else _current_diagnostics(status["last_diagnostics"])
    )
    validated_at = (
        None
        if status["last_validated_at"] is None
        else _date_string(
            status["last_validated_at"],
            "Studio document status.last_validated_at",
        )
    )

    meta = _exact_object(
        raw["meta"],
        allowed={"title", "tags"},
        required={"title", "tags"},
        label="Studio document meta",
    )
    identity = _exact_object(
        raw["identity"],
        allowed={"name", "aliases", "description", "personality", "scenario"},
        required={"name", "aliases", "description", "personality", "scenario"},
        label="Studio document identity",
    )
    identity_name = _required_string(
        identity["name"],
        "Studio document identity.name",
    )
    meta_title = _required_string(meta["title"], "Studio document meta.title")
    raw_title = (
        _required_string(raw["title"], "Studio document title")
        if "title" in raw
        else identity_name
    )
    explicit_title = (
        _required_string(title, "Studio document title")
        if title is not None
        else identity_name
    )
    if len({identity_name, meta_title, raw_title, explicit_title}) != 1:
        raise ValueError(
            "title, meta.title and identity.name must agree"
        )

    core = _exact_object(
        raw["coreMessages"],
        allowed={
            "first_message",
            "message_example",
            "alternate_greetings",
            "system_prompt",
            "post_history_instructions",
            "creator_notes",
            "character_version",
        },
        required={
            "first_message",
            "message_example",
            "alternate_greetings",
            "system_prompt",
            "post_history_instructions",
            "creator_notes",
            "character_version",
        },
        label="Studio document coreMessages",
    )
    lorebook = _exact_object(
        raw["lorebook"],
        allowed={"name", "entries"},
        required={"name", "entries"},
        label="Studio document lorebook",
    )

    return {
        "bookId": book_id,
        "title": identity_name,
        "origin": {
            "type": origin_type,
            "source_character": _nullable_string(
                origin["source_character"],
                "Studio document origin.source_character",
            ),
        },
        "status": {
            "is_favorite": _boolean(
                status["is_favorite"],
                "Studio document status.is_favorite",
            ),
            "frozen_sections": frozen_sections,
            "last_diagnostics": diagnostics,
            "last_validated_at": validated_at,
        },
        "meta": {
            "title": identity_name,
            "tags": _string_array(meta["tags"], "Studio document meta.tags"),
        },
        "identity": {
            "name": identity_name,
            "aliases": _string_array(
                identity["aliases"],
                "Studio document identity.aliases",
            ),
            "description": _string(
                identity["description"],
                "Studio document identity.description",
            ),
            "personality": _string(
                identity["personality"],
                "Studio document identity.personality",
            ),
            "scenario": _string(
                identity["scenario"],
                "Studio document identity.scenario",
            ),
        },
        "coreMessages": {
            "first_message": _string(
                core["first_message"],
                "Studio document coreMessages.first_message",
            ),
            "message_example": _string(
                core["message_example"],
                "Studio document coreMessages.message_example",
            ),
            "alternate_greetings": _string_array(
                core["alternate_greetings"],
                "Studio document coreMessages.alternate_greetings",
            ),
            "system_prompt": _string(
                core["system_prompt"],
                "Studio document coreMessages.system_prompt",
            ),
            "post_history_instructions": _string(
                core["post_history_instructions"],
                "Studio document coreMessages.post_history_instructions",
            ),
            "creator_notes": _string(
                core["creator_notes"],
                "Studio document coreMessages.creator_notes",
            ),
            "character_version": _string(
                core["character_version"],
                "Studio document coreMessages.character_version",
            ),
        },
        "lorebook": {
            "name": _string(lorebook["name"], "Studio document lorebook.name"),
            "entries": _current_entries(lorebook["entries"]),
        },
        "regexScripts": _current_regex_scripts(raw["regexScripts"]),
        "stateTasks": _current_state_tasks(raw["stateTasks"]),
        "exportArtifacts": deepcopy(
            _mapping_value(
                raw["exportArtifacts"],
                "Studio document exportArtifacts",
            )
        ),
    }


def _exact_object(
    value: object,
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result = dict(value)
    if not required.issubset(result) or not set(result).issubset(allowed):
        raise ValueError(f"{label} fields are invalid")
    return result


def _mapping_value(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _required_string(value: object, label: str) -> str:
    result = _string(value, label)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _nullable_string(value: object, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _string_array(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a string array")
    return [_string(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _date_string(value: object, label: str) -> str:
    rendered = _required_string(value, label)
    parsed = rendered[:-1] + "+00:00" if rendered.endswith("Z") else rendered
    try:
        datetime.fromisoformat(parsed)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    return rendered


def _current_diagnostics(value: object) -> dict[str, Any]:
    diagnostics = _exact_object(
        value,
        allowed={"valid", "errors", "warnings", "checks"},
        required={"valid", "errors", "warnings", "checks"},
        label="Studio document diagnostics",
    )
    checks = _exact_object(
        diagnostics["checks"],
        allowed={"document", "v3_export", "v2_export"},
        required={"document", "v3_export", "v2_export"},
        label="Studio document diagnostics.checks",
    )
    return {
        "valid": _boolean(
            diagnostics["valid"],
            "Studio document diagnostics.valid",
        ),
        "errors": _string_array(
            diagnostics["errors"],
            "Studio document diagnostics.errors",
        ),
        "warnings": _string_array(
            diagnostics["warnings"],
            "Studio document diagnostics.warnings",
        ),
        "checks": {
            key: _boolean(
                checks[key],
                f"Studio document diagnostics.checks.{key}",
            )
            for key in ("document", "v3_export", "v2_export")
        },
    }


def _current_entries(value: object, *, label: str = "Studio lorebook entries") -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [
        _current_entry(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _current_entry(value: object, *, label: str) -> dict[str, Any]:
    optional = {
        "secondary_keys",
        "probability",
        "prevent_recursion",
        "use_regex",
        "match_persona_description",
        "match_character_description",
        "match_character_personality",
        "match_character_depth_prompt",
        "match_scenario",
    }
    required = {
        "id",
        "comment",
        "keys",
        "content",
        "enabled",
        "constant",
        "selective",
        "priority",
        "position",
        "depth",
        "children",
    }
    entry = _exact_object(
        value,
        allowed=required | optional,
        required=required,
        label=label,
    )
    result: dict[str, Any] = {
        "id": _required_string(entry["id"], f"{label}.id"),
        "comment": _string(entry["comment"], f"{label}.comment"),
        "keys": _string_array(entry["keys"], f"{label}.keys"),
        "content": _string(entry["content"], f"{label}.content"),
        "enabled": _boolean(entry["enabled"], f"{label}.enabled"),
        "constant": _boolean(entry["constant"], f"{label}.constant"),
        "selective": _boolean(entry["selective"], f"{label}.selective"),
        "priority": _integer(entry["priority"], f"{label}.priority"),
        "position": _required_string(entry["position"], f"{label}.position"),
        "depth": _integer(entry["depth"], f"{label}.depth", minimum=0),
        "children": _current_entries(
            entry["children"],
            label=f"{label}.children",
        ),
    }
    if "secondary_keys" in entry:
        result["secondary_keys"] = _string_array(
            entry["secondary_keys"],
            f"{label}.secondary_keys",
        )
    if "probability" in entry:
        probability = _integer(
            entry["probability"],
            f"{label}.probability",
            minimum=0,
        )
        if probability > 100:
            raise ValueError(f"{label}.probability must not exceed 100")
        result["probability"] = probability
    for key in optional - {"secondary_keys", "probability"}:
        if key in entry:
            result[key] = _boolean(entry[key], f"{label}.{key}")
    return result


def _current_regex_scripts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Studio document regexScripts must be an array")
    result = []
    fields = {
        "id",
        "scriptName",
        "findRegex",
        "replaceString",
        "placement",
        "markdownOnly",
        "promptOnly",
        "runOnEdit",
        "disabled",
    }
    for index, raw in enumerate(value):
        label = f"Studio document regexScripts[{index}]"
        script = _exact_object(
            raw,
            allowed=fields,
            required=fields,
            label=label,
        )
        placement = script["placement"]
        if not isinstance(placement, list):
            raise ValueError(f"{label}.placement must be an integer array")
        result.append(
            {
                "id": _required_string(script["id"], f"{label}.id"),
                "scriptName": _string(
                    script["scriptName"],
                    f"{label}.scriptName",
                ),
                "findRegex": _string(
                    script["findRegex"],
                    f"{label}.findRegex",
                ),
                "replaceString": _string(
                    script["replaceString"],
                    f"{label}.replaceString",
                ),
                "placement": [
                    _integer(item, f"{label}.placement[{item_index}]")
                    for item_index, item in enumerate(placement)
                ],
                **{
                    key: _boolean(script[key], f"{label}.{key}")
                    for key in (
                        "markdownOnly",
                        "promptOnly",
                        "runOnEdit",
                        "disabled",
                    )
                },
            }
        )
    return result


def _current_state_tasks(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Studio document stateTasks must be an array")
    result = []
    fields = {"id", "name", "triggerTiming", "interval", "commands", "disabled"}
    for index, raw in enumerate(value):
        label = f"Studio document stateTasks[{index}]"
        task = _exact_object(
            raw,
            allowed=fields,
            required=fields,
            label=label,
        )
        result.append(
            {
                "id": _required_string(task["id"], f"{label}.id"),
                "name": _string(task["name"], f"{label}.name"),
                "triggerTiming": _string(
                    task["triggerTiming"],
                    f"{label}.triggerTiming",
                ),
                "interval": _integer(task["interval"], f"{label}.interval"),
                "commands": _string(task["commands"], f"{label}.commands"),
                "disabled": _boolean(task["disabled"], f"{label}.disabled"),
            }
        )
    return result


def build_export_bundle(document: Mapping[str, Any]) -> dict[str, Any]:
    book_id = _required_string(
        document.get("bookId"),
        "Studio document bookId",
    )
    doc = validate_current_document(document, book_id=book_id)
    identity = doc["identity"]
    core = doc["coreMessages"]
    lorebook = doc["lorebook"]
    v3_entries = [_entry_v3(entry) for entry in lorebook["entries"]]
    shared = {
        "name": identity["name"],
        "description": identity["description"],
        "personality": identity["personality"],
        "scenario": identity["scenario"],
        "first_mes": core["first_message"],
        "mes_example": core["message_example"],
        "creator_notes": core["creator_notes"],
        "system_prompt": core["system_prompt"],
        "post_history_instructions": core["post_history_instructions"],
        "alternate_greetings": core["alternate_greetings"],
        "tags": doc["meta"]["tags"],
        "creator": "Saber Translator",
        "character_version": core["character_version"],
        "extensions": {
            "fav": doc["status"]["is_favorite"],
            "regex_scripts": deepcopy(doc["regexScripts"]),
            "xiaobaix-tasks": {
                "tasks": deepcopy(doc["stateTasks"])
            },
        },
    }
    v3_data = {
        **shared,
        "character_book": {
            "name": lorebook["name"],
            "entries": v3_entries,
        },
    }
    v2_data = {
        **shared,
        "character_book": {
            "name": lorebook["name"],
            "entries": [
                _entry_v2(entry, index)
                for index, entry in enumerate(_flatten(lorebook["entries"]), 1)
            ],
        },
    }
    return {
        "document": doc,
        "v3": {
            "spec": "chara_card_v3",
            "spec_version": "3.0",
            "name": identity["name"],
            "data": v3_data,
        },
        "v2": {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": v2_data,
        },
        "worldbook": {
            "name": lorebook["name"],
            "entries": v3_entries,
        },
    }


def import_document_payload(
    book_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_spec = payload.get("spec")
    if raw_spec is not None and not isinstance(raw_spec, str):
        raise ValueError("Studio import spec must be a string")
    spec = raw_spec.lower() if raw_spec is not None else ""
    if spec in {"chara_card_v2", "chara_card_v3"}:
        data = _external_object(payload, "data", label="Studio card data")
        name = _external_name(
            (data, "name", "Studio card data.name"),
            (payload, "name", "Studio card name"),
            fallback="导入角色",
        )
        doc = create_empty_document(book_id, title=name)
        doc["origin"]["type"] = "imported"
        doc["identity"].update(
            {
                "name": name,
                "description": _external_string(
                    data,
                    "description",
                    label="Studio card description",
                ),
                "personality": _external_string(
                    data,
                    "personality",
                    label="Studio card personality",
                ),
                "scenario": _external_string(
                    data,
                    "scenario",
                    label="Studio card scenario",
                ),
            }
        )
        doc["coreMessages"].update(
            {
                "first_message": _external_string(
                    data,
                    "first_mes",
                    label="Studio card first_mes",
                ),
                "message_example": _external_string(
                    data,
                    "mes_example",
                    label="Studio card mes_example",
                ),
                "alternate_greetings": _external_string_array(
                    data,
                    "alternate_greetings",
                    label="Studio card alternate_greetings",
                ),
                "system_prompt": _external_string(
                    data,
                    "system_prompt",
                    label="Studio card system_prompt",
                ),
                "post_history_instructions": _external_string(
                    data,
                    "post_history_instructions",
                    label="Studio card post_history_instructions",
                ),
                "creator_notes": _external_string(
                    data,
                    "creator_notes",
                    label="Studio card creator_notes",
                ),
                "character_version": _external_string(
                    data,
                    "character_version",
                    label="Studio card character_version",
                    default="2.0.0",
                ),
            }
        )
        doc["meta"]["tags"] = _external_string_array(
            data,
            "tags",
            label="Studio card tags",
        )
        extensions = _external_object(
            data,
            "extensions",
            label="Studio card extensions",
            required=False,
        )
        doc["status"]["is_favorite"] = _external_boolean(
            extensions,
            "fav",
            label="Studio card extensions.fav",
        )
        doc["regexScripts"] = _external_regex_scripts(
            extensions.get("regex_scripts")
        )
        task_extension = _external_object(
            extensions,
            "xiaobaix-tasks",
            label="Studio card xiaobaix-tasks",
            required=False,
        )
        doc["stateTasks"] = _external_state_tasks(
            task_extension.get("tasks")
        )
        book = _external_object(
            data,
            "character_book",
            label="Studio card character_book",
            required=False,
        )
        doc["lorebook"] = {
            "name": _external_name(
                (book, "name", "Studio card character_book.name"),
                fallback=f"{name} 世界书",
            ),
            "entries": [
                _entry_internal(entry, index)
                for index, entry in enumerate(
                    _entry_values(book.get("entries"))
                )
            ],
        }
        return validate_current_document(doc, book_id=book_id)
    if "entries" in payload:
        name = _external_name(
            (payload, "name", "Studio worldbook name"),
            fallback="导入世界书",
        )
        doc = create_empty_document(book_id, title=name)
        doc["origin"]["type"] = "imported"
        doc["lorebook"] = {
            "name": name,
            "entries": [
                _entry_internal(entry, index)
                for index, entry in enumerate(
                    _entry_values(payload.get("entries"))
                )
            ],
        }
        return validate_current_document(doc, book_id=book_id)
    raise ValueError("unable to recognize Studio import format")


def build_diagnostics_report(document: Mapping[str, Any]) -> dict[str, Any]:
    book_id = _required_string(
        document.get("bookId"),
        "Studio document bookId",
    )
    doc = validate_current_document(document, book_id=book_id)
    errors: list[str] = []
    warnings: list[str] = []
    if not doc["coreMessages"]["first_message"].strip():
        warnings.append("coreMessages.first_message 为空")
    for index, script in enumerate(doc["regexScripts"]):
        pattern = script["findRegex"]
        try:
            re.compile(pattern)
        except re.error as exc:
            errors.append(f"regexScripts[{index}] 正则非法: {exc}")
    for index, entry in enumerate(_flatten(doc["lorebook"]["entries"])):
        if not entry["keys"]:
            errors.append(f"lorebook.entries[{index}].keys 必须为非空数组")
    allowed_task_events = {
        "initialization",
        "message_received",
        "message_sent",
    }
    for index, task in enumerate(doc["stateTasks"]):
        if not task["name"].strip():
            errors.append(f"stateTasks[{index}].name 不能为空")
        trigger = task["triggerTiming"]
        if trigger not in allowed_task_events:
            errors.append(
                f"stateTasks[{index}].triggerTiming 不支持值: {trigger}"
            )
        if task["interval"] < 0:
            errors.append(f"stateTasks[{index}].interval 不能为负数")
        commands = task["commands"]
        if not commands.strip():
            errors.append(f"stateTasks[{index}].commands 不能为空")
        if "<<taskjs>>" in commands and "<</taskjs>>" not in commands:
            errors.append(f"stateTasks[{index}] 缺少 <</taskjs>> 结束标记")
    bundle = build_export_bundle(doc)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "document": not errors,
            "v3_export": bundle["v3"]["spec"] == "chara_card_v3",
            "v2_export": bundle["v2"]["spec"] == "chara_card_v2",
        },
    }


def apply_regex_scripts(
    text: str,
    scripts: list[Mapping[str, Any]],
    placement: int,
    *,
    respect_run_on_edit: bool = False,
) -> tuple[str, str, list[dict[str, Any]]]:
    visible = text
    prompt = text
    hits: list[dict[str, Any]] = []
    for script in scripts:
        if script["disabled"]:
            continue
        if respect_run_on_edit and not script["runOnEdit"]:
            continue
        placements = script["placement"]
        if placement not in placements:
            continue
        pattern = script["findRegex"]
        if not pattern:
            continue
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        replacement = script["replaceString"]
        if regex.search(visible) or regex.search(prompt):
            hits.append(
                {
                    "type": "regex",
                    "scriptName": script["scriptName"],
                    "pattern": pattern,
                }
            )
        if script["promptOnly"]:
            prompt = regex.sub(replacement, prompt)
        elif script["markdownOnly"]:
            visible = regex.sub(replacement, visible)
        else:
            visible = regex.sub(replacement, visible)
            prompt = regex.sub(replacement, prompt)
    return visible, prompt, hits


def match_lorebook(
    entries: list[Mapping[str, Any]],
    text: str,
    *,
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    runtime = session.setdefault("_runtime", {})
    matched_ids = set(runtime.setdefault("matched_lorebook_ids", []))
    matched: list[dict[str, Any]] = []
    for entry in _flatten(entries):
        entry = dict(entry)
        if not entry["enabled"]:
            continue
        entry_id = entry["id"]
        if entry.get("prevent_recursion") and entry_id in matched_ids:
            continue
        keys = entry["keys"]
        secondary = entry.get("secondary_keys", [])
        use_regex = entry.get("use_regex", False)
        primary_hit = _matches(text, keys, use_regex)
        secondary_hit = _matches(
            text,
            secondary,
            use_regex,
        )
        if entry["constant"]:
            hit = True
        elif secondary:
            hit = (
                primary_hit and secondary_hit
                if entry["selective"]
                else primary_hit or secondary_hit
            )
        else:
            hit = primary_hit
        if not hit or not _probability(entry, text):
            continue
        matched.append(entry)
        if entry.get("prevent_recursion") and entry_id:
            matched_ids.add(entry_id)
    runtime["matched_lorebook_ids"] = list(matched_ids)
    return matched


def sort_lorebook_hits(entries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = {"before_char": 0, "at_depth": 1, "after_char": 2}
    return sorted(
        (dict(entry) for entry in entries),
        key=lambda entry: (
            order.get(entry["position"], 1),
            -entry["priority"],
            entry["comment"],
        ),
    )


def run_state_tasks(
    session: dict[str, Any],
    tasks: list[Mapping[str, Any]],
    *,
    event: str,
) -> list[dict[str, Any]]:
    runtime = session.setdefault("_runtime", {})
    counts = runtime.setdefault("event_counts", {})
    previous_count = counts.get(event, 0)
    if isinstance(previous_count, bool) or not isinstance(previous_count, int):
        raise ValueError("Studio runtime event count must be an integer")
    if event != "initialization":
        counts[event] = previous_count + 1
    current_count = counts.get(event, 0)
    logs: list[dict[str, Any]] = []
    for task in tasks:
        if task["disabled"] or task["triggerTiming"] != event:
            continue
        interval = task["interval"]
        if (
            event != "initialization"
            and interval > 1
            and current_count % interval
        ):
            continue
        for line in task["commands"].splitlines():
            match = re.search(
                r"/setvar\s+key=([A-Za-z0-9_\-\.]+)\s+([^'\")]+)",
                line.strip(),
            )
            if match:
                session.setdefault("variables", {})[match.group(1)] = (
                    match.group(2).strip().strip("'\"")
                )
        logs.append(
            {
                "type": "task",
                "name": task["name"],
                "event": event,
                "interval": interval,
            }
        )
    return logs


def _provider_section(
    config: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    value = config.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Studio provider section {name} must be an object")
    return dict(value)


def _flatten(entries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        result.append(dict(entry))
        result.extend(_flatten(entry["children"]))
    return result


def _entry_v3(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "keys": entry["keys"],
        "secondary_keys": entry.get("secondary_keys", []),
        "comment": entry["comment"],
        "content": entry["content"],
        "constant": entry["constant"],
        "selective": entry["selective"],
        "insertion_order": entry["priority"],
        "enabled": entry["enabled"],
        "position": entry["position"],
        "use_regex": entry.get("use_regex", False),
        "extensions": {
            "depth": entry["depth"],
            "probability": entry.get("probability", 100),
            "prevent_recursion": entry.get("prevent_recursion", True),
        },
        "children": [
            _entry_v3(child) for child in entry["children"]
        ],
    }


def _entry_v2(entry: Mapping[str, Any], uid: int) -> dict[str, Any]:
    return {
        "uid": uid,
        "key": entry["keys"],
        "keysecondary": entry.get("secondary_keys", []),
        "comment": entry["comment"],
        "content": entry["content"],
        "constant": entry["constant"],
        "selective": entry["selective"],
        "enabled": entry["enabled"],
        "position": entry["position"],
        "extensions": {
            "depth": entry["depth"],
            "probability": entry.get("probability", 100),
        },
    }


def _external_name(
    *candidates: tuple[Mapping[str, Any], str, str],
    fallback: str,
) -> str:
    for source, field, label in candidates:
        value = source.get(field)
        if value is None:
            continue
        rendered = _string(value, label).strip()
        if rendered:
            return rendered
    return fallback


def _external_object(
    source: Mapping[str, Any],
    field: str,
    *,
    label: str,
    required: bool = True,
) -> dict[str, Any]:
    if field not in source or source[field] is None:
        if required:
            raise ValueError(f"{label} must be an object")
        return {}
    return dict(_mapping_value(source[field], label))


def _external_string(
    source: Mapping[str, Any],
    field: str,
    *,
    label: str,
    default: str = "",
) -> str:
    value = source.get(field)
    return default if value is None else _string(value, label)


def _external_string_array(
    source: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> list[str]:
    value = source.get(field)
    return [] if value is None else _string_array(value, label)


def _external_boolean(
    source: Mapping[str, Any],
    field: str,
    *,
    label: str,
    default: bool = False,
) -> bool:
    value = source.get(field)
    return default if value is None else _boolean(value, label)


def _external_alias(
    source: Mapping[str, Any],
    *fields: str,
    label: str,
    default: object,
) -> object:
    present = [field for field in fields if source.get(field) is not None]
    if not present:
        return default
    value = source[present[0]]
    if any(source[field] != value for field in present[1:]):
        raise ValueError(f"{label} aliases disagree")
    return value


def _external_regex_scripts(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Studio card regex_scripts must be an array")
    result = []
    for index, raw in enumerate(value):
        label = f"Studio card regex_scripts[{index}]"
        script = dict(_mapping_value(raw, label))
        placement = _external_alias(
            script,
            "placement",
            label=f"{label}.placement",
            default=[2],
        )
        if not isinstance(placement, list):
            raise ValueError(f"{label}.placement must be an integer array")
        result.append(
            {
                "id": _string(
                    _external_alias(
                        script,
                        "id",
                        label=f"{label}.id",
                        default=f"regex_{index}",
                    ),
                    f"{label}.id",
                ),
                "scriptName": _string(
                    _external_alias(
                        script,
                        "scriptName",
                        "script_name",
                        label=f"{label}.scriptName",
                        default="",
                    ),
                    f"{label}.scriptName",
                ),
                "findRegex": _string(
                    _external_alias(
                        script,
                        "findRegex",
                        "find_regex",
                        label=f"{label}.findRegex",
                        default="",
                    ),
                    f"{label}.findRegex",
                ),
                "replaceString": _string(
                    _external_alias(
                        script,
                        "replaceString",
                        "replace_string",
                        label=f"{label}.replaceString",
                        default="",
                    ),
                    f"{label}.replaceString",
                ),
                "placement": [
                    _integer(item, f"{label}.placement[{item_index}]")
                    for item_index, item in enumerate(placement)
                ],
                **{
                    internal: _boolean(
                        _external_alias(
                            script,
                            internal,
                            external,
                            label=f"{label}.{internal}",
                            default=default,
                        ),
                        f"{label}.{internal}",
                    )
                    for internal, external, default in (
                        ("markdownOnly", "markdown_only", False),
                        ("promptOnly", "prompt_only", False),
                        ("runOnEdit", "run_on_edit", True),
                        ("disabled", "disabled", False),
                    )
                },
            }
        )
    return result


def _external_state_tasks(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Studio card state tasks must be an array")
    result = []
    for index, raw in enumerate(value):
        label = f"Studio card state tasks[{index}]"
        task = dict(_mapping_value(raw, label))
        result.append(
            {
                "id": _string(task.get("id", f"task_{index}"), f"{label}.id"),
                "name": _external_string(task, "name", label=f"{label}.name"),
                "triggerTiming": _string(
                    _external_alias(
                        task,
                        "triggerTiming",
                        "trigger_timing",
                        label=f"{label}.triggerTiming",
                        default="initialization",
                    ),
                    f"{label}.triggerTiming",
                ),
                "interval": _integer(
                    task.get("interval", 0),
                    f"{label}.interval",
                ),
                "commands": _external_string(
                    task,
                    "commands",
                    label=f"{label}.commands",
                ),
                "disabled": _external_boolean(
                    task,
                    "disabled",
                    label=f"{label}.disabled",
                ),
            }
        )
    return result


def _entry_values(value: object) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        items = list(value.values())
    if isinstance(value, list):
        items = value
    elif not isinstance(value, Mapping):
        raise ValueError("Studio worldbook entries must be an array or object")
    if not all(isinstance(item, Mapping) for item in items):
        raise ValueError("Studio worldbook entries must contain objects")
    return items


def _entry_internal(entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    extensions = _external_object(
        entry,
        "extensions",
        label=f"Studio worldbook entries[{index}].extensions",
        required=False,
    )
    raw_id = entry.get("id", entry.get("uid", f"entry_{index}"))
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
        raise ValueError(f"Studio worldbook entries[{index}].id is invalid")
    enabled = (
        _boolean(
            entry["enabled"],
            f"Studio worldbook entries[{index}].enabled",
        )
        if entry.get("enabled") is not None
        else not _external_boolean(
            entry,
            "disable",
            label=f"Studio worldbook entries[{index}].disable",
        )
    )
    priority = entry.get("insertion_order", entry.get("priority", 100))
    depth = extensions.get("depth", entry.get("depth", 4))
    probability = extensions.get(
        "probability",
        entry.get("probability", 100),
    )
    return {
        "id": str(raw_id),
        "comment": _string(
            entry.get("comment", entry.get("name", "")),
            f"Studio worldbook entries[{index}].comment",
        ),
        "keys": _string_array(
            entry.get("keys", entry.get("key", [])),
            f"Studio worldbook entries[{index}].keys",
        ),
        "secondary_keys": _string_array(
            entry.get("secondary_keys", entry.get("keysecondary", [])),
            f"Studio worldbook entries[{index}].secondary_keys",
        ),
        "content": _string(
            entry.get("content", ""),
            f"Studio worldbook entries[{index}].content",
        ),
        "enabled": enabled,
        "constant": _boolean(
            entry.get("constant", False),
            f"Studio worldbook entries[{index}].constant",
        ),
        "selective": _boolean(
            entry.get("selective", True),
            f"Studio worldbook entries[{index}].selective",
        ),
        "priority": _integer(
            priority,
            f"Studio worldbook entries[{index}].priority",
        ),
        "position": _string(
            entry.get("position", "before_char"),
            f"Studio worldbook entries[{index}].position",
        ),
        "depth": _integer(
            depth,
            f"Studio worldbook entries[{index}].depth",
            minimum=0,
        ),
        "probability": _integer(
            probability,
            f"Studio worldbook entries[{index}].probability",
            minimum=0,
        ),
        "prevent_recursion": _boolean(
            extensions.get("prevent_recursion", True),
            f"Studio worldbook entries[{index}].prevent_recursion",
        ),
        "use_regex": _boolean(
            entry.get("use_regex", False),
            f"Studio worldbook entries[{index}].use_regex",
        ),
        "children": [
            _entry_internal(child, child_index)
            for child_index, child in enumerate(
                _entry_values(entry.get("children"))
            )
        ],
    }


def _matches(text: str, keys: list[str], use_regex: bool) -> bool:
    for key in keys:
        try:
            if use_regex and re.search(key, text, re.IGNORECASE):
                return True
        except re.error:
            continue
        if not use_regex and key.lower() in text.lower():
            return True
    return False


def _probability(entry: Mapping[str, Any], text: str) -> bool:
    probability = entry.get("probability", 100)
    if probability <= 0:
        return False
    if probability >= 100:
        return True
    token = f"{entry['id']}|{text}".encode("utf-8")
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 100 < probability
