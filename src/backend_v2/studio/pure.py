"""Dependency-light Character Studio document and runtime functions.

This module is intentionally independent from the legacy Manga Insight package
so the v2 API import graph remains free of Worker/legacy implementation code.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any, Mapping
import uuid


def create_empty_document(book_id: str, *, title: str = "新角色") -> dict[str, Any]:
    return {
        "bookId": book_id,
        "origin": {"type": "manual", "source_character": None, "source_pages": []},
        "status": {
            "is_favorite": False,
            "frozen_sections": [],
            "last_validated_at": None,
        },
        "meta": {"title": title, "tags": []},
        "avatar": {"mode": "none", "source_page": None},
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


def ensure_document_shape(
    document: Mapping[str, Any],
    *,
    book_id: str,
) -> dict[str, Any]:
    base = create_empty_document(book_id)
    result = _deep_merge(base, document)
    result["bookId"] = book_id
    result.pop("grounding", None)
    result.pop("chatPreset", None)
    identity = _object(result.get("identity"))
    meta = _object(result.get("meta"))
    name = str(identity.get("name") or meta.get("title") or "新角色").strip()
    identity["name"] = name
    identity["aliases"] = _strings(identity.get("aliases"))
    meta["title"] = name
    meta["tags"] = _strings(meta.get("tags"))
    result["identity"] = identity
    result["meta"] = meta
    core = _object(result.get("coreMessages"))
    core["alternate_greetings"] = _strings(core.get("alternate_greetings"))
    result["coreMessages"] = core
    lorebook = _object(result.get("lorebook"))
    lorebook["entries"] = _normalize_entries(lorebook.get("entries"))
    result["lorebook"] = lorebook
    result["regexScripts"] = _normalize_items(
        result.get("regexScripts"),
        prefix="regex",
    )
    result["stateTasks"] = _normalize_items(
        result.get("stateTasks"),
        prefix="task",
    )
    avatar = _object(result.get("avatar"))
    avatar.pop("asset_path", None)
    result["avatar"] = avatar
    return result


def build_export_bundle(document: Mapping[str, Any]) -> dict[str, Any]:
    book_id = str(document.get("bookId", "unknown"))
    doc = ensure_document_shape(document, book_id=book_id)
    identity = _object(doc["identity"])
    core = _object(doc["coreMessages"])
    lorebook = _object(doc["lorebook"])
    v3_entries = [_entry_v3(entry) for entry in lorebook["entries"]]
    shared = {
        "name": identity["name"],
        "description": identity.get("description", ""),
        "personality": identity.get("personality", ""),
        "scenario": identity.get("scenario", ""),
        "first_mes": core.get("first_message", ""),
        "mes_example": core.get("message_example", ""),
        "creator_notes": core.get("creator_notes", ""),
        "system_prompt": core.get("system_prompt", ""),
        "post_history_instructions": core.get(
            "post_history_instructions",
            "",
        ),
        "alternate_greetings": core.get("alternate_greetings", []),
        "tags": _object(doc["meta"]).get("tags", []),
        "creator": "Saber Translator",
        "character_version": core.get("character_version", "2.0.0"),
        "extensions": {
            "fav": bool(_object(doc["status"]).get("is_favorite")),
            "regex_scripts": deepcopy(doc.get("regexScripts", [])),
            "xiaobaix-tasks": {
                "tasks": deepcopy(doc.get("stateTasks", []))
            },
        },
    }
    v3_data = {
        **shared,
        "character_book": {
            "name": lorebook.get("name", ""),
            "entries": v3_entries,
        },
    }
    v2_data = {
        **shared,
        "character_book": {
            "name": lorebook.get("name", ""),
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
            "name": lorebook.get("name", ""),
            "entries": v3_entries,
        },
    }


def import_document_payload(
    book_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    spec = str(payload.get("spec", "")).lower()
    if spec in {"chara_card_v2", "chara_card_v3"}:
        data = _object(payload.get("data"))
        name = str(data.get("name") or payload.get("name") or "导入角色")
        doc = create_empty_document(book_id, title=name)
        doc["origin"]["type"] = "imported"
        doc["identity"].update(
            {
                "name": name,
                "description": data.get("description", ""),
                "personality": data.get("personality", ""),
                "scenario": data.get("scenario", ""),
            }
        )
        doc["coreMessages"].update(
            {
                "first_message": data.get("first_mes", ""),
                "message_example": data.get("mes_example", ""),
                "alternate_greetings": data.get(
                    "alternate_greetings",
                    [],
                ),
                "system_prompt": data.get("system_prompt", ""),
                "post_history_instructions": data.get(
                    "post_history_instructions",
                    "",
                ),
                "creator_notes": data.get("creator_notes", ""),
                "character_version": data.get(
                    "character_version",
                    "2.0.0",
                ),
            }
        )
        doc["meta"]["tags"] = data.get("tags", [])
        extensions = _object(data.get("extensions"))
        doc["status"]["is_favorite"] = bool(extensions.get("fav", False))
        doc["regexScripts"] = extensions.get("regex_scripts", [])
        doc["stateTasks"] = _object(
            extensions.get("xiaobaix-tasks")
        ).get("tasks", [])
        book = _object(data.get("character_book"))
        doc["lorebook"] = {
            "name": book.get("name") or f"{name} 世界书",
            "entries": [
                _entry_internal(entry, index)
                for index, entry in enumerate(
                    _entry_values(book.get("entries"))
                )
            ],
        }
        return ensure_document_shape(doc, book_id=book_id)
    if "entries" in payload:
        name = str(payload.get("name") or "导入世界书")
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
        return ensure_document_shape(doc, book_id=book_id)
    raise ValueError("unable to recognize Studio import format")


def build_diagnostics_report(document: Mapping[str, Any]) -> dict[str, Any]:
    doc = ensure_document_shape(
        document,
        book_id=str(document.get("bookId", "unknown")),
    )
    errors: list[str] = []
    warnings: list[str] = []
    if not str(_object(doc["identity"]).get("name", "")).strip():
        errors.append("identity.name 不能为空")
    if not str(_object(doc["coreMessages"]).get("first_message", "")).strip():
        warnings.append("coreMessages.first_message 为空")
    for index, script in enumerate(doc.get("regexScripts", [])):
        pattern = str(_object(script).get("findRegex", ""))
        try:
            re.compile(pattern)
        except re.error as exc:
            errors.append(f"regexScripts[{index}] 正则非法: {exc}")
    for index, entry in enumerate(_object(doc["lorebook"]).get("entries", [])):
        if not _object(entry).get("keys"):
            errors.append(f"lorebook.entries[{index}].keys 必须为非空数组")
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
    for script in scripts or []:
        if script.get("disabled"):
            continue
        if respect_run_on_edit and not bool(script.get("runOnEdit", True)):
            continue
        placements = script.get("placement", [2])
        if isinstance(placements, int):
            placements = [placements]
        if placement not in placements:
            continue
        pattern = str(script.get("findRegex", ""))
        if not pattern:
            continue
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        replacement = str(script.get("replaceString", ""))
        if regex.search(visible) or regex.search(prompt):
            hits.append(
                {
                    "type": "regex",
                    "scriptName": script.get("scriptName", ""),
                    "pattern": pattern,
                }
            )
        if script.get("promptOnly"):
            prompt = regex.sub(replacement, prompt)
        elif script.get("markdownOnly"):
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
        if not entry.get("enabled", True):
            continue
        entry_id = str(entry.get("id", ""))
        if entry.get("prevent_recursion") and entry_id in matched_ids:
            continue
        keys = _strings(entry.get("keys"))
        secondary = _strings(entry.get("secondary_keys"))
        primary_hit = _matches(text, keys, bool(entry.get("use_regex")))
        secondary_hit = _matches(
            text,
            secondary,
            bool(entry.get("use_regex")),
        )
        if entry.get("constant"):
            hit = True
        elif secondary:
            hit = (
                primary_hit and secondary_hit
                if entry.get("selective", True)
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
            order.get(str(entry.get("position", "before_char")), 1),
            -int(entry.get("priority", 100) or 100),
            str(entry.get("comment", "")),
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
    counts[event] = int(counts.get(event, 0)) + 1
    logs: list[dict[str, Any]] = []
    for task in tasks or []:
        if task.get("disabled") or task.get("triggerTiming") != event:
            continue
        interval = int(task.get("interval", 0) or 0)
        if interval > 1 and counts[event] % interval:
            continue
        for line in str(task.get("commands", "")).splitlines():
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
                "name": task.get("name", ""),
                "event": event,
            }
        )
    return logs


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _normalize_items(value: object, *, prefix: str) -> list[dict[str, Any]]:
    result = []
    for raw in value if isinstance(value, list) else []:
        if isinstance(raw, Mapping):
            item = deepcopy(dict(raw))
            item["id"] = str(item.get("id") or f"{prefix}_{uuid.uuid4().hex[:8]}")
            result.append(item)
    return result


def _normalize_entries(value: object) -> list[dict[str, Any]]:
    result = []
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, Mapping):
            continue
        entry = deepcopy(dict(raw))
        entry["id"] = str(entry.get("id") or f"entry_{uuid.uuid4().hex[:8]}")
        entry["children"] = _normalize_entries(entry.get("children"))
        result.append(entry)
    return result


def _flatten(entries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries or []:
        result.append(dict(entry))
        result.extend(_flatten(_object(entry).get("children", [])))
    return result


def _entry_v3(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": entry.get("id"),
        "keys": entry.get("keys", []),
        "secondary_keys": entry.get("secondary_keys", []),
        "comment": entry.get("comment", ""),
        "content": entry.get("content", ""),
        "constant": bool(entry.get("constant", False)),
        "selective": bool(entry.get("selective", True)),
        "insertion_order": int(entry.get("priority", 100) or 100),
        "enabled": bool(entry.get("enabled", True)),
        "position": entry.get("position", "before_char"),
        "use_regex": bool(entry.get("use_regex", False)),
        "extensions": {
            "depth": int(entry.get("depth", 4) or 4),
            "probability": int(entry.get("probability", 100) or 100),
            "prevent_recursion": bool(entry.get("prevent_recursion", True)),
        },
        "children": [
            _entry_v3(child) for child in entry.get("children", []) or []
        ],
    }


def _entry_v2(entry: Mapping[str, Any], uid: int) -> dict[str, Any]:
    return {
        "uid": uid,
        "key": entry.get("keys", []),
        "keysecondary": entry.get("secondary_keys", []),
        "comment": entry.get("comment", ""),
        "content": entry.get("content", ""),
        "constant": bool(entry.get("constant", False)),
        "selective": bool(entry.get("selective", True)),
        "enabled": bool(entry.get("enabled", True)),
        "position": entry.get("position", "before_char"),
        "extensions": {
            "depth": int(entry.get("depth", 4) or 4),
            "probability": int(entry.get("probability", 100) or 100),
        },
    }


def _entry_values(value: object) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [item for item in value.values() if isinstance(item, Mapping)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _entry_internal(entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    extensions = _object(entry.get("extensions"))
    return {
        "id": str(entry.get("id", entry.get("uid", f"entry_{index}"))),
        "comment": entry.get("comment", entry.get("name", "")),
        "keys": entry.get("keys", entry.get("key", [])) or [],
        "secondary_keys": entry.get(
            "secondary_keys",
            entry.get("keysecondary", []),
        )
        or [],
        "content": entry.get("content", ""),
        "enabled": bool(entry.get("enabled", not entry.get("disable", False))),
        "constant": bool(entry.get("constant", False)),
        "selective": bool(entry.get("selective", True)),
        "priority": int(
            entry.get("insertion_order", entry.get("priority", 100)) or 100
        ),
        "position": entry.get("position", "before_char"),
        "depth": int(extensions.get("depth", entry.get("depth", 4)) or 4),
        "probability": int(
            extensions.get(
                "probability",
                entry.get("probability", 100),
            )
            or 100
        ),
        "prevent_recursion": bool(
            extensions.get("prevent_recursion", True)
        ),
        "use_regex": bool(entry.get("use_regex", False)),
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
    probability = int(entry.get("probability", 100) or 100)
    if probability <= 0:
        return False
    if probability >= 100:
        return True
    token = f"{entry.get('id', '')}|{text}".encode("utf-8")
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 100 < probability
