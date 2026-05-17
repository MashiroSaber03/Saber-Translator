"""
Adapters between Character Studio documents and export/import formats.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .png_codec import CharacterStudioPngCodec


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_name(value: str) -> str:
    return (value or "").strip()


def create_empty_document(book_id: str, *, title: str = "新角色", origin_type: str = "manual") -> Dict[str, Any]:
    now = _now_iso()
    return {
        "id": f"doc_{uuid.uuid4().hex[:12]}",
        "bookId": book_id,
        "origin": {
            "type": origin_type,
            "source_character": None,
            "source_pages": [],
        },
        "status": {
            "is_favorite": False,
            "frozen_sections": [],
            "last_validated_at": None,
        },
        "meta": {
            "title": title,
            "tags": [],
            "created_at": now,
            "updated_at": now,
        },
        "avatar": {
            "mode": "none",
            "asset_path": None,
            "source_page": None,
        },
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
        "lorebook": {
            "name": f"{title} 世界书",
            "entries": [],
        },
        "regexScripts": [],
        "stateTasks": [],
        "chatPreset": {
            "opening_mode": "first_message",
        },
        "grounding": {
            "timeline_mode": "",
            "sample_pages": [],
            "relationships": [],
            "key_moments": [],
        },
        "exportArtifacts": {},
    }


def ensure_document_shape(document: Dict[str, Any], *, book_id: Optional[str] = None) -> Dict[str, Any]:
    base = create_empty_document(book_id or document.get("bookId") or "unknown")
    normalized = copy.deepcopy(base)
    for key, value in (document or {}).items():
        if isinstance(value, dict) and isinstance(normalized.get(key), dict):
            normalized[key].update(value)
        else:
            normalized[key] = value
    normalized["bookId"] = book_id or normalized.get("bookId") or base["bookId"]
    normalized["id"] = normalized.get("id") or base["id"]
    normalized["meta"]["updated_at"] = _now_iso()
    normalized["identity"]["name"] = _normalize_name(normalized["identity"].get("name") or normalized["meta"].get("title"))
    normalized["meta"]["title"] = normalized["identity"]["name"] or normalized["meta"].get("title") or "新角色"
    if not isinstance(normalized["identity"].get("aliases"), list):
        normalized["identity"]["aliases"] = []
    if not isinstance(normalized["coreMessages"].get("alternate_greetings"), list):
        normalized["coreMessages"]["alternate_greetings"] = []
    if not isinstance(normalized["lorebook"].get("entries"), list):
        normalized["lorebook"]["entries"] = []
    if not isinstance(normalized.get("regexScripts"), list):
        normalized["regexScripts"] = []
    if not isinstance(normalized.get("stateTasks"), list):
        normalized["stateTasks"] = []
    if not isinstance(normalized["meta"].get("tags"), list):
        normalized["meta"]["tags"] = []
    return normalized


def _convert_lorebook_entry_to_v3(entry: Dict[str, Any]) -> Dict[str, Any]:
    children = [_convert_lorebook_entry_to_v3(child) for child in entry.get("children", []) or []]
    return {
        "id": entry.get("id") or f"entry_{uuid.uuid4().hex[:8]}",
        "keys": entry.get("keys", []) or [],
        "secondary_keys": entry.get("secondary_keys", []) or [],
        "comment": entry.get("comment", "") or "",
        "content": entry.get("content", "") or "",
        "constant": bool(entry.get("constant", False)),
        "selective": bool(entry.get("selective", True)),
        "insertion_order": int(entry.get("priority", 100) or 100),
        "enabled": bool(entry.get("enabled", True)),
        "position": entry.get("position", "before_char") or "before_char",
        "use_regex": bool(entry.get("use_regex", False)),
        "extensions": {
            "depth": int(entry.get("depth", 4) or 4),
            "probability": int(entry.get("probability", 100) or 100),
            "prevent_recursion": bool(entry.get("prevent_recursion", True)),
            "match_persona_description": bool(entry.get("match_persona_description", True)),
            "match_character_description": bool(entry.get("match_character_description", True)),
            "match_character_personality": bool(entry.get("match_character_personality", True)),
            "match_character_depth_prompt": bool(entry.get("match_character_depth_prompt", True)),
            "match_scenario": bool(entry.get("match_scenario", True)),
        },
        "children": children,
    }


def _flatten_lorebook_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for entry in entries:
        flattened.append(copy.deepcopy(entry))
        children = entry.get("children", []) or []
        if children:
            flattened.extend(_flatten_lorebook_entries(children))
    return flattened


def _build_v3_card(document: Dict[str, Any]) -> Dict[str, Any]:
    doc = ensure_document_shape(document)
    lorebook_entries = [_convert_lorebook_entry_to_v3(entry) for entry in doc["lorebook"]["entries"]]
    data = {
        "name": doc["identity"]["name"],
        "description": doc["identity"]["description"],
        "personality": doc["identity"]["personality"],
        "scenario": doc["identity"]["scenario"],
        "first_mes": doc["coreMessages"]["first_message"],
        "mes_example": doc["coreMessages"]["message_example"],
        "creator_notes": doc["coreMessages"]["creator_notes"] or "Created with Manga Insight Character Studio",
        "system_prompt": doc["coreMessages"]["system_prompt"],
        "post_history_instructions": doc["coreMessages"]["post_history_instructions"],
        "tags": doc["meta"]["tags"],
        "creator": "Saber Translator",
        "character_version": doc["coreMessages"]["character_version"] or "2.0.0",
        "alternate_greetings": doc["coreMessages"]["alternate_greetings"],
        "group_only_greetings": [],
        "extensions": {
            "fav": bool(doc["status"]["is_favorite"]),
            "regex_scripts": copy.deepcopy(doc["regexScripts"]),
            "xiaobaix-tasks": {
                "tasks": copy.deepcopy(doc["stateTasks"]),
            },
        },
        "character_book": {
            "name": doc["lorebook"]["name"] or f"{doc['identity']['name']} 世界书",
            "entries": lorebook_entries,
        },
    }
    return {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "name": data["name"],
        "description": data["description"],
        "personality": data["personality"],
        "scenario": data["scenario"],
        "first_mes": data["first_mes"],
        "mes_example": data["mes_example"],
        "creatorcomment": data["creator_notes"],
        "post_history_instructions": data["post_history_instructions"],
        "tags": data["tags"],
        "avatar": "none",
        "data": data,
    }


def _convert_lorebook_entry_to_v2(entry: Dict[str, Any], *, uid_start: int = 1) -> List[Dict[str, Any]]:
    flattened = _flatten_lorebook_entries([entry])
    result: List[Dict[str, Any]] = []
    for idx, item in enumerate(flattened, start=uid_start):
        result.append({
            "uid": idx,
            "id": idx,
            "name": item.get("comment") or f"entry_{idx}",
            "key": item.get("keys", []) or [],
            "keys": item.get("keys", []) or [],
            "keysecondary": item.get("secondary_keys", []) or [],
            "secondary_keys": item.get("secondary_keys", []) or [],
            "content": item.get("content", "") or "",
            "comment": item.get("comment", "") or "",
            "constant": bool(item.get("constant", False)),
            "selective": bool(item.get("selective", True)),
            "insertion_order": int(item.get("priority", 100) or 100),
            "priority": int(item.get("priority", 100) or 100),
            "enabled": bool(item.get("enabled", True)),
            "position": item.get("position", "before_char") or "before_char",
            "case_sensitive": bool(item.get("case_sensitive", False)),
            "extensions": {
                "depth": int(item.get("depth", 4) or 4),
                "probability": int(item.get("probability", 100) or 100),
                "prevent_recursion": bool(item.get("prevent_recursion", True)),
            },
        })
    return result


def _build_v2_card(document: Dict[str, Any]) -> Dict[str, Any]:
    doc = ensure_document_shape(document)
    book_entries: List[Dict[str, Any]] = []
    for entry in doc["lorebook"]["entries"]:
        book_entries.extend(_convert_lorebook_entry_to_v2(entry, uid_start=len(book_entries) + 1))
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": doc["identity"]["name"],
            "description": doc["identity"]["description"],
            "personality": doc["identity"]["personality"],
            "scenario": doc["identity"]["scenario"],
            "first_mes": doc["coreMessages"]["first_message"],
            "mes_example": doc["coreMessages"]["message_example"],
            "creator_notes": doc["coreMessages"]["creator_notes"] or "Created with Manga Insight Character Studio",
            "system_prompt": doc["coreMessages"]["system_prompt"],
            "post_history_instructions": doc["coreMessages"]["post_history_instructions"],
            "alternate_greetings": doc["coreMessages"]["alternate_greetings"],
            "tags": doc["meta"]["tags"],
            "creator": "Saber Translator",
            "character_version": doc["coreMessages"]["character_version"] or "2.0.0",
            "character_book": {
                "name": doc["lorebook"]["name"] or f"{doc['identity']['name']} 世界书",
                "description": "",
                "scan_depth": 4,
                "token_budget": 1024,
                "recursive_scanning": True,
                "extensions": {},
                "entries": book_entries,
            },
            "extensions": {
                "regex_scripts": copy.deepcopy(doc["regexScripts"]),
                "xiaobaix-tasks": {
                    "tasks": copy.deepcopy(doc["stateTasks"]),
                },
                "fav": bool(doc["status"]["is_favorite"]),
            },
        },
    }


def _build_worldbook_export(document: Dict[str, Any]) -> Dict[str, Any]:
    doc = ensure_document_shape(document)
    entries = [_convert_lorebook_entry_to_v3(entry) for entry in doc["lorebook"]["entries"]]
    return {
        "name": doc["lorebook"]["name"] or f"{doc['identity']['name']} 世界书",
        "entries": entries,
    }


def build_export_bundle(document: Dict[str, Any]) -> Dict[str, Any]:
    doc = ensure_document_shape(document)
    v3 = _build_v3_card(doc)
    v2 = _build_v2_card(doc)
    worldbook = _build_worldbook_export(doc)
    return {
        "document": doc,
        "v3": v3,
        "v2": v2,
        "worldbook": worldbook,
    }


def _document_from_v3(book_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data", {}) or {}
    document = create_empty_document(book_id, title=data.get("name") or payload.get("name") or "导入角色", origin_type="imported")
    document["identity"].update({
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "personality": data.get("personality", ""),
        "scenario": data.get("scenario", ""),
    })
    document["coreMessages"].update({
        "first_message": data.get("first_mes", ""),
        "message_example": data.get("mes_example", ""),
        "alternate_greetings": data.get("alternate_greetings", []) or [],
        "system_prompt": data.get("system_prompt", ""),
        "post_history_instructions": data.get("post_history_instructions", ""),
        "creator_notes": data.get("creator_notes", "") or payload.get("creatorcomment", ""),
        "character_version": data.get("character_version", "2.0.0"),
    })
    document["meta"]["tags"] = data.get("tags", []) or []
    document["status"]["is_favorite"] = bool((data.get("extensions", {}) or {}).get("fav", False))
    document["regexScripts"] = copy.deepcopy((data.get("extensions", {}) or {}).get("regex_scripts", []) or [])
    document["stateTasks"] = copy.deepcopy((((data.get("extensions", {}) or {}).get("xiaobaix-tasks", {}) or {}).get("tasks", []) or []))
    document["lorebook"]["name"] = ((data.get("character_book", {}) or {}).get("name")) or f"{document['identity']['name']} 世界书"
    document["lorebook"]["entries"] = [_internal_entry_from_v3(item) for item in (((data.get("character_book", {}) or {}).get("entries", [])) or [])]
    return ensure_document_shape(document, book_id=book_id)


def _document_from_v2(book_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data", {}) or {}
    document = create_empty_document(book_id, title=data.get("name") or "导入角色", origin_type="imported")
    document["identity"].update({
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "personality": data.get("personality", ""),
        "scenario": data.get("scenario", ""),
    })
    document["coreMessages"].update({
        "first_message": data.get("first_mes", ""),
        "message_example": data.get("mes_example", ""),
        "alternate_greetings": data.get("alternate_greetings", []) or [],
        "system_prompt": data.get("system_prompt", ""),
        "post_history_instructions": data.get("post_history_instructions", ""),
        "creator_notes": data.get("creator_notes", ""),
        "character_version": data.get("character_version", "2.0.0"),
    })
    document["meta"]["tags"] = data.get("tags", []) or []
    document["regexScripts"] = copy.deepcopy((data.get("extensions", {}) or {}).get("regex_scripts", []) or [])
    document["stateTasks"] = copy.deepcopy((((data.get("extensions", {}) or {}).get("xiaobaix-tasks", {}) or {}).get("tasks", []) or []))
    book = data.get("character_book", {}) or {}
    document["lorebook"]["name"] = book.get("name") or f"{document['identity']['name']} 世界书"
    document["lorebook"]["entries"] = [_internal_entry_from_v2(item, idx) for idx, item in enumerate(book.get("entries", []) or [])]
    return ensure_document_shape(document, book_id=book_id)


def _document_from_worldbook(book_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("name") or "导入世界书"
    document = create_empty_document(book_id, title=name, origin_type="imported")
    document["identity"]["name"] = name
    document["lorebook"]["name"] = name
    entries = payload.get("entries", []) or []
    if isinstance(entries, dict):
        entries = list(entries.values())
    document["lorebook"]["entries"] = [_internal_entry_from_tavern(item, idx) for idx, item in enumerate(entries)]
    document["meta"]["tags"] = ["worldbook"]
    return ensure_document_shape(document, book_id=book_id)


def _internal_entry_from_v3(entry: Dict[str, Any]) -> Dict[str, Any]:
    ext = entry.get("extensions", {}) or {}
    return {
        "id": str(entry.get("id") or f"entry_{uuid.uuid4().hex[:8]}"),
        "comment": entry.get("comment", "") or "",
        "keys": copy.deepcopy(entry.get("keys", []) or []),
        "secondary_keys": copy.deepcopy(entry.get("secondary_keys", []) or []),
        "content": entry.get("content", "") or "",
        "enabled": bool(entry.get("enabled", True)),
        "constant": bool(entry.get("constant", False)),
        "selective": bool(entry.get("selective", True)),
        "priority": int(entry.get("insertion_order", 100) or 100),
        "position": entry.get("position", "before_char") or "before_char",
        "depth": int(ext.get("depth", 4) or 4),
        "probability": int(ext.get("probability", 100) or 100),
        "prevent_recursion": bool(ext.get("prevent_recursion", True)),
        "use_regex": bool(entry.get("use_regex", False)),
        "match_persona_description": bool(ext.get("match_persona_description", True)),
        "match_character_description": bool(ext.get("match_character_description", True)),
        "match_character_personality": bool(ext.get("match_character_personality", True)),
        "match_character_depth_prompt": bool(ext.get("match_character_depth_prompt", True)),
        "match_scenario": bool(ext.get("match_scenario", True)),
        "children": [_internal_entry_from_v3(child) for child in entry.get("children", []) or []],
    }


def _internal_entry_from_v2(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    ext = entry.get("extensions", {}) or {}
    keys = copy.deepcopy(entry.get("key", entry.get("keys", [])) or [])
    secondary_keys = copy.deepcopy(entry.get("keysecondary", entry.get("secondary_keys", [])) or [])
    return {
        "id": str(entry.get("id", entry.get("uid", f'entry_{index}'))),
        "comment": entry.get("comment", entry.get("name", "")) or "",
        "keys": keys,
        "secondary_keys": secondary_keys,
        "content": entry.get("content", "") or "",
        "enabled": bool(entry.get("enabled", True)),
        "constant": bool(entry.get("constant", False)),
        "selective": bool(entry.get("selective", True)),
        "priority": int(entry.get("priority", entry.get("insertion_order", 100)) or 100),
        "position": entry.get("position", "before_char") or "before_char",
        "depth": int(ext.get("depth", 4) or 4),
        "probability": int(ext.get("probability", 100) or 100),
        "prevent_recursion": bool(ext.get("prevent_recursion", True)),
        "use_regex": bool(entry.get("use_regex", False)),
        "match_persona_description": True,
        "match_character_description": True,
        "match_character_personality": True,
        "match_character_depth_prompt": True,
        "match_scenario": True,
        "children": [],
    }


def _internal_entry_from_tavern(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    ext = entry.get("extensions", {}) or {}
    keys = copy.deepcopy(entry.get("key", entry.get("keys", [])) or [])
    secondary_keys = copy.deepcopy(entry.get("keysecondary", entry.get("secondary_keys", [])) or [])
    children = [_internal_entry_from_tavern(child, child_index) for child_index, child in enumerate(entry.get("children", []) or [])]
    return {
        "id": str(entry.get("id", entry.get("uid", f'entry_{index}'))),
        "comment": entry.get("comment", entry.get("name", "")) or "",
        "keys": keys,
        "secondary_keys": secondary_keys,
        "content": entry.get("content", "") or "",
        "enabled": not bool(entry.get("disable", False)) if "disable" in entry else bool(entry.get("enabled", True)),
        "constant": bool(entry.get("constant", False)),
        "selective": bool(entry.get("selective", True)),
        "priority": int(entry.get("order", entry.get("priority", entry.get("insertion_order", 100))) or 100),
        "position": entry.get("position", "before_char") or "before_char",
        "depth": int(ext.get("depth", entry.get("depth", 4)) or 4),
        "probability": int(ext.get("probability", entry.get("probability", 100)) or 100),
        "prevent_recursion": bool(ext.get("prevent_recursion", entry.get("preventRecursion", True))),
        "use_regex": bool(entry.get("use_regex", False)),
        "match_persona_description": True,
        "match_character_description": True,
        "match_character_personality": True,
        "match_character_depth_prompt": True,
        "match_scenario": True,
        "children": children,
    }


def import_document_payload(book_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("导入数据必须为对象")
    spec = str(payload.get("spec", "")).strip().lower()
    if spec == "chara_card_v3":
        return _document_from_v3(book_id, payload)
    if spec == "chara_card_v2":
        return _document_from_v2(book_id, payload)
    if "entries" in payload:
        return _document_from_worldbook(book_id, payload)
    raise ValueError("无法识别导入格式")


def export_png_bytes(document: Dict[str, Any], *, base_image_path: Optional[str] = None) -> bytes:
    bundle = build_export_bundle(document)
    return CharacterStudioPngCodec.write_card_png(bundle["v3"], base_image_path=base_image_path)
