"""
术语库与翻译记忆管理

数据存储结构:
data/glossary/entries.json
{
  "entries": [...]
}
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.shared.path_helpers import resource_path
from src.shared.security import validate_safe_id

logger = logging.getLogger("GlossaryManager")

_LOCK = threading.Lock()
_GLOSSARY_FILE = resource_path(os.path.join("data", "glossary", "entries.json"))
_ALLOWED_ENTRY_TYPES = {"term", "memory"}
_ALLOWED_SCOPE = {"global", "book"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_store() -> None:
    os.makedirs(os.path.dirname(_GLOSSARY_FILE), exist_ok=True)


def _load_store() -> Dict[str, Any]:
    _ensure_store()
    if not os.path.exists(_GLOSSARY_FILE):
        return {"entries": []}

    try:
        with open(_GLOSSARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"entries": []}
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        return {"entries": entries}
    except Exception as e:
        logger.error(f"加载术语库失败: {e}", exc_info=True)
        return {"entries": []}


def _save_store(data: Dict[str, Any]) -> bool:
    try:
        _ensure_store()
        tmp_file = _GLOSSARY_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, _GLOSSARY_FILE)
        return True
    except Exception as e:
        logger.error(f"保存术语库失败: {e}", exc_info=True)
        return False


def _build_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_text = str(payload.get("source_text", "")).strip()
    target_text = str(payload.get("target_text", "")).strip()
    note = str(payload.get("note", "")).strip()
    enabled = bool(payload.get("enabled", True))

    if not source_text:
        raise ValueError("source_text 不能为空")
    if not target_text:
        raise ValueError("target_text 不能为空")

    entry_type = str(payload.get("entry_type", "term")).strip() or "term"
    if entry_type not in _ALLOWED_ENTRY_TYPES:
        raise ValueError("entry_type 必须是 term 或 memory")

    scope = str(payload.get("scope", "global")).strip() or "global"
    if scope not in _ALLOWED_SCOPE:
        raise ValueError("scope 必须是 global 或 book")

    book_id = payload.get("book_id")
    if scope == "book":
        if not isinstance(book_id, str) or not validate_safe_id(book_id):
            raise ValueError("book 级术语必须提供合法 book_id")
    else:
        book_id = None

    now = _now_iso()
    return {
        "id": f"glo_{uuid.uuid4().hex[:10]}",
        "source_text": source_text,
        "target_text": target_text,
        "note": note,
        "entry_type": entry_type,
        "scope": scope,
        "book_id": book_id,
        "enabled": enabled,
        "created_at": now,
        "updated_at": now,
    }


def list_entries(
    book_id: Optional[str] = None,
    include_global: bool = True,
    query: Optional[str] = None,
    entry_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """列出术语/记忆条目。"""
    if book_id is not None and not validate_safe_id(book_id):
        raise ValueError("无效的 book_id")

    q = (query or "").strip().lower()
    entry_type = (entry_type or "").strip()
    if entry_type and entry_type not in _ALLOWED_ENTRY_TYPES:
        raise ValueError("entry_type 必须是 term 或 memory")

    with _LOCK:
        store = _load_store()
        entries = store.get("entries", [])

    filtered: List[Dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if entry_type and item.get("entry_type") != entry_type:
            continue

        scope = item.get("scope", "global")
        item_book_id = item.get("book_id")

        if book_id:
            if scope == "book" and item_book_id != book_id:
                continue
            if scope == "global" and not include_global:
                continue
        elif scope == "book" and item_book_id and not include_global:
            continue

        if q:
            source_text = str(item.get("source_text", "")).lower()
            target_text = str(item.get("target_text", "")).lower()
            note = str(item.get("note", "")).lower()
            if q not in source_text and q not in target_text and q not in note:
                continue

        filtered.append(item)

    filtered.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return filtered


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    """按 ID 获取条目。"""
    if not validate_safe_id(entry_id):
        raise ValueError("无效的条目 ID")

    with _LOCK:
        store = _load_store()
        for item in store.get("entries", []):
            if isinstance(item, dict) and item.get("id") == entry_id:
                return item
    return None


def create_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """创建条目。"""
    entry = _build_entry(payload)
    with _LOCK:
        store = _load_store()
        entries = store.get("entries", [])
        entries.append(entry)
        store["entries"] = entries
        if not _save_store(store):
            raise RuntimeError("保存术语库失败")
    return entry


def update_entry(entry_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """更新条目。"""
    if not validate_safe_id(entry_id):
        raise ValueError("无效的条目 ID")

    with _LOCK:
        store = _load_store()
        entries = store.get("entries", [])
        target_index = -1
        for i, item in enumerate(entries):
            if isinstance(item, dict) and item.get("id") == entry_id:
                target_index = i
                break

        if target_index < 0:
            return None

        entry = entries[target_index]
        assert isinstance(entry, dict)

        if "source_text" in payload:
            source_text = str(payload.get("source_text", "")).strip()
            if not source_text:
                raise ValueError("source_text 不能为空")
            entry["source_text"] = source_text

        if "target_text" in payload:
            target_text = str(payload.get("target_text", "")).strip()
            if not target_text:
                raise ValueError("target_text 不能为空")
            entry["target_text"] = target_text

        if "note" in payload:
            entry["note"] = str(payload.get("note", "")).strip()

        if "enabled" in payload:
            entry["enabled"] = bool(payload.get("enabled"))

        if "entry_type" in payload:
            entry_type = str(payload.get("entry_type", "")).strip()
            if entry_type not in _ALLOWED_ENTRY_TYPES:
                raise ValueError("entry_type 必须是 term 或 memory")
            entry["entry_type"] = entry_type

        if "scope" in payload:
            scope = str(payload.get("scope", "")).strip()
            if scope not in _ALLOWED_SCOPE:
                raise ValueError("scope 必须是 global 或 book")
            entry["scope"] = scope
            if scope == "global":
                entry["book_id"] = None

        if "book_id" in payload:
            book_id = payload.get("book_id")
            if book_id is None or book_id == "":
                entry["book_id"] = None
            else:
                if not isinstance(book_id, str) or not validate_safe_id(book_id):
                    raise ValueError("无效的 book_id")
                entry["book_id"] = book_id

        if entry.get("scope") == "book" and not entry.get("book_id"):
            raise ValueError("book 级条目必须包含 book_id")

        entry["updated_at"] = _now_iso()
        entries[target_index] = entry
        store["entries"] = entries
        if not _save_store(store):
            raise RuntimeError("保存术语库失败")
        return entry


def delete_entry(entry_id: str) -> bool:
    """删除条目。"""
    if not validate_safe_id(entry_id):
        raise ValueError("无效的条目 ID")

    with _LOCK:
        store = _load_store()
        entries = store.get("entries", [])
        before = len(entries)
        entries = [item for item in entries if not (isinstance(item, dict) and item.get("id") == entry_id)]
        if len(entries) == before:
            return False
        store["entries"] = entries
        if not _save_store(store):
            raise RuntimeError("保存术语库失败")
    return True

