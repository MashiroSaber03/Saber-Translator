"""normalize translation constraints to the structured v2 schema

Revision ID: 0013
Revises: 0012
Create Date: 2026-07-30
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0013"
down_revision: Union[str, Sequence[str], None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DEFAULT_PROMPT = """请从以下 OCR 文本中提取适合加入漫画术语表的实体。

提取范围：
1. 人名
2. 专有名词

输出要求：
1. 只输出 JSON 数组
2. 每项必须包含 source 和 target 字段
3. 不要输出空字段
4. 不要输出解释性文字
5. 如果没有可提取内容，返回 []

OCR 文本：
{ocr_text}"""


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _glossary_entries(value: object) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(value, list):
        return entries
    for raw in value:
        item = _mapping(raw)
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        mode = "regex" if item.get("matchMode") == "regex" else "text"
        if not source or not target or (mode, source) in seen:
            continue
        seen.add((mode, source))
        entries.append(
            {
                "source": source,
                "target": target,
                "note": str(item.get("note", "")).strip(),
                "matchMode": mode,
            }
        )
    return entries


def _non_translate_entries(value: object) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(value, list):
        return entries
    for raw in value:
        item = _mapping(raw)
        pattern = (
            str(raw).strip()
            if isinstance(raw, str)
            else str(item.get("pattern", item.get("content", ""))).strip()
        )
        mode = "regex" if item.get("matchMode") == "regex" else "text"
        if not pattern or (mode, pattern) in seen:
            continue
        seen.add((mode, pattern))
        entries.append(
            {
                "pattern": pattern,
                "note": str(item.get("note", "")).strip(),
                "matchMode": mode,
            }
        )
    return entries


def _upgrade_payload(value: object) -> dict[str, Any]:
    payload = _mapping(value)
    raw_glossary = payload.get("glossary")
    glossary = _mapping(raw_glossary)
    raw_non_translate = payload.get("nonTranslate", payload.get("non_translate"))
    non_translate = _mapping(raw_non_translate)
    return {
        "glossary": {
            "enabled": bool(glossary.get("enabled", bool(raw_glossary))),
            "autoExtractEnabled": bool(glossary.get("autoExtractEnabled", False)),
            "autoExtractPrompt": (
                str(glossary.get("autoExtractPrompt", "")).strip() or _DEFAULT_PROMPT
            ),
            "entries": _glossary_entries(
                glossary.get("entries", raw_glossary if isinstance(raw_glossary, list) else [])
            ),
        },
        "nonTranslate": {
            "enabled": bool(non_translate.get("enabled", bool(raw_non_translate))),
            "entries": _non_translate_entries(
                non_translate.get(
                    "entries",
                    raw_non_translate if isinstance(raw_non_translate, list) else [],
                )
            ),
        },
    }


def upgrade() -> None:
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT book_id, payload_json FROM translation_constraints "
            "WHERE schema_version < 2"
        )
    ).mappings()
    for row in rows:
        try:
            raw = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            raw = {}
        payload = json.dumps(
            _upgrade_payload(raw),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            sa.text(
                "UPDATE translation_constraints "
                "SET payload_json = :payload, schema_version = 2 "
                "WHERE book_id = :book_id"
            ),
            {"payload": payload, "book_id": row["book_id"]},
        )


def downgrade() -> None:
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT book_id, payload_json FROM translation_constraints "
            "WHERE schema_version = 2"
        )
    ).mappings()
    for row in rows:
        try:
            payload = _mapping(json.loads(row["payload_json"]))
        except (TypeError, json.JSONDecodeError):
            payload = {}
        glossary = _mapping(payload.get("glossary"))
        non_translate = _mapping(payload.get("nonTranslate"))
        legacy = {
            "glossary": glossary.get("entries", []),
            "nonTranslate": [
                entry.get("pattern", "")
                for entry in non_translate.get("entries", [])
                if isinstance(entry, Mapping) and entry.get("pattern")
            ],
        }
        connection.execute(
            sa.text(
                "UPDATE translation_constraints "
                "SET payload_json = :payload, schema_version = 1 "
                "WHERE book_id = :book_id"
            ),
            {
                "payload": json.dumps(
                    legacy,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "book_id": row["book_id"],
            },
        )
