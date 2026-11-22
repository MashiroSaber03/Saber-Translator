"""Resolve immutable v2 font records to paths accepted by the pure renderer."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping

from sqlalchemy import or_, select
from sqlalchemy.engine import Connection

from src.backend_v2.content.page_style import (
    rgb_to_hex,
    validate_page_style,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.font_files import font_paths, prepare_font_directory
from src.backend_v2.storage.defaults import DEFAULT_FONT_ID
from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.storage.schema import bubbles, fonts, pages
from src.core.config_models import validate_bubble_payload


def character_font_style(path: str) -> dict[str, object]:
    """Describe the first face, matching the shared native font registration."""
    return dict(_character_font_style(path, Path(path).stat().st_mtime_ns))


@lru_cache(maxsize=128)
def _character_font_style(path: str, modified_ns: int) -> dict[str, object]:
    from fontTools.ttLib import TTFont

    with TTFont(path, fontNumber=0, lazy=True) as font:
        names = font["name"]

        def english_name(name_id: int) -> str | None:
            for language in (0x409, 0):
                for record in names.names:
                    if record.nameID == name_id and record.langID == language:
                        try:
                            value = record.toUnicode()
                        except UnicodeDecodeError:
                            continue
                        if value:
                            return value
            return None

        return {
            # Match the native collection's canonical family, before localized labels.
            "font_family": english_name(16) or english_name(1) or names.getDebugName(16) or names.getDebugName(1) or "",
            "font_style": names.getDebugName(17) or names.getDebugName(2) or "",
            "postscript_name": names.getDebugName(6),
            "weight": int(font["OS/2"].usWeightClass) if "OS/2" in font else 400,
            "italic": bool(font["head"].macStyle & 2) if "head" in font else False,
        }


def resolve_font_path(
    connection: Connection,
    storage: AssetStorageService,
    font_id: str | None,
) -> str:
    return resolve_font_paths(connection, storage, [font_id or DEFAULT_FONT_ID])[0]["path"]


def resolve_font_paths(connection: Connection, storage: AssetStorageService,
                       font_ids: list[str] | None = None) -> list[dict[str, str]]:
    """Resolve a catalog in one query and prepare the shared directory only once."""
    query = select(fonts).where(or_(fonts.c.owner_user_id.is_(None),
                                   fonts.c.owner_user_id == effective_owner_id()))
    if font_ids is not None:
        query = query.where(fonts.c.id.in_(font_ids))
    rows = list(connection.execute(query).mappings())
    if font_ids is not None and {row["id"] for row in rows} != set(font_ids):
        raise LookupError("font not found")
    prepare_font_directory(storage.data_root)
    result = []
    paths = font_paths(storage.data_root, [(row["relative_path"], row["owner_user_id"]) for row in rows], require_files=True)
    for row, path in zip(rows, paths):
        result.append({"id": row["id"], "path": str(path)})
    return result


def materialize_render_payloads(
    connection: Connection,
    storage: AssetStorageService,
    page_id: str,
    *,
    initialize_auto_fields: frozenset[str] = frozenset(),
    style_defaults_override: Mapping[str, object] | None = None,
    override_font_id: bool = False,
    font_id_override: str | None = None,
) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    page = connection.execute(
        select(
            pages.c.default_font_id,
            pages.c.page_style_defaults_json,
        ).where(pages.c.id == page_id)
    ).mappings().one()
    rows = list(
        connection.execute(
            select(
                bubbles.c.id,
                bubbles.c.font_id,
                bubbles.c.payload_json,
            )
            .where(bubbles.c.page_id == page_id)
            .order_by(bubbles.c.ordinal)
        ).mappings()
    )
    style_defaults = validate_page_style(
        (
            style_defaults_override
            if style_defaults_override is not None
            else json.loads(page["page_style_defaults_json"])
        ),
        partial=False,
    )
    font_ids = [
        (font_id_override if override_font_id else row["font_id"] or page["default_font_id"])
        or DEFAULT_FONT_ID
        for row in rows
    ]
    if not font_ids:
        return []
    paths_by_id = {
        item["id"]: item["path"]
        for item in resolve_font_paths(connection, storage, list(dict.fromkeys(font_ids)))
    }
    result = []
    for row, font_id in zip(rows, font_ids, strict=True):
        persisted = validate_bubble_payload(
            json.loads(row["payload_json"]),
            render=False,
        )
        if (
            "layoutDirection" in initialize_auto_fields
            and style_defaults["layoutDirection"] == "auto"
        ):
            persisted["textDirection"] = persisted["autoTextDirection"]
        if (
            initialize_auto_fields.intersection({"textColor", "fillColor"})
            and style_defaults["useAutoTextColor"]
        ):
            if (
                "textColor" in initialize_auto_fields
                and persisted["autoFgColor"] is not None
            ):
                persisted["textColor"] = rgb_to_hex(persisted["autoFgColor"])
            if (
                "fillColor" in initialize_auto_fields
                and persisted["autoBgColor"] is not None
            ):
                persisted["fillColor"] = rgb_to_hex(persisted["autoBgColor"])
        persisted = validate_bubble_payload(persisted, render=False)
        render_payload = validate_bubble_payload(
            {**persisted, "fontFamily": paths_by_id[font_id]},
            render=True,
        )
        result.append((str(row["id"]), persisted, render_payload))
    if result and "fontSize" in initialize_auto_fields and style_defaults["autoFontSize"]:
        from src.backend_v2.editing.native import NativeEditor

        fitted = NativeEditor.fit_payloads([render for _, _, render in result], fonts=NativeEditor(storage).font_files(connection))
        for index, (_, persisted, render) in enumerate(result):
            size = fitted.get(str(index))
            if size is not None:
                persisted["fontSize"] = render["fontSize"] = size
    return result
