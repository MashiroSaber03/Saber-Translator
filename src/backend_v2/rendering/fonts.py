"""Resolve immutable v2 font records to paths accepted by the pure renderer."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.engine import Connection

from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import assets, bubbles, fonts, pages
from src.shared import constants


def resolve_font_path(
    connection: Connection,
    storage: AssetStorageService,
    font_id: str | None,
) -> str:
    if not font_id:
        return constants.DEFAULT_FONT_RELATIVE_PATH
    row = connection.execute(
        select(
            fonts.c.kind,
            fonts.c.builtin_key,
            assets.c.relative_path,
        )
        .outerjoin(assets, assets.c.id == fonts.c.asset_id)
        .where(fonts.c.id == font_id)
    ).mappings().one_or_none()
    if row is None:
        return constants.DEFAULT_FONT_RELATIVE_PATH
    if row["kind"] == "uploaded" and row["relative_path"]:
        return str(storage.resolve_relative_path(str(row["relative_path"])))
    return constants.DEFAULT_FONT_RELATIVE_PATH


def materialize_render_payloads(
    connection: Connection,
    storage: AssetStorageService,
    page_id: str,
) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    from src.core.rendering import calculate_auto_font_size

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
    style_defaults = json.loads(page["page_style_defaults_json"] or "{}")
    result = []
    for row in rows:
        persisted = json.loads(row["payload_json"])
        font_path = resolve_font_path(
            connection,
            storage,
            row["font_id"] or page["default_font_id"],
        )
        if style_defaults.get("layoutDirection") == "auto":
            auto_direction = persisted.get("autoTextDirection")
            if auto_direction in {"vertical", "horizontal"}:
                persisted["textDirection"] = auto_direction
        if style_defaults.get("useAutoTextColor"):
            if persisted.get("autoFgColor") is not None:
                persisted["textColor"] = _rgb_hex(persisted["autoFgColor"])
            if persisted.get("autoBgColor") is not None:
                persisted["fillColor"] = _rgb_hex(persisted["autoBgColor"])
        if style_defaults.get("autoFontSize"):
            coords = persisted.get("coords")
            if isinstance(coords, list) and len(coords) == 4:
                persisted["fontSize"] = calculate_auto_font_size(
                    str(persisted.get("translatedText", "")),
                    max(0, float(coords[2]) - float(coords[0])),
                    max(0, float(coords[3]) - float(coords[1])),
                    str(persisted.get("textDirection", "vertical")),
                    font_path,
                )
        render_payload = {**persisted, "fontFamily": font_path}
        result.append((str(row["id"]), persisted, render_payload))
    return result


def _rgb_hex(value: object) -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return "#000000"
    red, green, blue = (
        max(0, min(255, int(part)))
        for part in value[:3]
    )
    return f"#{red:02X}{green:02X}{blue:02X}"
