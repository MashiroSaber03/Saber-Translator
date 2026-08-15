"""Resolve immutable v2 font records to paths accepted by the pure renderer."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.engine import Connection

from src.backend_v2.content.page_style import rgb_to_hex, validate_page_style
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.builtin_fonts import resolve_bundled_font_path
from src.backend_v2.storage.schema import assets, bubbles, fonts, pages
from src.core.config_models import validate_bubble_payload


def resolve_font_path(
    connection: Connection,
    storage: AssetStorageService,
    font_id: str | None,
) -> str:
    if not font_id:
        return resolve_bundled_font_path("default")
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
        raise LookupError("font not found")
    if row["kind"] == "uploaded":
        if not row["relative_path"]:
            raise RuntimeError("uploaded font asset is missing")
        return str(storage.resolve_relative_path(str(row["relative_path"])))
    if row["kind"] == "builtin":
        return resolve_bundled_font_path(str(row["builtin_key"]))
    raise RuntimeError("unsupported builtin font")


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
    if "fontSize" in initialize_auto_fields:
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
    style_defaults = validate_page_style(
        (
            style_defaults_override
            if style_defaults_override is not None
            else json.loads(page["page_style_defaults_json"])
        ),
        partial=False,
    )
    result = []
    for row in rows:
        persisted = validate_bubble_payload(
            json.loads(row["payload_json"]),
            render=False,
        )
        font_path = resolve_font_path(
            connection,
            storage,
            (
                font_id_override
                if override_font_id
                else row["font_id"] or page["default_font_id"]
            ),
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
        if (
            "fontSize" in initialize_auto_fields
            and style_defaults["autoFontSize"]
        ):
            coords = persisted["coords"]
            persisted["fontSize"] = calculate_auto_font_size(
                persisted["translatedText"],
                coords[2] - coords[0],
                coords[3] - coords[1],
                persisted["textDirection"],
                font_path,
            )
        persisted = validate_bubble_payload(persisted, render=False)
        render_payload = validate_bubble_payload(
            {**persisted, "fontFamily": font_path},
            render=True,
        )
        result.append((str(row["id"]), persisted, render_payload))
    return result
