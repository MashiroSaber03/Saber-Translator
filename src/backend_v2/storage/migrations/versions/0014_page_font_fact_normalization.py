"""normalize text-style ownership and store page fonts as foreign keys

Revision ID: 0014
Revises: 0013
Create Date: 2026-07-30
"""

from __future__ import annotations

import json
import math
import re
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0014"
down_revision: Union[str, Sequence[str], None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_PAGE_DEFAULTS = {
    "fontSize": 26,
    "autoFontSize": True,
    "layoutDirection": "auto",
    "textColor": "#000000",
    "fillColor": "#FFFFFF",
    "inpaintMethod": "solid",
    "useAutoTextColor": False,
    "strokeEnabled": True,
    "strokeColor": "#FFFFFF",
    "strokeWidth": 3,
    "lineSpacing": 1.0,
    "textAlign": "start",
}
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _json_object(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(value or "{}") if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _canonical_page_style(value: object) -> dict[str, object]:
    raw = _json_object(value)
    result = dict(_PAGE_DEFAULTS)
    font_size = raw.get("fontSize")
    if (
        isinstance(font_size, int)
        and not isinstance(font_size, bool)
        and 1 <= font_size <= 512
    ):
        result["fontSize"] = font_size
    for field in ("autoFontSize", "useAutoTextColor", "strokeEnabled"):
        if isinstance(raw.get(field), bool):
            result[field] = raw[field]
    if raw.get("layoutDirection") in {"auto", "vertical", "horizontal"}:
        result["layoutDirection"] = raw["layoutDirection"]
    if raw.get("inpaintMethod") in {"solid", "lama_mpe", "litelama"}:
        result["inpaintMethod"] = raw["inpaintMethod"]
    if raw.get("textAlign") in {"start", "center", "end"}:
        result["textAlign"] = raw["textAlign"]
    for field in ("textColor", "fillColor", "strokeColor"):
        candidate = raw.get(field)
        if isinstance(candidate, str) and _COLOR.fullmatch(candidate):
            result[field] = candidate
    stroke_width = raw.get("strokeWidth")
    if (
        isinstance(stroke_width, int)
        and not isinstance(stroke_width, bool)
        and 0 <= stroke_width <= 64
    ):
        result["strokeWidth"] = stroke_width
    line_spacing = raw.get("lineSpacing")
    if (
        isinstance(line_spacing, (int, float))
        and not isinstance(line_spacing, bool)
        and math.isfinite(float(line_spacing))
        and 0 < float(line_spacing) <= 10
    ):
        result["lineSpacing"] = float(line_spacing)
    return result


def upgrade() -> None:
    connection = op.get_bind()
    font_ids = set(
        connection.execute(sa.text("SELECT id FROM fonts")).scalars()
    )
    defaults_row = connection.execute(
        sa.text(
            "SELECT payload_json FROM app_settings "
            "WHERE domain = 'text_style_defaults'"
        )
    ).scalar_one_or_none()
    defaults_payload = _json_object(defaults_row)
    default_font = defaults_payload.get("fontFamily")
    if not isinstance(default_font, str) or default_font not in font_ids:
        default_font = next(iter(sorted(font_ids)), None)
    canonical_defaults = _canonical_page_style(defaults_payload)
    if defaults_row is not None and default_font is not None:
        connection.execute(
            sa.text(
                "UPDATE app_settings SET payload_json = :payload "
                "WHERE domain = 'text_style_defaults'"
            ),
            {
                "payload": json.dumps(
                    {**canonical_defaults, "fontFamily": default_font},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            },
        )

    translation_row = connection.execute(
        sa.text(
            "SELECT payload_json FROM app_settings WHERE domain = 'translation'"
        )
    ).scalar_one_or_none()
    if translation_row is not None:
        translation_payload = _json_object(translation_row)
        translation_payload.pop("textStyle", None)
        for key in (
            "translation",
            "hqTranslation",
            "pluginAgent",
            "aiVisionOcr",
        ):
            section = translation_payload.get(key)
            if isinstance(section, dict):
                section.pop("apiKey", None)
                section.pop("api_key", None)
        baidu = translation_payload.get("baiduOcr")
        if isinstance(baidu, dict):
            baidu.pop("apiKey", None)
            baidu.pop("secretKey", None)
        proofreading = translation_payload.get("proofreading")
        if isinstance(proofreading, dict):
            rounds = proofreading.get("rounds")
            if isinstance(rounds, list):
                for round_config in rounds:
                    if isinstance(round_config, dict):
                        round_config.pop("apiKey", None)
                        round_config.pop("api_key", None)
        connection.execute(
            sa.text(
                "UPDATE app_settings SET payload_json = :payload "
                "WHERE domain = 'translation'"
            ),
            {
                "payload": json.dumps(
                    translation_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            },
        )

    rows = connection.execute(
        sa.text(
            "SELECT id, default_font_id, page_style_defaults_json FROM pages"
        )
    ).mappings()
    for row in rows:
        raw_style = _json_object(row["page_style_defaults_json"])
        embedded_font = raw_style.pop("fontFamily", None)
        style = _canonical_page_style(
            {**canonical_defaults, **raw_style}
        )
        default_font_id = row["default_font_id"]
        if isinstance(embedded_font, str) and embedded_font in font_ids:
            default_font_id = embedded_font
        elif default_font_id not in font_ids:
            default_font_id = default_font
        connection.execute(
            sa.text(
                "UPDATE pages "
                "SET default_font_id = :default_font_id, "
                "page_style_defaults_json = :style "
                "WHERE id = :page_id"
            ),
            {
                "default_font_id": default_font_id,
                "style": json.dumps(
                    style,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "page_id": row["id"],
            },
        )


def downgrade() -> None:
    connection = op.get_bind()
    defaults_row = connection.execute(
        sa.text(
            "SELECT payload_json FROM app_settings "
            "WHERE domain = 'text_style_defaults'"
        )
    ).scalar_one_or_none()
    translation_row = connection.execute(
        sa.text(
            "SELECT payload_json FROM app_settings WHERE domain = 'translation'"
        )
    ).scalar_one_or_none()
    if defaults_row is not None and translation_row is not None:
        translation_payload = _json_object(translation_row)
        translation_payload["textStyle"] = _json_object(defaults_row)
        connection.execute(
            sa.text(
                "UPDATE app_settings SET payload_json = :payload "
                "WHERE domain = 'translation'"
            ),
            {
                "payload": json.dumps(
                    translation_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            },
        )
    rows = connection.execute(
        sa.text(
            "SELECT id, default_font_id, page_style_defaults_json FROM pages"
        )
    ).mappings()
    for row in rows:
        try:
            style = json.loads(row["page_style_defaults_json"] or "{}")
        except (TypeError, ValueError):
            style = {}
        if not isinstance(style, dict):
            style = {}
        if row["default_font_id"] is not None:
            style["fontFamily"] = row["default_font_id"]
        connection.execute(
            sa.text(
                "UPDATE pages SET page_style_defaults_json = :style "
                "WHERE id = :page_id"
            ),
            {
                "style": json.dumps(
                    style,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "page_id": row["id"],
            },
        )
