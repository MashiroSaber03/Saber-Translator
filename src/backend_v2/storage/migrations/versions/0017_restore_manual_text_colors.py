"""restore manual colors accidentally materialized from automatic backups

Revision ID: 0017
Revises: 0016
Create Date: 2026-07-31
"""

from __future__ import annotations

import json
import re
from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa


revision: str = "0017"
down_revision: Union[str, Sequence[str], None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _json_object(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(value or "{}") if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _rgb_hex(value: object) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        parts = [max(0, min(255, int(part))) for part in value[:3]]
    except (TypeError, ValueError):
        return None
    return f"#{parts[0]:02X}{parts[1]:02X}{parts[2]:02X}"


def _same_color(left: object, right: object) -> bool:
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and _COLOR.fullmatch(left) is not None
        and _COLOR.fullmatch(right) is not None
        and left.casefold() == right.casefold()
    )


def upgrade() -> None:
    connection = op.get_bind()
    rows = list(
        connection.execute(
            sa.text(
                "SELECT p.id AS page_id, p.document_revision, "
                "p.page_style_defaults_json, b.id AS bubble_id, "
                "b.payload_json "
                "FROM pages AS p "
                "JOIN bubbles AS b ON b.page_id = p.id "
                "ORDER BY p.id, b.ordinal"
            )
        ).mappings()
    )
    changed_by_page: dict[str, list[tuple[str, str]]] = {}
    page_revisions: dict[str, int] = {}
    for row in rows:
        style = _json_object(row["page_style_defaults_json"])
        if style.get("useAutoTextColor") is not False:
            continue
        manual_text = style.get("textColor")
        manual_fill = style.get("fillColor")
        if (
            not isinstance(manual_text, str)
            or _COLOR.fullmatch(manual_text) is None
            or not isinstance(manual_fill, str)
            or _COLOR.fullmatch(manual_fill) is None
        ):
            continue
        payload = _json_object(row["payload_json"])
        changed = False
        automatic_text = _rgb_hex(payload.get("autoFgColor"))
        if _same_color(payload.get("textColor"), automatic_text):
            payload["textColor"] = manual_text
            changed = True
        automatic_fill = _rgb_hex(payload.get("autoBgColor"))
        if _same_color(payload.get("fillColor"), automatic_fill):
            payload["fillColor"] = manual_fill
            changed = True
        if not changed:
            continue
        page_id = str(row["page_id"])
        changed_by_page.setdefault(page_id, []).append(
            (
                str(row["bubble_id"]),
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        page_revisions[page_id] = int(row["document_revision"])

    for page_id, changed_bubbles in changed_by_page.items():
        new_revision = page_revisions[page_id] + 1
        for bubble_id, payload_json in changed_bubbles:
            connection.execute(
                sa.text(
                    "UPDATE bubbles SET payload_json = :payload, "
                    "updated_revision = :revision, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = :bubble_id AND page_id = :page_id"
                ),
                {
                    "payload": payload_json,
                    "revision": new_revision,
                    "bubble_id": bubble_id,
                    "page_id": page_id,
                },
            )
        has_translated_asset = connection.execute(
            sa.text(
                "SELECT 1 FROM page_assets "
                "WHERE page_id = :page_id AND role = 'translated' LIMIT 1"
            ),
            {"page_id": page_id},
        ).scalar_one_or_none()
        connection.execute(
            sa.text(
                "UPDATE pages SET document_revision = :revision, "
                "render_status = CASE WHEN :has_translated = 1 "
                "THEN 'stale' ELSE render_status END, "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE id = :page_id"
            ),
            {
                "revision": new_revision,
                "has_translated": 1 if has_translated_asset is not None else 0,
                "page_id": page_id,
            },
        )
        if has_translated_asset is None:
            continue
        connection.execute(
            sa.text(
                "INSERT INTO render_requests "
                "(id, page_id, requested_revision, status, created_at, updated_at) "
                "VALUES (:id, :page_id, :revision, 'pending', "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                "ON CONFLICT(page_id) DO UPDATE SET "
                "requested_revision = excluded.requested_revision, "
                "rendering_revision = NULL, completed_revision = NULL, "
                "status = 'pending', executor_epoch_id = NULL, "
                "attempt_id = NULL, lease_token = NULL, "
                "lease_expires_at = NULL, error_json = NULL, "
                "updated_at = CURRENT_TIMESTAMP"
            ),
            {
                "id": str(uuid.uuid4()),
                "page_id": page_id,
                "revision": new_revision,
            },
        )


def downgrade() -> None:
    # This migration repairs contradictory persisted facts. Recreating the
    # accidental materialization would intentionally corrupt user data.
    pass
