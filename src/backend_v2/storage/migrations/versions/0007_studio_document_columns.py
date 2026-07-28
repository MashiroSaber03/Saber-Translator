"""Normalize Character Studio documents into explicit domain columns.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-29
"""

from __future__ import annotations

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "studio_documents",
        sa.Column(
            "origin_type",
            sa.String(length=32),
            server_default="manual",
            nullable=False,
        ),
    )
    op.add_column(
        "studio_documents",
        sa.Column("source_character", sa.String(length=500)),
    )
    for name, default in (
        ("tags_json", "[]"),
        ("identity_json", "{}"),
        ("core_messages_json", "{}"),
        ("lorebook_json", "{}"),
        ("regex_scripts_json", "[]"),
        ("state_tasks_json", "[]"),
        ("frozen_sections_json", "[]"),
    ):
        op.add_column(
            "studio_documents",
            sa.Column(
                name,
                sa.Text(),
                server_default=default,
                nullable=False,
            ),
        )
    op.add_column(
        "studio_documents",
        sa.Column(
            "is_favorite",
            sa.Boolean(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "studio_documents",
        sa.Column("last_diagnostics_json", sa.Text()),
    )
    op.add_column(
        "studio_documents",
        sa.Column(
            "last_validated_at",
            sa.DateTime(timezone=True),
        ),
    )

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT id, kind, payload_json "
            "FROM studio_documents"
        )
    ).mappings()
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        origin = _object(payload.get("origin"))
        status = _object(payload.get("status"))
        meta = _object(payload.get("meta"))
        identity = _object(payload.get("identity"))
        identity.pop("name", None)
        origin_type = str(
            origin.get("type") or row["kind"] or "manual"
        )
        if origin_type not in {"analysis", "manual", "imported"}:
            origin_type = "manual"
        connection.execute(
            sa.text(
                "UPDATE studio_documents SET "
                "origin_type=:origin_type, "
                "source_character=:source_character, "
                "tags_json=:tags_json, "
                "is_favorite=:is_favorite, "
                "identity_json=:identity_json, "
                "core_messages_json=:core_messages_json, "
                "lorebook_json=:lorebook_json, "
                "regex_scripts_json=:regex_scripts_json, "
                "state_tasks_json=:state_tasks_json, "
                "frozen_sections_json=:frozen_sections_json, "
                "last_diagnostics_json=:last_diagnostics_json, "
                "last_validated_at=:last_validated_at "
                "WHERE id=:id"
            ),
            {
                "id": row["id"],
                "origin_type": origin_type,
                "source_character": origin.get("source_character"),
                "tags_json": _json(meta.get("tags", [])),
                "is_favorite": (
                    1 if status.get("is_favorite") else 0
                ),
                "identity_json": _json(identity),
                "core_messages_json": _json(
                    _object(payload.get("coreMessages"))
                ),
                "lorebook_json": _json(
                    _object(payload.get("lorebook"))
                ),
                "regex_scripts_json": _json(
                    payload.get("regexScripts", [])
                ),
                "state_tasks_json": _json(
                    payload.get("stateTasks", [])
                ),
                "frozen_sections_json": _json(
                    status.get("frozen_sections", [])
                ),
                "last_diagnostics_json": (
                    _json(status["last_diagnostics"])
                    if status.get("last_diagnostics") is not None
                    else None
                ),
                "last_validated_at": status.get(
                    "last_validated_at"
                ),
            },
        )

    with op.batch_alter_table(
        "studio_documents",
        recreate="always",
    ) as batch:
        batch.drop_constraint(
            "generation_positive",
            type_="check",
        )
        batch.drop_column("kind")
        batch.drop_column("generation")
        batch.drop_column("payload_json")
        batch.alter_column(
            "origin_type",
            server_default=None,
            existing_type=sa.String(length=32),
            existing_nullable=False,
        )
        batch.alter_column(
            "schema_version",
            server_default="2",
            existing_type=sa.Integer(),
            existing_nullable=False,
        )
        batch.create_check_constraint(
            "origin_type_values",
            "origin_type IN ('analysis','manual','imported')",
        )


def downgrade() -> None:
    op.add_column(
        "studio_documents",
        sa.Column(
            "kind",
            sa.String(length=32),
            server_default="manual",
            nullable=False,
        ),
    )
    op.add_column(
        "studio_documents",
        sa.Column(
            "generation",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
    )
    op.add_column(
        "studio_documents",
        sa.Column(
            "payload_json",
            sa.Text(),
            server_default="{}",
            nullable=False,
        ),
    )

    connection = op.get_bind()
    rows = connection.execute(
        sa.text("SELECT * FROM studio_documents")
    ).mappings()
    for row in rows:
        payload = {
            "origin": {
                "type": row["origin_type"],
                "source_character": row["source_character"],
                "source_pages": [],
            },
            "status": {
                "is_favorite": bool(row["is_favorite"]),
                "frozen_sections": _loads(
                    row["frozen_sections_json"],
                    [],
                ),
                "last_diagnostics": _loads(
                    row["last_diagnostics_json"],
                    None,
                ),
                "last_validated_at": _iso(
                    row["last_validated_at"]
                ),
            },
            "meta": {"tags": _loads(row["tags_json"], [])},
            "avatar": {"mode": "none", "source_page": None},
            "identity": _loads(row["identity_json"], {}),
            "coreMessages": _loads(
                row["core_messages_json"],
                {},
            ),
            "lorebook": _loads(row["lorebook_json"], {}),
            "regexScripts": _loads(
                row["regex_scripts_json"],
                [],
            ),
            "stateTasks": _loads(row["state_tasks_json"], []),
            "exportArtifacts": {},
        }
        connection.execute(
            sa.text(
                "UPDATE studio_documents SET "
                "kind=:kind, generation=1, payload_json=:payload "
                "WHERE id=:id"
            ),
            {
                "id": row["id"],
                "kind": row["origin_type"],
                "payload": _json(payload),
            },
        )

    with op.batch_alter_table(
        "studio_documents",
        recreate="always",
    ) as batch:
        batch.drop_constraint(
            "origin_type_values",
            type_="check",
        )
        for name in (
            "last_validated_at",
            "last_diagnostics_json",
            "frozen_sections_json",
            "state_tasks_json",
            "regex_scripts_json",
            "lorebook_json",
            "core_messages_json",
            "identity_json",
            "is_favorite",
            "tags_json",
            "source_character",
            "origin_type",
        ):
            batch.drop_column(name)
        batch.alter_column(
            "kind",
            server_default=None,
            existing_type=sa.String(length=32),
            existing_nullable=False,
        )
        batch.alter_column(
            "payload_json",
            server_default=None,
            existing_type=sa.Text(),
            existing_nullable=False,
        )
        batch.alter_column(
            "schema_version",
            server_default="1",
            existing_type=sa.Integer(),
            existing_nullable=False,
        )
        batch.create_check_constraint(
            "generation_positive",
            "generation >= 1",
        )


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _object(value: object) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _loads(value: object, default: object) -> object:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def _iso(value: object) -> str | None:
    if value is None:
        return None
    rendered = (
        value.isoformat()
        if hasattr(value, "isoformat")
        else str(value)
    )
    return rendered if rendered.endswith("Z") else rendered + "Z"
