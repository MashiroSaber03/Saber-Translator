"""store studio AI review independently from structural diagnostics

Revision ID: 0015
Revises: 0014
Create Date: 2026-07-31
"""

from __future__ import annotations

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0015"
down_revision: Union[str, Sequence[str], None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _normalize_review(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    nested = value.get("review")
    source = nested if isinstance(nested, dict) else value
    if any(
        key in source
        for key in ("checks", "errors", "passed", "warnings")
    ):
        return None
    summary = str(source.get("summary") or source.get("notes") or "").strip()
    if not summary:
        return None

    def string_list(candidate: object) -> list[str]:
        if not isinstance(candidate, list):
            return []
        return [
            rendered
            for item in candidate
            if (rendered := str(item).strip())
        ]

    return {
        "summary": summary,
        "issues": string_list(source.get("issues")),
        "suggestions": string_list(source.get("suggestions")),
    }


def upgrade() -> None:
    op.add_column(
        "studio_documents",
        sa.Column("last_review_json", sa.Text()),
    )

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT id, last_diagnostics_json "
            "FROM studio_documents "
            "WHERE last_diagnostics_json IS NOT NULL"
        )
    ).mappings()
    for row in rows:
        try:
            value = json.loads(str(row["last_diagnostics_json"]))
        except (TypeError, ValueError):
            continue
        review = _normalize_review(value)
        if review is None:
            continue
        connection.execute(
            sa.text(
                "UPDATE studio_documents SET "
                "last_review_json=:review, "
                "last_diagnostics_json=NULL, "
                "last_validated_at=NULL "
                "WHERE id=:id"
            ),
            {
                "id": row["id"],
                "review": json.dumps(
                    review,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "studio_documents",
        recreate="always",
    ) as batch:
        batch.drop_column("last_review_json")
