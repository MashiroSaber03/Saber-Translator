"""Allow immutable plugin builds to share a package version label.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "0009"
down_revision: Union[str, Sequence[str], None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table(
        "plugin_versions",
        recreate="always",
    ) as batch:
        batch.drop_constraint(
            "uq_plugin_versions_plugin_id",
            type_="unique",
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "plugin_versions",
        recreate="always",
    ) as batch:
        batch.create_unique_constraint(
            "uq_plugin_versions_plugin_id",
            ["plugin_id", "version"],
        )
