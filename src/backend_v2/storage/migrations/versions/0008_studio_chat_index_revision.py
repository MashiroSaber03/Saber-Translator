"""Add an independent Studio chat-session index revision.

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table(
        "studio_documents",
        recreate="always",
    ) as batch:
        batch.add_column(
            sa.Column(
                "chat_index_revision",
                sa.Integer(),
                server_default="1",
                nullable=False,
            )
        )
        batch.create_check_constraint(
            "chat_index_revision_positive",
            "chat_index_revision >= 1",
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "studio_documents",
        recreate="always",
    ) as batch:
        batch.drop_constraint(
            "chat_index_revision_positive",
            type_="check",
        )
        batch.drop_column("chat_index_revision")
