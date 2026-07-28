"""Studio runtime contract and durable operation events.

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("studio_chat_sessions") as batch:
        batch.add_column(
            sa.Column(
                "greeting_source_json",
                sa.Text(),
                server_default="{}",
                nullable=False,
            )
        )
    op.create_table(
        "operation_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("operation_id", sa.String(length=36), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column(
            "payload_json",
            sa.Text(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["operation_id"],
            ["operations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_operation_events_operation_cursor",
        "operation_events",
        ["operation_id", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_operation_events_operation_cursor",
        table_name="operation_events",
    )
    op.drop_table("operation_events")
    with op.batch_alter_table("studio_chat_sessions") as batch:
        batch.drop_column("greeting_source_json")
