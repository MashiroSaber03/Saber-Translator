"""Add the durable Worker model lifecycle control channel.

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0011"
down_revision: Union[str, Sequence[str], None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "worker_commands",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("worker_epoch_id", sa.String(length=36)),
        sa.Column("result_json", sa.Text()),
        sa.Column("error_json", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "kind IN ('release_models')",
            name=op.f("ck_worker_commands_kind_values"),
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','completed','failed')",
            name=op.f("ck_worker_commands_status_values"),
        ),
        sa.ForeignKeyConstraint(
            ["worker_epoch_id"],
            ["process_epochs.id"],
            name=op.f(
                "fk_worker_commands_worker_epoch_id_process_epochs"
            ),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_worker_commands"),
        ),
    )
    op.create_index(
        "uq_worker_commands_one_active_kind",
        "worker_commands",
        ["kind"],
        unique=True,
        sqlite_where=sa.text("status IN ('pending','running')"),
    )
    op.create_index(
        "ix_worker_commands_claim",
        "worker_commands",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_worker_commands_claim",
        table_name="worker_commands",
    )
    op.drop_index(
        "uq_worker_commands_one_active_kind",
        table_name="worker_commands",
        sqlite_where=sa.text("status IN ('pending','running')"),
    )
    op.drop_table("worker_commands")
