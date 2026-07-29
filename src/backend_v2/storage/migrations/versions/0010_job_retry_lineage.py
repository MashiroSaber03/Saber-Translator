"""Add durable lineage for replacement retry jobs.

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0010"
down_revision: Union[str, Sequence[str], None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _restore_expression_indexes() -> None:
    # SQLite batch reflection cannot represent this expression index and drops
    # it while recreating jobs. Recreate it explicitly on both directions.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_one_current "
        "ON jobs ((1)) "
        "WHERE status IN ('running','pausing','paused','cancelling')"
    )


def upgrade() -> None:
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.add_column(sa.Column("retry_of_job_id", sa.String(36)))
        batch.add_column(sa.Column("retry_mode", sa.String(16)))
        batch.create_foreign_key(
            "fk_jobs_retry_of_job_id_jobs",
            "jobs",
            ["retry_of_job_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_check_constraint(
            "ck_jobs_retry_mode_values",
            "retry_mode IS NULL OR retry_mode IN ('current','original')",
        )
        batch.create_check_constraint(
            "ck_jobs_retry_lineage_complete",
            "(retry_of_job_id IS NULL AND retry_mode IS NULL) OR "
            "(retry_of_job_id IS NOT NULL AND retry_mode IS NOT NULL)",
        )
        batch.create_index(
            "ix_jobs_retry_source",
            ["retry_of_job_id"],
            unique=False,
        )
    _restore_expression_indexes()


def downgrade() -> None:
    with op.batch_alter_table("jobs", recreate="always") as batch:
        batch.drop_index("ix_jobs_retry_source")
        batch.drop_constraint(
            "fk_jobs_retry_of_job_id_jobs",
            type_="foreignkey",
        )
        batch.drop_constraint(
            "ck_jobs_retry_mode_values",
            type_="check",
        )
        batch.drop_constraint(
            "ck_jobs_retry_lineage_complete",
            type_="check",
        )
        batch.drop_column("retry_mode")
        batch.drop_column("retry_of_job_id")
    _restore_expression_indexes()
