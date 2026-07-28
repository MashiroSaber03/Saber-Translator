"""guard one nonterminal web-import commit per draft

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_jobs_one_nonterminal_web_commit_per_draft",
        "jobs",
        ["web_import_draft_id"],
        unique=True,
        sqlite_where=sa.text(
            "kind = 'web_import_commit' "
            "AND status IN "
            "('queued','running','pausing','paused','cancelling','interrupted') "
            "AND web_import_draft_id IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_jobs_one_nonterminal_web_commit_per_draft",
        table_name="jobs",
    )
