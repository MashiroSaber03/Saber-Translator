"""bind transient vector queries to one active book

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("transient_requests") as batch:
        batch.add_column(sa.Column("book_id", sa.String(length=36)))
        batch.create_foreign_key(
            "fk_transient_requests_book_id_books",
            "books",
            ["book_id"],
            ["id"],
            ondelete="CASCADE",
        )
    op.create_index(
        "uq_transient_active_vector_query_book",
        "transient_requests",
        ["book_id"],
        unique=True,
        sqlite_where=sa.text(
            "kind = 'vector_query' "
            "AND book_id IS NOT NULL "
            "AND connection_open IS 1 "
            "AND status IN ('pending','running','completed')"
        ),
    )
    op.create_index(
        "ix_transient_requests_claim",
        "transient_requests",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_transient_requests_claim",
        table_name="transient_requests",
    )
    op.drop_index(
        "uq_transient_active_vector_query_book",
        table_name="transient_requests",
        sqlite_where=sa.text(
            "kind = 'vector_query' "
            "AND book_id IS NOT NULL "
            "AND connection_open IS 1 "
            "AND status IN ('pending','running','completed')"
        ),
    )
    with op.batch_alter_table("transient_requests") as batch:
        batch.drop_constraint(
            "fk_transient_requests_book_id_books",
            type_="foreignkey",
        )
        batch.drop_column("book_id")
