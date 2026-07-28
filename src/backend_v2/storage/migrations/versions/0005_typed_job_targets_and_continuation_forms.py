"""typed job targets and continuation form image versions

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("continuation_image_versions") as batch:
        batch.add_column(
            sa.Column(
                "thumbnail_asset_id",
                sa.String(length=36),
                nullable=False,
            )
        )
        batch.create_foreign_key(
            "fk_continuation_image_versions_thumbnail_asset_id_assets",
            "assets",
            ["thumbnail_asset_id"],
            ["id"],
            ondelete="RESTRICT",
        )
    op.create_table(
        "continuation_project_reference_assets",
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("asset_id", sa.String(length=36), nullable=False),
        sa.CheckConstraint(
            "ordinal >= 1",
            name="ck_continuation_project_reference_assets_ordinal_positive",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["continuation_projects.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("project_id", "ordinal"),
        sa.UniqueConstraint("project_id", "asset_id"),
    )
    with op.batch_alter_table("continuation_characters") as batch:
        batch.add_column(
            sa.Column(
                "revision",
                sa.Integer(),
                server_default="1",
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            )
        )
    with op.batch_alter_table("continuation_character_forms") as batch:
        batch.add_column(
            sa.Column(
                "reference_thumbnail_asset_id",
                sa.String(length=36),
            )
        )
        batch.add_column(
            sa.Column(
                "revision",
                sa.Integer(),
                server_default="1",
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            )
        )
        batch.create_foreign_key(
            "fk_continuation_character_forms_reference_thumbnail_asset_id_assets",
            "assets",
            ["reference_thumbnail_asset_id"],
            ["id"],
            ondelete="SET NULL",
        )
    with op.batch_alter_table("notes") as batch:
        batch.add_column(
            sa.Column(
                "kind",
                sa.String(length=16),
                server_default="text",
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "tags_json",
                sa.Text(),
                server_default="[]",
                nullable=False,
            )
        )
        batch.add_column(
            sa.Column(
                "comments_json",
                sa.Text(),
                server_default="[]",
                nullable=False,
            )
        )
        batch.create_check_constraint(
            "ck_notes_kind_values",
            "kind IN ('text','qa')",
        )
    with op.batch_alter_table("note_citations") as batch:
        batch.add_column(
            sa.Column("source_analysis_id", sa.String(length=36))
        )
        batch.add_column(
            sa.Column(
                "excerpt",
                sa.Text(),
                server_default="",
                nullable=False,
            )
        )
        batch.add_column(sa.Column("score", sa.Float()))
        batch.create_foreign_key(
            "fk_note_citations_source_analysis_id_analysis_page_results",
            "analysis_page_results",
            ["source_analysis_id"],
            ["id"],
            ondelete="SET NULL",
        )
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("analysis_run_id", sa.String(length=36)))
        batch.add_column(
            sa.Column("continuation_project_id", sa.String(length=36))
        )
        batch.create_foreign_key(
            "fk_jobs_analysis_run_id_analysis_runs",
            "analysis_runs",
            ["analysis_run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_foreign_key(
            "fk_jobs_continuation_project_id_continuation_projects",
            "continuation_projects",
            ["continuation_project_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.execute(
        "CREATE UNIQUE INDEX uq_jobs_one_current ON jobs (1) "
        "WHERE status IN "
        "('running','pausing','paused','cancelling')"
    )
    op.create_table(
        "continuation_form_image_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("form_id", sa.String(length=36), nullable=False),
        sa.Column("asset_id", sa.String(length=36), nullable=False),
        sa.Column("thumbnail_asset_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "is_adopted",
            sa.Boolean(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["form_id"],
            ["continuation_character_forms.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["thumbnail_asset_id"],
            ["assets.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("form_id", "version"),
    )


def downgrade() -> None:
    op.drop_table("continuation_project_reference_assets")
    with op.batch_alter_table("continuation_image_versions") as batch:
        batch.drop_constraint(
            "fk_continuation_image_versions_thumbnail_asset_id_assets",
            type_="foreignkey",
        )
        batch.drop_column("thumbnail_asset_id")
    op.drop_table("continuation_form_image_versions")
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint(
            "fk_jobs_continuation_project_id_continuation_projects",
            type_="foreignkey",
        )
        batch.drop_constraint(
            "fk_jobs_analysis_run_id_analysis_runs",
            type_="foreignkey",
        )
        batch.drop_column("continuation_project_id")
        batch.drop_column("analysis_run_id")
    op.execute(
        "CREATE UNIQUE INDEX uq_jobs_one_current ON jobs (1) "
        "WHERE status IN "
        "('running','pausing','paused','cancelling')"
    )
    with op.batch_alter_table("note_citations") as batch:
        batch.drop_constraint(
            "fk_note_citations_source_analysis_id_analysis_page_results",
            type_="foreignkey",
        )
        batch.drop_column("score")
        batch.drop_column("excerpt")
        batch.drop_column("source_analysis_id")
    with op.batch_alter_table("notes") as batch:
        batch.drop_constraint("ck_notes_kind_values", type_="check")
        batch.drop_column("comments_json")
        batch.drop_column("tags_json")
        batch.drop_column("kind")
    with op.batch_alter_table("continuation_character_forms") as batch:
        batch.drop_constraint(
            "fk_continuation_character_forms_reference_thumbnail_asset_id_assets",
            type_="foreignkey",
        )
        batch.drop_column("updated_at")
        batch.drop_column("created_at")
        batch.drop_column("revision")
        batch.drop_column("reference_thumbnail_asset_id")
    with op.batch_alter_table("continuation_characters") as batch:
        batch.drop_column("updated_at")
        batch.drop_column("created_at")
        batch.drop_column("revision")
