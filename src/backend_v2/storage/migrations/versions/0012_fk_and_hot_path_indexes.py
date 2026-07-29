"""Add complete foreign-key and hot-path lookup indexes.

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-29
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}

_ACTIVE_OPERATION_TARGET_SHAPE = (
    "((kind IN ('bubble_ocr','bubble_color','bubble_translate') "
    "AND studio_document_id IS NULL AND studio_session_id IS NULL "
    "AND ((status IN ('pending','running') "
    "AND page_id IS NOT NULL AND bubble_id IS NOT NULL) "
    "OR status IN ('completed','failed','cancelled'))) OR "
    "(kind IN ('page_detect','page_repair') "
    "AND bubble_id IS NULL AND studio_document_id IS NULL "
    "AND studio_session_id IS NULL "
    "AND ((status IN ('pending','running') AND page_id IS NOT NULL) "
    "OR status IN ('completed','failed','cancelled'))) OR "
    "(kind = 'studio_generate' AND page_id IS NULL AND bubble_id IS NULL "
    "AND studio_session_id IS NULL "
    "AND ((status IN ('pending','running') "
    "AND studio_document_id IS NOT NULL) "
    "OR status IN ('completed','failed','cancelled'))) OR "
    "(kind IN ('studio_chat','studio_summary') "
    "AND page_id IS NULL AND bubble_id IS NULL "
    "AND studio_document_id IS NULL "
    "AND ((status IN ('pending','running') "
    "AND studio_session_id IS NOT NULL) "
    "OR status IN ('completed','failed','cancelled'))))"
)

_LEGACY_OPERATION_TARGET_SHAPE = (
    "(kind IN ('bubble_ocr','bubble_color','bubble_translate') "
    "AND page_id IS NOT NULL AND bubble_id IS NOT NULL "
    "AND studio_document_id IS NULL AND studio_session_id IS NULL) OR "
    "(kind IN ('page_detect','page_repair') "
    "AND page_id IS NOT NULL AND studio_document_id IS NULL "
    "AND studio_session_id IS NULL) OR "
    "(kind = 'studio_generate' AND studio_document_id IS NOT NULL "
    "AND page_id IS NULL AND bubble_id IS NULL AND studio_session_id IS NULL) OR "
    "(kind IN ('studio_chat','studio_summary') "
    "AND studio_session_id IS NOT NULL "
    "AND page_id IS NULL AND bubble_id IS NULL AND studio_document_id IS NULL)"
)

_INDEXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "ix_job_items_job_status_ordinal",
        "job_items",
        ("job_id", "status", "ordinal"),
    ),
    ("ix_plugin_versions_plugin_id", "plugin_versions", ("plugin_id",)),
    ("ix_fonts_asset_id", "fonts", ("asset_id",)),
    ("ix_books_cover_asset_id", "books", ("cover_asset_id",)),
    ("ix_pages_default_font_id", "pages", ("default_font_id",)),
    ("ix_chapter_navigation_state_last_page", "chapter_navigation_state", ("last_visited_page_id",)),
    ("ix_bubbles_font_id", "bubbles", ("font_id",)),
    ("ix_provider_settings_credential_version", "provider_settings", ("credential_version_id",)),
    ("ix_web_import_drafts_book_id", "web_import_drafts", ("book_id",)),
    ("ix_web_import_draft_pages_thumbnail_asset", "web_import_draft_pages", ("thumbnail_asset_id",)),
    ("ix_jobs_book_id", "jobs", ("book_id",)),
    ("ix_jobs_page_id", "jobs", ("page_id",)),
    ("ix_jobs_analysis_run_id", "jobs", ("analysis_run_id",)),
    ("ix_jobs_continuation_project_id", "jobs", ("continuation_project_id",)),
    ("ix_jobs_blocked_by_job_id", "jobs", ("blocked_by_job_id",)),
    ("ix_jobs_blocked_by_import_lease_id", "jobs", ("blocked_by_import_lease_id",)),
    ("ix_jobs_worker_epoch_id", "jobs", ("worker_epoch_id",)),
    ("ix_job_items_page_id", "job_items", ("page_id",)),
    ("ix_job_drain_acks_last_step_id", "job_drain_acks", ("last_step_id",)),
    ("ix_job_asset_inputs_job_item_id", "job_asset_inputs", ("job_item_id",)),
    ("ix_job_step_asset_outputs_asset_id", "job_step_asset_outputs", ("asset_id",)),
    ("ix_job_artifacts_asset_id", "job_artifacts", ("asset_id",)),
    ("ix_worker_commands_worker_epoch_id", "worker_commands", ("worker_epoch_id",)),
    ("ix_studio_documents_book_id", "studio_documents", ("book_id",)),
    ("ix_studio_documents_avatar_asset_id", "studio_documents", ("avatar_asset_id",)),
    ("ix_studio_chat_sessions_summary_message", "studio_chat_sessions", ("summary_through_message_id",)),
    ("ix_analysis_run_targets_chapter_id", "analysis_run_targets", ("chapter_id",)),
    ("ix_analysis_run_targets_source_asset_id", "analysis_run_targets", ("source_asset_id",)),
    ("ix_analysis_page_results_source_asset_id", "analysis_page_results", ("source_asset_id",)),
    ("ix_analysis_heads_active_run_id", "analysis_heads", ("active_run_id",)),
    ("ix_analysis_heads_active_result_id", "analysis_heads", ("active_result_id",)),
    ("ix_analysis_layer_results_chapter_id", "analysis_layer_results", ("chapter_id",)),
    ("ix_analysis_layer_result_pages_page_id", "analysis_layer_result_pages", ("page_id",)),
    ("ix_analysis_artifacts_run_id", "analysis_artifacts", ("run_id",)),
    ("ix_analysis_artifacts_asset_id", "analysis_artifacts", ("asset_id",)),
    ("ix_timeline_versions_run_id", "timeline_versions", ("run_id",)),
    ("ix_vector_generations_run_id", "vector_generations", ("run_id",)),
    ("ix_note_citations_page_id", "note_citations", ("page_id",)),
    ("ix_note_citations_source_analysis_id", "note_citations", ("source_analysis_id",)),
    ("ix_continuation_projects_source_run_id", "continuation_projects", ("source_run_id",)),
    ("ix_continuation_scripts_project_id", "continuation_scripts", ("project_id",)),
    ("ix_continuation_image_versions_asset_id", "continuation_image_versions", ("asset_id",)),
    ("ix_continuation_image_versions_thumbnail_asset_id", "continuation_image_versions", ("thumbnail_asset_id",)),
    ("ix_continuation_project_reference_assets_asset_id", "continuation_project_reference_assets", ("asset_id",)),
    ("ix_continuation_character_forms_reference_asset_id", "continuation_character_forms", ("reference_asset_id",)),
    ("ix_continuation_character_forms_reference_thumbnail_asset_id", "continuation_character_forms", ("reference_thumbnail_asset_id",)),
    ("ix_continuation_character_forms_adopted_asset_id", "continuation_character_forms", ("adopted_asset_id",)),
    ("ix_continuation_form_image_versions_asset_id", "continuation_form_image_versions", ("asset_id",)),
    ("ix_continuation_form_image_versions_thumbnail_asset_id", "continuation_form_image_versions", ("thumbnail_asset_id",)),
    ("ix_operations_page_id", "operations", ("page_id",)),
    ("ix_operations_bubble_id", "operations", ("bubble_id",)),
    ("ix_operations_executor_epoch_id", "operations", ("executor_epoch_id",)),
    ("ix_operation_asset_inputs_asset_id", "operation_asset_inputs", ("asset_id",)),
    ("ix_operation_artifacts_asset_id", "operation_artifacts", ("asset_id",)),
    ("ix_operation_artifacts_page_id", "operation_artifacts", ("page_id",)),
    ("ix_transient_requests_worker_epoch_id", "transient_requests", ("worker_epoch_id",)),
    ("ix_render_requests_executor_epoch_id", "render_requests", ("executor_epoch_id",)),
    ("ix_page_assets_asset_id", "page_assets", ("asset_id",)),
    ("ix_page_assets_parent_asset_id", "page_assets", ("parent_asset_id",)),
    ("ix_page_assets_producer_job_step_id", "page_assets", ("producer_job_step_id",)),
    ("ix_page_assets_producer_operation_id", "page_assets", ("producer_operation_id",)),
    ("ix_page_assets_producer_render_request_id", "page_assets", ("producer_render_request_id",)),
    ("ix_jobs_chapter_status", "jobs", ("chapter_id", "status")),
    ("ix_jobs_batch_status", "jobs", ("batch_id", "status")),
    (
        "ix_web_import_drafts_chapter_status_expiry",
        "web_import_drafts",
        ("chapter_id", "status", "expires_at"),
    ),
)


def _set_business_asset_delete_policy(ondelete: str) -> None:
    targets = (
        (
            "studio_documents",
            ("avatar_asset_id",),
        ),
        (
            "continuation_character_forms",
            (
                "reference_asset_id",
                "reference_thumbnail_asset_id",
                "adopted_asset_id",
            ),
        ),
    )
    for table_name, columns in targets:
        with op.batch_alter_table(
            table_name,
            recreate="always",
            naming_convention=_NAMING_CONVENTION,
        ) as batch:
            for column in columns:
                constraint_name = (
                    f"fk_{table_name}_{column}_assets"
                )
                batch.drop_constraint(
                    constraint_name,
                    type_="foreignkey",
                )
                batch.create_foreign_key(
                    constraint_name,
                    "assets",
                    [column],
                    ["id"],
                    ondelete=ondelete,
                )


def _set_operation_target_shape(expression: str) -> None:
    with op.batch_alter_table(
        "operations",
        recreate="always",
        naming_convention=_NAMING_CONVENTION,
    ) as batch:
        batch.drop_constraint(
            op.f("ck_operations_kind_target_shape"),
            type_="check",
        )
        batch.create_check_constraint(
            op.f("ck_operations_kind_target_shape"),
            expression,
        )


def upgrade() -> None:
    _set_operation_target_shape(_ACTIVE_OPERATION_TARGET_SHAPE)
    _set_business_asset_delete_policy("RESTRICT")
    for name, table_name, columns in _INDEXES:
        op.create_index(name, table_name, list(columns))


def downgrade() -> None:
    for name, table_name, _columns in reversed(_INDEXES):
        op.drop_index(name, table_name=table_name)
    _set_business_asset_delete_policy("SET NULL")
    _set_operation_target_shape(_LEGACY_OPERATION_TARGET_SHAPE)
