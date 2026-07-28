"""Phase-0 SQLAlchemy Core metadata for the v2 fact model.

The metadata intentionally encodes constraints that must be enforced by SQLite
rather than by handler-level prechecks.  Later phases add domain repositories
and may add tables through new migrations; they must not weaken these baseline
invariants.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    literal,
    text,
)


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

UUID_LENGTH = 36
HASH_LENGTH = 64

JOB_STATUSES = (
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "cancelled",
    "completed",
    "completed_with_errors",
    "failed",
    "interrupted",
)
CURRENT_JOB_STATUSES = ("running", "pausing", "paused", "cancelling")
NONTERMINAL_JOB_STATUSES = (
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "interrupted",
)
JOB_KINDS = (
    "translation",
    "remove_text",
    "detect",
    "style_apply",
    "text_import",
    "container_import",
    "web_extract",
    "web_import_commit",
    "export",
    "insight_analysis",
    "insight_export",
    "vector_rebuild",
    "continuation",
    "derived_rebuild",
    "plugin_agent",
)
OPERATION_STATUSES = ("pending", "running", "completed", "failed", "cancelled")
OPERATION_KINDS = (
    "bubble_ocr",
    "bubble_color",
    "page_detect",
    "page_repair",
    "bubble_translate",
    "studio_generate",
    "studio_chat",
    "studio_summary",
)
PAGE_WRITE_OPERATION_KINDS = (
    "bubble_ocr",
    "bubble_color",
    "page_detect",
    "page_repair",
    "bubble_translate",
)
PAGE_ASSET_ROLES = (
    "source",
    "thumbnail_source",
    "clean",
    "translated",
    "thumbnail_translated",
    "text_mask",
)
PROMPT_TYPES = (
    "translate",
    "textbox",
    "ai_vision_ocr",
    "hq_translate",
    "proofreading",
    "batch_analysis",
    "segment_summary",
    "chapter_summary",
    "book_overview",
    "group_summary",
    "qa_response",
    "question_decompose",
    "analysis_system",
)


def _sql_values(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _timestamps() -> tuple[Column[DateTime], Column[DateTime]]:
    return (
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )


assets = Table(
    "assets",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("relative_path", Text, nullable=False, unique=True),
    Column("mime_type", String(127), nullable=False),
    Column("checksum", String(HASH_LENGTH), nullable=False),
    Column("byte_size", BigInteger, nullable=False),
    Column("width", Integer),
    Column("height", Integer),
    Column("integrity_status", String(16), nullable=False, server_default="ok"),
    Column("gc_marked_at", DateTime(timezone=True)),
    *_timestamps(),
    CheckConstraint("byte_size >= 0", name="byte_size_nonnegative"),
    CheckConstraint(
        "integrity_status IN ('ok', 'missing')",
        name="integrity_status_values",
    ),
)
Index("ix_assets_checksum", assets.c.checksum)
Index("ix_assets_gc_marked_at", assets.c.gc_marked_at)

credentials = Table(
    "credentials",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("domain", String(64), nullable=False),
    Column("provider", String(64), nullable=False),
    *_timestamps(),
    UniqueConstraint("domain", "provider"),
)

credential_versions = Table(
    "credential_versions",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column(
        "credential_id",
        String(UUID_LENGTH),
        ForeignKey("credentials.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("version", Integer, nullable=False),
    Column("secret_json", Text, nullable=False),
    Column("key_fingerprint", String(HASH_LENGTH), nullable=False),
    Column("retired_at", DateTime(timezone=True)),
    *_timestamps(),
    UniqueConstraint("credential_id", "version"),
    CheckConstraint("version >= 1", name="version_positive"),
)

credential_current_versions = Table(
    "credential_current_versions",
    metadata,
    Column(
        "credential_id",
        String(UUID_LENGTH),
        ForeignKey("credentials.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "credential_version_id",
        String(UUID_LENGTH),
        ForeignKey("credential_versions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("revision", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    CheckConstraint("revision >= 1", name="revision_positive"),
)

plugins = Table(
    "plugins",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("state", String(16), nullable=False, server_default="enabled"),
    *_timestamps(),
    CheckConstraint("state IN ('enabled', 'disabled', 'error')", name="state_values"),
)

plugin_versions = Table(
    "plugin_versions",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("plugin_id", String(UUID_LENGTH), ForeignKey("plugins.id", ondelete="CASCADE"), nullable=False),
    Column("version", String(64), nullable=False),
    Column("package_relative_path", Text, nullable=False, unique=True),
    Column("checksum", String(HASH_LENGTH), nullable=False),
    Column("manifest_json", Text, nullable=False),
    Column("manifest_schema_version", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    UniqueConstraint("plugin_id", "version"),
)

plugin_current_versions = Table(
    "plugin_current_versions",
    metadata,
    Column(
        "plugin_id",
        String(UUID_LENGTH),
        ForeignKey("plugins.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "plugin_version_id",
        String(UUID_LENGTH),
        ForeignKey("plugin_versions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    ),
    Column("revision", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    CheckConstraint("revision >= 1", name="revision_positive"),
)

fonts = Table(
    "fonts",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("kind", String(16), nullable=False),
    Column("display_name", String(200), nullable=False),
    Column("asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT")),
    Column("builtin_key", String(200)),
    *_timestamps(),
    CheckConstraint("kind IN ('builtin', 'uploaded')", name="kind_values"),
    CheckConstraint(
        "(kind = 'builtin' AND builtin_key IS NOT NULL AND asset_id IS NULL) OR "
        "(kind = 'uploaded' AND builtin_key IS NULL AND asset_id IS NOT NULL)",
        name="source_shape",
    ),
)
Index(
    "uq_fonts_builtin_key",
    fonts.c.builtin_key,
    unique=True,
    sqlite_where=fonts.c.builtin_key.is_not(None),
)

books = Table(
    "books",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("kind", String(24), nullable=False, server_default="library"),
    Column("title", String(500), nullable=False),
    Column("chapter_order_revision", Integer, nullable=False, server_default="1"),
    Column("cover_asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT")),
    *_timestamps(),
    CheckConstraint("kind IN ('library', 'quick_workspace')", name="kind_values"),
    CheckConstraint("chapter_order_revision >= 1", name="chapter_order_revision_positive"),
)
Index(
    "uq_books_one_quick_workspace",
    books.c.kind,
    unique=True,
    sqlite_where=books.c.kind == "quick_workspace",
)

chapters = Table(
    "chapters",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("book_id", String(UUID_LENGTH), ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("title", String(500), nullable=False),
    Column("page_order_revision", Integer, nullable=False, server_default="1"),
    Column("write_intent_generation", Integer, nullable=False, server_default="0"),
    Column("settings_memory_json", Text, nullable=False, server_default="{}"),
    Column("settings_memory_schema_version", Integer, nullable=False, server_default="1"),
    Column("settings_memory_revision", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    UniqueConstraint("book_id", "ordinal"),
    CheckConstraint("ordinal >= 1", name="ordinal_positive"),
    CheckConstraint("page_order_revision >= 1", name="page_order_revision_positive"),
    CheckConstraint("write_intent_generation >= 0", name="intent_generation_nonnegative"),
)

pages = Table(
    "pages",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("chapter_id", String(UUID_LENGTH), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("logical_source_path", Text, nullable=False),
    Column("source_revision", Integer, nullable=False, server_default="1"),
    Column("document_revision", Integer, nullable=False, server_default="1"),
    Column("rendered_revision", Integer),
    Column("render_status", String(32), nullable=False, server_default="not_rendered"),
    Column("detection_state", String(32), nullable=False, server_default="unprocessed"),
    Column("default_font_id", String(UUID_LENGTH), ForeignKey("fonts.id", ondelete="RESTRICT")),
    Column("page_style_defaults_json", Text, nullable=False, server_default="{}"),
    Column("page_style_schema_version", Integer, nullable=False, server_default="1"),
    Column("warnings_json", Text, nullable=False, server_default="[]"),
    *_timestamps(),
    UniqueConstraint("chapter_id", "ordinal"),
    UniqueConstraint("chapter_id", "logical_source_path"),
    CheckConstraint("ordinal >= 1", name="ordinal_positive"),
    CheckConstraint("source_revision >= 1", name="source_revision_positive"),
    CheckConstraint("document_revision >= 1", name="document_revision_positive"),
    CheckConstraint(
        "render_status IN ("
        "'not_rendered','ready','stale','rendering','render_failed',"
        "'awaiting_repair','repair_failed')",
        name="render_status_values",
    ),
)

chapter_navigation_state = Table(
    "chapter_navigation_state",
    metadata,
    Column(
        "chapter_id",
        String(UUID_LENGTH),
        ForeignKey("chapters.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "last_visited_page_id",
        String(UUID_LENGTH),
        ForeignKey("pages.id", ondelete="SET NULL"),
    ),
    Column("revision", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    CheckConstraint("revision >= 1", name="revision_positive"),
)

bubbles = Table(
    "bubbles",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("page_id", String(UUID_LENGTH), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("font_id", String(UUID_LENGTH), ForeignKey("fonts.id", ondelete="RESTRICT")),
    Column("payload_json", Text, nullable=False),
    Column("payload_schema_version", Integer, nullable=False, server_default="1"),
    Column("updated_revision", Integer, nullable=False),
    *_timestamps(),
    UniqueConstraint("page_id", "ordinal"),
    CheckConstraint("ordinal >= 1", name="ordinal_positive"),
    CheckConstraint("updated_revision >= 1", name="updated_revision_positive"),
)

tags = Table(
    "tags",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("color", String(16), nullable=False),
    *_timestamps(),
)

book_tags = Table(
    "book_tags",
    metadata,
    Column("book_id", String(UUID_LENGTH), ForeignKey("books.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", String(UUID_LENGTH), ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)

translation_constraints = Table(
    "translation_constraints",
    metadata,
    Column("book_id", String(UUID_LENGTH), ForeignKey("books.id", ondelete="CASCADE"), primary_key=True),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("payload_json", Text, nullable=False, server_default="{}"),
    Column("schema_version", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    CheckConstraint("revision >= 1", name="revision_positive"),
)

app_settings = Table(
    "app_settings",
    metadata,
    Column("domain", String(64), primary_key=True),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("payload_json", Text, nullable=False),
    Column("schema_version", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    CheckConstraint("revision >= 1", name="revision_positive"),
)

book_settings = Table(
    "book_settings",
    metadata,
    Column("book_id", String(UUID_LENGTH), ForeignKey("books.id", ondelete="CASCADE"), primary_key=True),
    Column("domain", String(64), primary_key=True),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("payload_json", Text, nullable=False),
    Column("schema_version", Integer, nullable=False, server_default="1"),
    *_timestamps(),
)

provider_settings = Table(
    "provider_settings",
    metadata,
    Column("domain", String(64), primary_key=True),
    Column("provider", String(64), primary_key=True),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("payload_json", Text, nullable=False),
    Column("schema_version", Integer, nullable=False, server_default="1"),
    Column(
        "credential_version_id",
        String(UUID_LENGTH),
        ForeignKey("credential_versions.id", ondelete="RESTRICT"),
    ),
    *_timestamps(),
)

prompts = Table(
    "prompts",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("type", String(32), nullable=False),
    Column("name", String(200), nullable=False),
    Column("content", Text, nullable=False),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("is_factory_default", Boolean, nullable=False, server_default="0"),
    *_timestamps(),
    UniqueConstraint("type", "name"),
    CheckConstraint(f"type IN ({_sql_values(PROMPT_TYPES)})", name="type_values"),
    CheckConstraint("revision >= 1", name="revision_positive"),
)

web_import_drafts = Table(
    "web_import_drafts",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("book_id", String(UUID_LENGTH), ForeignKey("books.id", ondelete="SET NULL")),
    Column("chapter_id", String(UUID_LENGTH), ForeignKey("chapters.id", ondelete="SET NULL")),
    Column("status", String(24), nullable=False),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("config_json", Text, nullable=False),
    Column("config_schema_version", Integer, nullable=False, server_default="1"),
    Column("temp_relative_path", Text, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    *_timestamps(),
    CheckConstraint(
        "status IN ('extracting','ready','committing','completed','failed','cancelled')",
        name="status_values",
    ),
)

web_import_draft_pages = Table(
    "web_import_draft_pages",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column(
        "draft_id",
        String(UUID_LENGTH),
        ForeignKey("web_import_drafts.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("selected", Boolean, nullable=False, server_default="1"),
    Column("source_url", Text, nullable=False),
    Column("temp_relative_path", Text, nullable=False),
    Column("thumbnail_asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT")),
    Column("checksum", String(HASH_LENGTH)),
    Column("error_json", Text),
    *_timestamps(),
    UniqueConstraint("draft_id", "ordinal"),
)

job_batches = Table(
    "job_batches",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("kind", String(64), nullable=False),
    Column("display_name", String(500), nullable=False),
    Column("status_summary_json", Text, nullable=False, server_default="{}"),
    *_timestamps(),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("batch_id", String(UUID_LENGTH), ForeignKey("job_batches.id", ondelete="SET NULL")),
    Column("kind", String(64), nullable=False),
    Column("status", String(32), nullable=False),
    Column("queue_rank", Integer, unique=True),
    Column("book_id", String(UUID_LENGTH), ForeignKey("books.id", ondelete="SET NULL")),
    Column("chapter_id", String(UUID_LENGTH), ForeignKey("chapters.id", ondelete="SET NULL")),
    Column("page_id", String(UUID_LENGTH), ForeignKey("pages.id", ondelete="SET NULL")),
    Column(
        "web_import_draft_id",
        String(UUID_LENGTH),
        ForeignKey("web_import_drafts.id", ondelete="SET NULL"),
    ),
    Column("blocked_reason", String(64)),
    Column("blocked_by_job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="SET NULL")),
    Column("blocked_by_import_lease_id", String(UUID_LENGTH)),
    Column("attempt_id", String(UUID_LENGTH)),
    Column("lease_token", String(200)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column(
        "worker_epoch_id",
        String(UUID_LENGTH),
        ForeignKey("process_epochs.id", ondelete="SET NULL"),
    ),
    Column("config_json", Text, nullable=False),
    Column("config_schema_version", Integer, nullable=False, server_default="1"),
    Column("latest_progress_json", Text, nullable=False, server_default="{}"),
    Column("target_display_json", Text, nullable=False, server_default="{}"),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    *_timestamps(),
    CheckConstraint(f"status IN ({_sql_values(JOB_STATUSES)})", name="status_values"),
    CheckConstraint(f"kind IN ({_sql_values(JOB_KINDS)})", name="kind_values"),
    CheckConstraint("queue_rank IS NULL OR queue_rank >= 1", name="queue_rank_positive"),
    CheckConstraint(
        "blocked_reason IS NULL OR blocked_reason IN ("
        "'blocked_by_job','blocked_by_import_lease','draining_immediate_writes')",
        name="blocked_reason_values",
    ),
    CheckConstraint(
        "(blocked_by_job_id IS NULL OR blocked_by_import_lease_id IS NULL)",
        name="single_blocker",
    ),
)

# The indexed expression is always 1 under the partial predicate, providing the
# same invariant as ``UNIQUE INDEX ... ON jobs ((1)) WHERE status IN (...)``
# while remaining naturally attached to SQLAlchemy metadata.
Index(
    "uq_jobs_one_current",
    (literal(1) + jobs.c.id.is_(None).cast(Integer)),
    unique=True,
    sqlite_where=jobs.c.status.in_(CURRENT_JOB_STATUSES),
)
Index(
    "uq_jobs_one_nonterminal_translation_per_chapter",
    jobs.c.chapter_id,
    unique=True,
    sqlite_where=and_(
        jobs.c.kind == "translation",
        jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
        jobs.c.chapter_id.is_not(None),
    ),
)
Index("ix_jobs_queue_claim", jobs.c.status, jobs.c.queue_rank)

queue_state = Table(
    "queue_state",
    metadata,
    Column("singleton_id", Integer, primary_key=True, server_default="1"),
    Column("queue_revision", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    CheckConstraint("singleton_id = 1", name="single_row"),
    CheckConstraint("queue_revision >= 1", name="revision_positive"),
)

job_items = Table(
    "job_items",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("page_id", String(UUID_LENGTH), ForeignKey("pages.id", ondelete="SET NULL")),
    Column("status", String(32), nullable=False, server_default="pending"),
    Column("input_fingerprint", String(HASH_LENGTH)),
    Column("result_json", Text),
    Column("error_json", Text),
    *_timestamps(),
    UniqueConstraint("job_id", "ordinal"),
    CheckConstraint("ordinal >= 1", name="ordinal_positive"),
    CheckConstraint(
        "status IN ('pending','running','completed','failed','skipped','cancelled')",
        name="status_values",
    ),
)

job_steps = Table(
    "job_steps",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("job_item_id", String(UUID_LENGTH), ForeignKey("job_items.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("kind", String(64), nullable=False),
    Column("status", String(32), nullable=False, server_default="pending"),
    Column("attempt_id", String(UUID_LENGTH)),
    Column("input_fingerprint", String(HASH_LENGTH)),
    Column("checkpoint_json", Text),
    Column("checkpoint_schema_version", Integer, nullable=False, server_default="1"),
    Column("error_json", Text),
    *_timestamps(),
    UniqueConstraint("job_item_id", "ordinal"),
    CheckConstraint("ordinal >= 1", name="ordinal_positive"),
    CheckConstraint(
        "status IN ('pending','running','completed','failed','skipped','cancelled')",
        name="status_values",
    ),
)

job_drain_acks = Table(
    "job_drain_acks",
    metadata,
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("attempt_id", String(UUID_LENGTH), primary_key=True),
    Column("pool_id", String(64), primary_key=True),
    Column("worker_slot", Integer, primary_key=True),
    Column("last_step_id", String(UUID_LENGTH), ForeignKey("job_steps.id", ondelete="SET NULL")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("worker_slot >= 0", name="worker_slot_nonnegative"),
)

job_events = Table(
    "job_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("event_type", String(64), nullable=False),
    Column("payload_json", Text, nullable=False, server_default="{}"),
    Column("payload_schema_version", Integer, nullable=False, server_default="1"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)
Index("ix_job_events_job_cursor", job_events.c.job_id, job_events.c.id)

job_config_snapshots = Table(
    "job_config_snapshots",
    metadata,
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("payload_json", Text, nullable=False),
    Column("schema_version", Integer, nullable=False),
)

job_credential_snapshots = Table(
    "job_credential_snapshots",
    metadata,
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "credential_version_id",
        String(UUID_LENGTH),
        ForeignKey("credential_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("role", String(64), primary_key=True),
)

job_plugin_snapshots = Table(
    "job_plugin_snapshots",
    metadata,
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "plugin_version_id",
        String(UUID_LENGTH),
        ForeignKey("plugin_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("config_json", Text, nullable=False, server_default="{}"),
)

job_font_snapshots = Table(
    "job_font_snapshots",
    metadata,
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("font_id", String(UUID_LENGTH), ForeignKey("fonts.id", ondelete="RESTRICT"), primary_key=True),
    Column("role", String(64), primary_key=True),
)

job_asset_inputs = Table(
    "job_asset_inputs",
    metadata,
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT"), primary_key=True),
    Column("role", String(64), primary_key=True),
    Column("binding_phase", String(16), nullable=False),
    Column("job_item_id", String(UUID_LENGTH), ForeignKey("job_items.id", ondelete="CASCADE")),
    CheckConstraint(
        "binding_phase IN ('create','item_start','checkpoint')",
        name="binding_phase_values",
    ),
)

job_step_asset_outputs = Table(
    "job_step_asset_outputs",
    metadata,
    Column("job_step_id", String(UUID_LENGTH), ForeignKey("job_steps.id", ondelete="CASCADE"), primary_key=True),
    Column("role", String(64), primary_key=True),
    Column("asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False),
)

job_artifacts = Table(
    "job_artifacts",
    metadata,
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("kind", String(64), primary_key=True),
    Column("asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False),
    Column("expires_at", DateTime(timezone=True)),
)

process_epochs = Table(
    "process_epochs",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("role", String(16), nullable=False),
    Column("token_hash", String(HASH_LENGTH), nullable=False),
    Column("pid", Integer, nullable=False),
    Column("status", String(16), nullable=False, server_default="active"),
    Column("heartbeat_at", DateTime(timezone=True), nullable=False),
    Column("lease_expires_at", DateTime(timezone=True), nullable=False),
    Column("recovery_completed_at", DateTime(timezone=True)),
    *_timestamps(),
    CheckConstraint("role IN ('launcher','api','worker')", name="role_values"),
    CheckConstraint("status IN ('active','lost','closed')", name="status_values"),
)
Index("ix_process_epochs_role_status", process_epochs.c.role, process_epochs.c.status)

worker_leases = Table(
    "worker_leases",
    metadata,
    Column("worker_epoch_id", String(UUID_LENGTH), ForeignKey("process_epochs.id", ondelete="CASCADE"), primary_key=True),
    Column("lease_token", String(200), nullable=False),
    Column("heartbeat_at", DateTime(timezone=True), nullable=False),
    Column("lease_expires_at", DateTime(timezone=True), nullable=False),
)

api_executor_leases = Table(
    "api_executor_leases",
    metadata,
    Column("api_epoch_id", String(UUID_LENGTH), ForeignKey("process_epochs.id", ondelete="CASCADE"), primary_key=True),
    Column("lease_token", String(200), nullable=False),
    Column("heartbeat_at", DateTime(timezone=True), nullable=False),
    Column("lease_expires_at", DateTime(timezone=True), nullable=False),
)

chapter_write_intents = Table(
    "chapter_write_intents",
    metadata,
    Column("chapter_id", String(UUID_LENGTH), ForeignKey("chapters.id", ondelete="CASCADE"), primary_key=True),
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("intent_set_id", String(UUID_LENGTH), nullable=False),
    Column("intent_generation", Integer, nullable=False),
    Column(
        "worker_epoch_id",
        String(UUID_LENGTH),
        ForeignKey("process_epochs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("lease_token", String(200), nullable=False),
    Column("lease_expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("intent_generation >= 1", name="generation_positive"),
)
Index(
    "ix_chapter_write_intents_job_set",
    chapter_write_intents.c.job_id,
    chapter_write_intents.c.intent_set_id,
)
Index(
    "ix_chapter_write_intents_epoch_expiry",
    chapter_write_intents.c.worker_epoch_id,
    chapter_write_intents.c.lease_expires_at,
)

chapter_write_locks = Table(
    "chapter_write_locks",
    metadata,
    Column("chapter_id", String(UUID_LENGTH), ForeignKey("chapters.id", ondelete="CASCADE"), primary_key=True),
    Column("job_id", String(UUID_LENGTH), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("lock_generation", Integer, nullable=False),
    Column("owner_attempt_id", String(UUID_LENGTH)),
    Column("lease_token", String(200)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint("lock_generation >= 1", name="generation_positive"),
)
Index("ix_chapter_write_locks_job", chapter_write_locks.c.job_id)

import_leases = Table(
    "import_leases",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("chapter_id", String(UUID_LENGTH), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False, unique=True),
    Column("owner_token_hash", String(HASH_LENGTH), nullable=False),
    Column("last_activity_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)
Index("ix_import_leases_expires_at", import_leases.c.expires_at)

jobs.c.blocked_by_import_lease_id.append_foreign_key(
    ForeignKey("import_leases.id", ondelete="SET NULL")
)

studio_documents = Table(
    "studio_documents",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("book_id", String(UUID_LENGTH), ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("title", String(500), nullable=False),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("generation", Integer, nullable=False, server_default="1"),
    Column("payload_json", Text, nullable=False),
    Column("schema_version", Integer, nullable=False, server_default="1"),
    *_timestamps(),
    CheckConstraint("revision >= 1", name="revision_positive"),
    CheckConstraint("generation >= 1", name="generation_positive"),
)

studio_chat_sessions = Table(
    "studio_chat_sessions",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column(
        "document_id",
        String(UUID_LENGTH),
        ForeignKey("studio_documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("title", String(500), nullable=False),
    Column("revision", Integer, nullable=False, server_default="1"),
    Column("generation", Integer, nullable=False, server_default="1"),
    Column("runtime_state_json", Text, nullable=False, server_default="{}"),
    Column("runtime_schema_version", Integer, nullable=False, server_default="1"),
    *_timestamps(),
)

studio_messages = Table(
    "studio_messages",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column(
        "session_id",
        String(UUID_LENGTH),
        ForeignKey("studio_chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("metadata_json", Text, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("session_id", "ordinal"),
    CheckConstraint("ordinal >= 1", name="ordinal_positive"),
    CheckConstraint("role IN ('system','user','assistant')", name="role_values"),
)

operations = Table(
    "operations",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("kind", String(64), nullable=False),
    Column("executor_role", String(16), nullable=False),
    Column("status", String(16), nullable=False, server_default="pending"),
    Column("page_id", String(UUID_LENGTH), ForeignKey("pages.id", ondelete="SET NULL")),
    Column("bubble_id", String(UUID_LENGTH), ForeignKey("bubbles.id", ondelete="SET NULL")),
    Column(
        "studio_document_id",
        String(UUID_LENGTH),
        ForeignKey("studio_documents.id", ondelete="SET NULL"),
    ),
    Column(
        "studio_session_id",
        String(UUID_LENGTH),
        ForeignKey("studio_chat_sessions.id", ondelete="SET NULL"),
    ),
    Column("base_revision", Integer),
    Column("base_generation", Integer),
    Column("request_json", Text, nullable=False),
    Column("request_schema_version", Integer, nullable=False, server_default="1"),
    Column("result_json", Text),
    Column("error_json", Text),
    Column("executor_epoch_id", String(UUID_LENGTH), ForeignKey("process_epochs.id", ondelete="SET NULL")),
    Column("attempt_id", String(UUID_LENGTH)),
    Column("lease_token", String(200)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    *_timestamps(),
    CheckConstraint(f"status IN ({_sql_values(OPERATION_STATUSES)})", name="status_values"),
    CheckConstraint(f"kind IN ({_sql_values(OPERATION_KINDS)})", name="kind_values"),
    CheckConstraint("executor_role IN ('api','worker')", name="executor_role_values"),
    CheckConstraint(
        "(kind IN ('bubble_ocr','bubble_color','page_detect') AND executor_role = 'worker') OR "
        "(kind IN ('bubble_translate','studio_generate','studio_chat','studio_summary') "
        "AND executor_role = 'api') OR "
        "(kind = 'page_repair')",
        name="kind_executor_shape",
    ),
    CheckConstraint(
        "(kind IN ('bubble_ocr','bubble_color','bubble_translate') "
        "AND page_id IS NOT NULL AND bubble_id IS NOT NULL "
        "AND studio_document_id IS NULL AND studio_session_id IS NULL) OR "
        "(kind IN ('page_detect','page_repair') "
        "AND page_id IS NOT NULL AND studio_document_id IS NULL AND studio_session_id IS NULL) OR "
        "(kind = 'studio_generate' AND studio_document_id IS NOT NULL "
        "AND page_id IS NULL AND bubble_id IS NULL AND studio_session_id IS NULL) OR "
        "(kind IN ('studio_chat','studio_summary') AND studio_session_id IS NOT NULL "
        "AND page_id IS NULL AND bubble_id IS NULL AND studio_document_id IS NULL)",
        name="kind_target_shape",
    ),
)
Index(
    "uq_operations_one_active_page_write",
    operations.c.page_id,
    unique=True,
    sqlite_where=and_(
        operations.c.page_id.is_not(None),
        operations.c.status.in_(("pending", "running")),
        operations.c.kind.in_(PAGE_WRITE_OPERATION_KINDS),
    ),
)
Index(
    "uq_operations_one_active_studio_generate",
    operations.c.studio_document_id,
    unique=True,
    sqlite_where=and_(
        operations.c.studio_document_id.is_not(None),
        operations.c.status.in_(("pending", "running")),
        operations.c.kind == "studio_generate",
    ),
)
Index(
    "uq_operations_one_active_studio_session",
    operations.c.studio_session_id,
    unique=True,
    sqlite_where=and_(
        operations.c.studio_session_id.is_not(None),
        operations.c.status.in_(("pending", "running")),
        operations.c.kind.in_(("studio_chat", "studio_summary")),
    ),
)
Index("ix_operations_executor_claim", operations.c.executor_role, operations.c.status, operations.c.created_at)

operation_credential_snapshots = Table(
    "operation_credential_snapshots",
    metadata,
    Column(
        "operation_id",
        String(UUID_LENGTH),
        ForeignKey("operations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "credential_version_id",
        String(UUID_LENGTH),
        ForeignKey("credential_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("role", String(64), primary_key=True),
)

operation_plugin_snapshots = Table(
    "operation_plugin_snapshots",
    metadata,
    Column(
        "operation_id",
        String(UUID_LENGTH),
        ForeignKey("operations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "plugin_version_id",
        String(UUID_LENGTH),
        ForeignKey("plugin_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("config_json", Text, nullable=False, server_default="{}"),
)

operation_font_snapshots = Table(
    "operation_font_snapshots",
    metadata,
    Column(
        "operation_id",
        String(UUID_LENGTH),
        ForeignKey("operations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("font_id", String(UUID_LENGTH), ForeignKey("fonts.id", ondelete="RESTRICT"), primary_key=True),
    Column("role", String(64), primary_key=True),
)

operation_asset_inputs = Table(
    "operation_asset_inputs",
    metadata,
    Column(
        "operation_id",
        String(UUID_LENGTH),
        ForeignKey("operations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("role", String(64), primary_key=True),
    Column("asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False),
)

operation_artifacts = Table(
    "operation_artifacts",
    metadata,
    Column(
        "operation_id",
        String(UUID_LENGTH),
        ForeignKey("operations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("kind", String(64), primary_key=True),
    Column("asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False),
    Column("page_id", String(UUID_LENGTH), ForeignKey("pages.id", ondelete="SET NULL")),
    Column("expires_at", DateTime(timezone=True)),
)

transient_requests = Table(
    "transient_requests",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("kind", String(64), nullable=False),
    Column("status", String(16), nullable=False, server_default="pending"),
    Column("connection_token_hash", String(HASH_LENGTH), nullable=False),
    Column("connection_open", Boolean, nullable=False, server_default="1"),
    Column("request_json", Text, nullable=False),
    Column("result_json", Text),
    Column("worker_epoch_id", String(UUID_LENGTH), ForeignKey("process_epochs.id", ondelete="SET NULL")),
    Column("attempt_id", String(UUID_LENGTH)),
    Column("lease_token", String(200)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("consumed_at", DateTime(timezone=True)),
    *_timestamps(),
    CheckConstraint("kind IN ('vector_query')", name="kind_values"),
    CheckConstraint(f"status IN ({_sql_values(OPERATION_STATUSES)})", name="status_values"),
)

render_requests = Table(
    "render_requests",
    metadata,
    Column("id", String(UUID_LENGTH), primary_key=True),
    Column("page_id", String(UUID_LENGTH), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, unique=True),
    Column("requested_revision", Integer, nullable=False),
    Column("rendering_revision", Integer),
    Column("completed_revision", Integer),
    Column("status", String(16), nullable=False, server_default="pending"),
    Column("executor_epoch_id", String(UUID_LENGTH), ForeignKey("process_epochs.id", ondelete="SET NULL")),
    Column("attempt_id", String(UUID_LENGTH)),
    Column("lease_token", String(200)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("error_json", Text),
    *_timestamps(),
    CheckConstraint("requested_revision >= 1", name="requested_revision_positive"),
    CheckConstraint("status IN ('pending','running','completed','failed')", name="status_values"),
)
Index("ix_render_requests_claim", render_requests.c.status, render_requests.c.updated_at)

page_assets = Table(
    "page_assets",
    metadata,
    Column("page_id", String(UUID_LENGTH), ForeignKey("pages.id", ondelete="CASCADE"), primary_key=True),
    Column("role", String(32), primary_key=True),
    Column("asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False),
    Column("input_source_revision", Integer),
    Column("input_document_revision", Integer),
    Column("parent_asset_id", String(UUID_LENGTH), ForeignKey("assets.id", ondelete="RESTRICT")),
    Column("producer_job_step_id", String(UUID_LENGTH), ForeignKey("job_steps.id", ondelete="SET NULL")),
    Column("producer_operation_id", String(UUID_LENGTH), ForeignKey("operations.id", ondelete="SET NULL")),
    Column(
        "producer_render_request_id",
        String(UUID_LENGTH),
        ForeignKey("render_requests.id", ondelete="SET NULL"),
    ),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint(f"role IN ({_sql_values(PAGE_ASSET_ROLES)})", name="role_values"),
    CheckConstraint(
        "(CASE WHEN producer_job_step_id IS NULL THEN 0 ELSE 1 END + "
        "CASE WHEN producer_operation_id IS NULL THEN 0 ELSE 1 END + "
        "CASE WHEN producer_render_request_id IS NULL THEN 0 ELSE 1 END) <= 1",
        name="single_producer",
    ),
    CheckConstraint(
        "(role IN ('thumbnail_source','thumbnail_translated') AND parent_asset_id IS NOT NULL) OR "
        "(role NOT IN ('thumbnail_source','thumbnail_translated'))",
        name="thumbnail_parent_required",
    ),
)

object_commit_journal = Table(
    "object_commit_journal",
    metadata,
    Column("asset_id", String(UUID_LENGTH), primary_key=True),
    Column("staging_relative_path", Text, nullable=False),
    Column("final_relative_path", Text, nullable=False),
    Column("state", String(24), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    CheckConstraint(
        "state IN ('staged','file_published','database_committed')",
        name="state_values",
    ),
)

provider_rate_limits = Table(
    "provider_rate_limits",
    metadata,
    Column("provider", String(64), primary_key=True),
    Column(
        "credential_version_id",
        String(UUID_LENGTH),
        ForeignKey("credential_versions.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("window_started_at", DateTime(timezone=True), nullable=False),
    Column("request_count", Integer, nullable=False, server_default="0"),
    Column("rpm_limit", Integer, nullable=False),
    Column("revision", Integer, nullable=False, server_default="1"),
    CheckConstraint("request_count >= 0", name="request_count_nonnegative"),
    CheckConstraint("rpm_limit >= 1", name="rpm_limit_positive"),
)

idempotency_records = Table(
    "idempotency_records",
    metadata,
    Column("scope", String(500), primary_key=True),
    Column("key", String(200), primary_key=True),
    Column("request_hash", String(HASH_LENGTH), nullable=False),
    Column("http_status", Integer, nullable=False),
    Column("response_json", Text, nullable=False),
    Column("resource_type", String(64)),
    Column("resource_id", String(UUID_LENGTH)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("http_status >= 200 AND http_status < 300", name="successful_status"),
)
Index("ix_idempotency_records_expires_at", idempotency_records.c.expires_at)
