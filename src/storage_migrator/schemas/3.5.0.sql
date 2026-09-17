-- Frozen 3.5.0 contract. Do not regenerate for later releases.

CREATE TABLE analysis_runs (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	job_id VARCHAR(36),
	scope VARCHAR(16) NOT NULL,
	status VARCHAR(32) DEFAULT 'staging' NOT NULL,
	config_json TEXT NOT NULL,
	missing_page_ids_json TEXT DEFAULT '[]' NOT NULL,
	target_count INTEGER DEFAULT '0' NOT NULL,
	success_count INTEGER DEFAULT '0' NOT NULL,
	failed_count INTEGER DEFAULT '0' NOT NULL,
	published_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_analysis_runs PRIMARY KEY (id),
	CONSTRAINT ck_analysis_runs_scope_values CHECK (scope IN ('full','incremental','chapter','page')),
	CONSTRAINT ck_analysis_runs_status_values CHECK (status IN ('staging','completed','completed_with_errors','failed','cancelled')),
	CONSTRAINT ck_analysis_runs_counts_nonnegative CHECK (target_count >= 0 AND success_count >= 0 AND failed_count >= 0),
	CONSTRAINT fk_analysis_runs_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT uq_analysis_runs_job_id UNIQUE (job_id),
	CONSTRAINT fk_analysis_runs_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE SET NULL
)

;


CREATE TABLE app_settings (
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	domain VARCHAR(64) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	payload_json TEXT NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_app_settings PRIMARY KEY (owner_user_id, domain),
	CONSTRAINT ck_app_settings_revision_positive CHECK (revision >= 1)
)

;


CREATE TABLE assets (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	relative_path TEXT NOT NULL,
	mime_type VARCHAR(127) NOT NULL,
	checksum VARCHAR(64) NOT NULL,
	byte_size BIGINT NOT NULL,
	width INTEGER,
	height INTEGER,
	integrity_status VARCHAR(16) DEFAULT 'ok' NOT NULL,
	gc_marked_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_assets PRIMARY KEY (id),
	CONSTRAINT ck_assets_byte_size_nonnegative CHECK (byte_size >= 0),
	CONSTRAINT ck_assets_dimensions_shape CHECK ((width IS NULL AND height IS NULL) OR (width >= 1 AND height >= 1)),
	CONSTRAINT ck_assets_integrity_status_values CHECK (integrity_status IN ('ok', 'missing')),
	CONSTRAINT uq_assets_relative_path UNIQUE (relative_path)
)

;


CREATE TABLE continuation_projects (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	source_run_id VARCHAR(36),
	revision INTEGER DEFAULT '1' NOT NULL,
	payload_json TEXT DEFAULT '{}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_continuation_projects PRIMARY KEY (id),
	CONSTRAINT ck_continuation_projects_revision_positive CHECK (revision >= 1),
	CONSTRAINT uq_continuation_projects_book_id UNIQUE (book_id),
	CONSTRAINT fk_continuation_projects_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_continuation_projects_source_run_id_analysis_runs FOREIGN KEY(source_run_id) REFERENCES analysis_runs (id) ON DELETE RESTRICT
)

;


CREATE TABLE credentials (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	domain VARCHAR(64) NOT NULL,
	provider VARCHAR(64) NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_credentials PRIMARY KEY (id),
	CONSTRAINT uq_credentials_owner_user_id UNIQUE (owner_user_id, domain, provider)
)

;


CREATE TABLE idempotency_records (
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	scope VARCHAR(500) NOT NULL,
	"key" VARCHAR(200) NOT NULL,
	request_hash VARCHAR(64) NOT NULL,
	http_status INTEGER NOT NULL,
	response_json TEXT NOT NULL,
	resource_type VARCHAR(64),
	resource_id VARCHAR(36),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	expires_at DATETIME NOT NULL,
	CONSTRAINT pk_idempotency_records PRIMARY KEY (owner_user_id, scope, "key"),
	CONSTRAINT ck_idempotency_records_successful_status CHECK (http_status >= 200 AND http_status < 300)
)

;


CREATE TABLE job_batches (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	display_name VARCHAR(500) NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_job_batches PRIMARY KEY (id)
)

;


CREATE TABLE jobs (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	batch_id VARCHAR(36),
	kind VARCHAR(64) NOT NULL,
	retry_of_job_id VARCHAR(36),
	retry_mode VARCHAR(16),
	status VARCHAR(32) NOT NULL,
	queue_rank INTEGER,
	book_id VARCHAR(36),
	chapter_id VARCHAR(36),
	page_id VARCHAR(36),
	analysis_run_id VARCHAR(36),
	continuation_project_id VARCHAR(36),
	web_import_draft_id VARCHAR(36),
	blocked_by_job_id VARCHAR(36),
	attempt_id VARCHAR(36),
	worker_epoch_id VARCHAR(36),
	config_json TEXT NOT NULL,
	latest_progress_json TEXT NOT NULL,
	target_display_json TEXT DEFAULT '{}' NOT NULL,
	started_at DATETIME,
	finished_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_jobs PRIMARY KEY (id),
	CONSTRAINT ck_jobs_status_values CHECK (status IN ('queued', 'running', 'paused', 'cancelled', 'completed', 'completed_with_errors', 'failed', 'interrupted')),
	CONSTRAINT ck_jobs_kind_values CHECK (kind IN ('translation', 'remove_text', 'detect', 'style_apply', 'text_import', 'container_import', 'web_extract', 'web_import_commit', 'export', 'insight_analysis', 'insight_export', 'vector_rebuild', 'continuation', 'derived_rebuild', 'plugin_agent')),
	CONSTRAINT ck_jobs_retry_mode_values CHECK (retry_mode IS NULL OR retry_mode IN ('current','original')),
	CONSTRAINT ck_jobs_retry_lineage_complete CHECK ((retry_of_job_id IS NULL AND retry_mode IS NULL) OR (retry_of_job_id IS NOT NULL AND retry_mode IS NOT NULL)),
	CONSTRAINT ck_jobs_queue_rank_positive CHECK (queue_rank IS NULL OR queue_rank >= 1),
	CONSTRAINT fk_jobs_batch_id_job_batches FOREIGN KEY(batch_id) REFERENCES job_batches (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_retry_of_job_id_jobs FOREIGN KEY(retry_of_job_id) REFERENCES jobs (id) ON DELETE RESTRICT,
	CONSTRAINT uq_jobs_queue_rank UNIQUE (queue_rank),
	CONSTRAINT fk_jobs_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_analysis_run_id_analysis_runs FOREIGN KEY(analysis_run_id) REFERENCES analysis_runs (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_continuation_project_id_continuation_projects FOREIGN KEY(continuation_project_id) REFERENCES continuation_projects (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_web_import_draft_id_web_import_drafts FOREIGN KEY(web_import_draft_id) REFERENCES web_import_drafts (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_blocked_by_job_id_jobs FOREIGN KEY(blocked_by_job_id) REFERENCES jobs (id) ON DELETE SET NULL,
	CONSTRAINT fk_jobs_worker_epoch_id_process_epochs FOREIGN KEY(worker_epoch_id) REFERENCES process_epochs (id) ON DELETE SET NULL
)

;


CREATE TABLE object_commit_journal (
	asset_id VARCHAR(36) NOT NULL,
	staging_relative_path TEXT NOT NULL,
	final_relative_path TEXT NOT NULL,
	state VARCHAR(24) NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_object_commit_journal PRIMARY KEY (asset_id),
	CONSTRAINT ck_object_commit_journal_state_values CHECK (state IN ('staged','file_published'))
)

;


CREATE TABLE platform_config (
	singleton_id INTEGER DEFAULT '1' NOT NULL,
	registration_requires_invite BOOLEAN DEFAULT '1' NOT NULL,
	asset_quota_bytes BIGINT DEFAULT '2147483648' NOT NULL,
	public_user_policy_json TEXT DEFAULT '{"features":{"characterStudio":true,"editMode":true,"insight":true,"translation":true},"models":{"aux_ysg_yolo":true,"detector_ctd":true,"detector_default":true,"detector_yolo":true,"lama_mpe":true,"litelama":true,"lama_manga":true,"manga_ocr":true,"ocr_48px":true,"paddle_ocr":true,"paddleocr_vl":true,"saber_yolo":true},"settings":{"lamaDisableResize":{"editable":false,"value":false},"parallel":{"allowed":false}}}' NOT NULL,
	scheduler_policy_json TEXT DEFAULT '{"apiOperationConcurrency":2,"interactiveBurst":1,"maxDeepLearningConcurrency":1,"minAvailableMemoryMiB":2048,"modelIdleSeconds":180,"pageQuantum":1,"queueDiscipline":"owner_round_robin"}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_platform_config PRIMARY KEY (singleton_id),
	CONSTRAINT ck_platform_config_single_row CHECK (singleton_id = 1),
	CONSTRAINT ck_platform_config_asset_quota_positive CHECK (asset_quota_bytes > 0)
)

;


CREATE TABLE plugins (
	id VARCHAR(100) NOT NULL,
	name VARCHAR(200) NOT NULL,
	state VARCHAR(16) DEFAULT 'enabled' NOT NULL,
	author VARCHAR(200) DEFAULT '' NOT NULL,
	description TEXT DEFAULT '' NOT NULL,
	default_enabled BOOLEAN DEFAULT '0' NOT NULL,
	runtime_enabled BOOLEAN DEFAULT '0' NOT NULL,
	config_json TEXT DEFAULT '{}' NOT NULL,
	config_revision INTEGER DEFAULT '1' NOT NULL,
	error_message TEXT,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_plugins PRIMARY KEY (id),
	CONSTRAINT ck_plugins_state_values CHECK (state IN ('enabled', 'disabled', 'error')),
	CONSTRAINT ck_plugins_config_revision_positive CHECK (config_revision >= 1)
)

;


CREATE TABLE process_epochs (
	id VARCHAR(36) NOT NULL,
	role VARCHAR(16) NOT NULL,
	token_hash VARCHAR(64) NOT NULL,
	pid INTEGER NOT NULL,
	status VARCHAR(16) DEFAULT 'active' NOT NULL,
	heartbeat_at DATETIME NOT NULL,
	lease_expires_at DATETIME NOT NULL,
	recovery_completed_at DATETIME,
	model_release_request_id VARCHAR(36),
	model_release_handled_id VARCHAR(36),
	model_release_result_json TEXT,
	model_release_error_json TEXT,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_process_epochs PRIMARY KEY (id),
	CONSTRAINT ck_process_epochs_role_values CHECK (role IN ('launcher','api','worker')),
	CONSTRAINT ck_process_epochs_status_values CHECK (status IN ('active','lost','closed')),
	CONSTRAINT ck_process_epochs_pid_nonnegative CHECK (pid >= 0)
)

;


CREATE TABLE prompts (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	type VARCHAR(32) NOT NULL,
	name VARCHAR(200) NOT NULL,
	content TEXT NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	is_factory_default BOOLEAN DEFAULT '0' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_prompts PRIMARY KEY (id),
	CONSTRAINT uq_prompts_owner_user_id UNIQUE (owner_user_id, type, name),
	CONSTRAINT ck_prompts_type_values CHECK (type IN ('translate', 'textbox', 'ai_vision_ocr', 'hq_translate', 'proofreading', 'batch_analysis', 'segment_summary', 'chapter_summary', 'book_overview', 'group_summary', 'qa_response', 'question_decompose', 'analysis_system')),
	CONSTRAINT ck_prompts_revision_positive CHECK (revision >= 1)
)

;


CREATE TABLE queue_state (
	singleton_id INTEGER DEFAULT '1' NOT NULL,
	admission_paused BOOLEAN DEFAULT '0' NOT NULL,
	CONSTRAINT pk_queue_state PRIMARY KEY (singleton_id),
	CONSTRAINT ck_queue_state_single_row CHECK (singleton_id = 1)
)

;


CREATE TABLE schema_metadata (
	singleton_id INTEGER DEFAULT '1' NOT NULL,
	runtime_profile VARCHAR(16) NOT NULL,
	storage_version VARCHAR(32) NOT NULL,
	CONSTRAINT pk_schema_metadata PRIMARY KEY (singleton_id),
	CONSTRAINT ck_schema_metadata_single_row CHECK (singleton_id = 1),
	CONSTRAINT ck_schema_metadata_runtime_profile_values CHECK (runtime_profile IN ('local','public'))
)

;


CREATE TABLE studio_chat_sessions (
	id VARCHAR(36) NOT NULL,
	document_id VARCHAR(36) NOT NULL,
	title VARCHAR(500) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	generation INTEGER DEFAULT '1' NOT NULL,
	greeting_source_json TEXT DEFAULT '{}' NOT NULL,
	variables_json TEXT DEFAULT '{}' NOT NULL,
	summary_blocks_json TEXT DEFAULT '[]' NOT NULL,
	summary_through_message_id VARCHAR(36),
	summary_generation INTEGER DEFAULT '0' NOT NULL,
	runtime_state_json TEXT DEFAULT '{}' NOT NULL,
	archived_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_studio_chat_sessions PRIMARY KEY (id),
	CONSTRAINT ck_studio_chat_sessions_revision_positive CHECK (revision >= 1),
	CONSTRAINT ck_studio_chat_sessions_generation_positive CHECK (generation >= 1),
	CONSTRAINT ck_studio_chat_sessions_summary_generation_nonnegative CHECK (summary_generation >= 0),
	CONSTRAINT fk_studio_chat_sessions_document_id_studio_documents FOREIGN KEY(document_id) REFERENCES studio_documents (id) ON DELETE CASCADE,
	CONSTRAINT fk_studio_chat_sessions_summary_through_message_id_studio_messages FOREIGN KEY(summary_through_message_id) REFERENCES studio_messages (id) ON DELETE SET NULL
)

;


CREATE TABLE studio_messages (
	id VARCHAR(36) NOT NULL,
	session_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	role VARCHAR(16) NOT NULL,
	content TEXT NOT NULL,
	runtime_log TEXT DEFAULT '[]' NOT NULL,
	variables_snapshot_json TEXT DEFAULT '{}' NOT NULL,
	generation_meta_json TEXT DEFAULT '{}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_studio_messages PRIMARY KEY (id),
	CONSTRAINT uq_studio_messages_session_id UNIQUE (session_id, ordinal),
	CONSTRAINT ck_studio_messages_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_studio_messages_role_values CHECK (role IN ('system','user','assistant')),
	CONSTRAINT fk_studio_messages_session_id_studio_chat_sessions FOREIGN KEY(session_id) REFERENCES studio_chat_sessions (id) ON DELETE CASCADE
)

;


CREATE TABLE tags (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	name VARCHAR(200) NOT NULL,
	color VARCHAR(16) NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_tags PRIMARY KEY (id),
	CONSTRAINT uq_tags_owner_user_id UNIQUE (owner_user_id, name)
)

;


CREATE TABLE users (
	id VARCHAR(36) NOT NULL,
	username VARCHAR(32) NOT NULL,
	password_hash TEXT,
	role VARCHAR(16) DEFAULT 'user' NOT NULL,
	status VARCHAR(16) DEFAULT 'active' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_users PRIMARY KEY (id),
	CONSTRAINT ck_users_role_values CHECK (role IN ('user','admin')),
	CONSTRAINT ck_users_status_values CHECK (status IN ('active','disabled')),
	CONSTRAINT uq_users_username UNIQUE (username)
)

;


CREATE TABLE books (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	kind VARCHAR(24) DEFAULT 'library' NOT NULL,
	title VARCHAR(500) NOT NULL,
	chapter_order_revision INTEGER DEFAULT '1' NOT NULL,
	cover_asset_id VARCHAR(36),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_books PRIMARY KEY (id),
	CONSTRAINT ck_books_kind_values CHECK (kind IN ('library', 'quick_workspace', 'browser_session')),
	CONSTRAINT ck_books_chapter_order_revision_positive CHECK (chapter_order_revision >= 1),
	CONSTRAINT fk_books_cover_asset_id_assets FOREIGN KEY(cover_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE continuation_characters (
	id VARCHAR(36) NOT NULL,
	project_id VARCHAR(36) NOT NULL,
	name VARCHAR(500) NOT NULL,
	aliases_json TEXT DEFAULT '[]' NOT NULL,
	enabled BOOLEAN DEFAULT '1' NOT NULL,
	payload_json TEXT DEFAULT '{}' NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_continuation_characters PRIMARY KEY (id),
	CONSTRAINT uq_continuation_characters_project_id UNIQUE (project_id, name),
	CONSTRAINT ck_continuation_characters_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_continuation_characters_project_id_continuation_projects FOREIGN KEY(project_id) REFERENCES continuation_projects (id) ON DELETE CASCADE
)

;


CREATE TABLE continuation_pages (
	id VARCHAR(36) NOT NULL,
	project_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	payload_json TEXT DEFAULT '{}' NOT NULL,
	CONSTRAINT pk_continuation_pages PRIMARY KEY (id),
	CONSTRAINT uq_continuation_pages_project_id UNIQUE (project_id, ordinal),
	CONSTRAINT ck_continuation_pages_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_continuation_pages_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_continuation_pages_project_id_continuation_projects FOREIGN KEY(project_id) REFERENCES continuation_projects (id) ON DELETE CASCADE
)

;


CREATE TABLE continuation_project_reference_assets (
	project_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	CONSTRAINT pk_continuation_project_reference_assets PRIMARY KEY (project_id, ordinal),
	CONSTRAINT uq_continuation_project_reference_assets_project_id UNIQUE (project_id, asset_id),
	CONSTRAINT ck_continuation_project_reference_assets_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT fk_continuation_project_reference_assets_project_id_continuation_projects FOREIGN KEY(project_id) REFERENCES continuation_projects (id) ON DELETE CASCADE,
	CONSTRAINT fk_continuation_project_reference_assets_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE continuation_scripts (
	id VARCHAR(36) NOT NULL,
	project_id VARCHAR(36) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	content TEXT DEFAULT '' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_continuation_scripts PRIMARY KEY (id),
	CONSTRAINT ck_continuation_scripts_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_continuation_scripts_project_id_continuation_projects FOREIGN KEY(project_id) REFERENCES continuation_projects (id) ON DELETE CASCADE
)

;


CREATE TABLE credential_versions (
	id VARCHAR(36) NOT NULL,
	credential_id VARCHAR(36) NOT NULL,
	version INTEGER NOT NULL,
	secret_json TEXT NOT NULL,
	key_fingerprint VARCHAR(64) NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_credential_versions PRIMARY KEY (id),
	CONSTRAINT uq_credential_versions_credential_id UNIQUE (credential_id, version),
	CONSTRAINT ck_credential_versions_version_positive CHECK (version >= 1),
	CONSTRAINT fk_credential_versions_credential_id_credentials FOREIGN KEY(credential_id) REFERENCES credentials (id) ON DELETE CASCADE
)

;


CREATE TABLE fonts (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	kind VARCHAR(16) NOT NULL,
	display_name VARCHAR(200) NOT NULL,
	asset_id VARCHAR(36),
	builtin_key VARCHAR(200),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_fonts PRIMARY KEY (id),
	CONSTRAINT ck_fonts_kind_values CHECK (kind IN ('builtin', 'uploaded')),
	CONSTRAINT ck_fonts_source_shape CHECK ((kind = 'builtin' AND builtin_key IS NOT NULL AND asset_id IS NULL) OR (kind = 'uploaded' AND builtin_key IS NULL AND asset_id IS NOT NULL)),
	CONSTRAINT fk_fonts_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE invite_codes (
	code_hash VARCHAR(64) NOT NULL,
	created_by_user_id VARCHAR(36) NOT NULL,
	used_by_user_id VARCHAR(36),
	expires_at DATETIME NOT NULL,
	used_at DATETIME,
	revoked_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_invite_codes PRIMARY KEY (code_hash),
	CONSTRAINT fk_invite_codes_created_by_user_id_users FOREIGN KEY(created_by_user_id) REFERENCES users (id) ON DELETE CASCADE,
	CONSTRAINT fk_invite_codes_used_by_user_id_users FOREIGN KEY(used_by_user_id) REFERENCES users (id) ON DELETE SET NULL
)

;


CREATE TABLE job_artifacts (
	job_id VARCHAR(36) NOT NULL,
	kind VARCHAR(64) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	expires_at DATETIME,
	CONSTRAINT pk_job_artifacts PRIMARY KEY (job_id, kind),
	CONSTRAINT fk_job_artifacts_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE,
	CONSTRAINT fk_job_artifacts_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE job_events (
	id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	job_id VARCHAR(36) NOT NULL,
	event_type VARCHAR(64) NOT NULL,
	payload_json TEXT DEFAULT '{}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT fk_job_events_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE
)

;


CREATE TABLE plugin_versions (
	id VARCHAR(36) NOT NULL,
	plugin_id VARCHAR(100) NOT NULL,
	version VARCHAR(64) NOT NULL,
	package_relative_path TEXT NOT NULL,
	checksum VARCHAR(64) NOT NULL,
	manifest_json TEXT NOT NULL,
	config_schema_json TEXT DEFAULT '{}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_plugin_versions PRIMARY KEY (id),
	CONSTRAINT fk_plugin_versions_plugin_id_plugins FOREIGN KEY(plugin_id) REFERENCES plugins (id) ON DELETE CASCADE,
	CONSTRAINT uq_plugin_versions_package_relative_path UNIQUE (package_relative_path)
)

;


CREATE TABLE recovery_codes (
	id VARCHAR(36) NOT NULL,
	user_id VARCHAR(36) NOT NULL,
	code_hash VARCHAR(64) NOT NULL,
	used_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_recovery_codes PRIMARY KEY (id),
	CONSTRAINT fk_recovery_codes_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE,
	CONSTRAINT uq_recovery_codes_code_hash UNIQUE (code_hash)
)

;


CREATE TABLE sessions (
	token_hash VARCHAR(64) NOT NULL,
	user_id VARCHAR(36) NOT NULL,
	csrf_token_hash VARCHAR(64) NOT NULL,
	expires_at DATETIME NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_sessions PRIMARY KEY (token_hash),
	CONSTRAINT fk_sessions_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
)

;


CREATE TABLE studio_message_assets (
	message_id VARCHAR(36) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	CONSTRAINT pk_studio_message_assets PRIMARY KEY (message_id, asset_id),
	CONSTRAINT uq_studio_message_assets_message_id UNIQUE (message_id, ordinal),
	CONSTRAINT ck_studio_message_assets_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT fk_studio_message_assets_message_id_studio_messages FOREIGN KEY(message_id) REFERENCES studio_messages (id) ON DELETE CASCADE,
	CONSTRAINT fk_studio_message_assets_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE analysis_artifacts (
	id VARCHAR(36) NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	run_id VARCHAR(36),
	kind VARCHAR(32) NOT NULL,
	template VARCHAR(64) DEFAULT 'default' NOT NULL,
	status VARCHAR(16) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	is_active BOOLEAN DEFAULT '0' NOT NULL,
	dependency_fingerprint VARCHAR(64) NOT NULL,
	payload_json TEXT DEFAULT '{}' NOT NULL,
	asset_id VARCHAR(36),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_analysis_artifacts PRIMARY KEY (id),
	CONSTRAINT ck_analysis_artifacts_status_values CHECK (status IN ('ready','stale','building','failed','degraded')),
	CONSTRAINT ck_analysis_artifacts_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_analysis_artifacts_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_analysis_artifacts_run_id_analysis_runs FOREIGN KEY(run_id) REFERENCES analysis_runs (id) ON DELETE SET NULL,
	CONSTRAINT fk_analysis_artifacts_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE book_settings (
	book_id VARCHAR(36) NOT NULL,
	domain VARCHAR(64) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	payload_json TEXT NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_book_settings PRIMARY KEY (book_id, domain),
	CONSTRAINT ck_book_settings_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_book_settings_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE
)

;


CREATE TABLE book_tags (
	book_id VARCHAR(36) NOT NULL,
	tag_id VARCHAR(36) NOT NULL,
	CONSTRAINT pk_book_tags PRIMARY KEY (book_id, tag_id),
	CONSTRAINT fk_book_tags_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_book_tags_tag_id_tags FOREIGN KEY(tag_id) REFERENCES tags (id) ON DELETE CASCADE
)

;


CREATE TABLE chapters (
	id VARCHAR(36) NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	title VARCHAR(500) NOT NULL,
	page_order_revision INTEGER DEFAULT '1' NOT NULL,
	settings_memory_json TEXT DEFAULT '{}' NOT NULL,
	settings_memory_revision INTEGER DEFAULT '1' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_chapters PRIMARY KEY (id),
	CONSTRAINT uq_chapters_book_id UNIQUE (book_id, ordinal),
	CONSTRAINT ck_chapters_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_chapters_page_order_revision_positive CHECK (page_order_revision >= 1),
	CONSTRAINT ck_chapters_settings_memory_revision_positive CHECK (settings_memory_revision >= 1),
	CONSTRAINT fk_chapters_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE
)

;


CREATE TABLE continuation_character_forms (
	id VARCHAR(36) NOT NULL,
	character_id VARCHAR(36) NOT NULL,
	name VARCHAR(500) NOT NULL,
	reference_asset_id VARCHAR(36),
	reference_thumbnail_asset_id VARCHAR(36),
	adopted_asset_id VARCHAR(36),
	payload_json TEXT DEFAULT '{}' NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_continuation_character_forms PRIMARY KEY (id),
	CONSTRAINT uq_continuation_character_forms_character_id UNIQUE (character_id, name),
	CONSTRAINT ck_continuation_character_forms_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_continuation_character_forms_character_id_continuation_characters FOREIGN KEY(character_id) REFERENCES continuation_characters (id) ON DELETE CASCADE,
	CONSTRAINT fk_continuation_character_forms_reference_asset_id_assets FOREIGN KEY(reference_asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_continuation_character_forms_reference_thumbnail_asset_id_assets FOREIGN KEY(reference_thumbnail_asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_continuation_character_forms_adopted_asset_id_assets FOREIGN KEY(adopted_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE continuation_image_versions (
	id VARCHAR(36) NOT NULL,
	continuation_page_id VARCHAR(36) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	thumbnail_asset_id VARCHAR(36) NOT NULL,
	version INTEGER NOT NULL,
	is_active BOOLEAN DEFAULT '0' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_continuation_image_versions PRIMARY KEY (id),
	CONSTRAINT uq_continuation_image_versions_continuation_page_id UNIQUE (continuation_page_id, version),
	CONSTRAINT ck_continuation_image_versions_version_positive CHECK (version >= 1),
	CONSTRAINT fk_continuation_image_versions_continuation_page_id_continuation_pages FOREIGN KEY(continuation_page_id) REFERENCES continuation_pages (id) ON DELETE CASCADE,
	CONSTRAINT fk_continuation_image_versions_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_continuation_image_versions_thumbnail_asset_id_assets FOREIGN KEY(thumbnail_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE credential_current_versions (
	credential_id VARCHAR(36) NOT NULL,
	credential_version_id VARCHAR(36) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_credential_current_versions PRIMARY KEY (credential_id),
	CONSTRAINT ck_credential_current_versions_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_credential_current_versions_credential_id_credentials FOREIGN KEY(credential_id) REFERENCES credentials (id) ON DELETE CASCADE,
	CONSTRAINT uq_credential_current_versions_credential_version_id UNIQUE (credential_version_id),
	CONSTRAINT fk_credential_current_versions_credential_version_id_credential_versions FOREIGN KEY(credential_version_id) REFERENCES credential_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE job_credential_snapshots (
	job_id VARCHAR(36) NOT NULL,
	credential_version_id VARCHAR(36) NOT NULL,
	role VARCHAR(64) NOT NULL,
	CONSTRAINT pk_job_credential_snapshots PRIMARY KEY (job_id, credential_version_id, role),
	CONSTRAINT fk_job_credential_snapshots_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE,
	CONSTRAINT fk_job_credential_snapshots_credential_version_id_credential_versions FOREIGN KEY(credential_version_id) REFERENCES credential_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE job_font_snapshots (
	job_id VARCHAR(36) NOT NULL,
	font_id VARCHAR(36) NOT NULL,
	role VARCHAR(64) NOT NULL,
	CONSTRAINT pk_job_font_snapshots PRIMARY KEY (job_id, font_id, role),
	CONSTRAINT fk_job_font_snapshots_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE,
	CONSTRAINT fk_job_font_snapshots_font_id_fonts FOREIGN KEY(font_id) REFERENCES fonts (id) ON DELETE RESTRICT
)

;


CREATE TABLE job_plugin_snapshots (
	job_id VARCHAR(36) NOT NULL,
	plugin_version_id VARCHAR(36) NOT NULL,
	config_json TEXT DEFAULT '{}' NOT NULL,
	CONSTRAINT pk_job_plugin_snapshots PRIMARY KEY (job_id, plugin_version_id),
	CONSTRAINT fk_job_plugin_snapshots_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE,
	CONSTRAINT fk_job_plugin_snapshots_plugin_version_id_plugin_versions FOREIGN KEY(plugin_version_id) REFERENCES plugin_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE notes (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	title VARCHAR(500) NOT NULL,
	content TEXT NOT NULL,
	kind VARCHAR(16) DEFAULT 'text' NOT NULL,
	tags_json TEXT DEFAULT '[]' NOT NULL,
	comments_json TEXT DEFAULT '[]' NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_notes PRIMARY KEY (id),
	CONSTRAINT ck_notes_revision_positive CHECK (revision >= 1),
	CONSTRAINT ck_notes_kind_values CHECK (kind IN ('text','qa')),
	CONSTRAINT fk_notes_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE
)

;


CREATE TABLE plugin_current_versions (
	plugin_id VARCHAR(100) NOT NULL,
	plugin_version_id VARCHAR(36) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_plugin_current_versions PRIMARY KEY (plugin_id),
	CONSTRAINT ck_plugin_current_versions_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_plugin_current_versions_plugin_id_plugins FOREIGN KEY(plugin_id) REFERENCES plugins (id) ON DELETE CASCADE,
	CONSTRAINT uq_plugin_current_versions_plugin_version_id UNIQUE (plugin_version_id),
	CONSTRAINT fk_plugin_current_versions_plugin_version_id_plugin_versions FOREIGN KEY(plugin_version_id) REFERENCES plugin_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE provider_rate_limits (
	provider VARCHAR(64) NOT NULL,
	credential_version_id VARCHAR(36) NOT NULL,
	window_started_at DATETIME NOT NULL,
	request_count INTEGER DEFAULT '0' NOT NULL,
	rpm_limit INTEGER NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	CONSTRAINT pk_provider_rate_limits PRIMARY KEY (provider, credential_version_id),
	CONSTRAINT ck_provider_rate_limits_request_count_nonnegative CHECK (request_count >= 0),
	CONSTRAINT ck_provider_rate_limits_rpm_limit_positive CHECK (rpm_limit >= 1),
	CONSTRAINT ck_provider_rate_limits_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_provider_rate_limits_credential_version_id_credential_versions FOREIGN KEY(credential_version_id) REFERENCES credential_versions (id) ON DELETE CASCADE
)

;


CREATE TABLE provider_settings (
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	domain VARCHAR(64) NOT NULL,
	provider VARCHAR(64) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	payload_json TEXT NOT NULL,
	credential_version_id VARCHAR(36),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_provider_settings PRIMARY KEY (owner_user_id, domain, provider),
	CONSTRAINT ck_provider_settings_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_provider_settings_credential_version_id_credential_versions FOREIGN KEY(credential_version_id) REFERENCES credential_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE studio_documents (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	origin_type VARCHAR(32) NOT NULL,
	source_character VARCHAR(500),
	title VARCHAR(500) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	chat_index_revision INTEGER DEFAULT '1' NOT NULL,
	avatar_asset_id VARCHAR(36),
	tags_json TEXT DEFAULT '[]' NOT NULL,
	is_favorite BOOLEAN DEFAULT '0' NOT NULL,
	identity_json TEXT DEFAULT '{}' NOT NULL,
	core_messages_json TEXT DEFAULT '{}' NOT NULL,
	lorebook_json TEXT DEFAULT '{}' NOT NULL,
	regex_scripts_json TEXT DEFAULT '[]' NOT NULL,
	state_tasks_json TEXT DEFAULT '[]' NOT NULL,
	frozen_sections_json TEXT DEFAULT '[]' NOT NULL,
	last_review_json TEXT,
	last_diagnostics_json TEXT,
	last_validated_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_studio_documents PRIMARY KEY (id),
	CONSTRAINT ck_studio_documents_revision_positive CHECK (revision >= 1),
	CONSTRAINT ck_studio_documents_chat_index_revision_positive CHECK (chat_index_revision >= 1),
	CONSTRAINT ck_studio_documents_origin_type_values CHECK (origin_type IN ('analysis','manual','imported')),
	CONSTRAINT fk_studio_documents_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_studio_documents_avatar_asset_id_assets FOREIGN KEY(avatar_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE timeline_versions (
	id VARCHAR(36) NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	run_id VARCHAR(36),
	mode VARCHAR(16) NOT NULL,
	status VARCHAR(16) NOT NULL,
	content_json TEXT DEFAULT '{}' NOT NULL,
	dependency_fingerprint VARCHAR(64) NOT NULL,
	is_active BOOLEAN DEFAULT '0' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_timeline_versions PRIMARY KEY (id),
	CONSTRAINT ck_timeline_versions_mode_values CHECK (mode IN ('enhanced','compressed','simple')),
	CONSTRAINT ck_timeline_versions_status_values CHECK (status IN ('ready','stale','building','failed','degraded')),
	CONSTRAINT fk_timeline_versions_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_timeline_versions_run_id_analysis_runs FOREIGN KEY(run_id) REFERENCES analysis_runs (id) ON DELETE SET NULL
)

;


CREATE TABLE transient_requests (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	kind VARCHAR(64) NOT NULL,
	book_id VARCHAR(36),
	status VARCHAR(16) DEFAULT 'pending' NOT NULL,
	connection_token_hash VARCHAR(64) NOT NULL,
	connection_open BOOLEAN DEFAULT '1' NOT NULL,
	request_json TEXT NOT NULL,
	result_json TEXT,
	worker_epoch_id VARCHAR(36),
	attempt_id VARCHAR(36),
	completed_at DATETIME,
	consumed_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_transient_requests PRIMARY KEY (id),
	CONSTRAINT ck_transient_requests_kind_values CHECK (kind IN ('vector_query')),
	CONSTRAINT ck_transient_requests_status_values CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
	CONSTRAINT fk_transient_requests_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_transient_requests_worker_epoch_id_process_epochs FOREIGN KEY(worker_epoch_id) REFERENCES process_epochs (id) ON DELETE SET NULL
)

;


CREATE TABLE translation_constraints (
	book_id VARCHAR(36) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	payload_json TEXT DEFAULT '{}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_translation_constraints PRIMARY KEY (book_id),
	CONSTRAINT ck_translation_constraints_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_translation_constraints_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE
)

;


CREATE TABLE vector_generations (
	id VARCHAR(36) NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	run_id VARCHAR(36),
	generation INTEGER NOT NULL,
	status VARCHAR(16) NOT NULL,
	dependency_fingerprint VARCHAR(64) NOT NULL,
	page_count INTEGER DEFAULT '0' NOT NULL,
	event_count INTEGER DEFAULT '0' NOT NULL,
	is_active BOOLEAN DEFAULT '0' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_vector_generations PRIMARY KEY (id),
	CONSTRAINT uq_vector_generations_book_id UNIQUE (book_id, generation),
	CONSTRAINT ck_vector_generations_generation_positive CHECK (generation >= 1),
	CONSTRAINT ck_vector_generations_status_values CHECK (status IN ('ready','stale','building','failed','degraded')),
	CONSTRAINT ck_vector_generations_counts_nonnegative CHECK (page_count >= 0 AND event_count >= 0),
	CONSTRAINT fk_vector_generations_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_vector_generations_run_id_analysis_runs FOREIGN KEY(run_id) REFERENCES analysis_runs (id) ON DELETE SET NULL
)

;


CREATE TABLE analysis_layer_results (
	id VARCHAR(36) NOT NULL,
	run_id VARCHAR(36) NOT NULL,
	layer_index INTEGER NOT NULL,
	layer_name VARCHAR(200) NOT NULL,
	unit_index INTEGER NOT NULL,
	chapter_id VARCHAR(36),
	page_range_snapshot_json TEXT DEFAULT '{}' NOT NULL,
	content_json TEXT NOT NULL,
	input_fingerprint VARCHAR(64) NOT NULL,
	status VARCHAR(16) DEFAULT 'staging' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_analysis_layer_results PRIMARY KEY (id),
	CONSTRAINT uq_analysis_layer_results_run_id UNIQUE (run_id, layer_index, unit_index),
	CONSTRAINT ck_analysis_layer_results_indices_nonnegative CHECK (layer_index >= 0 AND unit_index >= 0),
	CONSTRAINT ck_analysis_layer_results_status_values CHECK (status IN ('staging','published','stale')),
	CONSTRAINT fk_analysis_layer_results_run_id_analysis_runs FOREIGN KEY(run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE,
	CONSTRAINT fk_analysis_layer_results_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE SET NULL
)

;


CREATE TABLE browser_sessions (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	chapter_id VARCHAR(36) NOT NULL,
	page_url TEXT NOT NULL,
	page_title VARCHAR(500) NOT NULL,
	settings_json TEXT,
	mode VARCHAR(16) DEFAULT 'standard' NOT NULL,
	status VARCHAR(16) DEFAULT 'active' NOT NULL,
	expires_at DATETIME NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_browser_sessions PRIMARY KEY (id),
	CONSTRAINT ck_browser_sessions_mode_values CHECK (mode IN ('standard','hq')),
	CONSTRAINT ck_browser_sessions_status_values CHECK (status IN ('active','cancelled')),
	CONSTRAINT uq_browser_sessions_book_id UNIQUE (book_id),
	CONSTRAINT fk_browser_sessions_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT uq_browser_sessions_chapter_id UNIQUE (chapter_id),
	CONSTRAINT fk_browser_sessions_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE CASCADE
)

;


CREATE TABLE chapter_write_locks (
	chapter_id VARCHAR(36) NOT NULL,
	job_id VARCHAR(36) NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_chapter_write_locks PRIMARY KEY (chapter_id),
	CONSTRAINT fk_chapter_write_locks_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE CASCADE,
	CONSTRAINT fk_chapter_write_locks_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE
)

;


CREATE TABLE continuation_form_image_versions (
	id VARCHAR(36) NOT NULL,
	form_id VARCHAR(36) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	thumbnail_asset_id VARCHAR(36) NOT NULL,
	version INTEGER NOT NULL,
	is_adopted BOOLEAN DEFAULT '0' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_continuation_form_image_versions PRIMARY KEY (id),
	CONSTRAINT uq_continuation_form_image_versions_form_id UNIQUE (form_id, version),
	CONSTRAINT ck_continuation_form_image_versions_version_positive CHECK (version >= 1),
	CONSTRAINT fk_continuation_form_image_versions_form_id_continuation_character_forms FOREIGN KEY(form_id) REFERENCES continuation_character_forms (id) ON DELETE CASCADE,
	CONSTRAINT fk_continuation_form_image_versions_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_continuation_form_image_versions_thumbnail_asset_id_assets FOREIGN KEY(thumbnail_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE pages (
	id VARCHAR(36) NOT NULL,
	chapter_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	logical_source_path TEXT NOT NULL,
	source_revision INTEGER DEFAULT '1' NOT NULL,
	document_revision INTEGER DEFAULT '1' NOT NULL,
	rendered_revision INTEGER,
	render_status VARCHAR(32) DEFAULT 'not_rendered' NOT NULL,
	detection_state VARCHAR(32) DEFAULT 'unprocessed' NOT NULL,
	default_font_id VARCHAR(36),
	page_style_defaults_json TEXT DEFAULT '{}' NOT NULL,
	warnings_json TEXT DEFAULT '[]' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_pages PRIMARY KEY (id),
	CONSTRAINT uq_pages_chapter_id_ordinal UNIQUE (chapter_id, ordinal),
	CONSTRAINT uq_pages_chapter_id_logical_source_path UNIQUE (chapter_id, logical_source_path),
	CONSTRAINT ck_pages_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_pages_source_revision_positive CHECK (source_revision >= 1),
	CONSTRAINT ck_pages_document_revision_positive CHECK (document_revision >= 1),
	CONSTRAINT ck_pages_rendered_revision_positive CHECK (rendered_revision IS NULL OR rendered_revision >= 1),
	CONSTRAINT ck_pages_detection_state_values CHECK (detection_state IN ('unprocessed','processed')),
	CONSTRAINT ck_pages_render_status_values CHECK (render_status IN ('not_rendered','ready','stale','rendering','render_failed','awaiting_repair','repair_failed')),
	CONSTRAINT fk_pages_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE CASCADE,
	CONSTRAINT fk_pages_default_font_id_fonts FOREIGN KEY(default_font_id) REFERENCES fonts (id) ON DELETE RESTRICT
)

;


CREATE TABLE timeline_characters (
	id VARCHAR(36) NOT NULL,
	timeline_version_id VARCHAR(36) NOT NULL,
	name VARCHAR(500) NOT NULL,
	payload_json TEXT NOT NULL,
	CONSTRAINT pk_timeline_characters PRIMARY KEY (id),
	CONSTRAINT uq_timeline_characters_timeline_version_id UNIQUE (timeline_version_id, name),
	CONSTRAINT fk_timeline_characters_timeline_version_id_timeline_versions FOREIGN KEY(timeline_version_id) REFERENCES timeline_versions (id) ON DELETE CASCADE
)

;


CREATE TABLE timeline_events (
	id VARCHAR(36) NOT NULL,
	timeline_version_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	payload_json TEXT NOT NULL,
	CONSTRAINT pk_timeline_events PRIMARY KEY (id),
	CONSTRAINT uq_timeline_events_timeline_version_id UNIQUE (timeline_version_id, ordinal),
	CONSTRAINT ck_timeline_events_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT fk_timeline_events_timeline_version_id_timeline_versions FOREIGN KEY(timeline_version_id) REFERENCES timeline_versions (id) ON DELETE CASCADE
)

;


CREATE TABLE web_import_drafts (
	id VARCHAR(36) NOT NULL,
	book_id VARCHAR(36),
	chapter_id VARCHAR(36),
	status VARCHAR(24) NOT NULL,
	revision INTEGER DEFAULT '1' NOT NULL,
	config_json TEXT NOT NULL,
	temp_relative_path TEXT NOT NULL,
	expires_at DATETIME NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_web_import_drafts PRIMARY KEY (id),
	CONSTRAINT ck_web_import_drafts_status_values CHECK (status IN ('extracting','ready','committing','completed','failed','cancelled')),
	CONSTRAINT ck_web_import_drafts_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_web_import_drafts_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE SET NULL,
	CONSTRAINT fk_web_import_drafts_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE SET NULL
)

;


CREATE TABLE analysis_layer_result_pages (
	layer_result_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	page_id VARCHAR(36),
	page_id_snapshot VARCHAR(36) NOT NULL,
	page_number_snapshot INTEGER NOT NULL,
	CONSTRAINT pk_analysis_layer_result_pages PRIMARY KEY (layer_result_id, ordinal),
	CONSTRAINT ck_analysis_layer_result_pages_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_analysis_layer_result_pages_page_number_positive CHECK (page_number_snapshot >= 1),
	CONSTRAINT fk_analysis_layer_result_pages_layer_result_id_analysis_layer_results FOREIGN KEY(layer_result_id) REFERENCES analysis_layer_results (id) ON DELETE CASCADE,
	CONSTRAINT fk_analysis_layer_result_pages_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL
)

;


CREATE TABLE analysis_page_results (
	id VARCHAR(36) NOT NULL,
	run_id VARCHAR(36) NOT NULL,
	page_id VARCHAR(36),
	source_asset_id VARCHAR(36) NOT NULL,
	source_checksum VARCHAR(64) NOT NULL,
	page_id_snapshot VARCHAR(36) NOT NULL,
	page_number_snapshot INTEGER NOT NULL,
	payload_json TEXT NOT NULL,
	status VARCHAR(16) DEFAULT 'staging' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_analysis_page_results PRIMARY KEY (id),
	CONSTRAINT uq_analysis_page_results_run_id UNIQUE (run_id, page_id_snapshot),
	CONSTRAINT ck_analysis_page_results_status_values CHECK (status IN ('staging','published','stale')),
	CONSTRAINT ck_analysis_page_results_page_number_positive CHECK (page_number_snapshot >= 1),
	CONSTRAINT fk_analysis_page_results_run_id_analysis_runs FOREIGN KEY(run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE,
	CONSTRAINT fk_analysis_page_results_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL,
	CONSTRAINT fk_analysis_page_results_source_asset_id_assets FOREIGN KEY(source_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE analysis_run_targets (
	run_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	page_id VARCHAR(36),
	chapter_id VARCHAR(36),
	source_asset_id VARCHAR(36) NOT NULL,
	source_checksum VARCHAR(64) NOT NULL,
	page_id_snapshot VARCHAR(36) NOT NULL,
	page_number_snapshot INTEGER NOT NULL,
	status VARCHAR(16) DEFAULT 'pending' NOT NULL,
	error_json TEXT,
	CONSTRAINT pk_analysis_run_targets PRIMARY KEY (run_id, ordinal),
	CONSTRAINT uq_analysis_run_targets_run_id UNIQUE (run_id, page_id_snapshot),
	CONSTRAINT ck_analysis_run_targets_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_analysis_run_targets_page_number_positive CHECK (page_number_snapshot >= 1),
	CONSTRAINT ck_analysis_run_targets_status_values CHECK (status IN ('pending','completed','failed','conflict')),
	CONSTRAINT fk_analysis_run_targets_run_id_analysis_runs FOREIGN KEY(run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE,
	CONSTRAINT fk_analysis_run_targets_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL,
	CONSTRAINT fk_analysis_run_targets_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE SET NULL,
	CONSTRAINT fk_analysis_run_targets_source_asset_id_assets FOREIGN KEY(source_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE browser_session_credentials (
	session_id VARCHAR(36) NOT NULL,
	credential_version_id VARCHAR(36) NOT NULL,
	CONSTRAINT pk_browser_session_credentials PRIMARY KEY (session_id, credential_version_id),
	CONSTRAINT fk_browser_session_credentials_session_id_browser_sessions FOREIGN KEY(session_id) REFERENCES browser_sessions (id) ON DELETE CASCADE,
	CONSTRAINT fk_browser_session_credentials_credential_version_id_credential_versions FOREIGN KEY(credential_version_id) REFERENCES credential_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE browser_session_pages (
	id VARCHAR(36) NOT NULL,
	session_id VARCHAR(36) NOT NULL,
	client_page_key VARCHAR(200) NOT NULL,
	ordinal INTEGER NOT NULL,
	logical_path TEXT NOT NULL,
	source_url TEXT,
	source_asset_id VARCHAR(36) NOT NULL,
	thumbnail_asset_id VARCHAR(36) NOT NULL,
	page_id VARCHAR(36),
	job_id VARCHAR(36),
	retry_count INTEGER DEFAULT '0' NOT NULL,
	error_json TEXT,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_browser_session_pages PRIMARY KEY (id),
	CONSTRAINT uq_browser_session_pages_session_id UNIQUE (session_id, client_page_key),
	CONSTRAINT uq_browser_session_pages_session_id UNIQUE (session_id, ordinal),
	CONSTRAINT ck_browser_session_pages_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_browser_session_pages_retry_count_nonnegative CHECK (retry_count >= 0),
	CONSTRAINT fk_browser_session_pages_session_id_browser_sessions FOREIGN KEY(session_id) REFERENCES browser_sessions (id) ON DELETE CASCADE,
	CONSTRAINT fk_browser_session_pages_source_asset_id_assets FOREIGN KEY(source_asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_browser_session_pages_thumbnail_asset_id_assets FOREIGN KEY(thumbnail_asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT uq_browser_session_pages_page_id UNIQUE (page_id),
	CONSTRAINT fk_browser_session_pages_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE CASCADE,
	CONSTRAINT fk_browser_session_pages_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE SET NULL
)

;


CREATE TABLE bubbles (
	id VARCHAR(36) NOT NULL,
	page_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	font_id VARCHAR(36),
	payload_json TEXT NOT NULL,
	updated_revision INTEGER NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_bubbles PRIMARY KEY (id),
	CONSTRAINT uq_bubbles_page_id UNIQUE (page_id, ordinal),
	CONSTRAINT ck_bubbles_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_bubbles_updated_revision_positive CHECK (updated_revision >= 1),
	CONSTRAINT fk_bubbles_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE CASCADE,
	CONSTRAINT fk_bubbles_font_id_fonts FOREIGN KEY(font_id) REFERENCES fonts (id) ON DELETE RESTRICT
)

;


CREATE TABLE chapter_navigation_state (
	chapter_id VARCHAR(36) NOT NULL,
	last_visited_page_id VARCHAR(36),
	revision INTEGER DEFAULT '1' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_chapter_navigation_state PRIMARY KEY (chapter_id),
	CONSTRAINT ck_chapter_navigation_state_revision_positive CHECK (revision >= 1),
	CONSTRAINT fk_chapter_navigation_state_chapter_id_chapters FOREIGN KEY(chapter_id) REFERENCES chapters (id) ON DELETE CASCADE,
	CONSTRAINT fk_chapter_navigation_state_last_visited_page_id_pages FOREIGN KEY(last_visited_page_id) REFERENCES pages (id) ON DELETE SET NULL
)

;


CREATE TABLE job_items (
	id VARCHAR(36) NOT NULL,
	job_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	page_id VARCHAR(36),
	status VARCHAR(32) DEFAULT 'pending' NOT NULL,
	result_json TEXT,
	error_json TEXT,
	CONSTRAINT pk_job_items PRIMARY KEY (id),
	CONSTRAINT uq_job_items_job_id UNIQUE (job_id, ordinal),
	CONSTRAINT ck_job_items_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_job_items_status_values CHECK (status IN ('pending','running','completed','failed','skipped','cancelled')),
	CONSTRAINT fk_job_items_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE,
	CONSTRAINT fk_job_items_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL
)

;


CREATE TABLE render_requests (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	page_id VARCHAR(36) NOT NULL,
	requested_revision INTEGER NOT NULL,
	rendering_revision INTEGER,
	completed_revision INTEGER,
	status VARCHAR(16) DEFAULT 'pending' NOT NULL,
	executor_epoch_id VARCHAR(36),
	attempt_id VARCHAR(36),
	error_json TEXT,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_render_requests PRIMARY KEY (id),
	CONSTRAINT ck_render_requests_requested_revision_positive CHECK (requested_revision >= 1),
	CONSTRAINT ck_render_requests_rendering_revision_positive CHECK (rendering_revision IS NULL OR rendering_revision >= 1),
	CONSTRAINT ck_render_requests_completed_revision_positive CHECK (completed_revision IS NULL OR completed_revision >= 1),
	CONSTRAINT ck_render_requests_status_values CHECK (status IN ('pending', 'running', 'completed', 'failed')),
	CONSTRAINT uq_render_requests_page_id UNIQUE (page_id),
	CONSTRAINT fk_render_requests_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE CASCADE,
	CONSTRAINT fk_render_requests_executor_epoch_id_process_epochs FOREIGN KEY(executor_epoch_id) REFERENCES process_epochs (id) ON DELETE SET NULL
)

;


CREATE TABLE web_import_draft_pages (
	id VARCHAR(36) NOT NULL,
	draft_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	selected BOOLEAN DEFAULT '1' NOT NULL,
	source_url TEXT NOT NULL,
	temp_relative_path TEXT NOT NULL,
	thumbnail_asset_id VARCHAR(36),
	checksum VARCHAR(64),
	error_json TEXT,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_web_import_draft_pages PRIMARY KEY (id),
	CONSTRAINT uq_web_import_draft_pages_draft_id UNIQUE (draft_id, ordinal),
	CONSTRAINT ck_web_import_draft_pages_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT fk_web_import_draft_pages_draft_id_web_import_drafts FOREIGN KEY(draft_id) REFERENCES web_import_drafts (id) ON DELETE CASCADE,
	CONSTRAINT fk_web_import_draft_pages_thumbnail_asset_id_assets FOREIGN KEY(thumbnail_asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE analysis_heads (
	id VARCHAR(36) NOT NULL,
	book_id VARCHAR(36) NOT NULL,
	page_id VARCHAR(36),
	active_run_id VARCHAR(36) NOT NULL,
	active_result_id VARCHAR(36),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_analysis_heads PRIMARY KEY (id),
	CONSTRAINT ck_analysis_heads_target_shape CHECK ((page_id IS NULL AND active_result_id IS NULL) OR (page_id IS NOT NULL AND active_result_id IS NOT NULL)),
	CONSTRAINT fk_analysis_heads_book_id_books FOREIGN KEY(book_id) REFERENCES books (id) ON DELETE CASCADE,
	CONSTRAINT fk_analysis_heads_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE CASCADE,
	CONSTRAINT fk_analysis_heads_active_run_id_analysis_runs FOREIGN KEY(active_run_id) REFERENCES analysis_runs (id) ON DELETE RESTRICT,
	CONSTRAINT fk_analysis_heads_active_result_id_analysis_page_results FOREIGN KEY(active_result_id) REFERENCES analysis_page_results (id) ON DELETE RESTRICT
)

;


CREATE TABLE job_asset_inputs (
	job_id VARCHAR(36) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	role VARCHAR(64) NOT NULL,
	binding_phase VARCHAR(16) NOT NULL,
	job_item_id VARCHAR(36),
	CONSTRAINT pk_job_asset_inputs PRIMARY KEY (job_id, asset_id, role),
	CONSTRAINT ck_job_asset_inputs_binding_phase_values CHECK (binding_phase IN ('create','item_start','checkpoint')),
	CONSTRAINT fk_job_asset_inputs_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE CASCADE,
	CONSTRAINT fk_job_asset_inputs_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_job_asset_inputs_job_item_id_job_items FOREIGN KEY(job_item_id) REFERENCES job_items (id) ON DELETE CASCADE
)

;


CREATE TABLE job_steps (
	id VARCHAR(36) NOT NULL,
	job_item_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	kind VARCHAR(64) NOT NULL,
	status VARCHAR(32) DEFAULT 'pending' NOT NULL,
	attempt_id VARCHAR(36),
	checkpoint_json TEXT,
	error_json TEXT,
	CONSTRAINT pk_job_steps PRIMARY KEY (id),
	CONSTRAINT uq_job_steps_job_item_id UNIQUE (job_item_id, ordinal),
	CONSTRAINT ck_job_steps_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_job_steps_status_values CHECK (status IN ('pending','running','completed','failed','skipped','cancelled')),
	CONSTRAINT fk_job_steps_job_item_id_job_items FOREIGN KEY(job_item_id) REFERENCES job_items (id) ON DELETE CASCADE
)

;


CREATE TABLE note_citations (
	note_id VARCHAR(36) NOT NULL,
	ordinal INTEGER NOT NULL,
	page_id VARCHAR(36),
	page_id_snapshot VARCHAR(36) NOT NULL,
	page_number_snapshot INTEGER NOT NULL,
	source_analysis_id VARCHAR(36),
	excerpt TEXT DEFAULT '' NOT NULL,
	score FLOAT,
	CONSTRAINT pk_note_citations PRIMARY KEY (note_id, ordinal),
	CONSTRAINT ck_note_citations_ordinal_positive CHECK (ordinal >= 1),
	CONSTRAINT ck_note_citations_page_number_positive CHECK (page_number_snapshot >= 1),
	CONSTRAINT fk_note_citations_note_id_notes FOREIGN KEY(note_id) REFERENCES notes (id) ON DELETE CASCADE,
	CONSTRAINT fk_note_citations_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL,
	CONSTRAINT fk_note_citations_source_analysis_id_analysis_page_results FOREIGN KEY(source_analysis_id) REFERENCES analysis_page_results (id) ON DELETE SET NULL
)

;


CREATE TABLE operations (
	id VARCHAR(36) NOT NULL,
	owner_user_id VARCHAR(36) DEFAULT '00000000-0000-0000-0000-000000000010' NOT NULL,
	kind VARCHAR(64) NOT NULL,
	executor_role VARCHAR(16) NOT NULL,
	status VARCHAR(16) DEFAULT 'pending' NOT NULL,
	page_id VARCHAR(36),
	bubble_id VARCHAR(36),
	studio_document_id VARCHAR(36),
	studio_session_id VARCHAR(36),
	base_revision INTEGER,
	base_generation INTEGER,
	request_json TEXT NOT NULL,
	result_json TEXT,
	error_json TEXT,
	executor_epoch_id VARCHAR(36),
	attempt_id VARCHAR(36),
	started_at DATETIME,
	finished_at DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_operations PRIMARY KEY (id),
	CONSTRAINT ck_operations_status_values CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
	CONSTRAINT ck_operations_kind_values CHECK (kind IN ('bubble_ocr', 'bubble_color', 'page_detect', 'page_repair', 'bubble_translate', 'studio_generate', 'studio_chat', 'studio_summary')),
	CONSTRAINT ck_operations_executor_role_values CHECK (executor_role IN ('api','worker')),
	CONSTRAINT ck_operations_kind_executor_shape CHECK ((kind IN ('bubble_ocr','bubble_color','page_detect') AND executor_role = 'worker') OR (kind IN ('bubble_translate','studio_generate','studio_chat','studio_summary') AND executor_role = 'api') OR (kind = 'page_repair')),
	CONSTRAINT ck_operations_kind_target_shape CHECK (((kind IN ('bubble_ocr','bubble_color','bubble_translate') AND studio_document_id IS NULL AND studio_session_id IS NULL AND ((status IN ('pending','running') AND page_id IS NOT NULL AND bubble_id IS NOT NULL) OR status IN ('completed','failed','cancelled'))) OR (kind IN ('page_detect','page_repair') AND bubble_id IS NULL AND studio_document_id IS NULL AND studio_session_id IS NULL AND ((status IN ('pending','running') AND page_id IS NOT NULL) OR status IN ('completed','failed','cancelled'))) OR (kind = 'studio_generate' AND page_id IS NULL AND bubble_id IS NULL AND studio_session_id IS NULL AND ((status IN ('pending','running') AND studio_document_id IS NOT NULL) OR status IN ('completed','failed','cancelled'))) OR (kind IN ('studio_chat','studio_summary') AND page_id IS NULL AND bubble_id IS NULL AND studio_document_id IS NULL AND ((status IN ('pending','running') AND studio_session_id IS NOT NULL) OR status IN ('completed','failed','cancelled'))))),
	CONSTRAINT ck_operations_base_revision_positive CHECK (base_revision IS NULL OR base_revision >= 1),
	CONSTRAINT ck_operations_base_generation_positive CHECK (base_generation IS NULL OR base_generation >= 1),
	CONSTRAINT fk_operations_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL,
	CONSTRAINT fk_operations_bubble_id_bubbles FOREIGN KEY(bubble_id) REFERENCES bubbles (id) ON DELETE SET NULL,
	CONSTRAINT fk_operations_studio_document_id_studio_documents FOREIGN KEY(studio_document_id) REFERENCES studio_documents (id) ON DELETE SET NULL,
	CONSTRAINT fk_operations_studio_session_id_studio_chat_sessions FOREIGN KEY(studio_session_id) REFERENCES studio_chat_sessions (id) ON DELETE SET NULL,
	CONSTRAINT fk_operations_executor_epoch_id_process_epochs FOREIGN KEY(executor_epoch_id) REFERENCES process_epochs (id) ON DELETE SET NULL
)

;


CREATE TABLE job_step_asset_outputs (
	job_step_id VARCHAR(36) NOT NULL,
	role VARCHAR(64) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	CONSTRAINT pk_job_step_asset_outputs PRIMARY KEY (job_step_id, role),
	CONSTRAINT fk_job_step_asset_outputs_job_step_id_job_steps FOREIGN KEY(job_step_id) REFERENCES job_steps (id) ON DELETE CASCADE,
	CONSTRAINT fk_job_step_asset_outputs_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE operation_artifacts (
	operation_id VARCHAR(36) NOT NULL,
	kind VARCHAR(64) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	page_id VARCHAR(36),
	expires_at DATETIME,
	CONSTRAINT pk_operation_artifacts PRIMARY KEY (operation_id, kind),
	CONSTRAINT fk_operation_artifacts_operation_id_operations FOREIGN KEY(operation_id) REFERENCES operations (id) ON DELETE CASCADE,
	CONSTRAINT fk_operation_artifacts_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_operation_artifacts_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE SET NULL
)

;


CREATE TABLE operation_asset_inputs (
	operation_id VARCHAR(36) NOT NULL,
	role VARCHAR(64) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	CONSTRAINT pk_operation_asset_inputs PRIMARY KEY (operation_id, role),
	CONSTRAINT fk_operation_asset_inputs_operation_id_operations FOREIGN KEY(operation_id) REFERENCES operations (id) ON DELETE CASCADE,
	CONSTRAINT fk_operation_asset_inputs_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT
)

;


CREATE TABLE operation_credential_snapshots (
	operation_id VARCHAR(36) NOT NULL,
	credential_version_id VARCHAR(36) NOT NULL,
	role VARCHAR(64) NOT NULL,
	CONSTRAINT pk_operation_credential_snapshots PRIMARY KEY (operation_id, credential_version_id, role),
	CONSTRAINT fk_operation_credential_snapshots_operation_id_operations FOREIGN KEY(operation_id) REFERENCES operations (id) ON DELETE CASCADE,
	CONSTRAINT fk_operation_credential_snapshots_credential_version_id_credential_versions FOREIGN KEY(credential_version_id) REFERENCES credential_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE operation_events (
	id INTEGER NOT NULL,
	operation_id VARCHAR(36) NOT NULL,
	type VARCHAR(64) NOT NULL,
	payload_json TEXT DEFAULT '{}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_operation_events PRIMARY KEY (id),
	CONSTRAINT fk_operation_events_operation_id_operations FOREIGN KEY(operation_id) REFERENCES operations (id) ON DELETE CASCADE
)

;


CREATE TABLE operation_font_snapshots (
	operation_id VARCHAR(36) NOT NULL,
	font_id VARCHAR(36) NOT NULL,
	role VARCHAR(64) NOT NULL,
	CONSTRAINT pk_operation_font_snapshots PRIMARY KEY (operation_id, font_id, role),
	CONSTRAINT fk_operation_font_snapshots_operation_id_operations FOREIGN KEY(operation_id) REFERENCES operations (id) ON DELETE CASCADE,
	CONSTRAINT fk_operation_font_snapshots_font_id_fonts FOREIGN KEY(font_id) REFERENCES fonts (id) ON DELETE RESTRICT
)

;


CREATE TABLE operation_plugin_snapshots (
	operation_id VARCHAR(36) NOT NULL,
	plugin_version_id VARCHAR(36) NOT NULL,
	config_json TEXT DEFAULT '{}' NOT NULL,
	CONSTRAINT pk_operation_plugin_snapshots PRIMARY KEY (operation_id, plugin_version_id),
	CONSTRAINT fk_operation_plugin_snapshots_operation_id_operations FOREIGN KEY(operation_id) REFERENCES operations (id) ON DELETE CASCADE,
	CONSTRAINT fk_operation_plugin_snapshots_plugin_version_id_plugin_versions FOREIGN KEY(plugin_version_id) REFERENCES plugin_versions (id) ON DELETE RESTRICT
)

;


CREATE TABLE page_assets (
	page_id VARCHAR(36) NOT NULL,
	role VARCHAR(32) NOT NULL,
	asset_id VARCHAR(36) NOT NULL,
	input_source_revision INTEGER,
	input_document_revision INTEGER,
	parent_asset_id VARCHAR(36),
	producer_job_step_id VARCHAR(36),
	producer_operation_id VARCHAR(36),
	producer_render_request_id VARCHAR(36),
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	CONSTRAINT pk_page_assets PRIMARY KEY (page_id, role),
	CONSTRAINT ck_page_assets_role_values CHECK (role IN ('source', 'thumbnail_source', 'clean', 'translated', 'text_mask')),
	CONSTRAINT ck_page_assets_single_producer CHECK ((CASE WHEN producer_job_step_id IS NULL THEN 0 ELSE 1 END + CASE WHEN producer_operation_id IS NULL THEN 0 ELSE 1 END + CASE WHEN producer_render_request_id IS NULL THEN 0 ELSE 1 END) <= 1),
	CONSTRAINT ck_page_assets_thumbnail_parent_required CHECK ((role = 'thumbnail_source' AND parent_asset_id IS NOT NULL) OR role != 'thumbnail_source'),
	CONSTRAINT ck_page_assets_input_source_revision_positive CHECK (input_source_revision IS NULL OR input_source_revision >= 1),
	CONSTRAINT ck_page_assets_input_document_revision_positive CHECK (input_document_revision IS NULL OR input_document_revision >= 1),
	CONSTRAINT fk_page_assets_page_id_pages FOREIGN KEY(page_id) REFERENCES pages (id) ON DELETE CASCADE,
	CONSTRAINT fk_page_assets_asset_id_assets FOREIGN KEY(asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_page_assets_parent_asset_id_assets FOREIGN KEY(parent_asset_id) REFERENCES assets (id) ON DELETE RESTRICT,
	CONSTRAINT fk_page_assets_producer_job_step_id_job_steps FOREIGN KEY(producer_job_step_id) REFERENCES job_steps (id) ON DELETE SET NULL,
	CONSTRAINT fk_page_assets_producer_operation_id_operations FOREIGN KEY(producer_operation_id) REFERENCES operations (id) ON DELETE SET NULL,
	CONSTRAINT fk_page_assets_producer_render_request_id_render_requests FOREIGN KEY(producer_render_request_id) REFERENCES render_requests (id) ON DELETE SET NULL
)

;

CREATE INDEX ix_analysis_runs_book_created ON analysis_runs (book_id, created_at);

CREATE INDEX ix_assets_checksum ON assets (checksum);

CREATE INDEX ix_assets_gc_marked_at ON assets (gc_marked_at);

CREATE INDEX ix_assets_owner_state ON assets (owner_user_id, integrity_status);

CREATE INDEX ix_continuation_projects_source_run_id ON continuation_projects (source_run_id);

CREATE INDEX ix_idempotency_records_expires_at ON idempotency_records (expires_at);

CREATE INDEX ix_jobs_analysis_run_id ON jobs (analysis_run_id);

CREATE INDEX ix_jobs_batch_status ON jobs (batch_id, status);

CREATE INDEX ix_jobs_blocked_by_job_id ON jobs (blocked_by_job_id);

CREATE INDEX ix_jobs_book_id ON jobs (book_id);

CREATE INDEX ix_jobs_chapter_status ON jobs (chapter_id, status);

CREATE INDEX ix_jobs_continuation_project_id ON jobs (continuation_project_id);

CREATE INDEX ix_jobs_page_id ON jobs (page_id);

CREATE INDEX ix_jobs_queue_claim ON jobs (status, queue_rank);

CREATE INDEX ix_jobs_retry_source ON jobs (retry_of_job_id);

CREATE INDEX ix_jobs_worker_epoch_id ON jobs (worker_epoch_id);

CREATE UNIQUE INDEX uq_jobs_one_current ON jobs (1 + CAST(id IS NULL AS INTEGER)) WHERE status IN ('running');

CREATE UNIQUE INDEX uq_jobs_one_nonterminal_translation_per_chapter ON jobs (chapter_id) WHERE kind = 'translation' AND status IN ('queued', 'running', 'paused', 'interrupted') AND chapter_id IS NOT NULL;

CREATE UNIQUE INDEX uq_jobs_one_nonterminal_web_commit_per_draft ON jobs (web_import_draft_id) WHERE kind = 'web_import_commit' AND status IN ('queued', 'running', 'paused', 'interrupted') AND web_import_draft_id IS NOT NULL;

CREATE INDEX ix_process_epochs_role_status ON process_epochs (role, status);

CREATE INDEX ix_studio_chat_sessions_document_updated ON studio_chat_sessions (document_id, updated_at);

CREATE INDEX ix_studio_chat_sessions_summary_message ON studio_chat_sessions (summary_through_message_id);

CREATE UNIQUE INDEX uq_studio_chat_sessions_one_active ON studio_chat_sessions (document_id) WHERE archived_at IS NULL;

CREATE INDEX ix_studio_messages_session_ordinal ON studio_messages (session_id, ordinal);

CREATE INDEX ix_books_cover_asset_id ON books (cover_asset_id);

CREATE UNIQUE INDEX uq_books_one_quick_workspace ON books (owner_user_id, kind) WHERE kind = 'quick_workspace';

CREATE INDEX ix_continuation_project_reference_assets_asset_id ON continuation_project_reference_assets (asset_id);

CREATE INDEX ix_continuation_scripts_project_id ON continuation_scripts (project_id);

CREATE INDEX ix_fonts_asset_id ON fonts (asset_id);

CREATE UNIQUE INDEX uq_fonts_builtin_key ON fonts (builtin_key) WHERE builtin_key IS NOT NULL;

CREATE INDEX ix_invite_codes_created_by ON invite_codes (created_by_user_id);

CREATE INDEX ix_invite_codes_expires_at ON invite_codes (expires_at);

CREATE INDEX ix_invite_codes_used_by ON invite_codes (used_by_user_id);

CREATE INDEX ix_job_artifacts_asset_id ON job_artifacts (asset_id);

CREATE INDEX ix_job_events_job_cursor ON job_events (job_id, id);

CREATE INDEX ix_plugin_versions_plugin_id ON plugin_versions (plugin_id);

CREATE INDEX ix_recovery_codes_user_id ON recovery_codes (user_id);

CREATE INDEX ix_sessions_expires_at ON sessions (expires_at);

CREATE INDEX ix_sessions_user_id ON sessions (user_id);

CREATE INDEX ix_studio_message_assets_asset_id ON studio_message_assets (asset_id);

CREATE INDEX ix_analysis_artifacts_asset_id ON analysis_artifacts (asset_id);

CREATE INDEX ix_analysis_artifacts_book_kind_created ON analysis_artifacts (book_id, kind, created_at);

CREATE INDEX ix_analysis_artifacts_run_id ON analysis_artifacts (run_id);

CREATE UNIQUE INDEX uq_analysis_artifacts_active ON analysis_artifacts (book_id, kind, template) WHERE is_active IS 1;

CREATE INDEX ix_continuation_character_forms_adopted_asset_id ON continuation_character_forms (adopted_asset_id);

CREATE INDEX ix_continuation_character_forms_reference_asset_id ON continuation_character_forms (reference_asset_id);

CREATE INDEX ix_continuation_character_forms_reference_thumbnail_asset_id ON continuation_character_forms (reference_thumbnail_asset_id);

CREATE INDEX ix_continuation_image_versions_asset_id ON continuation_image_versions (asset_id);

CREATE INDEX ix_continuation_image_versions_thumbnail_asset_id ON continuation_image_versions (thumbnail_asset_id);

CREATE UNIQUE INDEX uq_continuation_image_versions_active ON continuation_image_versions (continuation_page_id) WHERE is_active IS 1;

CREATE INDEX ix_notes_book_updated ON notes (book_id, updated_at);

CREATE INDEX ix_provider_settings_credential_version ON provider_settings (credential_version_id);

CREATE INDEX ix_studio_documents_avatar_asset_id ON studio_documents (avatar_asset_id);

CREATE INDEX ix_studio_documents_book_id ON studio_documents (book_id);

CREATE INDEX ix_timeline_versions_run_id ON timeline_versions (run_id);

CREATE UNIQUE INDEX uq_timeline_versions_active ON timeline_versions (book_id) WHERE is_active IS 1;

CREATE INDEX ix_transient_requests_claim ON transient_requests (status, created_at);

CREATE INDEX ix_transient_requests_worker_epoch_id ON transient_requests (worker_epoch_id);

CREATE UNIQUE INDEX uq_transient_active_vector_query_book ON transient_requests (book_id) WHERE kind = 'vector_query' AND book_id IS NOT NULL AND connection_open IS 1 AND status IN ('pending', 'running', 'completed');

CREATE INDEX ix_vector_generations_run_id ON vector_generations (run_id);

CREATE UNIQUE INDEX uq_vector_generations_active ON vector_generations (book_id) WHERE is_active IS 1;

CREATE INDEX ix_analysis_layer_results_chapter_id ON analysis_layer_results (chapter_id);

CREATE INDEX ix_analysis_layer_results_run_layer ON analysis_layer_results (run_id, layer_index, unit_index);

CREATE INDEX ix_browser_sessions_expiry ON browser_sessions (expires_at);

CREATE INDEX ix_chapter_write_locks_job ON chapter_write_locks (job_id);

CREATE INDEX ix_continuation_form_image_versions_asset_id ON continuation_form_image_versions (asset_id);

CREATE INDEX ix_continuation_form_image_versions_thumbnail_asset_id ON continuation_form_image_versions (thumbnail_asset_id);

CREATE UNIQUE INDEX uq_continuation_form_image_versions_adopted ON continuation_form_image_versions (form_id) WHERE is_adopted IS 1;

CREATE INDEX ix_pages_default_font_id ON pages (default_font_id);

CREATE INDEX ix_timeline_events_version_ordinal ON timeline_events (timeline_version_id, ordinal);

CREATE INDEX ix_web_import_drafts_book_id ON web_import_drafts (book_id);

CREATE INDEX ix_web_import_drafts_chapter_status_expiry ON web_import_drafts (chapter_id, status, expires_at);

CREATE INDEX ix_analysis_layer_result_pages_page_id ON analysis_layer_result_pages (page_id);

CREATE INDEX ix_analysis_page_results_page_created ON analysis_page_results (page_id, created_at);

CREATE INDEX ix_analysis_page_results_source_asset_id ON analysis_page_results (source_asset_id);

CREATE INDEX ix_analysis_run_targets_chapter_id ON analysis_run_targets (chapter_id);

CREATE INDEX ix_analysis_run_targets_page ON analysis_run_targets (page_id, run_id);

CREATE INDEX ix_analysis_run_targets_source_asset_id ON analysis_run_targets (source_asset_id);

CREATE INDEX ix_browser_session_credentials_version ON browser_session_credentials (credential_version_id);

CREATE INDEX ix_browser_session_pages_job ON browser_session_pages (job_id);

CREATE INDEX ix_browser_session_pages_pending ON browser_session_pages (session_id, page_id, job_id);

CREATE INDEX ix_browser_session_pages_source_asset ON browser_session_pages (source_asset_id);

CREATE INDEX ix_browser_session_pages_thumbnail_asset ON browser_session_pages (thumbnail_asset_id);

CREATE INDEX ix_bubbles_font_id ON bubbles (font_id);

CREATE INDEX ix_chapter_navigation_state_last_page ON chapter_navigation_state (last_visited_page_id);

CREATE INDEX ix_job_items_job_status_ordinal ON job_items (job_id, status, ordinal);

CREATE INDEX ix_job_items_page_id ON job_items (page_id);

CREATE INDEX ix_render_requests_claim ON render_requests (status, updated_at);

CREATE INDEX ix_render_requests_executor_epoch_id ON render_requests (executor_epoch_id);

CREATE INDEX ix_web_import_draft_pages_thumbnail_asset ON web_import_draft_pages (thumbnail_asset_id);

CREATE INDEX ix_analysis_heads_active_result_id ON analysis_heads (active_result_id);

CREATE INDEX ix_analysis_heads_active_run_id ON analysis_heads (active_run_id);

CREATE UNIQUE INDEX uq_analysis_heads_book ON analysis_heads (book_id) WHERE page_id IS NULL;

CREATE UNIQUE INDEX uq_analysis_heads_page ON analysis_heads (page_id) WHERE page_id IS NOT NULL;

CREATE INDEX ix_job_asset_inputs_asset_id ON job_asset_inputs (asset_id);

CREATE INDEX ix_job_asset_inputs_job_item_id ON job_asset_inputs (job_item_id);

CREATE INDEX ix_note_citations_page_id ON note_citations (page_id);

CREATE INDEX ix_note_citations_source_analysis_id ON note_citations (source_analysis_id);

CREATE INDEX ix_operations_bubble_id ON operations (bubble_id);

CREATE INDEX ix_operations_executor_claim ON operations (executor_role, status, created_at);

CREATE INDEX ix_operations_executor_epoch_id ON operations (executor_epoch_id);

CREATE INDEX ix_operations_page_id ON operations (page_id);

CREATE UNIQUE INDEX uq_operations_one_active_page_write ON operations (page_id) WHERE page_id IS NOT NULL AND status IN ('pending', 'running') AND kind IN ('bubble_ocr', 'bubble_color', 'page_detect', 'page_repair', 'bubble_translate');

CREATE UNIQUE INDEX uq_operations_one_active_studio_generate ON operations (studio_document_id) WHERE studio_document_id IS NOT NULL AND status IN ('pending', 'running') AND kind = 'studio_generate';

CREATE UNIQUE INDEX uq_operations_one_active_studio_session ON operations (studio_session_id) WHERE studio_session_id IS NOT NULL AND status IN ('pending', 'running') AND kind IN ('studio_chat', 'studio_summary');

CREATE INDEX ix_job_step_asset_outputs_asset_id ON job_step_asset_outputs (asset_id);

CREATE INDEX ix_operation_artifacts_asset_id ON operation_artifacts (asset_id);

CREATE INDEX ix_operation_artifacts_page_id ON operation_artifacts (page_id);

CREATE INDEX ix_operation_asset_inputs_asset_id ON operation_asset_inputs (asset_id);

CREATE INDEX ix_operation_events_operation_cursor ON operation_events (operation_id, id);

CREATE INDEX ix_page_assets_asset_id ON page_assets (asset_id);

CREATE INDEX ix_page_assets_parent_asset_id ON page_assets (parent_asset_id);

CREATE INDEX ix_page_assets_producer_job_step_id ON page_assets (producer_job_step_id);

CREATE INDEX ix_page_assets_producer_operation_id ON page_assets (producer_operation_id);

CREATE INDEX ix_page_assets_producer_render_request_id ON page_assets (producer_render_request_id);
