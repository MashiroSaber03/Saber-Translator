from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import json
from copy import deepcopy
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from flask import Flask
from fontTools.ttLib import TTCollection, TTFont
from sqlalchemy import event, insert, select, text, update

from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage import builtin_fonts
from src.backend_v2.storage.builtin_fonts import discover_bundled_fonts
from src.backend_v2.storage.consistency import ConsistencyChecker
from src.backend_v2.storage.database import (
    create_sqlite_engine,
    immediate_transaction,
)
from src.backend_v2.storage.defaults import (
    DEFAULT_INSIGHT_SETTINGS,
    DEFAULT_TEXT_STYLE,
    default_translation_settings,
)
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
    hash_epoch_token,
    utcnow,
)
from src.backend_v2.storage.lifecycle import (
    UnsupportedDataRoot,
    initialize_database,
    schema_smoke_test,
)
from src.backend_v2.storage.platform_repositories import (
    BookSettingMutation,
    CredentialEdit,
    FontRepository,
    PromptMutation,
    PromptRepository,
    ProviderRateLimiter,
    ProviderSettingMutation,
    RevisionConflict,
    SettingMutation,
    SettingsRepository,
)
from src.backend_v2.storage.schema import (
    app_settings,
    assets,
    api_executor_leases,
    books,
    bubbles,
    chapter_write_intents,
    chapter_write_locks,
    chapters,
    credential_versions,
    credentials,
    fonts,
    idempotency_records,
    job_events,
    job_items,
    job_steps,
    jobs,
    metadata,
    object_commit_journal,
    operation_events,
    operations,
    pages,
    process_epochs,
    render_requests,
    worker_leases,
)
from src.backend_v2.storage.seeding import (
    QUICK_WORKSPACE_BOOK_ID,
    QUICK_WORKSPACE_CHAPTER_ID,
    seed_system_records,
)
from src.backend_v2.storage.single_instance import (
    DataRootAlreadyLocked,
    DataRootLock,
)
from src.backend_v2.settings.validation import (
    validate_credential_secret,
    validate_provider_setting_payload,
    validate_setting_payload,
)
from src.backend_v2.settings.diagnostics import ProviderDiagnostics
from src.backend_v2.settings.routes import create_settings_blueprint
from src.backend_v2.worker.maintenance import WorkerMaintenance
from src.shared import constants as shared_constants


def _stored_job_progress(status: str) -> str:
    return json.dumps(
        {
            "executionMode": "sequential",
            "jobStatus": status,
            "totalItems": 0,
            "completedItems": 0,
            "failedItems": 0,
            "skippedItems": 0,
            "cancelledItems": 0,
            "pools": [],
        },
        separators=(",", ":"),
    )


@pytest.fixture()
def platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    try:
        yield data_root, engine
    finally:
        engine.dispose()


def test_launcher_initialization_seeds_one_persistent_quick_workspace(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    first = initialize_database(data_root)
    assert first.schema_revision == "v2_foundation_20260819"
    assert first.created is True
    assert schema_smoke_test(first.database_path) == "v2_foundation_20260819"

    engine = create_sqlite_engine(first.database_path)
    with engine.connect() as connection:
        quick_books = connection.execute(
            select(books.c.id).where(books.c.kind == "quick_workspace")
        ).scalars().all()
        quick_chapters = connection.execute(
            select(chapters.c.id).where(chapters.c.book_id == quick_books[0])
        ).scalars().all()
        seeded_fonts = connection.execute(
            select(
                fonts.c.id,
                fonts.c.builtin_key,
                fonts.c.display_name,
            ).where(fonts.c.kind == "builtin")
        ).mappings().all()
        seeded_setting_domains = set(
            connection.execute(select(app_settings.c.domain)).scalars()
        )
    engine.dispose()
    assert quick_books == [QUICK_WORKSPACE_BOOK_ID]
    assert quick_chapters == [QUICK_WORKSPACE_CHAPTER_ID]
    assert seeded_setting_domains == {
        "insight",
        "text_style_defaults",
        "translation",
        "web_import",
        "workflow_preferences",
    }
    default_font = next(
        font for font in discover_bundled_fonts() if font.builtin_key == "default"
    )
    assert default_font.file_name == "思源黑体SourceHanSansK-Bold.TTF"
    assert default_font.display_name == "思源黑体"
    assert {
        (str(row["id"]), str(row["builtin_key"]), str(row["display_name"]))
        for row in seeded_fonts
    } == {
        (font.id, font.builtin_key, font.display_name)
        for font in discover_bundled_fonts()
    }

    second = initialize_database(data_root)
    assert second.created is False

    engine = create_sqlite_engine(second.database_path)
    try:
        listed_fonts = FontRepository(engine).list()
        assert len(listed_fonts) == len(discover_bundled_fonts())
        assert listed_fonts[0]["builtinKey"] == "default"
        assert listed_fonts[0]["displayName"] == "思源黑体"
    finally:
        engine.dispose()


def test_bundled_font_catalog_uses_an_available_font_when_preferred_is_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    font_root = tmp_path / "fonts"
    font_root.mkdir()
    fallback = font_root / "CustomDefault.ttf"
    fallback.write_bytes(b"custom-font")
    (font_root / "OtherFont.otf").write_bytes(b"other-font")
    monkeypatch.setattr(builtin_fonts, "_font_resource_root", lambda: font_root)
    discover_bundled_fonts.cache_clear()

    try:
        catalog = discover_bundled_fonts()
        default_font = next(font for font in catalog if font.builtin_key == "default")

        assert default_font.file_name == fallback.name
        assert builtin_fonts.resolve_bundled_font_path("default") == str(fallback.resolve())
        data_root = tmp_path / "data-v2"
        data_root.mkdir()
        assert initialize_database(data_root).created is True
    finally:
        discover_bundled_fonts.cache_clear()


def test_storage_initialization_rejects_conflicting_builtin_font_metadata(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    initialized = initialize_database(data_root)
    engine = create_sqlite_engine(initialized.database_path)
    with engine.begin() as connection:
        connection.execute(
            update(fonts)
            .where(fonts.c.builtin_key == "default")
            .values(display_name="默认字体")
        )
    engine.dispose()

    with pytest.raises(RuntimeError, match="display name mismatch"):
        initialize_database(data_root)


@pytest.mark.parametrize(
    "retired_revision",
    [None, "0017", "v2_foundation_20260810"],
)
def test_storage_initialization_rejects_nonformal_database_without_rewriting_it(
    tmp_path: Path,
    retired_revision: str | None,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    database_path = data_root / "saber.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel VALUES ('untouched')")
        if retired_revision is not None:
            connection.execute(
                "CREATE TABLE alembic_version(version_num VARCHAR(32) NOT NULL)"
            )
            connection.execute(
                "INSERT INTO alembic_version VALUES (?)",
                (retired_revision,),
            )

    with pytest.raises(UnsupportedDataRoot, match="旧数据不会被读取或迁移"):
        initialize_database(data_root)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT value FROM sentinel").fetchall() == [
            ("untouched",)
        ]


def test_storage_initialization_rejects_extra_nonformal_tables(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    initialized = initialize_database(data_root)
    with sqlite3.connect(initialized.database_path) as connection:
        connection.execute("CREATE TABLE retired_payload(value TEXT NOT NULL)")

    with pytest.raises(RuntimeError, match="unexpected=.*retired_payload"):
        initialize_database(data_root)


def test_custom_insight_architecture_requires_at_least_two_layers() -> None:
    payload = deepcopy(DEFAULT_INSIGHT_SETTINGS)
    payload["analysis"]["batch"]["architecturePreset"] = "custom"
    payload["analysis"]["batch"]["customLayers"] = []

    with pytest.raises(ValueError, match="must contain at least 2 layers"):
        validate_setting_payload("insight", payload, schema_version=1)


def test_custom_insight_architecture_has_no_arbitrary_size_gate() -> None:
    payload = deepcopy(DEFAULT_INSIGHT_SETTINGS)
    payload["analysis"]["batch"].update(
        {
            "architecturePreset": "custom",
            "pagesPerBatch": 21,
            "contextBatchCount": 11,
            "customLayers": [
                {
                    "name": f"Layer {index}",
                    "unitsPerGroup": 101 + index,
                    "alignToChapter": False,
                }
                for index in range(9)
            ],
        }
    )

    validated = validate_setting_payload("insight", payload, schema_version=1)

    assert len(validated["analysis"]["batch"]["customLayers"]) == 9


def _current_insight_provider_payload(domain: str) -> dict[str, object]:
    common: dict[str, object] = {
        "modelName": "test-model",
        "customBaseUrl": "",
    }
    if domain in {"insight_vlm", "insight_chat"}:
        common["openaiOptions"] = {
            "request": {
                "force_json_output": False,
                "temperature": 0.3 if domain == "insight_vlm" else None,
                "extra_body": {},
            },
            "execution": {
                "use_stream": False,
                "rpm_limit": 0,
                "transport_retries": 1,
                "business_retries": 0,
            },
        }
        if domain == "insight_vlm":
            common["imageMaxSize"] = 0
        return common
    if domain == "insight_embedding":
        common["rpmLimit"] = 0
    common.update(
        {
            "transportRetries": 1,
            "businessRetries": 0,
            "timeoutSeconds": 0,
        }
    )
    return common


def _current_web_import_settings() -> dict[str, object]:
    return {
        "firecrawl": {},
        "agent": {
            "provider": "custom",
            "customBaseUrl": "https://agent.example/v1",
            "modelName": "agent-model",
            "useStream": False,
            "forceJsonOutput": True,
            "maxRetries": 3,
            "timeout": 120,
        },
        "extraction": {"prompt": "extract", "maxIterations": 10},
        "download": {
            "concurrency": 3,
            "timeout": 30,
            "retries": 3,
            "delay": 100,
            "useReferer": True,
        },
        "imagePreprocess": {
            "enabled": False,
            "autoRotate": True,
            "compression": {
                "enabled": False,
                "quality": 85,
                "maxWidth": 0,
                "maxHeight": 0,
            },
            "formatConvert": {
                "enabled": False,
                "targetFormat": "original",
            },
        },
        "advanced": {"bypassProxy": False},
        "ui": {"showAgentLogs": True, "autoImport": False},
    }


def test_web_import_settings_reject_noncurrent_field_types() -> None:
    valid = _current_web_import_settings()
    assert validate_setting_payload("web_import", valid, schema_version=1) == valid

    invalid_boolean = deepcopy(valid)
    invalid_boolean["download"]["useReferer"] = "true"
    with pytest.raises(ValueError, match="download.useReferer must be boolean"):
        validate_setting_payload("web_import", invalid_boolean, schema_version=1)

    invalid_string = deepcopy(valid)
    invalid_string["agent"]["modelName"] = None
    with pytest.raises(ValueError, match="agent.modelName must be a string"):
        validate_setting_payload("web_import", invalid_string, schema_version=1)


def test_web_import_settings_have_no_arbitrary_numeric_upper_gates() -> None:
    payload = _current_web_import_settings()
    payload["agent"]["maxRetries"] = 101
    payload["agent"]["timeout"] = 3_601
    payload["extraction"]["maxIterations"] = 101
    payload["download"].update(
        {
            "concurrency": 33,
            "timeout": 3_601,
            "retries": 101,
            "delay": 60_001,
        }
    )
    payload["imagePreprocess"]["compression"].update(
        {"maxWidth": 100_001, "maxHeight": 100_001}
    )

    assert validate_setting_payload("web_import", payload, schema_version=1) == payload


def test_web_import_provider_setting_accepts_only_current_fields() -> None:
    valid = {
        "modelName": "agent-model",
        "customBaseUrl": "https://agent.example/v1",
    }
    assert validate_provider_setting_payload(
        "web_import_agent",
        "custom",
        valid,
        schema_version=1,
    ) == valid

    with pytest.raises(ValueError, match="invalid fields"):
        validate_provider_setting_payload(
            "web_import_agent",
            "custom",
            {**valid, "openaiOptions": {}},
            schema_version=1,
        )


def test_provider_settings_reject_fields_owned_by_other_domains() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        validate_provider_setting_payload(
            "translation",
            "custom",
            {"batchSize": 3},
            schema_version=1,
        )


def test_ai_batch_provider_settings_have_no_fixed_upper_bound() -> None:
    assert validate_provider_setting_payload(
        "hq",
        "custom",
        {"batchSize": 128},
        schema_version=1,
    ) == {"batchSize": 128}

    assert validate_provider_setting_payload(
        "proofreading_11111111-1111-4111-8111-111111111111",
        "custom",
        {"batchSize": 256},
        schema_version=1,
    ) == {"batchSize": 256}


def test_credentials_require_the_current_domain_provider_identity() -> None:
    with pytest.raises(ValueError, match="does not support"):
        validate_credential_secret(
            "ai_vision_ocr",
            "deepseek",
            {"ai_vision_api_key": "secret"},
        )
    with pytest.raises(ValueError, match="unsupported credential"):
        validate_credential_secret(
            "web_import_firecrawl",
            "custom",
            {"api_key": "secret"},
        )
    with pytest.raises(ValueError, match="HTTP credential is invalid"):
        validate_credential_secret(
            "web_import_http",
            "headers",
            {"headers": {"Referer": ""}},
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_credential_secret(
            "translation",
            "custom",
            {"api_key": "   "},
        )


@pytest.mark.parametrize(
    ("domain", "provider", "field", "value"),
    [
        ("insight_embedding", "openai", "rpmLimit", -1),
        ("insight_embedding", "openai", "rpmLimit", 1.5),
        ("insight_embedding", "openai", "transportRetries", -1),
        ("insight_vlm", "gemini", "imageMaxSize", -1),
        ("insight_image_gen", "gpt2api", "timeoutSeconds", -0.5),
    ],
)
def test_provider_numeric_settings_reject_invalid_values(
    domain: str,
    provider: str,
    field: str,
    value: float,
) -> None:
    payload = _current_insight_provider_payload(domain)
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        validate_provider_setting_payload(
            domain,
            provider,
            payload,
            schema_version=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("auxYoloConfThreshold", 1.01, "must be from 0 to 1"),
        ("auxYoloOverlapThreshold", -0.01, "must be from 0 to 1"),
        (
            "saberYoloRefineOverlapThreshold",
            101,
            "must be from 0 to 100",
        ),
    ],
)
def test_translation_detection_thresholds_use_one_current_unit(
    field: str,
    value: float,
    message: str,
) -> None:
    payload = default_translation_settings()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_setting_payload("translation", payload, schema_version=6)


def test_translation_settings_validate_paddleocr_vl_prompt_language() -> None:
    payload = default_translation_settings()
    assert payload["paddleOcrVl"] == {"sourceLanguage": "japanese"}

    payload["paddleOcrVl"]["sourceLanguage"] = "unsupported"
    with pytest.raises(ValueError, match="paddleOcrVl.sourceLanguage"):
        validate_setting_payload("translation", payload, schema_version=6)


def test_factory_translation_defaults_match_algorithm_prompt_protocols() -> None:
    payload = default_translation_settings()
    translation = payload["translation"]

    assert translation["batchNormalPrompt"] == (
        shared_constants.BATCH_TRANSLATE_SYSTEM_TEMPLATE
    )
    assert translation["batchJsonPrompt"] == (
        shared_constants.BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE
    )
    assert translation["singleNormalPrompt"] == shared_constants.DEFAULT_PROMPT
    assert translation["singleJsonPrompt"] == (
        shared_constants.DEFAULT_TRANSLATE_JSON_PROMPT
    )
    assert payload["aiVisionOcr"]["prompt"] == (
        shared_constants.DEFAULT_AI_VISION_OCR_PROMPT
    )
    assert payload["hqTranslation"]["prompt"] == (
        shared_constants.DEFAULT_HQ_TRANSLATE_PROMPT
    )


def test_translation_settings_reject_nullable_browser_temperature() -> None:
    payload = default_translation_settings()
    payload["translation"]["openaiOptions"]["request"]["temperature"] = None

    with pytest.raises(ValueError, match="temperature must be from 0 to 2"):
        validate_setting_payload("translation", payload, schema_version=6)


def test_parallel_deep_learning_concurrency_has_no_arbitrary_upper_gate() -> None:
    payload = default_translation_settings()
    payload["parallel"]["deepLearningLockSize"] = 8

    validated = validate_setting_payload("translation", payload, schema_version=6)

    assert validated["parallel"]["deepLearningLockSize"] == 8


def test_ai_translation_batch_sizes_have_no_fixed_upper_bound() -> None:
    payload = default_translation_settings()
    payload["hqTranslation"]["batchSize"] = 128
    payload["proofreading"]["rounds"] = [
        {
            **payload["hqTranslation"],
            "id": "11111111-1111-4111-8111-111111111111",
            "name": "第1轮",
            "batchSize": 256,
        }
    ]

    validated = validate_setting_payload("translation", payload, schema_version=6)

    assert validated["hqTranslation"]["batchSize"] == 128
    assert validated["proofreading"]["rounds"][0]["batchSize"] == 256


def test_translation_settings_require_unique_proofreading_round_ids() -> None:
    payload = default_translation_settings()
    round_id = "11111111-1111-4111-8111-111111111111"
    payload["proofreading"]["rounds"] = [
        {**payload["hqTranslation"], "id": round_id, "name": "第1轮"},
        {**payload["hqTranslation"], "id": round_id, "name": "第2轮"},
    ]

    with pytest.raises(ValueError, match="unique IDs"):
        validate_setting_payload("translation", payload, schema_version=6)


def test_translation_settings_drop_the_unused_global_proofreading_retry() -> None:
    payload = default_translation_settings()

    assert payload["settingsSchemaVersion"] == 6
    assert set(payload["proofreading"]) == {"enabled", "rounds"}

    retired = deepcopy(payload)
    retired["proofreading"]["maxRetries"] = 2
    with pytest.raises(ValueError, match="invalid fields"):
        validate_setting_payload("translation", retired, schema_version=6)

    with pytest.raises(ValueError, match="schema version must be 6"):
        validate_setting_payload("translation", payload, schema_version=5)


def test_text_style_defaults_reject_the_legacy_schema_version() -> None:
    with pytest.raises(ValueError, match="schema version must be 2"):
        validate_setting_payload(
            "text_style_defaults",
            DEFAULT_TEXT_STYLE,
            schema_version=1,
        )


def test_removing_middle_proofreading_round_prunes_only_current_provider_setting(
    platform,
) -> None:
    _data_root, engine = platform
    repository = SettingsRepository(engine)
    round_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        "33333333-3333-4333-8333-333333333333",
    )
    domains = tuple(f"proofreading_{round_id}" for round_id in round_ids)
    payload = default_translation_settings()
    payload["proofreading"] = {
        "enabled": True,
        "rounds": [
            {
                **payload["hqTranslation"],
                "id": round_id,
                "name": f"第{index + 1}轮",
                "provider": "custom",
                "modelName": f"proof-model-{index + 1}",
            }
            for index, round_id in enumerate(round_ids)
        ],
    }
    repository.save_transaction(
        settings=(SettingMutation("translation", payload, 0, 6),),
        credentials_edits=tuple(
            CredentialEdit(
                domain=domain,
                provider="custom",
                secret={"api_key": f"secret-{index + 1}"},
                base_revision=0,
                client_ref=domain,
            )
            for index, domain in enumerate(domains)
        ),
        providers=tuple(
            ProviderSettingMutation(
                domain=domain,
                provider="custom",
                payload={"modelName": f"proof-model-{index + 1}"},
                base_revision=0,
                schema_version=1,
                credential_edit_ref=domain,
            )
            for index, domain in enumerate(domains)
        ),
    )
    stale_credential = next(
        row for row in repository.credential_summaries()
        if row["domain"] == domains[1]
    )

    updated = deepcopy(payload)
    updated["proofreading"]["rounds"].pop(1)
    repository.save_transaction(
        settings=(SettingMutation("translation", updated, 1, 6),),
    )

    loaded = repository.load()
    assert {
        row["domain"]
        for row in loaded["providerSettings"]
        if str(row["domain"]).startswith("proofreading_")
    } == {domains[0], domains[2]}
    assert {
        row["domain"]
        for row in loaded["credentials"]
        if str(row["domain"]).startswith("proofreading_")
    } == set(domains)
    assert repository.resolve_secret(
        str(stale_credential["credentialVersionId"])
    ) == {"api_key": "secret-2"}


def test_settings_load_rejects_noncurrent_persisted_schema_versions(
    platform,
) -> None:
    _data_root, engine = platform
    repository = SettingsRepository(engine)
    repository.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=default_translation_settings(),
                base_revision=0,
                schema_version=6,
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={"modelName": "current-model"},
                base_revision=0,
                schema_version=1,
            ),
        ),
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE app_settings SET schema_version = 2 "
                "WHERE domain = 'translation'"
            )
        )
    with pytest.raises(ValueError, match="translation settings schema version"):
        repository.load(domains=("translation",))

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE app_settings SET schema_version = 6 "
                "WHERE domain = 'translation'"
            )
        )
        connection.execute(
            text(
                "UPDATE provider_settings SET schema_version = 2 "
                "WHERE domain = 'translation' AND provider = 'custom'"
            )
        )
    with pytest.raises(ValueError, match="provider setting schema version"):
        repository.load(domains=("translation",))

    book_id = "current-schema-book"
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id=book_id, kind="library", title="Book")
        )
        connection.execute(
            text(
                "UPDATE provider_settings SET schema_version = 1 "
                "WHERE domain = 'translation' AND provider = 'custom'"
            )
        )
    repository.save_transaction(
        book_settings_edits=(
            BookSettingMutation(
                book_id=book_id,
                domain="insight",
                payload=deepcopy(DEFAULT_INSIGHT_SETTINGS),
                base_revision=0,
                schema_version=1,
            ),
        ),
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE book_settings SET schema_version = 2 "
                "WHERE book_id = :book_id AND domain = 'insight'"
            ),
            {"book_id": book_id},
        )
    with pytest.raises(ValueError, match="book setting schema version"):
        repository.load(domains=("insight",), book_id=book_id)


def test_settings_load_uses_one_consistent_read_snapshot(platform) -> None:
    data_root, engine = platform
    repository = SettingsRepository(engine)
    payload = default_translation_settings()
    payload["translation"]["provider"] = "custom"
    repository.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=0,
                schema_version=6,
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={"modelName": "old-model"},
                base_revision=0,
                schema_version=1,
            ),
        ),
    )
    updated_payload = deepcopy(payload)
    updated_payload["targetLanguage"] = "en"
    triggered = False

    def update_between_reads(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany: bool,
    ) -> None:
        nonlocal triggered
        if triggered or "FROM app_settings" not in statement:
            return
        triggered = True
        with sqlite3.connect(data_root / "saber.sqlite3", timeout=5) as writer:
            writer.execute("PRAGMA busy_timeout=5000")
            writer.execute(
                "UPDATE app_settings SET revision = 2, payload_json = ? "
                "WHERE domain = 'translation'",
                (json.dumps(updated_payload, separators=(",", ":")),),
            )
            writer.execute(
                "UPDATE provider_settings SET revision = 2, payload_json = ? "
                "WHERE domain = 'translation' AND provider = 'custom'",
                (json.dumps({"modelName": "new-model"}),),
            )

    event.listen(engine, "after_cursor_execute", update_between_reads)
    try:
        loaded = repository.load(domains=("translation",))
    finally:
        event.remove(engine, "after_cursor_execute", update_between_reads)

    assert triggered is True
    assert loaded["settings"][0]["revision"] == 1
    assert loaded["settings"][0]["payload"]["targetLanguage"] == "zh"
    assert loaded["providerSettings"][0]["revision"] == 1
    assert loaded["providerSettings"][0]["payload"]["modelName"] == "old-model"


def test_second_launcher_is_rejected_for_the_same_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data-v2"
    (data_root / "runtime").mkdir(parents=True)
    first = DataRootLock(data_root)
    second = DataRootLock(data_root)
    first.acquire()
    try:
        with pytest.raises(DataRootAlreadyLocked):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_process_epoch_registration_allows_unbound_pid_and_rejects_negative(
    platform,
) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration("unbound-api", "token", "api", 0)

    repository.register(registration)

    with engine.connect() as connection:
        assert connection.execute(
            select(process_epochs.c.pid).where(process_epochs.c.id == "unbound-api")
        ).scalar_one() == 0
    with pytest.raises(ValueError, match="non-negative"):
        repository.register(EpochRegistration("invalid-api", "token", "api", -1))


def test_active_epoch_processes_only_returns_current_role(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine)
    repository.register(EpochRegistration("api-active", "api-token", "api", 101))
    repository.register(EpochRegistration("worker-active", "worker-token", "worker", 202))

    assert repository.active_epoch_processes("api") == [("api-active", 101)]
    assert repository.active_epoch_processes("worker") == [("worker-active", 202)]


def test_worker_recovery_is_idempotent_and_preserves_chapter_lock(platform) -> None:
    _data_root, engine = platform
    seed_system_records(engine)
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration("worker-epoch", "worker-token", "worker", 123)
    repository.register(registration)
    now = utcnow()
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="book", kind="library", title="Book")
        )
        connection.execute(
            insert(chapters).values(
                id="chapter",
                book_id="book",
                ordinal=1,
                title="Chapter",
            )
        )
        connection.execute(
            insert(jobs).values(
                id="job",
                kind="translation",
                status="running",
                chapter_id="chapter",
                config_json="{}",
                latest_progress_json=_stored_job_progress("running"),
                worker_epoch_id=registration.epoch_id,
                attempt_id="attempt",
                lease_token="attempt-token",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(chapter_write_locks).values(
                chapter_id="chapter",
                job_id="job",
                lock_generation=1,
                owner_attempt_id="attempt",
                lease_token="attempt-token",
            )
        )
        connection.execute(
            insert(chapter_write_intents).values(
                chapter_id="chapter",
                job_id="job",
                intent_set_id="intent-set",
                intent_generation=1,
                worker_epoch_id=registration.epoch_id,
                lease_token="worker-token",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )

    first = repository.reconcile_dead_worker(registration.epoch_id)
    second = repository.reconcile_dead_worker(registration.epoch_id)
    assert first.changed and first.jobs_interrupted == 1 and first.intents_removed == 1
    assert not second.changed
    with engine.connect() as connection:
        job = connection.execute(
            select(
                jobs.c.status,
                jobs.c.attempt_id,
                jobs.c.latest_progress_json,
            ).where(jobs.c.id == "job")
        ).one()
        lock_count = connection.execute(
            select(text("COUNT(*)")).select_from(chapter_write_locks)
        ).scalar_one()
    assert job[:2] == ("interrupted", None)
    assert json.loads(job.latest_progress_json)["jobStatus"] == "interrupted"
    assert lock_count == 1


@pytest.mark.parametrize(
    "initial_status,expected_status,lock_is_retained",
    [
        ("pausing", "interrupted", True),
        ("cancelling", "cancelled", False),
    ],
)
def test_worker_recovery_resolves_drain_transition_states(
    platform,
    initial_status: str,
    expected_status: str,
    lock_is_retained: bool,
) -> None:
    _data_root, engine = platform
    seed_system_records(engine)
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration(
        f"worker-{initial_status}",
        f"token-{initial_status}",
        "worker",
        456,
    )
    repository.register(registration)
    now = utcnow()
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="book", kind="library", title="Book")
        )
        connection.execute(
            insert(chapters).values(
                id="chapter",
                book_id="book",
                ordinal=1,
                title="Chapter",
            )
        )
        connection.execute(
            insert(jobs).values(
                id="job",
                kind="translation",
                status=initial_status,
                chapter_id="chapter",
                config_json="{}",
                latest_progress_json=_stored_job_progress(initial_status),
                worker_epoch_id=registration.epoch_id,
                attempt_id="attempt",
                lease_token="attempt-token",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(job_items).values(
                id=f"item-{initial_status}",
                job_id="job",
                ordinal=1,
                status="running",
            )
        )
        connection.execute(
            insert(job_steps).values(
                id=f"step-{initial_status}",
                job_item_id=f"item-{initial_status}",
                ordinal=1,
                kind="detect",
                status="running",
                attempt_id="attempt",
                checkpoint_schema_version=1,
            )
        )
        connection.execute(
            insert(chapter_write_locks).values(
                chapter_id="chapter",
                job_id="job",
                lock_generation=1,
                owner_attempt_id="attempt",
                lease_token="attempt-token",
            )
        )

    result = repository.reconcile_dead_worker(registration.epoch_id)
    with engine.connect() as connection:
        job = connection.execute(
            select(jobs.c.status, jobs.c.latest_progress_json).where(
                jobs.c.id == "job"
            )
        ).one()
        lock = connection.execute(
            select(chapter_write_locks.c.job_id)
        ).scalar_one_or_none()
        item_status = connection.execute(
            select(job_items.c.status).where(job_items.c.job_id == "job")
        ).scalar_one()
        step = connection.execute(
            select(job_steps.c.status, job_steps.c.attempt_id).where(
                job_steps.c.job_item_id == f"item-{initial_status}"
            )
        ).one()
        recovery_event = connection.execute(
            select(job_events.c.payload_json).where(
                job_events.c.job_id == "job",
                job_events.c.event_type == f"job_{expected_status}",
            )
        ).scalar_one()
    progress = json.loads(job.latest_progress_json)
    assert job.status == expected_status
    assert progress["jobStatus"] == expected_status
    assert json.loads(recovery_event)["progress"] == progress
    assert (lock is not None) is lock_is_retained
    expected_graph_status = (
        "cancelled" if expected_status == "cancelled" else "pending"
    )
    assert item_status == expected_graph_status
    assert step == (expected_graph_status, None)
    assert result.jobs_interrupted == int(expected_status == "interrupted")
    assert result.jobs_cancelled == int(expected_status == "cancelled")


def test_worker_recovery_requeues_operation_with_a_durable_event(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration(
        "worker-operation-epoch",
        "worker-operation-token",
        "worker",
        654,
    )
    repository.register(registration)
    now = utcnow()
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="book", kind="library", title="Book")
        )
        connection.execute(
            insert(chapters).values(
                id="chapter", book_id="book", ordinal=1, title="Chapter"
            )
        )
        connection.execute(
            insert(pages).values(
                id="page",
                chapter_id="chapter",
                ordinal=1,
                logical_source_path="page.png",
            )
        )
        connection.execute(
            insert(operations).values(
                id="worker-operation",
                kind="page_detect",
                executor_role="worker",
                status="running",
                page_id="page",
                base_revision=1,
                request_json="{}",
                executor_epoch_id=registration.epoch_id,
                attempt_id="attempt",
                lease_token="lease",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )

    result = repository.reconcile_dead_worker(registration.epoch_id)

    assert result.operations_requeued == 1
    with engine.connect() as connection:
        operation_status = connection.execute(
            select(operations.c.status).where(
                operations.c.id == "worker-operation"
            )
        ).scalar_one()
        event = connection.execute(
            select(operation_events.c.type, operation_events.c.payload_json).where(
                operation_events.c.operation_id == "worker-operation"
            )
        ).one()
    assert operation_status == "pending"
    assert event.type == "operation_requeued"
    assert json.loads(event.payload_json) == {"reason": "WORKER_EPOCH_LOST"}


def test_api_recovery_fails_remote_work_and_requeues_safe_render(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine)
    registration = EpochRegistration("api-epoch", "api-token", "api", 321)
    repository.register(registration)
    now = utcnow()
    with engine.begin() as connection:
        connection.execute(
            insert(books).values(id="book", kind="library", title="Book")
        )
        connection.execute(
            insert(chapters).values(
                id="chapter", book_id="book", ordinal=1, title="Chapter"
            )
        )
        connection.execute(
            insert(pages).values(
                id="page",
                chapter_id="chapter",
                ordinal=1,
                logical_source_path="page.png",
                render_status="rendering",
            )
        )
        connection.execute(
            insert(pages).values(
                id="repair-page",
                chapter_id="chapter",
                ordinal=2,
                logical_source_path="repair-page.png",
            )
        )
        connection.execute(
            insert(bubbles).values(
                id="bubble",
                page_id="page",
                ordinal=1,
                payload_json="{}",
                updated_revision=1,
            )
        )
        connection.execute(
            insert(operations).values(
                id="operation",
                kind="bubble_translate",
                executor_role="api",
                status="running",
                page_id="page",
                bubble_id="bubble",
                base_revision=1,
                request_json="{}",
                executor_epoch_id=registration.epoch_id,
                attempt_id="attempt",
                lease_token="lease",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(render_requests).values(
                id="render",
                page_id="page",
                requested_revision=1,
                rendering_revision=1,
                status="running",
                executor_epoch_id=registration.epoch_id,
                attempt_id="render-attempt",
                lease_token="render-lease",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(operations).values(
                id="safe-operation",
                kind="page_repair",
                executor_role="api",
                status="running",
                page_id="repair-page",
                base_revision=1,
                request_json="{}",
                executor_epoch_id=registration.epoch_id,
                attempt_id="safe-attempt",
                lease_token="safe-lease",
                lease_expires_at=now + timedelta(minutes=1),
            )
        )

    result = repository.reconcile_dead_api(registration.epoch_id)
    assert result.operations_failed == 1
    assert result.operations_requeued == 1
    assert result.renders_requeued == 1
    with engine.connect() as connection:
        operation = connection.execute(
            select(operations.c.status, operations.c.error_json).where(
                operations.c.id == "operation"
            )
        ).one()
        render = connection.execute(
            select(render_requests.c.status, render_requests.c.rendering_revision).where(
                render_requests.c.id == "render"
            )
        ).one()
        page_render_status = connection.execute(
            select(pages.c.render_status).where(pages.c.id == "page")
        ).scalar_one()
        safe_operation_status = connection.execute(
            select(operations.c.status).where(
                operations.c.id == "safe-operation"
            )
        ).scalar_one()
        events = {
            str(row.operation_id): (
                str(row.type),
                json.loads(str(row.payload_json)),
            )
            for row in connection.execute(
                select(
                    operation_events.c.operation_id,
                    operation_events.c.type,
                    operation_events.c.payload_json,
                ).where(
                    operation_events.c.operation_id.in_(
                        ("operation", "safe-operation")
                    )
                )
            )
        }
    assert operation.status == "failed"
    assert json.loads(operation.error_json)["code"] == "API_EXECUTOR_LOST"
    assert render == ("pending", None)
    assert page_render_status == "stale"
    assert safe_operation_status == "pending"
    assert events["operation"][0] == "operation_failed"
    assert events["operation"][1]["error"]["code"] == "API_EXECUTOR_LOST"
    assert events["safe-operation"] == (
        "operation_requeued",
        {"reason": "API_EPOCH_LOST"},
    )


def test_expired_or_replaced_epoch_cannot_be_renewed(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine, lease_seconds=3)
    registration = EpochRegistration("worker", "secret", "worker", 123)
    repository.register(registration)
    assert repository.renew(role="worker", epoch_id="worker", token="secret")
    expired_at = utcnow() - timedelta(seconds=1)
    with engine.begin() as connection:
        connection.execute(
            update(process_epochs)
            .where(process_epochs.c.id == "worker")
            .values(lease_expires_at=expired_at)
        )
        connection.execute(
            update(worker_leases)
            .where(worker_leases.c.worker_epoch_id == "worker")
            .values(lease_expires_at=expired_at)
        )
    assert not repository.is_active_epoch(role="worker", epoch_id="worker")
    assert not repository.renew(role="worker", epoch_id="worker", token="secret")
    with engine.begin() as connection:
        connection.execute(
            process_epochs.update()
            .where(process_epochs.c.id == "worker")
            .values(status="lost")
        )
    assert not repository.renew(role="worker", epoch_id="worker", token="secret")
    assert not repository.renew(role="worker", epoch_id="worker", token="wrong")


def test_launcher_epoch_tokens_are_never_persisted_in_plaintext(platform) -> None:
    _data_root, engine = platform
    repository = ProcessEpochRepository(engine, lease_seconds=3)
    registrations = (
        EpochRegistration("worker-secret-epoch", "worker-secret", "worker", 123),
        EpochRegistration("api-secret-epoch", "api-secret", "api", 321),
    )
    for registration in registrations:
        repository.register(registration)

    with engine.connect() as connection:
        epoch_tokens = {
            str(row.id): str(row.token_hash)
            for row in connection.execute(
                select(process_epochs.c.id, process_epochs.c.token_hash).where(
                    process_epochs.c.id.in_(
                        registration.epoch_id for registration in registrations
                    )
                )
            )
        }
        worker_token = connection.execute(
            select(worker_leases.c.token_hash).where(
                worker_leases.c.worker_epoch_id == "worker-secret-epoch"
            )
        ).scalar_one()
        api_token = connection.execute(
            select(api_executor_leases.c.token_hash).where(
                api_executor_leases.c.api_epoch_id == "api-secret-epoch"
            )
        ).scalar_one()

    assert epoch_tokens == {
        "worker-secret-epoch": hash_epoch_token("worker-secret"),
        "api-secret-epoch": hash_epoch_token("api-secret"),
    }
    assert worker_token == hash_epoch_token("worker-secret")
    assert api_token == hash_epoch_token("api-secret")
    assert repository.renew(
        role="worker",
        epoch_id="worker-secret-epoch",
        token="worker-secret",
    )
    assert repository.renew(
        role="api",
        epoch_id="api-secret-epoch",
        token="api-secret",
    )


@pytest.mark.parametrize(
    "crash_point,committed",
    [
        ("staging_fsynced", False),
        ("journal_staged", False),
        ("file_published", False),
        ("journal_file_published", False),
        ("database_before_commit", False),
        ("database_committed", True),
    ],
)
def test_asset_publication_failure_windows_are_recoverable(
    platform,
    crash_point: str,
    committed: bool,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)

    def crash(point: str) -> None:
        if point == crash_point:
            raise RuntimeError(f"injected crash at {point}")

    with pytest.raises(RuntimeError, match=crash_point):
        storage.publish_bytes(
            f"payload-{crash_point}".encode(),
            extension="bin",
            mime_type="application/octet-stream",
            failpoint=crash,
        )

    storage.recover_journal(orphan_grace_seconds=0)
    with engine.connect() as connection:
        stored_assets = list(
            connection.execute(select(assets)).mappings()
        )
        assert connection.execute(select(object_commit_journal)).all() == []
    assert len(stored_assets) == int(committed)
    assert not list((data_root / "temp" / "staging").glob("*.part"))
    object_files = [
        path
        for path in (data_root / "objects").rglob("*")
        if path.is_file()
    ]
    assert len(object_files) == int(committed)
    if committed:
        stored_path = storage.resolve_relative_path(
            str(stored_assets[0]["relative_path"])
        )
        assert stored_path.read_bytes() == (
            f"payload-{crash_point}".encode()
        )


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (1, None, "provided together"),
        (None, 1, "provided together"),
        (0, 1, "must be positive"),
        (1, 0, "must be positive"),
    ],
)
def test_asset_publication_rejects_incomplete_or_invalid_dimensions(
    platform,
    width: int | None,
    height: int | None,
    message: str,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)

    with pytest.raises(ValueError, match=message):
        storage.publish_bytes(
            b"payload",
            extension="png",
            mime_type="image/png",
            width=width,
            height=height,
        )


def test_integrity_scan_and_two_pass_gc_never_delete_referenced_assets(
    platform,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    unreferenced = storage.publish_bytes(
        b"unused", extension="bin", mime_type="application/octet-stream"
    )
    referenced = storage.publish_bytes(
        b"font", extension="ttf", mime_type="font/ttf"
    )
    FontRepository(engine).register_uploaded(
        asset_id=referenced.id,
        display_name="Uploaded",
    )
    storage.resolve_relative_path(referenced.relative_path).unlink()
    scan = storage.scan_integrity()
    assert scan.missing == 1

    first = storage.collect_garbage(grace_seconds=10, now=utcnow())
    second = storage.collect_garbage(
        grace_seconds=10,
        now=utcnow() + timedelta(seconds=11),
    )
    assert first.marked == 1
    assert second.deleted_rows == 1
    with engine.connect() as connection:
        remaining = set(connection.execute(select(assets.c.id)).scalars())
    assert unreferenced.id not in remaining
    assert referenced.id in remaining


def test_asset_gc_limits_each_mark_and_delete_batch(platform) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    asset_ids = {
        storage.publish_bytes(
            f"unused-{index}".encode(),
            extension="bin",
            mime_type="application/octet-stream",
        ).id
        for index in range(3)
    }
    started_at = utcnow()

    first = storage.collect_garbage(
        grace_seconds=10,
        now=started_at,
        batch_limit=2,
    )
    second = storage.collect_garbage(
        grace_seconds=10,
        now=started_at,
        batch_limit=2,
    )
    third = storage.collect_garbage(
        grace_seconds=10,
        now=started_at + timedelta(seconds=11),
        batch_limit=2,
    )
    fourth = storage.collect_garbage(
        grace_seconds=10,
        now=started_at + timedelta(seconds=11),
        batch_limit=2,
    )

    assert first.marked == 2
    assert second.marked == 1
    assert third.deleted_rows == 2
    assert fourth.deleted_rows == 1
    with engine.connect() as connection:
        remaining = set(
            connection.execute(
                select(assets.c.id).where(assets.c.id.in_(asset_ids))
            ).scalars()
        )
    assert remaining == set()


def test_asset_gc_does_not_request_a_write_lock_when_nothing_can_change(
    platform,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    referenced = storage.publish_bytes(
        b"font",
        extension="ttf",
        mime_type="font/ttf",
    )
    FontRepository(engine).register_uploaded(
        asset_id=referenced.id,
        display_name="Referenced",
    )

    with immediate_transaction(engine):
        result = storage.collect_garbage(batch_limit=1)

    assert result.marked == 0
    assert result.deleted_rows == 0


def test_orphan_object_reconciliation_honors_database_journal_and_grace(
    platform,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    referenced = storage.publish_bytes(
        b"referenced",
        extension="bin",
        mime_type="application/octet-stream",
    )
    old_orphan = data_root / "objects" / "orphan.bin"
    young_orphan = data_root / "objects" / "young.bin"
    old_orphan.write_bytes(b"old")
    young_orphan.write_bytes(b"young")
    old_timestamp = (utcnow() - timedelta(hours=2)).timestamp()
    os.utime(old_orphan, (old_timestamp, old_timestamp))

    result = storage.reconcile_orphan_objects(grace_seconds=3600)

    assert result.scanned == 3
    assert result.deleted == 1
    assert result.protected == 1
    assert result.grace_retained == 1
    assert not old_orphan.exists()
    assert young_orphan.exists()
    assert storage.resolve_relative_path(referenced.relative_path).exists()


def test_consistency_checker_and_cli_report_storage_divergence(
    platform,
) -> None:
    data_root, engine = platform
    storage = AssetStorageService(data_root, engine)
    missing = storage.publish_bytes(
        b"missing",
        extension="bin",
        mime_type="application/octet-stream",
    )
    storage.resolve_relative_path(missing.relative_path).unlink()
    orphan = data_root / "objects" / "orphan.bin"
    orphan.write_bytes(b"orphan")

    report = ConsistencyChecker(
        data_root=data_root,
        engine=engine,
    ).check(include_vectors=False)
    assert report.ok is False
    assert report.missing_asset_files == (missing.id,)
    assert report.integrity_status_mismatches == (missing.id,)
    assert report.orphan_object_files == ("objects/orphan.bin",)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_v2_consistency",
            "--data-dir",
            str(data_root),
            "--skip-vectors",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["missing_asset_files"] == [missing.id]


def test_consistency_cli_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_v2_consistency.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "Check Saber Translator v2 storage consistency" in completed.stdout


def test_worker_maintenance_runs_only_when_due(platform) -> None:
    data_root, engine = platform
    current = [100.0]
    maintenance = WorkerMaintenance(
        data_root=data_root,
        engine=engine,
        interval_seconds=60,
        clock=lambda: current[0],
    )

    assert maintenance.run_if_due(force=True) is True
    assert maintenance.run_if_due() is False
    current[0] += 60
    assert maintenance.run_if_due() is True


def test_worker_maintenance_continues_after_failed_action(
    platform,
    monkeypatch,
) -> None:
    data_root, engine = platform
    maintenance = WorkerMaintenance(
        data_root=data_root,
        engine=engine,
        interval_seconds=60,
    )
    completed: list[str] = []

    def fail_recovery():
        raise RuntimeError("broken journal fixture")

    monkeypatch.setattr(maintenance.storage, "recover_journal", fail_recovery)
    monkeypatch.setattr(
        maintenance.vector_store,
        "collect_orphan_collections",
        lambda _engine: completed.append("vector_gc"),
    )

    assert maintenance.run_if_due(force=True) is True
    assert completed == ["vector_gc"]


def test_settings_credentials_plugins_fonts_and_shared_limiter(platform) -> None:
    _data_root, engine = platform
    settings = SettingsRepository(engine)
    translation_payload = default_translation_settings()
    translation_payload["translation"]["provider"] = "custom"
    result = settings.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=translation_payload,
                base_revision=0,
                schema_version=6,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "never-return-me"},
                base_revision=0,
                client_ref="translation-fake",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={"modelName": "fake-model"},
                base_revision=0,
                schema_version=1,
                credential_edit_ref="translation-fake",
            ),
        ),
    )
    credential_summary = result["credentials"][0]
    assert credential_summary["hasKey"] is True
    assert "never-return-me" not in json.dumps(result)
    assert "secret" not in json.dumps(settings.credential_summaries()).lower()

    credential_id = str(credential_summary["credentialId"])
    with engine.connect() as connection:
        version_id = connection.execute(
            select(credential_versions.c.id).where(
                credential_versions.c.credential_id == credential_id
            )
        ).scalar_one()
    assert settings.resolve_secret(version_id) == {"api_key": "never-return-me"}
    assert settings.resolve_provider_secret(
        domain="translation",
        provider="custom",
    ) == {"api_key": "never-return-me"}

    with pytest.raises(RevisionConflict, match="already exists"):
        settings.save_transaction(
            credentials_edits=(
                CredentialEdit(
                    domain="translation",
                    provider="custom",
                    secret={"api_key": "replacement-without-current-id"},
                    base_revision=0,
                    client_ref="invalid-replacement",
                ),
            ),
        )
    loaded = settings.load(domains=("translation",))
    assert loaded["providerSettings"] == [
        {
            "domain": "translation",
            "provider": "custom",
            "revision": 1,
            "schemaVersion": 1,
            "credentialVersionId": version_id,
            "payload": {"modelName": "fake-model"},
        }
    ]

    idempotent_body = {
        "settings": [
            {
                "domain": "workflow_preferences",
                "payload": {
                    "rememberWorkflowModeEnabled": True,
                    "lastWorkflowMode": "hq-batch",
                },
                "baseRevision": 0,
            }
        ]
    }
    first, first_replayed = settings.save_transaction_idempotent(
        idempotency_key="settings-save-1",
        request_body=idempotent_body,
        settings=(
            SettingMutation(
                domain="workflow_preferences",
                payload={
                    "rememberWorkflowModeEnabled": True,
                    "lastWorkflowMode": "hq-batch",
                },
                base_revision=0,
                schema_version=1,
            ),
        ),
    )
    second, second_replayed = settings.save_transaction_idempotent(
        idempotency_key="settings-save-1",
        request_body=idempotent_body,
        settings=(
            SettingMutation(
                domain="workflow_preferences",
                payload={
                    "rememberWorkflowModeEnabled": True,
                    "lastWorkflowMode": "hq-batch",
                },
                base_revision=0,
                schema_version=1,
            ),
        ),
    )
    assert first == second
    assert first_replayed is False
    assert second_replayed is True

    with pytest.raises(RevisionConflict):
        settings.save_transaction(
            settings=(
                SettingMutation(
                    domain="translation",
                    payload=translation_payload,
                    base_revision=0,
                    schema_version=6,
                ),
            ),
            credentials_edits=(
                CredentialEdit(
                    domain="hq",
                    provider="custom",
                    secret={"api_key": "must-rollback"},
                    base_revision=0,
                ),
            ),
        )
    with engine.connect() as connection:
        assert connection.execute(select(credentials.c.id)).scalars().all() == [
            credential_id
        ]

    limiter = ProviderRateLimiter(engine)
    first = limiter.acquire(
        provider="fake",
        credential_version_id=version_id,
        rpm_limit=1,
    )
    second = limiter.acquire(
        provider="fake",
        credential_version_id=version_id,
        rpm_limit=5,
    )
    assert first.allowed and not second.allowed and second.retry_after_seconds > 0


def test_provider_rate_limiter_retries_transient_sqlite_busy(
    platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _data_root, engine = platform
    limiter = ProviderRateLimiter(engine)
    saved = SettingsRepository(engine).save_transaction(
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "rate-limit-test-key"},
                base_revision=0,
            ),
        ),
    )
    credential_version_id = str(saved["credentials"][0]["credentialVersionId"])
    original_begin = engine.begin
    attempts = 0

    def flaky_begin():
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise sqlite3.OperationalError("database is locked")
        return original_begin()

    monkeypatch.setattr(engine, "begin", flaky_begin)
    monkeypatch.setattr(
        "src.backend_v2.storage.platform_repositories.time.sleep",
        lambda _seconds: None,
    )

    decision = limiter.acquire(
        provider="custom",
        credential_version_id=credential_version_id,
        rpm_limit=10,
    )

    assert decision.allowed is True
    assert attempts == 3


def test_provider_rate_limiter_stops_after_finite_sqlite_busy_retries(
    platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _data_root, engine = platform
    limiter = ProviderRateLimiter(engine)
    attempts = 0

    def locked_begin():
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "begin", locked_begin)
    monkeypatch.setattr(
        "src.backend_v2.storage.platform_repositories.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        limiter.acquire(
            provider="custom",
            credential_version_id="00000000-0000-4000-8000-000000000001",
            rpm_limit=10,
        )

    assert attempts == 3


def test_settings_http_rejects_unknown_transaction_fields(platform) -> None:
    data_root, engine = platform
    app = Flask("settings-strict-contract-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()

    top_level = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "settings-extra-top-level"},
        json={
            "settings": [],
            "bookSettings": [],
            "providerSettings": [],
            "credentialEdits": [],
            "legacySettings": [],
        },
    )
    assert top_level.status_code == 422
    assert "legacySettings" in top_level.get_data(as_text=True)

    nested = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "settings-extra-nested"},
        json={
            "settings": [
                {
                    "domain": "workflow_preferences",
                    "payload": {
                        "rememberWorkflowModeEnabled": True,
                        "lastWorkflowMode": "hq-batch",
                    },
                    "baseRevision": 0,
                    "schemaVersion": 1,
                    "legacyPayload": {},
                }
            ],
            "bookSettings": [],
            "providerSettings": [],
            "credentialEdits": [],
        },
    )
    assert nested.status_code == 422
    assert "legacyPayload" in nested.get_data(as_text=True)


def test_expired_settings_idempotency_key_can_be_reused(platform) -> None:
    _data_root, engine = platform
    settings = SettingsRepository(engine)
    first_body = {
        "settings": [{
            "domain": "workflow_preferences",
            "payload": {
                "rememberWorkflowModeEnabled": True,
                "lastWorkflowMode": "hq-batch",
            },
            "baseRevision": 0,
            "schemaVersion": 1,
        }],
    }
    first, replayed = settings.save_transaction_idempotent(
        idempotency_key="reusable-settings-key",
        request_body=first_body,
        settings=(SettingMutation(
            domain="workflow_preferences",
            payload=first_body["settings"][0]["payload"],
            base_revision=0,
            schema_version=1,
        ),),
    )
    assert replayed is False
    assert first["settings"][0]["revision"] == 1

    with engine.begin() as connection:
        connection.execute(
            update(idempotency_records)
            .where(
                idempotency_records.c.scope == "settings-transaction",
                idempotency_records.c.key == "reusable-settings-key",
            )
            .values(expires_at=utcnow() - timedelta(seconds=1))
        )

    second_body = {
        "settings": [{
            "domain": "workflow_preferences",
            "payload": {
                "rememberWorkflowModeEnabled": False,
                "lastWorkflowMode": "translate-current",
            },
            "baseRevision": 1,
            "schemaVersion": 1,
        }],
    }
    second, replayed = settings.save_transaction_idempotent(
        idempotency_key="reusable-settings-key",
        request_body=second_body,
        settings=(SettingMutation(
            domain="workflow_preferences",
            payload=second_body["settings"][0]["payload"],
            base_revision=1,
            schema_version=1,
        ),),
    )
    assert replayed is False
    assert second["settings"][0]["revision"] == 2


def test_clean_temporary_assets_replays_without_running_twice(
    platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, engine = platform
    recover_calls = 0

    def recover_once(_storage: AssetStorageService) -> int:
        nonlocal recover_calls
        recover_calls += 1
        return 3

    monkeypatch.setattr(AssetStorageService, "recover_journal", recover_once)
    app = Flask("settings-maintenance-idempotency-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()

    first = client.post(
        "/api/v2/maintenance/clean-temp",
        headers={"Idempotency-Key": "clean-temp-once"},
    )
    replayed = client.post(
        "/api/v2/maintenance/clean-temp",
        headers={"Idempotency-Key": "clean-temp-once"},
    )

    assert first.status_code == 200
    assert first.get_json() == {"recovered": 3}
    assert replayed.status_code == 200
    assert replayed.get_json() == first.get_json()
    assert replayed.headers["Idempotency-Replayed"] == "true"
    assert recover_calls == 1


def test_settings_transaction_rolls_back_settings_when_prompt_cas_fails(
    platform,
) -> None:
    _data_root, engine = platform
    settings = SettingsRepository(engine)
    prompt_repository = PromptRepository(engine)
    initial_setting = settings.save_transaction(
        settings=(
            SettingMutation(
                domain="workflow_preferences",
                payload={
                    "rememberWorkflowModeEnabled": True,
                    "lastWorkflowMode": "hq-batch",
                },
                base_revision=0,
                schema_version=1,
            ),
        ),
    )["settings"][0]
    prompt = prompt_repository.create(
        prompt_type="batch_analysis",
        name="Atomic prompt",
        content="before",
    )

    with pytest.raises(RevisionConflict, match="prompt revision changed"):
        settings.save_transaction(
            settings=(
                SettingMutation(
                    domain="workflow_preferences",
                    payload={
                        "rememberWorkflowModeEnabled": False,
                        "lastWorkflowMode": "translate-current",
                    },
                    base_revision=int(initial_setting["revision"]),
                    schema_version=1,
                ),
            ),
            prompt_edits=(
                PromptMutation(
                    prompt_id=str(prompt["id"]),
                    name="Atomic prompt",
                    content="after",
                    base_revision=99,
                ),
            ),
        )

    loaded = settings.load(domains=("workflow_preferences",))
    assert loaded["settings"][0]["payload"] == {
        "rememberWorkflowModeEnabled": True,
        "lastWorkflowMode": "hq-batch",
    }
    current_prompt = prompt_repository.list("batch_analysis")[0]
    assert current_prompt["content"] == "before"
    assert current_prompt["revision"] == 1


def test_prompt_http_update_is_strict_and_returns_the_complete_resource(
    platform,
) -> None:
    data_root, engine = platform
    prompt = PromptRepository(engine).create(
        prompt_type="batch_analysis",
        name="Strict prompt",
        content="before",
    )
    app = Flask("settings-prompt-contract-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()
    prompt_url = f'/api/v2/prompts/{prompt["id"]}'

    invalid_content = client.put(
        prompt_url,
        headers={"Idempotency-Key": "strict-prompt-content"},
        json={
            "name": "Strict prompt",
            "content": {"legacy": "value"},
            "baseRevision": 1,
        },
    )
    assert invalid_content.status_code == 422
    invalid_revision = client.put(
        prompt_url,
        headers={"Idempotency-Key": "strict-prompt-revision"},
        json={
            "name": "Strict prompt",
            "content": "after",
            "baseRevision": True,
        },
    )
    assert invalid_revision.status_code == 422

    updated = client.put(
        prompt_url,
        headers={"Idempotency-Key": "strict-prompt-valid"},
        json={
            "name": "Strict prompt",
            "content": "after",
            "baseRevision": 1,
        },
    )
    assert updated.status_code == 200
    assert updated.get_json() == {
        "id": prompt["id"],
        "type": "batch_analysis",
        "name": "Strict prompt",
        "content": "after",
        "revision": 2,
        "isFactoryDefault": False,
    }
    replayed = client.put(
        prompt_url,
        headers={"Idempotency-Key": "strict-prompt-valid"},
        json={
            "name": "Strict prompt",
            "content": "after",
            "baseRevision": 1,
        },
    )
    assert replayed.status_code == 200
    assert replayed.headers["Idempotency-Replayed"] == "true"
    assert replayed.get_json() == updated.get_json()
    assert PromptRepository(engine).list("batch_analysis")[0]["revision"] == 2


def test_prompt_http_create_replays_without_creating_a_duplicate(platform) -> None:
    data_root, engine = platform
    app = Flask("settings-prompt-create-idempotency-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()
    payload = {
        "type": "batch_analysis",
        "name": "Idempotent prompt",
        "content": "content",
    }
    created = client.post(
        "/api/v2/prompts",
        headers={"Idempotency-Key": "create-prompt-once"},
        json=payload,
    )
    replayed = client.post(
        "/api/v2/prompts",
        headers={"Idempotency-Key": "create-prompt-once"},
        json=payload,
    )
    assert created.status_code == 201
    assert replayed.status_code == 201
    assert replayed.headers["Idempotency-Replayed"] == "true"
    assert replayed.get_json() == created.get_json()
    assert len(PromptRepository(engine).list("batch_analysis")) == 1


def test_prompt_content_has_no_arbitrary_size_gate(platform) -> None:
    _data_root, engine = platform
    content = "长提示词" * 50_001

    prompt = PromptRepository(engine).create(
        prompt_type="batch_analysis",
        name="Large prompt",
        content=content,
    )

    assert prompt["content"] == content


def test_prompt_reset_distinguishes_missing_and_non_factory_resources(platform) -> None:
    _data_root, engine = platform
    prompts = PromptRepository(engine)
    custom = prompts.create(
        prompt_type="batch_analysis",
        name="Custom prompt",
        content="content",
    )

    with pytest.raises(LookupError, match="prompt not found"):
        prompts.reset("00000000-0000-0000-0000-000000000000", base_revision=1)
    with pytest.raises(RevisionConflict, match="only factory prompts"):
        prompts.reset(str(custom["id"]), base_revision=1)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 rate limit exceeded for API key", "服务请求过于频繁，请稍后重试"),
        ("503 upstream unavailable", "服务暂时不可用，请稍后重试"),
        ("401 invalid API key", "API Key 无效或已过期"),
    ],
)
def test_provider_diagnostic_errors_classify_external_failures_before_credentials(
    message: str,
    expected: str,
) -> None:
    assert ProviderDiagnostics._friendly_error(RuntimeError(message)) == expected


def test_settings_http_transaction_updates_prompts_idempotently(platform) -> None:
    data_root, engine = platform
    prompt = PromptRepository(engine).create(
        prompt_type="batch_analysis",
        name="Transaction prompt",
        content="before",
    )
    app = Flask("settings-prompt-transaction-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()
    payload = {
        "promptEdits": [{
            "id": prompt["id"],
            "name": "Transaction prompt",
            "content": "after",
            "baseRevision": 1,
        }],
    }

    saved = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "prompt-transaction"},
        json=payload,
    )
    replayed = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "prompt-transaction"},
        json=payload,
    )

    assert saved.status_code == 200
    assert saved.get_json()["prompts"] == [{
        "id": prompt["id"],
        "type": "batch_analysis",
        "name": "Transaction prompt",
        "content": "after",
        "revision": 2,
        "isFactoryDefault": False,
    }]
    assert replayed.status_code == 200
    assert replayed.headers["Idempotency-Replayed"] == "true"
    assert replayed.get_json() == saved.get_json()
    assert PromptRepository(engine).list("batch_analysis")[0]["revision"] == 2


def test_settings_http_accepts_true_type_collections(platform) -> None:
    data_root, engine = platform
    source_path = next(
        font.path for font in discover_bundled_fonts() if font.file_name == "ALGER.TTF"
    )
    source_font = TTFont(source_path)
    collection = TTCollection()
    collection.fonts = [source_font]
    payload = BytesIO()
    collection.save(payload)
    source_font.close()

    app = Flask("settings-font-collection-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()
    response = client.post(
        "/api/v2/fonts",
        headers={"Idempotency-Key": "upload-font-collection"},
        data={"file": (BytesIO(payload.getvalue()), "custom.ttc")},
        content_type="multipart/form-data",
    )
    replayed = client.post(
        "/api/v2/fonts",
        headers={"Idempotency-Key": "upload-font-collection"},
        data={"file": (BytesIO(payload.getvalue()), "custom.ttc")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert replayed.status_code == 201
    assert replayed.headers["Idempotency-Replayed"] == "true"
    assert replayed.get_json() == response.get_json()
    uploaded = response.get_json()
    assert uploaded == {
        "id": uploaded["id"],
        "kind": "uploaded",
        "displayName": "custom",
        "builtinKey": None,
        "assetUrl": uploaded["assetUrl"],
    }
    uploaded_id = uploaded["id"]
    listed = client.get("/api/v2/fonts").get_json()["items"]
    assert any(
        item["id"] == uploaded_id
        and item["kind"] == "uploaded"
        and item["displayName"] == "custom"
        for item in listed
    )
    assert sum(item["kind"] == "uploaded" for item in listed) == 1


def test_insight_provider_accepts_its_snake_case_openai_wire_contract(
    platform,
) -> None:
    _data_root, engine = platform
    settings = SettingsRepository(engine)
    settings.save_transaction(
        providers=(
            ProviderSettingMutation(
                domain="insight_vlm",
                provider="siliconflow",
                payload={
                    "modelName": "Qwen/Qwen3.6-27B",
                    "customBaseUrl": "",
                    "imageMaxSize": 1280,
                    "openaiOptions": {
                        "request": {
                            "force_json_output": False,
                            "temperature": 0.3,
                            "extra_body": {},
                        },
                        "execution": {
                            "use_stream": True,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
                base_revision=0,
                schema_version=1,
            ),
        ),
    )

    loaded = settings.load(domains=("insight_vlm",))
    assert loaded["providerSettings"][0]["payload"]["openaiOptions"][
        "request"
    ]["force_json_output"] is False


def test_v2_provider_diagnostics_resolve_backend_credentials_and_routes(
    platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, engine = platform
    settings = SettingsRepository(engine)
    settings.save_transaction(
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "stored-only-on-server"},
                base_revision=0,
                client_ref="openai-key",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "gpt-test",
                    "customBaseUrl": "https://example.test/v1",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="openai-key",
            ),
        ),
    )
    diagnostics = ProviderDiagnostics(settings)
    captured: dict[str, object] = {}

    def list_models(request):
        captured["api_key"] = request.api_key
        return [{"id": "gpt-test", "name": "gpt-test"}]

    monkeypatch.setattr(diagnostics.chat, "list_models", list_models)
    assert diagnostics.model_catalog(
        {
            "provider": "custom",
            "baseUrl": "https://example.test/v1",
            "domain": "translation",
        }
    ) == {
        "models": [{"id": "gpt-test", "name": "gpt-test"}],
    }
    assert captured == {"api_key": "stored-only-on-server"}

    with pytest.raises(ValueError, match="exactly: api_key"):
        diagnostics.model_catalog(
            {
                "provider": "openai",
                "secret": {"apiKey": "retired-field-name"},
            }
        )

    monkeypatch.setattr(
        ProviderDiagnostics,
        "model_catalog",
        lambda _self, body: {
            "models": [{"id": str(body["provider"]), "name": "model"}],
        },
    )
    monkeypatch.setattr(
        ProviderDiagnostics,
        "connection_test",
        lambda _self, kind, _body: {
            "success": True,
            "message": f"{kind} ok",
        },
    )
    app = Flask("settings-diagnostics-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()

    catalog = client.post(
        "/api/v2/model-catalog",
        json={"provider": "openai", "domain": "translation"},
    )
    assert catalog.status_code == 200
    assert catalog.get_json()["models"][0]["id"] == "openai"
    tested = client.post(
        "/api/v2/connection-tests/llm",
        json={"provider": "openai", "domain": "translation"},
    )
    assert tested.status_code == 200
    assert tested.get_json() == {"success": True, "message": "llm ok"}
    unsupported = client.post(
        "/api/v2/connection-tests/not-real",
        json={},
    )
    assert unsupported.status_code == 422


def test_provider_diagnostics_enforce_capabilities_and_fatal_memory_errors(
    platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _data_root, engine = platform
    diagnostics = ProviderDiagnostics(SettingsRepository(engine))

    monkeypatch.setattr(
        diagnostics.chat,
        "complete_vision",
        lambda _request: "vision ok",
    )
    assert diagnostics.connection_test(
        "vlm",
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "secret": {"api_key": "test-key"},
        },
    )["success"] is True

    with pytest.raises(ValueError, match="does not support vision"):
        diagnostics.connection_test(
            "ai_vision_ocr",
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "secret": {"ai_vision_api_key": "test-key"},
            },
        )
    with pytest.raises(ValueError, match="does not support translation"):
        diagnostics.connection_test(
            "ai_translate",
            {
                "provider": "qwen",
                "domain": "translation",
                "model": "qwen-test",
                "secret": {"api_key": "test-key"},
            },
        )

    def fail_with_memory(_kind, _body):
        raise MemoryError("diagnostic allocation failed")

    monkeypatch.setattr(diagnostics, "_run_test", fail_with_memory)
    with pytest.raises(MemoryError, match="allocation failed"):
        diagnostics.connection_test("lama_repair", {})


def test_provider_diagnostics_do_not_report_failed_firecrawl_as_success(
    platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _data_root, engine = platform
    diagnostics = ProviderDiagnostics(SettingsRepository(engine))

    class FailedResponse:
        def raise_for_status(self) -> None:
            raise RuntimeError("Firecrawl status 500")

    monkeypatch.setattr(
        "src.backend_v2.settings.diagnostics.httpx.get",
        lambda *_args, **_kwargs: FailedResponse(),
    )
    result = diagnostics.connection_test(
        "firecrawl",
        {"secret": {"api_key": "test-key"}},
    )
    assert result == {
        "success": False,
        "message": "服务暂时不可用，请稍后重试",
    }


def test_provider_diagnostics_reject_retired_credential_id(platform) -> None:
    _data_root, engine = platform
    diagnostics = ProviderDiagnostics(SettingsRepository(engine))
    with pytest.raises(ValueError, match="credentialId"):
        diagnostics.model_catalog(
            {
                "provider": "openai",
                "credentialId": "retired-indirect-secret-selector",
            }
        )


def test_settings_http_transaction_persists_secret_without_returning_it(
    platform,
) -> None:
    data_root, engine = platform
    app = Flask("settings-credential-persistence-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    client = app.test_client()
    secret = "sk-must-never-return-to-browser"

    saved = client.put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "save-translation-credential"},
        json={
            "settings": [],
            "bookSettings": [],
            "providerSettings": [{
                "domain": "translation",
                "provider": "deepseek",
                    "payload": {"modelName": "deepseek-chat"},
                    "baseRevision": 0,
                    "schemaVersion": 1,
                    "credentialEditRef": "translation-deepseek",
                }],
            "credentialEdits": [{
                "domain": "translation",
                "provider": "deepseek",
                "secret": {"api_key": secret},
                "baseRevision": 0,
                "clientRef": "translation-deepseek",
            }],
        },
    )
    assert saved.status_code == 200
    assert secret not in saved.get_data(as_text=True)

    loaded = client.get("/api/v2/settings?domains=translation")
    assert loaded.status_code == 200
    document = loaded.get_json()
    assert secret not in loaded.get_data(as_text=True)
    assert document["credentials"] == [
        {
            "credentialId": document["credentials"][0]["credentialId"],
            "credentialVersionId": document["credentials"][0][
                "credentialVersionId"
            ],
            "currentVersion": 1,
            "domain": "translation",
            "hasKey": True,
            "provider": "deepseek",
            "revision": 1,
        }
    ]
    assert document["providerSettings"][0]["credentialVersionId"] == (
        document["credentials"][0]["credentialVersionId"]
    )
    assert SettingsRepository(engine).resolve_provider_secret(
        domain="translation",
        provider="deepseek",
    ) == {"api_key": secret}


def test_settings_http_rejects_empty_transaction(platform) -> None:
    data_root, engine = platform
    app = Flask("settings-empty-transaction-test")
    app.register_blueprint(
        create_settings_blueprint(data_root=data_root, engine=engine)
    )
    response = app.test_client().put(
        "/api/v2/settings/transactions",
        headers={"Idempotency-Key": "empty-settings-transaction"},
        json={},
    )
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "validation_error"
