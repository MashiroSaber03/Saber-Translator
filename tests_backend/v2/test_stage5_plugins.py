from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import threading
import uuid
import zipfile

import pytest
from sqlalchemy import select

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.plugins.contract import (
    PluginContractError,
    parse_manifest,
    validate_atomic_hook_data,
    validate_hook_source_contract,
)
from src.backend_v2.plugins.package import build_archive, parse_archive
from src.backend_v2.plugins.repository import (
    PluginConflict,
    PluginLocked,
    PluginNotFound,
    PluginRegistry,
)
from src.backend_v2.plugins.runtime import (
    PluginHookFailure,
    PluginJobRuntime,
)
from src.backend_v2.plugins.snapshots import enabled_plugin_snapshots
from src.backend_v2.plugins.agent import PluginAgentSessionService
from src.backend_v2.plugins.agent_worker import (
    PluginAgentWorkerService,
)
from src.backend_v2.operations.repository import OperationRepository
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    assets,
    books,
    chapters,
    job_events,
    job_items,
    job_plugin_snapshots,
    metadata,
    operation_plugin_snapshots,
    page_assets,
    pages,
    jobs as jobs_table,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.core.plugin_agent.controller import PluginAgentController
from src.core.plugin_agent.models import (
    LockedPluginTarget,
    PluginAgentMessage,
    PluginAgentSession,
)
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
)


def _enabled_snapshots(engine):
    with engine.connect() as connection:
        return enabled_plugin_snapshots(connection)


def _plugin_archive(
    *,
    plugin_id: str = "test_v3",
    package_version: str = "1.0.0",
    default_enabled: bool = True,
    hooks: list[str] | None = None,
    supported_steps: list[str] | None = None,
    priority: int = 50,
    failure_policy: str = "continue",
    source: str | None = None,
    config_schema: dict[str, dict[str, object]] | None = None,
) -> bytes:
    declared_hooks = hooks or ["before_translate", "after_translate"]
    manifest = {
        "schema_version": 3,
        "plugin_id": plugin_id,
        "display_name": f"{plugin_id} Plugin",
        "package_version": package_version,
        "entrypoint": "plugin.py:Plugin",
        "hooks": declared_hooks,
        "supported_steps": supported_steps or sorted(
            {hook.split("_", 1)[1] for hook in declared_hooks}
        ),
        "supported_modes": ["standard", "hq"],
        "priority": priority,
        "failure_policy": failure_policy,
        "author": "tests",
        "description": "immutable v3 test plugin",
        "default_enabled": default_enabled,
        "config_schema": (
            config_schema
            if config_schema is not None
            else {
                "prefix": {
                    "type": "text",
                    "default": "[v3]",
                },
                "strict": {
                    "type": "boolean",
                    "default": False,
                },
            }
        ),
    }
    output = BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "plugin.json",
            json.dumps(manifest, ensure_ascii=False),
        )
        archive.writestr(
            "plugin.py",
            source
            or (
                "class Plugin:\n"
                "    def before_translate(self, context, payload):\n"
                "        return payload\n"
                "    def after_translate(self, context, result):\n"
                "        return result\n"
            ),
        )
    return output.getvalue()


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 3,
        "plugin_id": "strict_v3",
        "display_name": "Strict v3",
        "package_version": "1.0.0",
        "entrypoint": "plugin.py:Plugin",
        "hooks": ["after_translate"],
        "supported_steps": ["translate"],
        "supported_modes": ["standard"],
        "priority": 100,
        "failure_policy": "continue",
        "author": "tests",
        "description": "strict current manifest",
        "default_enabled": False,
        "config_schema": {},
    }


@pytest.fixture()
def plugin_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "plugins.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    try:
        yield data_root, engine
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "field",
    (
        "supported_steps",
        "supported_modes",
        "priority",
        "failure_policy",
        "author",
        "description",
        "default_enabled",
        "config_schema",
    ),
)
def test_plugin_manifest_does_not_fill_omitted_v3_fields(field: str) -> None:
    manifest = _valid_manifest()
    del manifest[field]

    with pytest.raises(PluginContractError, match="field mismatch"):
        parse_manifest(manifest)


def test_plugin_manifest_rejects_unknown_fields_and_type_coercion() -> None:
    unknown = _valid_manifest()
    unknown["legacy_mode"] = "standard"
    with pytest.raises(PluginContractError, match="field mismatch"):
        parse_manifest(unknown)

    wrong_type = _valid_manifest()
    wrong_type["supported_steps"] = [123]
    with pytest.raises(PluginContractError, match="must be an array"):
        parse_manifest(wrong_type)


def test_plugin_versions_are_immutable_and_config_is_revisioned(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    archive = _plugin_archive()
    first = registry.import_archive(
        data=archive,
        base_revision=0,
        idempotency_key="install-v1",
    )
    replay = registry.import_archive(
        data=archive,
        base_revision=0,
        idempotency_key="install-v1",
    )
    assert replay == first
    plugin = registry.get_plugin("test_v3")
    assert plugin["runtimeEnabled"] is True
    assert plugin["config"] == {"prefix": "[v3]", "strict": False}
    first_path = (
        data_root
        / "plugins"
        / "test_v3"
        / "versions"
        / str(first["pluginVersionId"])
    )
    assert first_path.is_dir()

    changed = registry.update_config(
        plugin_id="test_v3",
        base_revision=1,
        config={"prefix": "new", "strict": True},
    )
    assert changed["configRevision"] == 2
    with pytest.raises(PluginConflict, match="config revision"):
        registry.update_config(
            plugin_id="test_v3",
            base_revision=1,
            config={"prefix": "stale", "strict": False},
        )

    second = registry.import_archive(
        data=archive,
        base_revision=1,
        idempotency_key="install-v2-build",
    )
    assert second["pluginVersionId"] != first["pluginVersionId"]
    assert first_path.is_dir()
    assert (
        data_root
        / "plugins"
        / "test_v3"
        / "versions"
        / str(second["pluginVersionId"])
    ).is_dir()
    exported, filename = registry.export_current("test_v3")
    assert filename == "test_v3-1.0.0.zip"
    assert parse_archive(exported).manifest.plugin_id == "test_v3"


def test_plugin_upgrade_drops_removed_config_fields_and_seeds_new_defaults(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    registry.import_archive(
        data=_plugin_archive(plugin_id="schema_v3"),
        base_revision=0,
        idempotency_key="schema-v1",
    )
    registry.update_config(
        plugin_id="schema_v3",
        base_revision=1,
        config={"prefix": "obsolete", "strict": True},
    )

    registry.import_archive(
        data=_plugin_archive(
            plugin_id="schema_v3",
            package_version="2.0.0",
            config_schema={
                "strict": {
                    "type": "boolean",
                    "default": False,
                },
                "suffix": {
                    "type": "text",
                    "default": "!",
                },
            },
        ),
        base_revision=1,
        idempotency_key="schema-v2",
    )

    assert registry.get_config("schema_v3")["value"] == {
        "strict": True,
        "suffix": "!",
    }


def test_runtime_default_snapshot_and_reference_lock(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    installed = registry.import_archive(
        data=_plugin_archive(),
        base_revision=0,
        idempotency_key="snapshot-plugin",
    )
    registry.set_runtime_enabled(plugin_id="test_v3", enabled=False)
    assert _enabled_snapshots(engine) == {}
    seed_system_records(engine)
    snapshots = _enabled_snapshots(engine)
    version_id = str(installed["pluginVersionId"])
    assert snapshots[version_id]["pluginId"] == "test_v3"
    assert snapshots[version_id]["config"] == {
        "prefix": "[v3]",
        "strict": False,
    }

    jobs = JobQueueRepository(engine)
    jobs.create_batch(
        kind="export",
        display_name="plugin reference",
        specs=[
            JobSpec(
                kind="export",
                config={},
                items=(
                    JobItemSpec(
                        page_id=None,
                        step_kinds=("export_package",),
                    ),
                ),
            )
        ],
    )
    with pytest.raises(PluginLocked, match="task history"):
        registry.delete_plugin(
            plugin_id="test_v3",
            base_revision=1,
        )
    assert (data_root / "plugins" / "test_v3").is_dir()


def test_refresh_detects_tampering_and_safe_archive_rejects_traversal(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    installed = registry.import_archive(
        data=_plugin_archive(plugin_id="integrity_v3"),
        base_revision=0,
        idempotency_key="integrity-plugin",
    )
    entrypoint = (
        data_root
        / "plugins"
        / "integrity_v3"
        / "versions"
        / str(installed["pluginVersionId"])
        / "plugin.py"
    )
    entrypoint.write_text("# tampered", encoding="utf-8")
    result = registry.refresh()
    assert result["failedVersions"] == 1
    plugin = registry.get_plugin("integrity_v3")
    assert plugin["state"] == "error"
    assert plugin["runtimeEnabled"] is False

    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../plugin.json", "{}")
    with pytest.raises(PluginContractError, match="unsafe path"):
        parse_archive(output.getvalue())


def test_python_runtime_cache_is_not_part_of_immutable_package(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    installed = registry.import_archive(
        data=_plugin_archive(plugin_id="cache_v3"),
        base_revision=0,
        idempotency_key="cache-plugin",
    )
    version_root = (
        data_root
        / "plugins"
        / "cache_v3"
        / "versions"
        / str(installed["pluginVersionId"])
    )
    cache_file = (
        version_root / "__pycache__" / "plugin.cpython-312.pyc"
    )
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"runtime-only")

    assert registry.refresh()["failedVersions"] == 0
    exported = build_archive(version_root)
    with zipfile.ZipFile(BytesIO(exported)) as archive:
        assert "__pycache__/plugin.cpython-312.pyc" not in archive.namelist()

    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("plugin.json", "{}")
        archive.writestr("__pycache__/plugin.cpython-312.pyc", b"cache")
    with pytest.raises(PluginContractError, match="runtime cache"):
        parse_archive(output.getvalue())


def test_plugin_management_http_api_is_metadata_only(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id=str(uuid.uuid4()),
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    client = app.test_client()
    query_revision = client.post(
        "/api/v2/plugins/import?baseRevision=0",
        data={
            "file": (
                BytesIO(_plugin_archive(plugin_id="query_revision")),
                "query_revision.zip",
            ),
        },
        content_type="multipart/form-data",
        headers={"Idempotency-Key": "query-revision-is-not-the-contract"},
    )
    assert query_revision.status_code == 422

    installed = client.post(
        "/api/v2/plugins/import",
        data={
            "baseRevision": "0",
            "file": (
                BytesIO(_plugin_archive(plugin_id="http_v3")),
                "http_v3.zip",
            ),
        },
        content_type="multipart/form-data",
        headers={"Idempotency-Key": "http-plugin-import"},
    )
    assert installed.status_code == 201
    conflict = client.post(
        "/api/v2/plugins/import",
        data={
            "baseRevision": "0",
            "file": (
                BytesIO(_plugin_archive(plugin_id="http_v3")),
                "http_v3.zip",
            ),
        },
        content_type="multipart/form-data",
        headers={"Idempotency-Key": "http-plugin-conflict"},
    )
    assert conflict.status_code == 409
    assert conflict.get_json()["error"]["details"] == {
        "pluginId": "http_v3",
        "currentRevision": 1,
    }
    listing = client.get("/api/v2/plugins")
    assert listing.status_code == 200
    item = listing.get_json()["items"][0]
    assert item["pluginId"] == "http_v3"
    assert item["manifest"]["schema_version"] == 3

    config = client.put(
        "/api/v2/plugins/http_v3/config",
        json={
            "baseRevision": 1,
            "config": {"prefix": "http", "strict": True},
        },
        headers={"Idempotency-Key": "http-plugin-config"},
    )
    assert config.status_code == 200
    assert config.get_json()["configRevision"] == 2
    exported = client.get("/api/v2/plugins/http_v3/export")
    assert exported.status_code == 200
    assert parse_archive(exported.data).manifest.plugin_id == "http_v3"


def test_api_import_does_not_execute_plugin_and_worker_uses_frozen_snapshot(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    broken = registry.import_archive(
        data=_plugin_archive(
            plugin_id="metadata_only",
            source="raise RuntimeError('must only run in Worker')\n",
        ),
        base_revision=0,
        idempotency_key="metadata-only",
    )
    assert registry.get_plugin("metadata_only")["pluginVersionId"] == (
        broken["pluginVersionId"]
    )
    registry.set_runtime_enabled(
        plugin_id="metadata_only",
        enabled=False,
    )

    installed = registry.import_archive(
        data=_plugin_archive(
            plugin_id="frozen_v3",
            hooks=["before_translate"],
            source=(
                "class Plugin:\n"
                "    def before_translate(self, context, payload):\n"
                "        result = dict(payload)\n"
                "        result['originalTexts'] = [\n"
                "            context.config['prefix'] + value\n"
                "            for value in payload['originalTexts']\n"
                "        ]\n"
                "        return result\n"
            ),
        ),
        base_revision=0,
        idempotency_key="frozen-v1",
    )
    frozen = _enabled_snapshots(engine)
    jobs = JobQueueRepository(engine)
    created = jobs.create_batch(
        kind="export",
        display_name="frozen plugin",
        specs=[
            JobSpec(
                kind="export",
                config={"mode": "standard"},
                items=(
                    JobItemSpec(
                        page_id=None,
                        step_kinds=("export_package",),
                    ),
                ),
                plugin_snapshots=frozen,
            )
        ],
    )
    registry.update_config(
        plugin_id="frozen_v3",
        base_revision=1,
        config={"prefix": "changed", "strict": False},
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            role="worker",
            epoch_id=epoch_id,
            token="worker-token",
            pid=1234,
        )
    )
    fence = jobs.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    runtime = PluginJobRuntime(
        data_root=data_root,
        engine=engine,
        repository=jobs,
    )
    hook_page_id = str(uuid.uuid4())
    changed = runtime.run_atomic(
        fence,
        phase="before",
        step="translate",
        page_id=hook_page_id,
        data={
            "pageId": hook_page_id,
            "originalTexts": ["source"],
            "translationConfig": {},
        },
    )
    assert changed["originalTexts"] == ["[v3]source"]
    loaded_version_root = (
        data_root
        / "plugins"
        / "frozen_v3"
        / "versions"
        / str(installed["pluginVersionId"])
    )
    assert not (loaded_version_root / "__pycache__").exists()
    with engine.connect() as connection:
        version_ids = set(
            connection.execute(
                select(
                    job_plugin_snapshots.c.plugin_version_id
                ).where(
                    job_plugin_snapshots.c.job_id
                    == created["jobIds"][0]
                )
            ).scalars()
        )
    assert str(installed["pluginVersionId"]) in version_ids


def test_worker_operation_snapshots_plugins_and_fail_policy_is_enforced(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    installed = registry.import_archive(
        data=_plugin_archive(
            plugin_id="strict_v3",
            hooks=["before_translate"],
            failure_policy="fail",
            source=(
                "class Plugin:\n"
                "    def before_translate(self, context, payload):\n"
                "        raise RuntimeError('strict failure')\n"
            ),
        ),
        base_revision=0,
        idempotency_key="strict-v1",
    )
    book_id = str(uuid.uuid4())
    chapter_id = str(uuid.uuid4())
    page_id = str(uuid.uuid4())
    source_asset_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            assets.insert().values(
                id=source_asset_id,
                relative_path="objects/test/plugin-operation-source.png",
                mime_type="image/png",
                checksum="0" * 64,
                byte_size=0,
            )
        )
        connection.execute(
            books.insert().values(
                id=book_id,
                kind="library",
                title="Plugin operation",
            )
        )
        connection.execute(
            chapters.insert().values(
                id=chapter_id,
                book_id=book_id,
                ordinal=1,
                title="Chapter",
            )
        )
        connection.execute(
            pages.insert().values(
                id=page_id,
                chapter_id=chapter_id,
                ordinal=1,
                logical_source_path="page.png",
            )
        )
        connection.execute(
            page_assets.insert().values(
                page_id=page_id,
                role="source",
                asset_id=source_asset_id,
                input_source_revision=1,
                input_document_revision=1,
            )
        )
    operation, replayed = OperationRepository(engine).create_page_operation(
        kind="page_detect",
        page_id=page_id,
        base_revision=1,
        bubble_id=None,
        payload={},
        idempotency_key="plugin-operation-snapshot",
    )
    assert replayed is False
    with engine.connect() as connection:
        operation_version = connection.execute(
            select(
                operation_plugin_snapshots.c.plugin_version_id
            ).where(
                operation_plugin_snapshots.c.operation_id
                == operation["operationId"]
            )
        ).scalar_one()
    assert operation_version == installed["pluginVersionId"]

    jobs = JobQueueRepository(engine)
    jobs.create_batch(
        kind="export",
        display_name="strict plugin",
        specs=[
            JobSpec(
                kind="export",
                config={"mode": "standard"},
                items=(
                    JobItemSpec(
                        page_id=None,
                        step_kinds=("export_package",),
                    ),
                ),
                plugin_snapshots=_enabled_snapshots(engine),
            )
        ],
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            role="worker",
            epoch_id=epoch_id,
            token="worker-token",
            pid=1235,
        )
    )
    fence = jobs.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    runtime = PluginJobRuntime(
        data_root=data_root,
        engine=engine,
        repository=jobs,
    )
    hook_page_id = str(uuid.uuid4())
    with pytest.raises(PluginHookFailure, match="strict failure"):
        runtime.run_atomic(
            fence,
            phase="before",
            step="translate",
            page_id=hook_page_id,
            data={
                "pageId": hook_page_id,
                "originalTexts": ["source"],
                "translationConfig": {},
            },
        )


def test_worker_plugin_lifecycle_is_job_once_and_pipeline_once_per_page(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    book_id = str(uuid.uuid4())
    chapter_id = str(uuid.uuid4())
    first_page_id = str(uuid.uuid4())
    second_page_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            books.insert().values(
                id=book_id,
                kind="library",
                title="Plugin lifecycle",
            )
        )
        connection.execute(
            chapters.insert().values(
                id=chapter_id,
                book_id=book_id,
                ordinal=1,
                title="Chapter",
            )
        )
        connection.execute(
            pages.insert(),
            [
                {
                    "id": first_page_id,
                    "chapter_id": chapter_id,
                    "ordinal": 1,
                    "logical_source_path": "first.png",
                },
                {
                    "id": second_page_id,
                    "chapter_id": chapter_id,
                    "ordinal": 2,
                    "logical_source_path": "second.png",
                },
            ],
        )
    PluginRegistry(data_root=data_root, engine=engine).import_archive(
        data=_plugin_archive(
            plugin_id="lifecycle_v3",
            hooks=[
                "before_job",
                "after_job",
                "before_pipeline",
                "after_pipeline",
            ],
            supported_steps=["job", "pipeline"],
            failure_policy="fail",
            source=(
                "class Plugin:\n"
                "    def before_job(self, context, payload):\n"
                "        result = dict(payload)\n"
                "        result['pluginInjected'] = True\n"
                "        return result\n"
                "    def after_job(self, context, payload):\n"
                "        return payload\n"
                "    def before_pipeline(self, context, payload):\n"
                "        return payload\n"
                "    def after_pipeline(self, context, payload):\n"
                f"        if context.page_id == {first_page_id!r}:\n"
                "            raise RuntimeError('first page rejected')\n"
                "        return payload\n"
            ),
        ),
        base_revision=0,
        idempotency_key="lifecycle-v1",
    )
    jobs = JobQueueRepository(engine)
    created = jobs.create_batch(
        kind="export",
        display_name="plugin lifecycle",
        specs=[
            JobSpec(
                kind="export",
                book_id=book_id,
                chapter_id=chapter_id,
                config={"mode": "standard", "executionMode": "sequential"},
                items=(
                    JobItemSpec(
                        page_id=first_page_id,
                        step_kinds=("fixture_prepare", "fixture_publish"),
                    ),
                    JobItemSpec(
                        page_id=second_page_id,
                        step_kinds=("fixture_prepare", "fixture_publish"),
                    ),
                ),
                plugin_snapshots=_enabled_snapshots(engine),
            )
        ],
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            role="worker",
            epoch_id=epoch_id,
            token="worker-token",
            pid=1236,
        )
    )
    fence = jobs.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    seen_configs: list[dict[str, object]] = []

    def handle(_fence, step):
        seen_configs.append(dict(step["config"]))
        return {"handled": True}

    runtime = PluginJobRuntime(
        data_root=data_root,
        engine=engine,
        repository=jobs,
    )
    JobWorkerLoop(
        jobs,
        worker_epoch_id=epoch_id,
        handlers={
            "fixture_prepare": handle,
            "fixture_publish": handle,
        },
        plugin_runtime=runtime,
    )._run_attempt(fence, threading.Event())

    assert len(seen_configs) == 4
    assert all(config["pluginInjected"] is True for config in seen_configs)
    with engine.connect() as connection:
        job = connection.execute(
            select(jobs_table.c.status, jobs_table.c.config_json).where(
                jobs_table.c.id == created["jobIds"][0]
            )
        ).mappings().one()
        item_statuses = list(
            connection.execute(
                select(job_items.c.status)
                .where(job_items.c.job_id == created["jobIds"][0])
                .order_by(job_items.c.ordinal)
            ).scalars()
        )
        stages = [
            json.loads(value)
            for value in connection.execute(
                select(job_events.c.payload_json).where(
                    job_events.c.job_id == created["jobIds"][0],
                    job_events.c.event_type == "plugin_stage_completed",
                )
            ).scalars()
        ]
    assert job["status"] == "completed_with_errors"
    assert json.loads(job["config_json"])["pluginInjected"] is True
    assert item_statuses == ["failed", "completed"]
    assert [stage["hook"] for stage in stages].count("before_job") == 1
    assert [stage["hook"] for stage in stages].count("after_job") == 1
    assert [stage["hook"] for stage in stages].count("before_pipeline") == 2
    assert [stage["hook"] for stage in stages].count("after_pipeline") == 2
    assert created["jobIds"][0] not in runtime._stage_cache
    first_after = next(
        stage
        for stage in stages
        if stage["hook"] == "after_pipeline"
        and stage["pageId"] == first_page_id
    )
    assert first_after["outcome"] == "failed"


class _FakeAgentProvider:
    def snapshot(self):
        return {
            "provider": "fake",
            "model_name": "fake-model",
            "custom_base_url": "",
            "openai_options": {
                "request": {"force_json_output": True},
                "execution": {"use_stream": False},
            },
            "settingsSnapshot": {
                "appRevision": 1,
                "providerRevision": 1,
            },
        }

    def runtime_config(self, _snapshot=None):
        return {"provider": "fake-runtime"}


class _FakeAgentController:
    def __init__(self, *, generated_plugin_id: str | None = None):
        self.generated_plugin_id = generated_plugin_id

    def plan_turn(self, _session, _skill, _config):
        return {
            "assistant_message": "方案已明确。",
            "target_proposal": {
                "plugin_id": "agent_v3",
                "display_name": "Agent v3",
                "supported_steps": ["translate"],
                "supported_modes": ["standard"],
            },
        }

    def execute(
        self,
        session,
        _skill,
        _config,
        tools,
        emit,
    ):
        manifest = {
            "schema_version": 3,
            "plugin_id": (
                self.generated_plugin_id
                or session.locked_target.plugin_id
            ),
            "display_name": session.locked_target.display_name,
            "package_version": "1.0.0",
            "entrypoint": "plugin.py:Plugin",
            "hooks": ["after_translate"],
            "supported_steps": ["translate"],
            "supported_modes": ["standard"],
            "priority": 100,
            "failure_policy": "continue",
            "author": "Plugin Agent",
            "description": "generated in a durable worktree",
            "default_enabled": False,
            "config_schema": {},
        }
        emit("tool_call", {"tool": "write_file"})
        tools.run_tool(
            "write_file",
            {
                "path": "plugin.json",
                "content": json.dumps(manifest),
            },
        )
        tools.run_tool(
            "write_file",
            {
                "path": "plugin.py",
                "content": (
                    "class Plugin:\n"
                    "    def after_translate(self, context, data):\n"
                    "        return dict(data)\n"
                ),
            },
        )
        validation = tools.run_tool("validate_plugin")
        return {
            "assistant_message": "生成完成。",
            "validation": validation,
        }


def test_plugin_agent_planning_hands_off_to_durable_worker_job(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    provider = _FakeAgentProvider()
    controller = _FakeAgentController()
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=controller,
        provider_resolver=provider,
    )
    created = sessions.create(mode="create", plugin_id=None)
    session_id = created["session_id"]
    planned = sessions.send_message(
        session_id=session_id,
        content="创建一个翻译后处理插件",
    )
    assert planned["run_state"] == "awaiting_target_lock"
    locked = sessions.lock_target(
        session_id=session_id,
        proposal=planned["pending_target"],
    )
    assert locked["run_state"] == "ready"
    started = sessions.start(
        session_id=session_id,
        idempotency_key="agent-start",
    )
    replayed = sessions.start(
        session_id=session_id,
        idempotency_key="agent-start",
    )
    job_id = started["jobId"]
    assert replayed["batchId"] == started["batchId"]
    assert replayed["jobId"] == job_id
    assert started["session"]["run_state"] == "running"
    serialized = json.dumps(started, ensure_ascii=False)
    assert "api_key" not in serialized
    assert "agent_config" not in serialized
    with engine.connect() as connection:
        row = connection.execute(
            select(
                jobs_table.c.kind,
                jobs_table.c.config_json,
            ).where(jobs_table.c.id == job_id)
        ).mappings().one()
    assert row["kind"] == "plugin_agent"
    assert "api_key" not in str(row["config_json"])

    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            role="worker",
            epoch_id=epoch_id,
            token="worker-token",
            pid=4321,
        )
    )
    queue = JobQueueRepository(engine)
    fence = queue.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    step = queue.next_step(fence)
    assert step is not None
    worker = PluginAgentWorkerService(
        data_root=data_root,
        engine=engine,
        jobs=queue,
        controller=controller,
        provider_resolver=provider,
    )
    result = worker.handle(fence, step)
    queue.complete_step(
        fence,
        step_id=str(step["stepId"]),
        checkpoint=result,
    )
    queue.finish_if_complete(fence)
    reconciled = sessions.get(session_id)
    assert reconciled["run_state"] == "completed"
    assert reconciled["execution_started_at"]
    assert reconciled["execution_finished_at"]
    assert reconciled["touched_files"] == ["plugin.json", "plugin.py"]
    assert set(reconciled["file_previews"]) == {
        "plugin.json",
        "plugin.py",
    }
    assert reconciled["last_validation"]["success"] is True
    assert reconciled["messages"][-1]["role"] == "assistant"
    assert reconciled["messages"][-1]["content"] == "生成完成。"
    plugin = PluginRegistry(
        data_root=data_root,
        engine=engine,
    ).get_plugin("agent_v3")
    assert plugin["packageVersion"] == "1.0.0"
    assert not (
        data_root
        / "temp"
        / "jobs"
        / job_id
        / "plugin-worktree"
    ).exists()


def test_plugin_agent_rejects_generated_manifest_for_another_plugin(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    provider = _FakeAgentProvider()
    controller = _FakeAgentController(
        generated_plugin_id="different_v3",
    )
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=controller,
        provider_resolver=provider,
    )
    session_id = sessions.create(
        mode="create",
        plugin_id=None,
    )["session_id"]
    planned = sessions.send_message(
        session_id=session_id,
        content="创建一个翻译后处理插件",
    )
    sessions.lock_target(
        session_id=session_id,
        proposal=planned["pending_target"],
    )
    started = sessions.start(
        session_id=session_id,
        idempotency_key="agent-mismatched-id",
    )

    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(
            role="worker",
            epoch_id=epoch_id,
            token="worker-token",
            pid=4322,
        )
    )
    queue = JobQueueRepository(engine)
    fence = queue.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    step = queue.next_step(fence)
    assert step is not None
    worker = PluginAgentWorkerService(
        data_root=data_root,
        engine=engine,
        jobs=queue,
        controller=controller,
        provider_resolver=provider,
    )

    with pytest.raises(ValueError, match="locked target"):
        worker.handle(fence, step)
    registry = PluginRegistry(data_root=data_root, engine=engine)
    with pytest.raises(PluginNotFound):
        registry.get_plugin("different_v3")
    with pytest.raises(PluginNotFound):
        registry.get_plugin("agent_v3")
    assert not (
        data_root
        / "temp"
        / "jobs"
        / started["jobId"]
        / "plugin-worktree"
    ).exists()


def test_modify_agent_job_protects_disabled_target_version(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    installed = registry.import_archive(
        data=_plugin_archive(
            plugin_id="disabled_target_v3",
            default_enabled=False,
        ),
        base_revision=0,
        idempotency_key="disabled-target",
    )
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=_FakeAgentController(),
        provider_resolver=_FakeAgentProvider(),
    )
    session_id = sessions.create(
        mode="modify",
        plugin_id="disabled_target_v3",
    )["session_id"]
    sessions.send_message(
        session_id=session_id,
        content="修改这个插件",
    )
    started = sessions.start(
        session_id=session_id,
        idempotency_key="modify-disabled-target",
    )

    with engine.connect() as connection:
        snapshot = connection.execute(
            select(job_plugin_snapshots.c.config_json).where(
                job_plugin_snapshots.c.job_id == started["jobId"],
                job_plugin_snapshots.c.plugin_version_id
                == installed["pluginVersionId"],
            )
        ).scalar_one()
    assert json.loads(snapshot) == {
        "pluginId": "disabled_target_v3",
        "protectOnly": True,
    }
    with pytest.raises(PluginLocked, match="task history"):
        registry.delete_plugin(
            plugin_id="disabled_target_v3",
            base_revision=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("supported_steps", ["unknown_step"]),
        ("supported_modes", ["unknown_mode"]),
        ("supported_steps", ["translate", "translate"]),
    ],
)
def test_plugin_agent_target_rejects_invalid_capability_values(
    plugin_platform,
    field: str,
    value: list[str],
) -> None:
    data_root, engine = plugin_platform
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=_FakeAgentController(),
        provider_resolver=_FakeAgentProvider(),
    )
    session_id = sessions.create(
        mode="create",
        plugin_id=None,
    )["session_id"]
    proposal = {
        "plugin_id": "validated_v3",
        "display_name": "Validated v3",
        "supported_steps": ["translate"],
        "supported_modes": ["standard"],
    }
    proposal[field] = value

    with pytest.raises(ValueError):
        sessions.lock_target(
            session_id=session_id,
            proposal=proposal,
        )


def test_plugin_builder_skill_states_the_actual_v3_manifest_contract() -> None:
    skill = (
        Path(__file__).parents[2]
        / "src"
        / "backend_v2"
        / "plugins"
        / "plugin_builder_skill.md"
    ).read_text(encoding="utf-8")

    assert '"hooks": ["after_translate"]' in skill
    for mode in ("standard", "hq", "proofread", "remove_text"):
        assert f"`{mode}`" in skill
    assert "not JSON Schema" in skill
    assert '"source_text": {' in skill
    assert '`data["translations"]`' in skill
    assert "`originalTexts`, `translations`, `textboxTexts`" in skill
    assert "translated_text" in skill


def test_atomic_plugin_schema_rejects_silent_invented_fields() -> None:
    with pytest.raises(
        PluginContractError,
        match="translated_text",
    ):
        validate_atomic_hook_data(
            "translate",
            "after",
            {
                "pageId": str(uuid.uuid4()),
                "originalTexts": ["先生"],
                "translations": ["老师"],
                "textboxTexts": ["老师"],
                "translated_text": "导师",
            },
        )


def test_plugin_agent_source_validation_rejects_invented_hook_field() -> None:
    manifest = parse_manifest(
        {
            "schema_version": 3,
            "plugin_id": "source_contract",
            "display_name": "Source contract",
            "package_version": "1.0.0",
            "entrypoint": "plugin.py:Plugin",
            "hooks": ["after_translate"],
            "supported_steps": ["translate"],
            "supported_modes": ["standard"],
            "priority": 100,
            "failure_policy": "continue",
            "author": "",
            "description": "",
            "default_enabled": False,
            "config_schema": {},
        }
    )
    with pytest.raises(
        PluginContractError,
        match="translated_text",
    ):
        validate_hook_source_contract(
            manifest,
            (
                "class Plugin:\n"
                "    def after_translate(self, context, data):\n"
                "        result = dict(data)\n"
                "        result['translated_text'] = 'wrong'\n"
                "        return result\n"
            ),
            filename="plugin.py",
        )


def test_plugin_agent_source_validation_rejects_wrong_field_container() -> None:
    manifest = parse_manifest(
        {
            "schema_version": 3,
            "plugin_id": "source_types",
            "display_name": "Source types",
            "package_version": "1.0.0",
            "entrypoint": "plugin.py:Plugin",
            "hooks": ["after_translate"],
            "supported_steps": ["translate"],
            "supported_modes": ["standard"],
            "priority": 100,
            "failure_policy": "continue",
            "author": "",
            "description": "",
            "default_enabled": False,
            "config_schema": {},
        }
    )
    with pytest.raises(
        PluginContractError,
        match="textboxTexts.*object.*array",
    ):
        validate_hook_source_contract(
            manifest,
            (
                "class Plugin:\n"
                "    def after_translate(self, context, data):\n"
                "        textbox = data.get('textboxTexts', {})\n"
                "        data['textboxTexts'] = {\n"
                "            key: value for key, value in textbox.items()\n"
                "        }\n"
                "        return data\n"
            ),
            filename="plugin.py",
        )


def test_plugin_source_validation_rejects_incompatible_constructor() -> None:
    manifest = parse_manifest(
        {
            "schema_version": 3,
            "plugin_id": "constructor_contract",
            "display_name": "Constructor contract",
            "package_version": "1.0.0",
            "entrypoint": "plugin.py:Plugin",
            "hooks": ["after_translate"],
            "supported_steps": ["translate"],
            "supported_modes": ["standard"],
            "priority": 100,
            "failure_policy": "continue",
            "author": "",
            "description": "",
            "default_enabled": False,
            "config_schema": {},
        }
    )
    with pytest.raises(
        PluginContractError,
        match="__init__ must be callable without arguments",
    ):
        validate_hook_source_contract(
            manifest,
            (
                "class Plugin:\n"
                "    def __init__(self, context):\n"
                "        self.context = context\n"
                "    def after_translate(self, context, data):\n"
                "        return data\n"
            ),
            filename="plugin.py",
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "class Plugin:\n"
            "    async def after_translate(self, context, data):\n"
            "        return data\n",
            "must be synchronous",
        ),
        (
            "class Plugin:\n"
            "    @staticmethod\n"
            "    def after_translate(self, context, data):\n"
            "        return data\n",
            "normal instance method",
        ),
        (
            "class Plugin:\n"
            "    def after_translate(self, context, data, required):\n"
            "        return data\n",
            "callable as",
        ),
    ],
)
def test_plugin_source_validation_rejects_incompatible_hook_call_shapes(
    source: str,
    message: str,
) -> None:
    manifest = parse_manifest(
        {
            "schema_version": 3,
            "plugin_id": "hook_call_contract",
            "display_name": "Hook call contract",
            "package_version": "1.0.0",
            "entrypoint": "plugin.py:Plugin",
            "hooks": ["after_translate"],
            "supported_steps": ["translate"],
            "supported_modes": ["standard"],
            "priority": 100,
            "failure_policy": "continue",
            "author": "",
            "description": "",
            "default_enabled": False,
            "config_schema": {},
        }
    )
    with pytest.raises(PluginContractError, match=message):
        validate_hook_source_contract(
            manifest,
            source,
            filename="plugin.py",
        )


@pytest.mark.parametrize(
    "content",
    [
        '"现在修改 plugin.py"',
        '["not", "an", "object"]',
    ],
)
def test_plugin_agent_non_object_response_is_business_retryable(
    content: str,
) -> None:
    with pytest.raises(OpenAICompatibleBusinessRetryableError):
        PluginAgentController._parse_agent_envelope(
            content,
            force_json_output=False,
        )


@pytest.mark.parametrize(
    "content",
    [
        '{"assistant_message":"只说明、不执行"}',
        '{"assistant_message":"参数错误","action":{"tool":"write_file","args":[]}}',
    ],
)
def test_plugin_agent_invalid_execution_action_is_business_retryable(
    content: str,
) -> None:
    with pytest.raises(OpenAICompatibleBusinessRetryableError):
        PluginAgentController._parse_agent_envelope(
            content,
            force_json_output=False,
            require_action=True,
        )


def test_plugin_agent_planning_uses_one_leading_system_message() -> None:
    session = PluginAgentSession(
        session_id="planning-system-message",
        mode="create",
        messages=[
            PluginAgentMessage(
                id="user-1",
                role="user",
                content="创建一个插件",
            )
        ],
    )

    messages = PluginAgentController()._build_chat_messages(
        session,
        "planning instructions",
        "plugin skill",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "planning instructions" in messages[0]["content"]
    assert "plugin skill" in messages[0]["content"]


def test_plugin_agent_execution_uses_one_leading_system_message() -> None:
    session = PluginAgentSession(
        session_id="execution-system-message",
        mode="create",
        locked_target=LockedPluginTarget(
            mode="create",
            plugin_id="system_message_test",
            display_name="System Message Test",
            plugin_dir="system_message_test",
        ),
        messages=[
            PluginAgentMessage(
                id="user-1",
                role="user",
                content="开始执行",
            )
        ],
    )

    messages = PluginAgentController()._build_execution_messages(
        session,
        "execution instructions",
        "plugin skill",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "execution instructions" in messages[0]["content"]
    assert "plugin skill" in messages[0]["content"]


def test_plugin_agent_http_rejects_browser_supplied_provider_secret(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id=str(uuid.uuid4()),
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    client = app.test_client()
    created = client.post(
        "/api/v2/plugin-agent/sessions",
        json={"mode": "create"},
    )
    assert created.status_code == 201
    session_id = created.get_json()["session"]["session_id"]
    rejected = client.post(
        f"/api/v2/plugin-agent/sessions/{session_id}/messages",
        json={
            "content": "secret test",
            "agent_config": {"api_key": "must-not-cross-http"},
        },
    )
    assert rejected.status_code == 422
    assert "unknown request fields" in rejected.get_json()["error"][
        "message"
    ]
