from __future__ import annotations

from io import BytesIO
import json
import logging
from pathlib import Path
import threading
import uuid
import zipfile

import pytest
from sqlalchemy import select

import src.backend_v2.plugins.repository as plugin_repository_module
from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.plugins.contract import (
    PluginContractError,
    normalize_config_schema,
    parse_manifest,
    validate_atomic_hook_data,
    validate_config,
    validate_hook_data,
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
    PluginAssetAccess,
    PluginHookFailure,
    PluginJobRuntime,
    _validate_atomic_page,
    _json_object,
)
from src.backend_v2.plugins.snapshots import enabled_plugin_snapshots
from src.backend_v2.plugins.agent import PluginAgentSessionService
from src.backend_v2.plugins.agent_worker import (
    PluginAgentWorkerService,
)
from src.backend_v2.plugins.agent_tools import PluginAgentWorktreeTools
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
from src.core.plugin_agent.controller import (
    PluginAgentControlRequested,
    PluginAgentController,
)
from src.core.plugin_agent.models import (
    LockedPluginTarget,
    PluginAgentMessage,
    PluginAgentSession,
)
from src.core.config_models import BubbleState, validate_bubble_payload
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
)
from src.shared.openai_options import OpenAICompatibleOptions


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    missing_step = _valid_manifest()
    missing_step["supported_steps"] = ["ocr"]
    with pytest.raises(PluginContractError, match="missing supported_steps"):
        parse_manifest(missing_step)

    duplicate_mode = _valid_manifest()
    duplicate_mode["supported_modes"] = ["standard", "standard"]
    with pytest.raises(PluginContractError, match="must be unique"):
        parse_manifest(duplicate_mode)

    unrestricted_metadata = _valid_manifest()
    unrestricted_metadata["priority"] = 1_000_000
    unrestricted_metadata["description"] = "x" * 25_000
    assert parse_manifest(unrestricted_metadata).priority == 1_000_000


def test_plugin_manifest_requires_current_complete_config_schema() -> None:
    with pytest.raises(PluginContractError, match="default is required"):
        normalize_config_schema({"missing": {"type": "text"}})
    with pytest.raises(PluginContractError, match="fields are invalid"):
        normalize_config_schema(
            {
                "choice": {
                    "type": "select",
                    "default": "one",
                    "options": ["one"],
                }
            }
        )
    with pytest.raises(PluginContractError, match="finite number"):
        normalize_config_schema(
            {
                "ratio": {
                    "type": "number",
                    "default": float("nan"),
                }
            }
        )
    with pytest.raises(PluginContractError, match="label must be text"):
        normalize_config_schema(
            {
                "name": {
                    "type": "text",
                    "default": "value",
                    "label": None,
                }
            }
        )
    with pytest.raises(PluginContractError, match="finite number"):
        normalize_config_schema(
            {
                "ratio": {
                    "type": "number",
                    "default": 1,
                    "minimum": None,
                }
            }
        )

    schema = normalize_config_schema(
        {
            "choice": {
                "type": "select",
                "default": "one",
                "options": [
                    {"value": "one", "label": "One"},
                    {"value": 2, "label": "Two"},
                ],
            }
        }
    )
    assert validate_config(schema, {"choice": 2}) == {"choice": 2}
    with pytest.raises(PluginContractError, match="field mismatch"):
        validate_config(schema, {})

    numeric_schema = normalize_config_schema(
        {
            "ratio": {
                "type": "number",
                "default": 1.25,
                "minimum": 0.5,
                "maximum": 2.5,
            }
        }
    )
    assert validate_config(numeric_schema, {"ratio": 1.75}) == {"ratio": 1.75}


def test_atomic_plugin_schema_requires_exact_fields_and_shape() -> None:
    page_id = str(uuid.uuid4())
    with pytest.raises(PluginContractError, match="translationConfig"):
        validate_atomic_hook_data(
            "translate",
            "before",
            {"pageId": page_id, "originalTexts": ["先生"]},
        )
    with pytest.raises(PluginContractError, match="preserve translations"):
        _validate_atomic_page(
            "translate",
            "after",
            page_id,
            {
                "pageId": page_id,
                "originalTexts": ["先生"],
                "translations": ["老师"],
                "textboxTexts": [],
            },
            expected_shape={
                "originalTexts": 1,
                "translations": 2,
                "textboxTexts": 0,
            },
        )
    with pytest.raises(PluginContractError, match="documentRevision"):
        _validate_atomic_page(
            "render",
            "after",
            page_id,
            {
                "pageId": page_id,
                "translatedAssetId": str(uuid.uuid4()),
                "documentRevision": 3,
            },
            expected_shape={"documentRevision": 2},
        )


def test_hook_json_validation_has_no_base64_heuristic_false_positive() -> None:
    value = {"message": "A" * 4096}
    assert validate_hook_data(value) == value
    with pytest.raises(PluginContractError, match="Base64"):
        validate_hook_data(
            {"image": "data:image/png;base64," + "A" * 4096}
        )


def test_plugin_asset_publish_has_no_broken_arbitrary_size_gate(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    access = PluginAssetAccess(data_root=data_root, engine=engine)

    asset_id = access.publish_bytes(
        b"plugin output",
        extension="bin",
        mime_type="application/octet-stream",
    )

    with engine.connect() as connection:
        row = connection.execute(
            select(assets.c.byte_size).where(assets.c.id == asset_id)
        ).scalar_one()
    assert row == len(b"plugin output")


def test_plugin_runtime_rejects_corrupt_stored_json() -> None:
    with pytest.raises(PluginContractError, match="JSON is invalid"):
        _json_object("{")
    with pytest.raises(PluginContractError, match="must be an object"):
        _json_object("[]")


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

    changed, replayed = registry.update_config(
        plugin_id="test_v3",
        base_revision=1,
        config={"prefix": "new", "strict": True},
        idempotency_key="config-v1",
    )
    assert replayed is False
    assert changed["configRevision"] == 2
    with pytest.raises(PluginConflict, match="config revision"):
        registry.update_config(
            plugin_id="test_v3",
            base_revision=1,
            config={"prefix": "stale", "strict": False},
            idempotency_key="config-stale",
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


def test_plugin_upgrade_discards_old_config_and_advances_revision(
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
        idempotency_key="schema-config-v1",
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

    config = registry.get_config("schema_v3")
    assert config["value"] == {
        "strict": False,
        "suffix": "!",
    }
    assert config["configRevision"] == 3


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
            idempotency_key="delete-locked-test-v3",
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


def test_plugin_import_cleans_partial_extraction(
    plugin_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)

    def fail_extraction(_archive, destination: Path) -> str:
        destination.mkdir(parents=True)
        (destination / "partial.py").write_text("partial", encoding="utf-8")
        raise OSError("extract failed")

    monkeypatch.setattr(
        plugin_repository_module,
        "extract_archive",
        fail_extraction,
    )

    with pytest.raises(OSError, match="extract failed"):
        registry.import_archive(
            data=_plugin_archive(plugin_id="partial_v3"),
            base_revision=0,
            idempotency_key="partial-plugin",
        )

    assert list((data_root / "temp" / "plugins").iterdir()) == []


def test_plugin_refresh_does_not_downgrade_memory_failure(
    plugin_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    registry.import_archive(
        data=_plugin_archive(plugin_id="refresh_memory_v3"),
        base_revision=0,
        idempotency_key="refresh-memory-plugin",
    )
    monkeypatch.setattr(
        plugin_repository_module,
        "directory_checksum",
        lambda _root: (_ for _ in ()).throw(MemoryError("refresh allocation failed")),
    )

    with pytest.raises(MemoryError, match="allocation failed"):
        registry.refresh()

    plugin = registry.get_plugin("refresh_memory_v3")
    assert plugin["state"] == "enabled"
    assert plugin["runtimeEnabled"] is True


def test_plugin_delete_does_not_restore_files_after_database_commit(
    plugin_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    registry.import_archive(
        data=_plugin_archive(plugin_id="delete_cleanup_v3"),
        base_revision=0,
        idempotency_key="delete-cleanup-plugin",
    )
    monkeypatch.setattr(
        plugin_repository_module.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    result, replayed = registry.delete_plugin(
        plugin_id="delete_cleanup_v3",
        base_revision=1,
        idempotency_key="delete-cleanup-v1",
    )

    assert replayed is False
    assert result["deleted"] is True
    with pytest.raises(PluginNotFound):
        registry.get_plugin("delete_cleanup_v3")
    assert not (data_root / "plugins" / "delete_cleanup_v3").exists()


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
    assert "packageRelativePath" not in item
    assert "checksum" not in item

    fractional_revision = client.put(
        "/api/v2/plugins/http_v3/config",
        json={
            "baseRevision": 1.5,
            "config": {"prefix": "invalid", "strict": True},
        },
    )
    assert fractional_revision.status_code == 422

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
    config_replay = client.put(
        "/api/v2/plugins/http_v3/config",
        json={
            "baseRevision": 1,
            "config": {"prefix": "http", "strict": True},
        },
        headers={"Idempotency-Key": "http-plugin-config"},
    )
    assert config_replay.status_code == 200
    assert config_replay.get_json() == config.get_json()
    assert config_replay.headers["Idempotency-Replayed"] == "true"
    default_state = client.put(
        "/api/v2/plugins/http_v3/default-enabled",
        json={"enabled": False},
        headers={"Idempotency-Key": "http-plugin-default"},
    )
    assert default_state.status_code == 200
    default_replay = client.put(
        "/api/v2/plugins/http_v3/default-enabled",
        json={"enabled": False},
        headers={"Idempotency-Key": "http-plugin-default"},
    )
    assert default_replay.status_code == 200
    assert default_replay.get_json() == default_state.get_json()
    assert default_replay.headers["Idempotency-Replayed"] == "true"
    default_conflict = client.put(
        "/api/v2/plugins/http_v3/default-enabled",
        json={"enabled": True},
        headers={"Idempotency-Key": "http-plugin-default"},
    )
    assert default_conflict.status_code == 409
    assert default_conflict.get_json()["error"]["code"] == (
        "idempotency_conflict"
    )
    exported = client.get("/api/v2/plugins/http_v3/export")
    assert exported.status_code == 200
    assert parse_archive(exported.data).manifest.plugin_id == "http_v3"
    deleted = client.delete(
        "/api/v2/plugins/http_v3",
        headers={
            "If-Match": "1",
            "Idempotency-Key": "http-plugin-delete",
        },
    )
    assert deleted.status_code == 200
    delete_replay = client.delete(
        "/api/v2/plugins/http_v3",
        headers={
            "If-Match": "1",
            "Idempotency-Key": "http-plugin-delete",
        },
    )
    assert delete_replay.status_code == 200
    assert delete_replay.get_json() == deleted.get_json()
    assert delete_replay.headers["Idempotency-Replayed"] == "true"


def test_api_import_does_not_execute_plugin_and_worker_uses_frozen_snapshot(
    plugin_platform,
    caplog,
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
                "        context.logger.info('frozen input', count=len(payload['originalTexts']))\n"
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
        idempotency_key="frozen-config-v1",
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
    with caplog.at_level(logging.INFO, logger="saber.user"):
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
    assert any(
        "插件 frozen_v3（before_translate）｜frozen input" in record.getMessage()
        and '"count": 1' in record.getMessage()
        for record in caplog.records
        if record.name == "saber.user"
    )
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


def test_continue_plugin_cannot_swallow_memory_failure(plugin_platform) -> None:
    data_root, engine = plugin_platform
    PluginRegistry(data_root=data_root, engine=engine).import_archive(
        data=_plugin_archive(
            plugin_id="memory_failure_v3",
            hooks=["before_translate"],
            failure_policy="continue",
            source=(
                "class Plugin:\n"
                "    def before_translate(self, context, payload):\n"
                "        raise MemoryError('native allocation failed')\n"
            ),
        ),
        base_revision=0,
        idempotency_key="memory-failure-v1",
    )
    jobs = JobQueueRepository(engine)
    jobs.create_batch(
        kind="export",
        display_name="memory failure plugin",
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
            pid=1237,
        )
    )
    fence = jobs.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    runtime = PluginJobRuntime(
        data_root=data_root,
        engine=engine,
        repository=jobs,
    )
    page_id = str(uuid.uuid4())
    with pytest.raises(MemoryError, match="allocation failed"):
        runtime.run_atomic(
            fence,
            phase="before",
            step="translate",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": ["source"],
                "translationConfig": {},
            },
        )


def test_plugin_load_cannot_swallow_memory_failure(plugin_platform) -> None:
    data_root, engine = plugin_platform
    registry = PluginRegistry(data_root=data_root, engine=engine)
    registry.import_archive(
        data=_plugin_archive(
            plugin_id="load_memory_failure_v3",
            hooks=["before_translate"],
            failure_policy="continue",
            source=(
                "raise MemoryError('plugin import allocation failed')\n"
                "class Plugin:\n"
                "    def before_translate(self, context, payload):\n"
                "        return payload\n"
            ),
        ),
        base_revision=0,
        idempotency_key="load-memory-failure-v1",
    )
    jobs = JobQueueRepository(engine)
    jobs.create_batch(
        kind="export",
        display_name="plugin load memory failure",
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
            pid=1238,
        )
    )
    fence = jobs.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    runtime = PluginJobRuntime(
        data_root=data_root,
        engine=engine,
        repository=jobs,
    )
    page_id = str(uuid.uuid4())

    with pytest.raises(MemoryError, match="import allocation failed"):
        runtime.run_atomic(
            fence,
            phase="before",
            step="translate",
            page_id=page_id,
            data={
                "pageId": page_id,
                "originalTexts": ["source"],
                "translationConfig": {},
            },
        )

    plugin = registry.get_plugin("load_memory_failure_v3")
    assert plugin["state"] == "enabled"
    assert plugin["runtimeEnabled"] is True


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
        validation = tools.run_tool("validate_plugin", {})
        return {
            "assistant_message": "生成完成。",
            "validation": validation,
        }


class _ControlRequestedAgentController:
    def execute(self, *_args, **_kwargs):
        raise PluginAgentControlRequested("control requested")


class _BlockingPlanningAgentController:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def plan_turn(self, *_args, **_kwargs):
        self.entered.set()
        assert self.release.wait(timeout=5)
        return {
            "assistant_message": "规划完成。",
            "target_proposal": {
                "plugin_id": "blocking_v3",
                "display_name": "Blocking v3",
                "supported_steps": ["translate"],
                "supported_modes": ["standard"],
            },
        }


class _BlockingPluginAgentJobs:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def create_batch(self, **_kwargs):
        self.entered.set()
        assert self.release.wait(timeout=5)
        return {"batchId": "batch-1", "jobIds": ["job-1"]}


def test_plugin_agent_planning_has_no_single_session_or_message_size_gate(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=_FakeAgentController(),
        provider_resolver=_FakeAgentProvider(),
    )
    first = sessions.create(mode="create", plugin_id=None)
    second = sessions.create(mode="create", plugin_id=None)

    assert first["session_id"] != second["session_id"]
    planned = sessions.send_message(
        session_id=str(first["session_id"]),
        content="需" * 100_001,
    )
    assert planned["run_state"] == "awaiting_target_lock"


def test_plugin_agent_session_cannot_be_deleted_during_planning(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    controller = _BlockingPlanningAgentController()
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=controller,
        provider_resolver=_FakeAgentProvider(),
    )
    session_id = sessions.create(mode="create", plugin_id=None)["session_id"]
    result: list[dict[str, object]] = []

    thread = threading.Thread(
        target=lambda: result.append(
            sessions.send_message(session_id=session_id, content="创建插件")
        )
    )
    thread.start()
    assert controller.entered.wait(timeout=5)
    with pytest.raises(ValueError, match="planning request"):
        sessions.delete(session_id)
    controller.release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result[0]["run_state"] == "awaiting_target_lock"


def test_plugin_agent_session_cannot_be_deleted_while_job_is_queued(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=_FakeAgentController(),
        provider_resolver=_FakeAgentProvider(),
    )
    session_id = sessions.create(mode="create", plugin_id=None)["session_id"]
    planned = sessions.send_message(session_id=session_id, content="创建插件")
    sessions.lock_target(
        session_id=session_id,
        proposal=planned["pending_target"],
    )
    jobs = _BlockingPluginAgentJobs()
    sessions.jobs = jobs
    result: list[dict[str, object]] = []

    thread = threading.Thread(
        target=lambda: result.append(
            sessions.start(session_id=session_id, idempotency_key="start-blocking")
        )
    )
    thread.start()
    assert jobs.entered.wait(timeout=5)
    with pytest.raises(ValueError, match="execution"):
        sessions.delete(session_id)
    jobs.release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result[0]["jobId"] == "job-1"
    assert result[0]["session"]["run_state"] == "running"


def test_plugin_agent_create_session_rejects_non_current_mode_payload(
    plugin_platform,
) -> None:
    data_root, engine = plugin_platform
    sessions = PluginAgentSessionService(
        data_root=data_root,
        engine=engine,
        controller=_FakeAgentController(),
        provider_resolver=_FakeAgentProvider(),
    )

    with pytest.raises(ValueError, match="mode must be"):
        sessions.create(mode=" CREATE ", plugin_id=None)
    with pytest.raises(ValueError, match="does not accept"):
        sessions.create(mode="create", plugin_id="unused_v3")


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
    first_fence = queue.claim_next(worker_epoch_id=epoch_id)
    assert first_fence is not None
    assert queue.request_pause(job_id)["status"] == "pausing"
    assert sessions.get(session_id)["run_state"] == "pausing"
    assert queue.finalize_control(first_fence) == "paused"
    assert sessions.get(session_id)["run_state"] == "paused"
    assert queue.resume(job_id)["status"] == "queued"
    assert sessions.get(session_id)["run_state"] == "running"

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


@pytest.mark.parametrize("status", ["pausing", "cancelling"])
def test_plugin_agent_control_yields_the_running_step(
    tmp_path: Path,
    status: str,
) -> None:
    checkpoints: list[tuple[str, dict[str, object]]] = []
    events: list[tuple[str, dict[str, object]]] = []

    class Jobs:
        @staticmethod
        def append_worker_event(
            _fence,
            *,
            event_type: str,
            payload: dict[str, object],
        ) -> None:
            events.append((event_type, payload))

        @staticmethod
        def control_status(_fence) -> str:
            return status

        @staticmethod
        def checkpoint_step(
            _fence,
            *,
            step_id: str,
            checkpoint: dict[str, object],
        ) -> str:
            checkpoints.append((step_id, checkpoint))
            return status

    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    worker = PluginAgentWorkerService(
        data_root=data_root,
        engine=engine,
        jobs=Jobs(),
        controller=_ControlRequestedAgentController(),
        provider_resolver=_FakeAgentProvider(),
    )
    fence = type("Fence", (), {"job_id": "job-1"})()
    result = worker.handle(
        fence,
        {
            "stepKind": "plugin_agent_execute",
                "stepId": "plugin-step",
                "config": {
                    "executionMode": "sequential",
                    "sessionId": "a" * 32,
                "target": {
                    "mode": "create",
                    "plugin_id": "agent_v3",
                    "display_name": "Agent v3",
                    "supported_steps": ["translate"],
                        "supported_modes": ["standard"],
                        "baseRevision": 0,
                        "pluginVersionId": None,
                    },
                "messages": [],
                "provider": {},
            },
        },
    )

    assert result == {
        "__already_published__": True,
        "__control_drained__": True,
    }
    assert checkpoints == [("plugin-step", {})]
    assert [event_type for event_type, _payload in events] == [
        "plugin_agent_state"
    ]
    assert not (
        data_root
        / "temp"
        / "jobs"
        / "job-1"
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
            idempotency_key="delete-disabled-target-v1",
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


def test_bundled_plugins_match_current_manifests_and_hook_payloads() -> None:
    plugin_root = PROJECT_ROOT / "plugins"
    page_id = str(uuid.uuid4())
    stored_bubble = BubbleState().to_dict()
    stored_bubble.pop("fontFamily")
    sample_payloads = {
        "before_job": {},
        "after_job": {},
        "before_pipeline": {},
        "after_pipeline": {},
        "before_detect": {
            "pageId": page_id,
            "sourceAssetId": str(uuid.uuid4()),
            "detectorConfig": {"detector_type": "default"},
        },
        "after_ocr": {
            "pageId": page_id,
            "originalTexts": ["source"],
            "ocrResults": [{}],
        },
        "before_translate": {
            "pageId": page_id,
            "originalTexts": ["source"],
            "translationConfig": {},
        },
        "after_translate": {
            "pageId": page_id,
            "originalTexts": ["source"],
            "translations": ["translated"],
            "textboxTexts": ["translated"],
        },
        "after_ai_translate": {
            "pageId": page_id,
            "originalTexts": ["source"],
            "translations": ["translated"],
        },
        "after_color": {
            "pageId": page_id,
            "colors": [
                {
                    "fgColor": [0, 0, 0],
                    "bgColor": [255, 255, 255],
                    "confidence": 0.9,
                }
            ],
        },
        "before_inpaint": {
            "pageId": page_id,
            "sourceAssetId": str(uuid.uuid4()),
            "inputAssetId": str(uuid.uuid4()),
            "textMaskAssetId": None,
            "bubbles": [stored_bubble],
            "method": "solid",
            "fillColor": "#FFFFFF",
        },
        "before_render": {
            "pageId": page_id,
            "inputAssetId": str(uuid.uuid4()),
            "bubbles": [BubbleState().to_dict()],
            "renderConfig": {},
        },
    }

    for directory in sorted(
        path
        for path in plugin_root.iterdir()
        if path.is_dir() and (path / "plugin.json").is_file()
    ):
        archive = parse_archive(build_archive(directory))
        module_name, class_name = archive.manifest.entrypoint.split(":", 1)
        namespace: dict[str, object] = {}
        source = (directory / module_name).read_text(encoding="utf-8")
        validate_hook_source_contract(
            archive.manifest,
            source,
            filename=module_name,
        )
        exec(compile(source, module_name, "exec"), namespace)
        plugin = namespace[class_name]()
        context = type(
            "Context",
            (),
            {
                "config": {
                    name: field["default"]
                    for name, field in archive.manifest.config_schema.items()
                },
                "job_id": "job",
                "book_id": "book",
                "chapter_id": "chapter",
                "mode": "standard",
                "logger": type(
                    "Logger",
                    (),
                    {"info": staticmethod(lambda *_args, **_kwargs: None)},
                )(),
            },
        )()
        for hook in archive.manifest.hooks:
            payload = sample_payloads[hook]
            output = getattr(plugin, hook)(context, payload)
            phase, step = hook.split("_", 1)
            if step in {"job", "pipeline"}:
                validate_hook_data(output)
            else:
                validate_atomic_hook_data(step, phase, output)
            if hook == "before_inpaint":
                for bubble in output["bubbles"]:
                    validate_bubble_payload(bubble, render=False)
            elif hook == "before_render":
                for bubble in output["bubbles"]:
                    validate_bubble_payload(bubble, render=True)


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


def test_plugin_source_validation_supports_positional_only_hook_data() -> None:
    manifest = parse_manifest(_valid_manifest())
    validate_hook_source_contract(
        manifest,
        (
            "class Plugin:\n"
            "    def after_translate(self, context, /, data):\n"
            "        result = dict(data)\n"
            "        result['translations'] = list(data['translations'])\n"
            "        return result\n"
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
            require_action=True,
        )


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"assistant_message":"ok","target_proposal":null}\n```',
        '{"assistant_message":"ok","target_proposal":null,"legacy":true}',
        '{"assistant_message":1,"target_proposal":null}',
        (
            '{"assistant_message":"ok","target_proposal":'
            '{"plugin_id":"demo","display_name":"Demo",'
            '"supported_steps":[],"supported_modes":[],"extra":true}}'
        ),
        (
            '{"assistant_message":"ok","target_proposal":'
            '{"plugin_id":"demo","display_name":"Demo",'
            '"supported_steps":["legacy"],"supported_modes":["standard"]}}'
        ),
    ],
)
def test_plugin_agent_planning_requires_the_exact_current_envelope(
    content: str,
) -> None:
    with pytest.raises(OpenAICompatibleBusinessRetryableError):
        PluginAgentController._parse_agent_envelope(content)


@pytest.mark.parametrize(
    "content",
    [
        '{"assistant_message":"x","action":{"tool":"unknown","args":{}}}',
        '{"assistant_message":"x","action":{"tool":"finish"}}',
        (
            '{"assistant_message":"x","action":'
            '{"tool":"finish","args":{},"legacy":true}}'
        ),
        '{"assistant_message":"x","action":{"tool":"finish","args":{"legacy":true}}}',
        '{"assistant_message":"x","action":{"tool":"write_file","args":{"path":1,"content":"x"}}}',
    ],
)
def test_plugin_agent_execution_requires_a_current_tool_action(
    content: str,
) -> None:
    with pytest.raises(OpenAICompatibleBusinessRetryableError):
        PluginAgentController._parse_agent_envelope(
            content,
            require_action=True,
        )


def test_plugin_agent_tool_result_requires_the_current_shape() -> None:
    with pytest.raises(TypeError, match="结果字段"):
        PluginAgentController._validate_tool_result(
            "read_file",
            {
                "success": True,
                "path": "plugin.py",
                "content": "code",
                "preview": "code",
                "legacy": True,
            },
        )


def test_plugin_agent_tools_require_current_posix_paths(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    tools = PluginAgentWorktreeTools(
        worktree=worktree,
        skill_markdown="skill",
        control_requested=lambda: False,
    )

    for path in ("../outside.py", "/absolute.py", "folder\\legacy.py", " file.py"):
        with pytest.raises(ValueError):
            tools.run_tool(
                "write_file",
                {"path": path, "content": "value = 1\n"},
            )


def test_plugin_agent_execution_keeps_full_tool_history() -> None:
    controller = PluginAgentController()
    session = PluginAgentSession(
        session_id="history",
        mode="create",
        locked_target=LockedPluginTarget(
            mode="create",
            plugin_id="history_v3",
            display_name="History v3",
            plugin_dir="worktree://history_v3",
        ),
    )
    content = "x" * 2_000
    system_prompt = controller._build_execution_system_prompt(
        session,
        [
            {
                "tool": "read_file",
                "args": {"path": "plugin.py"},
                "result": {
                    "success": True,
                    "path": "plugin.py",
                    "content": content,
                    "preview": content[:1_200],
                },
            }
        ],
        13,
    )

    assert content in system_prompt
    assert "当前迭代: 13" in system_prompt
    assert "/12" not in system_prompt


def test_plugin_agent_persists_the_full_tool_result() -> None:
    content = "x" * 2_000
    payload = PluginAgentController._build_tool_result_payload(
        "read_file",
        {
            "success": True,
            "path": "plugin.py",
            "content": content,
            "preview": content[:1_200],
        },
        "tool-1",
    )

    assert payload["debug_result"]["content"] == content


def test_plugin_agent_stream_callback_observes_job_control() -> None:
    class Transport:
        def complete(
            self,
            request,
            *,
            resolved_invocation=None,
            before_request=None,
        ):
            assert before_request is not None
            before_request()
            callback = request.runtime_options.on_stream_chunk
            assert callback is not None
            callback("{", "{")
            raise AssertionError("control callback should stop the stream")

    class Tools:
        checks = 0

        def is_control_requested(self) -> bool:
            self.checks += 1
            return self.checks >= 3

    controller = PluginAgentController(transport=Transport())
    session = PluginAgentSession(
        session_id="stream-control",
        mode="create",
        locked_target=LockedPluginTarget(
            mode="create",
            plugin_id="stream_control",
            display_name="Stream control",
            plugin_dir="worktree://stream_control",
        ),
        messages=[
            PluginAgentMessage(
                id="user-1",
                role="user",
                content="create plugin",
            )
        ],
    )
    config = {
        "provider": "custom",
        "api_key": "key",
        "model_name": "model",
        "custom_base_url": "https://example.com/v1",
        "credential_version_id": None,
        "openai_options": OpenAICompatibleOptions.from_dict(
            {
                "request": {
                    "force_json_output": True,
                    "temperature": None,
                    "extra_body": {},
                },
                "execution": {
                    "use_stream": True,
                    "rpm_limit": 0,
                    "transport_retries": 0,
                    "business_retries": 0,
                },
            }
        ),
    }

    with pytest.raises(PluginAgentControlRequested):
        controller.execute(
            session,
            "skill",
            config,
            Tools(),
            lambda *_args: None,
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
