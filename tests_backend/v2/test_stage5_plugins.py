from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
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
    PluginRegistry,
)
from src.backend_v2.plugins.runtime import (
    PluginHookFailure,
    PluginJobRuntime,
)
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
    books,
    chapters,
    job_plugin_snapshots,
    metadata,
    operation_plugin_snapshots,
    pages,
    jobs as jobs_table,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.core.plugin_agent.controller import PluginAgentController
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetryableError,
)


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
        "config_schema": {
            "prefix": {
                "type": "text",
                "default": "[v3]",
            },
            "strict": {
                "type": "boolean",
                "default": False,
            },
        },
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
    assert registry.enabled_snapshots() == {}
    registry.reset_runtime_enabled()
    snapshots = registry.enabled_snapshots()
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
    frozen = registry.enabled_snapshots()
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
    with engine.begin() as connection:
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
    operation = OperationRepository(engine).create_internal(
        kind="page_detect",
        executor_role="worker",
        request_payload={},
        page_id=page_id,
        base_revision=1,
    )
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
    created = jobs.create_batch(
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
                plugin_snapshots=registry.enabled_snapshots(),
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
            "plugin_id": session.locked_target.plugin_id,
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
            "failure_policy": "continue",
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
            "failure_policy": "continue",
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
            "failure_policy": "continue",
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
            "failure_policy": "continue",
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
