"""Transactional plugin v3 registry with immutable package publication."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from sqlalchemy import Engine, case, delete, insert, select, update

from src.backend_v2.jobs.repository import utcnow
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.plugins.contract import (
    PluginContractError,
    default_config,
    parse_manifest,
    validate_config,
)
from src.backend_v2.plugins.package import (
    build_archive,
    directory_checksum,
    extract_archive,
    parse_archive,
)
from src.backend_v2.plugins.snapshots import enabled_plugin_snapshots
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    idempotency_records,
    job_plugin_snapshots,
    operation_plugin_snapshots,
    plugin_current_versions,
    plugin_versions,
    plugins,
)


class PluginNotFound(LookupError):
    pass


class PluginConflict(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details or {}


class PluginLocked(PluginConflict):
    pass


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


class PluginRegistry:
    def __init__(self, *, data_root: Path, engine: Engine) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.plugins_root = self.data_root / "plugins"
        self.temp_root = self.data_root / "temp" / "plugins"
        self.plugins_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def reset_runtime_enabled(self) -> None:
        with immediate_transaction(self.engine) as connection:
            connection.execute(
                update(plugins).values(
                    runtime_enabled=plugins.c.default_enabled,
                    state=case(
                        (plugins.c.state == "error", "error"),
                        (
                            plugins.c.default_enabled.is_(True),
                            "enabled",
                        ),
                        else_="disabled",
                    ),
                    updated_at=utcnow(),
                )
            )

    def list_plugins(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        plugins,
                        plugin_current_versions.c.plugin_version_id,
                        plugin_current_versions.c.revision.label(
                            "current_revision"
                        ),
                        plugin_versions.c.version.label(
                            "package_version"
                        ),
                        plugin_versions.c.manifest_json,
                        plugin_versions.c.config_schema_json,
                    )
                    .join(
                        plugin_current_versions,
                        plugin_current_versions.c.plugin_id
                        == plugins.c.id,
                    )
                    .join(
                        plugin_versions,
                        plugin_versions.c.id
                        == plugin_current_versions.c.plugin_version_id,
                    )
                    .order_by(plugins.c.name, plugins.c.id)
                ).mappings()
            )
        return {"items": [self._dto(row) for row in rows]}

    def get_plugin(self, plugin_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    plugins,
                    plugin_current_versions.c.plugin_version_id,
                    plugin_current_versions.c.revision.label(
                        "current_revision"
                    ),
                    plugin_versions.c.version.label("package_version"),
                    plugin_versions.c.manifest_json,
                    plugin_versions.c.config_schema_json,
                    plugin_versions.c.package_relative_path,
                    plugin_versions.c.checksum,
                )
                .join(
                    plugin_current_versions,
                    plugin_current_versions.c.plugin_id == plugins.c.id,
                )
                .join(
                    plugin_versions,
                    plugin_versions.c.id
                    == plugin_current_versions.c.plugin_version_id,
                )
                .where(plugins.c.id == plugin_id)
            ).mappings().one_or_none()
        if row is None:
            raise PluginNotFound("plugin not found")
        return self._dto(row)

    def import_archive(
        self,
        *,
        data: bytes,
        base_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        parsed = parse_archive(data)
        plugin_id = parsed.manifest.plugin_id
        request = {
            "pluginId": plugin_id,
            "baseRevision": base_revision,
            "archiveChecksum": parsed.archive_checksum,
        }
        request_hash = hashlib.sha256(
            _json(request).encode("utf-8")
        ).hexdigest()
        scope = f"POST:importPlugin:{plugin_id}"
        now = utcnow()
        replay = self._replay_idempotency(
            scope=scope,
            key=idempotency_key,
            request_hash=request_hash,
            now=now,
        )
        if replay is not None:
            return replay

        plugin_version_id = str(uuid.uuid4())
        staging = self.temp_root / f"import-{plugin_version_id}"
        final = (
            self.plugins_root
            / plugin_id
            / "versions"
            / plugin_version_id
        )
        tree_checksum = extract_archive(parsed, staging)
        published = False
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            with immediate_transaction(self.engine) as connection:
                replay = self._idempotency_replay_in_connection(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    now=now,
                )
                if replay is not None:
                    return replay
                plugin = connection.execute(
                    select(plugins).where(plugins.c.id == plugin_id)
                ).mappings().one_or_none()
                current = connection.execute(
                    select(plugin_current_versions).where(
                        plugin_current_versions.c.plugin_id == plugin_id
                    )
                ).mappings().one_or_none()
                current_revision = (
                    int(current["revision"]) if current is not None else 0
                )
                if current_revision != base_revision:
                    raise PluginConflict(
                        "plugin current version revision changed",
                        details={
                            "pluginId": plugin_id,
                            "currentRevision": current_revision,
                        },
                    )
                if plugin is None and base_revision != 0:
                    raise PluginConflict(
                        "new plugin import requires baseRevision 0"
                    )
                schema = parsed.manifest.config_schema
                if plugin is None:
                    config = default_config(schema)
                    connection.execute(
                        insert(plugins).values(
                            id=plugin_id,
                            name=parsed.manifest.display_name,
                            state=(
                                "enabled"
                                if parsed.manifest.default_enabled
                                else "disabled"
                            ),
                            author=parsed.manifest.author,
                            description=parsed.manifest.description,
                            default_enabled=(
                                parsed.manifest.default_enabled
                            ),
                            runtime_enabled=(
                                parsed.manifest.default_enabled
                            ),
                            config_json=_json(config),
                            config_revision=1,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                else:
                    loaded_config = _load(
                        str(plugin["config_json"]),
                        {},
                    )
                    if not isinstance(loaded_config, Mapping):
                        raise PluginConflict(
                            "stored plugin config is invalid"
                        )
                    config = validate_config(schema, loaded_config)
                    connection.execute(
                        update(plugins)
                        .where(plugins.c.id == plugin_id)
                        .values(
                            name=parsed.manifest.display_name,
                            author=parsed.manifest.author,
                            description=parsed.manifest.description,
                            config_json=_json(config),
                            error_message=None,
                            state=(
                                "enabled"
                                if plugin["runtime_enabled"]
                                else "disabled"
                            ),
                            updated_at=now,
                        )
                    )
                if final.exists():
                    raise PluginConflict(
                        "immutable plugin version directory already exists"
                    )
                os.replace(staging, final)
                published = True
                relative_path = final.relative_to(
                    self.data_root
                ).as_posix()
                connection.execute(
                    insert(plugin_versions).values(
                        id=plugin_version_id,
                        plugin_id=plugin_id,
                        version=parsed.manifest.package_version,
                        package_relative_path=relative_path,
                        checksum=tree_checksum,
                        manifest_json=_json(
                            parsed.manifest.to_dict()
                        ),
                        config_schema_json=_json(schema),
                        manifest_schema_version=3,
                        created_at=now,
                        updated_at=now,
                    )
                )
                next_revision = current_revision + 1
                if current is None:
                    connection.execute(
                        insert(plugin_current_versions).values(
                            plugin_id=plugin_id,
                            plugin_version_id=plugin_version_id,
                            revision=next_revision,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                else:
                    changed = connection.execute(
                        update(plugin_current_versions)
                        .where(
                            plugin_current_versions.c.plugin_id
                            == plugin_id,
                            plugin_current_versions.c.revision
                            == base_revision,
                        )
                        .values(
                            plugin_version_id=plugin_version_id,
                            revision=next_revision,
                            updated_at=now,
                        )
                    )
                    if changed.rowcount != 1:
                        raise PluginConflict(
                            "plugin current version revision changed"
                        )
                response = {
                    "pluginId": plugin_id,
                    "pluginVersionId": plugin_version_id,
                    "packageVersion": parsed.manifest.package_version,
                    "currentRevision": next_revision,
                }
                connection.execute(
                    insert(idempotency_records).values(
                        scope=scope,
                        key=idempotency_key,
                        request_hash=request_hash,
                        http_status=201,
                        response_json=_json(response),
                        resource_type="plugin_version",
                        resource_id=plugin_version_id,
                        created_at=now,
                        expires_at=now + timedelta(days=7),
                    )
                )
            return response
        except Exception:
            if published and final.exists():
                shutil.rmtree(final, ignore_errors=True)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def set_runtime_enabled(
        self,
        *,
        plugin_id: str,
        enabled: bool,
    ) -> dict[str, Any]:
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(plugins).where(plugins.c.id == plugin_id)
            ).mappings().one_or_none()
            if row is None:
                raise PluginNotFound("plugin not found")
            if row["state"] == "error" and enabled:
                raise PluginConflict(
                    "plugin with an integrity error cannot be enabled"
                )
            connection.execute(
                update(plugins)
                .where(plugins.c.id == plugin_id)
                .values(
                    runtime_enabled=enabled,
                    state="enabled" if enabled else "disabled",
                    updated_at=utcnow(),
                )
            )
        return self.get_plugin(plugin_id)

    def set_default_enabled(
        self,
        *,
        plugin_id: str,
        enabled: bool,
    ) -> dict[str, Any]:
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(plugins)
                .where(plugins.c.id == plugin_id)
                .values(default_enabled=enabled, updated_at=utcnow())
            )
            if changed.rowcount != 1:
                raise PluginNotFound("plugin not found")
        return self.get_plugin(plugin_id)

    def get_config(self, plugin_id: str) -> dict[str, Any]:
        plugin = self.get_plugin(plugin_id)
        return {
            "pluginId": plugin_id,
            "schema": plugin["configSchema"],
            "value": plugin["config"],
            "configRevision": plugin["configRevision"],
        }

    def update_config(
        self,
        *,
        plugin_id: str,
        base_revision: int,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(
                    plugins.c.config_revision,
                    plugin_versions.c.config_schema_json,
                )
                .join(
                    plugin_current_versions,
                    plugin_current_versions.c.plugin_id == plugins.c.id,
                )
                .join(
                    plugin_versions,
                    plugin_versions.c.id
                    == plugin_current_versions.c.plugin_version_id,
                )
                .where(plugins.c.id == plugin_id)
            ).mappings().one_or_none()
            if row is None:
                raise PluginNotFound("plugin not found")
            if int(row["config_revision"]) != base_revision:
                raise PluginConflict("plugin config revision changed")
            schema = _load(str(row["config_schema_json"]), {})
            if not isinstance(schema, Mapping):
                raise PluginConflict("plugin config schema is invalid")
            normalized = validate_config(schema, config)
            changed = connection.execute(
                update(plugins)
                .where(
                    plugins.c.id == plugin_id,
                    plugins.c.config_revision == base_revision,
                )
                .values(
                    config_json=_json(normalized),
                    config_revision=base_revision + 1,
                    updated_at=utcnow(),
                )
            )
            if changed.rowcount != 1:
                raise PluginConflict("plugin config revision changed")
        return self.get_config(plugin_id)

    def export_current(self, plugin_id: str) -> tuple[bytes, str]:
        plugin = self.get_plugin(plugin_id)
        version_id = str(plugin["pluginVersionId"])
        relative = plugin.get("packageRelativePath")
        if not isinstance(relative, str):
            with self.engine.connect() as connection:
                relative = connection.execute(
                    select(plugin_versions.c.package_relative_path).where(
                        plugin_versions.c.id == version_id
                    )
                ).scalar_one()
        root = self._managed_path(str(relative))
        return (
            build_archive(root),
            f"{plugin_id}-{plugin['packageVersion']}.zip",
        )

    def refresh(self) -> dict[str, Any]:
        checked = 0
        failed = 0
        with immediate_transaction(self.engine) as connection:
            rows = list(connection.execute(select(plugin_versions)).mappings())
            failed_plugins: dict[str, str] = {}
            for row in rows:
                checked += 1
                try:
                    root = self._managed_path(
                        str(row["package_relative_path"])
                    )
                    actual = directory_checksum(root)
                    if actual != row["checksum"]:
                        raise PluginContractError(
                            "immutable package checksum mismatch"
                        )
                    raw_manifest = _load(str(row["manifest_json"]), {})
                    if not isinstance(raw_manifest, Mapping):
                        raise PluginContractError(
                            "stored manifest is invalid"
                        )
                    parse_manifest(raw_manifest)
                except Exception as exc:
                    failed += 1
                    failed_plugins[str(row["plugin_id"])] = (
                        redact_sensitive_text(exc)
                    )
            for plugin_id, message in failed_plugins.items():
                connection.execute(
                    update(plugins)
                    .where(plugins.c.id == plugin_id)
                    .values(
                        state="error",
                        runtime_enabled=False,
                        error_message=message,
                        updated_at=utcnow(),
                    )
                )
            healthy_ids = {
                str(row["plugin_id"]) for row in rows
            } - set(failed_plugins)
            for plugin_id in healthy_ids:
                connection.execute(
                    update(plugins)
                    .where(plugins.c.id == plugin_id)
                    .values(
                        state=case(
                            (
                                plugins.c.runtime_enabled.is_(True),
                                "enabled",
                            ),
                            else_="disabled",
                        ),
                        error_message=None,
                        updated_at=utcnow(),
                    )
                )
        return {"checkedVersions": checked, "failedVersions": failed}

    def delete_plugin(
        self,
        *,
        plugin_id: str,
        base_revision: int,
    ) -> dict[str, Any]:
        plugin_root = self.plugins_root / plugin_id
        trash = self.temp_root / f"delete-{uuid.uuid4()}"
        moved = False
        try:
            if plugin_root.exists():
                os.replace(plugin_root, trash)
                moved = True
            with immediate_transaction(self.engine) as connection:
                current = connection.execute(
                    select(plugin_current_versions.c.revision).where(
                        plugin_current_versions.c.plugin_id == plugin_id
                    )
                ).scalar_one_or_none()
                if current is None:
                    raise PluginNotFound("plugin not found")
                if int(current) != base_revision:
                    raise PluginConflict(
                        "plugin current version revision changed"
                    )
                version_ids = tuple(
                    connection.execute(
                        select(plugin_versions.c.id).where(
                            plugin_versions.c.plugin_id == plugin_id
                        )
                    ).scalars()
                )
                if version_ids:
                    job_reference = connection.execute(
                        select(job_plugin_snapshots.c.job_id)
                        .where(
                            job_plugin_snapshots.c.plugin_version_id.in_(
                                version_ids
                            )
                        )
                        .limit(1)
                    ).scalar_one_or_none()
                    operation_reference = connection.execute(
                        select(operation_plugin_snapshots.c.operation_id)
                        .where(
                            operation_plugin_snapshots.c.plugin_version_id.in_(
                                version_ids
                            )
                        )
                        .limit(1)
                    ).scalar_one_or_none()
                    if (
                        job_reference is not None
                        or operation_reference is not None
                    ):
                        raise PluginLocked(
                            "plugin version is referenced by task history"
                        )
                connection.execute(
                    delete(plugin_current_versions).where(
                        plugin_current_versions.c.plugin_id == plugin_id
                    )
                )
                connection.execute(
                    delete(plugin_versions).where(
                        plugin_versions.c.plugin_id == plugin_id
                    )
                )
                connection.execute(
                    delete(plugins).where(plugins.c.id == plugin_id)
                )
            if moved:
                shutil.rmtree(trash)
            return {"deleted": True, "pluginId": plugin_id}
        except Exception:
            if moved and trash.exists() and not plugin_root.exists():
                plugin_root.parent.mkdir(parents=True, exist_ok=True)
                os.replace(trash, plugin_root)
            raise

    def enabled_snapshots(self) -> dict[str, dict[str, Any]]:
        with self.engine.connect() as connection:
            return enabled_plugin_snapshots(connection)

    def _managed_path(self, relative: str) -> Path:
        path = (self.data_root / Path(relative)).resolve()
        try:
            path.relative_to(self.plugins_root)
        except ValueError as exc:
            raise PluginConflict(
                "plugin package path escapes the managed root"
            ) from exc
        return path

    @staticmethod
    def _dto(row: Mapping[str, Any]) -> dict[str, Any]:
        manifest = _load(str(row["manifest_json"]), {})
        schema = _load(str(row["config_schema_json"]), {})
        config = _load(str(row["config_json"]), {})
        result = {
            "pluginId": str(row["id"]),
            "displayName": str(row["name"]),
            "author": str(row["author"]),
            "description": str(row["description"]),
            "state": str(row["state"]),
            "defaultEnabled": bool(row["default_enabled"]),
            "runtimeEnabled": bool(row["runtime_enabled"]),
            "config": config if isinstance(config, dict) else {},
            "configRevision": int(row["config_revision"]),
            "errorMessage": row["error_message"],
            "pluginVersionId": str(row["plugin_version_id"]),
            "packageVersion": str(row["package_version"]),
            "currentRevision": int(row["current_revision"]),
            "manifest": manifest if isinstance(manifest, dict) else {},
            "configSchema": schema if isinstance(schema, dict) else {},
        }
        if "package_relative_path" in row:
            result["packageRelativePath"] = str(
                row["package_relative_path"]
            )
        if "checksum" in row:
            result["checksum"] = str(row["checksum"])
        return result

    def _replay_idempotency(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        now,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            return self._idempotency_replay_in_connection(
                connection,
                scope=scope,
                key=key,
                request_hash=request_hash,
                now=now,
            )

    @staticmethod
    def _idempotency_replay_in_connection(
        connection,
        *,
        scope: str,
        key: str,
        request_hash: str,
        now,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            select(idempotency_records).where(
                idempotency_records.c.scope == scope,
                idempotency_records.c.key == key,
                idempotency_records.c.expires_at > now,
            )
        ).mappings().one_or_none()
        if row is None:
            return None
        if str(row["request_hash"]) != request_hash:
            raise PluginConflict(
                "Idempotency-Key was reused for different plugin input"
            )
        value = _load(str(row["response_json"]), {})
        if not isinstance(value, dict):
            raise PluginConflict("stored idempotency response is invalid")
        return value
