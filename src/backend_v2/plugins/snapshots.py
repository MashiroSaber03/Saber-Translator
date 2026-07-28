"""Database-only helpers for freezing enabled plugin versions.

This module intentionally contains no package loading or dynamic imports, so
API command paths can snapshot plugin metadata without executing plugin code.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from src.backend_v2.storage.schema import (
    plugin_current_versions,
    plugin_versions,
    plugins,
)


def enabled_plugin_snapshots(
    connection: Connection,
) -> dict[str, dict[str, Any]]:
    """Return the immutable version/config snapshot for all enabled plugins."""

    rows = list(
        connection.execute(
            select(
                plugins.c.id,
                plugins.c.config_json,
                plugins.c.config_revision,
                plugin_current_versions.c.plugin_version_id,
                plugin_versions.c.manifest_json,
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
            .where(
                plugins.c.runtime_enabled.is_(True),
                plugins.c.state != "error",
            )
            .order_by(plugins.c.id)
        ).mappings()
    )
    snapshots: dict[str, dict[str, Any]] = {}
    for row in rows:
        manifest = _object(row["manifest_json"])
        config = _object(row["config_json"])
        snapshots[str(row["plugin_version_id"])] = {
            "pluginId": str(row["id"]),
            "configRevision": int(row["config_revision"]),
            "config": config,
            "hooks": manifest.get("hooks", []),
        }
    return snapshots


def _object(raw: object) -> dict[str, Any]:
    try:
        value = json.loads(str(raw)) if raw else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}
