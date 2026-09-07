"""Secret-free settings captured for one disposable comic session."""

import json
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.serialization import canonical_json
from src.backend_v2.settings.scope import SettingsScope
from src.backend_v2.storage.schema import app_settings, book_settings, provider_settings

SNAPSHOT_DOMAIN = "browser_extension_snapshot"


def session_settings(connection: Connection, book_id: str) -> dict[str, Any]:
    stored = connection.execute(
        select(book_settings.c.payload_json).where(
            book_settings.c.book_id == book_id,
            book_settings.c.domain == SNAPSHOT_DOMAIN,
        )
    ).scalar_one_or_none()
    if stored is None:
        raise ValueError("网页翻译配置快照不存在，请重新开始翻译")
    return json.loads(stored)


def capture_session_settings(connection: Connection, book_id: str) -> None:
    scope = SettingsScope(browser_extension=True)
    document: dict[str, Any] = {"settings": [], "providerSettings": []}
    for table, key in ((app_settings, "settings"), (provider_settings, "providerSettings")):
        for row in connection.execute(
            select(table).where(
                table.c.owner_user_id == effective_owner_id(),
                scope.condition(table.c.domain),
            )
        ).mappings():
            value = {
                "domain": scope.public_domain(row["domain"]),
                "payload": json.loads(row["payload_json"]),
                "revision": row["revision"],
                "schemaVersion": row["schema_version"],
            }
            if key == "providerSettings":
                value.update(
                    provider=row["provider"],
                    credentialVersionId=row["credential_version_id"],
                )
            document[key].append(value)
    scope.add_factory_defaults(document, ())
    connection.execute(
        insert(book_settings).values(
            book_id=book_id,
            domain=SNAPSHOT_DOMAIN,
            payload_json=canonical_json(document),
            schema_version=1,
        )
    )
