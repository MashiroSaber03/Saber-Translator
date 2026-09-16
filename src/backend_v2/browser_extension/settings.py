"""Secret-free settings captured for one disposable comic session."""

import json
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.serialization import canonical_json
from src.backend_v2.settings.scope import SettingsScope
from src.backend_v2.storage.schema import (
    app_settings, browser_sessions, browser_session_credentials, provider_settings,
    credentials, credential_versions,
)


def session_settings(connection: Connection, book_id: str) -> dict[str, Any]:
    stored = connection.execute(
        select(browser_sessions.c.id, browser_sessions.c.settings_json).where(
            browser_sessions.c.book_id == book_id,
        )
    ).mappings().one_or_none()
    if stored is None or stored["settings_json"] is None:
        raise ValueError("网页翻译配置快照不存在，请重新开始翻译")
    references = {
        (row.domain, row.provider): row.credential_version_id
        for row in connection.execute(
            select(credentials.c.domain, credentials.c.provider,
                   browser_session_credentials.c.credential_version_id)
            .select_from(browser_session_credentials)
            .join(credential_versions, credential_versions.c.id == browser_session_credentials.c.credential_version_id)
            .join(credentials, credentials.c.id == credential_versions.c.credential_id)
            .where(browser_session_credentials.c.session_id == stored["id"])
        )
    }
    document = json.loads(stored["settings_json"])
    for row in document["providerSettings"]:
        row["credentialVersionId"] = references.get((row["domain"], row["provider"]))
    return document


def capture_session_settings(connection: Connection, book_id: str) -> None:
    session_id = connection.execute(
        select(browser_sessions.c.id).where(browser_sessions.c.book_id == book_id)
    ).scalar_one()
    scope = SettingsScope(browser_extension=True)
    document: dict[str, Any] = {"settings": [], "providerSettings": []}
    # Translation services are shared with the translator; only page style is plugin-owned.
    for table, key, domains in (
        (app_settings, "settings", ("translation", "browser_extension:text_style_defaults")),
        (provider_settings, "providerSettings", ("translation", "hq", "ai_vision_ocr", "ocr")),
    ):
        for row in connection.execute(
            select(table).where(
                table.c.owner_user_id == effective_owner_id(),
                table.c.domain.in_(domains),
            )
        ).mappings():
            value = {
                "domain": scope.public_domain(row["domain"]),
                "payload": json.loads(row["payload_json"]),
                "revision": row["revision"],
            }
            if key == "providerSettings":
                value.update(
                    provider=row["provider"],
                    credentialVersionId=row["credential_version_id"],
                )
            document[key].append(value)
    scope.add_factory_defaults(document, ("text_style_defaults",))
    credential_ids = {
        row.pop("credentialVersionId") for row in document["providerSettings"]
    } - {None}
    connection.execute(
        update(browser_sessions).where(browser_sessions.c.id == session_id)
        .values(settings_json=canonical_json(document))
    )
    connection.execute(delete(browser_session_credentials).where(
        browser_session_credentials.c.session_id == session_id,
    ))
    if credential_ids:
        connection.execute(insert(browser_session_credentials), [
            {"session_id": session_id, "credential_version_id": credential_id}
            for credential_id in sorted(credential_ids)
        ])
