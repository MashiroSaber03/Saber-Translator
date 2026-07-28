"""Repositories for settings, immutable credentials/plugins, fonts, and rate limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import uuid
from typing import Any

from sqlalchemy import Engine, and_, insert, select, update
from sqlalchemy.exc import IntegrityError

from src.backend_v2.storage.schema import (
    app_settings,
    credential_current_versions,
    credential_versions,
    credentials,
    fonts,
    plugin_current_versions,
    plugin_versions,
    plugins,
    provider_rate_limits,
    provider_settings,
)


class RevisionConflict(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class SettingMutation:
    domain: str
    payload: dict[str, Any]
    base_revision: int
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class ProviderSettingMutation:
    domain: str
    provider: str
    payload: dict[str, Any]
    base_revision: int
    credential_version_id: str | None = None
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class CredentialEdit:
    domain: str
    provider: str
    secret: dict[str, Any]
    base_revision: int
    credential_id: str | None = None


class SettingsRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def save_transaction(
        self,
        *,
        settings: tuple[SettingMutation, ...] = (),
        providers: tuple[ProviderSettingMutation, ...] = (),
        credentials_edits: tuple[CredentialEdit, ...] = (),
    ) -> dict[str, list[dict[str, object]]]:
        with self.engine.begin() as connection:
            setting_results = [
                self._save_setting(connection, mutation) for mutation in settings
            ]
            provider_results = [
                self._save_provider_setting(connection, mutation)
                for mutation in providers
            ]
            credential_results = [
                self._save_credential(connection, edit) for edit in credentials_edits
            ]
        return {
            "settings": setting_results,
            "providerSettings": provider_results,
            "credentials": credential_results,
        }

    @staticmethod
    def _save_setting(connection: object, mutation: SettingMutation) -> dict[str, object]:
        if mutation.base_revision < 0 or mutation.schema_version < 1:
            raise ValueError("setting revisions must be non-negative")
        payload_json = _canonical_json(_require_object(mutation.payload, "setting payload"))
        current = connection.execute(  # type: ignore[attr-defined]
            select(app_settings.c.revision).where(app_settings.c.domain == mutation.domain)
        ).scalar_one_or_none()
        if current is None:
            if mutation.base_revision != 0:
                raise RevisionConflict(f"setting {mutation.domain} does not exist at requested revision")
            connection.execute(  # type: ignore[attr-defined]
                insert(app_settings).values(
                    domain=mutation.domain,
                    revision=1,
                    payload_json=payload_json,
                    schema_version=mutation.schema_version,
                )
            )
            return {"domain": mutation.domain, "revision": 1}
        if current != mutation.base_revision:
            raise RevisionConflict(f"setting {mutation.domain} revision changed")
        changed = connection.execute(  # type: ignore[attr-defined]
            update(app_settings)
            .where(
                app_settings.c.domain == mutation.domain,
                app_settings.c.revision == mutation.base_revision,
            )
            .values(
                revision=mutation.base_revision + 1,
                payload_json=payload_json,
                schema_version=mutation.schema_version,
                updated_at=_utcnow(),
            )
        )
        if changed.rowcount != 1:
            raise RevisionConflict(f"setting {mutation.domain} revision changed")
        return {"domain": mutation.domain, "revision": mutation.base_revision + 1}

    @staticmethod
    def _save_provider_setting(
        connection: object,
        mutation: ProviderSettingMutation,
    ) -> dict[str, object]:
        if mutation.base_revision < 0 or mutation.schema_version < 1:
            raise ValueError("provider setting revisions must be non-negative")
        payload_json = _canonical_json(
            _require_object(mutation.payload, "provider setting payload")
        )
        key = and_(
            provider_settings.c.domain == mutation.domain,
            provider_settings.c.provider == mutation.provider,
        )
        current = connection.execute(  # type: ignore[attr-defined]
            select(provider_settings.c.revision).where(key)
        ).scalar_one_or_none()
        values = {
            "payload_json": payload_json,
            "schema_version": mutation.schema_version,
            "credential_version_id": mutation.credential_version_id,
        }
        if current is None:
            if mutation.base_revision != 0:
                raise RevisionConflict("provider setting does not exist at requested revision")
            connection.execute(  # type: ignore[attr-defined]
                insert(provider_settings).values(
                    domain=mutation.domain,
                    provider=mutation.provider,
                    revision=1,
                    **values,
                )
            )
            revision = 1
        else:
            if current != mutation.base_revision:
                raise RevisionConflict("provider setting revision changed")
            changed = connection.execute(  # type: ignore[attr-defined]
                update(provider_settings)
                .where(key, provider_settings.c.revision == mutation.base_revision)
                .values(
                    revision=mutation.base_revision + 1,
                    updated_at=_utcnow(),
                    **values,
                )
            )
            if changed.rowcount != 1:
                raise RevisionConflict("provider setting revision changed")
            revision = mutation.base_revision + 1
        return {
            "domain": mutation.domain,
            "provider": mutation.provider,
            "revision": revision,
        }

    @staticmethod
    def _save_credential(connection: object, edit: CredentialEdit) -> dict[str, object]:
        secret = _require_object(edit.secret, "credential secret")
        if not secret or not any(value not in (None, "") for value in secret.values()):
            raise ValueError("credential secret must contain at least one value")
        secret_json = _canonical_json(secret)
        fingerprint = hashlib.sha256(secret_json.encode("utf-8")).hexdigest()

        if edit.credential_id is None:
            if edit.base_revision != 0:
                raise RevisionConflict("credential does not exist at requested revision")
            credential_id = str(uuid.uuid4())
            version_id = str(uuid.uuid4())
            connection.execute(  # type: ignore[attr-defined]
                insert(credentials).values(
                    id=credential_id,
                    domain=edit.domain,
                    provider=edit.provider,
                )
            )
            connection.execute(  # type: ignore[attr-defined]
                insert(credential_versions).values(
                    id=version_id,
                    credential_id=credential_id,
                    version=1,
                    secret_json=secret_json,
                    key_fingerprint=fingerprint,
                )
            )
            connection.execute(  # type: ignore[attr-defined]
                insert(credential_current_versions).values(
                    credential_id=credential_id,
                    credential_version_id=version_id,
                    revision=1,
                )
            )
            return {
                "credentialId": credential_id,
                "domain": edit.domain,
                "provider": edit.provider,
                "hasKey": True,
                "currentVersion": 1,
                "revision": 1,
            }

        current = connection.execute(  # type: ignore[attr-defined]
            select(
                credentials.c.domain,
                credentials.c.provider,
                credential_current_versions.c.revision,
                credential_versions.c.version,
            )
            .join(
                credential_current_versions,
                credential_current_versions.c.credential_id == credentials.c.id,
            )
            .join(
                credential_versions,
                credential_versions.c.id
                == credential_current_versions.c.credential_version_id,
            )
            .where(credentials.c.id == edit.credential_id)
        ).mappings().one_or_none()
        if current is None or current["revision"] != edit.base_revision:
            raise RevisionConflict("credential revision changed")
        if current["domain"] != edit.domain or current["provider"] != edit.provider:
            raise ValueError("credential identity domain/provider cannot change")

        version = int(current["version"]) + 1
        version_id = str(uuid.uuid4())
        connection.execute(  # type: ignore[attr-defined]
            insert(credential_versions).values(
                id=version_id,
                credential_id=edit.credential_id,
                version=version,
                secret_json=secret_json,
                key_fingerprint=fingerprint,
            )
        )
        changed = connection.execute(  # type: ignore[attr-defined]
            update(credential_current_versions)
            .where(
                credential_current_versions.c.credential_id == edit.credential_id,
                credential_current_versions.c.revision == edit.base_revision,
            )
            .values(
                credential_version_id=version_id,
                revision=edit.base_revision + 1,
                updated_at=_utcnow(),
            )
        )
        if changed.rowcount != 1:
            raise RevisionConflict("credential revision changed")
        return {
            "credentialId": edit.credential_id,
            "domain": edit.domain,
            "provider": edit.provider,
            "hasKey": True,
            "currentVersion": version,
            "revision": edit.base_revision + 1,
        }

    def credential_summaries(self) -> list[dict[str, object]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    credentials.c.id,
                    credentials.c.domain,
                    credentials.c.provider,
                    credential_current_versions.c.revision,
                    credential_versions.c.version,
                )
                .join(
                    credential_current_versions,
                    credential_current_versions.c.credential_id == credentials.c.id,
                )
                .join(
                    credential_versions,
                    credential_versions.c.id
                    == credential_current_versions.c.credential_version_id,
                )
            ).mappings()
            return [
                {
                    "credentialId": row["id"],
                    "domain": row["domain"],
                    "provider": row["provider"],
                    "hasKey": True,
                    "currentVersion": row["version"],
                    "revision": row["revision"],
                }
                for row in rows
            ]

    def resolve_secret(self, credential_version_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(credential_versions.c.secret_json).where(
                    credential_versions.c.id == credential_version_id
                )
            ).scalar_one()
        return _require_object(json.loads(value), "stored credential secret")


class PluginVersionRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def install_version(
        self,
        *,
        plugin_id: str | None,
        name: str,
        version: str,
        package_relative_path: str,
        checksum: str,
        manifest: dict[str, Any],
        base_revision: int,
    ) -> dict[str, object]:
        path = PurePosixPath(package_relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("plugin package path must be data-root-relative")
        with self.engine.begin() as connection:
            if plugin_id is None:
                if base_revision != 0:
                    raise RevisionConflict("new plugin requires base revision zero")
                plugin_id = str(uuid.uuid4())
                connection.execute(insert(plugins).values(id=plugin_id, name=name))
                current_revision = 0
            else:
                current_revision = connection.execute(
                    select(plugin_current_versions.c.revision).where(
                        plugin_current_versions.c.plugin_id == plugin_id
                    )
                ).scalar_one_or_none()
                if current_revision != base_revision:
                    raise RevisionConflict("plugin current version changed")

            version_id = str(uuid.uuid4())
            connection.execute(
                insert(plugin_versions).values(
                    id=version_id,
                    plugin_id=plugin_id,
                    version=version,
                    package_relative_path=path.as_posix(),
                    checksum=checksum,
                    manifest_json=_canonical_json(_require_object(manifest, "manifest")),
                )
            )
            if current_revision == 0:
                connection.execute(
                    insert(plugin_current_versions).values(
                        plugin_id=plugin_id,
                        plugin_version_id=version_id,
                        revision=1,
                    )
                )
                revision = 1
            else:
                changed = connection.execute(
                    update(plugin_current_versions)
                    .where(
                        plugin_current_versions.c.plugin_id == plugin_id,
                        plugin_current_versions.c.revision == base_revision,
                    )
                    .values(
                        plugin_version_id=version_id,
                        revision=base_revision + 1,
                        updated_at=_utcnow(),
                    )
                )
                if changed.rowcount != 1:
                    raise RevisionConflict("plugin current version changed")
                revision = base_revision + 1
        return {
            "pluginId": plugin_id,
            "pluginVersionId": version_id,
            "revision": revision,
        }


class FontRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def ensure_builtin(self, *, builtin_key: str, display_name: str) -> str:
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(fonts.c.id).where(fonts.c.builtin_key == builtin_key)
            ).scalar_one_or_none()
            if existing is not None:
                return str(existing)
            font_id = str(uuid.uuid4())
            connection.execute(
                insert(fonts).values(
                    id=font_id,
                    kind="builtin",
                    builtin_key=builtin_key,
                    display_name=display_name,
                )
            )
            return font_id

    def register_uploaded(self, *, asset_id: str, display_name: str) -> str:
        font_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                insert(fonts).values(
                    id=font_id,
                    kind="uploaded",
                    asset_id=asset_id,
                    display_name=display_name,
                )
            )
        return font_id


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: float


class ProviderRateLimiter:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def acquire(
        self,
        *,
        provider: str,
        credential_version_id: str,
        rpm_limit: int,
        now: datetime | None = None,
    ) -> RateLimitDecision:
        if rpm_limit < 1:
            raise ValueError("rpm_limit must be positive")
        current_time = now or _utcnow()
        window_cutoff = current_time - timedelta(minutes=1)

        for _attempt in range(8):
            with self.engine.begin() as connection:
                row = connection.execute(
                    select(provider_rate_limits).where(
                        provider_rate_limits.c.provider == provider,
                        provider_rate_limits.c.credential_version_id
                        == credential_version_id,
                    )
                ).mappings().one_or_none()
                if row is None:
                    try:
                        connection.execute(
                            insert(provider_rate_limits).values(
                                provider=provider,
                                credential_version_id=credential_version_id,
                                window_started_at=current_time,
                                request_count=1,
                                rpm_limit=rpm_limit,
                                revision=1,
                            )
                        )
                    except IntegrityError:
                        continue
                    return RateLimitDecision(
                        allowed=True,
                        remaining=rpm_limit - 1,
                        retry_after_seconds=0,
                    )

                revision = int(row["revision"])
                window_started_at = row["window_started_at"]
                if window_started_at <= window_cutoff:
                    count = 1
                    started_at = current_time
                    allowed = True
                elif int(row["request_count"]) < rpm_limit:
                    count = int(row["request_count"]) + 1
                    started_at = window_started_at
                    allowed = True
                else:
                    retry_after = max(
                        0.0,
                        (window_started_at + timedelta(minutes=1) - current_time).total_seconds(),
                    )
                    return RateLimitDecision(
                        allowed=False,
                        remaining=0,
                        retry_after_seconds=retry_after,
                    )

                changed = connection.execute(
                    update(provider_rate_limits)
                    .where(
                        provider_rate_limits.c.provider == provider,
                        provider_rate_limits.c.credential_version_id
                        == credential_version_id,
                        provider_rate_limits.c.revision == revision,
                    )
                    .values(
                        window_started_at=started_at,
                        request_count=count,
                        rpm_limit=rpm_limit,
                        revision=revision + 1,
                    )
                )
                if changed.rowcount == 1:
                    return RateLimitDecision(
                        allowed=allowed,
                        remaining=max(0, rpm_limit - count),
                        retry_after_seconds=0,
                    )
        raise RuntimeError("provider limiter CAS remained contended")
