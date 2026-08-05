"""Repositories for settings, immutable credentials, fonts, and rate limits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import Engine, and_, case, delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from src.backend_v2.serialization import canonical_json as _canonical_json
from src.backend_v2.timestamps import utcnow as _utcnow
from src.backend_v2.content.page_style import validate_text_style_defaults
from src.backend_v2.settings.validation import (
    validate_book_setting_payload,
    validate_credential_secret,
    validate_provider_setting_payload,
    validate_setting_payload,
)
from src.backend_v2.storage.defaults import FACTORY_PROMPTS
from src.backend_v2.storage.schema import (
    PROMPT_TYPES,
    app_settings,
    book_settings,
    credential_current_versions,
    credential_versions,
    credentials,
    fonts,
    idempotency_records,
    provider_rate_limits,
    provider_settings,
    prompts,
)
from src.backend_v2.storage.database import immediate_transaction
from src.shared.openai_rate_limits import RateLimitDecision


class RevisionConflict(RuntimeError):
    pass


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class SettingMutation:
    domain: str
    payload: dict[str, Any]
    base_revision: int
    schema_version: int


@dataclass(frozen=True, slots=True)
class ProviderSettingMutation:
    domain: str
    provider: str
    payload: dict[str, Any]
    base_revision: int
    schema_version: int
    credential_version_id: str | None = None
    credential_edit_ref: str | None = None


@dataclass(frozen=True, slots=True)
class CredentialEdit:
    domain: str
    provider: str
    secret: dict[str, Any]
    base_revision: int
    credential_id: str | None = None
    client_ref: str | None = None


@dataclass(frozen=True, slots=True)
class BookSettingMutation:
    book_id: str
    domain: str
    payload: dict[str, Any]
    base_revision: int
    schema_version: int


class SettingsRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def save_transaction(
        self,
        *,
        settings: tuple[SettingMutation, ...] = (),
        book_settings_edits: tuple[BookSettingMutation, ...] = (),
        providers: tuple[ProviderSettingMutation, ...] = (),
        credentials_edits: tuple[CredentialEdit, ...] = (),
    ) -> dict[str, list[dict[str, object]]]:
        with immediate_transaction(self.engine) as connection:
            return self._save_transaction(
                connection,
                settings=settings,
                book_settings_edits=book_settings_edits,
                providers=providers,
                credentials_edits=credentials_edits,
            )

    def save_transaction_idempotent(
        self,
        *,
        idempotency_key: str,
        request_body: dict[str, Any],
        settings: tuple[SettingMutation, ...] = (),
        book_settings_edits: tuple[BookSettingMutation, ...] = (),
        providers: tuple[ProviderSettingMutation, ...] = (),
        credentials_edits: tuple[CredentialEdit, ...] = (),
    ) -> tuple[dict[str, list[dict[str, object]]], bool]:
        now = _utcnow()
        request_hash = hashlib.sha256(
            _canonical_json(request_body).encode("utf-8")
        ).hexdigest()
        scope = "settings-transaction"
        with immediate_transaction(self.engine) as connection:
            replay = connection.execute(
                select(
                    idempotency_records.c.request_hash,
                    idempotency_records.c.response_json,
                ).where(
                    idempotency_records.c.scope == scope,
                    idempotency_records.c.key == idempotency_key,
                    idempotency_records.c.expires_at > now,
                )
            ).mappings().one_or_none()
            if replay is not None:
                if replay["request_hash"] != request_hash:
                    raise RevisionConflict(
                        "Idempotency-Key was reused for a different settings transaction"
                    )
                return json.loads(replay["response_json"]), True
            result = self._save_transaction(
                connection,
                settings=settings,
                book_settings_edits=book_settings_edits,
                providers=providers,
                credentials_edits=credentials_edits,
            )
            connection.execute(
                insert(idempotency_records).values(
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    http_status=200,
                    response_json=_canonical_json(result),
                    resource_type="settings",
                    expires_at=now + timedelta(hours=24),
                )
            )
            return result, False

    def _save_transaction(
        self,
        connection: object,
        *,
        settings: tuple[SettingMutation, ...],
        book_settings_edits: tuple[BookSettingMutation, ...],
        providers: tuple[ProviderSettingMutation, ...],
        credentials_edits: tuple[CredentialEdit, ...],
    ) -> dict[str, list[dict[str, object]]]:
        self._validate_transaction_keys(
            settings=settings,
            book_settings_edits=book_settings_edits,
            providers=providers,
            credentials_edits=credentials_edits,
        )
        setting_results = [
            self._save_setting(connection, mutation) for mutation in settings
        ]
        book_setting_results = [
            self._save_book_setting(connection, mutation)
            for mutation in book_settings_edits
        ]
        credential_results = [
            self._save_credential(connection, edit) for edit in credentials_edits
        ]
        created_versions = {
            edit.client_ref: str(result["credentialVersionId"])
            for edit, result in zip(credentials_edits, credential_results, strict=True)
            if edit.client_ref is not None
        }
        provider_results = []
        for mutation in providers:
            credential_version_id = mutation.credential_version_id
            if mutation.credential_edit_ref is not None:
                credential_version_id = created_versions[mutation.credential_edit_ref]
            provider_results.append(
                self._save_provider_setting(
                    connection,
                    mutation,
                    credential_version_id=credential_version_id,
                )
            )
        return {
            "settings": setting_results,
            "bookSettings": book_setting_results,
            "providerSettings": provider_results,
            "credentials": credential_results,
        }

    @staticmethod
    def _validate_transaction_keys(
        *,
        settings: tuple[SettingMutation, ...],
        book_settings_edits: tuple[BookSettingMutation, ...],
        providers: tuple[ProviderSettingMutation, ...],
        credentials_edits: tuple[CredentialEdit, ...],
    ) -> None:
        def require_unique(values: list[object], label: str) -> None:
            if len(values) != len(set(values)):
                raise ValueError(f"{label} contains duplicate identities")

        require_unique([row.domain for row in settings], "settings")
        require_unique(
            [(row.book_id, row.domain) for row in book_settings_edits],
            "book settings",
        )
        require_unique(
            [(row.domain, row.provider) for row in providers],
            "provider settings",
        )
        require_unique(
            [(row.domain, row.provider) for row in credentials_edits],
            "credential edits",
        )
        client_refs = [
            row.client_ref for row in credentials_edits if row.client_ref is not None
        ]
        require_unique(client_refs, "credential edit references")
        known_refs = set(client_refs)
        for provider in providers:
            if (
                provider.credential_version_id is not None
                and provider.credential_edit_ref is not None
            ):
                raise ValueError(
                    "provider setting cannot specify both credentialVersionId "
                    "and credentialEditRef"
                )
            if (
                provider.credential_edit_ref is not None
                and provider.credential_edit_ref not in known_refs
            ):
                raise ValueError("provider setting references an unknown credential edit")

    def load(
        self,
        *,
        domains: tuple[str, ...] = (),
        book_id: str | None = None,
    ) -> dict[str, object]:
        setting_condition = (
            app_settings.c.domain.in_(domains) if domains else True
        )
        provider_condition = (
            provider_settings.c.domain.in_(domains) if domains else True
        )
        with self.engine.connect() as connection:
            setting_rows = list(
                connection.execute(
                    select(app_settings)
                    .where(setting_condition)
                    .order_by(app_settings.c.domain)
                ).mappings()
            )
            provider_rows = list(
                connection.execute(
                    select(provider_settings)
                    .where(provider_condition)
                    .order_by(
                        provider_settings.c.domain,
                        provider_settings.c.provider,
                    )
                ).mappings()
            )
            book_rows = (
                list(
                    connection.execute(
                        select(book_settings)
                        .where(
                            book_settings.c.book_id == book_id,
                            (
                                book_settings.c.domain.in_(domains)
                                if domains
                                else True
                            ),
                        )
                        .order_by(book_settings.c.domain)
                    ).mappings()
                )
                if book_id
                else []
            )
        return {
            "settings": [
                {
                    "domain": row["domain"],
                    "revision": row["revision"],
                    "schemaVersion": row["schema_version"],
                    "payload": validate_setting_payload(
                        str(row["domain"]),
                        _require_object(
                            json.loads(row["payload_json"]),
                            f"{row['domain']} setting",
                        ),
                        schema_version=int(row["schema_version"]),
                    ),
                }
                for row in setting_rows
            ],
            "bookSettings": [
                {
                    "bookId": row["book_id"],
                    "domain": row["domain"],
                    "revision": row["revision"],
                    "schemaVersion": row["schema_version"],
                    "payload": validate_book_setting_payload(
                        str(row["domain"]),
                        _require_object(
                            json.loads(row["payload_json"]),
                            f"book {row['domain']} setting",
                        ),
                        schema_version=int(row["schema_version"]),
                    ),
                }
                for row in book_rows
            ],
            "providerSettings": [
                {
                    "domain": row["domain"],
                    "provider": row["provider"],
                    "revision": row["revision"],
                    "schemaVersion": row["schema_version"],
                    "credentialVersionId": row["credential_version_id"],
                    "payload": validate_provider_setting_payload(
                        str(row["domain"]),
                        str(row["provider"]),
                        _require_object(
                            json.loads(row["payload_json"]),
                            (
                                f"{row['domain']}/{row['provider']} "
                                "provider setting"
                            ),
                        ),
                        schema_version=int(row["schema_version"]),
                    ),
                }
                for row in provider_rows
            ],
            "credentials": self.credential_summaries(),
        }

    @staticmethod
    def _save_setting(connection: object, mutation: SettingMutation) -> dict[str, object]:
        if mutation.base_revision < 0 or mutation.schema_version < 1:
            raise ValueError("setting revisions must be non-negative")
        payload = validate_setting_payload(
            mutation.domain,
            mutation.payload,
            schema_version=mutation.schema_version,
        )
        if mutation.domain == "text_style_defaults":
            font_id, page_style = validate_text_style_defaults(  # type: ignore[arg-type]
                connection,
                payload,
            )
            payload = {**page_style, "fontFamily": font_id}
        payload_json = _canonical_json(payload)
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
        *,
        credential_version_id: str | None = None,
    ) -> dict[str, object]:
        if mutation.base_revision < 0 or mutation.schema_version < 1:
            raise ValueError("provider setting revisions must be non-negative")
        payload_json = _canonical_json(
            validate_provider_setting_payload(
                mutation.domain,
                mutation.provider,
                mutation.payload,
                schema_version=mutation.schema_version,
            )
        )
        key = and_(
            provider_settings.c.domain == mutation.domain,
            provider_settings.c.provider == mutation.provider,
        )
        current = connection.execute(  # type: ignore[attr-defined]
            select(provider_settings.c.revision).where(key)
        ).scalar_one_or_none()
        if credential_version_id is not None:
            owner = connection.execute(  # type: ignore[attr-defined]
                select(credentials.c.domain, credentials.c.provider)
                .join(
                    credential_versions,
                    credential_versions.c.credential_id == credentials.c.id,
                )
                .where(credential_versions.c.id == credential_version_id)
            ).mappings().one_or_none()
            if owner is None:
                raise ValueError("credential version does not exist")
            if (
                owner["domain"] != mutation.domain
                or owner["provider"] != mutation.provider
            ):
                raise ValueError(
                    "credential version domain/provider does not match provider setting"
                )
        values = {
            "payload_json": payload_json,
            "schema_version": mutation.schema_version,
            "credential_version_id": credential_version_id,
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
    def _save_book_setting(
        connection: object,
        mutation: BookSettingMutation,
    ) -> dict[str, object]:
        if mutation.base_revision < 0 or mutation.schema_version < 1:
            raise ValueError("book setting revisions must be non-negative")
        key = and_(
            book_settings.c.book_id == mutation.book_id,
            book_settings.c.domain == mutation.domain,
        )
        current = connection.execute(  # type: ignore[attr-defined]
            select(book_settings.c.revision).where(key)
        ).scalar_one_or_none()
        values = {
            "payload_json": _canonical_json(
                validate_book_setting_payload(
                    mutation.domain,
                    mutation.payload,
                    schema_version=mutation.schema_version,
                )
            ),
            "schema_version": mutation.schema_version,
        }
        if current is None:
            if mutation.base_revision != 0:
                raise RevisionConflict("book setting does not exist at requested revision")
            connection.execute(  # type: ignore[attr-defined]
                insert(book_settings).values(
                    book_id=mutation.book_id,
                    domain=mutation.domain,
                    revision=1,
                    **values,
                )
            )
            revision = 1
        else:
            if current != mutation.base_revision:
                raise RevisionConflict("book setting revision changed")
            changed = connection.execute(  # type: ignore[attr-defined]
                update(book_settings)
                .where(key, book_settings.c.revision == mutation.base_revision)
                .values(
                    revision=mutation.base_revision + 1,
                    updated_at=_utcnow(),
                    **values,
                )
            )
            if changed.rowcount != 1:
                raise RevisionConflict("book setting revision changed")
            revision = mutation.base_revision + 1
        return {
            "bookId": mutation.book_id,
            "domain": mutation.domain,
            "revision": revision,
        }

    @staticmethod
    def _save_credential(connection: object, edit: CredentialEdit) -> dict[str, object]:
        secret = validate_credential_secret(
            edit.domain,
            edit.provider,
            edit.secret,
        )
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
                "credentialVersionId": version_id,
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
            "credentialVersionId": version_id,
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
                    credential_current_versions.c.credential_version_id,
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
                    "credentialVersionId": row["credential_version_id"],
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
            ).scalar_one_or_none()
        if value is None:
            raise LookupError("credential version not found")
        return _require_object(json.loads(value), "stored credential secret")

    def resolve_credential_sections(
        self,
        config: Mapping[str, Any],
        section_names: Iterable[str],
    ) -> dict[str, Any]:
        """Materialize frozen credential versions into selected config sections."""

        result = deepcopy(dict(config))
        requested: dict[str, str] = {}
        for section_name in section_names:
            raw_section = result.get(section_name)
            section = dict(raw_section) if isinstance(raw_section, Mapping) else {}
            version_id = section.pop("credentialVersionId", None)
            result[section_name] = section
            if version_id is None:
                continue
            if not isinstance(version_id, str) or not version_id:
                raise ValueError(
                    f"{section_name}.credentialVersionId must be a non-empty string"
                )
            requested[section_name] = version_id
            section["credential_version_id"] = version_id
        if not requested:
            return result

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    credential_versions.c.id,
                    credential_versions.c.secret_json,
                ).where(
                    credential_versions.c.id.in_(set(requested.values()))
                )
            ).mappings()
            secrets = {
                str(row["id"]): _require_object(
                    json.loads(str(row["secret_json"])),
                    "stored credential secret",
                )
                for row in rows
            }
        for section_name, version_id in requested.items():
            secret = secrets.get(version_id)
            if secret is None:
                raise LookupError(
                    f"credential version for {section_name} not found"
                )
            result[section_name].update(secret)
        return result

    def resolve_current_secret(self, credential_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(credential_versions.c.secret_json)
                .join(
                    credential_current_versions,
                    credential_current_versions.c.credential_version_id
                    == credential_versions.c.id,
                )
                .where(
                    credential_current_versions.c.credential_id
                    == credential_id
                )
            ).scalar_one_or_none()
        if value is None:
            raise LookupError("credential not found")
        return _require_object(json.loads(value), "stored credential secret")

    def resolve_provider_secret(
        self,
        *,
        domain: str,
        provider: str,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(credential_versions.c.secret_json)
                .select_from(
                    provider_settings.join(
                        credential_versions,
                        credential_versions.c.id
                        == provider_settings.c.credential_version_id,
                    )
                )
                .where(
                    provider_settings.c.domain == domain,
                    provider_settings.c.provider == provider,
                )
            ).scalar_one_or_none()
        if value is None:
            raise LookupError(
                f"no stored credential for {domain}/{provider}"
            )
        return _require_object(json.loads(value), "stored credential secret")

    def delete_credential(self, credential_id: str) -> None:
        try:
            with immediate_transaction(self.engine) as connection:
                removed = connection.execute(
                    delete(credentials).where(credentials.c.id == credential_id)
                )
                if removed.rowcount != 1:
                    raise LookupError("credential not found")
        except IntegrityError as exc:
            raise RevisionConflict(
                "credential is still referenced by settings or history"
            ) from exc


class PromptRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def list(self, prompt_type: str | None = None) -> list[dict[str, object]]:
        if prompt_type is not None and prompt_type not in PROMPT_TYPES:
            raise ValueError("unsupported prompt type")
        condition = prompts.c.type == prompt_type if prompt_type else True
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(prompts)
                .where(condition)
                .order_by(prompts.c.type, prompts.c.name)
            ).mappings()
            return [self._dto(row) for row in rows]

    def create(
        self,
        *,
        prompt_type: str,
        name: str,
        content: str,
    ) -> dict[str, object]:
        self._validate(prompt_type, name, content)
        prompt_id = str(uuid.uuid4())
        try:
            with immediate_transaction(self.engine) as connection:
                connection.execute(
                    insert(prompts).values(
                        id=prompt_id,
                        type=prompt_type,
                        name=name.strip(),
                        content=content,
                    )
                )
        except IntegrityError as exc:
            raise RevisionConflict("prompt name already exists for this type") from exc
        return {
            "id": prompt_id,
            "type": prompt_type,
            "name": name.strip(),
            "content": content,
            "revision": 1,
            "isFactoryDefault": False,
        }

    def update(
        self,
        *,
        prompt_id: str,
        name: str,
        content: str,
        base_revision: int,
    ) -> dict[str, object]:
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(prompts.c.type).where(prompts.c.id == prompt_id)
            ).scalar_one_or_none()
            if row is None:
                raise LookupError("prompt not found")
            self._validate(str(row), name, content)
            try:
                changed = connection.execute(
                    update(prompts)
                    .where(
                        prompts.c.id == prompt_id,
                        prompts.c.revision == base_revision,
                    )
                    .values(
                        name=name.strip(),
                        content=content,
                        revision=base_revision + 1,
                        updated_at=_utcnow(),
                    )
                )
            except IntegrityError as exc:
                raise RevisionConflict(
                    "prompt name already exists for this type"
                ) from exc
            if changed.rowcount != 1:
                raise RevisionConflict("prompt revision changed")
        return {
            "id": prompt_id,
            "type": str(row),
            "name": name.strip(),
            "content": content,
            "revision": base_revision + 1,
        }

    def delete(self, prompt_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            factory = connection.execute(
                select(prompts.c.is_factory_default).where(
                    prompts.c.id == prompt_id
                )
            ).scalar_one_or_none()
            if factory is None:
                raise LookupError("prompt not found")
            if factory:
                raise RevisionConflict("factory prompt cannot be deleted")
            connection.execute(delete(prompts).where(prompts.c.id == prompt_id))

    def reset(self, prompt_id: str, *, base_revision: int) -> dict[str, object]:
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(prompts.c.type, prompts.c.name).where(
                    prompts.c.id == prompt_id,
                    prompts.c.is_factory_default.is_(True),
                )
            ).mappings().one_or_none()
            if row is None:
                raise RevisionConflict("only factory prompts can be reset")
            changed = connection.execute(
                update(prompts)
                .where(
                    prompts.c.id == prompt_id,
                    prompts.c.revision == base_revision,
                )
                .values(
                    content=FACTORY_PROMPTS[str(row["type"])],
                    revision=base_revision + 1,
                    updated_at=_utcnow(),
                )
            )
            if changed.rowcount != 1:
                raise RevisionConflict("prompt revision changed")
        return {
            "id": prompt_id,
            "type": row["type"],
            "name": row["name"],
            "content": FACTORY_PROMPTS[str(row["type"])],
            "revision": base_revision + 1,
            "isFactoryDefault": True,
        }

    @staticmethod
    def _validate(prompt_type: str, name: str, content: str) -> None:
        if prompt_type not in PROMPT_TYPES:
            raise ValueError("unsupported prompt type")
        if not name.strip() or len(name.strip()) > 200:
            raise ValueError("prompt name must contain 1-200 characters")
        if len(content) > 200_000:
            raise ValueError("prompt content is too large")

    @staticmethod
    def _dto(row: object) -> dict[str, object]:
        return {
            "id": row["id"],  # type: ignore[index]
            "type": row["type"],  # type: ignore[index]
            "name": row["name"],  # type: ignore[index]
            "content": row["content"],  # type: ignore[index]
            "revision": row["revision"],  # type: ignore[index]
            "isFactoryDefault": bool(row["is_factory_default"]),  # type: ignore[index]
        }


class FontRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

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

    def list(self) -> list[dict[str, object]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(fonts).order_by(
                    case((fonts.c.builtin_key == "default", 0), else_=1),
                    fonts.c.kind,
                    func.lower(fonts.c.display_name),
                )
            ).mappings()
            return [
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "displayName": row["display_name"],
                    "assetUrl": (
                        f"/api/v2/assets/{row['asset_id']}"
                        if row["asset_id"]
                        else None
                    ),
                    "builtinKey": row["builtin_key"],
                }
                for row in rows
            ]

    def delete_uploaded(self, font_id: str) -> str:
        try:
            with immediate_transaction(self.engine) as connection:
                row = connection.execute(
                    select(fonts.c.kind, fonts.c.asset_id).where(
                        fonts.c.id == font_id
                    )
                ).one_or_none()
                if row is None:
                    raise LookupError("font not found")
                if row.kind != "uploaded":
                    raise ValueError("built-in fonts cannot be deleted")
                connection.execute(delete(fonts).where(fonts.c.id == font_id))
        except IntegrityError as exc:
            raise RevisionConflict(
                "font is still referenced by content or history"
            ) from exc
        return str(row.asset_id)


class ProviderRateLimiter:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def acquire(
        self,
        *,
        provider: str,
        credential_version_id: str,
        rpm_limit: int,
    ) -> RateLimitDecision:
        if rpm_limit < 1:
            raise ValueError("rpm_limit must be positive")
        current_time = _utcnow()
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
                    effective_limit = rpm_limit
                else:
                    effective_limit = min(int(row["rpm_limit"]), rpm_limit)
                    if int(row["request_count"]) >= effective_limit:
                        retry_after = max(
                            0.0,
                            (
                                window_started_at
                                + timedelta(minutes=1)
                                - current_time
                            ).total_seconds(),
                        )
                        return RateLimitDecision(
                            allowed=False,
                            remaining=0,
                            retry_after_seconds=retry_after,
                        )
                    count = int(row["request_count"]) + 1
                    started_at = window_started_at

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
                        rpm_limit=effective_limit,
                        revision=revision + 1,
                    )
                )
                if changed.rowcount == 1:
                    return RateLimitDecision(
                        allowed=True,
                        remaining=max(0, effective_limit - count),
                        retry_after_seconds=0,
                    )
        raise RuntimeError("provider limiter CAS remained contended")
