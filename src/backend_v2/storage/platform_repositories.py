"""Repositories for settings, immutable credentials, fonts, and rate limits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import time
import uuid
from typing import Any, cast

from sqlalchemy import Engine, and_, case, delete, func, insert, or_, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from src.backend_v2.auth.credential_broker import (
    credential_reference,
    resolve_credential_reference,
)
from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.runtime_profile import resolve_runtime_profile
from src.backend_v2.serialization import canonical_json as _canonical_json
from src.backend_v2.timestamps import utcnow as _utcnow
from src.backend_v2.content.page_style import validate_text_style_defaults
from src.backend_v2.settings.validation import (
    is_proofreading_provider_domain,
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
    books,
    credential_current_versions,
    credential_versions,
    credentials,
    fonts,
    idempotency_records,
    provider_rate_limits,
    provider_settings,
    prompts,
)
from src.backend_v2.storage.database import (
    immediate_transaction,
    is_sqlite_busy_error,
    read_transaction,
)
from src.shared.openai_rate_limits import RateLimitDecision


class RevisionConflict(RuntimeError):
    pass


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _idempotency_replay(
    connection: Connection,
    *,
    scope: str,
    key: str,
    request_body: Mapping[str, Any],
    now: datetime,
) -> tuple[str, dict[str, Any] | None]:
    request_hash = hashlib.sha256(
        _canonical_json(dict(request_body)).encode("utf-8")
    ).hexdigest()
    row = connection.execute(
        select(
            idempotency_records.c.request_hash,
            idempotency_records.c.response_json,
            idempotency_records.c.expires_at,
        ).where(
            idempotency_records.c.scope == scope,
            idempotency_records.c.key == key,
            idempotency_records.c.owner_user_id == effective_owner_id(),
        )
    ).mappings().one_or_none()
    if row is None:
        return request_hash, None
    expires_at = row["expires_at"]
    if not isinstance(expires_at, datetime):
        raise RevisionConflict(
            "stored idempotency record is invalid; clear current data"
        )
    if expires_at <= now:
        connection.execute(
            delete(idempotency_records).where(
                idempotency_records.c.scope == scope,
                idempotency_records.c.key == key,
                idempotency_records.c.owner_user_id == effective_owner_id(),
            )
        )
        return request_hash, None
    if row["request_hash"] != request_hash:
        raise RevisionConflict(
            "Idempotency-Key was reused for a different request"
        )
    try:
        response = json.loads(row["response_json"])
    except (TypeError, ValueError) as exc:
        raise RevisionConflict(
            "stored idempotency response is invalid; clear current data"
        ) from exc
    if not isinstance(response, dict):
        raise RevisionConflict(
            "stored idempotency response is invalid; clear current data"
        )
    return request_hash, response


def _record_idempotency(
    connection: Connection,
    *,
    scope: str,
    key: str,
    request_hash: str,
    response: Mapping[str, Any],
    http_status: int,
    resource_type: str,
    resource_id: str | None = None,
    now: datetime,
) -> None:
    connection.execute(
        insert(idempotency_records).values(
            owner_user_id=effective_owner_id(),
            scope=scope,
            key=key,
            request_hash=request_hash,
            http_status=http_status,
            response_json=_canonical_json(dict(response)),
            resource_type=resource_type,
            resource_id=resource_id,
            expires_at=now + timedelta(hours=24),
        )
    )


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


@dataclass(frozen=True, slots=True)
class PromptMutation:
    prompt_id: str
    name: str
    content: str
    base_revision: int


def _validate_prompt(prompt_type: str, name: str) -> None:
    if prompt_type not in PROMPT_TYPES:
        raise ValueError("unsupported prompt type")
    if not name.strip() or len(name.strip()) > 200:
        raise ValueError("prompt name must contain 1-200 characters")


def _update_prompt(
    connection: object,
    mutation: PromptMutation,
) -> dict[str, object]:
    row = connection.execute(
        select(prompts.c.type, prompts.c.is_factory_default).where(
            prompts.c.id == mutation.prompt_id,
            prompts.c.owner_user_id == effective_owner_id(),
        )
    ).mappings().one_or_none()
    if row is None:
        raise LookupError("prompt not found")
    prompt_type = str(row["type"])
    _validate_prompt(prompt_type, mutation.name)
    try:
        changed = connection.execute(
            update(prompts)
            .where(
                prompts.c.id == mutation.prompt_id,
                prompts.c.revision == mutation.base_revision,
                prompts.c.owner_user_id == effective_owner_id(),
            )
            .values(
                name=mutation.name.strip(),
                content=mutation.content,
                revision=mutation.base_revision + 1,
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
        "id": mutation.prompt_id,
        "type": prompt_type,
        "name": mutation.name.strip(),
        "content": mutation.content,
        "revision": mutation.base_revision + 1,
        "isFactoryDefault": bool(row["is_factory_default"]),
    }


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
        prompt_edits: tuple[PromptMutation, ...] = (),
    ) -> dict[str, list[dict[str, object]]]:
        with immediate_transaction(self.engine) as connection:
            return self._save_transaction(
                connection,
                settings=settings,
                book_settings_edits=book_settings_edits,
                providers=providers,
                credentials_edits=credentials_edits,
                prompt_edits=prompt_edits,
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
        prompt_edits: tuple[PromptMutation, ...] = (),
    ) -> tuple[dict[str, list[dict[str, object]]], bool]:
        now = _utcnow()
        scope = "settings-transaction"
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_body=request_body,
                now=now,
            )
            if replay is not None:
                return cast(dict[str, list[dict[str, object]]], replay), True
            result = self._save_transaction(
                connection,
                settings=settings,
                book_settings_edits=book_settings_edits,
                providers=providers,
                credentials_edits=credentials_edits,
                prompt_edits=prompt_edits,
            )
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=result,
                http_status=200,
                resource_type="settings",
                now=now,
            )
            return result, False

    def replay_idempotent_command(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_body: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        with immediate_transaction(self.engine) as connection:
            _request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_body=request_body,
                now=_utcnow(),
            )
            return replay

    def record_idempotent_command(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_body: Mapping[str, Any],
        response: Mapping[str, Any],
        resource_type: str,
    ) -> tuple[dict[str, Any], bool]:
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_body=request_body,
                now=now,
            )
            if replay is not None:
                return replay, True
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type=resource_type,
                now=now,
            )
            return dict(response), False

    def _save_transaction(
        self,
        connection: object,
        *,
        settings: tuple[SettingMutation, ...],
        book_settings_edits: tuple[BookSettingMutation, ...],
        providers: tuple[ProviderSettingMutation, ...],
        credentials_edits: tuple[CredentialEdit, ...],
        prompt_edits: tuple[PromptMutation, ...],
    ) -> dict[str, list[dict[str, object]]]:
        self._validate_transaction_keys(
            settings=settings,
            book_settings_edits=book_settings_edits,
            providers=providers,
            credentials_edits=credentials_edits,
            prompt_edits=prompt_edits,
        )
        setting_results = [
            self._save_setting(connection, mutation) for mutation in settings
        ]
        translation_mutation = next(
            (mutation for mutation in settings if mutation.domain == "translation"),
            None,
        )
        if translation_mutation is not None:
            self._prune_proofreading_provider_settings(
                connection,
                translation_payload=translation_mutation.payload,
                providers=providers,
            )
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
        prompt_results = [
            _update_prompt(connection, mutation) for mutation in prompt_edits
        ]
        return {
            "settings": setting_results,
            "bookSettings": book_setting_results,
            "providerSettings": provider_results,
            "credentials": credential_results,
            "prompts": prompt_results,
        }

    @staticmethod
    def _prune_proofreading_provider_settings(
        connection: Connection,
        *,
        translation_payload: Mapping[str, Any],
        providers: tuple[ProviderSettingMutation, ...],
    ) -> None:
        proofreading = cast(Mapping[str, Any], translation_payload["proofreading"])
        rounds = cast(list[Mapping[str, Any]], proofreading["rounds"])
        active_pairs = {
            (f"proofreading_{round_config['id']}", str(round_config["provider"]))
            for round_config in rounds
        }
        submitted_pairs = {
            (mutation.domain, mutation.provider)
            for mutation in providers
            if is_proofreading_provider_domain(mutation.domain)
        }
        orphaned_submissions = submitted_pairs - active_pairs
        if orphaned_submissions:
            raise ValueError(
                "proofreading provider settings must belong to active rounds"
            )

        stored_provider_pairs = connection.execute(
            select(provider_settings.c.domain, provider_settings.c.provider).where(
                provider_settings.c.domain.like("proofreading_%"),
                provider_settings.c.owner_user_id == effective_owner_id(),
            )
        ).all()
        for domain, provider in stored_provider_pairs:
            pair = (str(domain), str(provider))
            if pair in active_pairs:
                continue
            connection.execute(
                delete(provider_settings).where(
                    provider_settings.c.domain == pair[0],
                    provider_settings.c.provider == pair[1],
                    provider_settings.c.owner_user_id == effective_owner_id(),
                )
            )

    @staticmethod
    def _validate_transaction_keys(
        *,
        settings: tuple[SettingMutation, ...],
        book_settings_edits: tuple[BookSettingMutation, ...],
        providers: tuple[ProviderSettingMutation, ...],
        credentials_edits: tuple[CredentialEdit, ...],
        prompt_edits: tuple[PromptMutation, ...],
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
        require_unique(
            [row.prompt_id for row in prompt_edits],
            "prompt edits",
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
        with read_transaction(self.engine) as connection:
            if book_id and connection.execute(
                select(books.c.id).where(
                    books.c.id == book_id,
                    books.c.owner_user_id == effective_owner_id(),
                )
            ).scalar_one_or_none() is None:
                raise LookupError("book not found")
            setting_rows = list(
                connection.execute(
                    select(app_settings)
                    .where(
                        setting_condition,
                        app_settings.c.owner_user_id == effective_owner_id(),
                    )
                    .order_by(app_settings.c.domain)
                ).mappings()
            )
            provider_rows = list(
                connection.execute(
                    select(provider_settings)
                    .where(
                        provider_condition,
                        provider_settings.c.owner_user_id == effective_owner_id(),
                    )
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
            credential_rows = self._credential_summaries_from_connection(
                connection,
                domains=domains,
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
            "credentials": credential_rows,
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
            select(app_settings.c.revision).where(
                app_settings.c.domain == mutation.domain,
                app_settings.c.owner_user_id == effective_owner_id(),
            )
        ).scalar_one_or_none()
        if current is None:
            if mutation.base_revision != 0:
                raise RevisionConflict(f"setting {mutation.domain} does not exist at requested revision")
            connection.execute(  # type: ignore[attr-defined]
                insert(app_settings).values(
                    owner_user_id=effective_owner_id(),
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
                app_settings.c.owner_user_id == effective_owner_id(),
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
            provider_settings.c.owner_user_id == effective_owner_id(),
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
                .where(credentials.c.owner_user_id == effective_owner_id())
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
                    owner_user_id=effective_owner_id(),
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
        if connection.execute(  # type: ignore[attr-defined]
            select(books.c.id).where(
                books.c.id == mutation.book_id,
                books.c.owner_user_id == effective_owner_id(),
            )
        ).scalar_one_or_none() is None:
            raise LookupError("book not found")
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
            existing_id = connection.execute(  # type: ignore[attr-defined]
                select(credentials.c.id).where(
                    credentials.c.domain == edit.domain,
                    credentials.c.provider == edit.provider,
                    credentials.c.owner_user_id == effective_owner_id(),
                )
            ).scalar_one_or_none()
            if existing_id is not None:
                raise RevisionConflict(
                    "credential already exists; its current ID and revision are required"
                )
            credential_id = str(uuid.uuid4())
            version_id = str(uuid.uuid4())
            connection.execute(  # type: ignore[attr-defined]
                insert(credentials).values(
                    id=credential_id,
                    owner_user_id=effective_owner_id(),
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
            .where(
                credentials.c.id == edit.credential_id,
                credentials.c.owner_user_id == effective_owner_id(),
            )
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
            return self._credential_summaries_from_connection(connection)

    @staticmethod
    def _credential_summaries_from_connection(
        connection: Connection,
        *,
        domains: tuple[str, ...] = (),
    ) -> list[dict[str, object]]:
        statement = (
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
            .where(credentials.c.owner_user_id == effective_owner_id())
            .order_by(credentials.c.domain, credentials.c.provider)
        )
        if domains:
            statement = statement.where(credentials.c.domain.in_(domains))
        rows = connection.execute(statement).mappings()
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
        browser_secret = resolve_credential_reference(
            credential_version_id, effective_owner_id()
        )
        if browser_secret is not None:
            return browser_secret
        with self.engine.connect() as connection:
            value = connection.execute(
                select(credential_versions.c.secret_json)
                .join(
                    credentials,
                    credentials.c.id == credential_versions.c.credential_id,
                )
                .where(
                    credential_versions.c.id == credential_version_id,
                    credentials.c.owner_user_id == effective_owner_id(),
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

        secrets: dict[str, dict[str, Any]] = {}
        database_ids: set[str] = set()
        for version_id in set(requested.values()):
            browser_secret = resolve_credential_reference(
                version_id, effective_owner_id()
            )
            if browser_secret is None:
                database_ids.add(version_id)
            else:
                secrets[version_id] = browser_secret
        if database_ids:
            with self.engine.connect() as connection:
                rows = connection.execute(
                    select(
                        credential_versions.c.id,
                        credential_versions.c.secret_json,
                    )
                    .join(
                        credentials,
                        credentials.c.id == credential_versions.c.credential_id,
                    )
                    .where(
                        credential_versions.c.id.in_(database_ids),
                        credentials.c.owner_user_id == effective_owner_id(),
                    )
                ).mappings()
                secrets.update(
                    {
                        str(row["id"]): _require_object(
                            json.loads(str(row["secret_json"])),
                            "stored credential secret",
                        )
                        for row in rows
                    }
                )
        for section_name, version_id in requested.items():
            secret = secrets.get(version_id)
            if secret is None:
                raise LookupError(
                    f"credential version for {section_name} not found"
                )
            result[section_name].update(secret)
        return result

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
                    provider_settings.c.owner_user_id == effective_owner_id(),
                )
            ).scalar_one_or_none()
        if value is None:
            if resolve_runtime_profile().browser_credentials:
                browser_secret = resolve_credential_reference(
                    credential_reference(domain, provider),
                    effective_owner_id(),
                )
                if browser_secret is not None:
                    return browser_secret
            raise LookupError(
                f"no stored credential for {domain}/{provider}"
            )
        return _require_object(json.loads(value), "stored credential secret")

    def delete_credential(self, credential_id: str) -> None:
        try:
            with immediate_transaction(self.engine) as connection:
                self._delete_credential(connection, credential_id)
        except IntegrityError as exc:
            raise RevisionConflict(
                "credential is still referenced by settings or history"
            ) from exc

    def delete_credential_idempotent(
        self,
        *,
        idempotency_key: str,
        credential_id: str,
    ) -> tuple[dict[str, object], bool]:
        scope = f"DELETE:deleteCredential:{credential_id}"
        now = _utcnow()
        try:
            with immediate_transaction(self.engine) as connection:
                request_hash, replay = _idempotency_replay(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request_body={},
                    now=now,
                )
                if replay is not None:
                    return replay, True
                self._delete_credential(connection, credential_id)
                result = {"deleted": True}
                _record_idempotency(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    response=result,
                    http_status=200,
                    resource_type="credential",
                    resource_id=credential_id,
                    now=now,
                )
                return result, False
        except IntegrityError as exc:
            raise RevisionConflict(
                "credential is still referenced by settings or history"
            ) from exc

    @staticmethod
    def _delete_credential(
        connection: Connection,
        credential_id: str,
    ) -> None:
        removed = connection.execute(
            delete(credentials).where(
                credentials.c.id == credential_id,
                credentials.c.owner_user_id == effective_owner_id(),
            )
        )
        if removed.rowcount != 1:
            raise LookupError("credential not found")


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
                .where(
                    condition,
                    prompts.c.owner_user_id == effective_owner_id(),
                )
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
        with immediate_transaction(self.engine) as connection:
            return self._create(
                connection,
                prompt_type=prompt_type,
                name=name,
                content=content,
            )

    def create_idempotent(
        self,
        *,
        idempotency_key: str,
        prompt_type: str,
        name: str,
        content: str,
    ) -> tuple[dict[str, object], bool]:
        body = {"type": prompt_type, "name": name, "content": content}
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope="POST:createPrompt",
                key=idempotency_key,
                request_body=body,
                now=now,
            )
            if replay is not None:
                return replay, True
            result = self._create(
                connection,
                prompt_type=prompt_type,
                name=name,
                content=content,
            )
            _record_idempotency(
                connection,
                scope="POST:createPrompt",
                key=idempotency_key,
                request_hash=request_hash,
                response=result,
                http_status=201,
                resource_type="prompt",
                resource_id=str(result["id"]),
                now=now,
            )
            return result, False

    @staticmethod
    def _create(
        connection: Connection,
        *,
        prompt_type: str,
        name: str,
        content: str,
    ) -> dict[str, object]:
        _validate_prompt(prompt_type, name)
        prompt_id = str(uuid.uuid4())
        try:
            connection.execute(
                insert(prompts).values(
                    id=prompt_id,
                    owner_user_id=effective_owner_id(),
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
            return _update_prompt(
                connection,
                PromptMutation(
                    prompt_id=prompt_id,
                    name=name,
                    content=content,
                    base_revision=base_revision,
                ),
            )

    def update_idempotent(
        self,
        *,
        idempotency_key: str,
        prompt_id: str,
        name: str,
        content: str,
        base_revision: int,
    ) -> tuple[dict[str, object], bool]:
        body = {
            "name": name,
            "content": content,
            "baseRevision": base_revision,
        }
        scope = f"PUT:updatePrompt:{prompt_id}"
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_body=body,
                now=now,
            )
            if replay is not None:
                return replay, True
            result = _update_prompt(
                connection,
                PromptMutation(
                    prompt_id=prompt_id,
                    name=name,
                    content=content,
                    base_revision=base_revision,
                ),
            )
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=result,
                http_status=200,
                resource_type="prompt",
                resource_id=prompt_id,
                now=now,
            )
            return result, False

    def delete(self, prompt_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            self._delete(connection, prompt_id)

    def delete_idempotent(
        self,
        *,
        idempotency_key: str,
        prompt_id: str,
    ) -> tuple[dict[str, object], bool]:
        scope = f"DELETE:deletePrompt:{prompt_id}"
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_body={},
                now=now,
            )
            if replay is not None:
                return replay, True
            self._delete(connection, prompt_id)
            result = {"deleted": True}
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=result,
                http_status=200,
                resource_type="prompt",
                resource_id=prompt_id,
                now=now,
            )
            return result, False

    @staticmethod
    def _delete(connection: Connection, prompt_id: str) -> None:
        factory = connection.execute(
            select(prompts.c.is_factory_default).where(
                prompts.c.id == prompt_id,
                prompts.c.owner_user_id == effective_owner_id(),
            )
        ).scalar_one_or_none()
        if factory is None:
            raise LookupError("prompt not found")
        if factory:
            raise RevisionConflict("factory prompt cannot be deleted")
        connection.execute(
            delete(prompts).where(
                prompts.c.id == prompt_id,
                prompts.c.owner_user_id == effective_owner_id(),
            )
        )

    def reset(self, prompt_id: str, *, base_revision: int) -> dict[str, object]:
        with immediate_transaction(self.engine) as connection:
            return self._reset(connection, prompt_id, base_revision=base_revision)

    def reset_idempotent(
        self,
        *,
        idempotency_key: str,
        prompt_id: str,
        base_revision: int,
    ) -> tuple[dict[str, object], bool]:
        scope = f"POST:resetPrompt:{prompt_id}"
        body = {"baseRevision": base_revision}
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_body=body,
                now=now,
            )
            if replay is not None:
                return replay, True
            result = self._reset(
                connection,
                prompt_id,
                base_revision=base_revision,
            )
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=result,
                http_status=200,
                resource_type="prompt",
                resource_id=prompt_id,
                now=now,
            )
            return result, False

    @staticmethod
    def _reset(
        connection: Connection,
        prompt_id: str,
        *,
        base_revision: int,
    ) -> dict[str, object]:
        row = connection.execute(
            select(
                prompts.c.type,
                prompts.c.name,
                prompts.c.is_factory_default,
            ).where(
                prompts.c.id == prompt_id,
                prompts.c.owner_user_id == effective_owner_id(),
            )
        ).mappings().one_or_none()
        if row is None:
            raise LookupError("prompt not found")
        if not row["is_factory_default"]:
            raise RevisionConflict("only factory prompts can be reset")
        changed = connection.execute(
            update(prompts)
            .where(
                prompts.c.id == prompt_id,
                prompts.c.revision == base_revision,
                prompts.c.owner_user_id == effective_owner_id(),
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
            self._register_uploaded(
                connection,
                font_id=font_id,
                asset_id=asset_id,
                display_name=display_name,
            )
        return font_id

    def replay_upload(
        self,
        *,
        idempotency_key: str,
        request_body: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        with immediate_transaction(self.engine) as connection:
            _request_hash, replay = _idempotency_replay(
                connection,
                scope="POST:uploadFont",
                key=idempotency_key,
                request_body=request_body,
                now=_utcnow(),
            )
            return replay

    def register_uploaded_idempotent(
        self,
        *,
        idempotency_key: str,
        request_body: Mapping[str, Any],
        asset_id: str,
        display_name: str,
    ) -> tuple[dict[str, object], bool]:
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            request_hash, replay = _idempotency_replay(
                connection,
                scope="POST:uploadFont",
                key=idempotency_key,
                request_body=request_body,
                now=now,
            )
            if replay is not None:
                return replay, True
            font_id = str(uuid.uuid4())
            self._register_uploaded(
                connection,
                font_id=font_id,
                asset_id=asset_id,
                display_name=display_name,
            )
            result = {
                "id": font_id,
                "kind": "uploaded",
                "displayName": display_name,
                "builtinKey": None,
                "assetUrl": f"/api/v2/assets/{asset_id}",
            }
            _record_idempotency(
                connection,
                scope="POST:uploadFont",
                key=idempotency_key,
                request_hash=request_hash,
                response=result,
                http_status=201,
                resource_type="font",
                resource_id=font_id,
                now=now,
            )
            return result, False

    @staticmethod
    def _register_uploaded(
        connection: Connection,
        *,
        font_id: str,
        asset_id: str,
        display_name: str,
    ) -> None:
        connection.execute(
            insert(fonts).values(
                id=font_id,
                owner_user_id=effective_owner_id(),
                kind="uploaded",
                asset_id=asset_id,
                display_name=display_name,
            )
        )

    def list(self) -> list[dict[str, object]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(fonts)
                .where(
                    or_(
                        fonts.c.kind == "builtin",
                        fonts.c.owner_user_id == effective_owner_id(),
                    )
                )
                .order_by(
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
                asset_id = self._delete_uploaded(connection, font_id)
        except IntegrityError as exc:
            raise RevisionConflict(
                "font is still referenced by content or history"
            ) from exc
        return asset_id

    def delete_uploaded_idempotent(
        self,
        *,
        idempotency_key: str,
        font_id: str,
    ) -> tuple[dict[str, object], bool]:
        scope = f"DELETE:deleteFont:{font_id}"
        now = _utcnow()
        try:
            with immediate_transaction(self.engine) as connection:
                request_hash, replay = _idempotency_replay(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request_body={},
                    now=now,
                )
                if replay is not None:
                    return replay, True
                self._delete_uploaded(connection, font_id)
                result = {"deleted": True}
                _record_idempotency(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    response=result,
                    http_status=200,
                    resource_type="font",
                    resource_id=font_id,
                    now=now,
                )
                return result, False
        except IntegrityError as exc:
            raise RevisionConflict(
                "font is still referenced by content or history"
            ) from exc

    @staticmethod
    def _delete_uploaded(connection: Connection, font_id: str) -> str:
        row = connection.execute(
            select(fonts.c.kind, fonts.c.asset_id).where(
                fonts.c.id == font_id,
                fonts.c.owner_user_id == effective_owner_id(),
            )
        ).one_or_none()
        if row is None:
            raise LookupError("font not found")
        if row.kind != "uploaded":
            raise ValueError("built-in fonts cannot be deleted")
        connection.execute(
            delete(fonts).where(
                fonts.c.id == font_id,
                fonts.c.owner_user_id == effective_owner_id(),
            )
        )
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

        busy_failures = 0
        for _attempt in range(8):
            try:
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
            except Exception as exc:
                if not is_sqlite_busy_error(exc) or busy_failures >= 2:
                    raise
                busy_failures += 1
                time.sleep(0.05)
        raise RuntimeError("provider limiter CAS remained contended")
