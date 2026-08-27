"""Versioned continuation projects and durable four-step workflow jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import timedelta
from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import Any, Protocol
import uuid
import zipfile

from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.insight.derived import (
    AnalysisInputSnapshot,
    InsightDerivedRepository,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
    _idempotency_replay,
    _record_idempotency,
)
from src.backend_v2.insight.provider_runtime import frozen_image_gen_config
from src.backend_v2.timestamps import utcnow
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    analysis_artifacts,
    analysis_heads,
    assets,
    continuation_character_forms,
    continuation_characters,
    continuation_form_image_versions,
    continuation_image_versions,
    continuation_pages,
    continuation_project_reference_assets,
    continuation_projects,
    continuation_scripts,
    chapters,
    jobs,
    job_artifacts,
    page_assets,
    pages,
    timeline_characters,
    timeline_versions,
)
from src.shared.user_logging import json_details, log_result


def _load(value: object) -> object:
    if not isinstance(value, str) or not value:
        raise ValueError("continuation JSON payload is missing")
    return json.loads(value)


class ContinuationRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.derived = InsightDerivedRepository(engine)

    def bootstrap(self, *, book_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            project = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.book_id == book_id
                )
            ).mappings().one_or_none()
            if project is None:
                checked_snapshot = self._snapshot_or_none(book_id=book_id)
                active_run_id = (
                    checked_snapshot.source_run_id
                    if checked_snapshot is not None
                    else None
                )
                active_only = True
                allow_stale = False
            else:
                checked_snapshot = self._project_snapshot_or_none(
                    book_id=book_id,
                    project=project,
                )
                active_run_id = connection.execute(
                    select(analysis_heads.c.active_run_id).where(
                        analysis_heads.c.book_id == book_id,
                        analysis_heads.c.page_id.is_(None),
                    )
                ).scalar_one_or_none()
                active_only = False
                allow_stale = True
            prerequisites, timeline = self._snapshot_prerequisites(
                connection,
                book_id=book_id,
                fingerprint=(
                    checked_snapshot.fingerprint if checked_snapshot else None
                ),
                active_only=active_only,
                allow_stale=allow_stale,
            )
            missing = _missing_prerequisites(
                analysis_ready=checked_snapshot is not None,
                prerequisites=prerequisites,
                timeline=timeline,
            )
            return {
                "bookId": book_id,
                "ready": not missing,
                "activeRunId": active_run_id,
                "missing": missing,
                "project": (
                    self._project_dto(connection, project)
                    if project is not None
                    else None
                ),
            }

    def sync_latest(
        self,
        *,
        idempotency_key: str,
        book_id: str,
    ) -> dict[str, Any]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"POST:syncContinuationAnalysis:{book_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={},
                now=now,
            )
            if replay is not None:
                return replay
            try:
                snapshot = InsightDerivedRepository._snapshot(
                    connection,
                    book_id=book_id,
                )
            except (InsightConflict, InsightNotFound):
                snapshot = None
            prerequisites, timeline = self._snapshot_prerequisites(
                connection,
                book_id=book_id,
                fingerprint=snapshot.fingerprint if snapshot else None,
                active_only=True,
                allow_stale=False,
            )
            missing = _missing_prerequisites(
                analysis_ready=snapshot is not None,
                prerequisites=prerequisites,
                timeline=timeline,
            )
            if missing:
                raise InsightConflict(
                    "continuation prerequisites are missing: "
                    + ", ".join(missing)
                )
            if snapshot is None or timeline is None:
                raise InsightConflict(
                    "continuation prerequisites changed during synchronization"
                )
            run_id = snapshot.source_run_id
            analysis_inputs = _snapshot_inputs(snapshot)
            project = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.book_id == book_id
                )
            ).mappings().one_or_none()
            if project is None:
                project_id = str(uuid.uuid4())
                connection.execute(
                    insert(continuation_projects).values(
                        id=project_id,
                        owner_user_id=effective_owner_id(),
                        book_id=book_id,
                        source_run_id=run_id,
                        revision=1,
                        payload_json=_json(
                            {
                                "pageCount": 15,
                                "styleReferencePages": 3,
                                "direction": "",
                                "analysisInputs": analysis_inputs,
                                "analysisInputFingerprint": snapshot.fingerprint,
                            }
                        ),
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                project_id = str(project["id"])
                try:
                    project_payload = _required_mapping_json(
                        project["payload_json"],
                        "continuation project",
                    )
                    _public_project_config(project_payload)
                    stored_inputs = _validated_analysis_inputs(
                        project_payload.get("analysisInputs")
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise InsightConflict(
                        "continuation project schema is invalid; clear the project"
                    ) from exc
                stored_fingerprint = project_payload.get(
                    "analysisInputFingerprint"
                )
                if (
                    not _is_sha256(stored_fingerprint)
                ):
                    raise InsightConflict(
                        "continuation project schema is invalid; clear the project"
                    )
                snapshot_changed = (
                    project["source_run_id"] != run_id
                    or stored_inputs != analysis_inputs
                    or stored_fingerprint != snapshot.fingerprint
                )
                if snapshot_changed:
                    project_payload.update(
                        {
                            "analysisInputs": analysis_inputs,
                            "analysisInputFingerprint": snapshot.fingerprint,
                        }
                    )
                    connection.execute(
                        update(continuation_projects)
                        .where(continuation_projects.c.id == project_id)
                        .values(
                            source_run_id=run_id,
                            payload_json=_json(project_payload),
                            revision=int(project["revision"]) + 1,
                            updated_at=now,
                        )
                    )
            timeline_id = str(timeline["id"])
            existing_names = set(
                connection.execute(
                    select(continuation_characters.c.name).where(
                        continuation_characters.c.project_id == project_id
                    )
                ).scalars()
            )
            characters = list(
                connection.execute(
                    select(
                        timeline_characters.c.name,
                        timeline_characters.c.payload_json,
                    ).where(
                        timeline_characters.c.timeline_version_id
                        == timeline_id
                    )
                )
            )
            for name, payload in characters:
                if str(name) in existing_names:
                    continue
                loaded = _mapping(_load(payload))
                aliases = loaded.get("aliases", [])
                if not isinstance(aliases, list) or any(
                    not isinstance(value, str) for value in aliases
                ):
                    raise InsightConflict(
                        "timeline character aliases are invalid"
                    )
                connection.execute(
                    insert(continuation_characters).values(
                        id=str(uuid.uuid4()),
                        project_id=project_id,
                        name=str(name),
                        aliases_json=_json(aliases),
                        enabled=True,
                        payload_json=_json(loaded),
                    )
                )
            project = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.id == project_id
                )
            ).mappings().one()
            response = self._project_dto(connection, project)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_project",
                resource_id=project_id,
                now=now,
            )
            return response

    def update_project(
        self,
        *,
        idempotency_key: str,
        project_id: str,
        base_revision: int,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        base_revision = _positive_integer(base_revision, "baseRevision")
        normalized = _validated_project_config(config)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"PATCH:updateContinuationProject:{project_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    "config": normalized,
                },
                now=now,
            )
            if replay is not None:
                return replay
            current = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.id == project_id
                )
            ).mappings().one_or_none()
            if current is None:
                raise InsightNotFound("continuation project not found")
            try:
                current_payload = _required_mapping_json(
                    current["payload_json"],
                    "continuation project",
                )
                analysis_inputs = _validated_analysis_inputs(
                    current_payload.get("analysisInputs")
                )
                analysis_fingerprint = current_payload.get(
                    "analysisInputFingerprint"
                )
                if not _is_sha256(analysis_fingerprint):
                    raise ValueError(
                        "analysisInputFingerprint must be SHA-256"
                    )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise InsightConflict(
                    "continuation project schema is invalid; clear the project"
                ) from exc
            normalized.update(
                {
                    "analysisInputs": analysis_inputs,
                    "analysisInputFingerprint": analysis_fingerprint,
                }
            )
            changed = connection.execute(
                update(continuation_projects)
                .where(
                    continuation_projects.c.id == project_id,
                    continuation_projects.c.revision == base_revision,
                )
                .values(
                    payload_json=_json(normalized),
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise InsightConflict("continuation project revision changed")
            connection.execute(
                delete(continuation_pages).where(
                    continuation_pages.c.project_id == project_id,
                    continuation_pages.c.ordinal > normalized["pageCount"],
                )
            )
            row = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.id == project_id
                )
            ).mappings().one()
            response = self._project_dto(connection, row)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_project",
                resource_id=project_id,
                now=now,
            )
            return response

    def set_project_references(
        self,
        *,
        idempotency_key: str,
        project_id: str,
        base_revision: int,
        asset_ids: Sequence[str],
    ) -> dict[str, Any]:
        base_revision = _positive_integer(base_revision, "baseRevision")
        normalized = list(asset_ids)
        if any(not isinstance(value, str) or not value for value in normalized):
            raise ValueError("reference assetIds must be non-empty strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError("reference assetIds must be unique")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"PUT:setContinuationReferences:{project_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    "assetIds": normalized,
                },
                now=now,
            )
            if replay is not None:
                return replay
            self._require_project(connection, project_id)
            if normalized:
                asset_rows = {
                    str(row["id"]): str(row["mime_type"])
                    for row in connection.execute(
                        select(assets.c.id, assets.c.mime_type).where(
                            assets.c.id.in_(tuple(normalized)),
                            assets.c.owner_user_id == effective_owner_id(),
                        )
                    ).mappings()
                }
                if set(asset_rows) != set(normalized):
                    raise InsightNotFound("reference asset not found")
                if any(
                    not mime_type.startswith("image/")
                    for mime_type in asset_rows.values()
                ):
                    raise ValueError("continuation references must be images")
            changed = connection.execute(
                update(continuation_projects)
                .where(
                    continuation_projects.c.id == project_id,
                    continuation_projects.c.revision == base_revision,
                )
                .values(
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise InsightConflict(
                    "continuation project revision changed"
                )
            connection.execute(
                delete(continuation_project_reference_assets).where(
                    continuation_project_reference_assets.c.project_id
                    == project_id
                )
            )
            if normalized:
                connection.execute(
                    insert(continuation_project_reference_assets),
                    [
                        {
                            "project_id": project_id,
                            "ordinal": ordinal,
                            "asset_id": asset_id,
                        }
                        for ordinal, asset_id in enumerate(
                            normalized,
                            start=1,
                        )
                    ],
                )
            row = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.id == project_id
                )
            ).mappings().one()
            response = self._project_dto(connection, row)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_project",
                resource_id=project_id,
                now=now,
            )
            return response

    def update_script(
        self,
        *,
        idempotency_key: str,
        project_id: str,
        base_revision: int,
        content: str,
    ) -> dict[str, Any]:
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 0
        ):
            raise ValueError("baseRevision must be a non-negative integer")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("continuation script content is required")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"PATCH:updateContinuationScript:{project_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    "content": content,
                },
                now=now,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                select(continuation_scripts).where(
                    continuation_scripts.c.project_id == project_id
                )
            ).mappings().one_or_none()
            if row is None:
                self._require_project(connection, project_id)
                if base_revision != 0:
                    raise InsightConflict("continuation script does not exist")
                script_id = str(uuid.uuid4())
                revision = 1
                connection.execute(
                    insert(continuation_scripts).values(
                        id=script_id,
                        project_id=project_id,
                        revision=revision,
                        content=content,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                if int(row["revision"]) != base_revision:
                    raise InsightConflict(
                        "continuation script revision changed"
                    )
                script_id = str(row["id"])
                revision = base_revision + 1
                connection.execute(
                    update(continuation_scripts)
                    .where(
                        continuation_scripts.c.id == script_id,
                        continuation_scripts.c.revision == base_revision,
                    )
                    .values(
                        content=content,
                        revision=revision,
                        updated_at=now,
                    )
                )
            connection.execute(
                update(continuation_pages)
                .where(continuation_pages.c.project_id == project_id)
                .values(
                    payload_json=func.json_set(
                        continuation_pages.c.payload_json,
                        "$.staleReason",
                        "script_changed",
                    ),
                    revision=continuation_pages.c.revision + 1,
                )
            )
            response = {
                "scriptId": script_id,
                "projectId": project_id,
                "revision": revision,
                "content": content,
            }
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_script",
                resource_id=script_id,
                now=now,
            )
            return response

    def update_page(
        self,
        *,
        idempotency_key: str,
        page_id: str,
        base_revision: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        base_revision = _positive_integer(base_revision, "baseRevision")
        normalized = _validated_page_payload(payload)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"PATCH:updateContinuationPage:{page_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    "payload": normalized,
                },
                now=now,
            )
            if replay is not None:
                return replay
            changed = connection.execute(
                update(continuation_pages)
                .where(
                    continuation_pages.c.id == page_id,
                    continuation_pages.c.revision == base_revision,
                )
                .values(
                    payload_json=_json(normalized),
                    revision=base_revision + 1,
                )
            )
            if changed.rowcount != 1:
                self._raise_page_cas(connection, page_id)
            row = connection.execute(
                select(continuation_pages).where(
                    continuation_pages.c.id == page_id
                )
            ).mappings().one()
            response = _page_dto(row)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_page",
                resource_id=page_id,
                now=now,
            )
            return response

    def switch_image_version(
        self,
        *,
        idempotency_key: str,
        continuation_page_id: str,
        version: int,
    ) -> dict[str, Any]:
        version = _positive_integer(version, "version")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = (
                "POST:activateContinuationImage:"
                f"{continuation_page_id}:{version}"
            )
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={},
                now=now,
            )
            if replay is not None:
                return replay
            target = connection.execute(
                select(continuation_image_versions).where(
                    continuation_image_versions.c.continuation_page_id
                    == continuation_page_id,
                    continuation_image_versions.c.version == version,
                )
            ).mappings().one_or_none()
            if target is None:
                raise InsightNotFound("continuation image version not found")
            connection.execute(
                update(continuation_image_versions)
                .where(
                    continuation_image_versions.c.continuation_page_id
                    == continuation_page_id
                )
                .values(is_active=False, updated_at=now)
            )
            connection.execute(
                update(continuation_image_versions)
                .where(continuation_image_versions.c.id == target["id"])
                .values(is_active=True, updated_at=now)
            )
            response = {
                "continuationPageId": continuation_page_id,
                "version": version,
                "assetId": str(target["asset_id"]),
            }
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_page",
                resource_id=continuation_page_id,
                now=now,
            )
            return response

    def create_character(
        self,
        *,
        idempotency_key: str,
        project_id: str,
        name: str,
        aliases: Sequence[str],
        enabled: bool,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        name, aliases = _normalize_character_identity(name, aliases)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(payload, Mapping):
            raise ValueError("character payload must be an object")
        normalized_payload = dict(payload)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"POST:createContinuationCharacter:{project_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "name": name,
                    "aliases": aliases,
                    "enabled": enabled,
                    "payload": normalized_payload,
                },
                now=now,
            )
            if replay is not None:
                return replay
            self._require_project(connection, project_id)
            character_id = str(uuid.uuid4())
            connection.execute(
                insert(continuation_characters).values(
                    id=character_id,
                    project_id=project_id,
                    name=name,
                    aliases_json=_json(aliases),
                    enabled=enabled,
                    payload_json=_json(normalized_payload),
                    revision=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            row = connection.execute(
                select(continuation_characters).where(
                    continuation_characters.c.id == character_id
                )
            ).mappings().one()
            response = _character_dto(row)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=201,
                resource_type="continuation_character",
                resource_id=character_id,
                now=now,
            )
            return response

    def update_character(
        self,
        *,
        idempotency_key: str,
        character_id: str,
        base_revision: int,
        name: str,
        aliases: Sequence[str],
        enabled: bool,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        base_revision = _positive_integer(base_revision, "baseRevision")
        name, aliases = _normalize_character_identity(name, aliases)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(payload, Mapping):
            raise ValueError("character payload must be an object")
        normalized_payload = dict(payload)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"PATCH:updateContinuationCharacter:{character_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    "name": name,
                    "aliases": aliases,
                    "enabled": enabled,
                    "payload": normalized_payload,
                },
                now=now,
            )
            if replay is not None:
                return replay
            changed = connection.execute(
                update(continuation_characters)
                .where(
                    continuation_characters.c.id == character_id,
                    continuation_characters.c.revision == base_revision,
                )
                .values(
                    name=name,
                    aliases_json=_json(aliases),
                    enabled=enabled,
                    payload_json=_json(normalized_payload),
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                self._raise_character_cas(connection, character_id)
            row = connection.execute(
                select(continuation_characters).where(
                    continuation_characters.c.id == character_id
                )
            ).mappings().one()
            response = _character_dto(row)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_character",
                resource_id=character_id,
                now=now,
            )
            return response

    def delete_character(
        self,
        *,
        idempotency_key: str,
        character_id: str,
        base_revision: int,
    ) -> None:
        base_revision = _positive_integer(base_revision, "baseRevision")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"DELETE:deleteContinuationCharacter:{character_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={"baseRevision": base_revision},
                now=now,
            )
            if replay is not None:
                return
            changed = connection.execute(
                delete(continuation_characters).where(
                    continuation_characters.c.id == character_id,
                    continuation_characters.c.revision == base_revision,
                )
            )
            if changed.rowcount != 1:
                self._raise_character_cas(connection, character_id)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response={"deleted": True},
                http_status=200,
                resource_type="continuation_character",
                resource_id=character_id,
                now=now,
            )

    def create_form(
        self,
        *,
        idempotency_key: str,
        character_id: str,
        name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        name = _form_name(name)
        if not isinstance(payload, Mapping):
            raise ValueError("form payload must be an object")
        normalized_payload = dict(payload)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"POST:createContinuationForm:{character_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={"name": name, "payload": normalized_payload},
                now=now,
            )
            if replay is not None:
                return replay
            if connection.execute(
                select(continuation_characters.c.id).where(
                    continuation_characters.c.id == character_id
                )
            ).scalar_one_or_none() is None:
                raise InsightNotFound("continuation character not found")
            form_id = str(uuid.uuid4())
            connection.execute(
                insert(continuation_character_forms).values(
                    id=form_id,
                    character_id=character_id,
                    name=name,
                    payload_json=_json(normalized_payload),
                    revision=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            row = connection.execute(
                select(continuation_character_forms).where(
                    continuation_character_forms.c.id == form_id
                )
            ).mappings().one()
            response = self._form_dto(row, image_versions=[])
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=201,
                resource_type="continuation_form",
                resource_id=form_id,
                now=now,
            )
            return response

    def update_form(
        self,
        *,
        idempotency_key: str,
        form_id: str,
        base_revision: int,
        name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        base_revision = _positive_integer(base_revision, "baseRevision")
        name = _form_name(name)
        if not isinstance(payload, Mapping):
            raise ValueError("form payload must be an object")
        normalized_payload = dict(payload)
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"PATCH:updateContinuationForm:{form_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    "name": name,
                    "payload": normalized_payload,
                },
                now=now,
            )
            if replay is not None:
                return replay
            changed = connection.execute(
                update(continuation_character_forms)
                .where(
                    continuation_character_forms.c.id == form_id,
                    continuation_character_forms.c.revision
                    == base_revision,
                )
                .values(
                    name=name,
                    payload_json=_json(normalized_payload),
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                self._raise_form_cas(connection, form_id)
            row = connection.execute(
                select(continuation_character_forms).where(
                    continuation_character_forms.c.id == form_id
                )
            ).mappings().one()
            versions = self._form_versions(connection, form_id)
            response = self._form_dto(row, image_versions=versions)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_form",
                resource_id=form_id,
                now=now,
            )
            return response

    def delete_form(
        self,
        *,
        idempotency_key: str,
        form_id: str,
        base_revision: int,
    ) -> None:
        base_revision = _positive_integer(base_revision, "baseRevision")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"DELETE:deleteContinuationForm:{form_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={"baseRevision": base_revision},
                now=now,
            )
            if replay is not None:
                return
            changed = connection.execute(
                delete(continuation_character_forms).where(
                    continuation_character_forms.c.id == form_id,
                    continuation_character_forms.c.revision
                    == base_revision,
                )
            )
            if changed.rowcount != 1:
                self._raise_form_cas(connection, form_id)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response={"deleted": True},
                http_status=200,
                resource_type="continuation_form",
                resource_id=form_id,
                now=now,
            )

    def bind_form_reference(
        self,
        *,
        idempotency_key: str,
        form_id: str,
        base_revision: int,
        asset_id: str | None,
        thumbnail_asset_id: str | None,
        content_checksum: str | None = None,
    ) -> dict[str, Any]:
        base_revision = _positive_integer(base_revision, "baseRevision")
        if (asset_id is None) != (thumbnail_asset_id is None):
            raise ValueError(
                "reference asset and thumbnail must be set together"
            )
        uploading = asset_id is not None
        if uploading:
            if (
                not isinstance(content_checksum, str)
                or len(content_checksum) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in content_checksum
                )
            ):
                raise ValueError("reference content checksum must be SHA-256")
        elif content_checksum is not None:
            raise ValueError("deleted reference cannot include a checksum")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = (
                f"POST:uploadContinuationReference:{form_id}"
                if uploading
                else f"DELETE:deleteContinuationReference:{form_id}"
            )
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    **(
                        {"contentChecksum": content_checksum}
                        if uploading
                        else {}
                    ),
                },
                now=now,
            )
            if replay is not None:
                return replay
            if uploading:
                if (
                    not isinstance(asset_id, str)
                    or not asset_id
                    or not isinstance(thumbnail_asset_id, str)
                    or not thumbnail_asset_id
                ):
                    raise ValueError(
                        "reference asset and thumbnail IDs are required"
                    )
                existing_assets = set(
                    connection.execute(
                        select(assets.c.id).where(
                            assets.c.id.in_((asset_id, thumbnail_asset_id)),
                            assets.c.owner_user_id == effective_owner_id(),
                        )
                    ).scalars()
                )
                if existing_assets != {asset_id, thumbnail_asset_id}:
                    raise InsightNotFound("reference asset not found")
            changed = connection.execute(
                update(continuation_character_forms)
                .where(
                    continuation_character_forms.c.id == form_id,
                    continuation_character_forms.c.revision
                    == base_revision,
                )
                .values(
                    reference_asset_id=asset_id,
                    reference_thumbnail_asset_id=thumbnail_asset_id,
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                self._raise_form_cas(connection, form_id)
            row = connection.execute(
                select(continuation_character_forms).where(
                    continuation_character_forms.c.id == form_id
                )
            ).mappings().one()
            versions = self._form_versions(connection, form_id)
            response = self._form_dto(row, image_versions=versions)
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_form",
                resource_id=form_id,
                now=now,
            )
            return response

    def replay_form_reference_upload(
        self,
        *,
        idempotency_key: str,
        form_id: str,
        base_revision: int,
        content_checksum: str,
    ) -> dict[str, Any] | None:
        base_revision = _positive_integer(base_revision, "baseRevision")
        if (
            not isinstance(content_checksum, str)
            or len(content_checksum) != 64
            or any(
                character not in "0123456789abcdef"
                for character in content_checksum
            )
        ):
            raise ValueError("reference content checksum must be SHA-256")
        with self.engine.connect() as connection:
            _, replay = _idempotency_replay(
                connection,
                scope=f"POST:uploadContinuationReference:{form_id}",
                key=idempotency_key,
                payload={
                    "baseRevision": base_revision,
                    "contentChecksum": content_checksum,
                },
                now=utcnow(),
            )
            return replay

    def list_forms(
        self,
        *,
        project_id: str,
        cursor: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        if (
            isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or cursor < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 200
        ):
            raise ValueError("invalid continuation form pagination")
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(continuation_character_forms)
                    .join(
                        continuation_characters,
                        continuation_characters.c.id
                        == continuation_character_forms.c.character_id,
                    )
                    .where(
                        continuation_characters.c.project_id == project_id
                    )
                    .order_by(
                        continuation_characters.c.name,
                        continuation_character_forms.c.name,
                    )
                    .offset(cursor)
                    .limit(limit + 1)
                ).mappings()
            )
            has_more = len(rows) > limit
            selected_rows = rows[:limit]
            versions_by_form: dict[str, list[Mapping[str, Any]]] = {}
            if selected_rows:
                for version in connection.execute(
                    select(continuation_form_image_versions)
                    .where(
                        continuation_form_image_versions.c.form_id.in_(
                            [str(row["id"]) for row in selected_rows]
                        )
                    )
                    .order_by(
                        continuation_form_image_versions.c.form_id,
                        continuation_form_image_versions.c.version.desc(),
                    )
                ).mappings():
                    versions_by_form.setdefault(
                        str(version["form_id"]),
                        [],
                    ).append(version)
            try:
                items = [
                    self._form_dto(
                        row,
                        image_versions=versions_by_form.get(
                            str(row["id"]),
                            (),
                        ),
                    )
                    for row in selected_rows
                ]
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise InsightConflict(
                    "continuation form data is invalid; clear the project"
                ) from exc
        return {
            "items": items,
            "nextCursor": cursor + limit if has_more else None,
        }

    def adopt_form_image(
        self,
        *,
        idempotency_key: str,
        form_id: str,
        version: int,
        base_revision: int,
    ) -> dict[str, Any]:
        version = _positive_integer(version, "version")
        base_revision = _positive_integer(base_revision, "baseRevision")
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"POST:adoptContinuationFormImage:{form_id}:{version}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={"baseRevision": base_revision},
                now=now,
            )
            if replay is not None:
                return replay
            target = connection.execute(
                select(continuation_form_image_versions).where(
                    continuation_form_image_versions.c.form_id == form_id,
                    continuation_form_image_versions.c.version == version,
                )
            ).mappings().one_or_none()
            if target is None:
                raise InsightNotFound(
                    "continuation form image version not found"
                )
            changed = connection.execute(
                update(continuation_character_forms)
                .where(
                    continuation_character_forms.c.id == form_id,
                    continuation_character_forms.c.revision
                    == base_revision,
                )
                .values(
                    adopted_asset_id=target["asset_id"],
                    revision=base_revision + 1,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                self._raise_form_cas(connection, form_id)
            connection.execute(
                update(continuation_form_image_versions)
                .where(
                    continuation_form_image_versions.c.form_id == form_id
                )
                .values(is_adopted=False, updated_at=now)
            )
            connection.execute(
                update(continuation_form_image_versions)
                .where(
                    continuation_form_image_versions.c.id == target["id"]
                )
                .values(is_adopted=True, updated_at=now)
            )
            response = {
                "formId": form_id,
                "version": version,
                "assetId": str(target["asset_id"]),
                "revision": base_revision + 1,
            }
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                http_status=200,
                resource_type="continuation_form",
                resource_id=form_id,
                now=now,
            )
            return response

    def clear(self, *, idempotency_key: str, book_id: str) -> None:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            scope = f"DELETE:clearContinuation:{book_id}"
            request_hash, replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                payload={},
                now=now,
            )
            if replay is not None:
                return
            project_id = connection.execute(
                select(continuation_projects.c.id).where(
                    continuation_projects.c.book_id == book_id
                )
            ).scalar_one_or_none()
            if project_id is not None:
                active_job = connection.execute(
                    select(jobs.c.id).where(
                        jobs.c.continuation_project_id == project_id,
                        jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                    ).limit(1)
                ).scalar_one_or_none()
                if active_job is not None:
                    raise InsightConflict(
                        "continuation project is referenced by active work"
                    )
                connection.execute(
                    delete(continuation_projects).where(
                        continuation_projects.c.id == project_id
                    )
                )
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response={"deleted": True},
                http_status=200,
                resource_type="continuation_project",
                resource_id=(str(project_id) if project_id is not None else None),
                now=now,
            )

    def project_by_book(self, book_id: str) -> Mapping[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.book_id == book_id
                )
            ).mappings().one_or_none()
        if row is None:
            raise InsightNotFound("continuation project not found")
        return row

    def _snapshot_or_none(
        self,
        *,
        book_id: str,
    ) -> AnalysisInputSnapshot | None:
        try:
            return self.derived.snapshot(book_id=book_id)
        except (InsightConflict, InsightNotFound):
            return None

    def _project_snapshot_or_none(
        self,
        *,
        book_id: str,
        project: Mapping[str, Any],
    ) -> AnalysisInputSnapshot | None:
        try:
            payload = _required_mapping_json(
                project["payload_json"],
                "continuation project",
            )
            _public_project_config(payload)
            frozen_inputs = _validated_analysis_inputs(
                payload.get("analysisInputs")
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InsightConflict(
                "continuation project schema is invalid; clear the project"
            ) from exc
        expected_fingerprint = payload.get("analysisInputFingerprint")
        if not _is_sha256(expected_fingerprint):
            raise InsightConflict(
                "continuation project schema is invalid; clear the project"
            )
        try:
            snapshot = self.derived.snapshot(
                book_id=book_id,
                frozen_inputs=frozen_inputs,
            )
        except (
            InsightConflict,
            InsightNotFound,
            KeyError,
            TypeError,
            ValueError,
        ):
            return None
        return (
            snapshot
            if snapshot.fingerprint == expected_fingerprint
            else None
        )

    @staticmethod
    def _snapshot_prerequisites(
        connection: Connection,
        *,
        book_id: str,
        fingerprint: str | None,
        active_only: bool,
        allow_stale: bool,
    ) -> tuple[set[tuple[str, str]], Mapping[str, Any] | None]:
        if not fingerprint:
            return set(), None
        statuses = ("ready", "degraded", "stale") if allow_stale else (
            "ready",
            "degraded",
        )
        artifact_conditions = [
            analysis_artifacts.c.book_id == book_id,
            analysis_artifacts.c.dependency_fingerprint == fingerprint,
            analysis_artifacts.c.status.in_(statuses),
        ]
        timeline_conditions = [
            timeline_versions.c.book_id == book_id,
            timeline_versions.c.dependency_fingerprint == fingerprint,
            timeline_versions.c.status.in_(statuses),
        ]
        if active_only:
            artifact_conditions.append(
                analysis_artifacts.c.is_active.is_(True)
            )
            timeline_conditions.append(
                timeline_versions.c.is_active.is_(True)
            )
        prerequisites = set(
            connection.execute(
                select(
                    analysis_artifacts.c.kind,
                    analysis_artifacts.c.template,
                ).where(*artifact_conditions)
            )
        )
        timeline = connection.execute(
            select(
                timeline_versions.c.id,
                timeline_versions.c.status,
                timeline_versions.c.run_id,
            )
            .where(*timeline_conditions)
            .order_by(timeline_versions.c.updated_at.desc())
            .limit(1)
        ).mappings().one_or_none()
        return prerequisites, timeline

    @staticmethod
    def _require_project(
        connection: Connection,
        project_id: str,
    ) -> None:
        if connection.execute(
            select(continuation_projects.c.id).where(
                continuation_projects.c.id == project_id
            )
        ).scalar_one_or_none() is None:
            raise InsightNotFound("continuation project not found")

    @staticmethod
    def _raise_character_cas(
        connection: Connection,
        character_id: str,
    ) -> None:
        if connection.execute(
            select(continuation_characters.c.id).where(
                continuation_characters.c.id == character_id
            )
        ).scalar_one_or_none() is None:
            raise InsightNotFound("continuation character not found")
        raise InsightConflict("continuation character revision changed")

    @staticmethod
    def _raise_page_cas(
        connection: Connection,
        page_id: str,
    ) -> None:
        if connection.execute(
            select(continuation_pages.c.id).where(
                continuation_pages.c.id == page_id
            )
        ).scalar_one_or_none() is None:
            raise InsightNotFound("continuation page not found")
        raise InsightConflict("continuation page revision changed")

    @staticmethod
    def _raise_form_cas(
        connection: Connection,
        form_id: str,
    ) -> None:
        if connection.execute(
            select(continuation_character_forms.c.id).where(
                continuation_character_forms.c.id == form_id
            )
        ).scalar_one_or_none() is None:
            raise InsightNotFound("continuation character form not found")
        raise InsightConflict("continuation character form revision changed")

    @staticmethod
    def _form_versions(
        connection: Connection,
        form_id: str,
    ) -> list[Mapping[str, Any]]:
        return list(
            connection.execute(
                select(continuation_form_image_versions)
                .where(
                    continuation_form_image_versions.c.form_id == form_id
                )
                .order_by(
                    continuation_form_image_versions.c.version.desc()
                )
            ).mappings()
        )

    @staticmethod
    def _form_dto(
        row: Mapping[str, Any],
        *,
        image_versions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        try:
            payload = _required_mapping_json(
                row["payload_json"],
                "continuation form",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InsightConflict(
                "continuation form data is invalid; clear the project"
            ) from exc
        return {
            "formId": str(row["id"]),
            "characterId": str(row["character_id"]),
            "name": str(row["name"]),
            "revision": int(row["revision"]),
            "payload": payload,
            "referenceAssetId": row["reference_asset_id"],
            "referenceAssetUrl": (
                f"/api/v2/assets/{row['reference_asset_id']}"
                if row["reference_asset_id"]
                else None
            ),
            "referenceThumbnailUrl": (
                f"/api/v2/assets/{row['reference_thumbnail_asset_id']}"
                if row["reference_thumbnail_asset_id"]
                else None
            ),
            "adoptedAssetId": row["adopted_asset_id"],
            "imageVersions": [
                {
                    "version": int(version["version"]),
                    "assetId": str(version["asset_id"]),
                    "assetUrl": (
                        f"/api/v2/assets/{version['asset_id']}"
                    ),
                    "thumbnailUrl": (
                        f"/api/v2/assets/"
                        f"{version['thumbnail_asset_id']}"
                    ),
                    "adopted": bool(version["is_adopted"]),
                }
                for version in image_versions
            ],
        }

    def _project_dto(
        self,
        connection: Connection,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        script = connection.execute(
            select(continuation_scripts).where(
                continuation_scripts.c.project_id == row["id"]
            )
        ).mappings().one_or_none()
        pages_rows = list(
            connection.execute(
                select(continuation_pages)
                .where(continuation_pages.c.project_id == row["id"])
                .order_by(continuation_pages.c.ordinal)
            ).mappings()
        )
        characters = list(
            connection.execute(
                select(continuation_characters)
                .where(continuation_characters.c.project_id == row["id"])
                .order_by(continuation_characters.c.name)
            ).mappings()
        )
        references = list(
            connection.execute(
                select(
                    continuation_project_reference_assets.c.asset_id
                )
                .where(
                    continuation_project_reference_assets.c.project_id
                    == row["id"]
                )
                .order_by(
                    continuation_project_reference_assets.c.ordinal
                )
            ).scalars()
        )
        versions_by_page: dict[str, list[Mapping[str, Any]]] = {}
        if pages_rows:
            for version in connection.execute(
                select(continuation_image_versions)
                .where(
                    continuation_image_versions.c.continuation_page_id.in_(
                        [str(page["id"]) for page in pages_rows]
                    )
                )
                .order_by(
                    continuation_image_versions.c.continuation_page_id,
                    continuation_image_versions.c.version.desc(),
                )
            ).mappings():
                versions_by_page.setdefault(
                    str(version["continuation_page_id"]),
                    [],
                ).append(version)
        thumbnail_by_reference: dict[str, str] = {}
        if references:
            for form in connection.execute(
                select(
                    continuation_character_forms.c.reference_asset_id,
                    continuation_character_forms.c.reference_thumbnail_asset_id,
                )
                .where(
                    continuation_character_forms.c.reference_asset_id.in_(
                        references
                    ),
                    continuation_character_forms.c.reference_thumbnail_asset_id.is_not(
                        None
                    ),
                )
                .order_by(continuation_character_forms.c.id)
            ).mappings():
                thumbnail_by_reference.setdefault(
                    str(form["reference_asset_id"]),
                    str(form["reference_thumbnail_asset_id"]),
                )
            for version in connection.execute(
                select(
                    continuation_image_versions.c.asset_id,
                    continuation_image_versions.c.thumbnail_asset_id,
                )
                .where(continuation_image_versions.c.asset_id.in_(references))
                .order_by(continuation_image_versions.c.id)
            ).mappings():
                thumbnail_by_reference.setdefault(
                    str(version["asset_id"]),
                    str(version["thumbnail_asset_id"]),
                )
            reference_source = page_assets.alias(
                "continuation_reference_source"
            )
            reference_thumbnail = page_assets.alias(
                "continuation_reference_thumbnail"
            )
            for pointer in connection.execute(
                select(
                    reference_source.c.asset_id.label("reference_asset_id"),
                    reference_thumbnail.c.asset_id.label("thumbnail_asset_id"),
                )
                .select_from(reference_source)
                .join(
                    reference_thumbnail,
                    (reference_thumbnail.c.page_id == reference_source.c.page_id)
                    & (reference_thumbnail.c.role == "thumbnail_source"),
                )
                .where(
                    reference_source.c.asset_id.in_(references),
                    reference_source.c.role == "source",
                )
            ).mappings():
                thumbnail_by_reference.setdefault(
                    str(pointer["reference_asset_id"]),
                    str(pointer["thumbnail_asset_id"]),
                )
        try:
            project_payload = _required_mapping_json(
                row["payload_json"],
                "continuation project",
            )
            public_config = _public_project_config(project_payload)
            _validated_analysis_inputs(project_payload.get("analysisInputs"))
            if not _is_sha256(
                project_payload.get("analysisInputFingerprint")
            ):
                raise ValueError(
                    "analysisInputFingerprint must be SHA-256"
                )
            page_items = [
                _page_dto(
                    page,
                    image_versions=versions_by_page.get(str(page["id"]), ()),
                )
                for page in pages_rows
            ]
            character_items = [
                _character_dto(character)
                for character in characters
            ]
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise InsightConflict(
                "continuation project data is invalid; clear the project"
            ) from exc
        return {
            "projectId": str(row["id"]),
            "bookId": str(row["book_id"]),
            "sourceRunId": row["source_run_id"],
            "revision": int(row["revision"]),
            "config": public_config,
            "script": (
                {
                    "scriptId": str(script["id"]),
                    "revision": int(script["revision"]),
                    "content": str(script["content"]),
                }
                if script
                else None
            ),
            "pages": page_items,
            "referenceAssets": [
                {
                    "assetId": str(asset_id),
                    "assetUrl": f"/api/v2/assets/{asset_id}",
                    "thumbnailUrl": (
                        f"/api/v2/assets/"
                        f"{thumbnail_by_reference.get(str(asset_id), str(asset_id))}"
                    ),
                }
                for asset_id in references
            ],
            "characters": character_items,
        }

    @staticmethod
    def insert_script_result(
        connection: Connection,
        *,
        project_id: str,
        base_revision: int,
        content: str,
    ) -> int:
        now = utcnow()
        row = connection.execute(
            select(continuation_scripts).where(
                continuation_scripts.c.project_id == project_id
            )
        ).mappings().one_or_none()
        if row is None:
            if base_revision != 0:
                raise JobConflict("frozen continuation script disappeared")
            connection.execute(
                insert(continuation_scripts).values(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    revision=1,
                    content=content,
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                update(continuation_pages)
                .where(continuation_pages.c.project_id == project_id)
                .values(
                    payload_json=func.json_set(
                        continuation_pages.c.payload_json,
                        "$.staleReason",
                        "script_changed",
                    ),
                    revision=continuation_pages.c.revision + 1,
                )
            )
            return 1
        if int(row["revision"]) != base_revision:
            raise JobConflict(
                "continuation script was edited while generation ran"
            )
        connection.execute(
            update(continuation_scripts)
            .where(
                continuation_scripts.c.id == row["id"],
                continuation_scripts.c.revision == base_revision,
            )
            .values(
                revision=base_revision + 1,
                content=content,
                updated_at=now,
            )
        )
        connection.execute(
            update(continuation_pages)
            .where(continuation_pages.c.project_id == project_id)
            .values(
                payload_json=func.json_set(
                    continuation_pages.c.payload_json,
                    "$.staleReason",
                    "script_changed",
                ),
                revision=continuation_pages.c.revision + 1,
            )
        )
        return base_revision + 1

    @staticmethod
    def insert_page_result(
        connection: Connection,
        *,
        page_id: str,
        base_revision: int,
        payload: Mapping[str, Any],
    ) -> int:
        normalized = _validated_page_payload(payload)
        changed = connection.execute(
            update(continuation_pages)
            .where(
                continuation_pages.c.id == page_id,
                continuation_pages.c.revision == base_revision,
            )
            .values(
                revision=base_revision + 1,
                payload_json=_json(normalized),
            )
        )
        if changed.rowcount != 1:
            raise JobConflict(
                "continuation page was edited while generation ran"
            )
        return base_revision + 1


class ContinuationCommandService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.repository = ContinuationRepository(engine)
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)

    @staticmethod
    def _assert_project_snapshot(
        connection: Connection,
        *,
        project_id: str,
        book_id: str,
        revision: int,
    ) -> None:
        current = connection.execute(
            select(
                continuation_projects.c.id,
                continuation_projects.c.revision,
            ).where(
                continuation_projects.c.id == project_id,
                continuation_projects.c.book_id == book_id,
            )
        ).mappings().one_or_none()
        if current is None or int(current["revision"]) != revision:
            raise InsightConflict(
                "continuation project changed while the job was being created"
            )

    @staticmethod
    def _assert_script_snapshot(
        connection: Connection,
        *,
        project_id: str,
        script_id: str | None,
        revision: int,
    ) -> None:
        current = connection.execute(
            select(
                continuation_scripts.c.id,
                continuation_scripts.c.revision,
            ).where(continuation_scripts.c.project_id == project_id)
        ).mappings().one_or_none()
        if script_id is None:
            unchanged = current is None and revision == 0
        else:
            unchanged = (
                current is not None
                and str(current["id"]) == script_id
                and int(current["revision"]) == revision
            )
        if not unchanged:
            raise InsightConflict(
                "continuation script changed while the job was being created"
            )

    @staticmethod
    def _assert_page_snapshots(
        connection: Connection,
        *,
        project_id: str,
        targets: Sequence[Mapping[str, Any]],
        ordinals: Sequence[int] | None,
    ) -> None:
        statement = select(
            continuation_pages.c.id,
            continuation_pages.c.ordinal,
            continuation_pages.c.revision,
        ).where(continuation_pages.c.project_id == project_id)
        if ordinals is not None:
            statement = statement.where(
                continuation_pages.c.ordinal.in_(tuple(ordinals))
            )
        current = {
            int(row["ordinal"]): (
                str(row["id"]),
                int(row["revision"]),
            )
            for row in connection.execute(statement).mappings()
        }
        expected = {
            int(target["ordinal"]): (
                str(target["pageId"]),
                int(target["baseRevision"]),
            )
            for target in targets
        }
        if current != expected:
            raise InsightConflict(
                "continuation pages changed while the job was being created"
            )

    def create_script_job(
        self,
        *,
        book_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        idempotency_scope = f"POST:createContinuationJob:{book_id}"
        idempotency_payload = {"kind": "script"}
        replay = self.jobs.idempotency_replay(
            scope=idempotency_scope,
            key=idempotency_key,
            payload=idempotency_payload,
        )
        if replay is not None:
            return replay
        project = self.repository.project_by_book(book_id)
        with self.engine.connect() as connection:
            script = connection.execute(
                select(continuation_scripts).where(
                    continuation_scripts.c.project_id == project["id"]
                )
            ).mappings().one_or_none()
        project_id = str(project["id"])
        project_revision = int(project["revision"])
        script_id = str(script["id"]) if script else None
        script_revision = int(script["revision"]) if script else 0
        config = self._config(book_id, project)
        config.update(
            {
                "continuationAction": "script",
                "projectId": project_id,
                "baseScriptRevision": script_revision,
            }
        )

        def initialize(connection: Connection, _batch_id: str) -> None:
            self._assert_project_snapshot(
                connection,
                project_id=project_id,
                book_id=book_id,
                revision=project_revision,
            )
            self._assert_script_snapshot(
                connection,
                project_id=project_id,
                script_id=script_id,
                revision=script_revision,
            )

        return self.jobs.create_batch(
            display_name="续写 · 生成脚本",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=(
                        str(project["source_run_id"])
                        if project["source_run_id"]
                        else None
                    ),
                    continuation_project_id=project_id,
                    config=config,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("continuation_generate_script",),
                        ),
                    ),
                ),
            ),
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            idempotency_payload=idempotency_payload,
            transaction_initializer=initialize,
        )

    def create_pages_job(
        self,
        *,
        book_id: str,
        ordinals: Sequence[int] | None,
        idempotency_key: str,
    ) -> dict[str, object]:
        requested_ordinals = _selected_ordinals(ordinals)
        idempotency_scope = f"POST:createContinuationJob:{book_id}"
        idempotency_payload = {
            "kind": "pages",
            "ordinals": requested_ordinals,
        }
        replay = self.jobs.idempotency_replay(
            scope=idempotency_scope,
            key=idempotency_key,
            payload=idempotency_payload,
        )
        if replay is not None:
            return replay
        project = self.repository.project_by_book(book_id)
        try:
            project_config = _public_project_config(
                _required_mapping_json(
                    project["payload_json"],
                    "continuation project",
                )
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InsightConflict(
                "continuation project snapshot is invalid; clear the project"
            ) from exc
        page_count = project_config["pageCount"]
        selected = requested_ordinals
        if selected is None:
            selected = list(range(1, page_count + 1))
        if selected[-1] > page_count:
            raise ValueError("continuation page ordinal is out of range")
        config = self._config(book_id, project)
        project_id = str(project["id"])
        project_revision = int(project["revision"])
        with self.engine.connect() as connection:
            script = connection.execute(
                select(continuation_scripts).where(
                    continuation_scripts.c.project_id == project_id
                )
            ).mappings().one_or_none()
            if script is None or not str(script["content"]).strip():
                raise InsightConflict("generate or save a script first")
            existing = {
                int(row["ordinal"]): row
                for row in connection.execute(
                    select(continuation_pages).where(
                        continuation_pages.c.project_id == project_id,
                        continuation_pages.c.ordinal.in_(tuple(selected)),
                    )
                ).mappings()
            }
            targets = []
            for ordinal in selected:
                row = existing.get(ordinal)
                if row is None:
                    page_id = str(uuid.uuid4())
                    targets.append(
                        {
                            "pageId": page_id,
                            "ordinal": ordinal,
                            "baseRevision": 1,
                        }
                    )
                else:
                    targets.append(
                        {
                            "pageId": str(row["id"]),
                            "ordinal": ordinal,
                            "baseRevision": int(row["revision"]),
                        }
                    )
        script_id = str(script["id"])
        script_revision = int(script["revision"])
        existing_targets = [
            target
            for target in targets
            if int(target["ordinal"]) in existing
        ]
        missing_targets = [
            target
            for target in targets
            if int(target["ordinal"]) not in existing
        ]
        config.update(
            {
                "continuationAction": "pages",
                "projectId": project_id,
                "script": str(script["content"]),
                "targets": targets,
            }
        )

        def initialize(connection: Connection, _batch_id: str) -> None:
            self._assert_project_snapshot(
                connection,
                project_id=project_id,
                book_id=book_id,
                revision=project_revision,
            )
            self._assert_script_snapshot(
                connection,
                project_id=project_id,
                script_id=script_id,
                revision=script_revision,
            )
            self._assert_page_snapshots(
                connection,
                project_id=project_id,
                targets=existing_targets,
                ordinals=selected,
            )
            if missing_targets:
                connection.execute(
                    insert(continuation_pages),
                    [
                        {
                            "id": str(target["pageId"]),
                            "project_id": project_id,
                            "ordinal": int(target["ordinal"]),
                            "revision": 1,
                            "payload_json": _json(_empty_page_payload()),
                        }
                        for target in missing_targets
                    ],
                )
            self._assert_page_snapshots(
                connection,
                project_id=project_id,
                targets=targets,
                ordinals=selected,
            )

        return self.jobs.create_batch(
            display_name="续写 · 页面剧情",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=(
                        str(project["source_run_id"])
                        if project["source_run_id"]
                        else None
                    ),
                    continuation_project_id=project_id,
                    config=config,
                    items=tuple(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("continuation_generate_page",),
                        )
                        for _target in targets
                    ),
                ),
            ),
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            idempotency_payload=idempotency_payload,
            transaction_initializer=initialize,
        )

    def create_images_job(
        self,
        *,
        book_id: str,
        ordinals: Sequence[int] | None,
        idempotency_key: str,
    ) -> dict[str, object]:
        selected = _selected_ordinals(ordinals)
        idempotency_scope = f"POST:createContinuationJob:{book_id}"
        idempotency_payload = {
            "kind": "images",
            "ordinals": selected,
        }
        replay = self.jobs.idempotency_replay(
            scope=idempotency_scope,
            key=idempotency_key,
            payload=idempotency_payload,
        )
        if replay is not None:
            return replay
        project = self.repository.project_by_book(book_id)
        project_id = str(project["id"])
        project_revision = int(project["revision"])
        try:
            project_config = _public_project_config(
                _required_mapping_json(
                    project["payload_json"],
                    "continuation project",
                )
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InsightConflict(
                "continuation project snapshot is invalid; clear the project"
            ) from exc
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(continuation_pages)
                    .where(
                        continuation_pages.c.project_id == project_id
                    )
                    .order_by(continuation_pages.c.ordinal)
                ).mappings()
            )
            initial_reference_ids = [
                str(value)
                for value in connection.execute(
                    select(
                        continuation_project_reference_assets.c.asset_id
                    )
                    .where(
                        continuation_project_reference_assets.c.project_id
                        == project_id
                    )
                    .order_by(
                        continuation_project_reference_assets.c.ordinal
                    )
                ).scalars()
            ]
            reference_count = project_config["styleReferencePages"]
            if len(initial_reference_ids) < reference_count:
                fallback_ids = [
                    str(value)
                    for value in connection.execute(
                        select(page_assets.c.asset_id)
                        .join(pages, pages.c.id == page_assets.c.page_id)
                        .join(chapters, chapters.c.id == pages.c.chapter_id)
                        .where(
                            chapters.c.book_id == book_id,
                            page_assets.c.role == "source",
                            page_assets.c.asset_id.not_in(
                                tuple(initial_reference_ids)
                            ),
                        )
                        .order_by(
                            chapters.c.ordinal.desc(),
                            pages.c.ordinal.desc(),
                        )
                        .limit(
                            reference_count - len(initial_reference_ids)
                        )
                    ).scalars()
                ]
                initial_reference_ids.extend(fallback_ids)
        selected_set = set(selected) if selected is not None else None
        available_ordinals = {int(row["ordinal"]) for row in rows}
        if selected_set is not None and not selected_set.issubset(available_ordinals):
            raise ValueError("continuation page ordinal is out of range")
        targets = [
            {
                "pageId": str(row["id"]),
                "ordinal": int(row["ordinal"]),
                "baseRevision": int(row["revision"]),
                "payload": _validated_page_payload(
                    _mapping(_load(row["payload_json"]))
                ),
            }
            for row in rows
            if selected_set is None or int(row["ordinal"]) in selected_set
        ]
        if not targets:
            raise InsightConflict("continuation has no pages to illustrate")
        if any(
            not str(target["payload"].get("finalPrompt", "")).strip()
            for target in targets
        ):
            raise InsightConflict(
                "every selected continuation page requires finalPrompt"
            )
        config = self._config(book_id, project)
        config.update(
            {
                "continuationAction": "images",
                "projectId": project_id,
                "targets": targets,
                "initialReferenceAssetIds": initial_reference_ids,
            }
        )
        frozen_references = {
            f"style_reference_{index}": asset_id
            for index, asset_id in enumerate(initial_reference_ids, start=1)
        }

        def initialize(connection: Connection, _batch_id: str) -> None:
            self._assert_project_snapshot(
                connection,
                project_id=project_id,
                book_id=book_id,
                revision=project_revision,
            )
            self._assert_page_snapshots(
                connection,
                project_id=project_id,
                targets=targets,
                ordinals=(selected if selected is not None else None),
            )

        return self.jobs.create_batch(
            display_name="续写 · 批量生图",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=(
                        str(project["source_run_id"])
                        if project["source_run_id"]
                        else None
                    ),
                    continuation_project_id=project_id,
                    config=config,
                    items=tuple(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("continuation_generate_image",),
                            asset_inputs=(
                                frozen_references
                                if index == 0 and frozen_references
                                else None
                            ),
                        )
                        for index, _target in enumerate(targets)
                    ),
                ),
            ),
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            idempotency_payload=idempotency_payload,
            transaction_initializer=initialize,
        )

    def create_export_job(
        self,
        *,
        book_id: str,
        output_format: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        if output_format not in {"zip", "pdf"}:
            raise ValueError("format must be zip or pdf")
        idempotency_scope = f"POST:createContinuationJob:{book_id}"
        idempotency_payload = {
            "kind": "export",
            "format": output_format,
        }
        replay = self.jobs.idempotency_replay(
            scope=idempotency_scope,
            key=idempotency_key,
            payload=idempotency_payload,
        )
        if replay is not None:
            return replay
        project = self.repository.project_by_book(book_id)
        project_id = str(project["id"])
        project_revision = int(project["revision"])
        with self.engine.connect() as connection:
            images = [
                {
                    "ordinal": int(ordinal),
                    "assetId": str(asset_id),
                }
                for ordinal, asset_id in connection.execute(
                    select(
                        continuation_pages.c.ordinal,
                        continuation_image_versions.c.asset_id,
                    )
                    .join(
                        continuation_image_versions,
                        continuation_image_versions.c.continuation_page_id
                        == continuation_pages.c.id,
                    )
                    .where(
                        continuation_pages.c.project_id == project_id,
                        continuation_image_versions.c.is_active.is_(True),
                    )
                    .order_by(continuation_pages.c.ordinal)
                )
            ]
        if not images:
            raise InsightConflict(
                "continuation has no active images to export"
            )
        config = self._config(book_id, project)
        config.update(
            {
                "continuationAction": "export",
                "projectId": project_id,
                "format": output_format,
                "images": images,
            }
        )

        def initialize(connection: Connection, _batch_id: str) -> None:
            self._assert_project_snapshot(
                connection,
                project_id=project_id,
                book_id=book_id,
                revision=project_revision,
            )
            current_images = [
                {
                    "ordinal": int(ordinal),
                    "assetId": str(asset_id),
                }
                for ordinal, asset_id in connection.execute(
                    select(
                        continuation_pages.c.ordinal,
                        continuation_image_versions.c.asset_id,
                    )
                    .join(
                        continuation_image_versions,
                        continuation_image_versions.c.continuation_page_id
                        == continuation_pages.c.id,
                    )
                    .where(
                        continuation_pages.c.project_id == project_id,
                        continuation_image_versions.c.is_active.is_(True),
                    )
                    .order_by(continuation_pages.c.ordinal)
                )
            ]
            if current_images != images:
                raise InsightConflict(
                    "continuation images changed while the job was being created"
                )

        return self.jobs.create_batch(
            display_name=f"续写 · 导出 {output_format.upper()}",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=(
                        str(project["source_run_id"])
                        if project["source_run_id"]
                        else None
                    ),
                    continuation_project_id=project_id,
                    config=config,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("continuation_export",),
                            asset_inputs={
                                f"continuation_page_{image['ordinal']}": str(
                                    image["assetId"]
                                )
                                for image in images
                            },
                        ),
                    ),
                ),
            ),
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            idempotency_payload=idempotency_payload,
            transaction_initializer=initialize,
        )

    def create_character_sheet_job(
        self,
        *,
        book_id: str,
        form_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        if not isinstance(form_id, str) or not form_id:
            raise ValueError("formId is required")
        idempotency_scope = f"POST:createContinuationJob:{book_id}"
        idempotency_payload = {
            "kind": "character_sheet",
            "formId": form_id,
        }
        replay = self.jobs.idempotency_replay(
            scope=idempotency_scope,
            key=idempotency_key,
            payload=idempotency_payload,
        )
        if replay is not None:
            return replay
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    continuation_character_forms,
                    continuation_characters.c.name.label("character_name"),
                    continuation_characters.c.revision.label(
                        "character_revision"
                    ),
                    continuation_characters.c.project_id,
                )
                .join(
                    continuation_characters,
                    continuation_characters.c.id
                    == continuation_character_forms.c.character_id,
                )
                .join(
                    continuation_projects,
                    continuation_projects.c.id
                    == continuation_characters.c.project_id,
                )
                .where(
                    continuation_character_forms.c.id == form_id,
                    continuation_projects.c.book_id == book_id,
                )
            ).mappings().one_or_none()
        if row is None:
            raise InsightNotFound("continuation character form not found")
        project = self.repository.project_by_book(book_id)
        project_id = str(project["id"])
        project_revision = int(project["revision"])
        character_id = str(row["character_id"])
        character_revision = int(row["character_revision"])
        form_revision = int(row["revision"])
        try:
            form_payload = _required_mapping_json(
                row["payload_json"],
                "continuation form",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InsightConflict(
                "continuation character form is invalid; delete and recreate it"
            ) from exc
        config = self._config(book_id, project)
        config.update(
            {
                "continuationAction": "character_sheet",
                "projectId": project_id,
                "formId": form_id,
                "baseFormRevision": form_revision,
                "characterName": str(row["character_name"]),
                "formName": str(row["name"]),
                "formPayload": form_payload,
                "referenceAssetId": row["reference_asset_id"],
            }
        )
        asset_inputs = (
            {"character_reference": str(row["reference_asset_id"])}
            if row["reference_asset_id"]
            else None
        )

        def initialize(connection: Connection, _batch_id: str) -> None:
            self._assert_project_snapshot(
                connection,
                project_id=project_id,
                book_id=book_id,
                revision=project_revision,
            )
            current = connection.execute(
                select(
                    continuation_character_forms.c.revision.label(
                        "form_revision"
                    ),
                    continuation_characters.c.id.label("character_id"),
                    continuation_characters.c.revision.label(
                        "character_revision"
                    ),
                    continuation_characters.c.project_id,
                )
                .join(
                    continuation_characters,
                    continuation_characters.c.id
                    == continuation_character_forms.c.character_id,
                )
                .where(continuation_character_forms.c.id == form_id)
            ).mappings().one_or_none()
            if (
                current is None
                or str(current["project_id"]) != project_id
                or str(current["character_id"]) != character_id
                or int(current["character_revision"]) != character_revision
                or int(current["form_revision"]) != form_revision
            ):
                raise InsightConflict(
                    "continuation character form changed while the job was being created"
                )

        return self.jobs.create_batch(
            display_name=(
                f"续写 · {row['character_name']} · {row['name']} 三视图"
            ),
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=(
                        str(project["source_run_id"])
                        if project["source_run_id"]
                        else None
                    ),
                    continuation_project_id=project_id,
                    config=config,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=(
                                "continuation_generate_character_sheet",
                            ),
                            asset_inputs=asset_inputs,
                        ),
                    ),
                ),
            ),
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            idempotency_payload=idempotency_payload,
            transaction_initializer=initialize,
        )

    def _config(
        self,
        book_id: str,
        project: Mapping[str, Any],
    ) -> dict[str, Any]:
        config = self.settings.resolve_insight(
            book_id=book_id,
            scope="full",
        )
        try:
            project_payload = _required_mapping_json(
                project["payload_json"],
                "continuation project",
            )
            project_config = _public_project_config(project_payload)
            analysis_inputs = _validated_analysis_inputs(
                project_payload.get("analysisInputs")
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InsightConflict(
                "continuation project snapshot is invalid; clear the project"
            ) from exc
        analysis_fingerprint = project_payload.get(
            "analysisInputFingerprint"
        )
        if not _is_sha256(analysis_fingerprint):
            raise InsightConflict(
                "continuation project analysis snapshot is invalid; clear the project"
            )
        config.update(
            {
                "bookId": book_id,
                "sourceRunId": project["source_run_id"],
                "projectRevision": int(project["revision"]),
                "projectConfig": project_config,
                "analysisInputs": analysis_inputs,
                "analysisInputFingerprint": analysis_fingerprint,
            }
        )
        return config


class ContinuationAlgorithms(Protocol):
    def generate_script(
        self,
        *,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> str: ...

    def generate_page(
        self,
        *,
        ordinal: int,
        script: str,
        previous: Mapping[str, Any] | None,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def generate_image(
        self,
        *,
        prompt: str,
        reference_paths: Sequence[Path],
        config: Mapping[str, Any],
    ) -> bytes: ...


class DefaultContinuationAlgorithms:
    def generate_script(
        self,
        *,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> str:
        from src.backend_v2.insight.derived import ProviderDerivedAlgorithms

        prompt = (
            "根据已发布漫画分析继续创作一话漫画脚本。"
            "严格遵守指定页数和方向，输出 JSON："
            '{"script":"可供逐页拆解的完整脚本"}。\n\n'
            + _json(context)
        )
        result = ProviderDerivedAlgorithms._chat_json(
            prompt,
            config=config,
            prompt_type="book_overview",
        )
        if not isinstance(result, Mapping):
            raise ValueError("continuation script response is not JSON")
        script = result.get("script")
        if not isinstance(script, str) or not script.strip():
            raise ValueError("continuation script response is missing script")
        return script.strip()

    def generate_page(
        self,
        *,
        ordinal: int,
        script: str,
        previous: Mapping[str, Any] | None,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from src.backend_v2.insight.derived import ProviderDerivedAlgorithms

        prompt = (
            f"把续写脚本拆成第 {ordinal} 页。输出 JSON："
            '{"storyText":"...","continuityText":"...",'
            '"dialogueText":"...","characters":[],'
            '"finalPrompt":"..."}。\n\n'
            f"上一页：{_json(previous or {})}\n\n脚本：{script}"
        )
        result = ProviderDerivedAlgorithms._chat_json(
            prompt,
            config=config,
            prompt_type="group_summary",
        )
        if not isinstance(result, Mapping):
            raise ValueError("continuation page response is not JSON")
        return _validated_generated_page(result)

    def generate_image(
        self,
        *,
        prompt: str,
        reference_paths: Sequence[Path],
        config: Mapping[str, Any],
    ) -> bytes:
        from src.core.manga_insight.clients.image_gen_client import (
            ImageGenClient,
        )
        client = ImageGenClient(frozen_image_gen_config(config))

        async def execute() -> bytes:
            try:
                return await client.generate(
                    prompt,
                    reference_images=[
                        {"path": str(path), "type": "style"}
                        for path in reference_paths
                    ],
                )
            finally:
                await client.close()

        return asyncio.run(execute())


class ContinuationWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        algorithms: ContinuationAlgorithms | None = None,
    ) -> None:
        self.engine = engine
        self.jobs = jobs
        self.storage = AssetStorageService(data_root, engine)
        self.derived = InsightDerivedRepository(engine)
        self.credentials = SettingsRepository(engine)
        self.algorithms = algorithms or DefaultContinuationAlgorithms()

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(step.get("config"), Mapping):
            raise JobConflict("continuation job configuration is invalid")
        raw_config = dict(step["config"])
        kind = str(step["stepKind"])
        if kind in {"continuation_generate_script", "continuation_generate_page"}:
            credential_sections = ("chat",)
        elif kind in {
            "continuation_generate_character_sheet",
            "continuation_generate_image",
        }:
            credential_sections = ("imageGen",)
        elif kind == "continuation_export":
            credential_sections = ()
        else:
            raise JobConflict(f"unsupported continuation step: {kind}")
        expected_action = {
            "continuation_generate_script": "script",
            "continuation_generate_page": "pages",
            "continuation_generate_character_sheet": "character_sheet",
            "continuation_generate_image": "images",
            "continuation_export": "export",
        }[kind]
        if raw_config.get("continuationAction") != expected_action:
            raise JobConflict("continuation job action does not match its step")
        project_id = raw_config.get("projectId")
        if not isinstance(project_id, str) or not project_id:
            raise JobConflict("continuation projectId is invalid")
        config = (
            self._with_credentials(
                raw_config,
                section_names=credential_sections,
            )
            if credential_sections
            else raw_config
        )
        if kind == "continuation_generate_script":
            base_script_revision = config.get("baseScriptRevision")
            if (
                isinstance(base_script_revision, bool)
                or not isinstance(base_script_revision, int)
                or base_script_revision < 0
            ):
                raise JobConflict("continuation script revision is invalid")
            context = self._script_context(config)
            content = self.algorithms.generate_script(
                context=context,
                config=config,
            )
            checkpoint: dict[str, Any] = {}

            def publish(connection: Connection) -> None:
                checkpoint.update(
                    {
                        "projectId": config["projectId"],
                        "scriptRevision": (
                            ContinuationRepository.insert_script_result(
                                connection,
                                project_id=project_id,
                                base_revision=base_script_revision,
                                content=content,
                            )
                        ),
                    }
                )

        elif kind == "continuation_generate_page":
            target = _continuation_job_target(config, step)
            script = config.get("script")
            if not isinstance(script, str) or not script.strip():
                raise JobConflict("continuation script snapshot is invalid")
            try:
                target_ordinal = _positive_integer(
                    target.get("ordinal"),
                    "continuation target ordinal",
                )
                target_revision = _positive_integer(
                    target.get("baseRevision"),
                    "continuation target revision",
                )
            except ValueError as exc:
                raise JobConflict(str(exc)) from exc
            page_id = target.get("pageId")
            if not isinstance(page_id, str) or not page_id:
                raise JobConflict("continuation target pageId is invalid")
            with self.engine.connect() as connection:
                current = connection.execute(
                    select(continuation_pages).where(
                        continuation_pages.c.id == page_id
                    )
                ).mappings().one_or_none()
                if current is not None:
                    existing = _validated_page_payload(
                        _mapping(_load(current["payload_json"]))
                    )
                    previous = connection.execute(
                        select(continuation_pages.c.payload_json).where(
                            continuation_pages.c.project_id
                            == current["project_id"],
                            continuation_pages.c.ordinal
                            == target_ordinal - 1,
                        )
                    ).scalar_one_or_none()
                else:
                    existing = None
                    previous = None
            if current is None or int(current["revision"]) != target_revision:
                payload = None
                skipped = True
                skip_reason = "page_changed_before_generation"
            elif (
                existing is not None
                and existing.get("status") == "ready"
                and existing.get("storyText")
                and not existing.get("staleReason")
            ):
                payload = existing
                skipped = True
                skip_reason = "existing_page_content"
            else:
                payload = _validated_page_payload(
                    self.algorithms.generate_page(
                        ordinal=target_ordinal,
                        script=script,
                        previous=(
                            _validated_page_payload(
                                _mapping(_load(previous))
                            )
                            if previous
                            else None
                        ),
                        config=config,
                    )
                )
                skipped = False
                skip_reason = None
            checkpoint = {}

            def publish(connection: Connection) -> bool:
                if skipped:
                    checkpoint.update(
                        {
                            "continuationPageId": page_id,
                            "ordinal": target_ordinal,
                            "skipped": True,
                            "reason": skip_reason,
                        }
                    )
                    return False
                current_revision = connection.execute(
                    select(continuation_pages.c.revision).where(
                        continuation_pages.c.id == page_id
                    )
                ).scalar_one_or_none()
                if current_revision != target_revision:
                    checkpoint.update(
                        {
                            "continuationPageId": page_id,
                            "ordinal": target_ordinal,
                            "skipped": True,
                            "reason": "page_edited_during_generation",
                        }
                    )
                    return False
                if payload is None:
                    raise JobConflict(
                        "continuation page result is missing before publication"
                    )
                revision = ContinuationRepository.insert_page_result(
                    connection,
                    page_id=page_id,
                    base_revision=target_revision,
                    payload=payload,
                )
                checkpoint.update(
                    {
                        "continuationPageId": page_id,
                        "ordinal": target_ordinal,
                        "pageRevision": revision,
                        "skipped": False,
                    }
                )
                return True
        elif kind == "continuation_generate_character_sheet":
            form_id = config.get("formId")
            character_name = config.get("characterName")
            form_name = config.get("formName")
            form_payload = config.get("formPayload")
            base_form_revision = config.get("baseFormRevision")
            if (
                not isinstance(form_id, str)
                or not form_id
                or not isinstance(character_name, str)
                or not character_name
                or not isinstance(form_name, str)
                or not form_name
                or not isinstance(form_payload, Mapping)
                or isinstance(base_form_revision, bool)
                or not isinstance(base_form_revision, int)
                or base_form_revision < 1
            ):
                raise JobConflict("continuation character form snapshot is invalid")
            reference_paths: list[Path] = []
            reference_asset_id = config.get("referenceAssetId")
            if reference_asset_id is not None and (
                not isinstance(reference_asset_id, str)
                or not reference_asset_id
            ):
                raise JobConflict("character form reference asset is invalid")
            if reference_asset_id:
                with self.engine.connect() as connection:
                    relative_path = connection.execute(
                        select(assets.c.relative_path).where(
                            assets.c.id == reference_asset_id
                        )
                    ).scalar_one_or_none()
                if relative_path is None:
                    raise JobConflict(
                        "character form reference asset is missing"
                    )
                reference_paths.append(
                    self.storage.resolve_relative_path(
                        str(relative_path)
                    )
                )
            prompt = (
                "生成同一角色同一形态的正面、侧面、背面三视图角色设定图。"
                "保持服装、发型、配色和比例一致，使用干净背景。\n"
                f"角色：{character_name}\n"
                f"形态：{form_name}\n"
                f"设定：{_json(dict(form_payload))}"
            )
            image_bytes = self.algorithms.generate_image(
                prompt=prompt,
                reference_paths=reference_paths,
                config=config,
            )
            asset, thumbnail_asset = self._publish_generated_image(image_bytes)
            checkpoint = {}

            def publish(connection: Connection) -> None:
                changed = connection.execute(
                    update(continuation_character_forms)
                    .where(
                        continuation_character_forms.c.id == form_id,
                        continuation_character_forms.c.revision
                        == base_form_revision,
                    )
                    .values(
                        revision=base_form_revision + 1,
                        updated_at=utcnow(),
                    )
                )
                if changed.rowcount != 1:
                    raise JobConflict(
                        "character form changed before sheet publication"
                    )
                version = int(
                    connection.execute(
                        select(
                            func.coalesce(
                                func.max(
                                    continuation_form_image_versions.c.version
                                ),
                                0,
                            )
                            + 1
                        ).where(
                            continuation_form_image_versions.c.form_id
                            == form_id
                        )
                    ).scalar_one()
                )
                connection.execute(
                    insert(continuation_form_image_versions).values(
                        id=str(uuid.uuid4()),
                        form_id=form_id,
                        asset_id=asset.id,
                        thumbnail_asset_id=thumbnail_asset.id,
                        version=version,
                        is_adopted=False,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )
                checkpoint.update(
                    {
                        "formId": form_id,
                        "formRevision": base_form_revision + 1,
                        "version": version,
                        "assetId": asset.id,
                        "thumbnailAssetId": thumbnail_asset.id,
                    }
                )
        elif kind == "continuation_generate_image":
            target = _continuation_job_target(config, step)
            page_id = target.get("pageId")
            try:
                target_ordinal = _positive_integer(
                    target.get("ordinal"),
                    "continuation target ordinal",
                )
                target_revision = _positive_integer(
                    target.get("baseRevision"),
                    "continuation target revision",
                )
                target_payload = _validated_page_payload(
                    _mapping(target.get("payload"))
                )
                project_config = _validated_project_config(
                    _mapping(config.get("projectConfig"))
                )
            except (TypeError, ValueError) as exc:
                raise JobConflict(
                    "continuation image snapshot is invalid"
                ) from exc
            if not isinstance(page_id, str) or not page_id:
                raise JobConflict("continuation target pageId is invalid")
            with self.engine.connect() as connection:
                current_revision = connection.execute(
                    select(continuation_pages.c.revision).where(
                        continuation_pages.c.id == page_id
                    )
                ).scalar_one_or_none()
            image_skipped = current_revision != target_revision
            checkpoint = {}
            if image_skipped:
                asset = None
                thumbnail_asset = None
            else:
                initial_reference_ids = config.get("initialReferenceAssetIds")
                if (
                    not isinstance(initial_reference_ids, list)
                    or any(
                        not isinstance(value, str) or not value
                        for value in initial_reference_ids
                    )
                ):
                    raise JobConflict(
                        "continuation initial reference snapshot is invalid"
                    )
                reference_asset_ids = self._reference_window_asset_ids(
                    project_id=project_id,
                    before_ordinal=target_ordinal,
                    count=project_config["styleReferencePages"],
                    initial_asset_ids=initial_reference_ids,
                )
                frozen_references = self.jobs.bind_explicit_item_inputs(
                    fence,
                    item_id=str(step["itemId"]),
                    assets_by_role={
                        f"continuation_reference_{index:03d}": asset_id
                        for index, asset_id in enumerate(
                            reference_asset_ids,
                            start=1,
                        )
                    },
                )
                reference_paths = [
                    self.storage.resolve_relative_path(
                        str(frozen_references[role]["relative_path"])
                    )
                    for role in sorted(frozen_references)
                ]
                image_bytes = self.algorithms.generate_image(
                    prompt=target_payload["finalPrompt"],
                    reference_paths=reference_paths,
                    config=config,
                )
                asset, thumbnail_asset = self._publish_generated_image(
                    image_bytes
                )

            def publish(connection: Connection) -> bool:
                if image_skipped:
                    checkpoint.update(
                        {
                            "continuationPageId": page_id,
                            "skipped": True,
                            "reason": "page_prompt_changed_before_generation",
                        }
                    )
                    return False
                current_revision = connection.execute(
                    select(continuation_pages.c.revision).where(
                        continuation_pages.c.id == page_id
                    )
                ).scalar_one_or_none()
                if current_revision != target_revision:
                    checkpoint.update(
                        {
                            "continuationPageId": page_id,
                            "skipped": True,
                            "reason": "page_prompt_changed_during_generation",
                        }
                    )
                    return False
                if asset is None or thumbnail_asset is None:
                    raise JobConflict(
                        "continuation image result is missing before publication"
                    )
                version = int(
                    connection.execute(
                        select(
                            func.coalesce(
                                func.max(
                                    continuation_image_versions.c.version
                                ),
                                0,
                            )
                            + 1
                        ).where(
                            continuation_image_versions.c.continuation_page_id
                            == page_id
                        )
                    ).scalar_one()
                )
                connection.execute(
                    update(continuation_image_versions)
                    .where(
                        continuation_image_versions.c.continuation_page_id
                        == page_id,
                        continuation_image_versions.c.is_active.is_(True),
                    )
                    .values(is_active=False, updated_at=utcnow())
                )
                connection.execute(
                    insert(continuation_image_versions).values(
                        id=str(uuid.uuid4()),
                        continuation_page_id=page_id,
                        asset_id=asset.id,
                        thumbnail_asset_id=thumbnail_asset.id,
                        version=version,
                        is_active=True,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )
                checkpoint.update(
                    {
                        "continuationPageId": page_id,
                        "version": version,
                        "assetId": asset.id,
                        "thumbnailAssetId": thumbnail_asset.id,
                    }
                )
                return True
        elif kind == "continuation_export":
            output_format = config.get("format")
            raw_images = config.get("images")
            if output_format not in {"zip", "pdf"}:
                raise JobConflict("continuation export format is invalid")
            if not isinstance(raw_images, list) or not raw_images:
                raise JobConflict("continuation export image snapshot is invalid")
            images: list[dict[str, Any]] = []
            for value in raw_images:
                if not isinstance(value, Mapping) or set(value) != {
                    "ordinal",
                    "assetId",
                }:
                    raise JobConflict(
                        "continuation export image snapshot is invalid"
                    )
                try:
                    ordinal = _positive_integer(
                        value["ordinal"],
                        "continuation export ordinal",
                    )
                except ValueError as exc:
                    raise JobConflict(str(exc)) from exc
                asset_id = value["assetId"]
                if not isinstance(asset_id, str) or not asset_id:
                    raise JobConflict(
                        "continuation export assetId is invalid"
                    )
                images.append({"ordinal": ordinal, "assetId": asset_id})
            if len({image["ordinal"] for image in images}) != len(images):
                raise JobConflict(
                    "continuation export image ordinals must be unique"
                )
            output = self._build_export(
                images=images,
                output_format=output_format,
            )
            try:
                asset = self.storage.publish_stream(
                    output,
                    extension=output_format,
                    mime_type=(
                        "application/zip"
                        if output_format == "zip"
                        else "application/pdf"
                    ),
                )
            finally:
                output.close()
            checkpoint = {}

            def publish(connection: Connection) -> None:
                connection.execute(
                    insert(job_artifacts).values(
                        job_id=fence.job_id,
                        kind="continuation_export",
                        asset_id=asset.id,
                        expires_at=utcnow() + timedelta(hours=24),
                    )
                )
                checkpoint.update(
                    {
                        "assetId": asset.id,
                        "format": output_format,
                        "expiresInSeconds": 86400,
                    }
                )
        else:
            raise JobConflict(f"unsupported continuation step: {kind}")
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        if kind == "continuation_generate_script":
            log_result(
                "续写脚本生成结果",
                str(content).splitlines(),
            )
        elif kind == "continuation_generate_page":
            if checkpoint.get("skipped"):
                log_result(
                    "续写页面方案已跳过",
                    (f"原因：{checkpoint.get('reason', '页面状态已变化')}",),
                )
            elif payload is not None:
                log_result(
                    f"第 {target_ordinal} 页续写方案",
                    json_details(payload),
                )
        elif kind == "continuation_generate_character_sheet":
            log_result(
                "角色设定图生成完成",
                (
                    f"角色：{character_name}",
                    f"形态：{form_name}",
                    f"版本：{checkpoint.get('version')}",
                ),
            )
        elif kind == "continuation_generate_image":
            if checkpoint.get("skipped"):
                log_result(
                    "续写图片已跳过",
                    (f"原因：{checkpoint.get('reason', '页面状态已变化')}",),
                )
            else:
                log_result(
                    f"第 {target_ordinal} 页续写图片生成完成",
                    (f"版本：{checkpoint.get('version')}",),
                )
        elif kind == "continuation_export":
            log_result(
                "续写作品导出完成",
                (
                    f"格式：{output_format.upper()}",
                    f"图片：{len(images)} 张",
                ),
            )
        return {**checkpoint, "__already_published__": True}

    def _reference_window_asset_ids(
        self,
        *,
        project_id: str,
        before_ordinal: int,
        count: int,
        initial_asset_ids: Sequence[str],
    ) -> list[str]:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(continuation_image_versions.c.asset_id)
                    .join(
                        continuation_pages,
                        continuation_pages.c.id
                        == continuation_image_versions.c.continuation_page_id,
                    )
                    .where(
                        continuation_pages.c.project_id == project_id,
                        continuation_pages.c.ordinal < before_ordinal,
                        continuation_image_versions.c.is_active.is_(True),
                    )
                    .order_by(continuation_pages.c.ordinal.desc())
                    .limit(count)
                ).scalars()
            )
            if len(rows) < count:
                existing_ids = (
                    {
                        str(asset_id)
                        for asset_id in connection.execute(
                            select(assets.c.id).where(
                                assets.c.id.in_(tuple(initial_asset_ids))
                            )
                        ).scalars()
                    }
                    if initial_asset_ids
                    else set()
                )
                selected = [
                    asset_id
                    for asset_id in reversed(initial_asset_ids)
                    if asset_id in existing_ids
                ][: count - len(rows)]
                rows.extend(selected)
        return [str(asset_id) for asset_id in reversed(rows)]

    def _publish_generated_image(self, payload: bytes):
        image_info = _image_info(payload)
        thumbnail = _thumbnail_image(payload)
        asset = self.storage.publish_bytes(
            payload,
            extension=image_info["extension"],
            mime_type=image_info["mimeType"],
            width=image_info["width"],
            height=image_info["height"],
        )
        thumbnail_asset = self.storage.publish_bytes(
            thumbnail["bytes"],
            extension="webp",
            mime_type="image/webp",
            width=thumbnail["width"],
            height=thumbnail["height"],
        )
        return asset, thumbnail_asset

    def _build_export(
        self,
        *,
        images: Sequence[Mapping[str, Any]],
        output_format: str,
    ):
        with self.engine.connect() as connection:
            asset_ids = tuple(
                str(value["assetId"]) for value in images
            )
            paths = {
                str(asset_id): str(relative_path)
                for asset_id, relative_path in connection.execute(
                    select(assets.c.id, assets.c.relative_path).where(
                        assets.c.id.in_(asset_ids)
                    )
                )
            } if asset_ids else {}
            rows = list(
                (
                    int(value["ordinal"]),
                    paths.get(str(value["assetId"])),
                )
                for value in images
            )
        if not rows or any(relative_path is None for _, relative_path in rows):
            raise JobConflict("frozen continuation export image is missing")
        temporary = tempfile.TemporaryFile()
        if output_format == "zip":
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                for ordinal, relative_path in rows:
                    path = self.storage.resolve_relative_path(
                        str(relative_path)
                    )
                    archive.write(
                        path,
                        arcname=f"page_{int(ordinal):03d}{path.suffix}",
                    )
        else:
            from PIL import Image
            from reportlab.pdfgen import canvas

            document = canvas.Canvas(temporary)
            for _ordinal, relative_path in rows:
                path = self.storage.resolve_relative_path(
                    str(relative_path)
                )
                with Image.open(path) as source:
                    width, height = source.size
                document.setPageSize((width, height))
                document.drawImage(
                    str(path),
                    0,
                    0,
                    width=width,
                    height=height,
                    preserveAspectRatio=True,
                )
                document.showPage()
            document.save()
        temporary.seek(0)
        return temporary

    def _script_context(self, config: Mapping[str, Any]) -> dict[str, Any]:
        try:
            frozen_inputs = _validated_analysis_inputs(
                config.get("analysisInputs")
            )
            project_config = _validated_project_config(
                _mapping(config.get("projectConfig"))
            )
        except (TypeError, ValueError) as exc:
            raise JobConflict(
                "continuation analysis snapshot is invalid"
            ) from exc
        result_ids = [value["resultId"] for value in frozen_inputs]
        fingerprint = config.get("analysisInputFingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise JobConflict("continuation analysis snapshot is invalid")
        book_id = config.get("bookId")
        if not isinstance(book_id, str) or not book_id:
            raise JobConflict("continuation bookId is invalid")
        snapshot = self.derived.snapshot(
            book_id=book_id,
            frozen_inputs=frozen_inputs,
        )
        if (
            snapshot.fingerprint != fingerprint
            or list(snapshot.result_ids) != result_ids
        ):
            raise JobConflict("continuation analysis snapshot changed")
        with self.engine.connect() as connection:
            artifact_conditions = [
                analysis_artifacts.c.status.in_(("ready", "degraded", "stale")),
                analysis_artifacts.c.book_id == book_id,
                analysis_artifacts.c.dependency_fingerprint == fingerprint,
            ]
            artifacts = {
                f"{kind}:{template}": _mapping(_load(payload))
                for kind, template, payload in connection.execute(
                    select(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                        analysis_artifacts.c.payload_json,
                    ).where(*artifact_conditions)
                    .order_by(analysis_artifacts.c.revision)
                )
            }
        return {
            "direction": project_config["direction"],
            "pageCount": project_config["pageCount"],
            "pages": [page["analysis"] for page in snapshot.pages[-10:]],
            "artifacts": artifacts,
        }

    def _with_credentials(
        self,
        config: Mapping[str, Any],
        *,
        section_names: Sequence[str],
    ) -> dict[str, Any]:
        try:
            return self.credentials.resolve_credential_sections(
                config,
                section_names,
            )
        except LookupError as exc:
            raise JobConflict(
                "continuation credential version is missing"
            ) from exc


def _missing_prerequisites(
    *,
    analysis_ready: bool,
    prerequisites: set[tuple[str, str]],
    timeline: Mapping[str, Any] | None,
) -> list[str]:
    missing: list[str] = []
    if not analysis_ready:
        missing.append("analysis")
    if ("overview", "story_summary") not in prerequisites:
        missing.append("story_summary")
    if ("compressed_context", "default") not in prerequisites:
        missing.append("compressed_context")
    if timeline is None:
        missing.append("timeline")
    return missing


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("continuation JSON value must be an object")
    return dict(value)


def _continuation_job_target(
    config: Mapping[str, Any],
    step: Mapping[str, Any],
) -> dict[str, Any]:
    targets = config.get("targets")
    item_ordinal = step.get("itemOrdinal")
    if (
        not isinstance(targets, list)
        or not targets
        or isinstance(item_ordinal, bool)
        or not isinstance(item_ordinal, int)
        or not 1 <= item_ordinal <= len(targets)
    ):
        raise JobConflict("continuation job target snapshot is invalid")
    target = targets[item_ordinal - 1]
    if not isinstance(target, Mapping):
        raise JobConflict("continuation job target snapshot is invalid")
    return dict(target)


def _required_mapping_json(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} payload must be JSON text")
    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{label} payload must be a JSON object")
    return dict(loaded)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_analysis_inputs(value: object) -> list[dict[str, Any]]:
    required = {
        "resultId",
        "pageId",
        "pageNumber",
        "currentSourceChecksum",
    }
    if not isinstance(value, list) or not value:
        raise ValueError("analysisInputs must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("analysisInputs contains an invalid item")
        result_id = item["resultId"]
        page_id = item["pageId"]
        checksum = item["currentSourceChecksum"]
        if any(
            not isinstance(field, str) or not field
            for field in (result_id, page_id)
        ) or (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checksum
            )
        ):
            raise ValueError("analysisInputs contains an invalid item")
        normalized.append(
            {
                "resultId": result_id,
                "pageId": page_id,
                "pageNumber": _positive_integer(
                    item["pageNumber"],
                    "analysisInputs.pageNumber",
                ),
                "currentSourceChecksum": checksum,
            }
        )
    if any(
        len({item[field] for item in normalized}) != len(normalized)
        for field in ("resultId", "pageId", "pageNumber")
    ):
        raise ValueError("analysisInputs contains duplicate identities")
    return normalized


def _validated_project_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {"pageCount", "styleReferencePages", "direction"}
    if set(config) != required:
        raise ValueError(
            "continuation config requires exactly pageCount, "
            "styleReferencePages, and direction"
        )
    direction = config["direction"]
    if not isinstance(direction, str):
        raise ValueError("direction must be a string")
    return {
        "pageCount": _positive_integer(config["pageCount"], "pageCount"),
        "styleReferencePages": _positive_integer(
            config["styleReferencePages"],
            "styleReferencePages",
        ),
        "direction": direction,
    }


def _public_project_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _validated_project_config(
        {
            key: payload[key]
            for key in ("pageCount", "styleReferencePages", "direction")
            if key in payload
        }
    )


def _selected_ordinals(
    ordinals: Sequence[int] | None,
) -> list[int] | None:
    if ordinals is None:
        return None
    values = list(ordinals)
    if (
        not values
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        )
        or any(value < 1 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "ordinals must be a non-empty sequence of unique positive integers"
        )
    return sorted(values)


def _empty_page_payload() -> dict[str, Any]:
    return {
        "storyText": "",
        "continuityText": "",
        "dialogueText": "",
        "characters": [],
        "finalPrompt": "",
        "status": "pending",
    }


def _validated_page_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "storyText",
        "continuityText",
        "dialogueText",
        "characters",
        "finalPrompt",
        "status",
    }
    unknown = set(payload) - required - {"staleReason"}
    missing = required - set(payload)
    if unknown or missing:
        raise ValueError("continuation page payload has invalid fields")
    text_fields = (
        "storyText",
        "continuityText",
        "dialogueText",
        "finalPrompt",
    )
    if any(not isinstance(payload[key], str) for key in text_fields):
        raise ValueError("continuation page payload has invalid text fields")
    characters = payload["characters"]
    if not isinstance(characters, list) or any(
        not isinstance(value, str) for value in characters
    ):
        raise ValueError("continuation page payload has invalid characters")
    status = payload["status"]
    if not isinstance(status, str) or status not in {
        "pending",
        "generating",
        "ready",
        "stale",
        "failed",
    }:
        raise ValueError("continuation page payload has invalid status")
    stale_reason = payload.get("staleReason")
    if stale_reason is not None and (
        not isinstance(stale_reason, str) or not stale_reason
    ):
        raise ValueError("continuation page payload has invalid staleReason")
    return {
        **{key: payload[key] for key in text_fields},
        "characters": list(characters),
        "status": status,
        **({"staleReason": stale_reason} if stale_reason is not None else {}),
    }


def _validated_generated_page(result: Mapping[str, Any]) -> dict[str, Any]:
    text_fields = (
        "storyText",
        "continuityText",
        "dialogueText",
        "finalPrompt",
    )
    if any(not isinstance(result.get(key), str) for key in text_fields):
        raise ValueError("continuation page response has invalid text fields")
    if not result["storyText"].strip() or not result["finalPrompt"].strip():
        raise ValueError(
            "continuation page response requires storyText and finalPrompt"
        )
    characters = result.get("characters")
    if not isinstance(characters, list) or any(
        not isinstance(value, str) for value in characters
    ):
        raise ValueError("continuation page response has invalid characters")
    return _validated_page_payload({
        **{key: result[key] for key in text_fields},
        "characters": list(characters),
        "status": "ready",
    })


def _snapshot_inputs(snapshot: AnalysisInputSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "resultId": page["resultId"],
            "pageId": page["pageId"],
            "pageNumber": page["pageNumber"],
            "currentSourceChecksum": page["currentSourceChecksum"],
        }
        for page in snapshot.pages
    ]


def _page_dto(
    row: Mapping[str, Any],
    *,
    image_versions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    try:
        payload = _validated_page_payload(
            _required_mapping_json(
                row["payload_json"],
                "continuation page",
            )
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise InsightConflict(
            "continuation page data is invalid; clear the project"
        ) from exc
    return {
        "continuationPageId": str(row["id"]),
        "ordinal": int(row["ordinal"]),
        "revision": int(row["revision"]),
        "payload": payload,
        "imageVersions": [
            {
                "version": int(version["version"]),
                "assetId": str(version["asset_id"]),
                "assetUrl": f"/api/v2/assets/{version['asset_id']}",
                "thumbnailUrl": (
                    f"/api/v2/assets/{version['thumbnail_asset_id']}"
                ),
                "active": bool(version["is_active"]),
            }
            for version in image_versions
        ],
    }


def _character_dto(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        aliases = _load(row["aliases_json"])
        if not isinstance(aliases, list) or any(
            not isinstance(value, str) for value in aliases
        ):
            raise ValueError("continuation character aliases are invalid")
        payload = _required_mapping_json(
            row["payload_json"],
            "continuation character",
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise InsightConflict(
            "continuation character data is invalid; clear the project"
        ) from exc
    return {
        "characterId": str(row["id"]),
        "projectId": str(row["project_id"]),
        "name": str(row["name"]),
        "aliases": aliases,
        "enabled": bool(row["enabled"]),
        "payload": payload,
        "revision": int(row["revision"]),
    }


def _normalize_character_identity(
    name: str,
    aliases: Sequence[str],
) -> tuple[str, list[str]]:
    normalized_name = name.strip()
    if not normalized_name or len(normalized_name) > 500:
        raise ValueError("character name must contain 1-500 characters")
    if any(not isinstance(value, str) for value in aliases):
        raise ValueError("character aliases must be strings")
    normalized_aliases = [value.strip() for value in aliases]
    if any(not value for value in normalized_aliases):
        raise ValueError("character aliases must not be blank")
    if len(set(normalized_aliases)) != len(normalized_aliases):
        raise ValueError("character aliases must be unique")
    return normalized_name, normalized_aliases


def _form_name(value: str) -> str:
    result = value.strip()
    if not result or len(result) > 500:
        raise ValueError("form name must contain 1-500 characters")
    return result


def _image_info(payload: bytes) -> dict[str, Any]:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        image_format = str(image.format or "PNG").upper()
        width, height = image.size
    media_type = {
        "JPEG": ("jpg", "image/jpeg"),
        "WEBP": ("webp", "image/webp"),
        "PNG": ("png", "image/png"),
    }.get(image_format)
    if media_type is None:
        raise ValueError(
            f"image provider returned unsupported {image_format} data"
        )
    extension, mime_type = media_type
    return {
        "extension": extension,
        "mimeType": mime_type,
        "width": width,
        "height": height,
    }


def _thumbnail_image(payload: bytes) -> dict[str, Any]:
    from PIL import Image, ImageOps

    with Image.open(BytesIO(payload)) as source:
        oriented = ImageOps.exif_transpose(source)
        oriented.load()
        thumbnail = oriented.copy()
        thumbnail.thumbnail((320, 320), Image.Resampling.LANCZOS)
        if thumbnail.mode not in {"RGB", "RGBA"}:
            thumbnail = thumbnail.convert("RGBA")
        output = BytesIO()
        thumbnail.save(output, format="WEBP", quality=80, method=4)
        width, height = thumbnail.size
        thumbnail.close()
        if oriented is not source:
            oriented.close()
    return {
        "bytes": output.getvalue(),
        "width": width,
        "height": height,
    }
