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

from sqlalchemy import Engine, delete, exists, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
    utcnow,
)
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
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_page_results,
    analysis_runs,
    assets,
    continuation_character_forms,
    continuation_characters,
    continuation_form_image_versions,
    continuation_image_versions,
    continuation_pages,
    continuation_project_reference_assets,
    continuation_projects,
    continuation_scripts,
    credential_versions,
    chapters,
    jobs,
    job_artifacts,
    job_asset_inputs,
    page_assets,
    pages,
    timeline_characters,
    timeline_versions,
)


NONTERMINAL_JOB_STATUSES = (
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "interrupted",
)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


class ContinuationRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def bootstrap(self, *, book_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            active_run = connection.execute(
                select(
                    analysis_heads.c.active_run_id,
                    analysis_runs.c.status,
                )
                .join(
                    analysis_runs,
                    analysis_runs.c.id == analysis_heads.c.active_run_id,
                )
                .where(
                    analysis_heads.c.book_id == book_id,
                    analysis_heads.c.page_id.is_(None),
                )
            ).mappings().one_or_none()
            prerequisites = set(
                connection.execute(
                    select(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                    ).where(
                        analysis_artifacts.c.book_id == book_id,
                        analysis_artifacts.c.is_active.is_(True),
                        analysis_artifacts.c.status.in_(("ready", "degraded")),
                    )
                )
            )
            timeline = connection.execute(
                select(
                    timeline_versions.c.id,
                    timeline_versions.c.status,
                    timeline_versions.c.run_id,
                ).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
            project = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.book_id == book_id
                )
            ).mappings().one_or_none()
            if project is None:
                return {
                    "bookId": book_id,
                    "ready": bool(
                        active_run
                        and ("overview", "story_summary") in prerequisites
                        and ("compressed_context", "default") in prerequisites
                        and timeline
                    ),
                    "activeRunId": (
                        str(active_run["active_run_id"])
                        if active_run
                        else None
                    ),
                    "missing": _missing_prerequisites(
                        active_run=active_run,
                        prerequisites=prerequisites,
                        timeline=timeline,
                    ),
                    "project": None,
                }
            return {
                "bookId": book_id,
                "ready": True,
                "activeRunId": (
                    str(active_run["active_run_id"])
                    if active_run
                    else None
                ),
                "missing": _missing_prerequisites(
                    active_run=active_run,
                    prerequisites=prerequisites,
                    timeline=timeline,
                ),
                "project": self._project_dto(connection, project),
            }

    def sync_latest(self, *, book_id: str) -> dict[str, Any]:
        state = self.bootstrap(book_id=book_id)
        if state["missing"]:
            raise InsightConflict(
                "continuation prerequisites are missing: "
                + ", ".join(state["missing"])
            )
        run_id = str(state["activeRunId"])
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
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
                        book_id=book_id,
                        source_run_id=run_id,
                        revision=1,
                        payload_json=_json(
                            {
                                "pageCount": 15,
                                "styleReferencePages": 3,
                                "direction": "",
                            }
                        ),
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                project_id = str(project["id"])
                if str(project["source_run_id"] or "") != run_id:
                    connection.execute(
                        update(continuation_projects)
                        .where(continuation_projects.c.id == project_id)
                        .values(
                            source_run_id=run_id,
                            revision=int(project["revision"]) + 1,
                            updated_at=now,
                        )
                    )
            timeline_id = connection.execute(
                select(timeline_versions.c.id).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).scalar_one()
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
                loaded = _load(payload, {})
                connection.execute(
                    insert(continuation_characters).values(
                        id=str(uuid.uuid4()),
                        project_id=project_id,
                        name=str(name),
                        aliases_json=_json(
                            loaded.get("aliases", [])
                            if isinstance(loaded, Mapping)
                            else []
                        ),
                        enabled=True,
                        payload_json=_json(
                            dict(loaded)
                            if isinstance(loaded, Mapping)
                            else {}
                        ),
                    )
                )
            project = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.id == project_id
                )
            ).mappings().one()
            return self._project_dto(connection, project)

    def update_project(
        self,
        *,
        project_id: str,
        base_revision: int,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = {
            "pageCount": max(1, min(100, int(config.get("pageCount", 15)))),
            "styleReferencePages": max(
                1,
                min(20, int(config.get("styleReferencePages", 3))),
            ),
            "direction": str(config.get("direction", "")),
        }
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
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
            row = connection.execute(
                select(continuation_projects).where(
                    continuation_projects.c.id == project_id
                )
            ).mappings().one()
            return self._project_dto(connection, row)

    def set_project_references(
        self,
        *,
        project_id: str,
        base_revision: int,
        asset_ids: Sequence[str],
    ) -> dict[str, Any]:
        normalized = list(dict.fromkeys(str(value) for value in asset_ids))
        if len(normalized) != len(asset_ids) or len(normalized) > 20:
            raise ValueError(
                "reference assetIds must be unique and contain at most 20 items"
            )
        with immediate_transaction(self.engine) as connection:
            if normalized:
                existing = set(
                    str(value)
                    for value in connection.execute(
                        select(assets.c.id).where(
                            assets.c.id.in_(tuple(normalized))
                        )
                    ).scalars()
                )
                if existing != set(normalized):
                    raise InsightNotFound("reference asset not found")
            changed = connection.execute(
                update(continuation_projects)
                .where(
                    continuation_projects.c.id == project_id,
                    continuation_projects.c.revision == base_revision,
                )
                .values(
                    revision=base_revision + 1,
                    updated_at=utcnow(),
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
            return self._project_dto(connection, row)

    def update_script(
        self,
        *,
        project_id: str,
        base_revision: int,
        content: str,
    ) -> dict[str, Any]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(continuation_scripts).where(
                    continuation_scripts.c.project_id == project_id
                )
            ).mappings().one_or_none()
            if row is None:
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
                    )
                )
            )
        return {
            "scriptId": script_id,
            "projectId": project_id,
            "revision": revision,
            "content": content,
        }

    def update_page(
        self,
        *,
        page_id: str,
        base_revision: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(continuation_pages)
                .where(
                    continuation_pages.c.id == page_id,
                    continuation_pages.c.revision == base_revision,
                )
                .values(
                    payload_json=_json(dict(payload)),
                    revision=base_revision + 1,
                )
            )
            if changed.rowcount != 1:
                raise InsightConflict("continuation page revision changed")
            row = connection.execute(
                select(continuation_pages).where(
                    continuation_pages.c.id == page_id
                )
            ).mappings().one()
        return _page_dto(row)

    def switch_image_version(
        self,
        *,
        continuation_page_id: str,
        version: int,
    ) -> dict[str, Any]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
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
        return {
            "continuationPageId": continuation_page_id,
            "version": version,
            "assetId": str(target["asset_id"]),
        }

    def create_character(
        self,
        *,
        project_id: str,
        name: str,
        aliases: Sequence[str],
        enabled: bool,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        name, aliases = _normalize_character_identity(name, aliases)
        character_id = str(uuid.uuid4())
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            self._require_project(connection, project_id)
            connection.execute(
                insert(continuation_characters).values(
                    id=character_id,
                    project_id=project_id,
                    name=name,
                    aliases_json=_json(aliases),
                    enabled=enabled,
                    payload_json=_json(dict(payload)),
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
        return _character_dto(row)

    def update_character(
        self,
        *,
        character_id: str,
        base_revision: int,
        name: str,
        aliases: Sequence[str],
        enabled: bool,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        name, aliases = _normalize_character_identity(name, aliases)
        with immediate_transaction(self.engine) as connection:
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
                    payload_json=_json(dict(payload)),
                    revision=base_revision + 1,
                    updated_at=utcnow(),
                )
            )
            if changed.rowcount != 1:
                self._raise_character_cas(connection, character_id)
            row = connection.execute(
                select(continuation_characters).where(
                    continuation_characters.c.id == character_id
                )
            ).mappings().one()
        return _character_dto(row)

    def delete_character(
        self,
        *,
        character_id: str,
        base_revision: int,
    ) -> None:
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                delete(continuation_characters).where(
                    continuation_characters.c.id == character_id,
                    continuation_characters.c.revision == base_revision,
                )
            )
            if changed.rowcount != 1:
                self._raise_character_cas(connection, character_id)

    def create_form(
        self,
        *,
        character_id: str,
        name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        name = _form_name(name)
        form_id = str(uuid.uuid4())
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            if connection.execute(
                select(continuation_characters.c.id).where(
                    continuation_characters.c.id == character_id
                )
            ).scalar_one_or_none() is None:
                raise InsightNotFound("continuation character not found")
            connection.execute(
                insert(continuation_character_forms).values(
                    id=form_id,
                    character_id=character_id,
                    name=name,
                    payload_json=_json(dict(payload)),
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
        return self._form_dto(row, image_versions=[])

    def update_form(
        self,
        *,
        form_id: str,
        base_revision: int,
        name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        name = _form_name(name)
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(continuation_character_forms)
                .where(
                    continuation_character_forms.c.id == form_id,
                    continuation_character_forms.c.revision
                    == base_revision,
                )
                .values(
                    name=name,
                    payload_json=_json(dict(payload)),
                    revision=base_revision + 1,
                    updated_at=utcnow(),
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
        return self._form_dto(row, image_versions=versions)

    def delete_form(
        self,
        *,
        form_id: str,
        base_revision: int,
    ) -> None:
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                delete(continuation_character_forms).where(
                    continuation_character_forms.c.id == form_id,
                    continuation_character_forms.c.revision
                    == base_revision,
                )
            )
            if changed.rowcount != 1:
                self._raise_form_cas(connection, form_id)

    def bind_form_reference(
        self,
        *,
        form_id: str,
        base_revision: int,
        asset_id: str | None,
        thumbnail_asset_id: str | None,
    ) -> dict[str, Any]:
        if (asset_id is None) != (thumbnail_asset_id is None):
            raise ValueError(
                "reference asset and thumbnail must be set together"
            )
        with immediate_transaction(self.engine) as connection:
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
                    updated_at=utcnow(),
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
        return self._form_dto(row, image_versions=versions)

    def list_forms(
        self,
        *,
        project_id: str,
        cursor: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        if cursor < 0 or not 1 <= limit <= 200:
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
            items = [
                self._form_dto(
                    row,
                    image_versions=self._form_versions(
                        connection,
                        str(row["id"]),
                    ),
                )
                for row in selected_rows
            ]
        return {
            "items": items,
            "nextCursor": cursor + limit if has_more else None,
        }

    def adopt_form_image(
        self,
        *,
        form_id: str,
        version: int,
        base_revision: int,
    ) -> dict[str, Any]:
        with immediate_transaction(self.engine) as connection:
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
                    updated_at=utcnow(),
                )
            )
            if changed.rowcount != 1:
                self._raise_form_cas(connection, form_id)
            connection.execute(
                update(continuation_form_image_versions)
                .where(
                    continuation_form_image_versions.c.form_id == form_id
                )
                .values(is_adopted=False, updated_at=utcnow())
            )
            connection.execute(
                update(continuation_form_image_versions)
                .where(
                    continuation_form_image_versions.c.id == target["id"]
                )
                .values(is_adopted=True, updated_at=utcnow())
            )
        return {
            "formId": form_id,
            "version": version,
            "assetId": str(target["asset_id"]),
            "revision": base_revision + 1,
        }

    def clear(self, *, book_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            project_id = connection.execute(
                select(continuation_projects.c.id).where(
                    continuation_projects.c.book_id == book_id
                )
            ).scalar_one_or_none()
            if project_id is None:
                return
            active_job = connection.execute(
                select(jobs.c.id).where(
                    jobs.c.continuation_project_id == project_id,
                    jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                )
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
                .limit(5)
            ).mappings()
        )

    @staticmethod
    def _form_dto(
        row: Mapping[str, Any],
        *,
        image_versions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "formId": str(row["id"]),
            "characterId": str(row["character_id"]),
            "name": str(row["name"]),
            "revision": int(row["revision"]),
            "payload": _load(row["payload_json"], {}),
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
        return {
            "projectId": str(row["id"]),
            "bookId": str(row["book_id"]),
            "sourceRunId": row["source_run_id"],
            "revision": int(row["revision"]),
            "config": _load(row["payload_json"], {}),
            "script": (
                {
                    "scriptId": str(script["id"]),
                    "revision": int(script["revision"]),
                    "content": str(script["content"]),
                }
                if script
                else None
            ),
            "pages": [
                _page_dto(
                    page,
                    image_versions=list(
                        connection.execute(
                            select(continuation_image_versions)
                            .where(
                                continuation_image_versions.c.continuation_page_id
                                == page["id"]
                            )
                            .order_by(
                                continuation_image_versions.c.version.desc()
                            )
                            .limit(5)
                        ).mappings()
                    ),
                )
                for page in pages_rows
            ],
            "referenceAssets": [
                {
                    "assetId": str(asset_id),
                    "assetUrl": f"/api/v2/assets/{asset_id}",
                    "thumbnailUrl": (
                        f"/api/v2/assets/"
                        f"{_reference_thumbnail_asset_id(connection, str(asset_id))}"
                    ),
                }
                for asset_id in references
            ],
            "characters": [
                _character_dto(character)
                for character in characters
            ],
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
        return base_revision + 1

    @staticmethod
    def insert_page_result(
        connection: Connection,
        *,
        page_id: str,
        base_revision: int,
        payload: Mapping[str, Any],
    ) -> int:
        changed = connection.execute(
            update(continuation_pages)
            .where(
                continuation_pages.c.id == page_id,
                continuation_pages.c.revision == base_revision,
            )
            .values(
                revision=base_revision + 1,
                payload_json=_json(dict(payload)),
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

    def create_script_job(
        self,
        *,
        book_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        project = self.repository.project_by_book(book_id)
        with self.engine.connect() as connection:
            script = connection.execute(
                select(continuation_scripts).where(
                    continuation_scripts.c.project_id == project["id"]
                )
            ).mappings().one_or_none()
        config = self._config(book_id, project)
        config.update(
            {
                "continuationAction": "script",
                "projectId": str(project["id"]),
                "baseScriptRevision": (
                    int(script["revision"]) if script else 0
                ),
            }
        )
        return self.jobs.create_batch(
            kind="continuation",
            display_name="续写 · 生成脚本",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=str(project["source_run_id"]),
                    continuation_project_id=str(project["id"]),
                    config=config,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("continuation_generate_script",),
                        ),
                    ),
                ),
            ),
            idempotency_scope=f"continuation-script:{project['id']}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "projectId": str(project["id"]),
                "projectRevision": int(project["revision"]),
                "baseScriptRevision": config["baseScriptRevision"],
            },
        )

    def create_pages_job(
        self,
        *,
        book_id: str,
        ordinals: Sequence[int] | None,
        idempotency_key: str,
    ) -> dict[str, object]:
        project = self.repository.project_by_book(book_id)
        page_count = int(_load(project["payload_json"], {}).get("pageCount", 15))
        selected = (
            sorted(set(int(value) for value in ordinals))
            if ordinals
            else list(range(1, page_count + 1))
        )
        if not selected or selected[0] < 1 or selected[-1] > page_count:
            raise ValueError("continuation page ordinal is out of range")
        with immediate_transaction(self.engine) as connection:
            script = connection.execute(
                select(continuation_scripts).where(
                    continuation_scripts.c.project_id == project["id"]
                )
            ).mappings().one_or_none()
            if script is None or not str(script["content"]).strip():
                raise InsightConflict("generate or save a script first")
            existing = {
                int(row["ordinal"]): row
                for row in connection.execute(
                    select(continuation_pages).where(
                        continuation_pages.c.project_id == project["id"],
                        continuation_pages.c.ordinal.in_(tuple(selected)),
                    )
                ).mappings()
            }
            now = utcnow()
            targets = []
            for ordinal in selected:
                row = existing.get(ordinal)
                if row is None:
                    page_id = str(uuid.uuid4())
                    connection.execute(
                        insert(continuation_pages).values(
                            id=page_id,
                            project_id=project["id"],
                            ordinal=ordinal,
                            revision=1,
                            payload_json="{}",
                        )
                    )
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
        config = self._config(book_id, project)
        config.update(
            {
                "continuationAction": "pages",
                "projectId": str(project["id"]),
                "script": str(script["content"]),
                "targets": targets,
            }
        )
        return self.jobs.create_batch(
            kind="continuation",
            display_name="续写 · 页面剧情",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=str(project["source_run_id"]),
                    continuation_project_id=str(project["id"]),
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
            idempotency_scope=f"continuation-pages:{project['id']}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "projectId": str(project["id"]),
                "targets": targets,
                "scriptRevision": int(script["revision"]),
            },
        )

    def create_images_job(
        self,
        *,
        book_id: str,
        ordinals: Sequence[int] | None,
        idempotency_key: str,
    ) -> dict[str, object]:
        project = self.repository.project_by_book(book_id)
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(continuation_pages)
                    .where(
                        continuation_pages.c.project_id == project["id"]
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
                        == project["id"]
                    )
                    .order_by(
                        continuation_project_reference_assets.c.ordinal
                    )
                ).scalars()
            ]
            reference_count = int(
                _load(project["payload_json"], {}).get(
                    "styleReferencePages",
                    3,
                )
            )
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
        selected_set = (
            set(int(value) for value in ordinals) if ordinals else None
        )
        targets = [
            {
                "pageId": str(row["id"]),
                "ordinal": int(row["ordinal"]),
                "baseRevision": int(row["revision"]),
                "payload": _load(row["payload_json"], {}),
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
                "projectId": str(project["id"]),
                "targets": targets,
                "initialReferenceAssetIds": initial_reference_ids,
            }
        )
        frozen_references = {
            f"style_reference_{index}": asset_id
            for index, asset_id in enumerate(initial_reference_ids, start=1)
        }
        return self.jobs.create_batch(
            kind="continuation",
            display_name="续写 · 批量生图",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=str(project["source_run_id"]),
                    continuation_project_id=str(project["id"]),
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
            idempotency_scope=f"continuation-images:{project['id']}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "projectId": str(project["id"]),
                "targets": [
                    {
                        "pageId": target["pageId"],
                        "revision": target["baseRevision"],
                    }
                    for target in targets
                ],
            },
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
        project = self.repository.project_by_book(book_id)
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
                        continuation_pages.c.project_id == project["id"],
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
                "projectId": str(project["id"]),
                "format": output_format,
                "images": images,
            }
        )
        return self.jobs.create_batch(
            kind="continuation",
            display_name=f"续写 · 导出 {output_format.upper()}",
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=str(project["source_run_id"]),
                    continuation_project_id=str(project["id"]),
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
            idempotency_scope=(
                f"continuation-export:{project['id']}:{output_format}"
            ),
            idempotency_key=idempotency_key,
            idempotency_payload={
                "projectId": str(project["id"]),
                "format": output_format,
                "projectRevision": int(project["revision"]),
                "images": images,
            },
        )

    def create_character_sheet_job(
        self,
        *,
        book_id: str,
        form_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    continuation_character_forms,
                    continuation_characters.c.name.label("character_name"),
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
        config = self._config(book_id, project)
        config.update(
            {
                "continuationAction": "character_sheet",
                "projectId": str(project["id"]),
                "formId": form_id,
                "baseFormRevision": int(row["revision"]),
                "characterName": str(row["character_name"]),
                "formName": str(row["name"]),
                "formPayload": _load(row["payload_json"], {}),
                "referenceAssetId": row["reference_asset_id"],
            }
        )
        asset_inputs = (
            {"character_reference": str(row["reference_asset_id"])}
            if row["reference_asset_id"]
            else None
        )
        return self.jobs.create_batch(
            kind="continuation",
            display_name=(
                f"续写 · {row['character_name']} · {row['name']} 三视图"
            ),
            specs=(
                JobSpec(
                    kind="continuation",
                    book_id=book_id,
                    analysis_run_id=str(project["source_run_id"]),
                    continuation_project_id=str(project["id"]),
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
            idempotency_scope=(
                f"continuation-character-sheet:{form_id}:"
                f"{int(row['revision'])}"
            ),
            idempotency_key=idempotency_key,
            idempotency_payload={
                "formId": form_id,
                "baseRevision": int(row["revision"]),
                "referenceAssetId": row["reference_asset_id"],
            },
        )

    def _config(
        self,
        book_id: str,
        project: Mapping[str, Any],
    ) -> dict[str, Any]:
        config = self.settings.resolve_insight(
            book_id=book_id,
            command={"scope": "full", "force": False},
        )
        config.update(
            {
                "bookId": book_id,
                "sourceRunId": project["source_run_id"],
                "projectRevision": int(project["revision"]),
                "projectConfig": _load(project["payload_json"], {}),
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
        from src.backend_v2.insight.derived import LegacyDerivedAlgorithms

        prompt = (
            "根据已发布漫画分析继续创作一话漫画脚本。"
            "严格遵守指定页数和方向，输出可供逐页拆解的脚本。\n\n"
            + _json(context)
        )
        result = LegacyDerivedAlgorithms._chat_json(
            prompt,
            config=config,
            prompt_type="book_overview",
        )
        if isinstance(result, Mapping):
            return str(
                result.get("script")
                or result.get("content")
                or _json(result)
            )
        return str(result)

    def generate_page(
        self,
        *,
        ordinal: int,
        script: str,
        previous: Mapping[str, Any] | None,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from src.backend_v2.insight.derived import LegacyDerivedAlgorithms

        prompt = (
            f"把续写脚本拆成第 {ordinal} 页。输出 JSON："
            '{"storyText":"...","continuityText":"...",'
            '"dialogueText":"...","characters":[],"characterForms":[],'
            '"finalPrompt":"..."}。\n\n'
            f"上一页：{_json(previous or {})}\n\n脚本：{script}"
        )
        result = LegacyDerivedAlgorithms._chat_json(
            prompt,
            config=config,
            prompt_type="group_summary",
        )
        if not isinstance(result, Mapping):
            raise ValueError("continuation page response is not JSON")
        return {
            "storyText": str(
                result.get("storyText", result.get("story_text", ""))
            ),
            "continuityText": str(
                result.get(
                    "continuityText",
                    result.get("continuity_text", ""),
                )
            ),
            "dialogueText": str(
                result.get(
                    "dialogueText",
                    result.get("dialogue_text", ""),
                )
            ),
            "characters": list(result.get("characters", [])),
            "characterForms": list(
                result.get(
                    "characterForms",
                    result.get("character_forms", []),
                )
            ),
            "finalPrompt": str(
                result.get(
                    "finalPrompt",
                    result.get("final_prompt", ""),
                )
            ),
            "status": "ready",
        }

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
        from src.core.manga_insight.config_models import ImageGenConfig

        section = (
            dict(config.get("imageGen"))
            if isinstance(config.get("imageGen"), Mapping)
            else {}
        )
        payload = {
            "provider": section.get("provider", ""),
            "api_key": section.get("api_key", section.get("apiKey", "")),
            "model": section.get(
                "model_name",
                section.get("modelName", ""),
            ),
            "base_url": section.get(
                "custom_base_url",
                section.get("base_url"),
            ),
        }
        client = ImageGenClient(ImageGenConfig.from_dict(payload))

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
        self.data_root = data_root
        self.engine = engine
        self.jobs = jobs
        self.repository = ContinuationRepository(engine)
        self.storage = AssetStorageService(data_root, engine)
        self.algorithms = algorithms or DefaultContinuationAlgorithms()

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._with_credentials(
            dict(step.get("config"))
            if isinstance(step.get("config"), Mapping)
            else {}
        )
        kind = str(step["stepKind"])
        if kind == "continuation_generate_script":
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
                                project_id=str(config["projectId"]),
                                base_revision=int(
                                    config["baseScriptRevision"]
                                ),
                                content=content,
                            )
                        ),
                    }
                )

        elif kind == "continuation_generate_page":
            targets = config.get("targets", [])
            target = targets[int(step["itemOrdinal"]) - 1]
            page_id = str(target["pageId"])
            with self.engine.connect() as connection:
                current = connection.execute(
                    select(continuation_pages).where(
                        continuation_pages.c.id == page_id
                    )
                ).mappings().one()
                existing = _load(current["payload_json"], {})
                previous = connection.execute(
                    select(continuation_pages.c.payload_json).where(
                        continuation_pages.c.project_id
                        == current["project_id"],
                        continuation_pages.c.ordinal
                        == int(target["ordinal"]) - 1,
                    )
                ).scalar_one_or_none()
            if (
                existing.get("status") == "ready"
                and existing.get("storyText")
                and not existing.get("staleReason")
            ):
                payload = existing
                skipped = True
            else:
                payload = dict(
                    self.algorithms.generate_page(
                        ordinal=int(target["ordinal"]),
                        script=str(config["script"]),
                        previous=(
                            _load(previous, {}) if previous else None
                        ),
                        config=config,
                    )
                )
                skipped = False
            checkpoint = {}

            def publish(connection: Connection) -> None:
                if skipped:
                    checkpoint.update(
                        {
                            "continuationPageId": page_id,
                            "ordinal": int(target["ordinal"]),
                            "skipped": True,
                        }
                    )
                    return
                current_revision = connection.execute(
                    select(continuation_pages.c.revision).where(
                        continuation_pages.c.id == page_id
                    )
                ).scalar_one_or_none()
                if current_revision != int(target["baseRevision"]):
                    checkpoint.update(
                        {
                            "continuationPageId": page_id,
                            "ordinal": int(target["ordinal"]),
                            "skipped": True,
                            "skipReason": "revision_conflict",
                        }
                    )
                    return
                revision = ContinuationRepository.insert_page_result(
                    connection,
                    page_id=page_id,
                    base_revision=int(target["baseRevision"]),
                    payload=payload,
                )
                checkpoint.update(
                    {
                        "continuationPageId": page_id,
                        "ordinal": int(target["ordinal"]),
                        "pageRevision": revision,
                        "skipped": False,
                    }
                )
        elif kind == "continuation_generate_character_sheet":
            reference_paths: list[Path] = []
            reference_asset_id = config.get("referenceAssetId")
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
                f"角色：{config.get('characterName', '')}\n"
                f"形态：{config.get('formName', '')}\n"
                f"设定：{_json(config.get('formPayload', {}))}"
            )
            image_bytes = self.algorithms.generate_image(
                prompt=prompt,
                reference_paths=reference_paths,
                config=config,
            )
            image_info = _image_info(image_bytes)
            thumbnail = _thumbnail_image(image_bytes)
            asset = self.storage.publish_bytes(
                image_bytes,
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
            checkpoint = {}

            def publish(connection: Connection) -> None:
                form_id = str(config["formId"])
                base_revision = int(config["baseFormRevision"])
                changed = connection.execute(
                    update(continuation_character_forms)
                    .where(
                        continuation_character_forms.c.id == form_id,
                        continuation_character_forms.c.revision
                        == base_revision,
                    )
                    .values(
                        revision=base_revision + 1,
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
                obsolete = list(
                    connection.execute(
                        select(continuation_form_image_versions.c.id)
                        .where(
                            continuation_form_image_versions.c.form_id
                            == form_id,
                            continuation_form_image_versions.c.is_adopted.is_(
                                False
                            ),
                            ~exists(
                                select(job_asset_inputs.c.asset_id).where(
                                    job_asset_inputs.c.asset_id
                                    == continuation_form_image_versions.c.asset_id
                                )
                            ),
                            ~exists(
                                select(job_artifacts.c.asset_id).where(
                                    job_artifacts.c.asset_id
                                    == continuation_form_image_versions.c.asset_id
                                )
                            ),
                            ~exists(
                                select(
                                    continuation_project_reference_assets.c.asset_id
                                ).where(
                                    continuation_project_reference_assets.c.asset_id
                                    == continuation_form_image_versions.c.asset_id
                                )
                            ),
                        )
                        .order_by(
                            continuation_form_image_versions.c.version.desc()
                        )
                        .offset(5)
                    ).scalars()
                )
                if obsolete:
                    connection.execute(
                        delete(continuation_form_image_versions).where(
                            continuation_form_image_versions.c.id.in_(
                                tuple(obsolete)
                            )
                        )
                    )
                checkpoint.update(
                    {
                        "formId": form_id,
                        "formRevision": base_revision + 1,
                        "version": version,
                        "assetId": asset.id,
                        "thumbnailAssetId": thumbnail_asset.id,
                    }
                )
        elif kind == "continuation_generate_image":
            targets = config.get("targets", [])
            target = targets[int(step["itemOrdinal"]) - 1]
            page_id = str(target["pageId"])
            reference_paths = self._reference_window(
                project_id=str(config["projectId"]),
                before_ordinal=int(target["ordinal"]),
                count=int(
                    dict(config.get("projectConfig", {})).get(
                        "styleReferencePages",
                        3,
                    )
                ),
                initial_asset_ids=[
                    str(value)
                    for value in config.get(
                        "initialReferenceAssetIds",
                        [],
                    )
                ],
            )
            image_bytes = self.algorithms.generate_image(
                prompt=str(target["payload"]["finalPrompt"]),
                reference_paths=reference_paths,
                config=config,
            )
            image_info = _image_info(image_bytes)
            thumbnail = _thumbnail_image(image_bytes)
            asset = self.storage.publish_bytes(
                image_bytes,
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
            checkpoint = {}

            def publish(connection: Connection) -> None:
                current_revision = connection.execute(
                    select(continuation_pages.c.revision).where(
                        continuation_pages.c.id == page_id
                    )
                ).scalar_one_or_none()
                if current_revision != int(target["baseRevision"]):
                    raise JobConflict(
                        "continuation page prompt changed before image publish"
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
                obsolete_ids = list(
                    connection.execute(
                        select(continuation_image_versions.c.id)
                        .where(
                            continuation_image_versions.c.continuation_page_id
                            == page_id,
                            ~exists(
                                select(job_asset_inputs.c.asset_id).where(
                                    job_asset_inputs.c.asset_id
                                    == continuation_image_versions.c.asset_id
                                )
                            ),
                            ~exists(
                                select(job_artifacts.c.asset_id).where(
                                    job_artifacts.c.asset_id
                                    == continuation_image_versions.c.asset_id
                                )
                            ),
                            ~exists(
                                select(
                                    continuation_project_reference_assets.c.asset_id
                                ).where(
                                    continuation_project_reference_assets.c.asset_id
                                    == continuation_image_versions.c.asset_id
                                )
                            ),
                            ~exists(
                                select(
                                    continuation_character_forms.c.reference_asset_id
                                ).where(
                                    (
                                        continuation_character_forms.c.reference_asset_id
                                        == continuation_image_versions.c.asset_id
                                    )
                                    | (
                                        continuation_character_forms.c.adopted_asset_id
                                        == continuation_image_versions.c.asset_id
                                    )
                                )
                            ),
                        )
                        .order_by(
                            continuation_image_versions.c.version.desc()
                        )
                        .offset(5)
                    ).scalars()
                )
                if obsolete_ids:
                    connection.execute(
                        delete(continuation_image_versions).where(
                            continuation_image_versions.c.id.in_(
                                tuple(obsolete_ids)
                            )
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
        elif kind == "continuation_export":
            output_format = str(config.get("format", "zip"))
            output = self._build_export(
                images=(
                    config.get("images", [])
                    if isinstance(config.get("images"), list)
                    else []
                ),
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
        return {**checkpoint, "__already_published__": True}

    def _reference_window(
        self,
        *,
        project_id: str,
        before_ordinal: int,
        count: int,
        initial_asset_ids: Sequence[str],
    ) -> list[Path]:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(assets.c.relative_path)
                    .join(
                        continuation_image_versions,
                        continuation_image_versions.c.asset_id == assets.c.id,
                    )
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
                path_by_id = {
                    str(asset_id): str(relative_path)
                    for asset_id, relative_path in connection.execute(
                        select(
                            assets.c.id,
                            assets.c.relative_path,
                        ).where(
                            assets.c.id.in_(tuple(initial_asset_ids))
                        )
                    )
                } if initial_asset_ids else {}
                selected = [
                    path_by_id[asset_id]
                    for asset_id in reversed(initial_asset_ids)
                    if asset_id in path_by_id
                ][: count - len(rows)]
                rows.extend(selected)
            if len(rows) < count:
                book_id = connection.execute(
                    select(continuation_projects.c.book_id).where(
                        continuation_projects.c.id == project_id
                    )
                ).scalar_one()
                original = list(
                    connection.execute(
                        select(assets.c.relative_path)
                        .join(page_assets, page_assets.c.asset_id == assets.c.id)
                        .join(pages, pages.c.id == page_assets.c.page_id)
                        .join(chapters, chapters.c.id == pages.c.chapter_id)
                        .where(
                            page_assets.c.role == "source",
                            chapters.c.book_id == book_id,
                        )
                        .order_by(
                            chapters.c.ordinal.desc(),
                            pages.c.ordinal.desc(),
                        )
                        .limit(count - len(rows))
                    ).scalars()
                )
                rows.extend(original)
        return [
            self.storage.resolve_relative_path(str(relative_path))
            for relative_path in reversed(rows)
        ]

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
        run_id = str(config["sourceRunId"])
        with self.engine.connect() as connection:
            pages_payload = [
                _load(value, {})
                for value in connection.execute(
                    select(analysis_page_results.c.payload_json)
                    .where(analysis_page_results.c.run_id == run_id)
                    .order_by(analysis_page_results.c.page_number_snapshot)
                ).scalars()
            ]
            artifacts = {
                f"{kind}:{template}": _load(payload, {})
                for kind, template, payload in connection.execute(
                    select(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                        analysis_artifacts.c.payload_json,
                    ).where(
                        analysis_artifacts.c.run_id == run_id,
                        analysis_artifacts.c.is_active.is_(True),
                    )
                )
            }
        return {
            "direction": dict(config.get("projectConfig", {})).get(
                "direction",
                "",
            ),
            "pageCount": dict(config.get("projectConfig", {})).get(
                "pageCount",
                15,
            ),
            "pages": pages_payload[-10:],
            "artifacts": artifacts,
        }

    def _with_credentials(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = json.loads(json.dumps(config))
        for key in ("vlm", "chat", "imageGen"):
            section = (
                dict(result.get(key))
                if isinstance(result.get(key), Mapping)
                else {}
            )
            credential_id = section.pop("credentialVersionId", None)
            if credential_id:
                with self.engine.connect() as connection:
                    secret = connection.execute(
                        select(credential_versions.c.secret_json).where(
                            credential_versions.c.id == credential_id
                        )
                    ).scalar_one_or_none()
                if secret is None:
                    raise JobConflict(
                        "continuation credential version is missing"
                    )
                loaded = json.loads(secret)
                if isinstance(loaded, Mapping):
                    section.update(loaded)
            result[key] = section
        return result


def _missing_prerequisites(
    *,
    active_run: Mapping[str, Any] | None,
    prerequisites: set[tuple[str, str]],
    timeline: Mapping[str, Any] | None,
) -> list[str]:
    missing: list[str] = []
    if active_run is None:
        missing.append("analysis")
    if ("overview", "story_summary") not in prerequisites:
        missing.append("story_summary")
    if ("compressed_context", "default") not in prerequisites:
        missing.append("compressed_context")
    if timeline is None or str(timeline["status"]) not in {
        "ready",
        "degraded",
    }:
        missing.append("timeline")
    return missing


def _page_dto(
    row: Mapping[str, Any],
    *,
    image_versions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "continuationPageId": str(row["id"]),
        "ordinal": int(row["ordinal"]),
        "revision": int(row["revision"]),
        "payload": _load(row["payload_json"], {}),
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
    return {
        "characterId": str(row["id"]),
        "projectId": str(row["project_id"]),
        "name": str(row["name"]),
        "aliases": _load(row["aliases_json"], []),
        "enabled": bool(row["enabled"]),
        "payload": _load(row["payload_json"], {}),
        "revision": int(row["revision"]),
    }


def _normalize_character_identity(
    name: str,
    aliases: Sequence[str],
) -> tuple[str, list[str]]:
    normalized_name = name.strip()
    if not normalized_name or len(normalized_name) > 500:
        raise ValueError("character name must contain 1-500 characters")
    normalized_aliases = list(
        dict.fromkeys(
            str(value).strip()
            for value in aliases
            if str(value).strip()
        )
    )
    if len(normalized_aliases) > 100 or any(
        len(value) > 500 for value in normalized_aliases
    ):
        raise ValueError("character aliases exceed the allowed size")
    return normalized_name, normalized_aliases


def _form_name(value: str) -> str:
    result = value.strip()
    if not result or len(result) > 500:
        raise ValueError("form name must contain 1-500 characters")
    return result


def _reference_thumbnail_asset_id(
    connection: Connection,
    asset_id: str,
) -> str:
    thumbnail = connection.execute(
        select(
            continuation_character_forms.c.reference_thumbnail_asset_id
        ).where(
            continuation_character_forms.c.reference_asset_id == asset_id
        ).limit(1)
    ).scalar_one_or_none()
    if thumbnail is None:
        thumbnail = connection.execute(
            select(
                continuation_image_versions.c.thumbnail_asset_id
            )
            .where(continuation_image_versions.c.asset_id == asset_id)
            .limit(1)
        ).scalar_one_or_none()
    if thumbnail is None:
        source_page_id = connection.execute(
            select(page_assets.c.page_id).where(
                page_assets.c.asset_id == asset_id,
                page_assets.c.role == "source",
            )
        ).scalar_one_or_none()
        if source_page_id is not None:
            thumbnail = connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == source_page_id,
                    page_assets.c.role == "thumbnail_source",
                )
            ).scalar_one_or_none()
    return str(thumbnail or asset_id)


def _image_info(payload: bytes) -> dict[str, Any]:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        image.verify()
    with Image.open(BytesIO(payload)) as image:
        image_format = str(image.format or "PNG").upper()
        width, height = image.size
    extension, mime_type = {
        "JPEG": ("jpg", "image/jpeg"),
        "WEBP": ("webp", "image/webp"),
        "PNG": ("png", "image/png"),
    }.get(image_format, ("png", "image/png"))
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
