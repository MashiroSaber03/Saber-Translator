"""Transactional commands for durable webpage-import drafts."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
import uuid

from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.jobs.repository import (
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    assets,
    books,
    chapters,
    jobs,
    idempotency_records,
    web_import_draft_pages,
    web_import_drafts,
)
from src.backend_v2.settings.resolver import SettingsResolver


WEB_ENGINES = {"auto", "gallery-dl", "ai-agent"}


class DraftLocked(JobConflict):
    pass


class WebImportCommandService:
    def __init__(self, *, data_root: Path, engine: Engine) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.storage = AssetStorageService(data_root, engine)
        self.settings = SettingsResolver(engine)

    def create_draft(
        self,
        *,
        chapter_id: str,
        source_url: str,
        requested_engine: str,
        idempotency_key: str,
        resolved_options: Mapping[str, Any] | None = None,
        retry_of_job_id: str | None = None,
        retry_mode: str | None = None,
        retry_failed_only: bool = False,
        credential_snapshots: Mapping[str, str] | None = None,
        plugin_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, object]:
        normalized_url = _validated_url(source_url)
        if requested_engine not in WEB_ENGINES:
            raise ValueError("engine must be auto, gallery-dl, or ai-agent")
        chapter = self._chapter(chapter_id)
        draft_id = str(uuid.uuid4())
        temp_relative = (
            Path("temp") / "web-import" / draft_id
        ).as_posix()
        if resolved_options is None:
            frozen_options = self.settings.resolve_web_import(
                source_url=normalized_url,
            )
        else:
            frozen_options = dict(resolved_options)
        frozen_config = {
            "draftId": draft_id,
            "sourceUrl": normalized_url,
            "requestedEngine": requested_engine,
            "actualEngine": None,
            "options": frozen_options,
            "executionMode": "sequential",
        }
        now = utcnow()

        def initialize(connection: Connection, _batch_id: str) -> None:
            connection.execute(
                insert(web_import_drafts).values(
                    id=draft_id,
                    book_id=chapter["book_id"],
                    chapter_id=chapter_id,
                    status="extracting",
                    config_json=_json(frozen_config),
                    temp_relative_path=temp_relative,
                    expires_at=now + timedelta(hours=24),
                    created_at=now,
                    updated_at=now,
                )
            )

        result = self.jobs.create_batch(
            kind="web_extract",
            display_name=f"网页提取 {chapter['book_title']} / {chapter['title']}",
            specs=[
                JobSpec(
                    kind="web_extract",
                    book_id=str(chapter["book_id"]),
                    chapter_id=chapter_id,
                    web_import_draft_id=draft_id,
                    config=frozen_config,
                    credential_snapshots=credential_snapshots,
                    plugin_snapshots=plugin_snapshots,
                    retry_of_job_id=retry_of_job_id,
                    retry_mode=retry_mode,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("web_extract_scan",),
                        ),
                    ),
                    target_display={
                        "book": chapter["book_title"],
                        "chapter": chapter["title"],
                        "url": normalized_url,
                        "engine": requested_engine,
                        **(
                            {
                                "retryOfJobId": retry_of_job_id,
                                "retryItemCount": 1,
                            }
                            if retry_of_job_id
                            else {}
                        ),
                    },
                )
            ],
            idempotency_scope=(
                f"job-retry:{retry_of_job_id}:"
                f"{'failed' if retry_failed_only else 'all'}"
                if retry_of_job_id
                else f"web-extract:{chapter_id}"
            ),
            idempotency_key=idempotency_key,
            response_extra={"draftId": draft_id},
            idempotency_payload={
                "chapterId": chapter_id,
                "sourceUrl": normalized_url,
                "engine": requested_engine,
                "config": frozen_options,
                **(
                    {
                        "sourceJobId": retry_of_job_id,
                        "failedOnly": retry_failed_only,
                        "strategy": retry_mode,
                    }
                    if retry_of_job_id
                    else {}
                ),
            },
            transaction_initializer=initialize,
        )
        return result

    def get_draft(self, draft_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            draft = connection.execute(
                select(web_import_drafts).where(
                    web_import_drafts.c.id == draft_id
                )
            ).mappings().one_or_none()
            if draft is None:
                raise LookupError("web import draft not found")
            counts = connection.execute(
                select(
                    func.count().label("candidate_count"),
                    func.count()
                    .filter(
                        web_import_draft_pages.c.selected.is_(True),
                        web_import_draft_pages.c.error_json.is_(None),
                    )
                    .label("selected_count"),
                    func.count()
                    .filter(web_import_draft_pages.c.error_json.is_not(None))
                    .label("failed_count"),
                ).where(web_import_draft_pages.c.draft_id == draft_id)
            ).mappings().one()
            job_rows = list(
                connection.execute(
                    select(jobs.c.id, jobs.c.kind, jobs.c.status).where(
                        jobs.c.web_import_draft_id == draft_id
                    ).order_by(jobs.c.created_at)
                ).mappings()
            )
        config = json.loads(draft["config_json"])
        options = config["options"]
        if not isinstance(options, Mapping):
            raise ValueError("web import draft settings snapshot is invalid")
        return {
            "id": draft["id"],
            "bookId": draft["book_id"],
            "chapterId": draft["chapter_id"],
            "status": str(draft["status"]),
            "revision": draft["revision"],
            "sourceUrl": config.get("sourceUrl"),
            "requestedEngine": config.get("requestedEngine"),
            "actualEngine": config.get("actualEngine"),
            "autoImport": bool(options["autoImport"]),
            "candidateCount": int(counts["candidate_count"]),
            "selectedCount": int(counts["selected_count"]),
            "failedCount": int(counts["failed_count"]),
            "expiresAt": draft["expires_at"].isoformat(),
            "jobs": [dict(row) for row in job_rows],
        }

    def list_draft_pages(
        self,
        *,
        draft_id: str,
        after_ordinal: int,
        limit: int,
    ) -> dict[str, object]:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        self._active_draft(draft_id)
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(web_import_draft_pages)
                    .where(
                        web_import_draft_pages.c.draft_id == draft_id,
                        web_import_draft_pages.c.ordinal > after_ordinal,
                    )
                    .order_by(web_import_draft_pages.c.ordinal)
                    .limit(limit + 1)
                ).mappings()
            )
        has_more = len(rows) > limit
        visible = rows[:limit]
        return {
            "items": [
                {
                    "id": row["id"],
                    "ordinal": row["ordinal"],
                    "selected": bool(row["selected"]),
                    "sourceUrl": row["source_url"],
                    "checksum": row["checksum"],
                    "error": (
                        json.loads(row["error_json"])
                        if row["error_json"]
                        else None
                    ),
                    "sourceMediaUrl": (
                        f"/api/v2/web-import/drafts/{draft_id}/pages/"
                        f"{row['id']}/media?variant=source"
                        if row["temp_relative_path"] and not row["error_json"]
                        else None
                    ),
                    "thumbnailUrl": (
                        f"/api/v2/web-import/drafts/{draft_id}/pages/"
                        f"{row['id']}/media?variant=thumbnail"
                        if row["thumbnail_asset_id"] and not row["error_json"]
                        else None
                    ),
                }
                for row in visible
            ],
            "nextCursor": (
                visible[-1]["ordinal"]
                if has_more and visible
                else None
            ),
        }

    def update_selection(
        self,
        *,
        draft_id: str,
        selected_page_ids: Sequence[str],
        base_revision: int,
        idempotency_key: str,
    ) -> dict[str, object]:
        if len(selected_page_ids) != len(set(selected_page_ids)):
            raise ValueError("selectedPageIds must contain unique IDs")
        now = utcnow()
        scope = f"PUT:updateWebImportSelection:{draft_id}"
        request_hash = _request_hash(
            {
                "draftId": draft_id,
                "baseRevision": base_revision,
                "selectedPageIds": sorted(selected_page_ids),
            }
        )
        with immediate_transaction(self.engine) as connection:
            replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                return replay
            draft = connection.execute(
                select(
                    web_import_drafts.c.status,
                    web_import_drafts.c.revision,
                    web_import_drafts.c.expires_at,
                ).where(web_import_drafts.c.id == draft_id)
            ).mappings().one_or_none()
            if draft is None or draft["expires_at"] <= now:
                raise LookupError("web import draft not found or expired")
            if draft["status"] != "ready":
                raise JobConflict("draft selection can only change while ready")
            if draft["revision"] != base_revision:
                raise JobConflict("draft revision changed")
            valid = set(
                connection.execute(
                    select(web_import_draft_pages.c.id).where(
                        web_import_draft_pages.c.draft_id == draft_id,
                        web_import_draft_pages.c.error_json.is_(None),
                    )
                ).scalars()
            )
            if not set(selected_page_ids).issubset(valid):
                raise ValueError(
                    "selectedPageIds must identify successful pages in this draft"
                )
            connection.execute(
                update(web_import_draft_pages)
                .where(web_import_draft_pages.c.draft_id == draft_id)
                .values(selected=False, updated_at=now)
            )
            if selected_page_ids:
                connection.execute(
                    update(web_import_draft_pages)
                    .where(
                        web_import_draft_pages.c.draft_id == draft_id,
                        web_import_draft_pages.c.id.in_(
                            tuple(selected_page_ids)
                        ),
                    )
                    .values(selected=True, updated_at=now)
                )
            changed = connection.execute(
                update(web_import_drafts)
                .where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.revision == base_revision,
                )
                .values(
                    revision=base_revision + 1,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise JobConflict("draft revision changed")
            response = {
                "draftId": draft_id,
                "revision": base_revision + 1,
                "selectedPageIds": list(selected_page_ids),
            }
            _record_idempotency(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                response=response,
                resource_id=draft_id,
                now=now,
            )
            return response

    def commit(
        self,
        *,
        draft_id: str,
        base_revision: int,
        idempotency_key: str,
        selected_only: bool = True,
    ) -> dict[str, object]:
        now = utcnow()
        with self.engine.connect() as connection:
            draft = connection.execute(
                select(web_import_drafts).where(
                    web_import_drafts.c.id == draft_id
                )
            ).mappings().one_or_none()
            if draft is None or draft["expires_at"] <= now:
                raise LookupError("web import draft not found or expired")
            if draft["status"] != "ready":
                existing = self._existing_root_commit(draft_id)
                if existing is not None:
                    return existing
                raise JobConflict("web import draft is not ready")
            rows = list(
                connection.execute(
                    select(web_import_draft_pages)
                    .where(
                        web_import_draft_pages.c.draft_id == draft_id,
                        *(
                            (web_import_draft_pages.c.selected.is_(True),)
                            if selected_only
                            else ()
                        ),
                        web_import_draft_pages.c.error_json.is_(None),
                    )
                    .order_by(web_import_draft_pages.c.ordinal)
                ).mappings()
            )
        chapter = self._chapter(str(draft["chapter_id"]))
        if not rows:
            raise ValueError("select at least one successful draft page")
        entries = [
            {
                "draftPageId": row["id"],
                "ordinal": row["ordinal"],
                "sourceUrl": row["source_url"],
                "relativePath": row["temp_relative_path"],
                "checksum": row["checksum"],
            }
            for row in rows
        ]
        config = {
            "draftId": draft_id,
            "draftRevision": base_revision,
            "chapterId": str(draft["chapter_id"]),
            "entries": entries,
            "executionMode": "sequential",
        }

        def hook(
            connection: Connection,
            _batch_id: str,
            _job_ids: Sequence[str],
        ) -> None:
            changed = connection.execute(
                update(web_import_drafts)
                .where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.status == "ready",
                    web_import_drafts.c.revision == base_revision,
                    web_import_drafts.c.expires_at > now,
                )
                .values(
                    status="committing",
                    revision=base_revision + 1,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise JobConflict(
                    "draft is no longer ready at the requested revision"
                )
            if not selected_only:
                connection.execute(
                    update(web_import_draft_pages)
                    .where(
                        web_import_draft_pages.c.draft_id == draft_id,
                        web_import_draft_pages.c.error_json.is_(None),
                    )
                    .values(selected=True, updated_at=now)
                )

        try:
            result = self.jobs.create_batch(
                kind="web_import_commit",
                display_name=(
                    f"网页入库 {chapter['book_title']} / {chapter['title']}"
                ),
                specs=[
                    JobSpec(
                        kind="web_import_commit",
                        book_id=str(chapter["book_id"]),
                        chapter_id=str(draft["chapter_id"]),
                        web_import_draft_id=draft_id,
                        config=config,
                        items=(
                            *(
                                JobItemSpec(
                                    page_id=None,
                                    step_kinds=("web_import_commit_page",),
                                )
                                for _entry in entries
                            ),
                            JobItemSpec(
                                page_id=None,
                                step_kinds=("web_import_commit_finalize",),
                            ),
                        ),
                        target_display={
                            "book": chapter["book_title"],
                            "chapter": chapter["title"],
                            "pageCount": len(entries),
                            "draftId": draft_id,
                        },
                    )
                ],
                idempotency_scope=f"web-import-commit:{draft_id}",
                idempotency_key=idempotency_key,
                idempotency_payload={
                    "draftId": draft_id,
                    "draftRevision": base_revision,
                    "draftPageIds": [
                        entry["draftPageId"] for entry in entries
                    ],
                },
                transaction_hook=hook,
            )
        except JobConflict:
            existing = self._existing_root_commit(draft_id)
            if existing is None:
                raise
            return existing
        return {**result, "draftId": draft_id}

    def delete_draft(
        self,
        draft_id: str,
        *,
        idempotency_key: str,
    ) -> dict[str, object]:
        now = utcnow()
        scope = f"DELETE:deleteWebImportDraft:{draft_id}"
        request_hash = _request_hash({"draftId": draft_id})
        with immediate_transaction(self.engine) as connection:
            replay = _idempotency_replay(
                connection,
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
                now=now,
            )
            if replay is not None:
                response = replay
            else:
                if connection.execute(
                    select(jobs.c.id).where(
                        jobs.c.web_import_draft_id == draft_id,
                        jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                    )
                ).scalar_one_or_none() is not None:
                    raise DraftLocked("draft is referenced by nonterminal work")
                removed = connection.execute(
                    delete(web_import_drafts).where(
                        web_import_drafts.c.id == draft_id
                    )
                )
                if removed.rowcount != 1:
                    raise LookupError("web import draft not found")
                response = {"deleted": True}
                _record_idempotency(
                    connection,
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    response=response,
                    resource_id=draft_id,
                    now=now,
                )
        directory = self.data_root / "temp" / "web-import" / draft_id
        shutil.rmtree(directory, ignore_errors=True)
        return response

    def media(
        self,
        *,
        draft_id: str,
        page_id: str,
        variant: str,
    ) -> tuple[Path, str]:
        self._active_draft(draft_id)
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    web_import_draft_pages.c.temp_relative_path,
                    web_import_draft_pages.c.thumbnail_asset_id,
                    assets.c.relative_path.label("thumbnail_relative_path"),
                    assets.c.mime_type.label("thumbnail_mime_type"),
                )
                .outerjoin(
                    assets,
                    assets.c.id
                    == web_import_draft_pages.c.thumbnail_asset_id,
                )
                .where(
                    web_import_draft_pages.c.id == page_id,
                    web_import_draft_pages.c.draft_id == draft_id,
                )
            ).mappings().one_or_none()
        if row is None:
            raise LookupError("draft page not found")
        if variant == "thumbnail":
            relative = row["thumbnail_relative_path"]
            mime = row["thumbnail_mime_type"]
        elif variant == "source":
            relative = row["temp_relative_path"]
            mime = _image_mime(self.storage.resolve_relative_path(relative))
        else:
            raise ValueError("variant must be source or thumbnail")
        if not relative:
            raise LookupError("draft media is unavailable")
        path = self.storage.resolve_relative_path(str(relative))
        if not path.is_file():
            raise LookupError("draft media is unavailable")
        return path, str(mime or "application/octet-stream")

    def _active_draft(self, draft_id: str):
        now = utcnow()
        with self.engine.connect() as connection:
            row = connection.execute(
                select(web_import_drafts).where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.expires_at > now,
                )
            ).mappings().one_or_none()
        if row is None:
            raise LookupError("web import draft not found or expired")
        return row

    def _chapter(self, chapter_id: str):
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    chapters.c.id,
                    chapters.c.book_id,
                    chapters.c.title,
                    books.c.title.label("book_title"),
                )
                .join(books, books.c.id == chapters.c.book_id)
                .where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
        if row is None:
            raise LookupError("chapter not found")
        return row

    def _existing_root_commit(
        self,
        draft_id: str,
    ) -> dict[str, object] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(jobs.c.id, jobs.c.batch_id, jobs.c.status)
                .where(
                    jobs.c.web_import_draft_id == draft_id,
                    jobs.c.kind == "web_import_commit",
                    jobs.c.retry_of_job_id.is_(None),
                )
                .order_by(jobs.c.created_at)
                .limit(1)
            ).mappings().one_or_none()
        if row is None or row["batch_id"] is None:
            return None
        return {
            "batchId": str(row["batch_id"]),
            "jobIds": [str(row["id"])],
            "status": str(row["status"]),
            "draftId": draft_id,
        }


def _request_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(dict(payload)).encode("utf-8")).hexdigest()


def _idempotency_replay(
    connection: Connection,
    *,
    scope: str,
    key: str,
    request_hash: str,
    now: datetime,
) -> dict[str, object] | None:
    if not key or len(key) > 200:
        raise ValueError("Idempotency-Key is required and must be at most 200 characters")
    row = connection.execute(
        select(
            idempotency_records.c.request_hash,
            idempotency_records.c.response_json,
            idempotency_records.c.expires_at,
        ).where(
            idempotency_records.c.scope == scope,
            idempotency_records.c.key == key,
        )
    ).mappings().one_or_none()
    if row is None:
        return None
    if row["expires_at"] <= now:
        connection.execute(
            delete(idempotency_records).where(
                idempotency_records.c.scope == scope,
                idempotency_records.c.key == key,
            )
        )
        return None
    if row["request_hash"] != request_hash:
        raise JobConflict("Idempotency-Key was reused for different web import input")
    return json.loads(str(row["response_json"]))


def _record_idempotency(
    connection: Connection,
    *,
    scope: str,
    key: str,
    request_hash: str,
    response: Mapping[str, object],
    resource_id: str,
    now: datetime,
) -> None:
    connection.execute(
        insert(idempotency_records).values(
            scope=scope,
            key=key,
            request_hash=request_hash,
            http_status=200,
            response_json=_json(dict(response)),
            resource_type="web_import_draft",
            resource_id=resource_id,
            created_at=now,
            expires_at=now + timedelta(days=7),
        )
    )

def _validated_url(value: str) -> str:
    normalized = value.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("sourceUrl must be an absolute HTTP(S) URL")
    if len(normalized) > 8_000:
        raise ValueError("sourceUrl is too long")
    return normalized


def _image_mime(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        image_format = str(image.format or "").upper()
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
    }.get(image_format, "application/octet-stream")
