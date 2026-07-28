"""Transactional commands for durable webpage-import drafts."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
import uuid

from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.jobs.repository import (
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
    utcnow,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import (
    assets,
    books,
    chapters,
    jobs,
    web_import_draft_pages,
    web_import_drafts,
)


WEB_ENGINES = {"auto", "gallery-dl", "ai-agent"}
NONTERMINAL = {
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "interrupted",
}
_SECRET_TOKENS = {
    "apikey",
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}


class WebImportCommandService:
    def __init__(self, *, data_root: Path, engine: Engine) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.storage = AssetStorageService(data_root, engine)

    def create_draft(
        self,
        *,
        chapter_id: str,
        source_url: str,
        requested_engine: str,
        config: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, object]:
        normalized_url = _validated_url(source_url)
        if requested_engine not in WEB_ENGINES:
            raise ValueError("engine must be auto, gallery-dl, or ai-agent")
        _reject_plaintext_secrets(config)
        chapter = self._chapter(chapter_id)
        draft_id = str(uuid.uuid4())
        temp_relative = (
            Path("temp") / "web-import" / draft_id
        ).as_posix()
        frozen_config = {
            "draftId": draft_id,
            "sourceUrl": normalized_url,
            "requestedEngine": requested_engine,
            "actualEngine": None,
            "options": dict(config),
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
                    },
                )
            ],
            idempotency_scope=f"web-extract:{chapter_id}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "chapterId": chapter_id,
                "sourceUrl": normalized_url,
                "engine": requested_engine,
                "config": dict(config),
            },
            transaction_initializer=initialize,
        )
        actual_draft_id = self._job_draft_id(str(result["jobIds"][0]))
        return {**result, "draftId": actual_draft_id}

    def get_draft(self, draft_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            draft = connection.execute(
                select(web_import_drafts).where(
                    web_import_drafts.c.id == draft_id
                )
            ).mappings().one_or_none()
            if draft is None:
                raise LookupError("web import draft not found")
            counts = list(
                connection.execute(
                    select(
                        web_import_draft_pages.c.selected,
                        web_import_draft_pages.c.error_json,
                    ).where(web_import_draft_pages.c.draft_id == draft_id)
                ).mappings()
            )
            job_rows = list(
                connection.execute(
                    select(jobs.c.id, jobs.c.kind, jobs.c.status).where(
                        jobs.c.web_import_draft_id == draft_id
                    ).order_by(jobs.c.created_at)
                ).mappings()
            )
        config = json.loads(draft["config_json"])
        return {
            "id": draft["id"],
            "bookId": draft["book_id"],
            "chapterId": draft["chapter_id"],
            "status": draft["status"],
            "revision": draft["revision"],
            "sourceUrl": config.get("sourceUrl"),
            "requestedEngine": config.get("requestedEngine"),
            "actualEngine": config.get("actualEngine"),
            "candidateCount": len(counts),
            "selectedCount": sum(
                1
                for row in counts
                if row["selected"] and row["error_json"] is None
            ),
            "failedCount": sum(
                1 for row in counts if row["error_json"] is not None
            ),
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
    ) -> dict[str, object]:
        if len(selected_page_ids) != len(set(selected_page_ids)):
            raise ValueError("selectedPageIds must contain unique IDs")
        now = utcnow()
        with self.engine.begin() as connection:
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
        return {
            "draftId": draft_id,
            "revision": base_revision + 1,
            "selectedPageIds": list(selected_page_ids),
        }

    def commit(
        self,
        *,
        draft_id: str,
        base_revision: int,
        idempotency_key: str,
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
            rows = list(
                connection.execute(
                    select(web_import_draft_pages)
                    .where(
                        web_import_draft_pages.c.draft_id == draft_id,
                        web_import_draft_pages.c.selected.is_(True),
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
                    items=tuple(
                        [
                            JobItemSpec(
                                page_id=None,
                                step_kinds=("web_import_commit_page",),
                            )
                            for _entry in entries
                        ]
                        + [
                            JobItemSpec(
                                page_id=None,
                                step_kinds=("web_import_commit_finalize",),
                            )
                        ]
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
        return {**result, "draftId": draft_id}

    def delete_draft(self, draft_id: str) -> None:
        with self.engine.begin() as connection:
            if connection.execute(
                select(jobs.c.id).where(
                    jobs.c.web_import_draft_id == draft_id,
                    jobs.c.status.in_(tuple(NONTERMINAL)),
                )
            ).scalar_one_or_none() is not None:
                raise JobConflict("draft is referenced by nonterminal work")
            removed = connection.execute(
                delete(web_import_drafts).where(
                    web_import_drafts.c.id == draft_id
                )
            )
            if removed.rowcount != 1:
                raise LookupError("web import draft not found")
        directory = self.data_root / "temp" / "web-import" / draft_id
        if directory.is_dir():
            import shutil

            shutil.rmtree(directory)
        self.storage.collect_garbage()

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

    def _job_draft_id(self, job_id: str) -> str:
        with self.engine.connect() as connection:
            value = connection.execute(
                select(jobs.c.web_import_draft_id).where(jobs.c.id == job_id)
            ).scalar_one()
        return str(value)


def _validated_url(value: str) -> str:
    normalized = value.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("sourceUrl must be an absolute HTTP(S) URL")
    if len(normalized) > 8_000:
        raise ValueError("sourceUrl is too long")
    return normalized


def _reject_plaintext_secrets(value: object, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).replace("-", "_").casefold()
            if (
                normalized in _SECRET_TOKENS
                or normalized.endswith("_secret")
                or normalized.endswith("_token")
                or normalized.endswith("_key")
            ) and child not in (None, ""):
                raise ValueError(
                    f"{path}.{key} must reference a backend credential version"
                )
            _reject_plaintext_secrets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_plaintext_secrets(child, f"{path}[{index}]")


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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
