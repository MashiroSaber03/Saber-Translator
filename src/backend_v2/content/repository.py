"""Relational content repository with revision CAS and atomic ordinals."""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
import uuid

from sqlalchemy import Engine, and_, delete, exists, func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.timestamps import utcnow as _utcnow
from src.backend_v2.content.translation_constraints import (
    TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
    empty_translation_constraints,
    validate_translation_constraints,
)
from src.backend_v2.content.page_style import (
    PAGE_STYLE_SCHEMA_VERSION,
    rgb_to_hex,
    validate_page_style,
    validate_text_style_defaults,
)
from src.backend_v2.insight.repository import (
    mark_book_insight_derived_stale,
)
from src.backend_v2.jobs.repository import decode_job_progress
from src.backend_v2.rendering.fonts import resolve_font_path
from src.backend_v2.settings.validation import validate_setting_payload
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.defaults import (
    TRANSLATION_SETTINGS_SCHEMA_VERSION,
    default_translation_settings,
)
from src.backend_v2.storage.schema import (
    ACTIVE_OPERATION_STATUSES,
    ACTIVE_RENDER_REQUEST_STATUSES,
    NONTERMINAL_JOB_STATUSES,
    assets,
    book_tags,
    books,
    bubbles,
    chapter_write_intents,
    chapter_write_locks,
    chapter_navigation_state,
    chapters,
    idempotency_records,
    fonts,
    job_items,
    jobs,
    operations,
    page_assets,
    pages,
    render_requests,
    studio_chat_sessions,
    studio_documents,
    tags,
    translation_constraints,
    web_import_drafts,
)
from src.backend_v2.storage.seeding import QUICK_WORKSPACE_BOOK_ID
from src.core.config_models import (
    STORED_BUBBLE_FIELDS,
    validate_bubble_payload,
)


CHAPTER_SETTINGS_MEMORY_SCHEMA_VERSION = 1
_MISSING = object()
_CLIENT_BUBBLE_MUTATION_FIELDS = (
    STORED_BUBBLE_FIELDS - {"autoTextDirection"}
) | {"fontId"}
_CHAPTER_WORK_STATE_KEYS = frozenset(
    {
        "ocrEngine",
        "sourceLanguage",
        "textDetector",
        "minTextBlockAreaPercent",
        "enableAuxYoloDetection",
        "auxYoloConfThreshold",
        "auxYoloOverlapThreshold",
        "enableSaberYoloRefine",
        "saberYoloRefineOverlapThreshold",
        "baiduOcr",
        "paddleOcrVl",
        "aiVisionOcr",
        "hybridOcr",
        "translation",
        "targetLanguage",
        "translatePrompt",
        "useTextboxPrompt",
        "textboxPrompt",
        "hqTranslation",
        "proofreading",
        "boxExpand",
        "preciseMask",
        "showDetectionDebug",
        "parallel",
        "removeTextWithOcr",
        "lamaDisableResize",
    }
)


def _auto_text_direction_from_coords(coords: list[int]) -> str:
    return (
        "vertical"
        if coords[3] - coords[1] > coords[2] - coords[0]
        else "horizontal"
    )


_CHAPTER_MEMORY_FORBIDDEN_KEYS = frozenset(
    {
        "apikey",
        "secretkey",
        "secret",
        "token",
        "password",
        "credentialversionid",
        "textstyle",
        "fontsize",
        "autofontsize",
        "fontfamily",
        "layoutdirection",
        "textcolor",
        "fillcolor",
        "inpaintmethod",
        "useautotextcolor",
        "strokeenabled",
        "strokecolor",
        "strokewidth",
        "linespacing",
        "textalign",
    }
)


class ContentNotFound(LookupError):
    pass


class ContentConflict(RuntimeError):
    pass


class ContentLocked(RuntimeError):
    pass


class IdempotencyConflict(ContentConflict):
    pass


def _validate_chapter_settings_memory(payload: dict[str, object]) -> None:
    unknown = sorted(set(payload) - _CHAPTER_WORK_STATE_KEYS)
    if unknown:
        raise ValueError(
            "chapter settings memory contains unsupported fields: "
            + ", ".join(unknown)
        )
    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if normalized in _CHAPTER_MEMORY_FORBIDDEN_KEYS:
                    raise ValueError(
                        f"chapter settings memory must not contain {key}"
                    )
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)

    def merge(
        base: dict[str, object],
        override: dict[str, object],
    ) -> dict[str, object]:
        result = json.loads(_json(base))
        for key, value in override.items():
            current = result.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                result[key] = merge(current, value)
            else:
                result[key] = value
        return result

    validate_setting_payload(
        "translation",
        merge(default_translation_settings(), payload),
        schema_version=TRANSLATION_SETTINGS_SCHEMA_VERSION,
    )


def normalize_logical_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or any(part in ("", ".") for part in path.parts)
    ):
        raise ValueError("logical source path must be a normalized relative path")
    return path.as_posix()


def deduplicate_logical_path(requested: str, existing: set[str]) -> str:
    if requested not in existing:
        return requested
    path = PurePosixPath(requested)
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    parent = "" if str(path.parent) == "." else f"{path.parent.as_posix()}/"
    counter = 2
    while True:
        candidate = f"{parent}{stem} ({counter}){suffix}"
        if candidate not in existing:
            return candidate
        counter += 1


def natural_sort_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


class ContentRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        database_path = engine.url.database
        self.storage = (
            AssetStorageService(Path(database_path).resolve().parent, engine)
            if database_path and database_path != ":memory:"
            else None
        )

    def list_books(
        self,
        *,
        search: str = "",
        tag_ids: tuple[str, ...] = (),
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> list[dict[str, object]]:
        if len(tag_ids) != len(set(tag_ids)):
            raise ValueError("tagIds must contain unique IDs")
        if sort_by not in {"title", "created_at", "updated_at"}:
            raise ValueError("sort_by must be title, created_at, or updated_at")
        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order must be asc or desc")
        chapter_counts = (
            select(
                chapters.c.book_id,
                func.count(chapters.c.id).label("chapter_count"),
            )
            .group_by(chapters.c.book_id)
            .subquery()
        )
        page_counts = (
            select(
                chapters.c.book_id,
                func.count(pages.c.id).label("page_count"),
            )
            .select_from(chapters.join(pages, pages.c.chapter_id == chapters.c.id))
            .group_by(chapters.c.book_id)
            .subquery()
        )
        statement = (
            select(
                books,
                func.coalesce(chapter_counts.c.chapter_count, 0).label("chapter_count"),
                func.coalesce(page_counts.c.page_count, 0).label("page_count"),
            )
            .outerjoin(chapter_counts, chapter_counts.c.book_id == books.c.id)
            .outerjoin(page_counts, page_counts.c.book_id == books.c.id)
            .where(books.c.kind == "library")
        )
        if sort_by != "title":
            sort_column = (
                books.c.created_at
                if sort_by == "created_at"
                else books.c.updated_at
            )
            statement = statement.order_by(
                sort_column.asc() if sort_order == "asc" else sort_column.desc(),
                books.c.id,
            )
        normalized_search = search.strip()
        if normalized_search:
            pattern = f"%{normalized_search}%"
            statement = statement.where(
                or_(
                    books.c.title.ilike(pattern),
                    books.c.id.in_(
                        select(book_tags.c.book_id)
                        .join(tags, tags.c.id == book_tags.c.tag_id)
                        .where(tags.c.name.ilike(pattern))
                    ),
                )
            )
        if tag_ids:
            statement = statement.where(
                books.c.id.in_(
                    select(book_tags.c.book_id)
                    .where(book_tags.c.tag_id.in_(tag_ids))
                    .group_by(book_tags.c.book_id)
                    .having(func.count(func.distinct(book_tags.c.tag_id)) == len(tag_ids))
                )
            )
        with self.engine.connect() as connection:
            rows = list(connection.execute(statement).mappings())
            tag_rows = list(
                connection.execute(
                    select(book_tags.c.book_id, tags.c.id, tags.c.name, tags.c.color)
                    .join(tags, tags.c.id == book_tags.c.tag_id)
                    .where(book_tags.c.book_id.in_([row["id"] for row in rows]))
                    .order_by(book_tags.c.book_id, tags.c.name, tags.c.id)
                ).mappings()
            ) if rows else []
            job_rows = list(
                connection.execute(
                    select(
                        jobs.c.book_id,
                        jobs.c.status,
                        func.count(jobs.c.id).label("job_count"),
                    )
                    .where(
                        jobs.c.book_id.in_([row["id"] for row in rows]),
                        jobs.c.status.in_(
                            (
                                "queued",
                                "running",
                                "pausing",
                                "paused",
                                "cancelling",
                                "interrupted",
                                "failed",
                            )
                        ),
                    )
                    .group_by(jobs.c.book_id, jobs.c.status)
                ).mappings()
            ) if rows else []
        if sort_by == "title":
            rows.sort(
                key=lambda row: natural_sort_key(row["title"]),
                reverse=sort_order == "desc",
            )
        tags_by_book: dict[str, list[dict[str, str]]] = {}
        for row in tag_rows:
            tags_by_book.setdefault(str(row["book_id"]), []).append(
                {"id": str(row["id"]), "name": str(row["name"]), "color": str(row["color"])}
            )
        jobs_by_book: dict[str, dict[str, int]] = {}
        for row in job_rows:
            jobs_by_book.setdefault(str(row["book_id"]), {})[
                str(row["status"])
            ] = int(row["job_count"])
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "coverAssetUrl": (
                    f"/api/v2/assets/{row['cover_asset_id']}"
                    if row["cover_asset_id"]
                    else None
                ),
                "chapterCount": row["chapter_count"],
                "pageCount": row["page_count"],
                "tags": tags_by_book.get(str(row["id"]), []),
                "jobStatusSummary": jobs_by_book.get(str(row["id"]), {}),
                "chapterOrderRevision": row["chapter_order_revision"],
                "createdAt": row["created_at"].isoformat(),
                "updatedAt": row["updated_at"].isoformat(),
            }
            for row in rows
        ]

    def create_book(
        self,
        *,
        title: str,
        tag_ids: list[str] | None = None,
        cover_asset_id: str | None = None,
    ) -> dict[str, object]:
        normalized = title.strip()
        if not normalized or len(normalized) > 500:
            raise ValueError("book title must contain 1-500 characters")
        book_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                insert(books).values(
                    id=book_id,
                    kind="library",
                    title=normalized,
                    cover_asset_id=cover_asset_id,
                )
            )
            connection.execute(
                insert(translation_constraints).values(
                    book_id=book_id,
                    payload_json=_json(empty_translation_constraints()),
                    schema_version=TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
                )
            )
            if tag_ids is not None:
                self._replace_book_tags(connection, book_id, tag_ids)
        return {
            "id": book_id,
            "title": normalized,
            "chapterOrderRevision": 1,
            "coverAssetUrl": (
                f"/api/v2/assets/{cover_asset_id}"
                if cover_asset_id
                else None
            ),
        }

    def get_book(self, book_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            book = connection.execute(
                select(books).where(
                    books.c.id == book_id,
                    books.c.kind == "library",
                )
            ).mappings().one_or_none()
            if book is None:
                raise ContentNotFound("book not found")
            chapter_rows = list(
                connection.execute(
                    select(
                        chapters.c.id,
                        chapters.c.ordinal,
                        chapters.c.title,
                        chapters.c.page_order_revision,
                        func.count(pages.c.id).label("page_count"),
                    )
                    .outerjoin(pages, pages.c.chapter_id == chapters.c.id)
                    .where(chapters.c.book_id == book_id)
                    .group_by(chapters.c.id)
                    .order_by(chapters.c.ordinal)
                ).mappings()
            )
            tag_rows = list(
                connection.execute(
                    select(tags.c.id, tags.c.name, tags.c.color)
                    .join(book_tags, book_tags.c.tag_id == tags.c.id)
                    .where(book_tags.c.book_id == book_id)
                    .order_by(tags.c.name)
                ).mappings()
            )
            chapter_job_rows = list(
                connection.execute(
                    select(
                        jobs.c.chapter_id,
                        jobs.c.status,
                        func.count(jobs.c.id).label("job_count"),
                    )
                    .where(
                        jobs.c.chapter_id.in_(
                            [row["id"] for row in chapter_rows]
                        ),
                        jobs.c.kind == "translation",
                        jobs.c.status.in_(
                            (
                                "queued",
                                "running",
                                "pausing",
                                "paused",
                                "cancelling",
                                "interrupted",
                                "failed",
                            )
                        ),
                    )
                    .group_by(jobs.c.chapter_id, jobs.c.status)
                ).mappings()
            ) if chapter_rows else []
        chapter_jobs: dict[str, dict[str, int]] = {}
        for row in chapter_job_rows:
            chapter_jobs.setdefault(str(row["chapter_id"]), {})[
                str(row["status"])
            ] = int(row["job_count"])
        return {
            "id": book["id"],
            "title": book["title"],
            "coverAssetUrl": (
                f"/api/v2/assets/{book['cover_asset_id']}"
                if book["cover_asset_id"]
                else None
            ),
            "chapterOrderRevision": book["chapter_order_revision"],
            "createdAt": book["created_at"].isoformat(),
            "updatedAt": book["updated_at"].isoformat(),
            "tags": [dict(row) for row in tag_rows],
            "chapters": [
                {
                    "id": row["id"],
                    "ordinal": row["ordinal"],
                    "title": row["title"],
                    "pageCount": row["page_count"],
                    "pageOrderRevision": row["page_order_revision"],
                    "jobStatusSummary": chapter_jobs.get(str(row["id"]), {}),
                }
                for row in chapter_rows
            ],
        }

    def update_book(
        self,
        *,
        book_id: str,
        title: str | None = None,
        tag_ids: list[str] | None = None,
        cover_asset_id: str | None = None,
        replace_cover: bool = False,
    ) -> dict[str, object]:
        normalized_title: str | None = None
        if title is not None:
            normalized_title = title.strip()
            if not normalized_title or len(normalized_title) > 500:
                raise ValueError("book title must contain 1-500 characters")
        with immediate_transaction(self.engine) as connection:
            values: dict[str, object] = {"updated_at": _utcnow()}
            if normalized_title is not None:
                values["title"] = normalized_title
            if replace_cover:
                values["cover_asset_id"] = cover_asset_id
            changed = connection.execute(
                update(books)
                .where(books.c.id == book_id, books.c.kind == "library")
                .values(**values)
            )
            if changed.rowcount != 1:
                raise ContentNotFound("book not found")
            if tag_ids is not None:
                self._replace_book_tags(connection, book_id, tag_ids)
        return self.get_book(book_id)

    def delete_book(self, book_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            kind = connection.execute(
                select(books.c.kind).where(books.c.id == book_id)
            ).scalar_one_or_none()
            if kind != "library":
                raise ContentNotFound("book not found")
            chapter_ids = [
                str(value)
                for value in connection.execute(
                    select(chapters.c.id).where(chapters.c.book_id == book_id)
                ).scalars()
            ]
            self._assert_targets_idle(connection, book_id, chapter_ids)
            connection.execute(delete(books).where(books.c.id == book_id))

    def batch_delete_books(
        self,
        book_ids: list[str],
    ) -> dict[str, list[dict[str, str]] | list[str]]:
        if not book_ids or len(book_ids) != len(set(book_ids)):
            raise ValueError("bookIds must contain unique IDs")
        deleted: list[str] = []
        rejected: list[dict[str, str]] = []
        for book_id in book_ids:
            try:
                self.delete_book(book_id)
                deleted.append(book_id)
            except (ContentLocked, ContentNotFound) as exc:
                rejected.append(
                    {
                        "bookId": book_id,
                        "reason": (
                            "locked"
                            if isinstance(exc, ContentLocked)
                            else "not_found"
                        ),
                        "message": str(exc),
                    }
                )
        return {"deleted": deleted, "rejected": rejected}

    def update_chapter(self, *, chapter_id: str, title: str) -> dict[str, object]:
        normalized = title.strip()
        if not normalized or len(normalized) > 500:
            raise ValueError("chapter title must contain 1-500 characters")
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(chapters.c.book_id).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if row is None:
                raise ContentNotFound("chapter not found")
            self._assert_targets_idle(
                connection,
                str(row["book_id"]),
                [chapter_id],
            )
            connection.execute(
                update(chapters)
                .where(chapters.c.id == chapter_id)
                .values(title=normalized, updated_at=_utcnow())
            )
        return {"id": chapter_id, "title": normalized}

    def delete_chapter(self, chapter_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(
                    chapters.c.book_id,
                    chapters.c.ordinal,
                    books.c.kind,
                )
                .join(books, books.c.id == chapters.c.book_id)
                .where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if row is None or row["kind"] != "library":
                raise ContentNotFound("chapter not found")
            book_id = str(row["book_id"])
            self._assert_targets_idle(connection, book_id, [chapter_id])
            if connection.execute(
                select(exists().where(pages.c.chapter_id == chapter_id))
            ).scalar_one():
                mark_book_insight_derived_stale(
                    connection,
                    book_id=book_id,
                    now=_utcnow(),
                )
            connection.execute(delete(chapters).where(chapters.c.id == chapter_id))
            connection.execute(
                update(chapters)
                .where(
                    chapters.c.book_id == book_id,
                    chapters.c.ordinal > row["ordinal"],
                )
                .values(ordinal=chapters.c.ordinal - 1)
            )
            connection.execute(
                update(books)
                .where(books.c.id == book_id)
                .values(
                    chapter_order_revision=books.c.chapter_order_revision + 1,
                    updated_at=_utcnow(),
                )
            )

    def list_tags(self) -> list[dict[str, object]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    tags,
                    func.count(book_tags.c.book_id).label("book_count"),
                )
                .outerjoin(book_tags, book_tags.c.tag_id == tags.c.id)
                .group_by(tags.c.id)
                .order_by(tags.c.name)
            ).mappings()
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "color": row["color"],
                    "bookCount": row["book_count"],
                }
                for row in rows
            ]

    def create_tag(self, *, name: str, color: str) -> dict[str, object]:
        normalized_name = name.strip()
        normalized_color = self._normalize_color(color)
        if not normalized_name or len(normalized_name) > 200:
            raise ValueError("tag name must contain 1-200 characters")
        tag_id = str(uuid.uuid4())
        try:
            with immediate_transaction(self.engine) as connection:
                connection.execute(
                    insert(tags).values(
                        id=tag_id,
                        name=normalized_name,
                        color=normalized_color,
                    )
                )
        except IntegrityError as exc:
            raise ContentConflict("tag name already exists") from exc
        return {
            "id": tag_id,
            "name": normalized_name,
            "color": normalized_color,
            "bookCount": 0,
        }

    def update_tag(
        self,
        *,
        tag_id: str,
        name: str,
        color: str,
    ) -> dict[str, object]:
        normalized_name = name.strip()
        normalized_color = self._normalize_color(color)
        if not normalized_name or len(normalized_name) > 200:
            raise ValueError("tag name must contain 1-200 characters")
        try:
            with immediate_transaction(self.engine) as connection:
                changed = connection.execute(
                    update(tags)
                    .where(tags.c.id == tag_id)
                    .values(
                        name=normalized_name,
                        color=normalized_color,
                        updated_at=_utcnow(),
                    )
                )
                if changed.rowcount != 1:
                    raise ContentNotFound("tag not found")
        except IntegrityError as exc:
            raise ContentConflict("tag name already exists") from exc
        return {
            "id": tag_id,
            "name": normalized_name,
            "color": normalized_color,
        }

    def delete_tag(self, tag_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            removed = connection.execute(delete(tags).where(tags.c.id == tag_id))
            if removed.rowcount != 1:
                raise ContentNotFound("tag not found")

    def batch_update_tags(
        self,
        *,
        book_ids: list[str],
        tag_ids: list[str],
        action: str,
    ) -> None:
        if action not in {"add", "remove"}:
            raise ValueError("tag action must be add or remove")
        if not book_ids or len(book_ids) != len(set(book_ids)):
            raise ValueError("bookIds must contain unique IDs")
        if len(tag_ids) != len(set(tag_ids)):
            raise ValueError("tagIds must contain unique IDs")
        with immediate_transaction(self.engine) as connection:
            existing_books = set(
                connection.execute(
                    select(books.c.id).where(
                        books.c.id.in_(book_ids),
                        books.c.kind == "library",
                    )
                ).scalars()
            )
            existing_tags = set(
                connection.execute(
                    select(tags.c.id).where(tags.c.id.in_(tag_ids))
                ).scalars()
            )
            if existing_books != set(book_ids) or existing_tags != set(tag_ids):
                raise ContentNotFound("book or tag not found")
            if action == "add" and tag_ids:
                connection.execute(
                    insert(book_tags).prefix_with("OR IGNORE"),
                    [
                        {"book_id": book_id, "tag_id": tag_id}
                        for book_id in book_ids
                        for tag_id in tag_ids
                    ],
                )
            elif tag_ids:
                connection.execute(
                    delete(book_tags).where(
                        book_tags.c.book_id.in_(book_ids),
                        book_tags.c.tag_id.in_(tag_ids),
                    )
                )

    def get_constraints(self, book_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(translation_constraints).where(
                    translation_constraints.c.book_id == book_id
                )
            ).mappings().one_or_none()
        if row is None:
            raise ContentNotFound("translation constraints not found")
        if row["schema_version"] != TRANSLATION_CONSTRAINTS_SCHEMA_VERSION:
            raise ValueError("translation constraints schema version is not current")
        return {
            "bookId": book_id,
            "revision": row["revision"],
            "schemaVersion": row["schema_version"],
            "payload": validate_translation_constraints(
                json.loads(row["payload_json"])
            ),
        }

    def update_constraints(
        self,
        *,
        book_id: str,
        base_revision: int,
        payload: dict[str, object],
    ) -> dict[str, object]:
        normalized = validate_translation_constraints(payload)
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(translation_constraints)
                .where(
                    translation_constraints.c.book_id == book_id,
                    translation_constraints.c.revision == base_revision,
                    translation_constraints.c.schema_version
                    == TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
                )
                .values(
                    payload_json=_json(normalized),
                    schema_version=TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
                    revision=base_revision + 1,
                    updated_at=_utcnow(),
                )
            )
            if changed.rowcount != 1:
                exists_book = connection.execute(
                    select(books.c.id).where(books.c.id == book_id)
                ).scalar_one_or_none()
                if exists_book is None:
                    raise ContentNotFound("book not found")
                schema_version = connection.execute(
                    select(translation_constraints.c.schema_version).where(
                        translation_constraints.c.book_id == book_id
                    )
                ).scalar_one_or_none()
                if schema_version != TRANSLATION_CONSTRAINTS_SCHEMA_VERSION:
                    raise ValueError(
                        "translation constraints schema version is not current"
                    )
                raise ContentConflict("translation constraints revision changed")
        return {
            "bookId": book_id,
            "revision": base_revision + 1,
            "schemaVersion": TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
            "payload": normalized,
        }

    def quick_workspace_context(self) -> dict[str, str]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(books.c.id, chapters.c.id.label("chapter_id"))
                .join(chapters, chapters.c.book_id == books.c.id)
                .where(books.c.kind == "quick_workspace")
                .order_by(chapters.c.ordinal)
                .limit(1)
            ).mappings().one_or_none()
        if row is None:
            raise ContentNotFound("quick workspace not found")
        return {"bookId": str(row["id"]), "chapterId": str(row["chapter_id"])}

    def translation_bootstrap(
        self,
        *,
        book_id: str | None = None,
        chapter_id: str | None = None,
    ) -> dict[str, object]:
        if bool(book_id) != bool(chapter_id):
            raise ValueError("bookId and chapterId must be provided together")
        if not book_id:
            context = self.quick_workspace_context()
            book_id = context["bookId"]
            chapter_id = context["chapterId"]
        assert chapter_id is not None
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    books.c.id.label("book_id"),
                    books.c.title.label("book_title"),
                    books.c.kind.label("book_kind"),
                    chapters.c.id.label("chapter_id"),
                    chapters.c.title.label("chapter_title"),
                    chapters.c.page_order_revision,
                    chapters.c.settings_memory_json,
                    chapters.c.settings_memory_schema_version,
                    chapters.c.settings_memory_revision,
                )
                .join(chapters, chapters.c.book_id == books.c.id)
                .where(
                    books.c.id == book_id,
                    chapters.c.id == chapter_id,
                )
            ).mappings().one_or_none()
            if row is None:
                raise ContentNotFound("book/chapter translation context not found")
            navigation = connection.execute(
                select(
                    chapter_navigation_state.c.last_visited_page_id,
                    chapter_navigation_state.c.revision,
                ).where(
                    chapter_navigation_state.c.chapter_id == chapter_id
                )
            ).mappings().one_or_none()
            constraints = connection.execute(
                select(
                    translation_constraints.c.payload_json,
                    translation_constraints.c.schema_version,
                    translation_constraints.c.revision,
                ).where(translation_constraints.c.book_id == book_id)
            ).mappings().one_or_none()
            active_jobs = list(
                connection.execute(
                    select(
                        jobs.c.id,
                        jobs.c.kind,
                        jobs.c.status,
                        jobs.c.queue_rank,
                        jobs.c.latest_progress_json,
                    )
                    .where(
                        jobs.c.chapter_id == chapter_id,
                        jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                    )
                    .order_by(jobs.c.queue_rank, jobs.c.created_at)
                ).mappings()
            )
            active_job_ids = [str(job["id"]) for job in active_jobs]
            active_job_pages: dict[str, list[dict[str, str]]] = {
                job_id: [] for job_id in active_job_ids
            }
            if active_job_ids:
                for job_id, page_id, status in connection.execute(
                    select(
                        job_items.c.job_id,
                        job_items.c.page_id,
                        job_items.c.status,
                    )
                    .where(
                        job_items.c.job_id.in_(active_job_ids),
                        job_items.c.page_id.is_not(None),
                    )
                    .order_by(job_items.c.job_id, job_items.c.ordinal)
                ):
                    active_job_pages[str(job_id)].append(
                        {"pageId": str(page_id), "status": str(status)}
                    )
            active_draft = connection.execute(
                select(
                    web_import_drafts.c.id,
                    web_import_drafts.c.status,
                    web_import_drafts.c.revision,
                    web_import_drafts.c.expires_at,
                )
                .where(
                    web_import_drafts.c.chapter_id == chapter_id,
                    web_import_drafts.c.expires_at > _utcnow(),
                    or_(
                        web_import_drafts.c.status == "ready",
                        and_(
                            web_import_drafts.c.status == "extracting",
                            exists(
                                select(jobs.c.id).where(
                                    jobs.c.web_import_draft_id
                                    == web_import_drafts.c.id,
                                    jobs.c.kind == "web_extract",
                                    jobs.c.status.in_(
                                        NONTERMINAL_JOB_STATUSES
                                    ),
                                )
                            ),
                        ),
                        and_(
                            web_import_drafts.c.status == "committing",
                            exists(
                                select(jobs.c.id).where(
                                    jobs.c.web_import_draft_id
                                    == web_import_drafts.c.id,
                                    jobs.c.kind == "web_import_commit",
                                    jobs.c.status.in_(
                                        NONTERMINAL_JOB_STATUSES
                                    ),
                                )
                            ),
                        ),
                    ),
                )
                .order_by(web_import_drafts.c.updated_at.desc())
                .limit(1)
            ).mappings().one_or_none()
        if constraints is None:
            raise ContentNotFound("translation constraints not found")
        if (
            row["settings_memory_schema_version"]
            != CHAPTER_SETTINGS_MEMORY_SCHEMA_VERSION
        ):
            raise ValueError("chapter settings memory schema version is not current")
        if constraints["schema_version"] != TRANSLATION_CONSTRAINTS_SCHEMA_VERSION:
            raise ValueError("translation constraints schema version is not current")
        settings_memory = json.loads(row["settings_memory_json"])
        if not isinstance(settings_memory, dict):
            raise ValueError("chapter settings memory must be an object")
        _validate_chapter_settings_memory(settings_memory)
        constraint_payload = validate_translation_constraints(
            json.loads(constraints["payload_json"])
        )
        return {
            "book": {
                "id": row["book_id"],
                "title": row["book_title"],
                "kind": row["book_kind"],
            },
            "chapter": {
                "id": row["chapter_id"],
                "title": row["chapter_title"],
                "pageOrderRevision": row["page_order_revision"],
                "settingsMemory": settings_memory,
                "settingsMemorySchemaVersion": row[
                    "settings_memory_schema_version"
                ],
                "settingsMemoryRevision": row["settings_memory_revision"],
            },
            "pages": self.list_pages(
                chapter_id=chapter_id,
                all_pages=True,
            ),
            "navigation": {
                "lastVisitedPageId": (
                    navigation["last_visited_page_id"]
                    if navigation
                    else None
                ),
                "revision": navigation["revision"] if navigation else 0,
            },
            "constraints": {
                "payload": constraint_payload,
                "schemaVersion": constraints["schema_version"],
                "revision": constraints["revision"],
            },
            "activeJobs": [
                {
                    "id": job["id"],
                    "kind": job["kind"],
                    "status": job["status"],
                    "queueRank": job["queue_rank"],
                    "progress": decode_job_progress(job),
                    "pages": active_job_pages.get(str(job["id"]), []),
                }
                for job in active_jobs
            ],
            "activeWebImportDraft": (
                {
                    "id": active_draft["id"],
                    "status": active_draft["status"],
                    "revision": active_draft["revision"],
                    "expiresAt": active_draft["expires_at"].isoformat(),
                }
                if active_draft
                else None
            ),
        }

    def update_chapter_settings_memory(
        self,
        *,
        chapter_id: str,
        base_revision: int,
        payload: dict[str, object],
    ) -> dict[str, object]:
        _validate_chapter_settings_memory(payload)
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(chapters)
                .where(
                    chapters.c.id == chapter_id,
                    chapters.c.settings_memory_revision == base_revision,
                    chapters.c.settings_memory_schema_version
                    == CHAPTER_SETTINGS_MEMORY_SCHEMA_VERSION,
                )
                .values(
                    settings_memory_json=_json(payload),
                    settings_memory_revision=base_revision + 1,
                    updated_at=_utcnow(),
                )
            )
            if changed.rowcount != 1:
                if connection.execute(
                    select(chapters.c.id).where(chapters.c.id == chapter_id)
                ).scalar_one_or_none() is None:
                    raise ContentNotFound("chapter not found")
                schema_version = connection.execute(
                    select(chapters.c.settings_memory_schema_version).where(
                        chapters.c.id == chapter_id
                    )
                ).scalar_one()
                if schema_version != CHAPTER_SETTINGS_MEMORY_SCHEMA_VERSION:
                    raise ValueError(
                        "chapter settings memory schema version is not current"
                    )
                raise ContentConflict("chapter settings memory revision changed")
        return {
            "chapterId": chapter_id,
            "revision": base_revision + 1,
            "payload": payload,
        }

    def update_last_visited_page(
        self,
        *,
        chapter_id: str,
        page_id: str,
    ) -> dict[str, object]:
        with immediate_transaction(self.engine) as connection:
            if connection.execute(
                select(pages.c.id).where(
                    pages.c.id == page_id,
                    pages.c.chapter_id == chapter_id,
                )
            ).scalar_one_or_none() is None:
                raise ContentNotFound("page not found in chapter")
            current = connection.execute(
                select(chapter_navigation_state.c.revision).where(
                    chapter_navigation_state.c.chapter_id == chapter_id
                )
            ).scalar_one_or_none()
            if current is None:
                connection.execute(
                    insert(chapter_navigation_state).values(
                        chapter_id=chapter_id,
                        last_visited_page_id=page_id,
                        revision=1,
                    )
                )
                revision = 1
            else:
                revision = int(current) + 1
                connection.execute(
                    update(chapter_navigation_state)
                    .where(chapter_navigation_state.c.chapter_id == chapter_id)
                    .values(
                        last_visited_page_id=page_id,
                        revision=revision,
                        updated_at=_utcnow(),
                    )
                )
        return {
            "chapterId": chapter_id,
            "lastVisitedPageId": page_id,
            "revision": revision,
        }

    def create_chapter(self, *, book_id: str, title: str) -> dict[str, object]:
        normalized = title.strip()
        if not normalized or len(normalized) > 500:
            raise ValueError("chapter title must contain 1-500 characters")
        chapter_id = str(uuid.uuid4())
        with immediate_transaction(self.engine) as connection:
            book_revision = connection.execute(
                select(books.c.chapter_order_revision).where(
                    books.c.id == book_id,
                    books.c.kind.in_(("library", "quick_workspace")),
                )
            ).scalar_one_or_none()
            if book_revision is None:
                raise ContentNotFound("book not found")
            ordinal = (
                connection.execute(
                    select(func.coalesce(func.max(chapters.c.ordinal), 0)).where(
                        chapters.c.book_id == book_id
                    )
                ).scalar_one()
                + 1
            )
            connection.execute(
                insert(chapters).values(
                    id=chapter_id,
                    book_id=book_id,
                    ordinal=ordinal,
                    title=normalized,
                )
            )
            connection.execute(
                update(books)
                .where(
                    books.c.id == book_id,
                    books.c.chapter_order_revision == book_revision,
                )
                .values(
                    chapter_order_revision=book_revision + 1,
                    updated_at=_utcnow(),
                )
            )
        return {
            "id": chapter_id,
            "bookId": book_id,
            "ordinal": ordinal,
            "title": normalized,
            "pageOrderRevision": 1,
        }

    def list_chapters(self, book_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            book = connection.execute(
                select(
                    books.c.id,
                    books.c.title,
                    books.c.kind,
                    books.c.chapter_order_revision,
                ).where(books.c.id == book_id)
            ).mappings().one_or_none()
            if book is None:
                raise ContentNotFound("book not found")
            rows = list(
                connection.execute(
                    select(
                        chapters.c.id,
                        chapters.c.ordinal,
                        chapters.c.title,
                        chapters.c.page_order_revision,
                        func.count(pages.c.id).label("page_count"),
                    )
                    .outerjoin(pages, pages.c.chapter_id == chapters.c.id)
                    .where(chapters.c.book_id == book_id)
                    .group_by(chapters.c.id)
                    .order_by(chapters.c.ordinal)
                ).mappings()
            )
        return {
            "book": dict(book),
            "chapters": [
                {
                    "id": row["id"],
                    "ordinal": row["ordinal"],
                    "title": row["title"],
                    "pageCount": row["page_count"],
                    "pageOrderRevision": row["page_order_revision"],
                }
                for row in rows
            ],
        }

    def reorder_chapters(
        self,
        *,
        book_id: str,
        ordered_ids: list[str],
        base_revision: int,
    ) -> int:
        with immediate_transaction(self.engine) as connection:
            revision = connection.execute(
                select(books.c.chapter_order_revision).where(books.c.id == book_id)
            ).scalar_one_or_none()
            if revision is None:
                raise ContentNotFound("book not found")
            if revision != base_revision:
                raise ContentConflict("chapter order revision changed")
            existing = list(
                connection.execute(
                    select(chapters.c.id)
                    .where(chapters.c.book_id == book_id)
                    .order_by(chapters.c.ordinal)
                ).scalars()
            )
            if len(ordered_ids) != len(set(ordered_ids)) or set(existing) != set(ordered_ids):
                raise ValueError("ordered chapter ids must be an exact permutation")
            offset = len(existing) * 2 + 1
            connection.execute(
                update(chapters)
                .where(chapters.c.book_id == book_id)
                .values(ordinal=chapters.c.ordinal + offset)
            )
            for ordinal, chapter_id in enumerate(ordered_ids, start=1):
                connection.execute(
                    update(chapters)
                    .where(chapters.c.id == chapter_id, chapters.c.book_id == book_id)
                    .values(ordinal=ordinal)
                )
            connection.execute(
                update(books)
                .where(
                    books.c.id == book_id,
                    books.c.chapter_order_revision == base_revision,
                )
                .values(
                    chapter_order_revision=base_revision + 1,
                    updated_at=_utcnow(),
                )
            )
            mark_book_insight_derived_stale(
                connection,
                book_id=book_id,
                now=_utcnow(),
            )
        return base_revision + 1

    def reorder_pages(
        self,
        *,
        chapter_id: str,
        ordered_ids: list[str],
        base_revision: int,
    ) -> int:
        with immediate_transaction(self.engine) as connection:
            chapter = connection.execute(
                select(
                    chapters.c.book_id,
                    chapters.c.page_order_revision,
                ).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter is None:
                raise ContentNotFound("chapter not found")
            if chapter["page_order_revision"] != base_revision:
                raise ContentConflict("page order revision changed")
            self._assert_chapter_writable(connection, chapter_id)
            existing = list(
                connection.execute(
                    select(pages.c.id)
                    .where(pages.c.chapter_id == chapter_id)
                    .order_by(pages.c.ordinal)
                ).scalars()
            )
            if len(ordered_ids) != len(set(ordered_ids)) or set(existing) != set(
                ordered_ids
            ):
                raise ValueError("ordered page ids must be an exact permutation")
            offset = len(existing) * 2 + 1
            connection.execute(
                update(pages)
                .where(pages.c.chapter_id == chapter_id)
                .values(ordinal=pages.c.ordinal + offset)
            )
            for ordinal, page_id in enumerate(ordered_ids, start=1):
                connection.execute(
                    update(pages)
                    .where(
                        pages.c.id == page_id,
                        pages.c.chapter_id == chapter_id,
                    )
                    .values(ordinal=ordinal)
                )
            connection.execute(
                update(chapters)
                .where(
                    chapters.c.id == chapter_id,
                    chapters.c.page_order_revision == base_revision,
                )
                .values(
                    page_order_revision=base_revision + 1,
                    updated_at=_utcnow(),
                )
            )
            mark_book_insight_derived_stale(
                connection,
                book_id=str(chapter["book_id"]),
                now=_utcnow(),
            )
        return base_revision + 1

    def delete_page(self, page_id: str) -> None:
        with immediate_transaction(self.engine) as connection:
            page = connection.execute(
                select(
                    pages.c.chapter_id,
                    pages.c.ordinal,
                    chapters.c.book_id,
                )
                .join(chapters, chapters.c.id == pages.c.chapter_id)
                .where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise ContentNotFound("page not found")
            chapter_id = str(page["chapter_id"])
            self._assert_targets_idle(
                connection,
                str(page["book_id"]),
                [chapter_id],
            )
            mark_book_insight_derived_stale(
                connection,
                book_id=str(page["book_id"]),
                now=_utcnow(),
            )
            connection.execute(delete(pages).where(pages.c.id == page_id))
            connection.execute(
                update(pages)
                .where(
                    pages.c.chapter_id == chapter_id,
                    pages.c.ordinal > page["ordinal"],
                )
                .values(ordinal=pages.c.ordinal - 1)
            )
            connection.execute(
                update(chapters)
                .where(chapters.c.id == chapter_id)
                .values(
                    page_order_revision=chapters.c.page_order_revision + 1,
                    updated_at=_utcnow(),
                )
            )

    def clear_chapter_pages(self, chapter_id: str) -> int:
        with immediate_transaction(self.engine) as connection:
            chapter = connection.execute(
                select(chapters.c.book_id).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter is None:
                raise ContentNotFound("chapter not found")
            self._assert_targets_idle(
                connection,
                str(chapter["book_id"]),
                [chapter_id],
            )
            has_pages = connection.execute(
                select(exists().where(pages.c.chapter_id == chapter_id))
            ).scalar_one()
            if has_pages:
                mark_book_insight_derived_stale(
                    connection,
                    book_id=str(chapter["book_id"]),
                    now=_utcnow(),
                )
            removed = connection.execute(
                delete(pages).where(pages.c.chapter_id == chapter_id)
            )
            deleted_count = int(removed.rowcount or 0)
            if deleted_count:
                connection.execute(
                    update(chapters)
                    .where(chapters.c.id == chapter_id)
                    .values(
                        page_order_revision=chapters.c.page_order_revision + 1,
                        updated_at=_utcnow(),
                    )
                )
            return deleted_count

    def replace_page_source(
        self,
        *,
        page_id: str,
        base_source_revision: int,
        source: AssetRecord,
        thumbnail: AssetRecord,
        idempotency_scope: str,
        idempotency_key: str,
        request_hash: str,
    ) -> tuple[dict[str, object], bool]:
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            replay = connection.execute(
                select(
                    idempotency_records.c.request_hash,
                    idempotency_records.c.response_json,
                    idempotency_records.c.expires_at,
                ).where(
                    idempotency_records.c.scope == idempotency_scope,
                    idempotency_records.c.key == idempotency_key,
                )
            ).mappings().one_or_none()
            if replay is not None:
                if replay["expires_at"] > now:
                    if replay["request_hash"] != request_hash:
                        raise IdempotencyConflict(
                            "Idempotency-Key was reused for different content"
                        )
                    return json.loads(replay["response_json"]), True
                connection.execute(
                    delete(idempotency_records).where(
                        idempotency_records.c.scope == idempotency_scope,
                        idempotency_records.c.key == idempotency_key,
                    )
                )
            page = connection.execute(
                select(
                    pages.c.chapter_id,
                    pages.c.source_revision,
                    pages.c.document_revision,
                    chapters.c.book_id,
                )
                .join(chapters, chapters.c.id == pages.c.chapter_id)
                .where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise ContentNotFound("page not found")
            if page["source_revision"] != base_source_revision:
                raise ContentConflict("page source revision changed")
            self._assert_chapter_writable(
                connection,
                str(page["chapter_id"]),
            )
            if connection.execute(
                select(operations.c.id).where(
                    operations.c.page_id == page_id,
                    operations.c.status.in_(ACTIVE_OPERATION_STATUSES),
                )
            ).scalar_one_or_none() is not None:
                raise ContentConflict("page has an active operation")
            new_source_revision = base_source_revision + 1
            new_document_revision = int(page["document_revision"]) + 1
            connection.execute(delete(bubbles).where(bubbles.c.page_id == page_id))
            connection.execute(
                delete(page_assets).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role.not_in(("source", "thumbnail_source")),
                )
            )
            for role, record, parent in (
                ("source", source, None),
                ("thumbnail_source", thumbnail, source.id),
            ):
                connection.execute(
                    update(page_assets)
                    .where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == role,
                    )
                    .values(
                        asset_id=record.id,
                        input_source_revision=new_source_revision,
                        input_document_revision=None,
                        parent_asset_id=parent,
                        producer_job_step_id=None,
                        producer_operation_id=None,
                        producer_render_request_id=None,
                    )
                )
            connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.source_revision == base_source_revision,
                )
                .values(
                    source_revision=new_source_revision,
                    document_revision=new_document_revision,
                    rendered_revision=None,
                    render_status="not_rendered",
                    detection_state="unprocessed",
                    updated_at=_utcnow(),
                )
            )
            mark_book_insight_derived_stale(
                connection,
                book_id=str(page["book_id"]),
                now=now,
            )
            result = {
                "pageId": page_id,
                "sourceRevision": new_source_revision,
                "documentRevision": new_document_revision,
                "sourceUrl": f"/api/v2/assets/{source.id}",
                "thumbnailSourceUrl": f"/api/v2/assets/{thumbnail.id}",
            }
            connection.execute(
                insert(idempotency_records).values(
                    scope=idempotency_scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    http_status=200,
                    response_json=_json(result),
                    resource_type="page",
                    resource_id=page_id,
                    expires_at=now.replace(microsecond=0) + timedelta(days=7),
                )
            )
        return result, False

    def append_page(
        self,
        *,
        chapter_id: str,
        requested_logical_path: str,
        text_style: dict[str, object],
        source: AssetRecord,
        thumbnail: AssetRecord,
        idempotency_scope: str,
        idempotency_key: str,
        request_hash: str,
    ) -> tuple[dict[str, object], bool]:
        logical_path = normalize_logical_path(requested_logical_path)
        now = _utcnow()
        with immediate_transaction(self.engine) as connection:
            replay = connection.execute(
                select(
                    idempotency_records.c.request_hash,
                    idempotency_records.c.http_status,
                    idempotency_records.c.response_json,
                    idempotency_records.c.expires_at,
                ).where(
                    idempotency_records.c.scope == idempotency_scope,
                    idempotency_records.c.key == idempotency_key,
                )
            ).mappings().one_or_none()
            if replay is not None:
                if replay["expires_at"] > now:
                    if replay["request_hash"] != request_hash:
                        raise IdempotencyConflict(
                            "Idempotency-Key was reused for different content"
                        )
                    return json.loads(replay["response_json"]), True
                connection.execute(
                    delete(idempotency_records).where(
                        idempotency_records.c.scope == idempotency_scope,
                        idempotency_records.c.key == idempotency_key,
                    )
                )

            chapter = connection.execute(
                select(
                    chapters.c.book_id,
                    chapters.c.page_order_revision,
                ).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter is None:
                raise ContentNotFound("chapter not found")
            chapter_revision = chapter["page_order_revision"]
            self._assert_chapter_writable(connection, chapter_id)
            existing_paths = set(
                connection.execute(
                    select(pages.c.logical_source_path).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalars()
            )
            final_logical_path = deduplicate_logical_path(logical_path, existing_paths)
            ordinal = (
                connection.execute(
                    select(func.coalesce(func.max(pages.c.ordinal), 0)).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalar_one()
                + 1
            )
            page_id = str(uuid.uuid4())
            default_font_id, style_defaults = validate_text_style_defaults(
                connection,
                text_style,
            )
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=chapter_id,
                    ordinal=ordinal,
                    logical_source_path=final_logical_path,
                    default_font_id=default_font_id,
                    page_style_defaults_json=_json(style_defaults),
                )
            )
            connection.execute(
                insert(page_assets),
                [
                    {
                        "page_id": page_id,
                        "role": "source",
                        "asset_id": source.id,
                        "input_source_revision": 1,
                        "parent_asset_id": None,
                    },
                    {
                        "page_id": page_id,
                        "role": "thumbnail_source",
                        "asset_id": thumbnail.id,
                        "input_source_revision": 1,
                        "parent_asset_id": source.id,
                    },
                ],
            )
            connection.execute(
                update(chapters)
                .where(
                    chapters.c.id == chapter_id,
                    chapters.c.page_order_revision == chapter_revision,
                )
                .values(
                    page_order_revision=chapter_revision + 1,
                    updated_at=now,
                )
            )
            mark_book_insight_derived_stale(
                connection,
                book_id=str(chapter["book_id"]),
                now=now,
            )
            response = {
                "page": {
                    "id": page_id,
                    "chapterId": chapter_id,
                    "ordinal": ordinal,
                    "logicalSourcePath": final_logical_path,
                    "sourceRevision": 1,
                    "documentRevision": 1,
                    "width": source.width,
                    "height": source.height,
                    "sourceUrl": f"/api/v2/assets/{source.id}",
                    "thumbnailSourceUrl": f"/api/v2/assets/{thumbnail.id}",
                    "translatedUrl": None,
                },
                "pageOrderRevision": chapter_revision + 1,
            }
            connection.execute(
                insert(idempotency_records).values(
                    scope=idempotency_scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    http_status=201,
                    response_json=_json(response),
                    resource_type="page",
                    resource_id=page_id,
                    expires_at=now.replace(microsecond=0) + timedelta(days=7),
                )
            )
        return response, False

    def replay_idempotency(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
    ) -> dict[str, object] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    idempotency_records.c.request_hash,
                    idempotency_records.c.response_json,
                ).where(
                    idempotency_records.c.scope == scope,
                    idempotency_records.c.key == key,
                    idempotency_records.c.expires_at > _utcnow(),
                )
            ).mappings().one_or_none()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise IdempotencyConflict("Idempotency-Key was reused for different content")
        return json.loads(row["response_json"])

    @staticmethod
    def _assert_chapter_writable(connection: object, chapter_id: str) -> None:
        intent = connection.execute(  # type: ignore[attr-defined]
            select(chapter_write_intents.c.chapter_id).where(
                chapter_write_intents.c.chapter_id == chapter_id
            )
        ).scalar_one_or_none()
        lock = connection.execute(  # type: ignore[attr-defined]
            select(chapter_write_locks.c.chapter_id).where(
                chapter_write_locks.c.chapter_id == chapter_id
            )
        ).scalar_one_or_none()
        if intent is not None or lock is not None:
            raise ContentLocked("chapter is reserved by backend work")

    def list_pages(
        self,
        *,
        chapter_id: str,
        after_ordinal: int = 0,
        limit: int = 50,
        all_pages: bool = False,
    ) -> dict[str, object]:
        if limit < 1 or limit > 200:
            raise ValueError("page limit must be between 1 and 200")
        statement = self._page_summary_statement().where(
            pages.c.chapter_id == chapter_id,
            pages.c.ordinal > after_ordinal,
        )
        statement = statement.order_by(pages.c.ordinal)
        if not all_pages:
            statement = statement.limit(limit + 1)
        with self.engine.connect() as connection:
            chapter_revision = connection.execute(
                select(chapters.c.page_order_revision).where(
                    chapters.c.id == chapter_id
                )
            ).scalar_one_or_none()
            if chapter_revision is None:
                raise ContentNotFound("chapter not found")
            rows = list(connection.execute(statement).mappings())
        has_more = not all_pages and len(rows) > limit
        visible = rows[:limit] if has_more else rows
        return {
            "items": [self._page_summary(row) for row in visible],
            "nextCursor": visible[-1]["ordinal"] if has_more and visible else None,
            "pageOrderRevision": chapter_revision,
        }

    def get_page_summary(self, page_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            row = connection.execute(
                self._page_summary_statement().where(pages.c.id == page_id)
            ).mappings().one_or_none()
        if row is None:
            raise ContentNotFound("page not found")
        return self._page_summary(row)

    def get_page_document(self, page_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            return self._get_page_document(connection, page_id)

    def mutate_page_document(
        self,
        *,
        page_id: str,
        base_revision: int,
        mutations: list[dict[str, object]],
        idempotency_key: str,
        default_font_id: object = _MISSING,
        page_style_defaults_patch: dict[str, object] | None = None,
        propagate_style_fields: list[str] | None = None,
    ) -> tuple[dict[str, object], bool]:
        style_patch = dict(page_style_defaults_patch or {})
        propagation = list(propagate_style_fields or [])
        if (
            not mutations
            and default_font_id is _MISSING
            and not style_patch
            and not propagation
        ):
            raise ValueError(
                "command must mutate bubbles, the default font, or page style"
            )
        normalized_mutations: list[dict[str, object]] = []
        for mutation in mutations:
            if not isinstance(mutation, dict):
                raise ValueError("each bubble mutation must be an object")
            normalized = dict(mutation)
            if isinstance(normalized.get("fields"), dict):
                normalized["fields"] = dict(normalized["fields"])  # type: ignore[arg-type]
            normalized_mutations.append(normalized)
        mutations = normalized_mutations
        style_patch = validate_page_style(style_patch, partial=True)
        if len(propagation) != len(set(propagation)):
            raise ValueError("propagateStyleFields must contain unique fields")
        allowed_propagation = frozenset(
            {
                "fontSize",
                "fontFamily",
                "layoutDirection",
                "textColor",
                "fillColor",
                "strokeEnabled",
                "strokeColor",
                "strokeWidth",
                "lineSpacing",
                "textAlign",
            }
        )
        unknown_propagation = set(propagation) - allowed_propagation
        if unknown_propagation:
            raise ValueError(
                "unknown propagateStyleFields: "
                + ", ".join(sorted(unknown_propagation))
            )
        if "fontFamily" in propagation and default_font_id is _MISSING:
            raise ValueError(
                "fontFamily propagation requires defaultFontId"
            )
        propagated_bubble_fields = {
            "fontSize": "fontSize",
            "fontFamily": "fontId",
            "layoutDirection": "textDirection",
            "textColor": "textColor",
            "fillColor": "fillColor",
            "strokeEnabled": "strokeEnabled",
            "strokeColor": "strokeColor",
            "strokeWidth": "strokeWidth",
            "lineSpacing": "lineSpacing",
            "textAlign": "textAlign",
        }
        propagated_targets = {
            propagated_bubble_fields[field] for field in propagation
        }
        seen_client_mutation_ids: set[str] = set()
        mutation_keys = {"op", "clientMutationId", "bubbleId", "fields"}
        for mutation in mutations:
            unknown_keys = set(mutation) - mutation_keys
            if unknown_keys:
                raise ValueError(
                    "unknown bubble mutation fields: "
                    + ", ".join(sorted(unknown_keys))
                )
            operation = mutation.get("op")
            if operation not in {"create", "patch", "delete", "reset"}:
                raise ValueError("bubble mutation op is invalid")
            client_mutation_id = mutation.get("clientMutationId")
            if (
                not isinstance(client_mutation_id, str)
                or not client_mutation_id.strip()
            ):
                raise ValueError("clientMutationId must be a non-empty string")
            if client_mutation_id in seen_client_mutation_ids:
                raise ValueError("clientMutationId must be unique within the batch")
            seen_client_mutation_ids.add(client_mutation_id)
            if operation == "create":
                if "bubbleId" in mutation:
                    raise ValueError("create mutations cannot provide bubbleId")
            else:
                bubble_id = mutation.get("bubbleId")
                if not isinstance(bubble_id, str) or not bubble_id:
                    raise ValueError("bubbleId is required for non-create mutations")
                try:
                    uuid.UUID(bubble_id)
                except ValueError as exc:
                    raise ValueError("bubbleId must be a UUID") from exc
            fields = mutation.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError("mutation fields must be an object")
            non_payload_fields = {
                "backendBubbleId",
                "bubbleId",
                "clientMutationId",
                "fontFamily",
                "ordinal",
            }.intersection(fields)
            if non_payload_fields:
                raise ValueError(
                    "fields contains non-payload identities: "
                    + ", ".join(sorted(non_payload_fields))
                )
            unknown_fields = set(fields) - _CLIENT_BUBBLE_MUTATION_FIELDS
            if unknown_fields:
                raise ValueError(
                    "unknown bubble fields: "
                    + ", ".join(sorted(unknown_fields))
                )
            if operation == "delete":
                if fields:
                    raise ValueError("delete mutations cannot provide fields")
            else:
                payload_fields = {
                    key: value for key, value in fields.items() if key != "fontId"
                }
                payload_fields = validate_bubble_payload(
                    payload_fields,
                    render=False,
                    partial=True,
                )
                if "coords" in payload_fields:
                    payload_fields["autoTextDirection"] = (
                        _auto_text_direction_from_coords(payload_fields["coords"])
                    )
                if operation != "patch":
                    payload_fields = validate_bubble_payload(
                        payload_fields,
                        render=False,
                    )
                if operation == "patch" and not fields:
                    raise ValueError("patch mutations must provide fields")
                mutation["fields"] = {
                    **payload_fields,
                    **({"fontId": fields["fontId"]} if "fontId" in fields else {}),
                }
            conflicts = (
                propagated_targets.intersection(fields)
                if operation == "patch"
                else set()
            )
            if conflicts:
                raise ValueError(
                    "bubble mutation conflicts with propagated style fields: "
                    + ", ".join(sorted(conflicts))
                )
        request_payload = {
            "baseRevision": base_revision,
            "mutations": mutations,
            "defaultFontId": (
                default_font_id if default_font_id is not _MISSING else "__missing__"
            ),
            "pageStyleDefaultsPatch": style_patch,
            "propagateStyleFields": propagation,
        }
        request_hash = hashlib.sha256(
            _json(request_payload).encode("utf-8")
        ).hexdigest()
        scope = f"page-document:{page_id}"
        now = _utcnow()
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
                    raise IdempotencyConflict(
                        "Idempotency-Key was reused for a different page mutation"
                    )
                return json.loads(replay["response_json"]), True
            page = connection.execute(
                select(
                    pages.c.chapter_id,
                    pages.c.document_revision,
                    pages.c.rendered_revision,
                    pages.c.render_status,
                    pages.c.default_font_id,
                    pages.c.page_style_defaults_json,
                    pages.c.page_style_schema_version,
                ).where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise ContentNotFound("page not found")
            if page["page_style_schema_version"] != PAGE_STYLE_SCHEMA_VERSION:
                raise ValueError("page style schema version is not current")
            if page["document_revision"] != base_revision:
                raise ContentConflict("page document revision changed")
            self._assert_chapter_writable(connection, str(page["chapter_id"]))
            if connection.execute(
                select(operations.c.id).where(
                    operations.c.page_id == page_id,
                    operations.c.status.in_(ACTIVE_OPERATION_STATUSES),
                )
            ).scalar_one_or_none() is not None:
                raise ContentConflict("page has an active operation")
            existing_rows = list(
                connection.execute(
                    select(
                        bubbles.c.id,
                        bubbles.c.ordinal,
                        bubbles.c.font_id,
                        bubbles.c.payload_json,
                    )
                    .where(bubbles.c.page_id == page_id)
                    .order_by(bubbles.c.ordinal)
                ).mappings()
            )
            documents: dict[str, dict[str, object]] = {
                str(row["id"]): {
                    "bubbleId": str(row["id"]),
                    "fontId": row["font_id"],
                    "payload": validate_bubble_payload(
                        json.loads(row["payload_json"]),
                        render=False,
                    ),
                }
                for row in existing_rows
            }
            order = [str(row["id"]) for row in existing_rows]
            deleted_ids: set[str] = set()
            created_ids: set[str] = set()
            mutation_results: list[dict[str, str]] = []
            renderable_change = False
            validated_font_ids: set[str] = set()

            def validate_font_reference(value: object, *, field: str) -> None:
                if value is None:
                    return
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{field} must be a non-empty string or null")
                if value in validated_font_ids:
                    return
                if connection.execute(
                    select(fonts.c.id).where(fonts.c.id == value)
                ).scalar_one_or_none() is None:
                    raise ContentNotFound("font not found")
                validated_font_ids.add(value)

            if default_font_id is not _MISSING:
                validate_font_reference(default_font_id, field="defaultFontId")
            current_style = validate_page_style(
                json.loads(page["page_style_defaults_json"]),
                partial=False,
            )
            if style_patch:
                current_style.update(style_patch)
            has_current_translated = (
                connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none()
                is not None
            )
            has_active_render = (
                connection.execute(
                    select(render_requests.c.id).where(
                        render_requests.c.page_id == page_id,
                        render_requests.c.status.in_(
                            ACTIVE_RENDER_REQUEST_STATUSES
                        ),
                    )
                ).scalar_one_or_none()
                is not None
            )
            calculate_auto_font_size = None
            if "fontSize" in propagation and bool(
                current_style["autoFontSize"]
            ):
                from src.core.rendering import calculate_auto_font_size
            for mutation in mutations:
                operation = str(mutation["op"])
                client_mutation_id = str(mutation["clientMutationId"])
                fields = mutation.get("fields", {})
                if operation == "create":
                    bubble_id = str(uuid.uuid4())
                    validate_font_reference(fields.get("fontId"), field="fontId")
                    documents[bubble_id] = {
                        "bubbleId": bubble_id,
                        "fontId": fields.get("fontId"),
                        "payload": {
                            key: value
                            for key, value in fields.items()
                            if key != "fontId"
                        },
                    }
                    order.append(bubble_id)
                    created_ids.add(bubble_id)
                    mutation_results.append(
                        {
                            "op": operation,
                            "clientMutationId": client_mutation_id,
                            "bubbleId": bubble_id,
                        }
                    )
                    renderable_change = True
                    continue
                bubble_id = str(mutation["bubbleId"])
                if bubble_id not in documents:
                    raise ContentNotFound(f"bubble {bubble_id} not found")
                if operation == "delete":
                    documents.pop(bubble_id)
                    order.remove(bubble_id)
                    deleted_ids.add(bubble_id)
                    mutation_results.append(
                        {
                            "op": operation,
                            "clientMutationId": client_mutation_id,
                            "bubbleId": bubble_id,
                        }
                    )
                    renderable_change = True
                    continue
                current = documents[bubble_id]
                before_font_id = current["fontId"]
                before_payload = dict(current["payload"])  # type: ignore[arg-type]
                if "fontId" in fields:
                    validate_font_reference(fields["fontId"], field="fontId")
                    current["fontId"] = fields["fontId"]
                payload_fields = {
                    key: value
                    for key, value in fields.items()
                    if key != "fontId"
                }
                if operation == "reset":
                    current["payload"] = payload_fields
                else:
                    current_payload = dict(before_payload)
                    current_payload.update(payload_fields)
                    current["payload"] = current_payload
                renderable_change = renderable_change or (
                    current["fontId"] != before_font_id
                    or self._render_payload_changed(
                        before_payload,
                        dict(current["payload"]),  # type: ignore[arg-type]
                    )
                )
                mutation_results.append(
                    {
                        "op": operation,
                        "clientMutationId": client_mutation_id,
                        "bubbleId": bubble_id,
                    }
                )

            if propagation:
                propagation_order = (
                    "fontFamily",
                    "layoutDirection",
                    "textColor",
                    "fillColor",
                    "strokeEnabled",
                    "strokeColor",
                    "strokeWidth",
                    "lineSpacing",
                    "textAlign",
                    "fontSize",
                )
                for document in documents.values():
                    payload = dict(document["payload"])  # type: ignore[arg-type]
                    for field in propagation_order:
                        if field not in propagation:
                            continue
                        if field == "fontFamily":
                            value = default_font_id
                            if document["fontId"] != value:
                                renderable_change = True
                            document["fontId"] = value
                            continue
                        if field == "layoutDirection":
                            value = current_style[field]
                            direction = (
                                payload["autoTextDirection"]
                                if value == "auto"
                                else value
                            )
                            if payload.get("textDirection") != direction:
                                renderable_change = True
                            payload["textDirection"] = direction
                        elif field in {"textColor", "fillColor"}:
                            materialized: object | None = None
                            if bool(current_style["useAutoTextColor"]):
                                automatic_field = (
                                    "autoFgColor"
                                    if field == "textColor"
                                    else "autoBgColor"
                                )
                                automatic = payload[automatic_field]
                                if automatic is not None:
                                    materialized = rgb_to_hex(automatic)
                            elif field in style_patch:
                                materialized = current_style[field]
                            if (
                                materialized is not None
                                and payload.get(field) != materialized
                            ):
                                if field == "textColor":
                                    renderable_change = True
                                payload[field] = materialized
                        elif field == "fontSize":
                            if (
                                bool(current_style["autoFontSize"])
                                and calculate_auto_font_size is not None
                            ):
                                coords = payload["coords"]
                                text = payload["translatedText"]
                                if text.strip():
                                    font_id = document["fontId"] or (
                                        default_font_id
                                        if default_font_id is not _MISSING
                                        else page["default_font_id"]
                                    )
                                    font_path = (
                                        resolve_font_path(
                                            connection,
                                            self.storage,
                                            str(font_id) if font_id else None,
                                        )
                                        if self.storage is not None
                                        else None
                                    )
                                    if font_path is None:
                                        from src.shared import constants

                                        font_path = constants.DEFAULT_FONT_RELATIVE_PATH
                                    calculated = calculate_auto_font_size(
                                        text,
                                        coords[2] - coords[0],
                                        coords[3] - coords[1],
                                        payload["textDirection"],
                                        font_path,
                                    )
                                    if payload.get("fontSize") != calculated:
                                        renderable_change = True
                                    payload["fontSize"] = calculated
                            else:
                                value = current_style["fontSize"]
                                if payload.get("fontSize") != value:
                                    renderable_change = True
                                payload["fontSize"] = value
                        else:
                            value = current_style[field]
                            if (
                                field != "fillColor"
                                and payload.get(field) != value
                            ):
                                renderable_change = True
                            payload[field] = value
                    document["payload"] = payload

            for document in documents.values():
                document["payload"] = validate_bubble_payload(
                    document["payload"],
                    render=False,
                )

            new_revision = base_revision + 1
            if existing_rows:
                offset = len(existing_rows) * 2 + len(created_ids) + 1
                connection.execute(
                    update(bubbles)
                    .where(bubbles.c.page_id == page_id)
                    .values(ordinal=bubbles.c.ordinal + offset)
                )
            if deleted_ids:
                connection.execute(
                    delete(bubbles).where(
                        bubbles.c.page_id == page_id,
                        bubbles.c.id.in_(deleted_ids),
                    )
                )
            for ordinal, bubble_id in enumerate(order, start=1):
                document = documents[bubble_id]
                values = {
                    "ordinal": ordinal,
                    "font_id": document["fontId"],
                    "payload_json": _json(document["payload"]),
                    "updated_revision": new_revision,
                    "updated_at": _utcnow(),
                }
                if bubble_id in created_ids:
                    connection.execute(
                        insert(bubbles).values(
                            id=bubble_id,
                            page_id=page_id,
                            **values,
                        )
                    )
                else:
                    connection.execute(
                        update(bubbles)
                        .where(
                            bubbles.c.id == bubble_id,
                            bubbles.c.page_id == page_id,
                        )
                        .values(**values)
                    )
            has_drawable_text = any(
                document["payload"]["translatedText"].strip()
                for document in documents.values()
            )
            needs_render = renderable_change and (
                has_current_translated or has_drawable_text
            )
            repair_blocks_render = page["render_status"] in {
                "awaiting_repair",
                "repair_failed",
            }
            advance_render_request = not repair_blocks_render and (
                needs_render or has_active_render
            )
            page_values: dict[str, object] = {
                "document_revision": new_revision,
                "updated_at": _utcnow(),
            }
            if default_font_id is not _MISSING:
                page_values["default_font_id"] = default_font_id
            if style_patch:
                page_values["page_style_defaults_json"] = _json(current_style)
            if advance_render_request:
                page_values["render_status"] = "stale"
            elif (
                page["render_status"] == "ready"
                and page["rendered_revision"] == base_revision
            ):
                page_values["rendered_revision"] = new_revision
                connection.execute(
                    update(page_assets)
                    .where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                        page_assets.c.input_document_revision == base_revision,
                    )
                    .values(input_document_revision=new_revision)
                )
            changed = connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == base_revision,
                )
                .values(**page_values)
            )
            if changed.rowcount != 1:
                raise ContentConflict("page document revision changed")
            if advance_render_request:
                from src.backend_v2.operations.repository import (
                    RenderRequestRepository,
                )

                RenderRequestRepository(self.engine).upsert(
                    connection,
                    page_id=page_id,
                    requested_revision=new_revision,
                )
            result = {
                "document": self._get_page_document(connection, page_id),
                "mutationResults": mutation_results,
            }
            connection.execute(
                insert(idempotency_records).values(
                    scope=scope,
                    key=idempotency_key,
                    request_hash=request_hash,
                    http_status=200,
                    response_json=_json(result),
                    resource_type="page_document",
                    resource_id=page_id,
                    expires_at=now + timedelta(hours=24),
                )
            )
        return result, False

    @staticmethod
    def _get_page_document(
        connection: Any,
        page_id: str,
    ) -> dict[str, object]:
        page = connection.execute(
            select(
                pages.c.id,
                pages.c.chapter_id,
                pages.c.document_revision,
                pages.c.render_status,
                pages.c.default_font_id,
                pages.c.page_style_defaults_json,
                pages.c.page_style_schema_version,
            ).where(pages.c.id == page_id)
        ).mappings().one_or_none()
        if page is None:
            raise ContentNotFound("page not found")
        if page["page_style_schema_version"] != PAGE_STYLE_SCHEMA_VERSION:
            raise ValueError("page style schema version is not current")
        page_style = validate_page_style(
            json.loads(page["page_style_defaults_json"]),
            partial=False,
        )
        bubble_rows = list(
            connection.execute(
                select(
                    bubbles.c.id,
                    bubbles.c.ordinal,
                    bubbles.c.font_id,
                    bubbles.c.payload_json,
                    bubbles.c.updated_revision,
                )
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).mappings()
        )
        bubble_documents: list[dict[str, object]] = []
        for row in bubble_rows:
            payload = validate_bubble_payload(
                json.loads(row["payload_json"]),
                render=False,
            )
            bubble_documents.append(
                {
                    "bubbleId": row["id"],
                    "ordinal": row["ordinal"],
                    "fontId": row["font_id"],
                    "payload": payload,
                    "updatedRevision": row["updated_revision"],
                }
            )
        return {
            "pageId": page["id"],
            "chapterId": page["chapter_id"],
            "documentRevision": page["document_revision"],
            "renderStatus": page["render_status"],
            "defaultFontId": page["default_font_id"],
            "pageStyleDefaults": page_style,
            "pageStyleSchemaVersion": page["page_style_schema_version"],
            "bubbles": bubble_documents,
        }

    @staticmethod
    def _render_payload_changed(
        before: dict[str, object],
        after: dict[str, object],
    ) -> bool:
        return any(
            before.get(field) != after.get(field)
            for field in (
                "translatedText",
                "coords",
                "fontSize",
                "textDirection",
                "textColor",
                "rotationAngle",
                "position",
                "strokeEnabled",
                "strokeColor",
                "strokeWidth",
                "lineSpacing",
                "textAlign",
            )
        )

    @staticmethod
    def _page_summary_statement():
        asset_aliases = {
            role: page_assets.alias(name=f"pa_{role}")
            for role in (
                "source",
                "thumbnail_source",
                "clean",
                "translated",
            )
        }
        statement = select(
            pages,
            *[
                alias.c.asset_id.label(f"{role}_asset_id")
                for role, alias in asset_aliases.items()
            ],
            assets.c.width.label("source_width"),
            assets.c.height.label("source_height"),
        )
        for role, alias in asset_aliases.items():
            statement = statement.outerjoin(
                alias,
                and_(alias.c.page_id == pages.c.id, alias.c.role == role),
            )
        return statement.outerjoin(
            assets,
            assets.c.id == asset_aliases["source"].c.asset_id,
        )

    @staticmethod
    def _page_summary(row: dict[str, Any]) -> dict[str, object]:
        def url(role: str) -> str | None:
            asset_id = row.get(f"{role}_asset_id")
            return f"/api/v2/assets/{asset_id}" if asset_id else None

        return {
            "id": row["id"],
            "chapterId": row["chapter_id"],
            "ordinal": row["ordinal"],
            "logicalSourcePath": row["logical_source_path"],
            "width": row.get("source_width"),
            "height": row.get("source_height"),
            "sourceRevision": row["source_revision"],
            "documentRevision": row["document_revision"],
            "renderedRevision": row["rendered_revision"],
            "renderStatus": row["render_status"],
            "detectionState": row["detection_state"],
            "sourceUrl": url("source"),
            "thumbnailSourceUrl": url("thumbnail_source"),
            "cleanUrl": url("clean"),
            "translatedUrl": url("translated"),
        }

    def reset_quick_workspace(self) -> dict[str, str]:
        with immediate_transaction(self.engine) as connection:
            book = connection.execute(
                select(books.c.id).where(books.c.kind == "quick_workspace")
            ).scalar_one_or_none()
            if book is None:
                connection.execute(
                    insert(books).values(
                        id=QUICK_WORKSPACE_BOOK_ID,
                        kind="quick_workspace",
                        title="快速翻译",
                    )
                )
                book = QUICK_WORKSPACE_BOOK_ID
            chapter_ids = list(
                connection.execute(
                    select(chapters.c.id).where(chapters.c.book_id == book)
                ).scalars()
            )
            self._assert_targets_idle(connection, book, chapter_ids)
            connection.execute(delete(chapters).where(chapters.c.book_id == book))
            connection.execute(
                delete(translation_constraints).where(
                    translation_constraints.c.book_id == book
                )
            )
            chapter_id = str(uuid.uuid4())
            connection.execute(
                insert(chapters).values(
                    id=chapter_id,
                    book_id=book,
                    ordinal=1,
                    title="快速翻译",
                )
            )
            connection.execute(
                insert(translation_constraints).values(
                    book_id=book,
                    payload_json=_json(empty_translation_constraints()),
                    schema_version=TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
                )
            )
            connection.execute(
                update(books)
                .where(books.c.id == book)
                .values(
                    chapter_order_revision=books.c.chapter_order_revision + 1,
                    updated_at=_utcnow(),
                )
            )
        return {"bookId": str(book), "chapterId": chapter_id}

    def promote_quick_workspace(
        self,
        *,
        chapter_title: str,
        new_book_title: str | None = None,
        target_book_id: str | None = None,
    ) -> dict[str, str]:
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title or len(normalized_chapter_title) > 500:
            raise ValueError("chapter title must contain 1-500 characters")
        if (new_book_title is None) == (target_book_id is None):
            raise ValueError("choose exactly one target: new book or existing book")
        normalized_new_book_title: str | None = None
        normalized_target_book_id: str | None = None
        if new_book_title is not None:
            normalized_new_book_title = new_book_title.strip()
            if not normalized_new_book_title or len(normalized_new_book_title) > 500:
                raise ValueError("new book title must contain 1-500 characters")
        else:
            assert target_book_id is not None
            normalized_target_book_id = target_book_id.strip()
            if not normalized_target_book_id:
                raise ValueError("target book ID is required")
        with immediate_transaction(self.engine) as connection:
            quick_book_id = connection.execute(
                select(books.c.id).where(books.c.kind == "quick_workspace")
            ).scalar_one_or_none()
            if quick_book_id is None:
                raise ContentNotFound("quick workspace not found")
            quick_chapters = list(
                connection.execute(
                    select(chapters.c.id)
                    .where(chapters.c.book_id == quick_book_id)
                    .order_by(chapters.c.ordinal)
                ).scalars()
            )
            if len(quick_chapters) != 1:
                raise ContentConflict("quick workspace must contain exactly one chapter")
            source_chapter_id = str(quick_chapters[0])
            self._assert_targets_idle(
                connection,
                str(quick_book_id),
                [source_chapter_id],
            )

            if normalized_new_book_title is not None:
                duplicate_book = connection.execute(
                    select(books.c.id).where(
                        books.c.kind == "library",
                        func.lower(books.c.title)
                        == normalized_new_book_title.lower(),
                    )
                ).scalar_one_or_none()
                if duplicate_book is not None:
                    raise ValueError("new book title already exists")
                destination_book_id = str(uuid.uuid4())
                connection.execute(
                    insert(books).values(
                        id=destination_book_id,
                        kind="library",
                        title=normalized_new_book_title,
                    )
                )
                source_constraints = connection.execute(
                    select(
                        translation_constraints.c.payload_json,
                        translation_constraints.c.schema_version,
                    ).where(
                        translation_constraints.c.book_id == quick_book_id
                    )
                ).mappings().one_or_none()
                if source_constraints is None:
                    raise ContentNotFound("translation constraints not found")
                if (
                    source_constraints["schema_version"]
                    != TRANSLATION_CONSTRAINTS_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "translation constraints schema version is not current"
                    )
                source_constraint_payload = validate_translation_constraints(
                    json.loads(source_constraints["payload_json"])
                )
                connection.execute(
                    insert(translation_constraints).values(
                        book_id=destination_book_id,
                        payload_json=_json(source_constraint_payload),
                        schema_version=TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
                    )
                )
            else:
                assert normalized_target_book_id is not None
                destination_book_id = normalized_target_book_id
                destination = connection.execute(
                    select(
                        books.c.kind,
                        translation_constraints.c.schema_version,
                    )
                    .outerjoin(
                        translation_constraints,
                        translation_constraints.c.book_id == books.c.id,
                    )
                    .where(books.c.id == destination_book_id)
                ).mappings().one_or_none()
                if destination is None or destination["kind"] != "library":
                    raise ContentNotFound("target library book not found")
                if (
                    destination["schema_version"]
                    != TRANSLATION_CONSTRAINTS_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "translation constraints schema version is not current"
                    )
                duplicate_chapter = connection.execute(
                    select(chapters.c.id).where(
                        chapters.c.book_id == destination_book_id,
                        func.lower(chapters.c.title)
                        == normalized_chapter_title.lower(),
                    )
                ).scalar_one_or_none()
                if duplicate_chapter is not None:
                    raise ValueError("chapter title already exists in target book")

            destination_ordinal = (
                connection.execute(
                    select(func.coalesce(func.max(chapters.c.ordinal), 0)).where(
                        chapters.c.book_id == destination_book_id
                    )
                ).scalar_one()
                + 1
            )
            connection.execute(
                update(chapters)
                .where(chapters.c.id == source_chapter_id)
                .values(
                    book_id=destination_book_id,
                    ordinal=destination_ordinal,
                    title=normalized_chapter_title,
                    updated_at=_utcnow(),
                )
            )
            connection.execute(
                update(books)
                .where(books.c.id == destination_book_id)
                .values(
                    chapter_order_revision=books.c.chapter_order_revision + 1,
                    updated_at=_utcnow(),
                )
            )
            connection.execute(
                delete(translation_constraints).where(
                    translation_constraints.c.book_id == quick_book_id
                )
            )
            connection.execute(
                insert(translation_constraints).values(
                    book_id=quick_book_id,
                    payload_json=_json(empty_translation_constraints()),
                    schema_version=TRANSLATION_CONSTRAINTS_SCHEMA_VERSION,
                )
            )
            new_quick_chapter_id = str(uuid.uuid4())
            connection.execute(
                insert(chapters).values(
                    id=new_quick_chapter_id,
                    book_id=quick_book_id,
                    ordinal=1,
                    title="快速翻译",
                )
            )
            connection.execute(
                update(books)
                .where(books.c.id == quick_book_id)
                .values(
                    chapter_order_revision=books.c.chapter_order_revision + 1,
                    updated_at=_utcnow(),
                )
            )
        return {
            "bookId": destination_book_id,
            "chapterId": source_chapter_id,
            "quickChapterId": new_quick_chapter_id,
        }

    @staticmethod
    def _assert_targets_idle(
        connection: object,
        book_id: str,
        chapter_ids: list[str],
    ) -> None:
        session_document = studio_documents.alias(
            "content_guard_session_document"
        )
        active_job = connection.execute(  # type: ignore[attr-defined]
            select(jobs.c.id).where(
                jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                or_(
                    jobs.c.book_id == book_id,
                    jobs.c.chapter_id.in_(chapter_ids) if chapter_ids else False,
                ),
            ).limit(1)
        ).scalar_one_or_none()
        active_operation = connection.execute(  # type: ignore[attr-defined]
            select(operations.c.id)
            .join(pages, pages.c.id == operations.c.page_id, isouter=True)
            .join(
                studio_documents,
                studio_documents.c.id == operations.c.studio_document_id,
                isouter=True,
            )
            .join(
                studio_chat_sessions,
                studio_chat_sessions.c.id == operations.c.studio_session_id,
                isouter=True,
            )
            .join(
                session_document,
                session_document.c.id == studio_chat_sessions.c.document_id,
                isouter=True,
            )
            .where(
                operations.c.status.in_(ACTIVE_OPERATION_STATUSES),
                or_(
                    (
                        pages.c.chapter_id.in_(chapter_ids)
                        if chapter_ids
                        else False
                    ),
                    studio_documents.c.book_id == book_id,
                    session_document.c.book_id == book_id,
                ),
            ).limit(1)
        ).scalar_one_or_none()
        if active_job or active_operation:
            raise ContentLocked("quick workspace is still referenced by active work")

    @staticmethod
    def _replace_book_tags(
        connection: object,
        book_id: str,
        tag_ids: list[str],
    ) -> None:
        if len(tag_ids) != len(set(tag_ids)):
            raise ValueError("tagIds must contain unique IDs")
        if tag_ids:
            found = set(
                connection.execute(  # type: ignore[attr-defined]
                    select(tags.c.id).where(tags.c.id.in_(tag_ids))
                ).scalars()
            )
            if found != set(tag_ids):
                raise ContentNotFound("tag not found")
        connection.execute(  # type: ignore[attr-defined]
            delete(book_tags).where(book_tags.c.book_id == book_id)
        )
        if tag_ids:
            connection.execute(  # type: ignore[attr-defined]
                insert(book_tags),
                [
                    {"book_id": book_id, "tag_id": tag_id}
                    for tag_id in tag_ids
                ],
            )

    @staticmethod
    def _normalize_color(value: str) -> str:
        normalized = value.strip()
        if (
            len(normalized) != 7
            or not normalized.startswith("#")
            or any(character not in "0123456789abcdefABCDEF" for character in normalized[1:])
        ):
            raise ValueError("tag color must be a six-digit hexadecimal color")
        return normalized.lower()
