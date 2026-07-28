"""Relational content repository with revision CAS and atomic ordinals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import secrets
from typing import Any
import uuid

from sqlalchemy import Engine, and_, delete, func, insert, or_, select, update

from src.backend_v2.storage.assets import AssetRecord
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    book_tags,
    books,
    bubbles,
    chapter_write_intents,
    chapter_write_locks,
    chapters,
    idempotency_records,
    import_leases,
    jobs,
    operations,
    page_assets,
    pages,
    tags,
    translation_constraints,
)
from src.backend_v2.storage.seeding import (
    QUICK_WORKSPACE_BOOK_ID,
    QUICK_WORKSPACE_CHAPTER_ID,
)


NONTERMINAL_JOB_STATUSES = (
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "interrupted",
)
ACTIVE_OPERATION_STATUSES = ("pending", "running")


class ContentNotFound(LookupError):
    pass


class ContentConflict(RuntimeError):
    pass


class ContentLocked(RuntimeError):
    pass


class IdempotencyConflict(ContentConflict):
    pass


@dataclass(frozen=True, slots=True)
class ImportLease:
    id: str
    owner_token: str
    expires_at: datetime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _deduplicate_logical_path(requested: str, existing: set[str]) -> str:
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


class ContentRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def list_books(
        self,
        *,
        search: str = "",
        tag_ids: tuple[str, ...] = (),
    ) -> list[dict[str, object]]:
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
            .order_by(books.c.updated_at.desc(), books.c.id)
        )
        if search:
            statement = statement.where(books.c.title.ilike(f"%{search.strip()}%"))
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
                ).mappings()
            ) if rows else []
        tags_by_book: dict[str, list[dict[str, str]]] = {}
        for row in tag_rows:
            tags_by_book.setdefault(str(row["book_id"]), []).append(
                {"id": str(row["id"]), "name": str(row["name"]), "color": str(row["color"])}
            )
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
                "chapterOrderRevision": row["chapter_order_revision"],
                "createdAt": row["created_at"].isoformat(),
                "updatedAt": row["updated_at"].isoformat(),
            }
            for row in rows
        ]

    def create_book(self, *, title: str) -> dict[str, object]:
        normalized = title.strip()
        if not normalized or len(normalized) > 500:
            raise ValueError("book title must contain 1-500 characters")
        book_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                insert(books).values(id=book_id, kind="library", title=normalized)
            )
            connection.execute(
                insert(translation_constraints).values(
                    book_id=book_id,
                    payload_json='{"glossary":[],"nonTranslate":[]}',
                )
            )
        return {"id": book_id, "title": normalized, "chapterOrderRevision": 1}

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
        return base_revision + 1

    def append_page(
        self,
        *,
        chapter_id: str,
        requested_logical_path: str,
        source: AssetRecord,
        thumbnail: AssetRecord,
        idempotency_scope: str,
        idempotency_key: str,
        request_hash: str,
        lease_id: str,
        owner_token: str,
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
                select(chapters.c.page_order_revision).where(
                    chapters.c.id == chapter_id
                )
            ).scalar_one_or_none()
            if chapter is None:
                raise ContentNotFound("chapter not found")
            self._assert_chapter_writable(connection, chapter_id)
            token_hash = hashlib.sha256(owner_token.encode("utf-8")).hexdigest()
            lease_expiry = now + timedelta(seconds=60)
            renewed = connection.execute(
                update(import_leases)
                .where(
                    import_leases.c.id == lease_id,
                    import_leases.c.chapter_id == chapter_id,
                    import_leases.c.owner_token_hash == token_hash,
                    import_leases.c.expires_at > now,
                )
                .values(last_activity_at=now, expires_at=lease_expiry)
            )
            if renewed.rowcount != 1:
                raise ContentLocked(
                    "import lease is missing, expired, or owned by another client"
                )
            existing_paths = set(
                connection.execute(
                    select(pages.c.logical_source_path).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalars()
            )
            final_logical_path = _deduplicate_logical_path(logical_path, existing_paths)
            ordinal = (
                connection.execute(
                    select(func.coalesce(func.max(pages.c.ordinal), 0)).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalar_one()
                + 1
            )
            page_id = str(uuid.uuid4())
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=chapter_id,
                    ordinal=ordinal,
                    logical_source_path=final_logical_path,
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
                    chapters.c.page_order_revision == chapter,
                )
                .values(
                    page_order_revision=chapter + 1,
                    updated_at=now,
                )
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
                    "thumbnailTranslatedUrl": None,
                },
                "pageOrderRevision": chapter + 1,
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
        asset_aliases = {}
        for role in (
            "source",
            "thumbnail_source",
            "clean",
            "translated",
            "thumbnail_translated",
        ):
            asset_aliases[role] = page_assets.alias(name=f"pa_{role}")

        statement = select(
            pages,
            *[
                asset_aliases[role].c.asset_id.label(f"{role}_asset_id")
                for role in asset_aliases
            ],
        ).where(
            pages.c.chapter_id == chapter_id,
            pages.c.ordinal > after_ordinal,
        )
        for role, alias in asset_aliases.items():
            statement = statement.outerjoin(
                alias,
                and_(alias.c.page_id == pages.c.id, alias.c.role == role),
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

    def get_page_document(self, page_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            page = connection.execute(
                select(
                    pages.c.id,
                    pages.c.chapter_id,
                    pages.c.document_revision,
                    pages.c.default_font_id,
                    pages.c.page_style_defaults_json,
                    pages.c.page_style_schema_version,
                ).where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise ContentNotFound("page not found")
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
        return {
            "pageId": page["id"],
            "chapterId": page["chapter_id"],
            "documentRevision": page["document_revision"],
            "defaultFontId": page["default_font_id"],
            "pageStyleDefaults": json.loads(page["page_style_defaults_json"]),
            "pageStyleSchemaVersion": page["page_style_schema_version"],
            "bubbles": [
                {
                    "bubbleId": row["id"],
                    "ordinal": row["ordinal"],
                    "fontId": row["font_id"],
                    "payload": json.loads(row["payload_json"]),
                    "updatedRevision": row["updated_revision"],
                }
                for row in bubble_rows
            ],
        }

    def mutate_page_document(
        self,
        *,
        page_id: str,
        base_revision: int,
        mutations: list[dict[str, object]],
    ) -> dict[str, object]:
        if not mutations or len(mutations) > 500:
            raise ValueError("mutations must contain 1-500 items")
        with immediate_transaction(self.engine) as connection:
            page = connection.execute(
                select(
                    pages.c.chapter_id,
                    pages.c.document_revision,
                    pages.c.rendered_revision,
                    pages.c.render_status,
                ).where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise ContentNotFound("page not found")
            if page["document_revision"] != base_revision:
                raise ContentConflict("page document revision changed")
            self._assert_chapter_writable(connection, str(page["chapter_id"]))
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
                    "payload": json.loads(row["payload_json"]),
                }
                for row in existing_rows
            }
            order = [str(row["id"]) for row in existing_rows]
            deleted_ids: set[str] = set()
            created_ids: set[str] = set()
            renderable_change = False
            has_current_translated = (
                connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none()
                is not None
            )
            for mutation in mutations:
                if not isinstance(mutation, dict):
                    raise ValueError("each bubble mutation must be an object")
                operation = mutation.get("op")
                bubble_id = mutation.get("bubbleId")
                fields = mutation.get("fields", {})
                if (
                    operation not in {"create", "patch", "delete", "reset"}
                    or not isinstance(bubble_id, str)
                    or not bubble_id
                ):
                    raise ValueError("bubble mutation op/bubbleId is invalid")
                if operation == "create":
                    if bubble_id in documents or bubble_id in deleted_ids:
                        raise ContentConflict("bubble id already exists in this document")
                    if not isinstance(fields, dict):
                        raise ValueError("create fields must be an object")
                    documents[bubble_id] = {
                        "bubbleId": bubble_id,
                        "fontId": fields.get("fontId"),
                        "payload": {
                            key: value
                            for key, value in fields.items()
                            if key not in {"fontId", "ordinal"}
                        },
                    }
                    order.append(bubble_id)
                    created_ids.add(bubble_id)
                    renderable_change = True
                    continue
                if bubble_id not in documents:
                    raise ContentNotFound(f"bubble {bubble_id} not found")
                if operation == "delete":
                    documents.pop(bubble_id)
                    order.remove(bubble_id)
                    deleted_ids.add(bubble_id)
                    renderable_change = True
                    continue
                if not isinstance(fields, dict):
                    raise ValueError("patch/reset fields must be an object")
                current = documents[bubble_id]
                if "fontId" in fields:
                    current["fontId"] = fields["fontId"]
                payload_fields = {
                    key: value
                    for key, value in fields.items()
                    if key not in {"fontId", "ordinal"}
                }
                if "fontId" in fields or self._affects_render(payload_fields):
                    renderable_change = True
                if operation == "reset":
                    current["payload"] = payload_fields
                else:
                    current_payload = dict(current["payload"])  # type: ignore[arg-type]
                    current_payload.update(payload_fields)
                    current["payload"] = current_payload

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
                str(document["payload"].get("translatedText", "")).strip()
                for document in documents.values()
            )
            needs_render = renderable_change and (
                has_current_translated or has_drawable_text
            )
            page_values: dict[str, object] = {
                "document_revision": new_revision,
                "updated_at": _utcnow(),
            }
            if needs_render:
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
                        page_assets.c.role.in_(
                            ("translated", "thumbnail_translated")
                        ),
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
            if needs_render:
                from src.backend_v2.operations.repository import (
                    RenderRequestRepository,
                )

                RenderRequestRepository(self.engine).upsert(
                    connection,
                    page_id=page_id,
                    requested_revision=new_revision,
                )
        return self.get_page_document(page_id)

    @staticmethod
    def _affects_render(fields: dict[str, object]) -> bool:
        return bool(
            {
                "translatedText",
                "coords",
                "fontSize",
                "fontFamily",
                "textDirection",
                "textColor",
                "rotationAngle",
                "position",
                "strokeEnabled",
                "strokeColor",
                "strokeWidth",
                "lineSpacing",
                "textAlign",
            }
            & fields.keys()
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
            "sourceRevision": row["source_revision"],
            "documentRevision": row["document_revision"],
            "renderedRevision": row["rendered_revision"],
            "renderStatus": row["render_status"],
            "detectionState": row["detection_state"],
            "sourceUrl": url("source"),
            "thumbnailSourceUrl": url("thumbnail_source"),
            "cleanUrl": url("clean"),
            "translatedUrl": url("translated"),
            "thumbnailTranslatedUrl": url("thumbnail_translated"),
        }

    def create_import_lease(self, chapter_id: str) -> ImportLease:
        now = _utcnow()
        expires_at = now + timedelta(seconds=60)
        owner_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(owner_token.encode("utf-8")).hexdigest()
        lease_id = str(uuid.uuid4())
        with immediate_transaction(self.engine) as connection:
            if connection.execute(
                select(chapters.c.id).where(chapters.c.id == chapter_id)
            ).scalar_one_or_none() is None:
                raise ContentNotFound("chapter not found")
            self._assert_chapter_writable(connection, chapter_id)
            connection.execute(
                delete(import_leases).where(
                    import_leases.c.chapter_id == chapter_id,
                    import_leases.c.expires_at <= now,
                )
            )
            if connection.execute(
                select(import_leases.c.id).where(
                    import_leases.c.chapter_id == chapter_id
                )
            ).scalar_one_or_none() is not None:
                raise ContentLocked("an image import is already active for this chapter")
            connection.execute(
                insert(import_leases).values(
                    id=lease_id,
                    chapter_id=chapter_id,
                    owner_token_hash=token_hash,
                    last_activity_at=now,
                    expires_at=expires_at,
                )
            )
        return ImportLease(lease_id, owner_token, expires_at)

    def validate_and_renew_import_lease(
        self,
        *,
        chapter_id: str,
        lease_id: str,
        owner_token: str,
    ) -> datetime:
        now = _utcnow()
        expires_at = now + timedelta(seconds=60)
        token_hash = hashlib.sha256(owner_token.encode("utf-8")).hexdigest()
        with immediate_transaction(self.engine) as connection:
            changed = connection.execute(
                update(import_leases)
                .where(
                    import_leases.c.id == lease_id,
                    import_leases.c.chapter_id == chapter_id,
                    import_leases.c.owner_token_hash == token_hash,
                    import_leases.c.expires_at > now,
                )
                .values(last_activity_at=now, expires_at=expires_at)
            )
            if changed.rowcount != 1:
                raise ContentLocked("import lease is missing, expired, or owned by another client")
        return expires_at

    def release_import_lease(
        self,
        *,
        chapter_id: str,
        lease_id: str,
        owner_token: str,
    ) -> None:
        token_hash = hashlib.sha256(owner_token.encode("utf-8")).hexdigest()
        with self.engine.begin() as connection:
            removed = connection.execute(
                delete(import_leases).where(
                    import_leases.c.id == lease_id,
                    import_leases.c.chapter_id == chapter_id,
                    import_leases.c.owner_token_hash == token_hash,
                )
            )
            if removed.rowcount != 1:
                raise ContentLocked("import lease is missing or owned by another client")

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
        if not normalized_chapter_title:
            raise ValueError("chapter title is required")
        if bool(new_book_title) == bool(target_book_id):
            raise ValueError("choose exactly one target: new book or existing book")
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

            if new_book_title:
                title = new_book_title.strip()
                if not title:
                    raise ValueError("new book title is required")
                destination_book_id = str(uuid.uuid4())
                connection.execute(
                    insert(books).values(
                        id=destination_book_id,
                        kind="library",
                        title=title,
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
                connection.execute(
                    insert(translation_constraints).values(
                        book_id=destination_book_id,
                        payload_json=(
                            source_constraints["payload_json"]
                            if source_constraints
                            else '{"glossary":[],"nonTranslate":[]}'
                        ),
                        schema_version=(
                            source_constraints["schema_version"]
                            if source_constraints
                            else 1
                        ),
                    )
                )
            else:
                destination_book_id = str(target_book_id)
                destination_kind = connection.execute(
                    select(books.c.kind).where(books.c.id == destination_book_id)
                ).scalar_one_or_none()
                if destination_kind != "library":
                    raise ContentNotFound("target library book not found")

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
            .where(
                operations.c.status.in_(ACTIVE_OPERATION_STATUSES),
                pages.c.chapter_id.in_(chapter_ids) if chapter_ids else False,
            ).limit(1)
        ).scalar_one_or_none()
        active_import = connection.execute(  # type: ignore[attr-defined]
            select(import_leases.c.id).where(
                import_leases.c.chapter_id.in_(chapter_ids),
                import_leases.c.expires_at > _utcnow(),
            ).limit(1)
        ).scalar_one_or_none() if chapter_ids else None
        if active_job or active_operation or active_import:
            raise ContentLocked("quick workspace is still referenced by active work")
