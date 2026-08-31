"""Durable browser-page sessions backed by the existing translation pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, BinaryIO
from urllib.parse import urlsplit
import uuid

from sqlalchemy import Engine, and_, delete, func, insert, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.browser_extension.retention import (
    cleanup_expired_browser_sessions,
)
from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.page_style import resolve_new_page_style
from src.backend_v2.content.repository import (
    ContentConflict,
    ContentLocked,
    ContentNotFound,
    ContentRepository,
)
from src.backend_v2.content.translation_constraints import (
    empty_translation_constraints,
    validate_translation_constraints,
    with_glossary_delta,
)
from src.backend_v2.insight.repository import mark_book_insight_derived_stale
from src.backend_v2.jobs.repository import (
    InvalidJobTransition,
    JobQueueRepository,
)
from src.backend_v2.runtime_profile import RuntimeProfile
from src.backend_v2.serialization import canonical_json
from src.backend_v2.settings.validation import validate_setting_payload
from src.backend_v2.storage.assets import AssetRecord, AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    app_settings,
    assets,
    books,
    browser_session_pages,
    browser_sessions,
    chapters,
    job_items,
    jobs,
    page_assets,
    pages,
    translation_constraints,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.translation.commands import TranslationJobCommandService


SESSION_TTL = timedelta(hours=24)


class BrowserSessionNotFound(LookupError):
    pass


class BrowserSessionConflict(RuntimeError):
    pass


def _error_payload(error: BaseException | str, code: str) -> str:
    message = str(error).strip() or "unknown error"
    return canonical_json({"code": code, "message": message[:2_000]})


def _decode_error(value: object) -> dict[str, object] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"code": "translation_failed", "message": value[:2_000]}
    if isinstance(parsed, Mapping):
        code = parsed.get("code")
        message = parsed.get("message")
        if isinstance(code, str) and isinstance(message, str):
            return {"code": code, "message": message}
    return {"code": "translation_failed", "message": "translation failed"}


class BrowserSessionService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        profile: RuntimeProfile,
    ) -> None:
        if profile.name != "local":
            raise ValueError("browser sessions are local-profile only")
        self.data_root = data_root.resolve()
        self.engine = engine
        self.content = ContentRepository(engine)
        self.storage = AssetStorageService(self.data_root, engine)
        self.importer = ImageImportService(
            data_root=self.data_root,
            repository=self.content,
            storage=self.storage,
        )
        self.jobs = JobQueueRepository(engine)
        self.translation = TranslationJobCommandService(engine, profile=profile)
        self._lock = threading.RLock()

    def create(
        self,
        *,
        page_url: str,
        page_title: str,
        mode: str,
        glossary_enabled: bool,
        auto_terms_enabled: bool,
    ) -> dict[str, object]:
        normalized_url = self._validate_page_url(page_url)
        normalized_title = self._title(page_title, normalized_url)
        normalized_mode = self._mode(mode)
        if not isinstance(glossary_enabled, bool) or not isinstance(
            auto_terms_enabled,
            bool,
        ):
            raise ValueError("glossary flags must be booleans")
        self.cleanup_expired()
        owner = effective_owner_id()
        now = utcnow()
        session_id = str(uuid.uuid4())
        book_id = str(uuid.uuid4())
        chapter_id = str(uuid.uuid4())
        with self._lock, immediate_transaction(self.engine) as connection:
            connection.execute(
                insert(books).values(
                    id=book_id,
                    owner_user_id=owner,
                    kind="browser_session",
                    title=normalized_title,
                )
            )
            connection.execute(
                insert(chapters).values(
                    id=chapter_id,
                    book_id=book_id,
                    ordinal=1,
                    title=normalized_title,
                )
            )
            constraints = empty_translation_constraints()
            constraints["glossary"]["enabled"] = glossary_enabled
            constraints["glossary"]["autoExtractEnabled"] = auto_terms_enabled
            connection.execute(
                insert(translation_constraints).values(
                    book_id=book_id,
                    payload_json=canonical_json(constraints),
                )
            )
            connection.execute(
                insert(browser_sessions).values(
                    id=session_id,
                    owner_user_id=owner,
                    book_id=book_id,
                    chapter_id=chapter_id,
                    page_url=normalized_url,
                    page_title=normalized_title,
                    mode=normalized_mode,
                    status="active",
                    expires_at=now + SESSION_TTL,
                )
            )
        return self.get(session_id)

    def add_page(
        self,
        *,
        session_id: str,
        client_page_key: str,
        ordinal: int,
        logical_path: str,
        source_url: str | None,
        upload: BinaryIO,
    ) -> dict[str, object]:
        client_key = client_page_key.strip()
        if not client_key or len(client_key) > 200:
            raise ValueError("clientPageKey must contain 1-200 characters")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError("ordinal must be an integer")
        if not 1 <= ordinal <= 1_000_000:
            raise ValueError("ordinal must be from 1 to 1000000")
        normalized_path = logical_path.strip()
        if not normalized_path or len(normalized_path) > 2_000:
            raise ValueError("logicalPath must contain 1-2000 characters")
        normalized_source_url = None
        if source_url is not None:
            if not isinstance(source_url, str) or len(source_url) > 20_000:
                raise ValueError("sourceUrl must be a string of at most 20000 characters")
            parsed_source = urlsplit(source_url)
            if (
                parsed_source.scheme not in {"http", "https"}
                or not parsed_source.hostname
            ):
                raise ValueError("sourceUrl must be an HTTP(S) URL")
            normalized_source_url = source_url
        with self.engine.connect() as connection:
            session = self._require_session(connection, session_id)
            existing = self._page_by_client_key(
                connection,
                session_id=session_id,
                client_page_key=client_key,
            )
        if existing is not None:
            return self.get_page(
                session_id=session_id,
                browser_page_id=str(existing["id"]),
            )
        if session["status"] == "cancelled":
            raise BrowserSessionConflict(
                "cancelled browser session does not accept new pages"
            )

        source, thumbnail = self.importer.publish_standalone_image(upload)
        page_record_id = str(uuid.uuid4())
        now = utcnow()
        try:
            with self._lock, immediate_transaction(self.engine) as connection:
                session = self._require_session(connection, session_id)
                existing = self._page_by_client_key(
                    connection,
                    session_id=session_id,
                    client_page_key=client_key,
                )
                if existing is not None:
                    return self.get_page(
                        session_id=session_id,
                        browser_page_id=str(existing["id"]),
                    )
                if session["status"] == "cancelled":
                    raise BrowserSessionConflict(
                        "cancelled browser session does not accept new pages"
                    )
                used_ordinal = connection.execute(
                    select(browser_session_pages.c.id).where(
                        browser_session_pages.c.session_id == session_id,
                        browser_session_pages.c.ordinal == ordinal,
                    )
                ).scalar_one_or_none()
                if used_ordinal is not None:
                    ordinal = int(
                        connection.execute(
                            select(
                                func.coalesce(
                                    func.max(browser_session_pages.c.ordinal),
                                    0,
                                )
                            ).where(
                                browser_session_pages.c.session_id == session_id
                            )
                        ).scalar_one()
                    ) + 1
                connection.execute(
                    insert(browser_session_pages).values(
                        id=page_record_id,
                        session_id=session_id,
                        client_page_key=client_key,
                        ordinal=ordinal,
                        logical_path=normalized_path,
                        source_url=normalized_source_url,
                        source_asset_id=source.id,
                        thumbnail_asset_id=thumbnail.id,
                    )
                )
                connection.execute(
                    update(browser_sessions)
                    .where(browser_sessions.c.id == session_id)
                    .values(updated_at=now, expires_at=now + SESSION_TTL)
                )
        except IntegrityError:
            with self.engine.connect() as connection:
                existing = self._page_by_client_key(
                    connection,
                    session_id=session_id,
                    client_page_key=client_key,
                )
            if existing is None:
                raise
            return self.get_page(
                session_id=session_id,
                browser_page_id=str(existing["id"]),
            )
        return self.get_page(session_id=session_id, browser_page_id=page_record_id)

    def get(self, session_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            session = self._require_session(connection, session_id)
            rows = self._session_page_rows(connection, session_id)
            glossary_enabled, auto_terms_enabled = self._glossary_flags(
                connection,
                book_id=str(session["book_id"]),
            )
        page_dtos = [self._page_from_joined_row(row) for row in rows]
        counts = {
            key: sum(1 for page in page_dtos if page["state"] == key)
            for key in (
                "queued",
                "translating",
                "completed",
                "failed",
                "cancelled",
            )
        }
        if session["status"] == "cancelled":
            state = "cancelled"
        elif counts["translating"]:
            state = "translating"
        elif counts["queued"]:
            state = "queued"
        elif counts["failed"] and counts["completed"]:
            state = "partial"
        elif counts["failed"]:
            state = "failed"
        elif counts["cancelled"] and counts["completed"]:
            state = "partial"
        elif counts["cancelled"]:
            state = "cancelled"
        elif page_dtos and counts["completed"] == len(page_dtos):
            state = "completed"
        else:
            state = "idle"
        return {
            "id": str(session["id"]),
            "pageUrl": str(session["page_url"]),
            "pageTitle": str(session["page_title"]),
            "bookId": str(session["book_id"]),
            "chapterId": str(session["chapter_id"]),
            "mode": str(session["mode"]),
            "glossaryEnabled": glossary_enabled,
            "autoTermsEnabled": auto_terms_enabled,
            "state": state,
            "expiresAt": (
                session["expires_at"].isoformat()
                if session["expires_at"] is not None
                else None
            ),
            "counts": {"total": len(page_dtos), **counts},
            "pages": page_dtos,
        }

    def get_page(
        self,
        *,
        session_id: str,
        browser_page_id: str,
    ) -> dict[str, object]:
        with self.engine.connect() as connection:
            self._require_session(connection, session_id)
            rows = self._session_page_rows(
                connection,
                session_id,
                browser_page_id=browser_page_id,
            )
        if not rows:
            raise BrowserSessionNotFound("browser page not found")
        return self._page_from_joined_row(rows[0])

    def update(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        glossary_enabled: bool | None = None,
        auto_terms_enabled: bool | None = None,
    ) -> dict[str, object]:
        now = utcnow()
        values: dict[str, object] = {"updated_at": now}
        if mode is not None:
            values["mode"] = self._mode(mode)
        for value, label in (
            (glossary_enabled, "glossaryEnabled"),
            (auto_terms_enabled, "autoTermsEnabled"),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean")
        with self._lock, immediate_transaction(self.engine) as connection:
            session = self._require_session(connection, session_id)
            connection.execute(
                update(browser_sessions)
                .where(browser_sessions.c.id == session_id)
                .values(**values)
            )
            if glossary_enabled is not None or auto_terms_enabled is not None:
                self._set_glossary_flags(
                    connection,
                    book_id=str(session["book_id"]),
                    glossary_enabled=glossary_enabled,
                    auto_terms_enabled=auto_terms_enabled,
                )
        return self.get(session_id)

    def start(self, session_id: str) -> dict[str, object]:
        """Start every uploaded page that is not already assigned to a job."""

        with self._lock:
            with immediate_transaction(self.engine) as connection:
                session = self._require_session(connection, session_id)
                has_active_job = self._has_active_translation_job(
                    connection,
                    chapter_id=str(session["chapter_id"]),
                )
            if has_active_job:
                return self.get(session_id)
            if session["status"] == "cancelled":
                return self.get(session_id)
            if not self._materialize_pending(session_id):
                return self.get(session_id)
            self._create_pending_job(session_id)
        return self.get(session_id)

    def terms(self, session_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            session = self._require_session(connection, session_id)
            row = connection.execute(
                select(translation_constraints).where(
                    translation_constraints.c.book_id == session["book_id"]
                )
            ).mappings().one()
        payload = validate_translation_constraints(json.loads(row["payload_json"]))
        return {
            "revision": int(row["revision"]),
            "glossary": payload["glossary"],
        }

    def cancel(self, session_id: str) -> dict[str, object]:
        with self._lock:
            with immediate_transaction(self.engine) as connection:
                self._require_session(connection, session_id)
                job_ids = list(
                    connection.execute(
                        select(browser_session_pages.c.job_id)
                        .join(jobs, jobs.c.id == browser_session_pages.c.job_id)
                        .where(
                            browser_session_pages.c.session_id == session_id,
                            browser_session_pages.c.job_id.is_not(None),
                            jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                        )
                        .distinct()
                    ).scalars()
                )
                connection.execute(
                    update(browser_sessions)
                    .where(browser_sessions.c.id == session_id)
                    .values(
                        status="cancelled",
                        updated_at=utcnow(),
                    )
                )
            for job_id in job_ids:
                try:
                    self.jobs.request_cancel(str(job_id))
                except (InvalidJobTransition, LookupError):
                    pass
        return self.get(session_id)

    def retry(self, *, session_id: str, browser_page_id: str) -> dict[str, object]:
        with self._lock:
            with immediate_transaction(self.engine) as connection:
                session = self._require_session(connection, session_id)
                translated = page_assets.alias("browser_retry_translated")
                row = connection.execute(
                    select(
                        browser_session_pages,
                        job_items.c.status.label("item_status"),
                        translated.c.asset_id.label("translated_asset_id"),
                    )
                    .outerjoin(
                        job_items,
                        and_(
                            job_items.c.job_id == browser_session_pages.c.job_id,
                            job_items.c.page_id == browser_session_pages.c.page_id,
                        ),
                    )
                    .outerjoin(
                        translated,
                        and_(
                            translated.c.page_id == browser_session_pages.c.page_id,
                            translated.c.role == "translated",
                        ),
                    )
                    .where(
                        browser_session_pages.c.id == browser_page_id,
                        browser_session_pages.c.session_id == session_id,
                    )
                ).mappings().one_or_none()
                if row is None:
                    raise BrowserSessionNotFound("browser page not found")
                if self._has_active_translation_job(
                    connection,
                    chapter_id=str(session["chapter_id"]),
                ):
                    raise BrowserSessionConflict(
                        "wait for the current browser translation batch before retrying"
                    )
                has_prior_result = row["translated_asset_id"] is not None
                if (
                    row["error_json"] is None
                    and row["item_status"] not in {"failed", "cancelled"}
                    and not has_prior_result
                ):
                    raise BrowserSessionConflict(
                        "only a failed, cancelled, or completed page can be retried"
                    )
                connection.execute(
                    update(browser_session_pages)
                    .where(browser_session_pages.c.id == browser_page_id)
                    .values(
                        job_id=None,
                        retry_count=browser_session_pages.c.retry_count + 1,
                        error_json=None,
                        updated_at=utcnow(),
                    )
                )
                connection.execute(
                    update(browser_sessions)
                    .where(browser_sessions.c.id == session_id)
                    .values(
                        status="active",
                        updated_at=utcnow(),
                    )
                )
            if not self._materialize_pending(session_id):
                return self.get_page(
                    session_id=session_id,
                    browser_page_id=browser_page_id,
                )
            self._create_pending_job(
                session_id,
                browser_page_ids=(browser_page_id,),
            )
        return self.get_page(
            session_id=session_id,
            browser_page_id=browser_page_id,
        )

    def library_books(self) -> list[dict[str, object]]:
        return [
            {
                "id": str(book["id"]),
                "title": str(book["title"]),
                "chapterCount": int(book["chapterCount"]),
            }
            for book in self.content.list_books(
                sort_by="updated_at",
                sort_order="desc",
            )
        ]

    def import_to_library(
        self,
        session_id: str,
        *,
        destination: str,
        book_title: str | None,
        target_book_id: str | None,
        chapter_title: str,
    ) -> dict[str, object]:
        if destination not in {"new", "existing"}:
            raise ValueError("destination must be new or existing")
        if destination == "new":
            if target_book_id is not None:
                raise ValueError("targetBookId is only valid for an existing book")
            normalized_book_title = self._required_title(book_title, "bookTitle")
        else:
            if book_title is not None:
                raise ValueError("bookTitle is only valid for a new book")
            if not isinstance(target_book_id, str) or not target_book_id.strip():
                raise ValueError("targetBookId is required for an existing book")
            target_book_id = target_book_id.strip()
            normalized_book_title = ""
        normalized_chapter_title = self._required_title(
            chapter_title,
            "chapterTitle",
        )
        now = utcnow()
        with self._lock, immediate_transaction(self.engine) as connection:
            session = self._require_session(connection, session_id)
            chapter_id = str(session["chapter_id"])
            source_book_id = str(session["book_id"])
            source_kind = connection.execute(
                select(books.c.kind).where(books.c.id == source_book_id)
            ).scalar_one_or_none()
            if source_kind != "browser_session":
                raise BrowserSessionConflict("browser session was already imported")
            page_dtos = [
                self._page_from_joined_row(row)
                for row in self._session_page_rows(connection, session_id)
            ]
            materialized_pages = sum(page["pageId"] is not None for page in page_dtos)
            if materialized_pages == 0:
                raise BrowserSessionConflict(
                    "at least one imported page is required before adding to the library"
                )
            if any(
                page["state"] in {"queued", "translating"}
                for page in page_dtos
            ) or self._has_active_translation_job(
                connection,
                chapter_id=chapter_id,
            ):
                raise BrowserSessionConflict(
                    "wait for the current browser translation batch before importing"
                )

            terms_added = 0
            if destination == "new":
                destination_book_id = source_book_id
                destination_book_title = normalized_book_title
                connection.execute(
                    update(books)
                    .where(
                        books.c.id == source_book_id,
                        books.c.kind == "browser_session",
                    )
                    .values(
                        kind="library",
                        title=normalized_book_title,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(chapters)
                    .where(chapters.c.id == chapter_id)
                    .values(title=normalized_chapter_title, updated_at=now)
                )
            else:
                if target_book_id == source_book_id:
                    raise ValueError("targetBookId must differ from the session book")
                target = connection.execute(
                    select(books.c.id, books.c.title).where(
                        books.c.id == target_book_id,
                        books.c.kind == "library",
                        books.c.owner_user_id == effective_owner_id(),
                    )
                ).mappings().one_or_none()
                if target is None:
                    raise BrowserSessionNotFound("target library book not found")
                destination_book_id = str(target["id"])
                destination_book_title = str(target["title"])
                next_ordinal = int(
                    connection.execute(
                        select(func.coalesce(func.max(chapters.c.ordinal), 0)).where(
                            chapters.c.book_id == destination_book_id
                        )
                    ).scalar_one()
                ) + 1
                source_constraints = connection.execute(
                    select(translation_constraints.c.payload_json).where(
                        translation_constraints.c.book_id == source_book_id
                    )
                ).scalar_one()
                target_constraints = connection.execute(
                    select(
                        translation_constraints.c.payload_json,
                        translation_constraints.c.revision,
                    ).where(
                        translation_constraints.c.book_id == destination_book_id
                    )
                ).mappings().one()
                source_payload = validate_translation_constraints(
                    json.loads(source_constraints)
                )
                target_payload = validate_translation_constraints(
                    json.loads(target_constraints["payload_json"])
                )
                merged_payload, terms_added = with_glossary_delta(
                    target_payload,
                    source_payload["glossary"]["entries"],
                )
                if terms_added:
                    connection.execute(
                        update(translation_constraints)
                        .where(
                            translation_constraints.c.book_id == destination_book_id
                        )
                        .values(
                            payload_json=canonical_json(merged_payload),
                            revision=int(target_constraints["revision"]) + 1,
                            updated_at=now,
                        )
                    )
                connection.execute(
                    update(jobs)
                    .where(jobs.c.chapter_id == chapter_id)
                    .values(book_id=destination_book_id, updated_at=now)
                )
                connection.execute(
                    update(chapters)
                    .where(chapters.c.id == chapter_id)
                    .values(
                        book_id=destination_book_id,
                        ordinal=next_ordinal,
                        title=normalized_chapter_title,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(books)
                    .where(books.c.id == destination_book_id)
                    .values(
                        chapter_order_revision=books.c.chapter_order_revision + 1,
                        updated_at=now,
                    )
                )
                mark_book_insight_derived_stale(
                    connection,
                    book_id=destination_book_id,
                    now=now,
                )

            connection.execute(
                delete(browser_sessions)
                .where(browser_sessions.c.id == session_id)
            )
            if destination == "existing":
                connection.execute(delete(books).where(books.c.id == source_book_id))
        return {
            "destination": destination,
            "bookId": destination_book_id,
            "bookTitle": destination_book_title,
            "chapterId": chapter_id,
            "chapterTitle": normalized_chapter_title,
            "importedPages": materialized_pages,
            "omittedPages": len(page_dtos) - materialized_pages,
            "termsAdded": terms_added,
        }

    def translated_asset(
        self,
        *,
        session_id: str,
        browser_page_id: str,
    ) -> AssetRecord:
        page = self.get_page(
            session_id=session_id,
            browser_page_id=browser_page_id,
        )
        if page["state"] != "completed" or not page["resultReady"]:
            raise BrowserSessionNotFound("translated result is not ready")
        with self.engine.connect() as connection:
            self._require_session(connection, session_id)
            row = connection.execute(
                select(assets)
                .join(page_assets, page_assets.c.asset_id == assets.c.id)
                .join(
                    browser_session_pages,
                    browser_session_pages.c.page_id == page_assets.c.page_id,
                )
                .where(
                    browser_session_pages.c.id == browser_page_id,
                    browser_session_pages.c.session_id == session_id,
                    page_assets.c.role == "translated",
                )
            ).mappings().one_or_none()
        if row is None:
            raise BrowserSessionNotFound("translated result is not ready")
        return AssetRecord(
            id=str(row["id"]),
            relative_path=str(row["relative_path"]),
            mime_type=str(row["mime_type"]),
            checksum=str(row["checksum"]),
            byte_size=int(row["byte_size"]),
            width=int(row["width"]) if row["width"] is not None else None,
            height=int(row["height"]) if row["height"] is not None else None,
        )

    def validate_result_binding(
        self,
        *,
        session_id: str,
        browser_page_id: str,
        asset_id: str,
    ) -> AssetRecord:
        asset = self.translated_asset(
            session_id=session_id,
            browser_page_id=browser_page_id,
        )
        if asset.id != asset_id:
            raise BrowserSessionNotFound("translated result is stale")
        return asset

    def cleanup_expired(self) -> int:
        with self._lock:
            return cleanup_expired_browser_sessions(self.engine)

    def _create_pending_job(
        self,
        session_id: str,
        *,
        browser_page_ids: tuple[str, ...] | None = None,
    ) -> str | None:
        with self.engine.connect() as connection:
            session = self._require_session(connection, session_id)
            statement = select(
                browser_session_pages.c.id,
                browser_session_pages.c.page_id,
                browser_session_pages.c.retry_count,
            ).where(
                browser_session_pages.c.session_id == session_id,
                browser_session_pages.c.page_id.is_not(None),
                browser_session_pages.c.job_id.is_(None),
                browser_session_pages.c.error_json.is_(None),
            )
            if browser_page_ids is not None:
                statement = statement.where(
                    browser_session_pages.c.id.in_(browser_page_ids)
                )
            else:
                translated_exists = select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == browser_session_pages.c.page_id,
                    page_assets.c.role == "translated",
                ).exists()
                statement = statement.where(~translated_exists)
            page_rows = list(
                connection.execute(
                    statement.order_by(browser_session_pages.c.ordinal)
                ).mappings()
            )
            execution_mode = (
                self._translation_execution_mode(connection)
                if page_rows
                else "sequential"
            )
        if not page_rows:
            return None
        page_ids = [str(row["page_id"]) for row in page_rows]
        fingerprint = hashlib.sha256(
            "\n".join(
                f"{row['id']}:{row['retry_count']}" for row in page_rows
            ).encode("utf-8")
        ).hexdigest()

        def bind_job(
            connection: Connection,
            _batch_id: str,
            job_ids: Sequence[str],
        ) -> None:
            if len(job_ids) != 1:
                raise RuntimeError("browser translation did not create exactly one job")
            connection.execute(
                update(browser_session_pages)
                .where(
                    browser_session_pages.c.id.in_(
                        [row["id"] for row in page_rows]
                    ),
                    browser_session_pages.c.job_id.is_(None),
                )
                .values(
                    job_id=job_ids[0],
                    error_json=None,
                    updated_at=utcnow(),
                )
            )

        created = self.translation.create_chapter_job(
            chapter_id=str(session["chapter_id"]),
            config={
                "mode": str(session["mode"]),
                "executionMode": execution_mode,
                "skipCompleted": False,
            },
            page_ids=page_ids,
            idempotency_key=f"browser-{fingerprint}",
            idempotency_scope=f"browser-session:{session_id}",
            transaction_hook=bind_job,
        )
        raw_ids = created.get("jobIds")
        if not isinstance(raw_ids, list) or len(raw_ids) != 1:
            raise RuntimeError("browser translation did not create exactly one job")
        return str(raw_ids[0])

    @staticmethod
    def _translation_execution_mode(connection: Connection) -> str:
        row = connection.execute(
            select(
                app_settings.c.payload_json,
                app_settings.c.schema_version,
            ).where(
                app_settings.c.domain == "translation",
                app_settings.c.owner_user_id == effective_owner_id(),
            )
        ).mappings().one_or_none()
        if row is None:
            raise ValueError("translation settings are missing")
        settings = validate_setting_payload(
            "translation",
            json.loads(row["payload_json"]),
            schema_version=int(row["schema_version"]),
        )
        return "parallel" if settings["parallel"]["enabled"] else "sequential"

    @staticmethod
    def _has_active_translation_job(
        connection: Connection,
        *,
        chapter_id: str,
    ) -> bool:
        value = connection.execute(
            select(jobs.c.id)
            .where(
                jobs.c.chapter_id == chapter_id,
                jobs.c.kind == "translation",
                jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
            )
            .limit(1)
        ).scalar_one_or_none()
        return value is not None

    def _materialize_pending(self, session_id: str) -> bool:
        with self.engine.connect() as connection:
            session = self._require_session(connection, session_id)
            pending = list(
                connection.execute(
                    select(browser_session_pages).where(
                        browser_session_pages.c.session_id == session_id,
                        browser_session_pages.c.page_id.is_(None),
                        browser_session_pages.c.error_json.is_(None),
                    ).order_by(browser_session_pages.c.ordinal)
                ).mappings()
            )
            font_id, page_style = resolve_new_page_style(connection)
        text_style: dict[str, object] = {
            "fontFamily": font_id,
            **page_style,
        }
        for row in pending:
            source = self.storage.get_record(str(row["source_asset_id"]))
            thumbnail = self.storage.get_record(str(row["thumbnail_asset_id"]))
            if source is None or thumbnail is None:
                self._record_page_error(
                    str(row["id"]),
                    "source asset is unavailable",
                    "source_missing",
                )
                continue
            request_hash = hashlib.sha256(
                (
                    f"{session_id}\n{row['id']}\n{source.checksum}\n"
                    f"{thumbnail.checksum}"
                ).encode("utf-8")
            ).hexdigest()
            try:
                result, _replayed = self.content.append_page(
                    chapter_id=str(session["chapter_id"]),
                    requested_logical_path=str(row["logical_path"]),
                    text_style=text_style,
                    source=source,
                    thumbnail=thumbnail,
                    idempotency_scope=f"browser-materialize:{session_id}",
                    idempotency_key=f"browser-page-{row['id']}",
                    request_hash=request_hash,
                )
            except ContentLocked:
                return False
            except (ContentNotFound, ContentConflict, ValueError) as error:
                self._record_page_error(
                    str(row["id"]),
                    error,
                    "materialize_failed",
                )
                continue
            page_payload = result.get("page")
            if not isinstance(page_payload, Mapping) or not isinstance(
                page_payload.get("id"),
                str,
            ):
                self._record_page_error(
                    str(row["id"]),
                    "materialized browser page response is invalid",
                    "materialize_failed",
                )
                continue
            with immediate_transaction(self.engine) as connection:
                connection.execute(
                    update(browser_session_pages)
                    .where(browser_session_pages.c.id == row["id"])
                    .values(
                        page_id=page_payload["id"],
                        error_json=None,
                        updated_at=utcnow(),
                    )
                )
        try:
            self._restore_page_order(session_id)
        except ContentLocked:
            return False
        return True

    def _restore_page_order(self, session_id: str) -> None:
        with self.engine.connect() as connection:
            session = self._require_session(connection, session_id)
            desired = [
                str(value)
                for value in connection.execute(
                    select(browser_session_pages.c.page_id)
                    .where(
                        browser_session_pages.c.session_id == session_id,
                        browser_session_pages.c.page_id.is_not(None),
                    )
                    .order_by(browser_session_pages.c.ordinal)
                ).scalars()
            ]
            actual = [
                str(value)
                for value in connection.execute(
                    select(pages.c.id)
                    .where(pages.c.chapter_id == session["chapter_id"])
                    .order_by(pages.c.ordinal)
                ).scalars()
            ]
            revision = connection.execute(
                select(chapters.c.page_order_revision).where(
                    chapters.c.id == session["chapter_id"]
                )
            ).scalar_one()
        if actual == desired:
            return
        if set(actual) != set(desired):
            raise RuntimeError("browser chapter contains an unexpected page")
        self.content.reorder_pages(
            chapter_id=str(session["chapter_id"]),
            ordered_ids=desired,
            base_revision=int(revision),
        )

    def _session_page_rows(
        self,
        connection: Connection,
        session_id: str,
        *,
        browser_page_id: str | None = None,
    ) -> list[Mapping[str, Any]]:
        translated = page_assets.alias("browser_translated_asset")
        statement = (
            select(
                browser_session_pages,
                job_items.c.status.label("item_status"),
                job_items.c.error_json.label("item_error_json"),
                translated.c.asset_id.label("translated_asset_id"),
            )
            .outerjoin(
                job_items,
                and_(
                    job_items.c.job_id == browser_session_pages.c.job_id,
                    job_items.c.page_id == browser_session_pages.c.page_id,
                ),
            )
            .outerjoin(
                translated,
                and_(
                    translated.c.page_id == browser_session_pages.c.page_id,
                    translated.c.role == "translated",
                ),
            )
            .where(browser_session_pages.c.session_id == session_id)
            .order_by(browser_session_pages.c.ordinal)
        )
        if browser_page_id is not None:
            statement = statement.where(
                browser_session_pages.c.id == browser_page_id
            )
        return list(connection.execute(statement).mappings())

    @staticmethod
    def _page_from_joined_row(row: Mapping[str, Any]) -> dict[str, object]:
        item_status = row.get("item_status")
        translated_asset_id = row.get("translated_asset_id")
        if row.get("error_json"):
            state = "failed"
            error = _decode_error(row.get("error_json"))
        elif row.get("page_id") is None:
            state = "queued"
            error = None
        elif row.get("job_id") is None:
            state = "completed" if translated_asset_id else "queued"
            error = None
        elif item_status == "running":
            state = "translating"
            error = None
        elif item_status == "failed":
            state = "failed"
            error = _decode_error(row.get("item_error_json")) or {
                "code": "translation_failed",
                "message": "translation failed",
            }
        elif item_status == "cancelled":
            state = "cancelled"
            error = None
        elif item_status == "completed":
            if translated_asset_id:
                state = "completed"
                error = None
            else:
                state = "failed"
                error = {
                    "code": "result_missing",
                    "message": "translation completed without a result image",
                }
        else:
            state = "queued"
            error = None
        return {
            "id": str(row["id"]),
            "clientPageKey": str(row["client_page_key"]),
            "ordinal": int(row["ordinal"]),
            "pageId": str(row["page_id"]) if row.get("page_id") else None,
            "state": state,
            "resultReady": bool(translated_asset_id and state == "completed"),
            "retryCount": int(row["retry_count"]),
            "error": error,
        }

    @staticmethod
    def _page_by_client_key(
        connection: Connection,
        *,
        session_id: str,
        client_page_key: str,
    ) -> Mapping[str, Any] | None:
        return connection.execute(
            select(browser_session_pages).where(
                browser_session_pages.c.session_id == session_id,
                browser_session_pages.c.client_page_key == client_page_key,
            )
        ).mappings().one_or_none()

    @staticmethod
    def _require_session(
        connection: Connection,
        session_id: str,
    ) -> Mapping[str, Any]:
        row = connection.execute(
            select(browser_sessions).where(
                browser_sessions.c.id == session_id,
                browser_sessions.c.owner_user_id == effective_owner_id(),
            )
        ).mappings().one_or_none()
        if row is None:
            raise BrowserSessionNotFound("browser session not found")
        return row

    @staticmethod
    def _glossary_flags(
        connection: Connection,
        *,
        book_id: str,
    ) -> tuple[bool, bool]:
        payload_json = connection.execute(
            select(translation_constraints.c.payload_json).where(
                translation_constraints.c.book_id == book_id
            )
        ).scalar_one()
        payload = validate_translation_constraints(json.loads(payload_json))
        glossary = payload["glossary"]
        return bool(glossary["enabled"]), bool(glossary["autoExtractEnabled"])

    @staticmethod
    def _set_glossary_flags(
        connection: Connection,
        *,
        book_id: str,
        glossary_enabled: bool | None,
        auto_terms_enabled: bool | None,
    ) -> None:
        row = connection.execute(
            select(
                translation_constraints.c.payload_json,
                translation_constraints.c.revision,
            ).where(translation_constraints.c.book_id == book_id)
        ).mappings().one()
        payload = validate_translation_constraints(json.loads(row["payload_json"]))
        glossary = payload["glossary"]
        next_glossary_enabled = (
            glossary_enabled
            if glossary_enabled is not None
            else bool(glossary["enabled"])
        )
        next_auto_terms_enabled = (
            auto_terms_enabled
            if auto_terms_enabled is not None
            else bool(glossary["autoExtractEnabled"])
        )
        if (
            glossary["enabled"] == next_glossary_enabled
            and glossary["autoExtractEnabled"] == next_auto_terms_enabled
        ):
            return
        payload["glossary"]["enabled"] = next_glossary_enabled
        payload["glossary"]["autoExtractEnabled"] = next_auto_terms_enabled
        connection.execute(
            update(translation_constraints)
            .where(translation_constraints.c.book_id == book_id)
            .values(
                payload_json=canonical_json(payload),
                revision=int(row["revision"]) + 1,
                updated_at=utcnow(),
            )
        )

    def _record_page_error(
        self,
        browser_page_id: str,
        error: BaseException | str,
        code: str,
    ) -> None:
        with immediate_transaction(self.engine) as connection:
            connection.execute(
                update(browser_session_pages)
                .where(browser_session_pages.c.id == browser_page_id)
                .values(
                    error_json=_error_payload(error, code),
                    updated_at=utcnow(),
                )
            )

    @staticmethod
    def _validate_page_url(value: str) -> str:
        if not isinstance(value, str) or len(value) > 20_000:
            raise ValueError("pageUrl must be a string of at most 20000 characters")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("pageUrl must be an HTTP(S) URL")
        return value

    @staticmethod
    def _title(value: str | None, page_url: str) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            normalized = urlsplit(page_url).hostname or "网页漫画"
        if len(normalized) > 500:
            normalized = normalized[:500].rstrip()
        return normalized or "网页漫画"

    @staticmethod
    def _required_title(value: str | None, field: str) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized or len(normalized) > 500:
            raise ValueError(f"{field} must contain 1-500 characters")
        return normalized

    @staticmethod
    def _mode(value: str) -> str:
        if value not in {"standard", "hq"}:
            raise ValueError("mode must be standard or hq")
        return value
