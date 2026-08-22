"""Detect/style job commands and Worker-side style application."""

from __future__ import annotations

import json
from typing import Any, Mapping

from sqlalchemy import Engine, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json
from src.backend_v2.content.page_style import (
    rgb_to_hex,
    validate_page_style,
)
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.public_policy import PublicUserPolicyAccess
from src.backend_v2.translation.commands import resolve_chapter_pages
from src.backend_v2.storage.schema import (
    assets,
    bubbles,
    chapter_write_locks,
    chapters,
    job_steps,
    page_assets,
    pages,
)
from src.core.config_models import validate_bubble_payload


STYLE_FIELDS = frozenset(
    {
        "fontSize",
        "fontFamily",
        "layoutDirection",
        "lineSpacing",
        "inlineAlign",
        "blockAlign",
        "textColor",
        "fillColor",
        "strokeEnabled",
        "strokeColor",
        "strokeWidth",
    }
)
RENDER_STYLE_FIELDS = STYLE_FIELDS - {"fillColor"}
TEXT_IMPORT_FIELDS = {
    "original_text": "originalText",
    "translated_text": "translatedText",
    "textbox_text": "textboxText",
    "text_direction": "textDirection",
}
TEXT_EXPORT_ROOT_FIELDS = frozenset(
    {"schema_version", "book_id", "chapter_id", "exported_at", "pages"}
)
TEXT_EXPORT_PAGE_FIELDS = frozenset(
    {
        "page_id",
        "page_number",
        "source_checksum",
        "document_revision",
        "bubbles",
    }
)
TEXT_EXPORT_BUBBLE_FIELDS = frozenset(
    {"bubble_id", *TEXT_IMPORT_FIELDS.keys()}
)
TEXT_IMPORT_PREVIEW_PAGE_FIELDS = frozenset(
    {
        "pageId",
        "status",
        "issues",
        "baseDocumentRevision",
        "sourceChecksum",
        "sourceAssetId",
        "changes",
    }
)
TEXT_IMPORT_CHANGE_FIELDS = frozenset(
    {"bubbleId", "fields", "differences"}
)


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    fields = set(value)
    missing = expected - fields
    unknown = fields - expected
    if missing:
        raise ValueError(
            f"{label} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ValueError(
            f"{label} contains unknown fields: {', '.join(sorted(unknown))}"
        )


def _require_non_empty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _load_current_bubble_payload(
    row: Mapping[str, Any],
    *,
    document_revision: int,
    label: str,
) -> dict[str, Any]:
    if row["updated_revision"] != document_revision:
        raise RuntimeError(f"{label} revision does not match page document")
    try:
        return validate_bubble_payload(
            json.loads(row["payload_json"]),
            render=False,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} payload does not match the current schema"
        ) from exc


def _bubble_text(payload: Mapping[str, Any], field: str) -> str:
    if field not in payload:
        raise RuntimeError(f"bubble {field} is missing")
    value = payload[field]
    if not isinstance(value, str):
        raise RuntimeError(f"bubble {field} must be a string")
    if field == "textDirection" and value not in {"vertical", "horizontal"}:
        raise RuntimeError("bubble textDirection is invalid")
    return value


class AuxiliaryTranslationCommands:
    def __init__(
        self,
        engine: Engine,
        *,
        public_access: PublicUserPolicyAccess | None = None,
    ) -> None:
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)
        self.public_access = public_access

    def create_detect_job(
        self,
        *,
        chapter_id: str,
        page_ids: list[str] | None,
        idempotency_key: str,
        retry_of_job_id: str | None = None,
        retry_mode: str | None = None,
        idempotency_scope: str | None = None,
    ) -> dict[str, object]:
        chapter, ordered = resolve_chapter_pages(
            self.engine,
            chapter_id=chapter_id,
            requested_page_ids=page_ids,
            empty_message="job requires at least one page",
        )
        resolved = self.settings.resolve_translation(
            chapter_id=chapter_id,
            command={
                "mode": "standard",
                "executionMode": "parallel",
                "skipCompleted": False,
                "reuseExistingBubbles": False,
            },
        )
        config = {
            "deepLearningConcurrency": int(resolved["deepLearningConcurrency"]),
            "detector": dict(resolved["detector"]),
            "executionMode": str(resolved["executionMode"]),
            "settingsSnapshot": dict(resolved["settingsSnapshot"]),
        }
        if self.public_access is not None:
            config = self.public_access.apply_resolved_translation(config)
        return self.jobs.create_batch(
            kind="detect",
            display_name=f"检测 {chapter['book_title']} / {chapter['title']}",
            specs=[
                JobSpec(
                    kind="detect",
                    book_id=str(chapter["book_id"]),
                    chapter_id=chapter_id,
                    config=config,
                    items=tuple(
                        JobItemSpec(
                            page_id=page_id,
                            step_kinds=("detect", "render", "save"),
                        )
                        for page_id in ordered
                    ),
                    target_display={
                        "book": chapter["book_title"],
                        "chapter": chapter["title"],
                        "pageCount": len(ordered),
                    },
                    retry_of_job_id=retry_of_job_id,
                    retry_mode=retry_mode,
                )
            ],
            idempotency_scope=idempotency_scope or f"chapter-detect:{chapter_id}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "chapterId": chapter_id,
                "pageIds": ordered,
                "detector": config["detector"],
                "retryOfJobId": retry_of_job_id,
                "retryMode": retry_mode,
            },
        )

    def create_style_apply_job(
        self,
        *,
        chapter_id: str,
        source_page_id: str,
        source_document_revision: int,
        selected_fields: list[str],
        idempotency_key: str,
    ) -> dict[str, object]:
        if not isinstance(source_page_id, str) or not source_page_id:
            raise ValueError("sourcePageId must be a non-empty string")
        _require_positive_integer(
            source_document_revision,
            label="sourceDocumentRevision",
        )
        if not isinstance(selected_fields, list) or not all(
            isinstance(field, str) for field in selected_fields
        ):
            raise ValueError("selectedFields must be a string array")
        selected = set(selected_fields)
        if (
            not selected
            or len(selected) != len(selected_fields)
            or not selected.issubset(STYLE_FIELDS)
        ):
            raise ValueError("selectedFields must be a non-empty unique style field list")
        chapter, ordered = resolve_chapter_pages(
            self.engine,
            chapter_id=chapter_id,
            requested_page_ids=None,
            empty_message="job requires at least one page",
        )
        with self.engine.connect() as connection:
            source = connection.execute(
                select(
                    pages.c.document_revision,
                    pages.c.default_font_id,
                    pages.c.page_style_defaults_json,
                ).where(
                    pages.c.id == source_page_id,
                    pages.c.chapter_id == chapter_id,
                )
            ).mappings().one_or_none()
        if source is None:
            raise ValueError("source page does not belong to the chapter")
        if source["document_revision"] != source_document_revision:
            raise ValueError("source page document revision changed")
        defaults = validate_page_style(
            json.loads(source["page_style_defaults_json"]),
            partial=False,
        )
        frozen = {
            field: defaults.get(field)
            for field in selected
            if field != "fontFamily"
        }
        if "fontSize" in selected:
            frozen["autoFontSize"] = bool(
                defaults.get("autoFontSize", False)
            )
        if selected.intersection({"textColor", "fillColor"}):
            frozen["useAutoTextColor"] = bool(
                defaults.get("useAutoTextColor", False)
            )
        if "fontFamily" in selected:
            frozen["fontFamily"] = source["default_font_id"]
        config = {
            "sourcePageId": source_page_id,
            "sourceDocumentRevision": source_document_revision,
            "selectedFields": selected_fields,
            "frozenStyle": frozen,
            "executionMode": "sequential",
        }
        return self.jobs.create_batch(
            kind="style_apply",
            display_name=f"应用样式 {chapter['book_title']} / {chapter['title']}",
            specs=[
                JobSpec(
                    kind="style_apply",
                    book_id=str(chapter["book_id"]),
                    chapter_id=chapter_id,
                    config=config,
                    items=tuple(
                        JobItemSpec(
                            page_id=page_id,
                            step_kinds=(
                                "style_apply_document",
                                "render",
                                "save",
                            ),
                        )
                        for page_id in ordered
                    ),
                    target_display={
                        "book": chapter["book_title"],
                        "chapter": chapter["title"],
                        "pageCount": len(ordered),
                    },
                    font_snapshots=(
                        {"style": str(source["default_font_id"])}
                        if "fontFamily" in selected
                        and source["default_font_id"] is not None
                        else None
                    ),
                )
            ],
            idempotency_scope=f"style-apply:{chapter_id}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "chapterId": chapter_id,
                "sourcePageId": source_page_id,
                "sourceDocumentRevision": source_document_revision,
                "selectedFields": selected_fields,
                "frozenStyle": frozen,
            },
        )

    def export_text(self, chapter_id: str) -> dict[str, object]:
        with self.engine.connect() as connection:
            chapter = connection.execute(
                select(chapters.c.book_id).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter is None:
                raise ValueError("chapter not found")
            page_rows = list(
                connection.execute(
                    select(
                        pages.c.id,
                        pages.c.ordinal,
                        pages.c.document_revision,
                        assets.c.checksum,
                    )
                    .join(
                        page_assets,
                        (page_assets.c.page_id == pages.c.id)
                        & (page_assets.c.role == "source"),
                    )
                    .join(assets, assets.c.id == page_assets.c.asset_id)
                    .where(pages.c.chapter_id == chapter_id)
                    .order_by(pages.c.ordinal)
                ).mappings()
            )
            bubble_rows_by_page: dict[str, list[Mapping[str, Any]]] = {}
            for row in connection.execute(
                select(
                    bubbles,
                    pages.c.document_revision.label(
                        "page_document_revision"
                    ),
                )
                .join(pages, pages.c.id == bubbles.c.page_id)
                .where(pages.c.chapter_id == chapter_id)
                .order_by(pages.c.ordinal, bubbles.c.ordinal)
            ).mappings():
                bubble_rows_by_page.setdefault(str(row["page_id"]), []).append(
                    row
                )
            exported_pages: list[dict[str, object]] = []
            for page in page_rows:
                exported_pages.append(
                    {
                        "page_id": page["id"],
                        "page_number": page["ordinal"],
                        "source_checksum": page["checksum"],
                        "document_revision": page["document_revision"],
                        "bubbles": [
                            self._text_bubble(row)
                            for row in bubble_rows_by_page.get(
                                str(page["id"]),
                                (),
                            )
                        ],
                    }
                )
        from datetime import datetime, timezone

        return {
            "schema_version": 1,
            "book_id": chapter["book_id"],
            "chapter_id": chapter_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "pages": exported_pages,
        }

    def preview_text_import(
        self,
        *,
        chapter_id: str,
        document: Mapping[str, Any],
    ) -> dict[str, object]:
        _require_exact_fields(
            document,
            TEXT_EXPORT_ROOT_FIELDS,
            label="text import root",
        )
        if document.get("schema_version") != 1:
            raise ValueError("text import schema_version must be 1")
        if document.get("chapter_id") != chapter_id:
            raise ValueError("text import belongs to a different chapter")
        _require_non_empty_text(
            document.get("book_id"),
            label="text import book_id",
        )
        _require_non_empty_text(
            document.get("exported_at"),
            label="text import exported_at",
        )
        imported_pages = document.get("pages")
        if not isinstance(imported_pages, list):
            raise ValueError("text import pages must be an array")
        seen_pages: set[str] = set()
        for page_index, imported in enumerate(imported_pages):
            if not isinstance(imported, Mapping):
                raise ValueError("each imported page must be an object")
            _require_exact_fields(
                imported,
                TEXT_EXPORT_PAGE_FIELDS,
                label=f"text import pages[{page_index}]",
            )
            page_id = _require_non_empty_text(
                imported["page_id"],
                label=f"text import pages[{page_index}].page_id",
            )
            if page_id in seen_pages:
                raise ValueError("imported page IDs must be unique")
            seen_pages.add(page_id)
            _require_positive_integer(
                imported["page_number"],
                label=f"text import pages[{page_index}].page_number",
            )
            _require_non_empty_text(
                imported["source_checksum"],
                label=f"text import pages[{page_index}].source_checksum",
            )
            _require_positive_integer(
                imported["document_revision"],
                label=f"text import pages[{page_index}].document_revision",
            )
            imported_bubbles = imported["bubbles"]
            if not isinstance(imported_bubbles, list):
                raise ValueError("imported page bubbles must be an array")
            seen_bubbles: set[str] = set()
            for bubble_index, imported_bubble in enumerate(imported_bubbles):
                if not isinstance(imported_bubble, Mapping):
                    raise ValueError("each imported bubble must be an object")
                _require_exact_fields(
                    imported_bubble,
                    TEXT_EXPORT_BUBBLE_FIELDS,
                    label=(
                        f"text import pages[{page_index}]."
                        f"bubbles[{bubble_index}]"
                    ),
                )
                bubble_id = _require_non_empty_text(
                    imported_bubble["bubble_id"],
                    label=(
                        f"text import pages[{page_index}]."
                        f"bubbles[{bubble_index}].bubble_id"
                    ),
                )
                if bubble_id in seen_bubbles:
                    raise ValueError(
                        "imported bubble IDs must be unique per page"
                    )
                seen_bubbles.add(bubble_id)
                try:
                    validate_bubble_payload(
                        {
                            payload_key: imported_bubble[imported_key]
                            for imported_key, payload_key in TEXT_IMPORT_FIELDS.items()
                        },
                        render=False,
                        partial=True,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "imported bubble text fields are invalid"
                    ) from exc
        with self.engine.connect() as connection:
            chapter = connection.execute(
                select(chapters.c.book_id).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter is None:
                raise ValueError("chapter not found")
            if document["book_id"] != chapter["book_id"]:
                raise ValueError("text import belongs to a different book")
            current_pages = {
                str(row["id"]): row
                for row in connection.execute(
                    select(
                        pages.c.id,
                        pages.c.document_revision,
                        assets.c.checksum,
                        page_assets.c.asset_id,
                    )
                    .join(
                        page_assets,
                        (page_assets.c.page_id == pages.c.id)
                        & (page_assets.c.role == "source"),
                    )
                    .join(assets, assets.c.id == page_assets.c.asset_id)
                    .where(pages.c.chapter_id == chapter_id)
                ).mappings()
            }
            current_bubbles = {
                str(row["id"]): row
                for row in connection.execute(
                    select(
                        bubbles,
                        pages.c.document_revision.label(
                            "page_document_revision"
                        ),
                    )
                    .join(pages, pages.c.id == bubbles.c.page_id)
                    .where(pages.c.chapter_id == chapter_id)
                ).mappings()
            }
        results: list[dict[str, object]] = []
        for imported in imported_pages:
            page_id = imported["page_id"]
            current = current_pages.get(page_id)
            issues: list[str] = []
            changes: list[dict[str, object]] = []
            if current is None:
                issues.append("missing_page")
            else:
                if imported.get("source_checksum") != current["checksum"]:
                    issues.append("checksum_conflict")
                if (
                    imported["document_revision"]
                    != int(current["document_revision"])
                ):
                    issues.append("revision_conflict")
                imported_bubbles = imported["bubbles"]
                for imported_bubble in imported_bubbles:
                    bubble_id = imported_bubble["bubble_id"]
                    existing = current_bubbles.get(bubble_id)
                    if existing is None or str(existing["page_id"]) != page_id:
                        issues.append(f"missing_bubble:{bubble_id}")
                        continue
                    payload = _load_current_bubble_payload(
                        existing,
                        document_revision=int(
                            existing["page_document_revision"]
                        ),
                        label=f"bubble {bubble_id}",
                    )
                    fields: dict[str, object] = {}
                    differences: dict[str, dict[str, object]] = {}
                    for imported_key, payload_key in TEXT_IMPORT_FIELDS.items():
                        value = imported_bubble[imported_key]
                        current_value = _bubble_text(payload, payload_key)
                        if current_value != value:
                            fields[payload_key] = value
                            differences[payload_key] = {
                                "before": current_value,
                                "after": value,
                            }
                    if fields:
                        changes.append(
                            {
                                "bubbleId": bubble_id,
                                "fields": fields,
                                "differences": differences,
                            }
                        )
            results.append(
                {
                    "pageId": page_id,
                    "status": "match" if not issues else "conflict",
                    "issues": issues,
                    "baseDocumentRevision": (
                        current["document_revision"] if current else None
                    ),
                    "sourceChecksum": (
                        current["checksum"] if current else None
                    ),
                    "sourceAssetId": (
                        current["asset_id"] if current else None
                    ),
                    "changes": changes,
                }
            )
        return {
            "schemaVersion": 1,
            "chapterId": chapter_id,
            "pages": results,
            "matchedPages": sum(
                1 for row in results if row["status"] == "match"
            ),
            "conflictedPages": sum(
                1 for row in results if row["status"] != "match"
            ),
        }

    def create_text_import_job(
        self,
        *,
        chapter_id: str,
        confirmed_pages: list[Mapping[str, Any]],
        idempotency_key: str,
    ) -> dict[str, object]:
        if not confirmed_pages:
            raise ValueError("confirmedPages must contain at least one page")
        chapter, ordered = resolve_chapter_pages(
            self.engine,
            chapter_id=chapter_id,
            requested_page_ids=None,
            empty_message="job requires at least one page",
        )
        order_index = {page_id: index for index, page_id in enumerate(ordered)}
        normalized: list[dict[str, object]] = []
        seen: set[str] = set()
        for page_index, item in enumerate(confirmed_pages):
            _require_exact_fields(
                item,
                TEXT_IMPORT_PREVIEW_PAGE_FIELDS,
                label=f"confirmedPages[{page_index}]",
            )
            page_id = _require_non_empty_text(
                item["pageId"],
                label=f"confirmedPages[{page_index}].pageId",
            )
            if page_id in seen or page_id not in order_index:
                raise ValueError(
                    "confirmed page IDs must be unique members of the chapter"
                )
            seen.add(page_id)
            if item["status"] != "match":
                raise ValueError("confirmed pages must have match status")
            issues = item["issues"]
            if not isinstance(issues, list) or any(
                not isinstance(issue, str) for issue in issues
            ):
                raise ValueError("confirmed page issues must be a string array")
            if issues:
                raise ValueError("confirmed pages cannot contain conflicts")
            base_revision = _require_positive_integer(
                item["baseDocumentRevision"],
                label=f"confirmedPages[{page_index}].baseDocumentRevision",
            )
            source_checksum = _require_non_empty_text(
                item["sourceChecksum"],
                label=f"confirmedPages[{page_index}].sourceChecksum",
            )
            source_asset_id = _require_non_empty_text(
                item["sourceAssetId"],
                label=f"confirmedPages[{page_index}].sourceAssetId",
            )
            changes = item["changes"]
            if not isinstance(changes, list) or not changes:
                raise ValueError("each confirmed page requires changes")
            normalized_changes: list[dict[str, object]] = []
            seen_bubbles: set[str] = set()
            for change_index, change in enumerate(changes):
                if not isinstance(change, Mapping):
                    raise ValueError("each text change must be an object")
                _require_exact_fields(
                    change,
                    TEXT_IMPORT_CHANGE_FIELDS,
                    label=(
                        f"confirmedPages[{page_index}]."
                        f"changes[{change_index}]"
                    ),
                )
                bubble_id = _require_non_empty_text(
                    change["bubbleId"],
                    label=(
                        f"confirmedPages[{page_index}]."
                        f"changes[{change_index}].bubbleId"
                    ),
                )
                fields = change["fields"]
                if (
                    bubble_id in seen_bubbles
                    or not isinstance(fields, Mapping)
                    or not fields
                    or not set(fields).issubset(TEXT_IMPORT_FIELDS.values())
                ):
                    raise ValueError("text change bubbleId/fields are invalid")
                try:
                    normalized_fields = validate_bubble_payload(
                        fields,
                        render=False,
                        partial=True,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("text change fields are invalid") from exc
                differences = change["differences"]
                if not isinstance(differences, Mapping) or set(
                    differences
                ) != set(fields):
                    raise ValueError(
                        "text change differences must match changed fields"
                    )
                for difference in differences.values():
                    if not isinstance(difference, Mapping) or set(
                        difference
                    ) != {"before", "after"}:
                        raise ValueError("text change difference is invalid")
                seen_bubbles.add(bubble_id)
                normalized_changes.append(
                    {"bubbleId": bubble_id, "fields": normalized_fields}
                )
            normalized.append(
                {
                    "pageId": page_id,
                    "baseDocumentRevision": base_revision,
                    "sourceChecksum": source_checksum,
                    "sourceAssetId": source_asset_id,
                    "changes": normalized_changes,
                }
            )
        normalized.sort(key=lambda item: order_index[str(item["pageId"])])
        with self.engine.connect() as connection:
            current_pages = {
                str(row["id"]): row
                for row in connection.execute(
                    select(
                        pages.c.id,
                        pages.c.document_revision,
                        page_assets.c.asset_id,
                        assets.c.checksum,
                    )
                    .join(
                        page_assets,
                        (page_assets.c.page_id == pages.c.id)
                        & (page_assets.c.role == "source"),
                    )
                    .join(assets, assets.c.id == page_assets.c.asset_id)
                    .where(
                        pages.c.id.in_(
                            [str(item["pageId"]) for item in normalized]
                        )
                    )
                ).mappings()
            }
            current_bubbles = {
                str(row["id"]): row
                for row in connection.execute(
                    select(
                        bubbles.c.id,
                        bubbles.c.page_id,
                        bubbles.c.updated_revision,
                    ).where(
                        bubbles.c.page_id.in_(
                            [str(item["pageId"]) for item in normalized]
                        )
                    )
                ).mappings()
            }
        for item in normalized:
            page_id = str(item["pageId"])
            current = current_pages.get(page_id)
            if (
                current is None
                or int(current["document_revision"])
                != int(item["baseDocumentRevision"])
                or str(current["asset_id"]) != item["sourceAssetId"]
                or str(current["checksum"]) != item["sourceChecksum"]
            ):
                raise ValueError(
                    f"page {page_id} no longer matches the preview"
                )
            for change in item["changes"]:  # type: ignore[union-attr]
                bubble = current_bubbles.get(str(change["bubbleId"]))
                if bubble is None or str(bubble["page_id"]) != page_id:
                    raise ValueError(
                        f"page {page_id} contains a missing or moved bubble"
                    )
                if bubble["updated_revision"] != current["document_revision"]:
                    raise ValueError(
                        f"page {page_id} bubble document is not current"
                    )
        config = {
            "pages": normalized,
            "executionMode": "sequential",
        }
        return self.jobs.create_batch(
            kind="text_import",
            display_name=f"导入文本 {chapter['book_title']} / {chapter['title']}",
            specs=[
                JobSpec(
                    kind="text_import",
                    book_id=str(chapter["book_id"]),
                    chapter_id=chapter_id,
                    config=config,
                    items=tuple(
                        JobItemSpec(
                            page_id=str(item["pageId"]),
                            step_kinds=(
                                "text_import_apply",
                                "render",
                                "save",
                            ),
                            asset_inputs={
                                "source": str(item["sourceAssetId"])
                            },
                        )
                        for item in normalized
                    ),
                    target_display={
                        "book": chapter["book_title"],
                        "chapter": chapter["title"],
                        "pageCount": len(normalized),
                    },
                )
            ],
            idempotency_scope=f"text-import:{chapter_id}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "chapterId": chapter_id,
                "pages": normalized,
            },
        )

    @staticmethod
    def _text_bubble(row: Mapping[str, Any]) -> dict[str, object]:
        payload = _load_current_bubble_payload(
            row,
            document_revision=int(row["page_document_revision"]),
            label=f"bubble {row['id']}",
        )
        return {
            "bubble_id": row["id"],
            "original_text": _bubble_text(payload, "originalText"),
            "translated_text": _bubble_text(payload, "translatedText"),
            "textbox_text": _bubble_text(payload, "textboxText"),
            "text_direction": _bubble_text(payload, "textDirection"),
        }

class StyleApplyWorkerService:
    def __init__(self, *, engine: Engine, jobs: JobQueueRepository) -> None:
        self.engine = engine
        self.jobs = jobs

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page_id = step.get("pageId")
        if not isinstance(page_id, str) or not page_id:
            raise RuntimeError("style job page is invalid")
        config = step.get("config")
        if not isinstance(config, Mapping) or set(config) != {
            "sourcePageId",
            "sourceDocumentRevision",
            "selectedFields",
            "frozenStyle",
            "executionMode",
        }:
            raise RuntimeError("style job configuration is invalid")
        selected_fields = config["selectedFields"]
        frozen_value = config["frozenStyle"]
        if (
            config["executionMode"] != "sequential"
            or not isinstance(selected_fields, list)
            or not all(isinstance(field, str) for field in selected_fields)
            or not isinstance(frozen_value, Mapping)
        ):
            raise RuntimeError("style job configuration is invalid")
        if (
            not isinstance(config["sourcePageId"], str)
            or not config["sourcePageId"]
        ):
            raise RuntimeError("style source page is invalid")
        if (
            isinstance(config["sourceDocumentRevision"], bool)
            or not isinstance(config["sourceDocumentRevision"], int)
            or config["sourceDocumentRevision"] < 1
        ):
            raise RuntimeError("style source revision is invalid")
        selected = set(selected_fields)
        if (
            not selected
            or len(selected) != len(selected_fields)
            or not selected.issubset(STYLE_FIELDS)
        ):
            raise RuntimeError("style selected fields are invalid")
        frozen = dict(frozen_value)
        expected_frozen = set(selected)
        if "fontSize" in selected:
            expected_frozen.add("autoFontSize")
        if selected.intersection({"textColor", "fillColor"}):
            expected_frozen.add("useAutoTextColor")
        if set(frozen) != expected_frozen:
            raise RuntimeError("style snapshot fields are invalid")
        font_id = frozen.get("fontFamily")
        if font_id is not None and (
            not isinstance(font_id, str) or not font_id
        ):
            raise RuntimeError("style snapshot font is invalid")
        try:
            validate_page_style(
                {
                    key: item
                    for key, item in frozen.items()
                    if key != "fontFamily"
                },
                partial=True,
            )
        except ValueError as exc:
            raise RuntimeError("style snapshot is invalid") from exc
        with self.engine.connect() as connection:
            page = connection.execute(
                select(pages).where(pages.c.id == page_id)
            ).mappings().one_or_none()
            if page is None:
                raise RuntimeError("style target page was deleted")
            bubble_rows = list(
                connection.execute(
                    select(bubbles)
                    .where(bubbles.c.page_id == page_id)
                    .order_by(bubbles.c.ordinal)
                ).mappings()
            )
            has_translated_asset = (
                connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none()
                is not None
            )
        defaults = validate_page_style(
            json.loads(page["page_style_defaults_json"]),
            partial=False,
        )
        new_defaults = dict(defaults)
        default_font_id = page["default_font_id"]
        for field in selected:
            if field == "fontFamily":
                default_font_id = frozen.get(field)
            elif (
                field == "fontSize"
                and bool(frozen.get("autoFontSize", False))
            ):
                # Automatic mode is shared, but every target page keeps its own
                # fixed-size fallback for a later switch back to manual mode.
                continue
            elif (
                field in {"textColor", "fillColor"}
                and bool(frozen.get("useAutoTextColor", False))
            ):
                # Automatic mode consumes each target bubble's own backups. The
                # target page's manual fallback colors remain page-local facts.
                continue
            else:
                new_defaults[field] = frozen.get(field)
        if "fontSize" in selected:
            new_defaults["autoFontSize"] = bool(
                frozen.get("autoFontSize", False)
            )
        if selected.intersection({"textColor", "fillColor"}):
            new_defaults["useAutoTextColor"] = bool(
                frozen.get("useAutoTextColor", False)
            )
        updated_payloads: list[tuple[str, dict[str, Any]]] = []
        changed = (
            new_defaults != defaults
            or default_font_id != page["default_font_id"]
        )
        render_changed = False
        has_drawable_text = False
        for row in bubble_rows:
            payload = _load_current_bubble_payload(
                row,
                document_revision=int(page["document_revision"]),
                label=f"style target bubble {row['id']}",
            )
            updated = dict(payload)
            if (
                "fontFamily" in selected
                and row["font_id"] != default_font_id
            ):
                changed = True
                render_changed = True
            for field in selected:
                if field == "fontFamily":
                    continue
                value = frozen.get(field)
                if field == "layoutDirection":
                    if value == "auto":
                        direction = updated["autoTextDirection"]
                    else:
                        direction = value
                    updated["textDirection"] = direction
                elif (
                    field == "fontSize"
                    and bool(frozen.get("autoFontSize", False))
                ):
                    if _bubble_text(
                        updated,
                        "translatedText",
                    ).strip():
                        # The following render step recalculates and persists the
                        # concrete size. Force a fresh revision even when the
                        # target page was already in automatic mode.
                        changed = True
                        render_changed = True
                elif (
                    field in {"textColor", "fillColor"}
                    and bool(frozen.get("useAutoTextColor", False))
                ):
                    automatic_field = (
                        "autoFgColor"
                        if field == "textColor"
                        else "autoBgColor"
                    )
                    automatic = updated.get(automatic_field)
                    if automatic is not None:
                        updated[field] = rgb_to_hex(automatic)
                    # Missing automatic backup preserves the target bubble's
                    # current effective value instead of copying a source-page
                    # manual fallback.
                else:
                    updated[field] = value
                if (
                    field in RENDER_STYLE_FIELDS
                    and field != "fontFamily"
                    and updated.get(
                        "textDirection"
                        if field == "layoutDirection"
                        else field
                    )
                    != payload.get(
                        "textDirection"
                        if field == "layoutDirection"
                        else field
                    )
                ):
                    render_changed = True
            changed = changed or updated != payload
            has_drawable_text = has_drawable_text or bool(
                _bubble_text(updated, "translatedText").strip()
            )
            updated_payloads.append((str(row["id"]), updated))
        needs_render = bool(
            render_changed
            and (has_translated_asset or has_drawable_text)
        )
        base_revision = int(page["document_revision"])
        new_revision = base_revision + 1 if changed else base_revision

        def publish(connection: Connection) -> None:
            if connection.execute(
                select(chapter_write_locks.c.job_id).where(
                    chapter_write_locks.c.chapter_id == page["chapter_id"],
                    chapter_write_locks.c.job_id == fence.job_id,
                )
            ).scalar_one_or_none() is None:
                raise RuntimeError("style job lost its chapter lock")
            if changed:
                for bubble_id, payload in updated_payloads:
                    values: dict[str, object] = {
                        "payload_json": canonical_json(payload),
                        "updated_revision": new_revision,
                    }
                    if "fontFamily" in selected:
                        values["font_id"] = default_font_id
                    bubble_changed = connection.execute(
                        update(bubbles)
                        .where(
                            bubbles.c.id == bubble_id,
                            bubbles.c.page_id == page_id,
                            bubbles.c.updated_revision == base_revision,
                        )
                        .values(**values)
                    )
                    if bubble_changed.rowcount != 1:
                        raise RuntimeError(
                            "style target bubble changed during publication"
                        )
                page_changed = connection.execute(
                    update(pages)
                    .where(
                        pages.c.id == page_id,
                        pages.c.document_revision == base_revision,
                    )
                    .values(
                        default_font_id=default_font_id,
                        page_style_defaults_json=canonical_json(new_defaults),
                        document_revision=new_revision,
                        rendered_revision=(
                            new_revision
                            if (
                                not needs_render
                                and page["render_status"] == "ready"
                                and page["rendered_revision"] == base_revision
                            )
                            else page["rendered_revision"]
                        ),
                        render_status=(
                            "stale" if needs_render else page["render_status"]
                        ),
                    )
                )
                if page_changed.rowcount != 1:
                    raise RuntimeError(
                        "style target page changed during publication"
                    )
                if (
                    not needs_render
                    and page["render_status"] == "ready"
                    and page["rendered_revision"] == base_revision
                ):
                    pointer_changed = connection.execute(
                        update(page_assets)
                        .where(
                            page_assets.c.page_id == page_id,
                            page_assets.c.role == "translated",
                            page_assets.c.input_document_revision
                            == base_revision,
                        )
                        .values(input_document_revision=new_revision)
                    )
                    if pointer_changed.rowcount != 1:
                        raise RuntimeError(
                            "current translated asset is missing"
                        )
            if not needs_render:
                connection.execute(
                    update(job_steps)
                    .where(
                        job_steps.c.job_item_id == step["itemId"],
                        job_steps.c.kind.in_(("render", "save")),
                        job_steps.c.status == "pending",
                    )
                    .values(status="skipped")
                )

        checkpoint = {
            "changed": changed,
            "renderRequired": needs_render,
            "documentRevision": new_revision,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}


class TextImportWorkerService:
    def __init__(self, *, engine: Engine, jobs: JobQueueRepository) -> None:
        self.engine = engine
        self.jobs = jobs

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        page_id = step.get("pageId")
        if not isinstance(page_id, str) or not page_id:
            raise RuntimeError("text import page is invalid")
        config = step.get("config")
        if (
            not isinstance(config, Mapping)
            or set(config) != {"pages", "executionMode"}
            or config["executionMode"] != "sequential"
            or not isinstance(config["pages"], list)
        ):
            raise RuntimeError("text import snapshot is invalid")
        page_entries = config["pages"]
        if not all(
            isinstance(value, Mapping)
            and set(value)
            == {
                "pageId",
                "baseDocumentRevision",
                "sourceChecksum",
                "sourceAssetId",
                "changes",
            }
            for value in page_entries
        ):
            raise RuntimeError("text import page snapshot is invalid")
        page_ids = [value["pageId"] for value in page_entries]
        if (
            not all(isinstance(value, str) and value for value in page_ids)
            or len(page_ids) != len(set(page_ids))
        ):
            raise RuntimeError("text import page snapshot is invalid")
        entry = next(
            (value for value in page_entries if value["pageId"] == page_id),
            None,
        )
        if entry is None or set(entry) != {
            "pageId",
            "baseDocumentRevision",
            "sourceChecksum",
            "sourceAssetId",
            "changes",
        }:
            raise RuntimeError("text import page snapshot is invalid")
        base_revision = entry["baseDocumentRevision"]
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 1
            or not isinstance(entry["sourceChecksum"], str)
            or not entry["sourceChecksum"]
            or not isinstance(entry["sourceAssetId"], str)
            or not entry["sourceAssetId"]
            or not isinstance(entry["changes"], list)
            or not entry["changes"]
        ):
            raise RuntimeError("text import page snapshot is invalid")
        with self.engine.connect() as connection:
            page = connection.execute(
                select(
                    pages.c.chapter_id,
                    pages.c.document_revision,
                    pages.c.rendered_revision,
                    pages.c.render_status,
                    page_assets.c.asset_id,
                    assets.c.checksum,
                )
                .join(
                    page_assets,
                    (page_assets.c.page_id == pages.c.id)
                    & (page_assets.c.role == "source"),
                )
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(pages.c.id == page_id)
            ).mappings().one_or_none()
            bubble_rows = {
                str(row["id"]): row
                for row in connection.execute(
                    select(bubbles).where(bubbles.c.page_id == page_id)
                ).mappings()
            }
            has_translated_asset = (
                connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none()
                is not None
            )
        if page is None:
            raise RuntimeError("text import page was deleted")
        if (
            page["document_revision"] != base_revision
            or page["asset_id"] != entry["sourceAssetId"]
            or page["checksum"] != entry["sourceChecksum"]
        ):
            raise RuntimeError("text import page changed after preview")
        bubble_payloads = {
            bubble_id: _load_current_bubble_payload(
                row,
                document_revision=base_revision,
                label=f"text import bubble {bubble_id}",
            )
            for bubble_id, row in bubble_rows.items()
        }
        updates: list[tuple[str, dict[str, Any]]] = []
        changed_fields: set[str] = set()
        seen_bubbles: set[str] = set()
        for change in entry["changes"]:
            if not isinstance(change, Mapping) or set(change) != {
                "bubbleId",
                "fields",
            }:
                raise RuntimeError("text import change snapshot is invalid")
            bubble_id = change["bubbleId"]
            fields = change["fields"]
            if (
                not isinstance(bubble_id, str)
                or not bubble_id
                or bubble_id in seen_bubbles
                or not isinstance(fields, Mapping)
                or not fields
                or not set(fields).issubset(TEXT_IMPORT_FIELDS.values())
            ):
                raise RuntimeError("text import change snapshot is invalid")
            seen_bubbles.add(bubble_id)
            current = bubble_rows.get(bubble_id)
            if current is None:
                raise RuntimeError(
                    f"text import bubble {bubble_id} changed after preview"
                )
            payload = dict(bubble_payloads[bubble_id])
            payload.update(dict(fields))
            try:
                payload = validate_bubble_payload(payload, render=False)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "text import bubble payload is invalid"
                ) from exc
            changed_fields.update(fields)
            updates.append((bubble_id, payload))
        render_changed = bool(
            changed_fields.intersection({"translatedText", "textDirection"})
        )
        updated_by_id = dict(updates)
        has_drawable_text = any(
            _bubble_text(
                updated_by_id.get(bubble_id, payload),
                "translatedText",
            ).strip()
            for bubble_id, payload in bubble_payloads.items()
        )
        needs_render = bool(
            render_changed and (has_translated_asset or has_drawable_text)
        )
        new_revision = base_revision + 1

        def publish(connection: Connection) -> None:
            if connection.execute(
                select(chapter_write_locks.c.job_id).where(
                    chapter_write_locks.c.chapter_id == page["chapter_id"],
                    chapter_write_locks.c.job_id == fence.job_id,
                )
            ).scalar_one_or_none() is None:
                raise RuntimeError("text import lost its chapter lock")
            for bubble_id, payload in updates:
                bubble_changed = connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.page_id == page_id,
                        bubbles.c.updated_revision == base_revision,
                    )
                    .values(
                        payload_json=canonical_json(payload),
                        updated_revision=new_revision,
                    )
                )
                if bubble_changed.rowcount != 1:
                    raise RuntimeError(
                        f"text import bubble {bubble_id} changed after preview"
                    )
            changed_ids = [bubble_id for bubble_id, _payload in updates]
            untouched = connection.execute(
                update(bubbles)
                .where(
                    bubbles.c.page_id == page_id,
                    bubbles.c.updated_revision == base_revision,
                    bubbles.c.id.not_in(changed_ids),
                )
                .values(updated_revision=new_revision)
            )
            if untouched.rowcount != len(bubble_rows) - len(updates):
                raise RuntimeError(
                    "text import bubble set changed after preview"
                )
            changed = connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == base_revision,
                )
                .values(
                    document_revision=new_revision,
                    rendered_revision=(
                        new_revision
                        if (
                            not needs_render
                            and page["render_status"] == "ready"
                            and page["rendered_revision"] == base_revision
                        )
                        else page["rendered_revision"]
                    ),
                    render_status=(
                        "stale" if needs_render else page["render_status"]
                    ),
                )
            )
            if changed.rowcount != 1:
                raise RuntimeError("text import page changed after preview")
            if (
                not needs_render
                and page["render_status"] == "ready"
                and page["rendered_revision"] == base_revision
            ):
                pointer_changed = connection.execute(
                    update(page_assets)
                    .where(
                        page_assets.c.page_id == page_id,
                        page_assets.c.role == "translated",
                        page_assets.c.input_document_revision == base_revision,
                    )
                    .values(input_document_revision=new_revision)
                )
                if pointer_changed.rowcount != 1:
                    raise RuntimeError("current translated asset is missing")
            if not needs_render:
                connection.execute(
                    update(job_steps)
                    .where(
                        job_steps.c.job_item_id == step["itemId"],
                        job_steps.c.kind.in_(("render", "save")),
                        job_steps.c.status == "pending",
                    )
                    .values(status="skipped")
                )

        checkpoint = {
            "pageId": page_id,
            "documentRevision": new_revision,
            "changedBubbles": len(updates),
            "renderRequired": needs_render,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}
