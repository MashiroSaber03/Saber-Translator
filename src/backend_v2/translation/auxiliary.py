"""Detect/style job commands and Worker-side style application."""

from __future__ import annotations

import json
from typing import Any, Mapping

from sqlalchemy import Engine, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json
from src.backend_v2.content.page_style import rgb_to_hex, validate_page_style
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
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


STYLE_FIELDS = frozenset(
    {
        "fontSize",
        "fontFamily",
        "layoutDirection",
        "lineSpacing",
        "textAlign",
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


class AuxiliaryTranslationCommands:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)

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
            "executionMode": "parallel",
            "settingsSnapshot": dict(resolved["settingsSnapshot"]),
        }
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
                select(bubbles)
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
        if document.get("schema_version") != 1:
            raise ValueError("text import schema_version must be 1")
        if document.get("chapter_id") != chapter_id:
            raise ValueError("text import belongs to a different chapter")
        imported_pages = document.get("pages")
        if not isinstance(imported_pages, list):
            raise ValueError("text import pages must be an array")
        with self.engine.connect() as connection:
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
                    select(bubbles)
                    .join(pages, pages.c.id == bubbles.c.page_id)
                    .where(pages.c.chapter_id == chapter_id)
                ).mappings()
            }
        results: list[dict[str, object]] = []
        seen_pages: set[str] = set()
        for imported in imported_pages:
            if not isinstance(imported, Mapping):
                raise ValueError("each imported page must be an object")
            page_id = str(imported.get("page_id", ""))
            if not page_id or page_id in seen_pages:
                raise ValueError("imported page IDs must be non-empty and unique")
            seen_pages.add(page_id)
            current = current_pages.get(page_id)
            issues: list[str] = []
            changes: list[dict[str, object]] = []
            if current is None:
                issues.append("missing_page")
            else:
                if imported.get("source_checksum") != current["checksum"]:
                    issues.append("checksum_conflict")
                if (
                    int(imported.get("document_revision", 0))
                    != int(current["document_revision"])
                ):
                    issues.append("revision_conflict")
                imported_bubbles = imported.get("bubbles")
                if not isinstance(imported_bubbles, list):
                    raise ValueError("imported page bubbles must be an array")
                seen_bubbles: set[str] = set()
                for imported_bubble in imported_bubbles:
                    if not isinstance(imported_bubble, Mapping):
                        raise ValueError("each imported bubble must be an object")
                    bubble_id = str(imported_bubble.get("bubble_id", ""))
                    if not bubble_id or bubble_id in seen_bubbles:
                        raise ValueError(
                            "imported bubble IDs must be non-empty and unique per page"
                        )
                    seen_bubbles.add(bubble_id)
                    existing = current_bubbles.get(bubble_id)
                    if existing is None or str(existing["page_id"]) != page_id:
                        issues.append(f"missing_bubble:{bubble_id}")
                        continue
                    payload = json.loads(existing["payload_json"])
                    fields: dict[str, object] = {}
                    differences: dict[str, dict[str, object]] = {}
                    for imported_key, payload_key in TEXT_IMPORT_FIELDS.items():
                        if imported_key not in imported_bubble:
                            continue
                        value = imported_bubble[imported_key]
                        if not isinstance(value, str):
                            raise ValueError(
                                f"{imported_key} must be a string"
                            )
                        if payload.get(payload_key, "") != value:
                            fields[payload_key] = value
                            differences[payload_key] = {
                                "before": payload.get(payload_key, ""),
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
        for item in confirmed_pages:
            page_id = str(item.get("pageId", ""))
            if page_id in seen or page_id not in order_index:
                raise ValueError(
                    "confirmed page IDs must be unique members of the chapter"
                )
            seen.add(page_id)
            changes = item.get("changes")
            if not isinstance(changes, list) or not changes:
                raise ValueError("each confirmed page requires changes")
            normalized_changes: list[dict[str, object]] = []
            seen_bubbles: set[str] = set()
            for change in changes:
                if not isinstance(change, Mapping):
                    raise ValueError("each text change must be an object")
                bubble_id = str(change.get("bubbleId", ""))
                fields = change.get("fields")
                if (
                    not bubble_id
                    or bubble_id in seen_bubbles
                    or not isinstance(fields, Mapping)
                    or not fields
                    or not set(fields).issubset(TEXT_IMPORT_FIELDS.values())
                    or not all(isinstance(value, str) for value in fields.values())
                ):
                    raise ValueError("text change bubbleId/fields are invalid")
                seen_bubbles.add(bubble_id)
                normalized_changes.append(
                    {"bubbleId": bubble_id, "fields": dict(fields)}
                )
            normalized.append(
                {
                    "pageId": page_id,
                    "baseDocumentRevision": int(
                        item.get("baseDocumentRevision", 0)
                    ),
                    "sourceChecksum": str(item.get("sourceChecksum", "")),
                    "sourceAssetId": str(item.get("sourceAssetId", "")),
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
            bubble_owners = {
                str(row["id"]): str(row["page_id"])
                for row in connection.execute(
                    select(bubbles.c.id, bubbles.c.page_id).where(
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
            if any(
                bubble_owners.get(str(change["bubbleId"])) != page_id
                for change in item["changes"]  # type: ignore[union-attr]
            ):
                raise ValueError(
                    f"page {page_id} contains a missing or moved bubble"
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
        payload = json.loads(row["payload_json"])
        return {
            "bubble_id": row["id"],
            "original_text": payload.get("originalText", ""),
            "translated_text": payload.get("translatedText", ""),
            "textbox_text": payload.get("textboxText", ""),
            "text_direction": payload.get("textDirection", "vertical"),
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
        page_id = str(step["pageId"])
        config = step.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("style job configuration is invalid")
        selected = set(config.get("selectedFields", []))
        frozen = config.get("frozenStyle")
        if not selected.issubset(STYLE_FIELDS) or not isinstance(frozen, Mapping):
            raise RuntimeError("style snapshot is invalid")
        with self.engine.connect() as connection:
            page = connection.execute(
                select(pages).where(pages.c.id == page_id)
            ).mappings().one()
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
            payload = json.loads(row["payload_json"])
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
                    direction = (
                        updated.get("autoTextDirection", "vertical")
                        if value == "auto"
                        else value
                    )
                    updated["textDirection"] = (
                        direction
                        if direction in {"vertical", "horizontal"}
                        else "vertical"
                    )
                elif (
                    field == "fontSize"
                    and bool(frozen.get("autoFontSize", False))
                ):
                    if str(updated.get("translatedText", "")).strip():
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
                str(updated.get("translatedText", "")).strip()
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
                    chapter_write_locks.c.owner_attempt_id == fence.attempt_id,
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
                    connection.execute(
                        update(bubbles)
                        .where(
                            bubbles.c.id == bubble_id,
                            bubbles.c.page_id == page_id,
                        )
                        .values(**values)
                    )
                connection.execute(
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
                if (
                    not needs_render
                    and page["render_status"] == "ready"
                    and page["rendered_revision"] == base_revision
                ):
                    connection.execute(
                        update(page_assets)
                        .where(
                            page_assets.c.page_id == page_id,
                            page_assets.c.role == "translated",
                            page_assets.c.input_document_revision
                            == base_revision,
                        )
                        .values(input_document_revision=new_revision)
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
        config = step.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("text import snapshot is invalid")
        entries = config.get("pages")
        if not isinstance(entries, list):
            raise RuntimeError("text import page snapshot is missing")
        page_id = str(step["pageId"])
        entry = next(
            (
                value
                for value in entries
                if isinstance(value, Mapping)
                and value.get("pageId") == page_id
            ),
            None,
        )
        if entry is None:
            raise RuntimeError("text import page snapshot is missing")
        base_revision = int(entry["baseDocumentRevision"])
        with self.engine.connect() as connection:
            page = connection.execute(
                select(
                    pages.c.chapter_id,
                    pages.c.document_revision,
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
        if page is None:
            raise RuntimeError("text import page was deleted")
        if (
            page["document_revision"] != base_revision
            or page["checksum"] != entry["sourceChecksum"]
        ):
            raise RuntimeError("text import page changed after preview")
        updates: list[tuple[str, dict[str, Any]]] = []
        for change in entry["changes"]:
            bubble_id = str(change["bubbleId"])
            current = bubble_rows.get(bubble_id)
            if current is None:
                raise RuntimeError(
                    f"text import bubble {bubble_id} changed after preview"
                )
            payload = json.loads(current["payload_json"])
            payload.update(dict(change["fields"]))
            updates.append((bubble_id, payload))
        new_revision = base_revision + 1

        def publish(connection: Connection) -> None:
            if connection.execute(
                select(chapter_write_locks.c.job_id).where(
                    chapter_write_locks.c.chapter_id == page["chapter_id"],
                    chapter_write_locks.c.job_id == fence.job_id,
                    chapter_write_locks.c.owner_attempt_id
                    == fence.attempt_id,
                )
            ).scalar_one_or_none() is None:
                raise RuntimeError("text import lost its chapter lock")
            for bubble_id, payload in updates:
                connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.id == bubble_id,
                        bubbles.c.page_id == page_id,
                    )
                    .values(
                        payload_json=canonical_json(payload),
                        updated_revision=new_revision,
                    )
                )
            changed = connection.execute(
                update(pages)
                .where(
                    pages.c.id == page_id,
                    pages.c.document_revision == base_revision,
                )
                .values(
                    document_revision=new_revision,
                    render_status="stale",
                )
            )
            if changed.rowcount != 1:
                raise RuntimeError("text import page changed after preview")

        checkpoint = {
            "pageId": page_id,
            "documentRevision": new_revision,
            "changedBubbles": len(updates),
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}
