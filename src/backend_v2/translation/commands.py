"""Closed translation job commands; there is intentionally no generic POST /jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

from sqlalchemy import Engine, func, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.content.page_style import validate_page_style
from src.backend_v2.jobs.repository import (
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.storage.schema import (
    NONTERMINAL_JOB_STATUSES,
    books,
    bubbles,
    chapters,
    jobs,
    page_assets,
    pages,
    render_requests,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.public_policy import PublicUserPolicyAccess
from src.backend_v2.runtime_profile import RuntimeProfile
from src.backend_v2.timestamps import utcnow
from src.shared.ai_providers import (
    HQ_TRANSLATION_CAPABILITY,
    TRANSLATION_CAPABILITY,
    VISION_OCR_CAPABILITY,
    get_provider_manifest,
    provider_requires_api_key,
)
from src.shared.paddleocr_vl import PADDLEOCR_VL_LANGUAGE_NAMES


ALLOWED_MODES = frozenset({"standard", "hq", "proofread", "remove_text"})
ALLOWED_EXECUTION_MODES = frozenset({"sequential", "parallel"})
ALLOWED_CONFIG_KEYS = frozenset(
    {
        "mode",
        "executionMode",
        "skipCompleted",
        "reuseExistingBubbles",
        "styleSourcePageId",
        "styleSourceDocumentRevision",
    }
)


def resolve_chapter_pages(
    engine: Engine,
    *,
    chapter_id: str,
    requested_page_ids: Sequence[str] | None,
    empty_message: str = "translation task requires at least one page",
) -> tuple[Mapping[str, Any], list[str]]:
    """Return one chapter and its requested pages in persisted order."""

    if not isinstance(chapter_id, str) or not chapter_id:
        raise ValueError("chapterId must be a non-empty string")
    if requested_page_ids is not None and any(
        not isinstance(page_id, str) or not page_id
        for page_id in requested_page_ids
    ):
        raise ValueError("pageIds must contain non-empty strings")

    with engine.connect() as connection:
        chapter = connection.execute(
            select(
                chapters.c.id,
                chapters.c.book_id,
                chapters.c.title,
                books.c.title.label("book_title"),
            )
            .join(books, books.c.id == chapters.c.book_id)
            .where(
                chapters.c.id == chapter_id,
                books.c.owner_user_id == effective_owner_id(),
            )
        ).mappings().one_or_none()
        if chapter is None:
            raise ValueError("chapter not found")
        ordered = [
            str(value)
            for value in connection.execute(
                select(pages.c.id)
                .where(pages.c.chapter_id == chapter_id)
                .order_by(pages.c.ordinal)
            ).scalars()
        ]
    if requested_page_ids is not None:
        if not requested_page_ids or len(set(requested_page_ids)) != len(
            requested_page_ids
        ):
            raise ValueError("pageIds must contain unique page IDs")
        requested = set(requested_page_ids)
        if not requested.issubset(set(ordered)):
            raise ValueError("pageIds must all belong to the chapter")
        ordered = [page_id for page_id in ordered if page_id in requested]
    if not ordered:
        raise ValueError(empty_message)
    return chapter, ordered


class TranslationJobCommandService:
    def __init__(
        self,
        engine: Engine,
        *,
        profile: RuntimeProfile,
    ) -> None:
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)
        self.public_access = PublicUserPolicyAccess(engine, profile)

    def create_chapter_job(
        self,
        *,
        chapter_id: str,
        config: Mapping[str, Any],
        page_ids: Sequence[str] | None,
        idempotency_key: str,
        retry_of_job_id: str | None = None,
        retry_mode: str | None = None,
        idempotency_scope: str | None = None,
    ) -> dict[str, object]:
        command = normalize_translation_command(config)
        self.public_access.enforce_translation_command(command)
        mode = command["mode"]
        job_kind = "remove_text" if mode == "remove_text" else "translation"
        _chapter, ordered_pages = resolve_chapter_pages(
            self.engine,
            chapter_id=chapter_id,
            requested_page_ids=page_ids,
        )
        request_payload = {
            "chapterId": chapter_id,
            "pageIds": ordered_pages,
            "config": command,
            "retryOfJobId": retry_of_job_id,
            "retryMode": retry_mode,
        }
        scope = idempotency_scope or f"chapter-translation:{chapter_id}"
        replay = self.jobs.idempotency_replay(
            scope=scope,
            key=idempotency_key,
            payload=request_payload,
        )
        if replay is not None:
            return replay
        chapter, ordered_pages, normalized, spec = self._translation_spec(
            chapter_id=chapter_id,
            requested_page_ids=page_ids,
            command=command,
            job_kind=job_kind,
            retry_of_job_id=retry_of_job_id,
            retry_mode=retry_mode,
        )
        return self.jobs.create_batch(
            display_name=f"{chapter['book_title']} / {chapter['title']}",
            specs=[spec],
            idempotency_scope=scope,
            idempotency_key=idempotency_key,
            idempotency_payload=request_payload,
            transaction_initializer=lambda connection, _batch_id: (
                self._materialize_text_styles(connection, (spec,))
            ),
        )

    def create_batch(
        self,
        *,
        chapter_ids: Sequence[str] | None = None,
        book_ids: Sequence[str] | None = None,
        config: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, object]:
        if (chapter_ids is None) == (book_ids is None):
            raise ValueError("provide exactly one of chapterIds or bookIds")
        requested_field: str
        requested_ids: list[str]
        if book_ids is not None:
            requested_field = "bookIds"
            requested_ids = list(book_ids)
        else:
            requested_field = "chapterIds"
            requested_ids = list(chapter_ids or ())
        if (
            not requested_ids
            or any(not isinstance(value, str) or not value for value in requested_ids)
            or len(set(requested_ids)) != len(requested_ids)
        ):
            raise ValueError(
                f"{requested_field} must contain unique "
                f"{'book' if requested_field == 'bookIds' else 'chapter'} IDs"
            )
        command = normalize_translation_command(config)
        self.public_access.enforce_translation_command(command)
        mode = command["mode"]
        job_kind = "remove_text" if mode == "remove_text" else "translation"
        idempotency_payload = {
            requested_field: requested_ids,
            "config": command,
        }
        replay = self.jobs.idempotency_replay(
            scope="translation-batch",
            key=idempotency_key,
            payload=idempotency_payload,
        )
        if replay is not None:
            return replay
        if book_ids is not None:
            chapter_ids = self._resolve_book_chapter_ids(requested_ids)
        else:
            chapter_ids = requested_ids
        specs: list[JobSpec] = []
        skipped: list[dict[str, str]] = []
        with self.engine.connect() as connection:
            occupied_chapter_ids = {
                str(value)
                for value in connection.execute(
                    select(jobs.c.chapter_id).where(
                        jobs.c.chapter_id.in_(chapter_ids),
                        jobs.c.kind == job_kind,
                        jobs.c.status.in_(NONTERMINAL_JOB_STATUSES),
                    )
                ).scalars()
                if value is not None
            }
        for chapter_id in chapter_ids:
            if chapter_id in occupied_chapter_ids:
                skipped.append(
                    {
                        "chapterId": chapter_id,
                        "reason": "active_job",
                        "message": "章节已有未结束的同类任务",
                    }
                )
                continue
            try:
                _chapter, _ordered_pages, _normalized, spec = self._translation_spec(
                    chapter_id=chapter_id,
                    requested_page_ids=None,
                    command=command,
                    job_kind=job_kind,
                )
            except ValueError as exc:
                message = str(exc)
                skipped.append(
                    {
                        "chapterId": chapter_id,
                        "reason": _batch_skip_reason(message),
                        "message": message,
                    }
                )
                continue
            specs.append(spec)
        if not specs:
            summary = "；".join(
                f"{item['chapterId']}: {item['message']}" for item in skipped
            )
            raise ValueError(f"没有可创建任务的章节：{summary}")
        return self.jobs.create_batch(
            display_name=(
                f"{len(specs)} 个章节"
                if len(specs) > 1
                else str(specs[0].target_display["chapter"])
            ),
            specs=specs,
            response_extra={"skipped": skipped},
            idempotency_scope="translation-batch",
            idempotency_key=idempotency_key,
            idempotency_payload=idempotency_payload,
            transaction_initializer=lambda connection, _batch_id: (
                self._materialize_text_styles(connection, specs)
            ),
        )

    def _resolve_book_chapter_ids(self, book_ids: Sequence[str]) -> list[str]:
        if (
            not book_ids
            or any(not isinstance(value, str) or not value for value in book_ids)
            or len(set(book_ids)) != len(book_ids)
        ):
            raise ValueError("bookIds must contain unique book IDs")
        requested_order = {book_id: index for index, book_id in enumerate(book_ids)}
        with self.engine.connect() as connection:
            existing = {
                str(value)
                for value in connection.execute(
                    select(books.c.id).where(
                        books.c.id.in_(book_ids),
                        books.c.kind == "library",
                        books.c.owner_user_id == effective_owner_id(),
                    )
                ).scalars()
            }
            if existing != set(book_ids):
                raise ValueError("bookIds must all identify library books")
            rows = list(
                connection.execute(
                    select(chapters.c.id, chapters.c.book_id, chapters.c.ordinal)
                    .where(chapters.c.book_id.in_(book_ids))
                ).mappings()
            )
        rows.sort(
            key=lambda row: (
                requested_order[str(row["book_id"])],
                int(row["ordinal"]),
            )
        )
        resolved = [str(row["id"]) for row in rows]
        if not resolved:
            raise ValueError("selected books contain no chapters")
        return resolved

    def _translation_spec(
        self,
        *,
        chapter_id: str,
        requested_page_ids: Sequence[str] | None,
        command: Mapping[str, Any],
        job_kind: str,
        retry_of_job_id: str | None = None,
        retry_mode: str | None = None,
    ) -> tuple[Mapping[str, Any], list[str], dict[str, Any], JobSpec]:
        chapter, ordered_pages = resolve_chapter_pages(
            self.engine,
            chapter_id=chapter_id,
            requested_page_ids=requested_page_ids,
        )
        normalized = self.settings.resolve_translation(
            chapter_id=chapter_id,
            command=command,
        )
        text_style_snapshot = self._text_style_snapshot(
            chapter_id=chapter_id,
            command=command,
        )
        if text_style_snapshot is not None:
            normalized["textStyleSnapshot"] = text_style_snapshot
        normalized = self.public_access.apply_resolved_translation(
            normalized,
            page_ids=ordered_pages,
        )
        mode = command["mode"]
        step_kinds = step_kinds_for_mode(
            mode,
            reuse_existing_bubbles=bool(command["reuseExistingBubbles"]),
            proofreading_rounds=len(normalized.get("proofreadingRounds", ())),
            remove_text_with_ocr=bool(normalized["removeTextWithOcr"]),
        )
        validate_translation_job_requirements(normalized, step_kinds)
        spec = JobSpec(
            kind=job_kind,
            book_id=str(chapter["book_id"]),
            chapter_id=chapter_id,
            config=normalized,
            items=tuple(
                JobItemSpec(page_id=page_id, step_kinds=step_kinds)
                for page_id in ordered_pages
            ),
            target_display={
                "book": chapter["book_title"],
                "chapter": chapter["title"],
                "pageCount": len(ordered_pages),
            },
            retry_of_job_id=retry_of_job_id,
            retry_mode=retry_mode,
            font_snapshots=(
                {"taskTextStyle": str(text_style_snapshot["defaultFontId"])}
                if text_style_snapshot is not None
                and text_style_snapshot.get("defaultFontId") is not None
                else None
            ),
        )
        return chapter, ordered_pages, normalized, spec

    def _text_style_snapshot(
        self,
        *,
        chapter_id: str,
        command: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        source_page_id = command.get("styleSourcePageId")
        source_revision = command.get("styleSourceDocumentRevision")
        if source_page_id is None and source_revision is None:
            return None
        with self.engine.connect() as connection:
            source = connection.execute(
                select(
                    pages.c.document_revision,
                    pages.c.default_font_id,
                    pages.c.page_style_defaults_json,
                ).where(
                    pages.c.id == str(source_page_id),
                    pages.c.chapter_id == chapter_id,
                )
            ).mappings().one_or_none()
        if source is None:
            raise ValueError("style source page does not belong to the chapter")
        if int(source["document_revision"]) != int(source_revision):
            raise ValueError("style source page document revision changed")
        return {
            "sourcePageId": str(source_page_id),
            "sourceDocumentRevision": int(source_revision),
            "defaultFontId": (
                str(source["default_font_id"])
                if source["default_font_id"] is not None
                else None
            ),
            "pageStyleDefaults": validate_page_style(
                json.loads(source["page_style_defaults_json"]),
                partial=False,
            ),
        }

    @staticmethod
    def _materialize_text_styles(
        connection: Connection,
        specs: Sequence[JobSpec],
    ) -> None:
        """Copy each frozen task style into its target page documents."""

        now = utcnow()
        for spec in specs:
            snapshot = spec.config.get("textStyleSnapshot")
            if not isinstance(snapshot, Mapping):
                continue
            if spec.chapter_id is None:
                raise ValueError("text style materialization requires a chapter")
            if set(snapshot) != {
                "sourcePageId",
                "sourceDocumentRevision",
                "defaultFontId",
                "pageStyleDefaults",
            }:
                raise ValueError("frozen text style snapshot fields are invalid")
            source_page_id = snapshot["sourcePageId"]
            source_revision = snapshot["sourceDocumentRevision"]
            if not isinstance(source_page_id, str) or not source_page_id:
                raise ValueError("frozen text style source page is invalid")
            if (
                isinstance(source_revision, bool)
                or not isinstance(source_revision, int)
                or source_revision < 1
            ):
                raise ValueError("frozen text style source revision is invalid")
            default_font_id = snapshot.get("defaultFontId")
            if default_font_id is not None and (
                not isinstance(default_font_id, str) or not default_font_id
            ):
                raise ValueError("frozen text style font is invalid")
            style_defaults = validate_page_style(
                snapshot.get("pageStyleDefaults"),
                partial=False,
            )
            source = connection.execute(
                select(
                    pages.c.document_revision,
                    pages.c.default_font_id,
                    pages.c.page_style_defaults_json,
                ).where(
                    pages.c.id == source_page_id,
                    pages.c.chapter_id == spec.chapter_id,
                )
            ).mappings().one_or_none()
            if source is None:
                raise ValueError("style source page does not belong to the chapter")
            source_defaults = validate_page_style(
                json.loads(source["page_style_defaults_json"]),
                partial=False,
            )
            if (
                int(source["document_revision"]) != source_revision
                or source["default_font_id"] != default_font_id
                or source_defaults != style_defaults
            ):
                raise ValueError("style source page document revision changed")

            target_page_ids = [
                str(item.page_id)
                for item in spec.items
                if item.page_id is not None
            ]
            target_rows = list(
                connection.execute(
                    select(
                        pages.c.id,
                        pages.c.document_revision,
                        pages.c.rendered_revision,
                        pages.c.render_status,
                        pages.c.default_font_id,
                        pages.c.page_style_defaults_json,
                    ).where(
                        pages.c.chapter_id == spec.chapter_id,
                        pages.c.id.in_(target_page_ids),
                    )
                ).mappings()
            )
            targets = {str(row["id"]): row for row in target_rows}
            if set(targets) != set(target_page_ids):
                raise ValueError("pageIds must all belong to the chapter")

            for page_id in target_page_ids:
                target = targets[page_id]
                current_defaults = validate_page_style(
                    json.loads(target["page_style_defaults_json"]),
                    partial=False,
                )
                if (
                    target["default_font_id"] == default_font_id
                    and current_defaults == style_defaults
                ):
                    continue

                base_revision = int(target["document_revision"])
                new_revision = base_revision + 1
                bubble_count = int(
                    connection.execute(
                        select(func.count())
                        .select_from(bubbles)
                        .where(bubbles.c.page_id == page_id)
                    ).scalar_one()
                )
                bubbles_changed = connection.execute(
                    update(bubbles)
                    .where(
                        bubbles.c.page_id == page_id,
                        bubbles.c.updated_revision == base_revision,
                    )
                    .values(updated_revision=new_revision, updated_at=now)
                )
                if bubbles_changed.rowcount != bubble_count:
                    raise JobConflict(
                        "bubble revision does not match page document"
                    )
                connection.execute(
                    update(render_requests)
                    .where(
                        render_requests.c.page_id == page_id,
                        render_requests.c.status.in_(("pending", "running")),
                    )
                    .values(requested_revision=new_revision, updated_at=now)
                )
                page_values: dict[str, object] = {
                    "default_font_id": default_font_id,
                    "document_revision": new_revision,
                    "page_style_defaults_json": _json(style_defaults),
                    "updated_at": now,
                }
                if (
                    target["render_status"] == "ready"
                    and target["rendered_revision"] == base_revision
                ):
                    page_values["rendered_revision"] = new_revision
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
                        raise JobConflict("current translated asset is missing")
                changed = connection.execute(
                    update(pages)
                    .where(
                        pages.c.id == page_id,
                        pages.c.chapter_id == spec.chapter_id,
                        pages.c.document_revision == base_revision,
                    )
                    .values(**page_values)
                )
                if changed.rowcount != 1:
                    raise JobConflict("target page document revision changed")


def normalize_translation_command(config: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(config) - ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown translation config fields: {', '.join(sorted(unknown))}"
        )
    mode = config.get("mode", "standard")
    execution_mode = config.get("executionMode", "sequential")
    if not isinstance(mode, str):
        raise ValueError("mode must be a string")
    if not isinstance(execution_mode, str):
        raise ValueError("executionMode must be a string")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"unsupported translation mode: {mode}")
    if execution_mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    source_page_id = config.get("styleSourcePageId")
    source_revision = config.get("styleSourceDocumentRevision")
    if (source_page_id is None) != (source_revision is None):
        raise ValueError(
            "styleSourcePageId and styleSourceDocumentRevision must be provided together"
        )
    if source_page_id is not None and (
        not isinstance(source_page_id, str) or not source_page_id
    ):
        raise ValueError("styleSourcePageId must be a non-empty string")
    if source_revision is not None and (
        isinstance(source_revision, bool)
        or not isinstance(source_revision, int)
        or source_revision < 1
    ):
        raise ValueError("styleSourceDocumentRevision must be a positive integer")
    skip_completed = config.get("skipCompleted", False)
    reuse_existing = config.get("reuseExistingBubbles", False)
    if not isinstance(skip_completed, bool):
        raise ValueError("skipCompleted must be a boolean")
    if not isinstance(reuse_existing, bool):
        raise ValueError("reuseExistingBubbles must be a boolean")
    normalized = {
        "mode": mode,
        "executionMode": execution_mode,
        "skipCompleted": skip_completed,
        "reuseExistingBubbles": reuse_existing,
    }
    if source_page_id is not None:
        normalized.update(
            {
                "styleSourcePageId": source_page_id,
                "styleSourceDocumentRevision": source_revision,
            }
        )
    return normalized


def _batch_skip_reason(message: str) -> str:
    lowered = message.lower()
    if "chapter not found" in lowered:
        return "not_found"
    if "requires at least one page" in lowered:
        return "empty_chapter"
    if "api key" in lowered or "credential" in lowered:
        return "missing_credentials"
    return "invalid_configuration"


def step_kinds_for_mode(
    mode: str,
    *,
    reuse_existing_bubbles: bool = False,
    proofreading_rounds: int = 1,
    remove_text_with_ocr: bool = False,
) -> tuple[str, ...]:
    if mode == "standard":
        steps = (
            "detect",
            "ocr",
            "color",
            "auto_terms",
            "translate",
            "repair",
            "render",
            "save",
        )
        return steps[1:] if reuse_existing_bubbles else steps
    if mode == "hq":
        steps = (
            "detect",
            "ocr",
            "color",
            "auto_terms",
            "hq_translate",
            "repair",
            "render",
            "save",
        )
        return steps[1:] if reuse_existing_bubbles else steps
    if mode == "proofread":
        if proofreading_rounds < 1:
            raise ValueError("proofread mode requires at least one proofreading round")
        return (
            *("proofread" for _ in range(proofreading_rounds)),
            "render",
            "save",
        )
    if mode == "remove_text":
        return (
            "detect",
            *(("ocr",) if remove_text_with_ocr else ()),
            "repair",
            "publish_clean",
        )
    raise ValueError(f"unsupported translation mode: {mode}")


def validate_translation_job_requirements(
    config: Mapping[str, Any],
    step_kinds: Sequence[str],
) -> None:
    """Reject incomplete backend settings before a durable job is admitted."""

    steps = set(step_kinds)
    translation_step = next(
        (step for step in ("translate", "hq_translate") if step in steps),
        None,
    )
    if translation_step is not None:
        capability = (
            TRANSLATION_CAPABILITY
            if translation_step == "translate"
            else HQ_TRANSLATION_CAPABILITY
        )
        _validate_ai_provider_section(
            config.get("translation"),
            capability=capability,
            label="翻译服务",
        )
    if "proofread" in steps:
        rounds = config.get("proofreadingRounds")
        if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)):
            raise ValueError("AI 校对任务缺少已冻结的轮次配置")
        if len(rounds) != sum(1 for step in step_kinds if step == "proofread"):
            raise ValueError("AI 校对步骤与冻结轮次数量不一致")
        for index, round_config in enumerate(rounds):
            _validate_ai_provider_section(
                round_config,
                capability=HQ_TRANSLATION_CAPABILITY,
                label=f"第 {index + 1} 轮校对",
            )

    if "ocr" in steps:
        _validate_ocr_section(config.get("ocr"))
    if "detect" in steps:
        _validate_detector_section(config.get("detector"))


def _validate_ai_provider_section(
    value: object,
    *,
    capability: str,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}配置必须是对象")
    section = dict(value)
    provider_value = section.get("provider")
    if not isinstance(provider_value, str):
        raise ValueError(f"{label}服务商必须是字符串")
    provider = provider_value.strip()
    if not provider:
        raise ValueError(f"{label}未选择服务商，请先在设置中完成配置")
    manifest = get_provider_manifest(provider)
    if capability not in manifest.capabilities:
        raise ValueError(f"{label}服务商 {manifest.display_name} 不支持当前任务")
    credential_version_id = section.get("credentialVersionId")
    if credential_version_id is not None and (
        not isinstance(credential_version_id, str)
        or not credential_version_id.strip()
    ):
        raise ValueError(f"{label}凭据版本必须是非空字符串")
    model_value = section.get("model_name", "")
    if not isinstance(model_value, str):
        raise ValueError(f"{label}模型名称必须是字符串")
    model_name = model_value.strip()
    if manifest.requires_model and not model_name:
        raise ValueError(f"{label}缺少模型名称，请先在设置中填写并保存")
    base_url_value = section.get("custom_base_url", "")
    if not isinstance(base_url_value, str):
        raise ValueError(f"{label} Base URL 必须是字符串")
    base_url = base_url_value.strip()
    if manifest.requires_base_url and not base_url:
        raise ValueError(f"{label}缺少 Base URL，请先在设置中填写并保存")
    if provider_requires_api_key(provider, base_url) and credential_version_id is None:
        raise ValueError(
            f"{label}缺少已保存的 API Key，请先在设置中填写并保存"
        )
    if (
        capability == HQ_TRANSLATION_CAPABILITY
        and not isinstance(section.get("compress_vision_images"), bool)
    ):
        raise ValueError(f"{label}图片压缩开关必须是布尔值")


def _validate_ocr_section(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("OCR 配置必须是对象")
    section = dict(value)
    engine = section.get("ocr_engine")
    if not isinstance(engine, str):
        raise ValueError("OCR 引擎必须是字符串")
    if engine not in {
        "manga_ocr",
        "paddle_ocr",
        "paddleocr_vl",
        "baidu_ocr",
        "ai_vision",
        "48px_ocr",
    }:
        raise ValueError("OCR 引擎无效")
    base_fields = {
        "ocr_engine",
        "enable_hybrid_ocr",
        "secondary_ocr_engine",
        "hybrid_ocr_threshold",
    }
    expected_fields = set(base_fields)
    if engine == "paddleocr_vl":
        expected_fields.add("paddleocr_vl_source_language")
    elif engine == "baidu_ocr":
        expected_fields.update(
            {"baidu_version", "baidu_ocr_language", "credentialVersionId"}
        )
    elif engine == "ai_vision":
        expected_fields.update(
            {
                "ai_vision_provider",
                "ai_vision_model_name",
                "custom_ai_vision_base_url",
                "ai_vision_openai_options",
                "ai_vision_ocr_prompt",
                "ai_vision_prompt_mode",
                "ai_vision_min_image_size",
                "compress_vision_images",
            }
        )
        if "credentialVersionId" in section:
            expected_fields.add("credentialVersionId")
    if set(section) != expected_fields:
        raise ValueError("OCR 配置字段无效")
    if engine == "paddleocr_vl":
        source_language = section["paddleocr_vl_source_language"]
        if (
            not isinstance(source_language, str)
            or source_language not in PADDLEOCR_VL_LANGUAGE_NAMES
        ):
            raise ValueError("PaddleOCR-VL 源语言无效")
    if not isinstance(section["enable_hybrid_ocr"], bool):
        raise ValueError("混合 OCR 开关必须是布尔值")
    if (
        not isinstance(section["secondary_ocr_engine"], str)
        or not section["secondary_ocr_engine"]
    ):
        raise ValueError("备用 OCR 引擎无效")
    threshold = section["hybrid_ocr_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= threshold <= 1
    ):
        raise ValueError("混合 OCR 阈值无效")
    if engine == "baidu_ocr":
        credential_version_id = section.get("credentialVersionId")
        if not isinstance(credential_version_id, str) or not credential_version_id:
            raise ValueError(
                "百度 OCR 缺少已保存的 API Key 和 Secret Key，"
                "请先在设置中填写并保存"
            )
        return
    if engine != "ai_vision":
        return
    if not isinstance(section["compress_vision_images"], bool):
        raise ValueError("视觉模型图片压缩开关必须是布尔值")
    _validate_ai_provider_section(
        {
            "provider": section.get("ai_vision_provider"),
            "model_name": section.get("ai_vision_model_name"),
            "custom_base_url": section.get("custom_ai_vision_base_url"),
            "credentialVersionId": section.get("credentialVersionId"),
        },
        capability=VISION_OCR_CAPABILITY,
        label="AI 视觉 OCR",
    )


def _validate_detector_section(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("检测配置必须是对象")
    section = dict(value)
    required_fields = {
        "detector_type",
        "expand_ratio",
        "expand_top",
        "expand_bottom",
        "expand_left",
        "expand_right",
        "enable_aux_yolo_detection",
        "aux_yolo_conf_threshold",
        "aux_yolo_overlap_threshold",
        "enable_saber_yolo_refine",
        "saber_yolo_refine_overlap_threshold",
        "min_text_block_area_percent",
    }
    if set(section) != required_fields:
        raise ValueError("检测配置字段无效")
    if section["detector_type"] not in {"default", "ctd", "yolo"}:
        raise ValueError("文本检测器无效")
    for field in ("enable_aux_yolo_detection", "enable_saber_yolo_refine"):
        if not isinstance(section[field], bool):
            raise ValueError(f"检测配置 {field} 必须是布尔值")
    for field in (
        "expand_ratio",
        "expand_top",
        "expand_bottom",
        "expand_left",
        "expand_right",
        "aux_yolo_conf_threshold",
        "aux_yolo_overlap_threshold",
        "saber_yolo_refine_overlap_threshold",
        "min_text_block_area_percent",
    ):
        field_value = section[field]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not math.isfinite(float(field_value))
        ):
            raise ValueError(f"检测配置 {field} 必须是有限数字")
