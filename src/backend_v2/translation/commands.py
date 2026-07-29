"""Closed translation job commands; there is intentionally no generic POST /jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Engine, select

from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.plugins.snapshots import enabled_plugin_snapshots
from src.backend_v2.storage.schema import books, chapters, pages
from src.backend_v2.settings.resolver import SettingsResolver
from src.shared.ai_providers import (
    HQ_TRANSLATION_CAPABILITY,
    TRANSLATION_CAPABILITY,
    VISION_OCR_CAPABILITY,
    get_provider_manifest,
)


ALLOWED_MODES = frozenset({"standard", "hq", "proofread", "remove_text"})
ALLOWED_EXECUTION_MODES = frozenset({"sequential", "parallel"})
ALLOWED_CONFIG_KEYS = frozenset(
    {
        "mode",
        "executionMode",
        "skipCompleted",
        "reuseExistingBubbles",
    }
)


class TranslationJobCommandService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)

    def create_chapter_job(
        self,
        *,
        chapter_id: str,
        config: Mapping[str, Any],
        page_ids: Sequence[str] | None,
        idempotency_key: str,
    ) -> dict[str, object]:
        command = normalize_translation_command(config)
        chapter, ordered_pages = self._resolve_chapter_pages(
            chapter_id=chapter_id,
            requested_page_ids=page_ids,
        )
        normalized = self.settings.resolve_translation(
            chapter_id=chapter_id,
            command=command,
        )
        plugin_snapshots = self._plugin_snapshots()
        mode = str(command["mode"])
        step_kinds = step_kinds_for_mode(
            mode,
            reuse_existing_bubbles=bool(command["reuseExistingBubbles"]),
            proofreading_rounds=len(normalized.get("proofreadingRounds", ())),
        )
        validate_translation_job_requirements(normalized, step_kinds)
        job_kind = "remove_text" if mode == "remove_text" else "translation"
        spec = JobSpec(
            kind=job_kind,
            book_id=str(chapter["book_id"]),
            chapter_id=chapter_id,
            config=normalized,
            items=tuple(
                JobItemSpec(
                    page_id=page_id,
                    step_kinds=step_kinds,
                )
                for page_id in ordered_pages
            ),
            target_display={
                "book": chapter["book_title"],
                "chapter": chapter["title"],
                "pageCount": len(ordered_pages),
            },
            plugin_snapshots=plugin_snapshots,
        )
        payload = {
            "chapterId": chapter_id,
            "pageIds": ordered_pages,
            "config": normalized,
        }
        return self.jobs.create_batch(
            kind=job_kind,
            display_name=f"{chapter['book_title']} / {chapter['title']}",
            specs=[spec],
            idempotency_scope=f"chapter-translation:{chapter_id}",
            idempotency_key=idempotency_key,
            idempotency_payload=payload,
        )

    def create_batch(
        self,
        *,
        chapter_ids: Sequence[str],
        config: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, object]:
        if not chapter_ids or len(set(chapter_ids)) != len(chapter_ids):
            raise ValueError("chapterIds must contain unique chapter IDs")
        command = normalize_translation_command(config)
        mode = str(command["mode"])
        job_kind = "remove_text" if mode == "remove_text" else "translation"
        plugin_snapshots = self._plugin_snapshots()
        specs: list[JobSpec] = []
        for chapter_id in chapter_ids:
            chapter, ordered_pages = self._resolve_chapter_pages(
                chapter_id=chapter_id,
                requested_page_ids=None,
            )
            normalized = self.settings.resolve_translation(
                chapter_id=chapter_id,
                command=command,
            )
            step_kinds = step_kinds_for_mode(
                mode,
                reuse_existing_bubbles=bool(command["reuseExistingBubbles"]),
                proofreading_rounds=len(normalized.get("proofreadingRounds", ())),
            )
            validate_translation_job_requirements(normalized, step_kinds)
            specs.append(
                JobSpec(
                    kind=job_kind,
                    book_id=str(chapter["book_id"]),
                    chapter_id=chapter_id,
                    config=normalized,
                    items=tuple(
                        JobItemSpec(
                            page_id=page_id,
                            step_kinds=step_kinds,
                        )
                        for page_id in ordered_pages
                    ),
                    target_display={
                        "book": chapter["book_title"],
                        "chapter": chapter["title"],
                        "pageCount": len(ordered_pages),
                    },
                    plugin_snapshots=plugin_snapshots,
                )
            )
        return self.jobs.create_batch(
            kind=job_kind,
            display_name=(
                f"{len(specs)} 个章节"
                if len(specs) > 1
                else str(specs[0].target_display["chapter"])
            ),
            specs=specs,
            idempotency_scope="translation-batch",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "chapterIds": list(chapter_ids),
                "config": command,
            },
        )

    def _resolve_chapter_pages(
        self,
        *,
        chapter_id: str,
        requested_page_ids: Sequence[str] | None,
    ) -> tuple[Mapping[str, Any], list[str]]:
        with self.engine.connect() as connection:
            chapter = connection.execute(
                select(
                    chapters.c.id,
                    chapters.c.book_id,
                    chapters.c.title,
                    books.c.title.label("book_title"),
                )
                .join(books, books.c.id == chapters.c.book_id)
                .where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter is None:
                raise ValueError("chapter not found")
            ordered = list(
                connection.execute(
                    select(pages.c.id)
                    .where(pages.c.chapter_id == chapter_id)
                    .order_by(pages.c.ordinal)
                ).scalars()
            )
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
            raise ValueError("translation task requires at least one page")
        return chapter, [str(page_id) for page_id in ordered]

    def _plugin_snapshots(self) -> dict[str, dict[str, Any]]:
        with self.engine.connect() as connection:
            return enabled_plugin_snapshots(connection)


def normalize_translation_command(config: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(config) - ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown translation config fields: {', '.join(sorted(unknown))}"
        )
    mode = str(config.get("mode", "standard"))
    execution_mode = str(config.get("executionMode", "sequential"))
    if mode not in ALLOWED_MODES:
        raise ValueError(f"unsupported translation mode: {mode}")
    if execution_mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    return {
        "mode": mode,
        "executionMode": execution_mode,
        "skipCompleted": bool(config.get("skipCompleted", False)),
        "reuseExistingBubbles": bool(config.get("reuseExistingBubbles", False)),
    }


def step_kinds_for_mode(
    mode: str,
    *,
    reuse_existing_bubbles: bool = False,
    proofreading_rounds: int = 1,
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
        )
        return steps[1:] if reuse_existing_bubbles else steps
    if mode == "proofread":
        if proofreading_rounds < 1:
            raise ValueError("proofread mode requires at least one proofreading round")
        return (*("proofread" for _ in range(proofreading_rounds)), "render")
    if mode == "remove_text":
        return ("detect", "ocr", "repair", "publish_clean")
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


def _validate_ai_provider_section(
    value: object,
    *,
    capability: str,
    label: str,
) -> None:
    section = dict(value) if isinstance(value, Mapping) else {}
    provider = str(
        section.get("provider", section.get("model_provider", ""))
    ).strip()
    if not provider:
        raise ValueError(f"{label}未选择服务商，请先在设置中完成配置")
    manifest = get_provider_manifest(provider)
    if capability not in manifest.capabilities:
        raise ValueError(f"{label}服务商 {manifest.display_name} 不支持当前任务")
    if manifest.requires_api_key and not section.get("credentialVersionId"):
        raise ValueError(
            f"{label}缺少已保存的 API Key，请先在设置中填写并保存"
        )
    model_name = str(section.get("model_name", "")).strip()
    if manifest.requires_model and not model_name:
        raise ValueError(f"{label}缺少模型名称，请先在设置中填写并保存")
    base_url = str(section.get("custom_base_url", "")).strip()
    if manifest.requires_base_url and not base_url:
        raise ValueError(f"{label}缺少 Base URL，请先在设置中填写并保存")


def _validate_ocr_section(value: object) -> None:
    section = dict(value) if isinstance(value, Mapping) else {}
    engine = str(section.get("ocr_engine", "manga_ocr"))
    if engine == "baidu_ocr":
        if not section.get("credentialVersionId"):
            raise ValueError(
                "百度 OCR 缺少已保存的 API Key 和 Secret Key，"
                "请先在设置中填写并保存"
            )
        return
    if engine != "ai_vision":
        return
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
