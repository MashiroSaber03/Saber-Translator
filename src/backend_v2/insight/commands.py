"""Closed commands that create durable, frozen Insight analysis runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import uuid

from sqlalchemy import Engine, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.insight.repository import (
    ANALYSIS_RUN_SCOPES,
    InsightConflict,
    InsightRepository,
)
from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.schema import (
    analysis_heads,
    analysis_page_results,
    assets,
    books,
    chapters,
    page_assets,
    pages,
    jobs,
)
from src.shared.ai_providers import (
    CHAT_CAPABILITY,
    EMBEDDING_CAPABILITY,
    VLM_CAPABILITY,
    get_provider_manifest,
    provider_requires_api_key,
)


ALLOWED_COMMAND_KEYS = frozenset(
    {
        "bookId",
        "scope",
        "chapterIds",
        "pageIds",
    }
)


@dataclass(frozen=True, slots=True)
class FrozenTarget:
    page_id: str
    chapter_id: str
    source_asset_id: str
    source_checksum: str
    page_number: int

    def mapping(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "chapter_id": self.chapter_id,
            "source_asset_id": self.source_asset_id,
            "source_checksum": self.source_checksum,
            "page_number": self.page_number,
        }


class InsightAnalysisCommandService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)

    def create_analysis_job(
        self,
        *,
        command: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, object]:
        normalized = normalize_analysis_command(command)
        book_id = normalized["bookId"]
        scope = normalized["scope"]
        idempotency_scope = f"insight-analysis:{book_id}"
        replay = self.jobs.idempotency_replay(
            scope=idempotency_scope,
            key=idempotency_key,
            payload=normalized,
        )
        if replay is not None:
            _required_string(replay.get("runId"), "Insight replay runId")
            return replay
        book, targets = self._resolve_targets(
            book_id=book_id,
            scope=scope,
            chapter_ids=normalized["chapterIds"],
            page_ids=normalized["pageIds"],
        )
        run_id = str(uuid.uuid4())
        config = self.settings.resolve_insight(
            book_id=book_id,
            scope=scope,
        )
        validate_insight_job_requirements(config, scope=scope)
        config["runId"] = run_id
        config["bookId"] = book_id
        config["targetCount"] = len(targets)

        single_chapter = {
            target.chapter_id for target in targets
        }
        if scope == "full":
            layer_steps: list[str] = []
            for layer in config["analysis"]["layers"]:
                if not isinstance(layer, Mapping):
                    raise ValueError("Insight layer must be an object")
                layer_index = layer.get("index")
                if (
                    isinstance(layer_index, bool)
                    or not isinstance(layer_index, int)
                    or layer_index < 0
                ):
                    raise ValueError(
                        "Insight layer index must be a non-negative integer"
                    )
                layer_steps.append(f"insight_build_layer_{layer_index}")
            final_steps = (
                "insight_validate_run",
                *layer_steps,
                "insight_stage_compressed_context",
                "insight_stage_overview_no_spoiler",
                "insight_stage_overview_story_summary",
                "insight_stage_timeline",
                "insight_stage_vectors",
                "insight_publish_run",
            )
        else:
            final_steps = ("insight_publish_run",)
        spec = JobSpec(
            kind="insight_analysis",
            book_id=book_id,
            chapter_id=(
                next(iter(single_chapter))
                if len(single_chapter) == 1
                else None
            ),
            page_id=targets[0].page_id if len(targets) == 1 else None,
            config=config,
            items=(
                *(
                    JobItemSpec(
                        page_id=target.page_id,
                        step_kinds=("insight_analyze_page",),
                        asset_inputs={"source": target.source_asset_id},
                    )
                    for target in targets
                ),
                JobItemSpec(
                    page_id=None,
                    step_kinds=final_steps,
                ),
            ),
            target_display={
                "book": _required_string(book.get("title"), "book title"),
                "scope": scope,
                "pageCount": len(targets),
            },
        )
        target_mappings = tuple(target.mapping() for target in targets)

        def assert_targets(connection, _batch_id: str) -> None:
            try:
                current_book, current_targets = self._resolve_targets_in_connection(
                    connection,
                    book_id=book_id,
                    scope=scope,
                    chapter_ids=normalized["chapterIds"],
                    page_ids=normalized["pageIds"],
                )
            except ValueError as exc:
                raise InsightConflict(
                    "Insight analysis targets changed before job admission"
                ) from exc
            if (
                _required_string(current_book.get("id"), "book id") != book_id
                or _required_string(current_book.get("title"), "book title")
                != _required_string(book.get("title"), "book title")
                or current_targets != targets
            ):
                raise InsightConflict(
                    "Insight analysis targets changed before job admission"
                )

        def initialize_run(
            connection,
            _batch_id: str,
            job_ids: Sequence[str],
        ) -> None:
            if len(job_ids) != 1:
                raise RuntimeError("Insight command must create exactly one job")
            InsightRepository.insert_run(
                connection,
                run_id=run_id,
                job_id=job_ids[0],
                book_id=book_id,
                scope=scope,
                config=config,
                targets=target_mappings,
            )
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job_ids[0])
                .values(analysis_run_id=run_id)
            )

        response = self.jobs.create_batch(
            kind="insight_analysis",
            display_name=f"{book['title']} · {_scope_label(scope)}",
            specs=(spec,),
            response_extra={"runId": run_id},
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            idempotency_payload=normalized,
            transaction_initializer=assert_targets,
            transaction_hook=initialize_run,
        )
        return response

    def _resolve_targets(
        self,
        *,
        book_id: str,
        scope: str,
        chapter_ids: Sequence[str],
        page_ids: Sequence[str],
    ) -> tuple[Mapping[str, Any], list[FrozenTarget]]:
        with self.engine.connect() as connection:
            return self._resolve_targets_in_connection(
                connection,
                book_id=book_id,
                scope=scope,
                chapter_ids=chapter_ids,
                page_ids=page_ids,
            )

    @staticmethod
    def _resolve_targets_in_connection(
        connection: Connection,
        *,
        book_id: str,
        scope: str,
        chapter_ids: Sequence[str],
        page_ids: Sequence[str],
    ) -> tuple[Mapping[str, Any], list[FrozenTarget]]:
        source_pointer = page_assets.alias("insight_command_source")
        page_head = analysis_heads.alias("insight_command_page_head")
        book = connection.execute(
            select(books.c.id, books.c.title).where(
                books.c.id == book_id,
                books.c.kind == "library",
                books.c.owner_user_id == effective_owner_id(),
            )
        ).mappings().one_or_none()
        if book is None:
            raise ValueError("book not found")
        rows = list(
            connection.execute(
                select(
                    pages.c.id.label("page_id"),
                    pages.c.chapter_id,
                    chapters.c.ordinal.label("chapter_ordinal"),
                    pages.c.ordinal.label("page_ordinal"),
                    source_pointer.c.asset_id.label("source_asset_id"),
                    assets.c.checksum.label("source_checksum"),
                    analysis_page_results.c.source_checksum.label(
                        "analysis_source_checksum"
                    ),
                )
                .join(chapters, chapters.c.id == pages.c.chapter_id)
                .join(
                    source_pointer,
                    (source_pointer.c.page_id == pages.c.id)
                    & (source_pointer.c.role == "source"),
                )
                .join(assets, assets.c.id == source_pointer.c.asset_id)
                .join(
                    page_head,
                    page_head.c.page_id == pages.c.id,
                    isouter=True,
                )
                .join(
                    analysis_page_results,
                    analysis_page_results.c.id == page_head.c.active_result_id,
                    isouter=True,
                )
                .where(chapters.c.book_id == book_id)
                .order_by(chapters.c.ordinal, pages.c.ordinal)
            ).mappings()
        )
        if not rows:
            raise ValueError("book has no pages")
        available_chapters = {
            _required_string(row.get("chapter_id"), "chapter id")
            for row in rows
        }
        available_pages = {
            _required_string(row.get("page_id"), "page id")
            for row in rows
        }
        if not set(chapter_ids).issubset(available_chapters):
            raise ValueError("all chapterIds must belong to the book")
        if not set(page_ids).issubset(available_pages):
            raise ValueError("all pageIds must belong to the book")

        selected: list[Mapping[str, Any]] = []
        for row in rows:
            page_id = _required_string(row.get("page_id"), "page id")
            chapter_id = _required_string(row.get("chapter_id"), "chapter id")
            if scope == "full":
                include = True
            elif scope == "incremental":
                include = (
                    row["analysis_source_checksum"] is None
                    or row["analysis_source_checksum"]
                    != row["source_checksum"]
                )
            elif scope == "chapter":
                include = chapter_id in chapter_ids
            else:
                include = page_id in page_ids
            if include:
                selected.append(row)
        if not selected:
            if scope == "incremental":
                raise ValueError("没有需要增量分析的页面")
            raise ValueError("analysis target is empty")
        page_numbers = {
            _required_string(row.get("page_id"), "page id"): index
            for index, row in enumerate(rows, start=1)
        }
        return book, [
            FrozenTarget(
                page_id=_required_string(row.get("page_id"), "page id"),
                chapter_id=_required_string(
                    row.get("chapter_id"),
                    "chapter id",
                ),
                source_asset_id=_required_string(
                    row.get("source_asset_id"),
                    "source asset id",
                ),
                source_checksum=_required_sha256(
                    row.get("source_checksum"),
                    "source checksum",
                ),
                page_number=page_numbers[
                    _required_string(row.get("page_id"), "page id")
                ],
            )
            for row in selected
        ]


def normalize_analysis_command(command: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(command) - ALLOWED_COMMAND_KEYS
    if unknown:
        raise ValueError(
            f"unknown Insight command fields: {', '.join(sorted(unknown))}"
        )
    book_id = command.get("bookId")
    if (
        not isinstance(book_id, str)
        or not book_id
        or book_id != book_id.strip()
    ):
        raise ValueError("bookId is required")
    scope = command.get("scope")
    if scope not in ANALYSIS_RUN_SCOPES:
        raise ValueError("scope must be full, incremental, chapter, or page")
    chapter_ids = _ids(command, "chapterIds")
    page_ids = _ids(command, "pageIds")
    if scope == "chapter" and not chapter_ids:
        raise ValueError("chapter scope requires chapterIds")
    if scope == "page" and not page_ids:
        raise ValueError("page scope requires pageIds")
    if scope in {"full", "incremental"} and (chapter_ids or page_ids):
        raise ValueError(f"{scope} scope does not accept chapter/page selectors")
    if scope == "chapter" and page_ids:
        raise ValueError("chapter scope does not accept page selectors")
    if scope == "page" and chapter_ids:
        raise ValueError("page scope does not accept chapter selectors")
    return {
        "bookId": book_id,
        "scope": scope,
        "chapterIds": chapter_ids,
        "pageIds": page_ids,
    }


def _ids(
    command: Mapping[str, Any],
    key: str,
) -> list[str]:
    raw = command.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a string array")
    values = list(raw)
    if not all(
        isinstance(value, str)
        and value
        and value == value.strip()
        for value in values
    ):
        raise ValueError(f"{key} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{key} must be unique")
    return values


def _scope_label(scope: str) -> str:
    return {
        "full": "全书分析",
        "incremental": "增量分析",
        "chapter": "章节分析",
        "page": "页面分析",
    }[scope]


def validate_insight_job_requirements(
    config: Mapping[str, Any],
    *,
    scope: str,
) -> None:
    """Reject incomplete model settings before admitting a durable run."""

    _validate_provider_section(
        config.get("vlm"),
        capability=VLM_CAPABILITY,
        label="漫画分析 VLM",
    )
    if scope == "full":
        _validate_provider_section(
            config.get("chat"),
            capability=CHAT_CAPABILITY,
            label="漫画分析 LLM",
        )
        _validate_provider_section(
            config.get("embedding"),
            capability=EMBEDDING_CAPABILITY,
            label="漫画分析 Embedding",
        )


def _validate_provider_section(
    value: object,
    *,
    capability: str,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 配置无效，请重新设置")
    section = dict(value)
    provider_value = section.get("provider")
    if not isinstance(provider_value, str) or not provider_value.strip():
        raise ValueError(f"{label} 未选择服务商，请先在分析设置中完成配置")
    provider = provider_value.strip()
    manifest = get_provider_manifest(provider)
    if capability not in manifest.capabilities:
        raise ValueError(
            f"{label} 服务商 {manifest.display_name} 不支持当前任务"
        )
    base_url = section.get("custom_base_url")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError(f"{label} Base URL 无效，请重新保存设置")
    credential_version_id = section.get("credentialVersionId")
    if credential_version_id is not None and (
        not isinstance(credential_version_id, str)
        or not credential_version_id.strip()
    ):
        raise ValueError(f"{label} API Key 版本无效，请重新保存设置")
    if provider_requires_api_key(provider, base_url) and (
        not isinstance(credential_version_id, str)
        or not credential_version_id.strip()
    ):
        raise ValueError(
            f"{label} 缺少已保存的 API Key，请先在分析设置中填写并保存"
        )
    model_name = section.get("model_name")
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError(f"{label} 模型名称无效，请重新保存设置")
    if manifest.requires_model and (
        not isinstance(model_name, str) or not model_name.strip()
    ):
        raise ValueError(
            f"{label} 缺少模型名称，请先在分析设置中填写并保存"
        )
    if manifest.requires_base_url and (
        not isinstance(base_url, str) or not base_url.strip()
    ):
        raise ValueError(
            f"{label} 缺少 Base URL，请先在分析设置中填写并保存"
        )


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_sha256(value: object, field: str) -> str:
    text = _required_string(value, field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text
