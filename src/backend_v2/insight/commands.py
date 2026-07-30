"""Closed commands that create durable, frozen Insight analysis runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import uuid

from sqlalchemy import Engine, select, update

from src.backend_v2.insight.repository import InsightRepository
from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.schema import (
    analysis_heads,
    analysis_page_results,
    analysis_runs,
    assets,
    books,
    chapters,
    page_assets,
    pages,
    jobs,
)


ANALYSIS_SCOPES = frozenset({"full", "incremental", "chapter", "page"})
ALLOWED_COMMAND_KEYS = frozenset(
    {
        "bookId",
        "scope",
        "chapterId",
        "chapterIds",
        "pageId",
        "pageIds",
        "force",
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
        book_id = str(normalized["bookId"])
        scope = str(normalized["scope"])
        book, targets = self._resolve_targets(
            book_id=book_id,
            scope=scope,
            chapter_ids=normalized["chapterIds"],
            page_ids=normalized["pageIds"],
            force=bool(normalized["force"]),
        )
        run_id = str(uuid.uuid4())
        config = self.settings.resolve_insight(
            book_id=book_id,
            command=normalized,
        )
        config["runId"] = run_id
        config["bookId"] = book_id
        config["targetCount"] = len(targets)

        single_chapter = {
            target.chapter_id for target in targets
        }
        if scope == "full":
            layer_steps = tuple(
                f"insight_build_layer_{int(layer['index'])}"
                for layer in config["analysis"]["layers"]
            )
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
                "book": str(book["title"]),
                "scope": scope,
                "pageCount": len(targets),
            },
        )
        target_mappings = tuple(target.mapping() for target in targets)

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
                job_id=str(job_ids[0]),
                book_id=book_id,
                scope=scope,
                config=config,
                targets=target_mappings,
            )
            connection.execute(
                update(jobs)
                .where(jobs.c.id == str(job_ids[0]))
                .values(analysis_run_id=run_id)
            )

        response = self.jobs.create_batch(
            kind="insight_analysis",
            display_name=f"{book['title']} · {_scope_label(scope)}",
            specs=(spec,),
            idempotency_scope=f"insight-analysis:{book_id}",
            idempotency_key=idempotency_key,
            idempotency_payload=normalized,
            transaction_hook=initialize_run,
        )
        with self.engine.connect() as connection:
            persisted_run_id = connection.execute(
                select(analysis_runs.c.id).where(
                    analysis_runs.c.job_id == str(response["jobIds"][0])
                )
            ).scalar_one()
        response["runId"] = str(persisted_run_id)
        return response

    def _resolve_targets(
        self,
        *,
        book_id: str,
        scope: str,
        chapter_ids: Sequence[str],
        page_ids: Sequence[str],
        force: bool,
    ) -> tuple[Mapping[str, Any], list[FrozenTarget]]:
        source_pointer = page_assets.alias("insight_command_source")
        page_head = analysis_heads.alias("insight_command_page_head")
        with self.engine.connect() as connection:
            book = connection.execute(
                select(books.c.id, books.c.title).where(
                    books.c.id == book_id,
                    books.c.kind == "library",
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
                        analysis_page_results.c.id
                        == page_head.c.active_result_id,
                        isouter=True,
                    )
                    .where(chapters.c.book_id == book_id)
                    .order_by(chapters.c.ordinal, pages.c.ordinal)
                ).mappings()
            )
        if not rows:
            raise ValueError("book has no pages")
        available_chapters = {str(row["chapter_id"]) for row in rows}
        available_pages = {str(row["page_id"]) for row in rows}
        if not set(chapter_ids).issubset(available_chapters):
            raise ValueError("all chapterIds must belong to the book")
        if not set(page_ids).issubset(available_pages):
            raise ValueError("all pageIds must belong to the book")

        selected: list[Mapping[str, Any]] = []
        for row in rows:
            page_id = str(row["page_id"])
            chapter_id = str(row["chapter_id"])
            if scope == "full":
                include = True
            elif scope == "incremental":
                include = force or (
                    row["analysis_source_checksum"] is None
                    or str(row["analysis_source_checksum"])
                    != str(row["source_checksum"])
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
            str(row["page_id"]): index
            for index, row in enumerate(rows, start=1)
        }
        return book, [
            FrozenTarget(
                page_id=str(row["page_id"]),
                chapter_id=str(row["chapter_id"]),
                source_asset_id=str(row["source_asset_id"]),
                source_checksum=str(row["source_checksum"]),
                page_number=page_numbers[str(row["page_id"])],
            )
            for row in selected
        ]


def normalize_analysis_command(command: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(command) - ALLOWED_COMMAND_KEYS
    if unknown:
        raise ValueError(
            f"unknown Insight command fields: {', '.join(sorted(unknown))}"
        )
    book_id = str(command.get("bookId", "")).strip()
    if not book_id:
        raise ValueError("bookId is required")
    scope = str(command.get("scope", "full"))
    if scope not in ANALYSIS_SCOPES:
        raise ValueError("scope must be full, incremental, chapter, or page")
    chapter_ids = _ids(command, singular="chapterId", plural="chapterIds")
    page_ids = _ids(command, singular="pageId", plural="pageIds")
    if scope == "chapter" and not chapter_ids:
        raise ValueError("chapter scope requires chapterId or chapterIds")
    if scope == "page" and not page_ids:
        raise ValueError("page scope requires pageId or pageIds")
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
        "force": bool(command.get("force", False)),
    }


def _ids(
    command: Mapping[str, Any],
    *,
    singular: str,
    plural: str,
) -> list[str]:
    values: list[object] = []
    if command.get(singular) is not None:
        values.append(command[singular])
    if command.get(plural) is not None:
        raw = command[plural]
        if not isinstance(raw, list):
            raise ValueError(f"{plural} must be a string array")
        values.extend(raw)
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{singular}/{plural} must contain non-empty strings")
    normalized = [str(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{singular}/{plural} must be unique")
    return normalized


def _scope_label(scope: str) -> str:
    return {
        "full": "全书分析",
        "incremental": "增量分析",
        "chapter": "章节分析",
        "page": "页面分析",
    }[scope]
