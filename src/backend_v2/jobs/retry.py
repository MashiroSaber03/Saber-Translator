"""Create replacement jobs from durable failure facts, never browser memory."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping

from sqlalchemy import Engine, select

from src.backend_v2.jobs.repository import (
    JobConflict,
    JobItemSpec,
    JobNotFound,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.storage.schema import (
    assets,
    job_asset_inputs,
    job_credential_snapshots,
    job_font_snapshots,
    job_items,
    job_plugin_snapshots,
    job_steps,
    jobs,
    pages,
)
from src.backend_v2.translation.auxiliary import AuxiliaryTranslationCommands
from src.backend_v2.translation.commands import TranslationJobCommandService


class JobRetryService:
    """Apply the plan's current-settings/original-snapshot retry semantics."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.repository = JobQueueRepository(engine)

    def retry(
        self,
        *,
        job_id: str,
        failed_only: bool,
        strategy: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        if strategy not in {"current", "original"}:
            raise ValueError("strategy must be current or original")
        source, selected_items = self._source(job_id, failed_only=failed_only)
        kind = str(source["kind"])
        if kind == "style_apply":
            strategy = "original"

        if strategy == "current" and kind in {"translation", "remove_text"}:
            response = self._retry_translation(
                source,
                selected_items,
                failed_only=failed_only,
                idempotency_key=idempotency_key,
            )
        elif strategy == "current" and kind == "detect":
            response = self._retry_detection(
                source,
                selected_items,
                failed_only=failed_only,
                idempotency_key=idempotency_key,
            )
        else:
            response = self._clone_original(
                source,
                selected_items,
                failed_only=failed_only,
                idempotency_key=idempotency_key,
            )
            strategy = "original"
        return {
            **response,
            "sourceJobId": job_id,
            "retryMode": strategy,
            "failedOnly": failed_only,
        }

    def _source(
        self,
        job_id: str,
        *,
        failed_only: bool,
    ) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
        with self.engine.connect() as connection:
            source = connection.execute(
                select(jobs).where(jobs.c.id == job_id)
            ).mappings().one_or_none()
            if source is None:
                raise JobNotFound("job not found")
            expected_status = "completed_with_errors" if failed_only else "failed"
            if str(source["status"]) != expected_status:
                raise JobConflict(
                    f"{expected_status} is required for this retry command"
                )
            conditions = [job_items.c.job_id == job_id]
            if failed_only:
                conditions.append(job_items.c.status == "failed")
            selected_items = list(
                connection.execute(
                    select(job_items)
                    .where(*conditions)
                    .order_by(job_items.c.ordinal)
                ).mappings()
            )
        if not selected_items:
            raise JobConflict("source job has no retryable items")
        return source, selected_items

    def _retry_translation(
        self,
        source: Mapping[str, Any],
        selected_items: list[Mapping[str, Any]],
        *,
        failed_only: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        chapter_id = source.get("chapter_id")
        if not chapter_id:
            raise JobConflict("translation retry target chapter no longer exists")
        page_ids = self._page_ids(selected_items)
        config = _json_object(source.get("config_json"))
        mode = str(config.get("mode", "standard"))
        return TranslationJobCommandService(self.engine).create_chapter_job(
            chapter_id=str(chapter_id),
            config={
                "mode": mode,
                "executionMode": str(
                    config.get("executionMode", "sequential")
                ),
                "skipCompleted": False,
                "reuseExistingBubbles": bool(
                    config.get("reuseExistingBubbles", False)
                ),
            },
            page_ids=page_ids,
            idempotency_key=idempotency_key,
            retry_of_job_id=str(source["id"]),
            retry_mode="current",
            idempotency_scope=(
                f"job-retry:{source['id']}:"
                f"{'failed' if failed_only else 'all'}"
            ),
        )

    def _retry_detection(
        self,
        source: Mapping[str, Any],
        selected_items: list[Mapping[str, Any]],
        *,
        failed_only: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        chapter_id = source.get("chapter_id")
        if not chapter_id:
            raise JobConflict("detection retry target chapter no longer exists")
        return AuxiliaryTranslationCommands(self.engine).create_detect_job(
            chapter_id=str(chapter_id),
            page_ids=self._page_ids(selected_items),
            idempotency_key=idempotency_key,
            retry_of_job_id=str(source["id"]),
            retry_mode="current",
            idempotency_scope=(
                f"job-retry:{source['id']}:"
                f"{'failed' if failed_only else 'all'}"
            ),
        )

    def _clone_original(
        self,
        source: Mapping[str, Any],
        selected_items: list[Mapping[str, Any]],
        *,
        failed_only: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        source_id = str(source["id"])
        selected_ids = {str(item["id"]) for item in selected_items}
        step_kinds: dict[str, list[str]] = defaultdict(list)
        create_inputs: dict[str, dict[str, str]] = defaultdict(dict)
        with self.engine.connect() as connection:
            for row in connection.execute(
                select(
                    job_steps.c.job_item_id,
                    job_steps.c.kind,
                )
                .where(job_steps.c.job_item_id.in_(selected_ids))
                .order_by(job_steps.c.job_item_id, job_steps.c.ordinal)
            ).mappings():
                step_kinds[str(row["job_item_id"])].append(str(row["kind"]))
            for row in connection.execute(
                select(
                    job_asset_inputs.c.job_item_id,
                    job_asset_inputs.c.role,
                    job_asset_inputs.c.asset_id,
                    assets.c.integrity_status,
                )
                .join(assets, assets.c.id == job_asset_inputs.c.asset_id)
                .where(
                    job_asset_inputs.c.job_id == source_id,
                    job_asset_inputs.c.binding_phase == "create",
                    job_asset_inputs.c.job_item_id.in_(selected_ids),
                )
            ).mappings():
                if str(row["integrity_status"]) != "ok":
                    raise JobConflict(
                        "original retry input is unavailable; recreate the task"
                    )
                create_inputs[str(row["job_item_id"])][str(row["role"])] = str(
                    row["asset_id"]
                )
            credentials = {
                str(row["role"]): str(row["credential_version_id"])
                for row in connection.execute(
                    select(
                        job_credential_snapshots.c.role,
                        job_credential_snapshots.c.credential_version_id,
                    ).where(job_credential_snapshots.c.job_id == source_id)
                ).mappings()
            }
            fonts = {
                str(row["role"]): str(row["font_id"])
                for row in connection.execute(
                    select(
                        job_font_snapshots.c.role,
                        job_font_snapshots.c.font_id,
                    ).where(job_font_snapshots.c.job_id == source_id)
                ).mappings()
            }
            plugins = {
                str(row["plugin_version_id"]): _json_object(row["config_json"])
                for row in connection.execute(
                    select(job_plugin_snapshots).where(
                        job_plugin_snapshots.c.job_id == source_id
                    )
                ).mappings()
            }
            page_ids = {
                str(value)
                for value in connection.execute(
                    select(pages.c.id).where(
                        pages.c.id.in_(
                            [
                                str(item["page_id"])
                                for item in selected_items
                                if item["page_id"] is not None
                            ]
                        )
                    )
                ).scalars()
            }
        requested_page_ids = {
            str(item["page_id"])
            for item in selected_items
            if item["page_id"] is not None
        }
        if page_ids != requested_page_ids:
            raise JobConflict("one or more retry target pages no longer exist")

        item_specs = tuple(
            JobItemSpec(
                page_id=(
                    str(item["page_id"])
                    if item["page_id"] is not None
                    else None
                ),
                step_kinds=tuple(step_kinds[str(item["id"])]),
                asset_inputs=create_inputs.get(str(item["id"])) or None,
            )
            for item in selected_items
        )
        config = _json_object(source.get("config_json"))
        display = _json_object(source.get("target_display_json"))
        spec = JobSpec(
            kind=str(source["kind"]),
            config=config,
            items=item_specs,
            book_id=_optional_text(source.get("book_id")),
            chapter_id=_optional_text(source.get("chapter_id")),
            page_id=_optional_text(source.get("page_id")),
            analysis_run_id=_optional_text(source.get("analysis_run_id")),
            continuation_project_id=_optional_text(
                source.get("continuation_project_id")
            ),
            web_import_draft_id=_optional_text(
                source.get("web_import_draft_id")
            ),
            target_display={
                **display,
                "retryOfJobId": source_id,
                "retryItemCount": len(item_specs),
            },
            credential_snapshots=credentials,
            font_snapshots=fonts,
            plugin_snapshots=plugins,
            retry_of_job_id=source_id,
            retry_mode="original",
        )
        return self.repository.create_batch(
            kind=str(source["kind"]),
            display_name=f"重试 · {display.get('chapter') or display.get('book') or source['kind']}",
            specs=(spec,),
            idempotency_scope=(
                f"job-retry:{source_id}:{'failed' if failed_only else 'all'}"
            ),
            idempotency_key=idempotency_key,
            idempotency_payload={
                "sourceJobId": source_id,
                "failedOnly": failed_only,
                "strategy": "original",
                "itemIds": [str(item["id"]) for item in selected_items],
            },
        )

    @staticmethod
    def _page_ids(items: list[Mapping[str, Any]]) -> list[str]:
        page_ids = [
            str(item["page_id"])
            for item in items
            if item["page_id"] is not None
        ]
        if len(page_ids) != len(items):
            raise JobConflict("retry contains a non-page item")
        return page_ids


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None
