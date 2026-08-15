"""Create replacement jobs from durable failure facts, never browser memory."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from sqlalchemy import Engine, select, update

from src.backend_v2.checksums import sha256_file
from src.backend_v2.insight.repository import InsightRepository
from src.backend_v2.jobs.repository import (
    JobConflict,
    JobDataInvalid,
    JobItemSpec,
    JobNotFound,
    JobQueueRepository,
    JobSpec,
    decode_job_config,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.schema import (
    assets,
    analysis_page_results,
    analysis_run_targets,
    job_asset_inputs,
    job_credential_snapshots,
    job_font_snapshots,
    job_items,
    job_plugin_snapshots,
    job_steps,
    jobs,
    page_assets,
    pages,
    web_import_drafts,
)
from src.backend_v2.translation.auxiliary import AuxiliaryTranslationCommands
from src.backend_v2.translation.commands import TranslationJobCommandService
from src.backend_v2.transfer.commands import (
    TransferDataInvalid,
    validate_container_config,
)
from src.backend_v2.web_import.commands import (
    WebImportCommandService,
    WebImportDataInvalid,
    validate_web_commit_config,
    validate_web_extract_config,
)


class JobRetryService:
    """Apply the plan's current-settings/original-snapshot retry semantics."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.repository = JobQueueRepository(engine)
        database_path = engine.url.database
        if not database_path:
            raise ValueError("job retries require a file-backed SQLite database")
        self.data_root = Path(database_path).resolve().parent

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

        if kind == "insight_analysis":
            response = self._retry_insight(
                source,
                selected_items,
                strategy=strategy,
                failed_only=failed_only,
                idempotency_key=idempotency_key,
            )
        elif kind == "container_import":
            response = self._retry_container_import(
                source,
                selected_items,
                strategy=strategy,
                failed_only=failed_only,
                idempotency_key=idempotency_key,
            )
        elif kind == "web_extract":
            response = self._retry_web_extract(
                source,
                strategy=strategy,
                failed_only=failed_only,
                idempotency_key=idempotency_key,
            )
        elif kind == "web_import_commit":
            response = self._retry_web_import_commit(
                source,
                selected_items,
                strategy=strategy,
                failed_only=failed_only,
                idempotency_key=idempotency_key,
            )
        elif strategy == "current" and kind in {"translation", "remove_text"}:
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
        try:
            detail = self.repository.get_job(job_id)
        except JobDataInvalid as exc:
            raise JobConflict(f"source job data is invalid: {exc}") from exc
        expected_status = "completed_with_errors" if failed_only else "failed"
        if str(detail["status"]) != expected_status:
            raise JobConflict(
                f"{expected_status} is required for this retry command"
            )
        with self.engine.connect() as connection:
            source = connection.execute(
                select(jobs).where(jobs.c.id == job_id)
            ).mappings().one_or_none()
            if source is None:
                raise JobNotFound("job not found")
            try:
                source_config = decode_job_config(source)
            except JobDataInvalid as exc:
                raise JobConflict(f"source job data is invalid: {exc}") from exc
            if str(source["status"]) != expected_status:
                raise JobConflict(
                    "source job changed while the retry was being prepared"
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
        normalized_source = dict(source)
        normalized_source["config_json"] = source_config
        return normalized_source, selected_items

    def _retry_container_import(
        self,
        source: Mapping[str, Any],
        selected_items: list[Mapping[str, Any]],
        *,
        strategy: str,
        failed_only: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        source_id = str(source["id"])
        chapter_id = _optional_text(source.get("chapter_id"))
        try:
            config = validate_container_config(source.get("config_json"))
        except TransferDataInvalid as exc:
            raise JobConflict(f"container retry snapshot is invalid: {exc}") from exc
        relative_path = _optional_text(config.get("containerRelativePath"))
        checksum = _optional_text(config.get("checksum"))
        if not chapter_id or not relative_path or not checksum:
            raise JobConflict("container retry target or input no longer exists")
        finished_at = source.get("finished_at") or source.get("updated_at")
        if finished_at is not None and finished_at <= utcnow() - timedelta(hours=24):
            raise JobConflict("container retry input has expired; upload it again")
        container_path = (self.data_root / Path(relative_path)).resolve()
        try:
            container_path.relative_to(self.data_root)
        except ValueError as exc:
            raise JobConflict("container retry input path is invalid") from exc
        if (
            not container_path.is_file()
            or sha256_file(container_path) != checksum
        ):
            raise JobConflict("container retry input is missing or changed")

        selected_ids = {str(item["id"]) for item in selected_items}
        with self.engine.connect() as connection:
            step_rows = list(
                connection.execute(
                    select(
                        job_steps.c.job_item_id,
                        job_steps.c.kind,
                    )
                    .where(job_steps.c.job_item_id.in_(selected_ids))
                    .order_by(job_steps.c.job_item_id, job_steps.c.ordinal)
                ).mappings()
            )
        step_kinds: dict[str, list[str]] = defaultdict(list)
        for row in step_rows:
            step_kinds[str(row["job_item_id"])].append(str(row["kind"]))

        raw_entries = config.get("entries")
        if raw_entries is None:
            entries: list[dict[str, Any]] = []
        elif not isinstance(raw_entries, list) or not all(
            isinstance(entry, Mapping) for entry in raw_entries
        ):
            raise JobConflict("container retry checkpoint is invalid")
        else:
            entries = [dict(entry) for entry in raw_entries]
        retry_entries: list[dict[str, Any]] = []
        retry_scan = not entries
        if entries:
            source_base = _required_integer(
                config,
                "entryItemOrdinalBase",
                minimum=1,
            )
            if failed_only:
                for item in selected_items:
                    kinds = step_kinds.get(str(item["id"]), [])
                    if "container_scan" in kinds:
                        retry_scan = True
                    if "container_import_page" not in kinds:
                        continue
                    index = int(item["ordinal"]) - source_base
                    if index < 0 or index >= len(entries):
                        raise JobConflict("container retry checkpoint is invalid")
                    retry_entries.append(entries[index])
            else:
                retry_entries = entries
        if retry_scan:
            retry_config = {
                key: value
                for key, value in config.items()
                if key not in {"entries", "entryItemOrdinalBase"}
            }
            item_specs = (
                JobItemSpec(page_id=None, step_kinds=("container_scan",)),
            )
        else:
            if not retry_entries:
                raise JobConflict("source job has no retryable container pages")
            retry_config = {
                **config,
                "entries": retry_entries,
                "entryItemOrdinalBase": 1,
            }
            item_specs = tuple(
                JobItemSpec(
                    page_id=None,
                    step_kinds=("container_import_page",),
                )
                for _entry in retry_entries
            )
        validate_container_config(retry_config)

        display = _json_object(source.get("target_display_json"))
        credentials, plugins = self._original_runtime_snapshots(source_id)
        spec = JobSpec(
            kind="container_import",
            book_id=_optional_text(source.get("book_id")),
            chapter_id=chapter_id,
            config=retry_config,
            items=item_specs,
            target_display={
                **display,
                "retryOfJobId": source_id,
                "retryItemCount": len(item_specs),
            },
            credential_snapshots=(
                credentials if strategy == "original" else None
            ),
            plugin_snapshots=plugins if strategy == "original" else None,
            retry_of_job_id=source_id,
            retry_mode=strategy,
        )
        return self.repository.create_batch(
            kind="container_import",
            display_name=(
                f"重试 · {display.get('chapter') or display.get('book') or '容器导入'}"
            ),
            specs=(spec,),
            idempotency_scope=(
                f"job-retry:{source_id}:{'failed' if failed_only else 'all'}"
            ),
            idempotency_key=idempotency_key,
            idempotency_payload={
                "sourceJobId": source_id,
                "failedOnly": failed_only,
                "strategy": strategy,
                "entryPaths": [
                    str(entry.get("logicalPath", ""))
                    for entry in retry_entries
                ],
            },
        )

    def _retry_web_extract(
        self,
        source: Mapping[str, Any],
        *,
        strategy: str,
        failed_only: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        try:
            config = validate_web_extract_config(source.get("config_json"))
        except WebImportDataInvalid as exc:
            raise JobConflict(f"web extraction retry snapshot is invalid: {exc}") from exc
        chapter_id = _optional_text(source.get("chapter_id"))
        source_url = _optional_text(config.get("sourceUrl"))
        requested_engine = _required_text(config, "requestedEngine")
        if not chapter_id or not source_url:
            raise JobConflict("web extraction retry target no longer exists")
        credentials, plugins = self._original_runtime_snapshots(
            str(source["id"])
        )
        accepted = WebImportCommandService(
            data_root=self.data_root,
            engine=self.engine,
        ).create_draft(
            chapter_id=chapter_id,
            source_url=source_url,
            requested_engine=requested_engine,
            idempotency_key=idempotency_key,
            resolved_options=(
                dict(config["options"])
                if strategy == "original"
                else None
            ),
            retry_of_job_id=str(source["id"]),
            retry_mode=strategy,
            retry_failed_only=failed_only,
            credential_snapshots=(
                credentials if strategy == "original" else None
            ),
            plugin_snapshots=plugins if strategy == "original" else None,
        )
        return {
            key: value
            for key, value in accepted.items()
            if key != "draftId"
        }

    def _retry_web_import_commit(
        self,
        source: Mapping[str, Any],
        selected_items: list[Mapping[str, Any]],
        *,
        strategy: str,
        failed_only: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        source_id = str(source["id"])
        draft_id = _optional_text(source.get("web_import_draft_id"))
        chapter_id = _optional_text(source.get("chapter_id"))
        try:
            config = validate_web_commit_config(source.get("config_json"))
        except WebImportDataInvalid as exc:
            raise JobConflict(f"web import retry snapshot is invalid: {exc}") from exc
        raw_entries = config.get("entries")
        if (
            not draft_id
            or not chapter_id
            or not isinstance(raw_entries, list)
            or not all(isinstance(entry, Mapping) for entry in raw_entries)
        ):
            raise JobConflict("web import retry draft no longer exists")
        entries = [dict(entry) for entry in raw_entries]
        selected_ids = {str(item["id"]) for item in selected_items}
        with self.engine.connect() as connection:
            draft = connection.execute(
                select(web_import_drafts).where(
                    web_import_drafts.c.id == draft_id
                )
            ).mappings().one_or_none()
            step_rows = list(
                connection.execute(
                    select(
                        job_steps.c.job_item_id,
                        job_steps.c.kind,
                    )
                    .where(job_steps.c.job_item_id.in_(selected_ids))
                    .order_by(job_steps.c.job_item_id, job_steps.c.ordinal)
                ).mappings()
            )
        if draft is None:
            raise JobConflict("web import retry draft no longer exists")
        if draft["expires_at"] <= utcnow():
            raise JobConflict("web import retry draft has expired")

        step_kinds: dict[str, list[str]] = defaultdict(list)
        for row in step_rows:
            step_kinds[str(row["job_item_id"])].append(str(row["kind"]))
        retry_entries: list[dict[str, Any]] = []
        needs_finalize = False
        for item in selected_items:
            if not failed_only and str(item["status"]) in {
                "completed",
                "skipped",
            }:
                continue
            kinds = step_kinds.get(str(item["id"]), [])
            if "web_import_commit_page" in kinds:
                entry_index = int(item["ordinal"]) - 1
                if entry_index < 0 or entry_index >= len(entries):
                    raise JobConflict("web import retry checkpoint is invalid")
                retry_entry = dict(entries[entry_index])
                retry_entry.pop("logicalPath", None)
                retry_entries.append(retry_entry)
            if "web_import_commit_finalize" in kinds:
                needs_finalize = True
        if not retry_entries and not needs_finalize:
            raise JobConflict("source job has no retryable web import items")

        base_revision = int(draft["revision"])
        now = utcnow()
        retry_config = {
            "draftId": draft_id,
            "draftRevision": base_revision + 1,
            "chapterId": chapter_id,
            "entries": retry_entries,
            "executionMode": "sequential",
        }
        validate_web_commit_config(retry_config)
        display = _json_object(source.get("target_display_json"))
        credentials, plugins = self._original_runtime_snapshots(source_id)
        item_specs = (
            *(
                JobItemSpec(
                    page_id=None,
                    step_kinds=("web_import_commit_page",),
                )
                for _entry in retry_entries
            ),
            JobItemSpec(
                page_id=None,
                step_kinds=("web_import_commit_finalize",),
            ),
        )
        spec = JobSpec(
            kind="web_import_commit",
            book_id=_optional_text(source.get("book_id")),
            chapter_id=chapter_id,
            web_import_draft_id=draft_id,
            config=retry_config,
            items=item_specs,
            target_display={
                **display,
                "pageCount": len(retry_entries),
                "retryOfJobId": source_id,
                "retryItemCount": len(item_specs),
            },
            credential_snapshots=(
                credentials if strategy == "original" else None
            ),
            plugin_snapshots=plugins if strategy == "original" else None,
            retry_of_job_id=source_id,
            retry_mode=strategy,
        )

        def reopen_draft(connection, _batch_id, _job_ids) -> None:
            changed = connection.execute(
                update(web_import_drafts)
                .where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.revision == base_revision,
                    web_import_drafts.c.status.in_(
                        ("failed", "cancelled", "completed")
                    ),
                    web_import_drafts.c.expires_at > now,
                )
                .values(
                    status="committing",
                    revision=base_revision + 1,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                raise JobConflict(
                    "web import draft changed before retry creation"
                )

        return self.repository.create_batch(
            kind="web_import_commit",
            display_name=(
                f"重试 · {display.get('chapter') or display.get('book') or '网页入库'}"
            ),
            specs=(spec,),
            idempotency_scope=(
                f"job-retry:{source_id}:"
                f"{'failed' if failed_only else 'all'}"
            ),
            idempotency_key=idempotency_key,
            idempotency_payload={
                "sourceJobId": source_id,
                "failedOnly": failed_only,
                "strategy": strategy,
                "draftId": draft_id,
                "draftRevision": base_revision,
                "draftPageIds": [
                    str(entry.get("draftPageId", ""))
                    for entry in retry_entries
                ],
            },
            transaction_hook=reopen_draft,
        )

    def _original_runtime_snapshots(
        self,
        job_id: str,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        with self.engine.connect() as connection:
            credentials = {
                str(row["role"]): str(row["credential_version_id"])
                for row in connection.execute(
                    select(
                        job_credential_snapshots.c.role,
                        job_credential_snapshots.c.credential_version_id,
                    ).where(job_credential_snapshots.c.job_id == job_id)
                ).mappings()
            }
            plugins = {
                str(row["plugin_version_id"]): _json_object(
                    row["config_json"]
                )
                for row in connection.execute(
                    select(job_plugin_snapshots).where(
                        job_plugin_snapshots.c.job_id == job_id
                    )
                ).mappings()
            }
        return credentials, plugins

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
        mode = _required_text(config, "mode")
        execution_mode = _required_text(config, "executionMode")
        reuse_existing_bubbles = _required_boolean(
            config,
            "reuseExistingBubbles",
        )
        style_source: dict[str, object] = {}
        frozen_style = config.get("textStyleSnapshot")
        if frozen_style is not None:
            if not isinstance(frozen_style, Mapping):
                raise JobConflict("translation style snapshot is invalid")
            source_page_id = _required_text(frozen_style, "sourcePageId")
            with self.engine.connect() as connection:
                source_revision = connection.execute(
                    select(pages.c.document_revision).where(
                        pages.c.id == source_page_id,
                        pages.c.chapter_id == str(chapter_id),
                    )
                ).scalar_one_or_none()
            if source_revision is None:
                raise JobConflict(
                    "translation style source page no longer exists"
                )
            style_source = {
                "styleSourcePageId": source_page_id,
                "styleSourceDocumentRevision": int(source_revision),
            }
        return TranslationJobCommandService(self.engine).create_chapter_job(
            chapter_id=str(chapter_id),
            config={
                "mode": mode,
                "executionMode": execution_mode,
                "skipCompleted": False,
                "reuseExistingBubbles": reuse_existing_bubbles,
                **style_source,
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

    def _retry_insight(
        self,
        source: Mapping[str, Any],
        selected_items: list[Mapping[str, Any]],
        *,
        strategy: str,
        failed_only: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        book_id = _optional_text(source.get("book_id"))
        source_run_id = _optional_text(source.get("analysis_run_id"))
        if not book_id or not source_run_id:
            raise JobConflict("Insight retry target run no longer exists")
        original = _json_object(source.get("config_json"))
        scope = _required_text(original, "scope")
        if scope not in {"full", "incremental", "chapter", "page"}:
            raise JobConflict("Insight retry scope is invalid")
        config = (
            SettingsResolver(self.engine).resolve_insight(
                book_id=book_id,
                scope=scope,
            )
            if strategy == "current"
            else dict(original)
        )

        source_id = str(source["id"])
        selected_page_ids = {
            str(item["page_id"])
            for item in selected_items
            if item["page_id"] is not None
        }
        with self.engine.connect() as connection:
            source_targets = list(
                connection.execute(
                    select(analysis_run_targets)
                    .where(analysis_run_targets.c.run_id == source_run_id)
                    .order_by(analysis_run_targets.c.ordinal)
                ).mappings()
            )
            source_results = {
                str(row["page_id_snapshot"]): row
                for row in connection.execute(
                    select(analysis_page_results).where(
                        analysis_page_results.c.run_id == source_run_id
                    )
                ).mappings()
            }
            original_credentials = {
                str(row["role"]): str(row["credential_version_id"])
                for row in connection.execute(
                    select(
                        job_credential_snapshots.c.role,
                        job_credential_snapshots.c.credential_version_id,
                    ).where(job_credential_snapshots.c.job_id == source_id)
                ).mappings()
            }
            original_fonts = {
                str(row["role"]): str(row["font_id"])
                for row in connection.execute(
                    select(
                        job_font_snapshots.c.role,
                        job_font_snapshots.c.font_id,
                    ).where(job_font_snapshots.c.job_id == source_id)
                ).mappings()
            }
            original_plugins = {
                str(row["plugin_version_id"]): _json_object(row["config_json"])
                for row in connection.execute(
                    select(job_plugin_snapshots).where(
                        job_plugin_snapshots.c.job_id == source_id
                    )
                ).mappings()
            }
        if not source_targets:
            raise JobConflict("Insight retry source run has no targets")

        source_target_page_ids = {
            str(row["page_id_snapshot"]) for row in source_targets
        }
        if scope == "full" or not selected_page_ids:
            retry_targets = source_targets
        else:
            retry_targets = [
                target
                for target in source_targets
                if str(target["page_id_snapshot"]) in selected_page_ids
            ]
        target_page_ids = {
            str(row["page_id_snapshot"]) for row in retry_targets
        }
        if not retry_targets or not selected_page_ids.issubset(
            source_target_page_ids
        ):
            raise JobConflict("Insight retry target no longer exists")

        with self.engine.connect() as connection:
            existing_page_ids = {
                str(value)
                for value in connection.execute(
                    select(pages.c.id).where(pages.c.id.in_(target_page_ids))
                ).scalars()
            }
            current_sources = {
                str(row["page_id"]): row
                for row in connection.execute(
                    select(
                        page_assets.c.page_id,
                        page_assets.c.asset_id,
                        assets.c.checksum,
                        assets.c.integrity_status,
                    )
                    .join(assets, assets.c.id == page_assets.c.asset_id)
                    .where(
                        page_assets.c.page_id.in_(target_page_ids),
                        page_assets.c.role == "source",
                    )
                ).mappings()
            }
        if existing_page_ids != target_page_ids:
            raise JobConflict("one or more Insight retry pages no longer exist")

        retry_page_ids = set(selected_page_ids)
        target_mappings: list[dict[str, Any]] = []
        for target in retry_targets:
            page_id = str(target["page_id_snapshot"])
            source_asset_id = str(target["source_asset_id"])
            source_checksum = str(target["source_checksum"])
            current_source = current_sources.get(page_id)
            if (
                current_source is None
                or str(current_source["integrity_status"]) != "ok"
            ):
                raise JobConflict(
                    "current Insight retry input is unavailable; recreate the task"
                )
            current_asset_id = str(current_source["asset_id"])
            current_checksum = str(current_source["checksum"])
            if (
                current_asset_id != source_asset_id
                or current_checksum != source_checksum
            ):
                retry_page_ids.add(page_id)
            source_asset_id = current_asset_id
            source_checksum = current_checksum
            if str(target["status"]) != "completed":
                retry_page_ids.add(page_id)
            if page_id not in source_results:
                retry_page_ids.add(page_id)
            target_mappings.append(
                {
                    "page_id": page_id,
                    "chapter_id": str(target["chapter_id"]),
                    "source_asset_id": source_asset_id,
                    "source_checksum": source_checksum,
                    "page_number": int(target["page_number_snapshot"]),
                }
            )

        new_run_id = str(uuid.uuid4())
        config["runId"] = new_run_id
        config["bookId"] = book_id
        config["targetCount"] = len(target_mappings)
        raw_layers = _json_object(config.get("analysis")).get("layers")
        if not isinstance(raw_layers, list) or not all(
            isinstance(layer, Mapping) for layer in raw_layers
        ):
            raise JobConflict("Insight retry layer configuration is invalid")
        layer_indices = [
            _required_integer(layer, "index", minimum=0)
            for layer in raw_layers
        ]
        if layer_indices != list(range(len(layer_indices))):
            raise JobConflict("Insight retry layer configuration is invalid")
        final_steps = (
            (
                "insight_validate_run",
                *(
                    f"insight_build_layer_{index}"
                    for index in layer_indices
                ),
                "insight_stage_compressed_context",
                "insight_stage_overview_no_spoiler",
                "insight_stage_overview_story_summary",
                "insight_stage_timeline",
                "insight_stage_vectors",
                "insight_publish_run",
            )
            if scope == "full"
            else ("insight_publish_run",)
        )
        target_by_page = {
            str(target["page_id"]): target for target in target_mappings
        }
        item_specs = tuple(
            [
                JobItemSpec(
                    page_id=page_id,
                    step_kinds=("insight_analyze_page",),
                    asset_inputs={
                        "source": str(target_by_page[page_id]["source_asset_id"])
                    },
                )
                for page_id in [
                    str(target["page_id"]) for target in target_mappings
                ]
                if page_id in retry_page_ids
            ]
            + [JobItemSpec(page_id=None, step_kinds=final_steps)]
        )
        retry_page_count = len(item_specs) - 1
        display = _json_object(source.get("target_display_json"))
        spec = JobSpec(
            kind="insight_analysis",
            config=config,
            items=item_specs,
            book_id=book_id,
            chapter_id=_optional_text(source.get("chapter_id")),
            target_display={
                **display,
                "pageCount": retry_page_count,
                "retryOfJobId": source_id,
                "retryItemCount": retry_page_count,
            },
            credential_snapshots=(
                original_credentials if strategy == "original" else None
            ),
            font_snapshots=original_fonts if strategy == "original" else None,
            plugin_snapshots=(
                original_plugins if strategy == "original" else None
            ),
            retry_of_job_id=source_id,
            retry_mode=strategy,
        )
        copied_page_ids = {
            page_id
            for page_id in target_page_ids
            if page_id not in retry_page_ids
        }

        def initialize_retry_run(
            connection,
            _batch_id: str,
            job_ids: list[str],
        ) -> None:
            if len(job_ids) != 1:
                raise RuntimeError("Insight retry must create exactly one job")
            InsightRepository.insert_run(
                connection,
                run_id=new_run_id,
                job_id=str(job_ids[0]),
                book_id=book_id,
                scope=scope,
                config=config,
                targets=target_mappings,
            )
            connection.execute(
                update(jobs)
                .where(jobs.c.id == str(job_ids[0]))
                .values(analysis_run_id=new_run_id)
            )
            InsightRepository.copy_page_successes(
                connection,
                run_id=new_run_id,
                scope=scope,
                copies=tuple(
                    {
                        "page_id": page_id,
                        "source_asset_id": target_by_page[page_id][
                            "source_asset_id"
                        ],
                        "source_checksum": target_by_page[page_id][
                            "source_checksum"
                        ],
                        "page_number": target_by_page[page_id]["page_number"],
                        "payload": _json_object(
                            source_results[page_id]["payload_json"]
                        ),
                    }
                    for page_id in copied_page_ids
                ),
            )

        response = self.repository.create_batch(
            kind="insight_analysis",
            display_name=(
                f"重试 · {display.get('chapter') or display.get('book') or 'Insight'}"
            ),
            specs=(spec,),
            idempotency_scope=(
                f"job-retry:{source_id}:"
                f"{'failed' if failed_only else 'all'}"
            ),
            idempotency_key=idempotency_key,
            idempotency_payload={
                "sourceJobId": source_id,
                "failedOnly": failed_only,
                "strategy": strategy,
                "itemIds": [str(item["id"]) for item in selected_items],
            },
            transaction_hook=initialize_retry_run,
        )
        with self.engine.connect() as connection:
            persisted_run_id = connection.execute(
                select(jobs.c.analysis_run_id).where(
                    jobs.c.id == str(response["jobIds"][0])
                )
            ).scalar_one()
        response["runId"] = str(persisted_run_id)
        return response

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
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise JobConflict("stored job JSON is invalid") from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise JobConflict("stored job JSON must be an object")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise JobConflict("stored job reference must be a non-empty string")
    return value


def _required_text(value: Mapping[str, Any], key: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str) or not selected:
        raise JobConflict(f"stored job field {key} must be a non-empty string")
    return selected


def _required_boolean(value: Mapping[str, Any], key: str) -> bool:
    selected = value.get(key)
    if not isinstance(selected, bool):
        raise JobConflict(f"stored job field {key} must be a boolean")
    return selected


def _required_integer(
    value: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    selected = value.get(key)
    if isinstance(selected, bool) or not isinstance(selected, int):
        raise JobConflict(f"stored job field {key} must be an integer")
    if minimum is not None and selected < minimum:
        raise JobConflict(f"stored job field {key} is out of range")
    return selected
