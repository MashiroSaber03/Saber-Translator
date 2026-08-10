"""Worker handlers for webpage extraction and draft-to-chapter commit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urljoin, urlparse
import uuid

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.content.image_import import (
    FORMAT_DETAILS,
    ImageImportService,
    ImportSafetyLimits,
)
from src.backend_v2.content.page_style import resolve_new_page_style
from src.backend_v2.content.repository import (
    ContentRepository,
    _deduplicate_logical_path,
    normalize_logical_path,
)
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobQueueRepository,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    assets,
    chapter_write_locks,
    chapters,
    job_items,
    job_steps,
    jobs,
    page_assets,
    pages,
    web_import_draft_pages,
    web_import_drafts,
)
from src.backend_v2.web_import.commands import WebImportCommandService
from src.shared.memory_errors import is_memory_allocation_error


IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}


class WebImportWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        limits: ImportSafetyLimits = ImportSafetyLimits(),
    ) -> None:
        self.engine = engine
        self.jobs = jobs
        self.limits = limits
        self.storage = AssetStorageService(data_root, engine)
        self.credentials = SettingsRepository(engine)
        self.commands = WebImportCommandService(
            data_root=data_root,
            engine=engine,
        )
        self.importer = ImageImportService(
            data_root=data_root,
            repository=ContentRepository(engine),
            storage=self.storage,
            limits=limits,
        )

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        kind = str(step["stepKind"])
        try:
            if kind == "web_extract_scan":
                return self._scan(fence, step)
            if kind == "web_extract_page":
                return self._download_page(fence, step)
            if kind == "web_extract_finalize":
                return self._finalize_extract(fence, step)
            if kind == "web_extract_auto_commit":
                return self._auto_commit(fence, step)
            if kind == "web_import_commit_page":
                return self._commit_page(fence, step)
            if kind == "web_import_commit_finalize":
                return self._finalize_commit(fence, step)
        except AttemptFenced:
            raise
        except Exception as exc:
            if kind.startswith("web_extract"):
                self._record_extract_failure(fence, step, exc)
            raise
        raise ValueError(f"unsupported web import step: {kind}")

    def handle_download_batch(
        self,
        fence: AttemptFence,
        steps: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Download one bounded batch while each page keeps its own checkpoint."""

        with ThreadPoolExecutor(
            max_workers=len(steps),
            thread_name_prefix="web-import-download",
        ) as executor:
            work = [
                (step, executor.submit(self.handle, fence, step))
                for step in steps
            ]
            for step, future in work:
                try:
                    future.result()
                except AttemptFenced:
                    raise
                except Exception as exc:
                    self.jobs.fail_step(
                        fence,
                        step_id=str(step["stepId"]),
                        code="WEB_IMPORT_DOWNLOAD_FAILED",
                        message=str(exc),
                    )
        return {
            "processed": len(steps),
            "__already_published__": True,
        }

    def _scan(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        source_url = str(config["sourceUrl"])
        requested = str(config.get("requestedEngine", "auto"))
        urls, actual_engine = self._extract_urls(
            fence,
            source_url,
            requested,
            config,
        )
        if not urls:
            raise ValueError("webpage extraction found no image candidates")
        entries = [
            {
                "draftPageId": str(uuid.uuid4()),
                "ordinal": ordinal,
                "sourceUrl": url,
            }
            for ordinal, url in enumerate(urls, start=1)
        ]
        new_config = {
            **config,
            "actualEngine": actual_engine,
            "entries": entries,
        }
        now = utcnow()
        draft_id = str(config["draftId"])

        def publish(connection: Connection) -> None:
            step_kinds = ["web_extract_page"] * len(entries)
            step_kinds.append("web_extract_finalize")
            options = self._options(config)
            if bool(options["autoImport"]):
                step_kinds.append("web_extract_auto_commit")
            item_rows: list[dict[str, Any]] = []
            step_rows: list[dict[str, Any]] = []
            for next_ordinal, step_kind in enumerate(step_kinds, start=2):
                item_id = str(uuid.uuid4())
                item_rows.append(
                    {
                        "id": item_id,
                        "job_id": fence.job_id,
                        "ordinal": next_ordinal,
                        "status": "pending",
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                step_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "job_item_id": item_id,
                        "ordinal": 1,
                        "kind": step_kind,
                        "status": "pending",
                        "checkpoint_schema_version": 1,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
            connection.execute(insert(job_items), item_rows)
            connection.execute(insert(job_steps), step_rows)
            serialized = _json(new_config)
            connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                )
                .values(config_json=serialized, updated_at=now)
            )
            connection.execute(
                update(web_import_drafts)
                .where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.status == "extracting",
                )
                .values(
                    config_json=serialized,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )

        checkpoint = {
            "candidateCount": len(entries),
            "actualEngine": actual_engine,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _download_page(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        entry = self._entry(config, int(step["itemOrdinal"]) - 2)
        draft_id = str(config["draftId"])
        draft_page_id = str(entry["draftPageId"])
        existing = self._existing_draft_page(draft_id, draft_page_id)
        if existing and existing["checksum"]:
            existing_path = self.storage.resolve_relative_path(
                str(existing["temp_relative_path"])
            )
            thumbnail_path = (
                self.storage.resolve_relative_path(
                    str(existing["thumbnail_relative_path"])
                )
                if existing["thumbnail_relative_path"]
                else None
            )
            if (
                existing_path.is_file()
                and thumbnail_path is not None
                and thumbnail_path.is_file()
                and _sha256_file(existing_path) == existing["checksum"]
            ):
                checkpoint = {
                    "draftPageId": draft_page_id,
                    "checksum": existing["checksum"],
                    "reused": True,
                }
                self.jobs.complete_step(
                    fence,
                    step_id=str(step["stepId"]),
                    checkpoint=checkpoint,
                )
                return {**checkpoint, "__already_published__": True}

        relative = (
            Path("temp")
            / "web-import"
            / draft_id
            / f"{int(entry['ordinal']):06d}-{draft_page_id}.image"
        )
        target = self.storage.resolve_relative_path(relative.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        options = self._options(config)
        downloaded_checksum = self._download(
            str(entry["sourceUrl"]),
            target,
            options,
        )
        checksum = self._preprocess_image(
            target,
            options["imagePreprocess"],
            downloaded_checksum,
        )
        thumbnail = self.importer.publish_draft_thumbnail(target)
        now = utcnow()

        def publish(connection: Connection) -> None:
            connection.execute(
                delete(web_import_draft_pages).where(
                    web_import_draft_pages.c.id == draft_page_id,
                    web_import_draft_pages.c.draft_id == draft_id,
                )
            )
            connection.execute(
                insert(web_import_draft_pages).values(
                    id=draft_page_id,
                    draft_id=draft_id,
                    ordinal=int(entry["ordinal"]),
                    selected=True,
                    source_url=str(entry["sourceUrl"]),
                    temp_relative_path=relative.as_posix(),
                    thumbnail_asset_id=thumbnail.id,
                    checksum=checksum,
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                update(web_import_drafts)
                .where(web_import_drafts.c.id == draft_id)
                .values(
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )

        checkpoint = {
            "draftPageId": draft_page_id,
            "checksum": checksum,
            "thumbnailAssetId": thumbnail.id,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            input_fingerprint=checksum,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _auto_commit(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        draft_id = str(self._config(step)["draftId"])
        draft = self.commands.get_draft(draft_id)
        if int(draft["candidateCount"]) == int(draft["failedCount"]):
            checkpoint = {
                "draftId": draft_id,
                "status": "skipped",
                "reason": "no_successful_pages",
            }
        else:
            accepted = self.commands.commit(
                draft_id=draft_id,
                base_revision=int(draft["revision"]),
                idempotency_key=f"web-import-auto-commit:{draft_id}",
                selected_only=False,
            )
            checkpoint = {
                "draftId": draft_id,
                "status": "queued",
                "batchId": accepted["batchId"],
                "jobIds": accepted["jobIds"],
            }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
        )
        return {**checkpoint, "__already_published__": True}

    def _finalize_extract(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        draft_id = str(config["draftId"])
        now = utcnow()

        with self.engine.connect() as connection:
            successful_count = int(
                connection.execute(
                    select(func.count()).where(
                        web_import_draft_pages.c.draft_id == draft_id,
                        web_import_draft_pages.c.checksum.is_not(None),
                        web_import_draft_pages.c.error_json.is_(None),
                    )
                ).scalar_one()
            )
        final_status = "ready" if successful_count else "failed"

        def publish(connection: Connection) -> None:
            connection.execute(
                update(web_import_drafts)
                .where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.status == "extracting",
                )
                .values(
                    status=final_status,
                    revision=web_import_drafts.c.revision + 1,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )

        checkpoint = {
            "draftId": draft_id,
            "status": final_status,
            "successfulCount": successful_count,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _commit_page(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        if not all(
            isinstance(entry, Mapping) and entry.get("logicalPath")
            for entry in config.get("entries", [])
        ):
            config = self._freeze_commit_paths(fence, config)
        entry = self._entry(config, int(step["itemOrdinal"]) - 1)
        draft_id = str(config["draftId"])
        source_path = self.storage.resolve_relative_path(
            str(entry["relativePath"])
        )
        if not source_path.is_file():
            raise ValueError("draft source file expired or is missing")
        if _sha256_file(source_path) != entry["checksum"]:
            raise ValueError("draft source checksum changed")
        thumbnail = self.storage.get_record(str(entry["thumbnailAssetId"]))
        if (
            thumbnail is None
            or thumbnail.mime_type != "image/webp"
            or not self.storage.resolve_relative_path(
                thumbnail.relative_path
            ).is_file()
        ):
            raise ValueError("draft thumbnail expired or is missing")
        with source_path.open("rb") as source:
            source_asset = self.importer.publish_standalone_source(source)
        page_id = str(uuid.uuid4())
        chapter_id = str(config["chapterId"])
        logical_path = normalize_logical_path(str(entry["logicalPath"]))
        now = utcnow()

        def publish(connection: Connection) -> None:
            if connection.execute(
                select(chapter_write_locks.c.job_id).where(
                    chapter_write_locks.c.chapter_id == chapter_id,
                    chapter_write_locks.c.job_id == fence.job_id,
                    chapter_write_locks.c.owner_attempt_id
                    == fence.attempt_id,
                )
            ).scalar_one_or_none() is None:
                raise RuntimeError("web import commit lost its chapter lock")
            draft = connection.execute(
                select(web_import_drafts.c.status).where(
                    web_import_drafts.c.id == draft_id
                )
            ).scalar_one_or_none()
            if draft != "committing":
                raise RuntimeError("web import draft is no longer committing")
            ordinal = int(
                connection.execute(
                    select(func.coalesce(func.max(pages.c.ordinal), 0)).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalar_one()
            ) + 1
            default_font_id, style_defaults = resolve_new_page_style(connection)
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=chapter_id,
                    ordinal=ordinal,
                    logical_source_path=logical_path,
                    default_font_id=default_font_id,
                    page_style_defaults_json=_json(style_defaults),
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(page_assets),
                [
                    {
                        "page_id": page_id,
                        "role": "source",
                        "asset_id": source_asset.id,
                        "input_source_revision": 1,
                        "parent_asset_id": None,
                    },
                    {
                        "page_id": page_id,
                        "role": "thumbnail_source",
                        "asset_id": thumbnail.id,
                        "input_source_revision": 1,
                        "parent_asset_id": source_asset.id,
                    },
                ],
            )
            connection.execute(
                update(job_items)
                .where(job_items.c.id == step["itemId"])
                .values(page_id=page_id, updated_at=now)
            )
            connection.execute(
                update(chapters)
                .where(chapters.c.id == chapter_id)
                .values(
                    page_order_revision=chapters.c.page_order_revision + 1,
                    updated_at=now,
                )
            )
            connection.execute(
                update(web_import_drafts)
                .where(web_import_drafts.c.id == draft_id)
                .values(
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )

        checkpoint = {
            "draftPageId": entry["draftPageId"],
            "pageId": page_id,
            "sourceAssetId": source_asset.id,
            "thumbnailAssetId": thumbnail.id,
        }
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            input_fingerprint=str(entry["checksum"]),
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _freeze_commit_paths(
        self,
        fence: AttemptFence,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw_entries = config.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise RuntimeError("web import commit snapshot is invalid")
        candidates: list[tuple[dict[str, Any], str]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise RuntimeError("web import commit entry is invalid")
            entry = dict(raw_entry)
            if not isinstance(entry.get("thumbnailAssetId"), str):
                raise RuntimeError("web import commit thumbnail is invalid")
            source_path = self.storage.resolve_relative_path(
                str(entry["relativePath"])
            )
            if not source_path.is_file():
                raise ValueError("draft source file expired or is missing")
            try:
                with Image.open(source_path) as image:
                    image_format = str(image.format or "").upper()
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError("draft source is not a decodable image") from exc
            details = FORMAT_DETAILS.get(image_format)
            if details is None:
                raise ValueError("draft source uses an unsupported image format")
            candidates.append(
                (
                    entry,
                    self._logical_path(entry, extension=details[0]),
                )
            )

        chapter_id = str(config["chapterId"])
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            if connection.execute(
                select(chapter_write_locks.c.job_id).where(
                    chapter_write_locks.c.chapter_id == chapter_id,
                    chapter_write_locks.c.job_id == fence.job_id,
                    chapter_write_locks.c.owner_attempt_id == fence.attempt_id,
                )
            ).scalar_one_or_none() is None:
                raise RuntimeError("web import commit lost its chapter lock")
            used_paths = set(
                connection.execute(
                    select(pages.c.logical_source_path).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalars()
            )
            entries: list[dict[str, Any]] = []
            for entry, candidate in candidates:
                logical_path = _deduplicate_logical_path(candidate, used_paths)
                used_paths.add(logical_path)
                entries.append({**entry, "logicalPath": logical_path})
            frozen = {**config, "entries": entries}
            changed = connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == fence.job_id,
                    jobs.c.attempt_id == fence.attempt_id,
                    jobs.c.lease_token == fence.lease_token,
                    jobs.c.status.in_(("running", "pausing", "cancelling")),
                )
                .values(config_json=_json(frozen), updated_at=now)
            )
            if changed.rowcount != 1:
                raise AttemptFenced(f"job attempt is no longer current: {fence.job_id}")
        return frozen

    def _finalize_commit(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        draft_id = str(config["draftId"])
        now = utcnow()

        def publish(connection: Connection) -> None:
            connection.execute(
                update(web_import_drafts)
                .where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.status == "committing",
                )
                .values(
                    status="completed",
                    revision=web_import_drafts.c.revision + 1,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )

        checkpoint = {"draftId": draft_id, "status": "completed"}
        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _extract_urls(
        self,
        fence: AttemptFence,
        source_url: str,
        requested: str,
        config: Mapping[str, Any],
    ) -> tuple[list[str], str]:
        options = self._options(config)
        if requested in {"gallery-dl", "auto"}:
            gallery_urls = _gallery_dl_urls(
                source_url,
                max_candidates=self.limits.max_container_pages,
            )
            if gallery_urls:
                return (
                    _deduplicate_urls(
                        gallery_urls,
                        max_candidates=self.limits.max_container_pages,
                    ),
                    "gallery-dl",
                )
            if requested == "gallery-dl":
                raise ValueError("gallery-dl does not support this URL")
        if requested == "ai-agent":
            agent = options.get("agent")
            if not isinstance(agent, Mapping) or not agent.get(
                "credentialVersionId"
            ):
                raise ValueError(
                    "ai-agent extraction requires a credentialVersionId"
                )
            return self._run_ai_agent(fence, source_url, options), "ai-agent"
        return (
            _html_image_urls(
                source_url,
                timeout=float(options["timeout"]),
                headers=self._request_headers(
                    options,
                    accept="text/html,image/*;q=0.8,*/*;q=0.5",
                ),
                bypass_proxy=bool(options["bypassProxy"]),
                max_html_bytes=self.limits.max_html_bytes,
                max_candidates=self.limits.max_container_pages,
                stream_chunk_bytes=self.limits.stream_chunk_bytes,
            ),
            "html",
        )

    def _run_ai_agent(
        self,
        fence: AttemptFence,
        source_url: str,
        options: Mapping[str, Any],
    ) -> list[str]:
        from src.core.web_import import MangaScraperAgent

        agent_options = options.get("agent")
        if not isinstance(agent_options, Mapping):
            raise ValueError("AI Agent settings are missing")
        agent_secret = self._credential_secret(
            agent_options.get("credentialVersionId")
        )
        firecrawl_options = options.get("firecrawl")
        firecrawl_secret = (
            self._credential_secret(
                firecrawl_options.get("credentialVersionId")
            )
            if isinstance(firecrawl_options, Mapping)
            else {}
        )
        api_key = agent_secret.get("api_key", "")
        firecrawl_key = firecrawl_secret.get("api_key", "")
        if not api_key:
            raise ValueError("AI Agent credential has no API key")
        if not firecrawl_key:
            raise ValueError("Firecrawl credential is required for AI Agent")
        extraction = options.get("extraction")
        extraction_options = (
            dict(extraction) if isinstance(extraction, Mapping) else {}
        )
        agent = MangaScraperAgent(
            {
                "firecrawl": {"apiKey": firecrawl_key},
                "agent": {
                    "provider": agent_options.get("provider", ""),
                    "apiKey": api_key,
                    "customBaseUrl": agent_options.get(
                        "custom_base_url",
                        "",
                    ),
                    "modelName": agent_options.get("model_name", ""),
                    "useStream": bool(agent_options.get("useStream", False)),
                    "forceJsonOutput": bool(
                        agent_options.get("forceJsonOutput", True)
                    ),
                    "maxRetries": int(agent_options.get("maxRetries", 3)),
                    "timeout": int(agent_options.get("timeout", 120)),
                },
                "extraction": extraction_options,
                "bypassProxy": bool(options["bypassProxy"]),
            }
        )
        result = agent.extract(
            source_url,
            on_log=lambda log: self.jobs.append_worker_event(
                fence,
                event_type="web_import_agent_log",
                payload={
                    "timestamp": log.timestamp,
                    "type": log.type,
                    "message": log.message,
                },
            ),
        )
        if not result.success:
            raise ValueError(result.error or "AI Agent extraction failed")
        urls = [
            str(page.get("imageUrl", "")).strip()
            for page in result.pages
            if isinstance(page, Mapping) and page.get("imageUrl")
        ]
        return _deduplicate_urls(
            urls,
            max_candidates=self.limits.max_container_pages,
        )

    def _download(
        self,
        url: str,
        target: Path,
        options_value: object,
    ) -> str:
        options = (
            dict(options_value)
            if isinstance(options_value, Mapping)
            else {}
        )
        timeout = float(options["timeout"])
        retries = int(options["retries"])
        delay_seconds = int(options["delay"]) / 1000
        headers = self._request_headers(
            options,
            accept="image/*,*/*;q=0.5",
        )
        bypass_proxy = bool(options["bypassProxy"])
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.unlink(missing_ok=True)
            try:
                digest = hashlib.sha256()
                byte_size = 0
                with httpx.Client(
                    follow_redirects=True,
                    timeout=timeout,
                    headers=headers,
                    trust_env=not bypass_proxy,
                ) as client:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        content_type = response.headers.get(
                            "content-type", ""
                        ).split(";", 1)[0].strip().casefold()
                        if (
                            content_type
                            and content_type not in IMAGE_CONTENT_TYPES
                        ):
                            raise ValueError(
                                f"candidate is not an image ({content_type})"
                            )
                        with temporary.open("xb") as output:
                            for chunk in response.iter_bytes(
                                self.limits.stream_chunk_bytes
                            ):
                                byte_size += len(chunk)
                                digest.update(chunk)
                                output.write(chunk)
                if byte_size == 0:
                    raise ValueError("candidate image is empty")
                temporary.replace(target)
                return digest.hexdigest()
            except Exception as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if not _retryable_download_error(exc) or attempt == retries:
                    raise
                if delay_seconds:
                    time.sleep(delay_seconds)
        assert last_error is not None
        raise last_error

    def _request_headers(
        self,
        options: Mapping[str, Any],
        *,
        accept: str,
    ) -> dict[str, str]:
        headers = {
            "User-Agent": "Saber-Translator/2",
            "Accept": accept,
        }
        if options.get("referer"):
            headers["Referer"] = str(options["referer"])
        http_credential = options.get("http")
        if isinstance(http_credential, Mapping):
            secret = self._credential_secret(
                http_credential.get("credentialVersionId")
            )
            cookie = secret.get("cookie")
            if cookie:
                headers["Cookie"] = str(cookie)
            custom_headers = secret.get("headers")
            if isinstance(custom_headers, Mapping):
                for name, value in custom_headers.items():
                    normalized_name = str(name).strip()
                    if (
                        normalized_name
                        and normalized_name.casefold()
                        not in {"host", "content-length", "connection"}
                    ):
                        headers[normalized_name] = str(value)
        return headers

    @staticmethod
    def _preprocess_image(
        path: Path,
        settings_value: object,
        source_checksum: str,
    ) -> str:
        if not isinstance(settings_value, Mapping):
            raise ValueError("web import image preprocessing settings are missing")
        settings = dict(settings_value)
        if not bool(settings["enabled"]):
            return source_checksum

        compression = settings["compression"]
        conversion = settings["formatConvert"]
        if not isinstance(compression, Mapping) or not isinstance(
            conversion,
            Mapping,
        ):
            raise ValueError("web import image preprocessing settings are invalid")

        temporary = path.with_suffix(path.suffix + ".processed")
        temporary.unlink(missing_ok=True)
        try:
            with Image.open(path) as source:
                source_format = str(source.format or "").upper()
                source.load()
                orientation = source.getexif().get(274, 1)
                rotate = bool(settings["autoRotate"]) and orientation != 1
                compress = bool(compression["enabled"])
                convert = (
                    bool(conversion["enabled"])
                    and str(conversion["targetFormat"]) != "original"
                )
                if not rotate and not compress and not convert:
                    return source_checksum

                image = (
                    ImageOps.exif_transpose(source)
                    if rotate
                    else source.copy()
                )
                try:
                    if compress:
                        max_width = int(compression["maxWidth"])
                        max_height = int(compression["maxHeight"])
                        bounds = (
                            max_width or image.width,
                            max_height or image.height,
                        )
                        image.thumbnail(bounds, Image.Resampling.LANCZOS)

                    target_format = (
                        {
                            "jpeg": "JPEG",
                            "png": "PNG",
                            "webp": "WEBP",
                        }[str(conversion["targetFormat"])]
                        if convert
                        else source_format
                    )
                    if target_format not in {
                        "JPEG",
                        "PNG",
                        "WEBP",
                        "GIF",
                        "BMP",
                        "TIFF",
                    }:
                        raise ValueError(
                            f"unsupported image format: {target_format or 'unknown'}"
                        )

                    output = image
                    if target_format == "JPEG" and image.mode not in {"RGB", "L"}:
                        if "A" in image.getbands():
                            rgba = image.convert("RGBA")
                            output = Image.new("RGB", rgba.size, "white")
                            output.paste(rgba, mask=rgba.getchannel("A"))
                            rgba.close()
                        else:
                            output = image.convert("RGB")
                    try:
                        save_options: dict[str, Any] = {}
                        quality = int(compression["quality"])
                        if target_format == "JPEG":
                            save_options = {"quality": quality, "optimize": True}
                        elif target_format == "WEBP":
                            save_options = {
                                "quality": quality,
                                "method": 4,
                            }
                        elif target_format == "PNG":
                            save_options = {"optimize": True}
                        output.save(
                            temporary,
                            format=target_format,
                            **save_options,
                        )
                    finally:
                        if output is not image:
                            output.close()
                finally:
                    image.close()
            temporary.replace(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("candidate is not a decodable image") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return _sha256_file(path)

    def _credential_secret(self, version_id: object) -> dict[str, Any]:
        if not isinstance(version_id, str) or not version_id:
            return {}
        try:
            return self.credentials.resolve_secret(version_id)
        except LookupError as exc:
            raise ValueError(
                "frozen web import credential no longer exists"
            ) from exc

    def _record_extract_failure(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        error: Exception,
    ) -> None:
        config = step.get("config")
        if not isinstance(config, Mapping) or not config.get("draftId"):
            return
        draft_id = str(config["draftId"])
        now = utcnow()
        if step.get("stepKind") == "web_extract_page":
            try:
                entry = self._entry(
                    config,
                    int(step["itemOrdinal"]) - 2,
                )
            except Exception:
                entry = None
            if entry:
                with self.engine.begin() as connection:
                    if not self._attempt_is_current(connection, fence):
                        return
                    connection.execute(
                        delete(web_import_draft_pages).where(
                            web_import_draft_pages.c.id
                            == entry["draftPageId"]
                        )
                    )
                    connection.execute(
                        insert(web_import_draft_pages).values(
                            id=str(entry["draftPageId"]),
                            draft_id=draft_id,
                            ordinal=int(entry["ordinal"]),
                            selected=False,
                            source_url=str(entry["sourceUrl"]),
                            temp_relative_path="",
                            error_json=_json(
                                {
                                    "code": "download_failed",
                                    "message": str(error),
                                }
                            ),
                            created_at=now,
                            updated_at=now,
                        )
                    )
                return
        with self.engine.begin() as connection:
            if not self._attempt_is_current(connection, fence):
                return
            connection.execute(
                update(web_import_drafts)
                .where(
                    web_import_drafts.c.id == draft_id,
                    web_import_drafts.c.status == "extracting",
                )
                .values(
                    status="failed",
                    revision=web_import_drafts.c.revision + 1,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )

    @staticmethod
    def _attempt_is_current(
        connection: Connection,
        fence: AttemptFence,
    ) -> bool:
        return connection.execute(
            select(jobs.c.id).where(
                jobs.c.id == fence.job_id,
                jobs.c.attempt_id == fence.attempt_id,
                jobs.c.lease_token == fence.lease_token,
                jobs.c.status.in_(("running", "pausing", "cancelling")),
            )
        ).scalar_one_or_none() is not None

    def _existing_draft_page(self, draft_id: str, page_id: str):
        with self.engine.connect() as connection:
            return connection.execute(
                select(
                    web_import_draft_pages,
                    assets.c.relative_path.label("thumbnail_relative_path"),
                )
                .outerjoin(
                    assets,
                    assets.c.id
                    == web_import_draft_pages.c.thumbnail_asset_id,
                )
                .where(
                    web_import_draft_pages.c.id == page_id,
                    web_import_draft_pages.c.draft_id == draft_id,
                )
            ).mappings().one_or_none()

    @staticmethod
    def _config(step: Mapping[str, Any]) -> dict[str, Any]:
        value = step.get("config")
        if not isinstance(value, Mapping):
            raise RuntimeError("web import job configuration is invalid")
        return dict(value)

    @staticmethod
    def _options(config: Mapping[str, Any]) -> dict[str, Any]:
        value = config.get("options")
        if not isinstance(value, Mapping):
            raise RuntimeError("web import settings snapshot is invalid")
        return dict(value)

    @staticmethod
    def _entry(
        config: Mapping[str, Any],
        index: int,
    ) -> dict[str, Any]:
        entries = config.get("entries")
        if (
            not isinstance(entries, list)
            or index < 0
            or index >= len(entries)
            or not isinstance(entries[index], dict)
        ):
            raise RuntimeError("web import item snapshot is invalid")
        return dict(entries[index])

    @staticmethod
    def _logical_path(
        entry: Mapping[str, Any],
        *,
        extension: str,
    ) -> str:
        parsed = urlparse(str(entry["sourceUrl"]))
        name = Path(unquote(parsed.path)).name
        if not name or "." not in name:
            name = f"page_{int(entry['ordinal']):05d}.{extension}"
        elif extension:
            name = f"{Path(name).stem}.{extension}"
        try:
            return normalize_logical_path(name)
        except ValueError:
            return f"page_{int(entry['ordinal']):05d}.png"


class _ImageTagParser(HTMLParser):
    def __init__(self, base_url: str, max_candidates: int) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.max_candidates = max_candidates
        self.urls: list[str] = []
        self._seen: set[str] = set()

    def _append(self, value: str) -> None:
        url = urljoin(self.base_url, value)
        if url in self._seen:
            return
        if len(self.urls) >= self.max_candidates:
            raise ValueError("webpage returned too many image candidates")
        self._seen.add(url)
        self.urls.append(url)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() not in {"img", "source"}:
            return
        values = {key.casefold(): value for key, value in attrs if value}
        for key in (
            "data-original",
            "data-src",
            "data-lazy-src",
            "src",
        ):
            if values.get(key):
                self._append(str(values[key]))
                break
        srcset = values.get("srcset") or values.get("data-srcset")
        if srcset:
            candidates = [
                part.strip().split()[0]
                for part in srcset.split(",")
                if part.strip()
            ]
            if candidates:
                self._append(candidates[-1])


def _html_image_urls(
    url: str,
    *,
    timeout: float,
    headers: Mapping[str, str],
    bypass_proxy: bool,
    max_html_bytes: int,
    max_candidates: int,
    stream_chunk_bytes: int = ImportSafetyLimits().stream_chunk_bytes,
) -> list[str]:
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers=dict(headers),
        trust_env=not bypass_proxy,
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get(
                "content-type", ""
            ).split(";", 1)[0]
            if content_type.casefold() in IMAGE_CONTENT_TYPES:
                return [str(response.url)]
            payload = bytearray()
            for chunk in response.iter_bytes(stream_chunk_bytes):
                payload.extend(chunk)
                if len(payload) > max_html_bytes:
                    raise ValueError("webpage HTML exceeds the configured byte limit")
            final_url = str(response.url)
            encoding = response.encoding or "utf-8"
    parser = _ImageTagParser(final_url, max_candidates)
    parser.feed(payload.decode(encoding, errors="replace"))
    return parser.urls


def _gallery_dl_urls(url: str, *, max_candidates: int) -> list[str]:
    try:
        from gallery_dl import job

        class Collector(job.Job):
            def __init__(self, target_url: str) -> None:
                self.urls: list[str] = []
                self.max_candidates = max_candidates
                super().__init__(target_url)

            def handle_url(
                self,
                found_url: str,
                _keywords: object,
            ) -> None:
                if len(self.urls) >= self.max_candidates:
                    raise ValueError("gallery-dl returned too many candidates")
                self.urls.append(found_url)

        collector = Collector(url)
        collector.run()
        return collector.urls
    except ValueError:
        raise
    except Exception as exc:
        if is_memory_allocation_error(exc):
            raise
        return []


def _deduplicate_urls(
    values: list[str],
    *,
    max_candidates: int,
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if value in seen:
            continue
        if len(result) >= max_candidates:
            raise ValueError("webpage returned too many image candidates")
        seen.add(value)
        result.append(value)
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _retryable_download_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in {408, 429} or status >= 500
    return isinstance(error, httpx.TransportError)
