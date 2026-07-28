"""Worker handlers for webpage extraction and draft-to-chapter commit."""

from __future__ import annotations

from datetime import timedelta
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping
from urllib.parse import unquote, urljoin, urlparse
import uuid

import httpx
from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import (
    ContentRepository,
    _deduplicate_logical_path,
    normalize_logical_path,
)
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobQueueRepository,
    utcnow,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import (
    app_settings,
    chapter_write_locks,
    chapters,
    fonts,
    job_items,
    job_steps,
    jobs,
    page_assets,
    pages,
    web_import_draft_pages,
    web_import_drafts,
    credential_versions,
)


MAX_CANDIDATES = 10_000
MAX_IMAGE_BYTES = 128 * 1024 * 1024
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
    ) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = jobs
        self.storage = AssetStorageService(data_root, engine)
        self.importer = ImageImportService(
            data_root=data_root,
            repository=ContentRepository(engine),
            storage=self.storage,
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
            if kind == "web_import_commit_page":
                return self._commit_page(fence, step)
            if kind == "web_import_commit_finalize":
                return self._finalize_commit(fence, step)
        except Exception as exc:
            if kind.startswith("web_extract"):
                self._record_extract_failure(step, exc)
            raise
        raise ValueError(f"unsupported web import step: {kind}")

    def _scan(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self._config(step)
        source_url = str(config["sourceUrl"])
        requested = str(config.get("requestedEngine", "auto"))
        urls, actual_engine = self._extract_urls(source_url, requested, config)
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
            next_ordinal = 2
            for _entry in entries:
                item_id = str(uuid.uuid4())
                connection.execute(
                    insert(job_items).values(
                        id=item_id,
                        job_id=fence.job_id,
                        ordinal=next_ordinal,
                        status="pending",
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(job_steps).values(
                        id=str(uuid.uuid4()),
                        job_item_id=item_id,
                        ordinal=1,
                        kind="web_extract_page",
                        status="pending",
                        checkpoint_schema_version=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
                next_ordinal += 1
            final_item_id = str(uuid.uuid4())
            connection.execute(
                insert(job_items).values(
                    id=final_item_id,
                    job_id=fence.job_id,
                    ordinal=next_ordinal,
                    status="pending",
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(job_steps).values(
                    id=str(uuid.uuid4()),
                    job_item_id=final_item_id,
                    ordinal=1,
                    kind="web_extract_finalize",
                    status="pending",
                    checkpoint_schema_version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
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
            if existing_path.is_file():
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
        checksum = self._download(
            str(entry["sourceUrl"]),
            target,
            config.get("options"),
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
        try:
            self.jobs.complete_step(
                fence,
                step_id=str(step["stepId"]),
                checkpoint=checkpoint,
                input_fingerprint=checksum,
                publisher=publish,
            )
        except Exception:
            self.storage.collect_garbage()
            raise
        return {**checkpoint, "__already_published__": True}

    def _finalize_extract(
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
                    web_import_drafts.c.status == "extracting",
                )
                .values(
                    status="ready",
                    revision=web_import_drafts.c.revision + 1,
                    expires_at=now + timedelta(hours=24),
                    updated_at=now,
                )
            )

        checkpoint = {"draftId": draft_id, "status": "ready"}
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
        entry = self._entry(config, int(step["itemOrdinal"]) - 1)
        draft_id = str(config["draftId"])
        source_path = self.storage.resolve_relative_path(
            str(entry["relativePath"])
        )
        if not source_path.is_file():
            raise ValueError("draft source file expired or is missing")
        if _sha256_file(source_path) != entry["checksum"]:
            raise ValueError("draft source checksum changed")
        with source_path.open("rb") as source:
            source_asset, thumbnail = self.importer.publish_replacement(source)
        page_id = str(uuid.uuid4())
        chapter_id = str(
            self._job_target(fence.job_id)["chapter_id"]
        )
        logical_path = self._logical_path(entry)
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
            existing_paths = set(
                connection.execute(
                    select(pages.c.logical_source_path).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalars()
            )
            final_path = _deduplicate_logical_path(
                logical_path,
                existing_paths,
            )
            ordinal = int(
                connection.execute(
                    select(func.coalesce(func.max(pages.c.ordinal), 0)).where(
                        pages.c.chapter_id == chapter_id
                    )
                ).scalar_one()
            ) + 1
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=chapter_id,
                    ordinal=ordinal,
                    logical_source_path=final_path,
                    default_font_id=connection.execute(
                        select(fonts.c.id)
                        .where(fonts.c.kind == "builtin")
                        .order_by(fonts.c.created_at)
                        .limit(1)
                    ).scalar_one_or_none(),
                    page_style_defaults_json=(
                        connection.execute(
                            select(app_settings.c.payload_json).where(
                                app_settings.c.domain
                                == "text_style_defaults"
                            )
                        ).scalar_one_or_none()
                        or "{}"
                    ),
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
        try:
            self.jobs.complete_step(
                fence,
                step_id=str(step["stepId"]),
                checkpoint=checkpoint,
                input_fingerprint=str(entry["checksum"]),
                publisher=publish,
            )
        except Exception:
            self.storage.collect_garbage()
            raise
        return {**checkpoint, "__already_published__": True}

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
        source_url: str,
        requested: str,
        config: Mapping[str, Any],
    ) -> tuple[list[str], str]:
        if requested in {"gallery-dl", "auto"}:
            gallery_urls = _gallery_dl_urls(source_url)
            if gallery_urls:
                return _deduplicate_urls(gallery_urls), "gallery-dl"
            if requested == "gallery-dl":
                raise ValueError("gallery-dl does not support this URL")
        if requested == "ai-agent":
            configured = config.get("options")
            agent = (
                configured.get("agent")
                if isinstance(configured, Mapping)
                else None
            )
            if not isinstance(agent, Mapping) or not agent.get(
                "credentialVersionId"
            ):
                raise ValueError(
                    "ai-agent extraction requires a credentialVersionId"
                )
            return self._run_ai_agent(source_url, configured), "ai-agent"
        return _html_image_urls(source_url), "html"

    def _run_ai_agent(
        self,
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
        api_key = agent_secret.get("api_key", agent_secret.get("apiKey", ""))
        firecrawl_key = firecrawl_secret.get(
            "api_key",
            firecrawl_secret.get("apiKey", ""),
        )
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
            }
        )
        result = agent.extract(source_url)
        if not result.success:
            raise ValueError(result.error or "AI Agent extraction failed")
        urls = [
            str(page.get("imageUrl", "")).strip()
            for page in result.pages
            if isinstance(page, Mapping) and page.get("imageUrl")
        ]
        return _deduplicate_urls(urls)

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
        timeout = min(max(float(options.get("timeout", 30)), 1), 300)
        retries = min(max(int(options.get("retries", 2)), 0), 10)
        delay = min(max(float(options.get("delay", 0)), 0), 30)
        headers = {
            "User-Agent": (
                str(options.get("userAgent"))
                if options.get("userAgent")
                else "Saber-Translator/2"
            ),
            "Accept": "image/*,*/*;q=0.5",
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
                            for chunk in response.iter_bytes(1024 * 1024):
                                byte_size += len(chunk)
                                if byte_size > MAX_IMAGE_BYTES:
                                    raise ValueError(
                                        "candidate exceeds the single-file byte limit"
                                    )
                                digest.update(chunk)
                                output.write(chunk)
                if byte_size == 0:
                    raise ValueError("candidate image is empty")
                temporary.replace(target)
                return digest.hexdigest()
            except Exception as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < retries and delay:
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _credential_secret(self, version_id: object) -> dict[str, Any]:
        if not isinstance(version_id, str) or not version_id:
            return {}
        with self.engine.connect() as connection:
            raw = connection.execute(
                select(credential_versions.c.secret_json).where(
                    credential_versions.c.id == version_id
                )
            ).scalar_one_or_none()
        if raw is None:
            raise ValueError("frozen web import credential no longer exists")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("frozen web import credential is invalid")
        return value

    def _record_extract_failure(
        self,
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

    def _existing_draft_page(self, draft_id: str, page_id: str):
        with self.engine.connect() as connection:
            return connection.execute(
                select(web_import_draft_pages).where(
                    web_import_draft_pages.c.id == page_id,
                    web_import_draft_pages.c.draft_id == draft_id,
                )
            ).mappings().one_or_none()

    def _job_target(self, job_id: str):
        with self.engine.connect() as connection:
            return connection.execute(
                select(jobs.c.chapter_id).where(jobs.c.id == job_id)
            ).mappings().one()

    @staticmethod
    def _config(step: Mapping[str, Any]) -> dict[str, Any]:
        value = step.get("config")
        if not isinstance(value, Mapping):
            raise RuntimeError("web import job configuration is invalid")
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
    def _logical_path(entry: Mapping[str, Any]) -> str:
        parsed = urlparse(str(entry["sourceUrl"]))
        name = Path(unquote(parsed.path)).name
        if not name or "." not in name:
            name = f"page_{int(entry['ordinal']):05d}.png"
        try:
            return normalize_logical_path(name)
        except ValueError:
            return f"page_{int(entry['ordinal']):05d}.png"


class _ImageTagParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.urls: list[str] = []

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
                self.urls.append(urljoin(self.base_url, str(values[key])))
                break
        srcset = values.get("srcset") or values.get("data-srcset")
        if srcset:
            candidates = [
                part.strip().split()[0]
                for part in srcset.split(",")
                if part.strip()
            ]
            if candidates:
                self.urls.append(urljoin(self.base_url, candidates[-1]))


def _html_image_urls(url: str) -> list[str]:
    max_html_bytes = 16 * 1024 * 1024
    with httpx.Client(
        follow_redirects=True,
        timeout=30,
        headers={"User-Agent": "Saber-Translator/2"},
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get(
                "content-type", ""
            ).split(";", 1)[0]
            if content_type.casefold() in IMAGE_CONTENT_TYPES:
                return [str(response.url)]
            payload = bytearray()
            for chunk in response.iter_bytes():
                payload.extend(chunk)
                if len(payload) > max_html_bytes:
                    raise ValueError("webpage HTML exceeds the 16 MiB limit")
            final_url = str(response.url)
            encoding = response.encoding or "utf-8"
    parser = _ImageTagParser(final_url)
    parser.feed(payload.decode(encoding, errors="replace"))
    return _deduplicate_urls(parser.urls)


def _gallery_dl_urls(url: str) -> list[str]:
    try:
        from gallery_dl import job

        class Collector(job.Job):
            def __init__(self, target_url: str) -> None:
                self.urls: list[str] = []
                super().__init__(target_url)

            def handle_url(
                self,
                found_url: str,
                _keywords: object,
            ) -> None:
                if len(self.urls) >= MAX_CANDIDATES:
                    raise ValueError("gallery-dl returned too many candidates")
                self.urls.append(found_url)

        collector = Collector(url)
        collector.run()
        return collector.urls
    except Exception:
        return []


def _deduplicate_urls(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= MAX_CANDIDATES:
            break
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
