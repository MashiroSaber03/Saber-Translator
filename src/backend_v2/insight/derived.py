"""Versioned Insight overview, timeline, compressed-context, and vector jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Protocol
import uuid

from sqlalchemy import Engine, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_layer_result_pages,
    analysis_layer_results,
    analysis_page_results,
    analysis_run_targets,
    analysis_runs,
    assets,
    chapters,
    pages,
    page_assets,
    timeline_characters,
    timeline_events,
    timeline_versions,
    vector_generations,
    jobs,
)
from src.shared.memory_errors import is_memory_allocation_error


DERIVED_KINDS = frozenset(
    {"overview", "compressed_context", "timeline", "vector"}
)


def _load(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_timeline_error(error: object) -> str:
    return redact_sensitive_text(error)[:1000]


def _normalized_timeline_result(
    result: object,
    *,
    mode: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("timeline response must be an object")
    events = result.get("events")
    characters = result.get("characters")
    if not isinstance(events, list) or not events:
        raise ValueError("timeline response must contain at least one event")
    if not isinstance(characters, list):
        raise ValueError("timeline response is missing characters")
    content = _object(result.get("content"))
    content.update(
        {
            "requested_mode": "enhanced",
            "actual_mode": mode,
            "fallback_reason": fallback_reason,
            "degraded": mode != "enhanced",
        }
    )
    return {
        "mode": mode,
        "content": content,
        "events": events,
        "characters": characters,
    }


def _timeline_thumbnail_page_numbers(
    *,
    content: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    characters: Sequence[Mapping[str, Any]],
) -> set[int]:
    numbers: set[int] = set()

    def add(value: object) -> None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return
        if parsed > 0:
            numbers.add(parsed)

    for event in events:
        page_numbers = event.get("page_numbers", [])
        for value in page_numbers if isinstance(page_numbers, list) else []:
            add(value)
    for character in characters:
        add(character.get("first_page"))
        related_pages = character.get("related_page_numbers", [])
        for value in related_pages if isinstance(related_pages, list) else []:
            add(value)
        key_moments = character.get("key_moments", [])
        for moment in key_moments if isinstance(key_moments, list) else []:
            if isinstance(moment, Mapping):
                add(moment.get("page"))
    plot_arcs = content.get("plot_arcs", [])
    for arc in plot_arcs if isinstance(plot_arcs, list) else []:
        if isinstance(arc, Mapping):
            page_range = _object(arc.get("page_range"))
            add(page_range.get("start"))
    plot_threads = content.get("plot_threads", [])
    for thread in plot_threads if isinstance(plot_threads, list) else []:
        if isinstance(thread, Mapping):
            add(thread.get("introduced_at"))
            add(thread.get("resolved_at"))
    return numbers


@dataclass(frozen=True, slots=True)
class AnalysisInputSnapshot:
    book_id: str
    source_run_id: str | None
    source_run_status: str | None
    result_ids: tuple[str, ...]
    pages: tuple[dict[str, Any], ...]
    fingerprint: str


class DerivedAlgorithms(Protocol):
    def build_layer(
        self,
        inputs: Sequence[Mapping[str, Any]],
        *,
        layer: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_overview(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        template: str,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_compressed_context(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def build_timeline(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def embed_documents(
        self,
        documents: Sequence[str],
        *,
        config: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]: ...


class ProviderDerivedAlgorithms:
    """Current derived-analysis implementation for the Worker."""

    def build_layer(
        self,
        inputs: Sequence[Mapping[str, Any]],
        *,
        layer: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prompt_type = str(layer.get("promptType", "segment_summary"))
        prompt = (
            f"请生成“{layer.get('name', '汇总')}”层级摘要。"
            "只依据输入，保留关键事件、连续性和因果关系。输出 JSON。\n\n"
            + "\n\n".join(_json(dict(value)) for value in inputs)
        )
        result = self._chat_json(
            prompt,
            config=config,
            prompt_type=prompt_type,
        )
        return (
            dict(result)
            if isinstance(result, Mapping)
            else {"summary": str(result)}
        )

    def build_overview(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        template: str,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prompt = (
            f"请根据以下逐页分析生成“{template}”漫画概览。"
            "只依据输入，不补写不存在的情节。输出 JSON，至少包含 title 与 content。\n\n"
            + _page_context(pages)
        )
        result = self._chat_json(prompt, config=config, prompt_type="book_overview")
        return result if isinstance(result, Mapping) else {"content": str(result)}

    def build_compressed_context(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prompt = (
            "把以下漫画逐页分析压缩成可供后续问答和剧情生成使用的上下文。"
            "保留事件顺序、因果、角色状态变化和未解决线索。输出 JSON。\n\n"
            + _page_context(pages)
        )
        result = self._chat_json(prompt, config=config, prompt_type="group_summary")
        return result if isinstance(result, Mapping) else {"content": str(result)}

    def build_timeline(
        self,
        pages: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        enhanced_prompt = (
            "根据以下漫画分析生成增强时间线。输出 JSON："
            '{"content":{"story_summary":"...","plot_arcs":'
            '[{"id":"...","name":"...","description":"...",'
            '"page_range":{"start":1,"end":2},"mood":"...",'
            '"event_ids":["..."]}],"plot_threads":[]},'
            '"events":[{"summary":"...","page_ids":["..."]}],'
            '"characters":[{"name":"...","aliases":[],"description":"...",'
            '"personality":"...","arc":"...","first_page":1,'
            '"key_moments":[{"summary":"...","page":1}],'
            '"related_page_numbers":[1]}]}。'
            "不要把推断写成事实。\n\n"
            + _page_context(pages)
        )
        enhanced_error: Exception | None = None
        try:
            result = self._chat_json(
                enhanced_prompt,
                config=config,
                prompt_type="book_overview",
            )
            return _normalized_timeline_result(
                result,
                mode="enhanced",
                fallback_reason=None,
            )
        except Exception as exc:
            if is_memory_allocation_error(exc):
                raise
            enhanced_error = exc

        compressed_payloads = [
            _object(_object(page.get("analysis")).get("compressed_context"))
            for page in pages
            if _object(_object(page.get("analysis")).get("compressed_context"))
        ]
        compressed_error: Exception | None = None
        if compressed_payloads:
            compressed_prompt = (
                "根据以下压缩上下文生成漫画时间线。输出 JSON，必须包含 "
                "content、events 和 characters；事件使用 page_ids 或 page_numbers "
                "关联来源页面。不要补写上下文中不存在的事实。\n\n"
                + "\n\n".join(_json(value) for value in compressed_payloads)
            )
            try:
                result = self._chat_json(
                    compressed_prompt,
                    config=config,
                    prompt_type="book_overview",
                )
                return _normalized_timeline_result(
                    result,
                    mode="compressed",
                    fallback_reason=_safe_timeline_error(enhanced_error),
                )
            except Exception as exc:
                if is_memory_allocation_error(exc):
                    raise
                compressed_error = exc

        events = []
        story_summary = ""
        for page in pages:
            payload = _object(page.get("analysis"))
            compressed_context = _object(payload.get("compressed_context"))
            if compressed_context and not story_summary:
                story_summary = str(
                    compressed_context.get("story_summary")
                    or compressed_context.get("summary")
                    or compressed_context.get("content")
                    or ""
                )
            for event in payload.get("key_events", []):
                if isinstance(event, Mapping):
                    page_ids = [
                        str(value)
                        for value in page.get(
                            "pageIds",
                            [page.get("pageId", "")],
                        )
                        if value
                    ]
                    page_numbers = [
                        int(value)
                        for value in page.get(
                            "pageNumbers",
                            [page.get("pageNumber", 0)],
                        )
                        if int(value) > 0
                    ]
                    events.append(
                        {
                            "summary": str(event.get("summary", "")),
                            "importance": str(
                                event.get("importance", "normal")
                            ),
                            "page_ids": page_ids,
                            "page_numbers": page_numbers,
                        }
                    )
        if not events:
            reasons = [f"enhanced: {_safe_timeline_error(enhanced_error)}"]
            if compressed_error is not None:
                reasons.append(
                    f"compressed: {_safe_timeline_error(compressed_error)}"
                )
            raise InsightConflict(
                "timeline generation failed in every mode; " + "; ".join(reasons)
            )
        fallback_reason = _safe_timeline_error(enhanced_error)
        if compressed_error is not None:
            fallback_reason += (
                f"; compressed: {_safe_timeline_error(compressed_error)}"
            )
        return {
            "mode": "simple",
            "content": {
                "story_summary": story_summary,
                "requested_mode": "enhanced",
                "actual_mode": "simple",
                "fallback_reason": fallback_reason,
                "degraded": True,
                "source": "page_key_events",
            },
            "events": events,
            "characters": [],
        }

    def embed_documents(
        self,
        documents: Sequence[str],
        *,
        config: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        from src.core.manga_insight.config_models import EmbeddingConfig
        from src.core.manga_insight.embedding_client import EmbeddingClient

        section = _object(config.get("embedding"))
        payload = {
            "provider": section.get("provider", ""),
            "api_key": section.get("api_key", ""),
            "model": section.get("model_name", ""),
            "base_url": section.get("custom_base_url"),
            "credential_version_id": section.get("credential_version_id"),
            "rpm_limit": int(section.get("rpm_limit", 0)),
            "transport_retries": int(section.get("transport_retries", 10)),
            "business_retries": int(section.get("business_retries", 10)),
            "timeout_seconds": float(section.get("timeout_seconds", 0)),
        }
        client = EmbeddingClient(EmbeddingConfig.from_dict(payload))

        async def execute() -> Sequence[Sequence[float]]:
            try:
                return await client.embed_batch(list(documents))
            finally:
                await client.close()

        return asyncio.run(execute())

    @staticmethod
    def _chat_json(
        prompt: str,
        *,
        config: Mapping[str, Any],
        prompt_type: str,
    ) -> object:
        from src.core.manga_insight.config_models import ChatLLMConfig
        from src.core.manga_insight.embedding_client import ChatClient

        section = _object(config.get("chat"))
        if not section.get("provider"):
            section = _object(config.get("vlm"))
        payload = {
            "provider": section.get("provider", ""),
            "api_key": section.get("api_key", ""),
            "model": section.get("model_name", ""),
            "base_url": section.get("custom_base_url"),
            "credential_version_id": section.get("credential_version_id"),
            "openai_options": _object(section.get("openai_options")),
        }
        system = str(
            _object(_object(config.get("prompts")).get("analysis_system")).get(
                "content",
                "",
            )
        )
        configured = str(
            _object(_object(config.get("prompts")).get(prompt_type)).get(
                "content",
                "",
            )
        )
        client = ChatClient(ChatLLMConfig.from_dict(payload))

        async def execute() -> object:
            try:
                return await client.generate_json(
                    f"{configured}\n\n{prompt}".strip(),
                    system=system or None,
                    temperature=0.2,
                )
            finally:
                await client.close()

        return asyncio.run(execute())


@dataclass(frozen=True, slots=True)
class VectorCollectionInspection:
    expected: tuple[str, ...]
    actual: tuple[str, ...]
    missing: tuple[str, ...]
    orphaned: tuple[str, ...]


class InsightVectorStore:
    """Generation-isolated Chroma collections owned exclusively by Worker."""

    def __init__(self, data_root: Path) -> None:
        self.path = data_root / "chroma"

    @staticmethod
    def names(book_id: str, generation: int) -> tuple[str, str]:
        prefix = hashlib.sha256(book_id.encode("utf-8")).hexdigest()[:20]
        return (
            f"b{prefix}_g{generation}_pages",
            f"b{prefix}_g{generation}_events",
        )

    def publish(
        self,
        *,
        book_id: str,
        generation: int,
        page_records: Sequence[Mapping[str, Any]],
        page_embeddings: Sequence[Sequence[float]],
        event_records: Sequence[Mapping[str, Any]],
        event_embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.publish_batches(
            book_id=book_id,
            generation=generation,
            page_batches=((page_records, page_embeddings),) if page_records else (),
            event_batches=((event_records, event_embeddings),) if event_records else (),
        )

    def publish_batches(
        self,
        *,
        book_id: str,
        generation: int,
        page_batches: Iterable[
            tuple[Sequence[Mapping[str, Any]], Sequence[Sequence[float]]]
        ],
        event_batches: Iterable[
            tuple[Sequence[Mapping[str, Any]], Sequence[Sequence[float]]]
        ],
        resume: bool = False,
        initial_page_count: int = 0,
        initial_event_count: int = 0,
        expected_page_count: int | None = None,
        expected_event_count: int | None = None,
        on_batch: Callable[[str, int], bool] | None = None,
    ) -> dict[str, object]:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise InsightConflict("ChromaDB is not installed") from exc
        self.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        page_name, event_name = self.names(book_id, generation)
        if resume:
            pages_collection = client.get_or_create_collection(
                page_name,
                metadata={"hnsw:space": "cosine"},
            )
            events_collection = client.get_or_create_collection(
                event_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            for name in (page_name, event_name):
                try:
                    client.delete_collection(name)
                except Exception:
                    pass
            pages_collection = client.create_collection(
                page_name,
                metadata={"hnsw:space": "cosine"},
            )
            events_collection = client.create_collection(
                event_name,
                metadata={"hnsw:space": "cosine"},
            )
        page_count = initial_page_count
        event_count = initial_event_count
        try:
            for page_records, page_embeddings in page_batches:
                if len(page_records) != len(page_embeddings):
                    raise InsightConflict("page embedding result count mismatch")
                pages_collection.upsert(
                    ids=[str(row["id"]) for row in page_records],
                    embeddings=[list(value) for value in page_embeddings],
                    documents=[str(row["document"]) for row in page_records],
                    metadatas=[dict(row["metadata"]) for row in page_records],
                )
                page_count += len(page_records)
                if on_batch is not None and not on_batch("pages", page_count):
                    return {
                        "completed": False,
                        "pageCount": page_count,
                        "eventCount": event_count,
                    }
            for event_records, event_embeddings in event_batches:
                if len(event_records) != len(event_embeddings):
                    raise InsightConflict("event embedding result count mismatch")
                events_collection.upsert(
                    ids=[str(row["id"]) for row in event_records],
                    embeddings=[list(value) for value in event_embeddings],
                    documents=[str(row["document"]) for row in event_records],
                    metadatas=[dict(row["metadata"]) for row in event_records],
                )
                event_count += len(event_records)
                if on_batch is not None and not on_batch("events", event_count):
                    return {
                        "completed": False,
                        "pageCount": page_count,
                        "eventCount": event_count,
                    }
            if (
                expected_page_count is not None
                and pages_collection.count() != expected_page_count
            ):
                raise InsightConflict("page vector coverage is incomplete")
            if (
                expected_event_count is not None
                and events_collection.count() != expected_event_count
            ):
                raise InsightConflict("event vector coverage is incomplete")
            return {
                "completed": True,
                "pageCount": page_count,
                "eventCount": event_count,
            }
        except AttemptFenced:
            # A newer attempt owns publication. Keep the last fenced checkpoint;
            # the replacement attempt will idempotently upsert from that offset.
            raise
        except Exception:
            for name in (page_name, event_name):
                try:
                    client.delete_collection(name)
                except Exception:
                    pass
            raise

    def expected_collection_names(self, engine: Engine) -> set[str]:
        expected: set[str] = set()
        with engine.connect() as connection:
            rows = connection.execute(
                select(
                    vector_generations.c.book_id,
                    vector_generations.c.generation,
                ).where(vector_generations.c.status != "failed")
            )
            for book_id, generation in rows:
                expected.update(self.names(str(book_id), int(generation)))
        return expected

    def inspect_collections(self, engine: Engine) -> VectorCollectionInspection:
        expected = self.expected_collection_names(engine)
        if not self.path.exists() or not any(self.path.iterdir()):
            return VectorCollectionInspection(
                expected=tuple(sorted(expected)),
                actual=(),
                missing=tuple(sorted(expected)),
                orphaned=(),
            )
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise InsightConflict("ChromaDB is not installed") from exc
        client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        actual = {
            str(getattr(collection, "name", collection))
            for collection in client.list_collections()
        }
        managed = {
            name
            for name in actual
            if re.fullmatch(r"b[0-9a-f]{20}_g[1-9][0-9]*_(?:pages|events)", name)
        }
        return VectorCollectionInspection(
            expected=tuple(sorted(expected)),
            actual=tuple(sorted(actual)),
            missing=tuple(sorted(expected - actual)),
            orphaned=tuple(sorted(managed - expected)),
        )

    def collect_orphan_collections(self, engine: Engine) -> int:
        inspection = self.inspect_collections(engine)
        if not inspection.orphaned:
            return 0
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        deleted = 0
        for name in inspection.orphaned:
            try:
                client.delete_collection(name)
            except Exception:
                continue
            deleted += 1
        return deleted


class InsightDerivedRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def snapshot(
        self,
        *,
        book_id: str,
        frozen_inputs: Sequence[Mapping[str, Any]] | None = None,
    ) -> AnalysisInputSnapshot:
        with self.engine.connect() as connection:
            return self._snapshot(
                connection,
                book_id=book_id,
                frozen_inputs=frozen_inputs,
            )

    def snapshot_for_run(self, *, run_id: str) -> AnalysisInputSnapshot:
        """Read only successful staging results from one isolated full run."""

        source_pointer = page_assets.alias("run_snapshot_source")
        with self.engine.connect() as connection:
            run = connection.execute(
                select(
                    analysis_runs.c.book_id,
                    analysis_runs.c.status,
                ).where(analysis_runs.c.id == run_id)
            ).mappings().one_or_none()
            if run is None:
                raise InsightNotFound("analysis run not found")
            rows = list(
                connection.execute(
                    select(
                        analysis_page_results,
                        analysis_run_targets.c.chapter_id,
                        analysis_run_targets.c.ordinal.label("target_ordinal"),
                        assets.c.checksum.label("current_source_checksum"),
                    )
                    .join(
                        analysis_run_targets,
                        (analysis_run_targets.c.run_id == analysis_page_results.c.run_id)
                        & (
                            analysis_run_targets.c.page_id_snapshot
                            == analysis_page_results.c.page_id_snapshot
                        ),
                    )
                    .join(
                        source_pointer,
                        (source_pointer.c.page_id == analysis_run_targets.c.page_id)
                        & (source_pointer.c.role == "source"),
                    )
                    .join(assets, assets.c.id == source_pointer.c.asset_id)
                    .where(
                        analysis_page_results.c.run_id == run_id,
                        analysis_run_targets.c.status == "completed",
                    )
                    .order_by(analysis_run_targets.c.ordinal)
                ).mappings()
            )
        if not rows:
            raise InsightConflict("analysis run has no successful page results")
        pages_payload = tuple(
            {
                "resultId": str(row["id"]),
                "pageId": str(row["page_id_snapshot"]),
                "pageNumber": int(row["page_number_snapshot"]),
                "chapterId": (
                    str(row["chapter_id"])
                    if row["chapter_id"] is not None
                    else None
                ),
                "sourceChecksum": str(row["source_checksum"]),
                "currentSourceChecksum": str(row["current_source_checksum"]),
                "analysis": _load(row["payload_json"], {}),
            }
            for row in rows
        )
        fingerprint = _analysis_input_fingerprint(pages_payload)
        return AnalysisInputSnapshot(
            book_id=str(run["book_id"]),
            source_run_id=run_id,
            source_run_status=str(run["status"]),
            result_ids=tuple(str(row["id"]) for row in rows),
            pages=pages_payload,
            fingerprint=fingerprint,
        )

    @staticmethod
    def _snapshot(
        connection: Connection,
        *,
        book_id: str,
        frozen_inputs: Sequence[Mapping[str, Any]] | None = None,
    ) -> AnalysisInputSnapshot:
        book_head = connection.execute(
                select(
                    analysis_heads.c.active_run_id,
                    analysis_runs.c.status,
                )
                .join(
                    analysis_runs,
                    analysis_runs.c.id == analysis_heads.c.active_run_id,
                )
                .where(
                    analysis_heads.c.book_id == book_id,
                    analysis_heads.c.page_id.is_(None),
                )
            ).mappings().one_or_none()
        if frozen_inputs is None:
            source_pointer = page_assets.alias("derived_current_source")
            rows = list(
                connection.execute(
                    select(
                        analysis_page_results,
                        chapters.c.ordinal.label("chapter_ordinal"),
                        pages.c.ordinal.label("page_ordinal"),
                        assets.c.checksum.label("current_source_checksum"),
                    )
                    .join(
                        analysis_heads,
                        analysis_heads.c.active_result_id
                        == analysis_page_results.c.id,
                    )
                    .join(pages, pages.c.id == analysis_heads.c.page_id)
                    .join(chapters, chapters.c.id == pages.c.chapter_id)
                    .join(
                        source_pointer,
                        (source_pointer.c.page_id == pages.c.id)
                        & (source_pointer.c.role == "source"),
                    )
                    .join(assets, assets.c.id == source_pointer.c.asset_id)
                    .where(analysis_heads.c.book_id == book_id)
                    .order_by(chapters.c.ordinal, pages.c.ordinal)
                ).mappings()
            )
            ordered_inputs = [
                {
                    "resultId": str(row["id"]),
                    "pageId": str(row["page_id_snapshot"]),
                    "pageNumber": int(row["page_number_snapshot"]),
                    "currentSourceChecksum": str(
                        row["current_source_checksum"]
                    ),
                }
                for row in rows
            ]
        else:
            ordered_inputs = [dict(value) for value in frozen_inputs]
            result_ids = [
                str(value.get("resultId", "")) for value in ordered_inputs
            ]
            if (
                not result_ids
                or any(not value for value in result_ids)
                or len(set(result_ids)) != len(result_ids)
            ):
                raise InsightConflict("frozen analysis inputs are invalid")
            selected = list(
                connection.execute(
                    select(analysis_page_results).where(
                        analysis_page_results.c.id.in_(tuple(result_ids))
                    )
                ).mappings()
            )
            by_id = {str(row["id"]): row for row in selected}
            if set(by_id) != set(result_ids):
                raise InsightNotFound("frozen analysis input no longer exists")
            rows = [by_id[value] for value in result_ids]
        if not rows:
            raise InsightConflict("book has no published page analysis")
        pages_payload = tuple(
            {
                "resultId": str(row["id"]),
                "pageId": str(frozen_input["pageId"]),
                "pageNumber": int(frozen_input["pageNumber"]),
                "sourceChecksum": str(row["source_checksum"]),
                "currentSourceChecksum": str(
                    frozen_input["currentSourceChecksum"]
                ),
                "analysis": _load(row["payload_json"], {}),
            }
            for row, frozen_input in zip(rows, ordered_inputs)
        )
        fingerprint = _analysis_input_fingerprint(pages_payload)
        return AnalysisInputSnapshot(
            book_id=book_id,
            source_run_id=(
                str(book_head["active_run_id"]) if book_head is not None else None
            ),
            source_run_status=(
                str(book_head["status"]) if book_head is not None else None
            ),
            result_ids=tuple(str(row["id"]) for row in rows),
            pages=pages_payload,
            fingerprint=fingerprint,
        )

    def publish_artifact(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        kind: str,
        template: str,
        payload: Mapping[str, Any],
        activate: bool = True,
    ) -> dict[str, Any]:
        status = "building"
        should_activate = False
        if activate:
            current = self._snapshot(connection, book_id=frozen.book_id)
            status = _publication_status(frozen, current)
            should_activate = status in {"ready", "degraded"}
        now = utcnow()
        revision = int(
            connection.execute(
                select(
                    func.coalesce(
                        func.max(analysis_artifacts.c.revision),
                        0,
                    )
                    + 1
                ).where(
                    analysis_artifacts.c.book_id == frozen.book_id,
                    analysis_artifacts.c.kind == kind,
                    analysis_artifacts.c.template == template,
                )
            ).scalar_one()
        )
        if should_activate:
            connection.execute(
                update(analysis_artifacts)
                .where(
                    analysis_artifacts.c.book_id == frozen.book_id,
                    analysis_artifacts.c.kind == kind,
                    analysis_artifacts.c.template == template,
                    analysis_artifacts.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
        artifact_id = str(uuid.uuid4())
        connection.execute(
            insert(analysis_artifacts).values(
                id=artifact_id,
                book_id=frozen.book_id,
                run_id=frozen.source_run_id,
                kind=kind,
                template=template,
                status=status,
                revision=revision,
                is_active=should_activate,
                dependency_fingerprint=frozen.fingerprint,
                payload_json=_json(dict(payload)),
                asset_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        return {
            "artifactId": artifact_id,
            "kind": kind,
            "template": template,
            "status": status,
            "revision": revision,
            "dependencyFingerprint": frozen.fingerprint,
        }

    def layer_units(
        self,
        *,
        run_id: str,
        layer_index: int,
        config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        analysis_config = _object(config.get("analysis"))
        raw_layers = analysis_config.get("layers", [])
        if (
            not isinstance(raw_layers, list)
            or layer_index < 0
            or layer_index >= len(raw_layers)
            or not isinstance(raw_layers[layer_index], Mapping)
        ):
            raise InsightConflict("frozen Insight layer definition is invalid")
        layer = dict(raw_layers[layer_index])
        units_per_group = int(layer.get("unitsPerGroup", 0))
        align_to_chapter = bool(layer.get("alignToChapter", False))
        if layer_index == 0:
            frozen = self.snapshot_for_run(run_id=run_id)
            source_units = [
                {
                    "content": _object(page.get("analysis")),
                    "pages": [
                        {
                            "pageId": str(page["pageId"]),
                            "pageNumber": int(page["pageNumber"]),
                            "chapterId": page.get("chapterId"),
                        }
                    ],
                }
                for page in frozen.pages
            ]
            group_size = int(
                analysis_config.get("pagesPerBatch", units_per_group or 5)
            )
        else:
            with self.engine.connect() as connection:
                rows = list(
                    connection.execute(
                        select(analysis_layer_results)
                        .where(
                            analysis_layer_results.c.run_id == run_id,
                            analysis_layer_results.c.layer_index
                            == layer_index - 1,
                            analysis_layer_results.c.status == "staging",
                        )
                        .order_by(analysis_layer_results.c.unit_index)
                    ).mappings()
                )
                covered_by_result: dict[str, list[Mapping[str, Any]]] = {}
                if rows:
                    for page in connection.execute(
                        select(analysis_layer_result_pages)
                        .where(
                            analysis_layer_result_pages.c.layer_result_id.in_(
                                tuple(str(row["id"]) for row in rows)
                            )
                        )
                        .order_by(
                            analysis_layer_result_pages.c.layer_result_id,
                            analysis_layer_result_pages.c.ordinal,
                        )
                    ).mappings():
                        covered_by_result.setdefault(
                            str(page["layer_result_id"]),
                            [],
                        ).append(page)
                source_units = []
                for row in rows:
                    covered = covered_by_result.get(str(row["id"]), ())
                    source_units.append(
                        {
                            "content": _load(row["content_json"], {}),
                            "pages": [
                                {
                                    "pageId": str(page["page_id_snapshot"]),
                                    "pageNumber": int(
                                        page["page_number_snapshot"]
                                    ),
                                    "chapterId": (
                                        str(row["chapter_id"])
                                        if row["chapter_id"] is not None
                                        else None
                                    ),
                                }
                                for page in covered
                            ],
                        }
                    )
            if not source_units:
                raise InsightConflict(
                    f"Insight layer {layer_index - 1} has no staging units"
                )
            group_size = units_per_group or len(source_units)

        grouped: list[list[dict[str, Any]]] = []
        if align_to_chapter:
            by_chapter: dict[str, list[dict[str, Any]]] = {}
            chapter_order: list[str] = []
            for unit in source_units:
                chapters_in_unit = {
                    str(page.get("chapterId") or "")
                    for page in unit["pages"]
                }
                chapter_key = next(iter(chapters_in_unit), "")
                if chapter_key not in by_chapter:
                    chapter_order.append(chapter_key)
                    by_chapter[chapter_key] = []
                by_chapter[chapter_key].append(unit)
            for chapter_key in chapter_order:
                chapter_units = by_chapter[chapter_key]
                size = group_size or len(chapter_units)
                grouped.extend(
                    chapter_units[offset : offset + size]
                    for offset in range(0, len(chapter_units), size)
                )
        else:
            size = group_size or len(source_units)
            grouped = [
                source_units[offset : offset + size]
                for offset in range(0, len(source_units), size)
            ]
        result: list[dict[str, Any]] = []
        for unit_index, group in enumerate(grouped):
            covered_pages: list[dict[str, Any]] = []
            seen: set[str] = set()
            for source in group:
                for page in source["pages"]:
                    page_id = str(page["pageId"])
                    if page_id not in seen:
                        seen.add(page_id)
                        covered_pages.append(dict(page))
            chapters_in_group = {
                str(page.get("chapterId") or "")
                for page in covered_pages
            }
            chapter_id = (
                next(iter(chapters_in_group))
                if align_to_chapter
                and len(chapters_in_group) == 1
                and next(iter(chapters_in_group))
                else None
            )
            result.append(
                {
                    "unitIndex": unit_index,
                    "chapterId": chapter_id,
                    "pages": covered_pages,
                    "inputs": [dict(source["content"]) for source in group],
                    "layer": {
                        **layer,
                        "promptType": _layer_prompt_type(
                            layer_index=layer_index,
                            layer_count=len(raw_layers),
                            align_to_chapter=align_to_chapter,
                        ),
                    },
                }
            )
        return result

    def summary_inputs(
        self,
        frozen: AnalysisInputSnapshot,
    ) -> tuple[dict[str, Any], ...]:
        """Use the highest complete summary layer, with compact pages as fallback."""

        fallback = tuple(
            {
                "resultId": str(page["resultId"]),
                "pageId": str(page["pageId"]),
                "pageIds": [str(page["pageId"])],
                "pageNumber": int(page["pageNumber"]),
                "pageNumbers": [int(page["pageNumber"])],
                "analysis": {
                    key: value
                    for key, value in _object(page.get("analysis")).items()
                    if key
                    in {
                        "page_summary",
                        "key_events",
                        "continuity_notes",
                        "warnings",
                    }
                },
            }
            for page in frozen.pages
        )
        if not frozen.source_run_id:
            return fallback
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(analysis_layer_results)
                    .where(
                        analysis_layer_results.c.run_id
                        == frozen.source_run_id,
                        analysis_layer_results.c.status.in_(
                            ("staging", "published")
                        ),
                    )
                    .order_by(
                        analysis_layer_results.c.layer_index.desc(),
                        analysis_layer_results.c.unit_index,
                    )
                ).mappings()
            )
            covered_rows = list(
                connection.execute(
                    select(analysis_layer_result_pages).where(
                        analysis_layer_result_pages.c.layer_result_id.in_(
                            tuple(str(row["id"]) for row in rows)
                        )
                    )
                ).mappings()
            ) if rows else []
        covered_by_result: dict[str, list[Mapping[str, Any]]] = {}
        for page in covered_rows:
            covered_by_result.setdefault(
                str(page["layer_result_id"]), []
            ).append(page)
        by_layer: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_layer.setdefault(int(row["layer_index"]), []).append(row)
        expected = {str(page["pageId"]) for page in frozen.pages}
        for layer_index in sorted(by_layer, reverse=True):
            layer_rows = by_layer[layer_index]
            covered = {
                str(page["page_id_snapshot"])
                for row in layer_rows
                for page in covered_by_result.get(str(row["id"]), ())
            }
            if covered != expected:
                continue
            inputs: list[dict[str, Any]] = []
            for row in layer_rows:
                pages_for_result = sorted(
                    covered_by_result.get(str(row["id"]), ()),
                    key=lambda page: int(page["ordinal"]),
                )
                page_ids = [
                    str(page["page_id_snapshot"])
                    for page in pages_for_result
                ]
                page_numbers = [
                    int(page["page_number_snapshot"])
                    for page in pages_for_result
                ]
                inputs.append(
                    {
                        "resultId": str(row["id"]),
                        "pageId": page_ids[0],
                        "pageIds": page_ids,
                        "pageNumber": page_numbers[0],
                        "pageNumbers": page_numbers,
                        "analysis": _load(row["content_json"], {}),
                    }
                )
            return tuple(inputs)
        return fallback

    def compressed_context_input(
        self,
        frozen: AnalysisInputSnapshot,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            statement = select(analysis_artifacts).where(
                analysis_artifacts.c.book_id == frozen.book_id,
                analysis_artifacts.c.kind == "compressed_context",
                analysis_artifacts.c.template == "default",
            )
            if frozen.source_run_id:
                staged = connection.execute(
                    statement.where(
                        analysis_artifacts.c.run_id == frozen.source_run_id,
                        analysis_artifacts.c.status.in_(
                            ("building", "ready", "degraded", "stale")
                        ),
                    ).order_by(analysis_artifacts.c.revision.desc())
                ).mappings().first()
            else:
                staged = None
            row = staged or connection.execute(
                statement.where(analysis_artifacts.c.is_active.is_(True))
                .order_by(analysis_artifacts.c.revision.desc())
            ).mappings().first()
        if row is None:
            return None
        return {
            "resultId": str(row["id"]),
            "pageId": str(frozen.pages[0]["pageId"]),
            "pageIds": [str(page["pageId"]) for page in frozen.pages],
            "pageNumber": int(frozen.pages[0]["pageNumber"]),
            "pageNumbers": [int(page["pageNumber"]) for page in frozen.pages],
            "analysis": {
                "compressed_context": _load(row["payload_json"], {}),
            },
        }

    @staticmethod
    def publish_layer(
        connection: Connection,
        *,
        run_id: str,
        layer_index: int,
        layer_name: str,
        units: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        now = utcnow()
        for unit in units:
            pages_covered = list(unit["pages"])
            content = _object(unit.get("content"))
            fingerprint = hashlib.sha256(
                _json(
                    {
                        "pages": pages_covered,
                        "content": content,
                    }
                ).encode("utf-8")
            ).hexdigest()
            result_id = str(uuid.uuid4())
            page_numbers = [
                int(page["pageNumber"]) for page in pages_covered
            ]
            connection.execute(
                insert(analysis_layer_results).values(
                    id=result_id,
                    run_id=run_id,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    unit_index=int(unit["unitIndex"]),
                    chapter_id=unit.get("chapterId"),
                    page_range_snapshot_json=_json(
                        {
                            "start": min(page_numbers),
                            "end": max(page_numbers),
                        }
                    ),
                    content_json=_json(content),
                    input_fingerprint=fingerprint,
                    status="staging",
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(analysis_layer_result_pages),
                [
                    {
                        "layer_result_id": result_id,
                        "ordinal": ordinal,
                        "page_id": page["pageId"],
                        "page_id_snapshot": page["pageId"],
                        "page_number_snapshot": page["pageNumber"],
                    }
                    for ordinal, page in enumerate(
                        pages_covered,
                        start=1,
                    )
                ],
            )
        return {
            "runId": run_id,
            "layerIndex": layer_index,
            "unitCount": len(units),
        }

    def publish_timeline(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        result: Mapping[str, Any],
        activate: bool = True,
    ) -> dict[str, Any]:
        status = "building"
        if activate:
            current = self._snapshot(connection, book_id=frozen.book_id)
            status = _publication_status(frozen, current)
        mode = str(result.get("mode", "simple"))
        if mode not in {"enhanced", "compressed", "simple"}:
            raise InsightConflict("timeline mode is invalid")
        raw_events = result.get("events", [])
        raw_characters = result.get("characters", [])
        if not isinstance(raw_events, list) or not isinstance(raw_characters, list):
            raise InsightConflict("timeline events/characters must be arrays")
        timeline_id = str(uuid.uuid4())
        now = utcnow()
        should_activate = activate and status in {"ready", "degraded"}
        if should_activate:
            connection.execute(
                update(timeline_versions)
                .where(
                    timeline_versions.c.book_id == frozen.book_id,
                    timeline_versions.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
        connection.execute(
            insert(timeline_versions).values(
                id=timeline_id,
                book_id=frozen.book_id,
                run_id=frozen.source_run_id,
                mode=mode,
                status=status,
                content_json=_json(_object(result.get("content"))),
                dependency_fingerprint=frozen.fingerprint,
                is_active=should_activate,
                created_at=now,
                updated_at=now,
            )
        )
        if raw_events:
            connection.execute(
                insert(timeline_events),
                [
                    {
                        "id": str(uuid.uuid4()),
                        "timeline_version_id": timeline_id,
                        "ordinal": index,
                        "payload_json": _json(
                            dict(event)
                            if isinstance(event, Mapping)
                            else {"summary": str(event)}
                        ),
                    }
                    for index, event in enumerate(raw_events, start=1)
                ],
            )
        names: set[str] = set()
        character_rows = []
        for character in raw_characters:
            if not isinstance(character, Mapping):
                continue
            name = str(character.get("name", "")).strip()
            if not name or name in names:
                continue
            names.add(name)
            character_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "timeline_version_id": timeline_id,
                    "name": name,
                    "payload_json": _json(dict(character)),
                }
            )
        if character_rows:
            connection.execute(insert(timeline_characters), character_rows)
        return {
            "timelineVersionId": timeline_id,
            "mode": mode,
            "status": status,
            "eventCount": len(raw_events),
            "characterCount": len(character_rows),
        }

    def next_vector_generation(self, book_id: str) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    select(
                        func.coalesce(
                            func.max(vector_generations.c.generation),
                            0,
                        )
                        + 1
                    ).where(vector_generations.c.book_id == book_id)
                ).scalar_one()
            )

    def checkpoint_vector_generation(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        generation: int,
        page_count: int,
        event_count: int,
    ) -> dict[str, Any]:
        if generation < 1 or page_count < 0 or event_count < 0:
            raise InsightConflict("vector generation checkpoint is invalid")
        row = connection.execute(
            select(vector_generations).where(
                vector_generations.c.book_id == frozen.book_id,
                vector_generations.c.generation == generation,
            )
        ).mappings().one_or_none()
        now = utcnow()
        if row is None:
            generation_id = str(uuid.uuid4())
            connection.execute(
                insert(vector_generations).values(
                    id=generation_id,
                    book_id=frozen.book_id,
                    run_id=frozen.source_run_id,
                    generation=generation,
                    status="building",
                    dependency_fingerprint=frozen.fingerprint,
                    page_count=page_count,
                    event_count=event_count,
                    is_active=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            if (
                row["run_id"] != frozen.source_run_id
                or row["dependency_fingerprint"] != frozen.fingerprint
                or bool(row["is_active"])
                or row["status"] != "building"
                or page_count < int(row["page_count"])
                or event_count < int(row["event_count"])
            ):
                raise InsightConflict("vector generation checkpoint conflicts")
            generation_id = str(row["id"])
            connection.execute(
                update(vector_generations)
                .where(vector_generations.c.id == generation_id)
                .values(
                    page_count=page_count,
                    event_count=event_count,
                    updated_at=now,
                )
            )
        return {
            "vectorGenerationId": generation_id,
            "generation": generation,
            "status": "building",
            "pageCount": page_count,
            "eventCount": event_count,
        }

    def fail_vector_generation(self, *, book_id: str, generation: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                update(vector_generations)
                .where(
                    vector_generations.c.book_id == book_id,
                    vector_generations.c.generation == generation,
                    vector_generations.c.status == "building",
                    vector_generations.c.is_active.is_(False),
                )
                .values(status="failed", updated_at=utcnow())
            )

    def publish_vector_generation(
        self,
        *,
        connection: Connection,
        frozen: AnalysisInputSnapshot,
        generation: int,
        page_count: int,
        event_count: int,
        activate: bool = True,
    ) -> dict[str, Any]:
        status = "building"
        if activate:
            current = self._snapshot(connection, book_id=frozen.book_id)
            status = _publication_status(frozen, current)
        now = utcnow()
        should_activate = activate and status in {"ready", "degraded"}
        if should_activate:
            connection.execute(
                update(vector_generations)
                .where(
                    vector_generations.c.book_id == frozen.book_id,
                    vector_generations.c.is_active.is_(True),
                )
                .values(is_active=False, updated_at=now)
            )
        existing = connection.execute(
            select(vector_generations).where(
                vector_generations.c.book_id == frozen.book_id,
                vector_generations.c.generation == generation,
            )
        ).mappings().one_or_none()
        if existing is None:
            generation_id = str(uuid.uuid4())
            connection.execute(
                insert(vector_generations).values(
                    id=generation_id,
                    book_id=frozen.book_id,
                    run_id=frozen.source_run_id,
                    generation=generation,
                    status=status,
                    dependency_fingerprint=frozen.fingerprint,
                    page_count=page_count,
                    event_count=event_count,
                    is_active=should_activate,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            if (
                existing["run_id"] != frozen.source_run_id
                or existing["dependency_fingerprint"] != frozen.fingerprint
                or bool(existing["is_active"])
                or existing["status"] != "building"
            ):
                raise InsightConflict("vector generation publication conflicts")
            generation_id = str(existing["id"])
            connection.execute(
                update(vector_generations)
                .where(vector_generations.c.id == generation_id)
                .values(
                    status=status,
                    page_count=page_count,
                    event_count=event_count,
                    is_active=should_activate,
                    updated_at=now,
                )
            )
        return {
            "vectorGenerationId": generation_id,
            "generation": generation,
            "status": status,
            "pageCount": page_count,
            "eventCount": event_count,
        }

    def get_artifact(
        self,
        *,
        book_id: str,
        kind: str,
        template: str,
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(analysis_artifacts).where(
                    analysis_artifacts.c.book_id == book_id,
                    analysis_artifacts.c.kind == kind,
                    analysis_artifacts.c.template == template,
                    analysis_artifacts.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
        if row is None:
            return None
        return {
            "artifactId": str(row["id"]),
            "bookId": str(row["book_id"]),
            "runId": row["run_id"],
            "kind": str(row["kind"]),
            "template": str(row["template"]),
            "status": str(row["status"]),
            "revision": int(row["revision"]),
            "dependencyFingerprint": str(row["dependency_fingerprint"]),
            "payload": _load(row["payload_json"], {}),
        }

    def get_timeline(
        self,
        *,
        book_id: str,
        event_after: int = 0,
        event_limit: int = 100,
        character_after: str | None = None,
        character_limit: int = 100,
    ) -> dict[str, Any] | None:
        if event_after < 0:
            raise ValueError("event cursor must be nonnegative")
        if not 1 <= event_limit <= 200:
            raise ValueError("event limit must be between 1 and 200")
        if not 1 <= character_limit <= 200:
            raise ValueError("character limit must be between 1 and 200")
        with self.engine.connect() as connection:
            row = connection.execute(
                select(timeline_versions).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
            if row is None:
                return None
            event_rows = list(
                connection.execute(
                    select(
                        timeline_events.c.id,
                        timeline_events.c.ordinal,
                        timeline_events.c.payload_json,
                    )
                    .where(
                        timeline_events.c.timeline_version_id == row["id"],
                        timeline_events.c.ordinal > event_after,
                    )
                    .order_by(timeline_events.c.ordinal)
                    .limit(event_limit + 1)
                )
            )
            character_statement = select(
                timeline_characters.c.id,
                timeline_characters.c.name,
                timeline_characters.c.payload_json,
            ).where(
                timeline_characters.c.timeline_version_id == row["id"]
            )
            if character_after:
                character_statement = character_statement.where(
                    timeline_characters.c.name > character_after
                )
            character_rows = list(
                connection.execute(
                    character_statement.order_by(
                        timeline_characters.c.name
                    ).limit(character_limit + 1)
                )
            )
            event_count = int(
                connection.execute(
                    select(func.count(timeline_events.c.id)).where(
                        timeline_events.c.timeline_version_id == row["id"]
                    )
                ).scalar_one()
            )
            character_count = int(
                connection.execute(
                    select(func.count(timeline_characters.c.id)).where(
                        timeline_characters.c.timeline_version_id == row["id"]
                    )
                ).scalar_one()
            )
            page_count = int(
                connection.execute(
                    select(func.count(pages.c.id))
                    .join(chapters, chapters.c.id == pages.c.chapter_id)
                    .where(chapters.c.book_id == book_id)
                ).scalar_one()
            )
            has_more_events = len(event_rows) > event_limit
            selected_events = event_rows[:event_limit]
            has_more_characters = len(character_rows) > character_limit
            selected_characters = character_rows[:character_limit]
            content_payload = _object(_load(row["content_json"], {}))
            event_payloads = [
                {
                    "eventId": str(event_id),
                    **_object(_load(value, {})),
                }
                for event_id, _ordinal, value in selected_events
            ]
            character_payloads = [
                {
                    "characterId": str(character_id),
                    **_object(_load(value, {})),
                }
                for character_id, _name, value in selected_characters
            ]
            referenced_page_ids = {
                str(page_id)
                for payload in event_payloads
                for page_id in (
                    payload.get("page_ids", [])
                    if isinstance(payload.get("page_ids"), list)
                    else []
                )
                if page_id
            }
            referenced_page_numbers = _timeline_thumbnail_page_numbers(
                content=content_payload,
                events=event_payloads,
                characters=character_payloads,
            )
            page_numbers_by_id: dict[str, int] = {}
            page_thumbnails: dict[str, str] = {}
            if referenced_page_ids or referenced_page_numbers:
                thumbnail_pointer = page_assets.alias(
                    "timeline_thumbnail_pointer"
                )
                numbered_pages = (
                    select(
                        pages.c.id.label("page_id"),
                        func.row_number()
                        .over(order_by=(chapters.c.ordinal, pages.c.ordinal))
                        .label("page_number"),
                        thumbnail_pointer.c.asset_id.label(
                            "thumbnail_asset_id"
                        ),
                    )
                    .join(chapters, chapters.c.id == pages.c.chapter_id)
                    .outerjoin(
                        thumbnail_pointer,
                        (
                            thumbnail_pointer.c.page_id == pages.c.id
                        )
                        & (
                            thumbnail_pointer.c.role == "thumbnail_source"
                        ),
                    )
                    .where(chapters.c.book_id == book_id)
                    .subquery()
                )
                page_filter = (
                    numbered_pages.c.page_id.in_(referenced_page_ids)
                    if referenced_page_ids
                    else numbered_pages.c.page_number.in_(
                        referenced_page_numbers
                    )
                )
                if referenced_page_ids and referenced_page_numbers:
                    page_filter = page_filter | numbered_pages.c.page_number.in_(
                        referenced_page_numbers
                    )
                page_rows = list(
                    connection.execute(
                        select(
                            numbered_pages.c.page_id,
                            numbered_pages.c.page_number,
                            numbered_pages.c.thumbnail_asset_id,
                        ).where(page_filter)
                    ).mappings()
                )
                page_numbers_by_id = {
                    str(page["page_id"]): int(page["page_number"])
                    for page in page_rows
                }
                page_thumbnails = {
                    str(int(page["page_number"])): (
                        f"/api/v2/assets/{page['thumbnail_asset_id']}"
                    )
                    for page in page_rows
                    if page["thumbnail_asset_id"]
                }
            for payload in event_payloads:
                if not isinstance(payload.get("page_numbers"), list):
                    payload["page_numbers"] = [
                        page_numbers_by_id[str(page_id)]
                        for page_id in (
                            payload.get("page_ids", [])
                            if isinstance(payload.get("page_ids"), list)
                            else []
                        )
                        if str(page_id) in page_numbers_by_id
                    ]
        return {
            "timelineVersionId": str(row["id"]),
            "bookId": str(row["book_id"]),
            "runId": row["run_id"],
            "mode": str(row["mode"]),
            "status": str(row["status"]),
            "content": content_payload,
            "events": event_payloads,
            "characters": character_payloads,
            "eventPage": {
                "totalCount": event_count,
                "nextCursor": (
                    int(selected_events[-1][1])
                    if has_more_events and selected_events
                    else None
                )
            },
            "characterPage": {
                "totalCount": character_count,
                "nextCursor": (
                    str(selected_characters[-1][1])
                    if has_more_characters and selected_characters
                    else None
                )
            },
            "pageCount": page_count,
            "pageThumbnails": page_thumbnails,
            "dependencyFingerprint": str(row["dependency_fingerprint"]),
        }

    def qa_status(
        self,
        *,
        book_id: str,
        mode: str = "exact",
    ) -> dict[str, Any]:
        if mode not in {"exact", "global"}:
            raise ValueError("mode must be exact or global")
        try:
            current = self.snapshot(book_id=book_id)
        except InsightConflict:
            return {
                "available": False,
                "reason": "analysis_missing",
                "repairAction": "analyze",
            }
        if mode == "global":
            with self.engine.connect() as connection:
                rows = list(
                    connection.execute(
                        select(
                            analysis_artifacts.c.kind,
                            analysis_artifacts.c.template,
                            analysis_artifacts.c.status,
                            analysis_artifacts.c.dependency_fingerprint,
                        ).where(
                            analysis_artifacts.c.book_id == book_id,
                            analysis_artifacts.c.is_active.is_(True),
                        )
                    ).mappings()
                )
            artifacts = {
                (str(row["kind"]), str(row["template"])): row
                for row in rows
            }
            required = (
                (
                    ("overview", "story_summary"),
                    "global_summary_missing",
                    "global_summary_stale",
                    "overview_rebuild",
                ),
                (
                    ("compressed_context", "default"),
                    "compressed_context_missing",
                    "compressed_context_stale",
                    "compressed_context_rebuild",
                ),
            )
            for key, missing_reason, stale_reason, repair_action in required:
                row = artifacts.get(key)
                if row is None:
                    return {
                        "available": False,
                        "reason": missing_reason,
                        "repairAction": repair_action,
                    }
                if (
                    str(row["status"]) not in {"ready", "degraded"}
                    or str(row["dependency_fingerprint"]) != current.fingerprint
                ):
                    return {
                        "available": False,
                        "reason": stale_reason,
                        "repairAction": repair_action,
                    }
            return {
                "available": True,
                "reason": None,
            }
        with self.engine.connect() as connection:
            vector = connection.execute(
                select(vector_generations).where(
                    vector_generations.c.book_id == book_id,
                    vector_generations.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
        if vector is None:
            return {
                "available": False,
                "reason": "vector_missing",
                "repairAction": "vector_rebuild",
            }
        if (
            str(vector["dependency_fingerprint"]) != current.fingerprint
            or str(vector["status"]) == "stale"
        ):
            return {
                "available": False,
                "reason": "vector_stale",
                "repairAction": "vector_rebuild",
            }
        return {
            "available": str(vector["status"]) in {"ready", "degraded"},
            "reason": None,
            "generation": int(vector["generation"]),
            "coverage": {
                "pages": int(vector["page_count"]),
                "events": int(vector["event_count"]),
            },
        }


class InsightDerivedCommandService:
    def __init__(self, engine: Engine) -> None:
        self.jobs = JobQueueRepository(engine)
        self.settings = SettingsResolver(engine)
        self.repository = InsightDerivedRepository(engine)

    def create_job(
        self,
        *,
        book_id: str,
        kind: str,
        template: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        if kind not in DERIVED_KINDS:
            raise ValueError("unsupported Insight derived kind")
        if not template or len(template) > 64:
            raise ValueError("template must contain 1-64 characters")
        frozen = self.repository.snapshot(book_id=book_id)
        config = self.settings.resolve_insight(
            book_id=book_id,
            command={"scope": "full", "force": False},
        )
        config.update(
            {
                "bookId": book_id,
                "derivedKind": kind,
                "template": template,
                "sourceRunId": frozen.source_run_id,
                "sourceRunStatus": frozen.source_run_status,
                "analysisInputs": [
                    {
                        "resultId": page["resultId"],
                        "pageId": page["pageId"],
                        "pageNumber": page["pageNumber"],
                        "currentSourceChecksum": page[
                            "currentSourceChecksum"
                        ],
                    }
                    for page in frozen.pages
                ],
                "analysisInputFingerprint": frozen.fingerprint,
            }
        )
        step = {
            "overview": "insight_build_overview",
            "compressed_context": "insight_build_compressed_context",
            "timeline": "insight_build_timeline",
            "vector": "insight_build_vectors",
        }[kind]
        job_kind = "vector_rebuild" if kind == "vector" else "derived_rebuild"
        return self.jobs.create_batch(
            kind=job_kind,
            display_name=f"Insight · {kind}",
            specs=(
                JobSpec(
                    kind=job_kind,
                    book_id=book_id,
                    analysis_run_id=frozen.source_run_id,
                    config=config,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=(step,),
                        ),
                    ),
                    target_display={
                        "bookId": book_id,
                        "kind": kind,
                        "template": template,
                    },
                ),
            ),
            idempotency_scope=f"insight-derived:{book_id}:{kind}:{template}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "bookId": book_id,
                "kind": kind,
                "template": template,
                "fingerprint": frozen.fingerprint,
            },
        )


class InsightDerivedWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        algorithms: DerivedAlgorithms | None = None,
        vector_store: InsightVectorStore | None = None,
    ) -> None:
        self.engine = engine
        self.jobs = jobs
        self.repository = InsightDerivedRepository(engine)
        self.credentials = SettingsRepository(engine)
        self.algorithms = algorithms or ProviderDerivedAlgorithms()
        self.vector_store = vector_store or InsightVectorStore(data_root)

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raw_config = _object(step.get("config"))
        kind = str(step["stepKind"])
        if kind in {"insight_build_vectors", "insight_stage_vectors"}:
            credential_sections = ("embedding",)
        elif (
            kind.startswith("insight_build_layer_")
            or kind
            in {
                "insight_build_overview",
                "insight_stage_overview_no_spoiler",
                "insight_stage_overview_story_summary",
                "insight_build_compressed_context",
                "insight_stage_compressed_context",
                "insight_build_timeline",
                "insight_stage_timeline",
            }
        ):
            credential_sections = _chat_credential_sections(raw_config)
        else:
            raise JobConflict(f"unsupported derived step: {kind}")
        config = self._with_credentials(
            raw_config,
            section_names=credential_sections,
        )
        book_id = str(config.get("bookId", ""))
        run_id = str(config.get("runId", ""))
        full_stage = (
            str(config.get("scope", "")) == "full"
            and bool(run_id)
            and (
                kind.startswith("insight_build_layer_")
                or kind.startswith("insight_stage_")
            )
        )
        if full_stage:
            frozen = self.repository.snapshot_for_run(run_id=run_id)
        else:
            frozen_inputs = config.get("analysisInputs")
            if (
                not book_id
                or not isinstance(frozen_inputs, list)
                or not all(
                    isinstance(value, Mapping) for value in frozen_inputs
                )
            ):
                raise JobConflict(
                    "derived job has an invalid frozen input snapshot"
                )
            frozen = self.repository.snapshot(
                book_id=book_id,
                frozen_inputs=frozen_inputs,
            )
            expected = str(config.get("analysisInputFingerprint", ""))
            if frozen.fingerprint != expected:
                raise JobConflict(
                    "frozen Insight input fingerprint is invalid"
                )
            frozen = AnalysisInputSnapshot(
                book_id=frozen.book_id,
                source_run_id=(
                    str(config["sourceRunId"])
                    if config.get("sourceRunId")
                    else None
                ),
                source_run_status=(
                    str(config["sourceRunStatus"])
                    if config.get("sourceRunStatus")
                    else None
                ),
                result_ids=frozen.result_ids,
                pages=frozen.pages,
                fingerprint=frozen.fingerprint,
            )
        try:
            if kind.startswith("insight_build_layer_"):
                layer_index = int(kind.rsplit("_", 1)[1])
                layer_units = self.repository.layer_units(
                    run_id=run_id,
                    layer_index=layer_index,
                    config=config,
                )
                completed_units = []
                for unit in layer_units:
                    content = self.algorithms.build_layer(
                        unit["inputs"],
                        layer=unit["layer"],
                        config=config,
                    )
                    completed_units.append(
                        {
                            **unit,
                            "content": _object(content),
                        }
                    )
                checkpoint: dict[str, Any] = {}

                def publish(connection: Connection) -> None:
                    layer = _object(
                        _object(config.get("analysis"))
                        .get("layers", [])[layer_index]
                    )
                    checkpoint.update(
                        self.repository.publish_layer(
                            connection,
                            run_id=run_id,
                            layer_index=layer_index,
                            layer_name=str(layer.get("name", "")),
                            units=completed_units,
                        )
                    )
            elif kind in {
                "insight_build_overview",
                "insight_stage_overview_no_spoiler",
                "insight_stage_overview_story_summary",
            }:
                template = {
                    "insight_stage_overview_no_spoiler": "no_spoiler",
                    "insight_stage_overview_story_summary": "story_summary",
                }.get(kind, str(config.get("template", "default")))
                summary_inputs = self.repository.summary_inputs(frozen)
                payload = self.algorithms.build_overview(
                    summary_inputs,
                    template=template,
                    config=config,
                )
                checkpoint = {}

                def publish(connection: Connection) -> None:
                    checkpoint.update(
                        self.repository.publish_artifact(
                            connection=connection,
                            frozen=frozen,
                            kind="overview",
                            template=template,
                            payload=payload,
                            activate=not full_stage,
                        )
                    )
            elif kind in {
                "insight_build_compressed_context",
                "insight_stage_compressed_context",
            }:
                summary_inputs = self.repository.summary_inputs(frozen)
                payload = self.algorithms.build_compressed_context(
                    summary_inputs,
                    config=config,
                )
                checkpoint = {}

                def publish(connection: Connection) -> None:
                    checkpoint.update(
                        self.repository.publish_artifact(
                            connection=connection,
                            frozen=frozen,
                            kind="compressed_context",
                            template="default",
                            payload=payload,
                            activate=not full_stage,
                        )
                    )
            elif kind in {
                "insight_build_timeline",
                "insight_stage_timeline",
            }:
                timeline_inputs = list(self.repository.summary_inputs(frozen))
                compressed_input = self.repository.compressed_context_input(frozen)
                if compressed_input is not None:
                    timeline_inputs.append(compressed_input)
                timeline = self.algorithms.build_timeline(
                    timeline_inputs,
                    config=config,
                )
                checkpoint = {}

                def publish(connection: Connection) -> None:
                    checkpoint.update(
                        self.repository.publish_timeline(
                            connection=connection,
                            frozen=frozen,
                            result=timeline,
                            activate=not full_stage,
                        )
                    )
            elif kind in {
                "insight_build_vectors",
                "insight_stage_vectors",
            }:
                vector_build = self._build_vectors(
                    fence=fence,
                    step=step,
                    frozen=frozen,
                    config=config,
                )
                checkpoint = {
                    key: value
                    for key, value in vector_build.items()
                    if not key.startswith("__")
                }
                if vector_build.get("__control_drained__"):
                    return {
                        **checkpoint,
                        "__already_published__": True,
                        "__control_drained__": True,
                    }

                def publish(connection: Connection) -> None:
                    job_status = connection.execute(
                        select(jobs.c.status).where(jobs.c.id == fence.job_id)
                    ).scalar_one()
                    if str(job_status) == "running":
                        checkpoint.update(
                            self.repository.publish_vector_generation(
                                connection=connection,
                                frozen=frozen,
                                generation=int(vector_build["generation"]),
                                page_count=int(vector_build["pageCount"]),
                                event_count=int(vector_build["eventCount"]),
                                activate=not full_stage,
                            )
                        )
                    else:
                        self.repository.checkpoint_vector_generation(
                            connection=connection,
                            frozen=frozen,
                            generation=int(vector_build["generation"]),
                            page_count=int(vector_build["pageCount"]),
                            event_count=int(vector_build["eventCount"]),
                        )
                completed = self.jobs.complete_step(
                    fence,
                    step_id=str(step["stepId"]),
                    checkpoint=checkpoint,
                    publisher=publish,
                    defer_on_control=True,
                )
                return {
                    **checkpoint,
                    "__already_published__": True,
                    **({"__control_drained__": True} if not completed else {}),
                }
            else:
                raise JobConflict(f"unsupported derived step: {kind}")
            self.jobs.complete_step(
                fence,
                step_id=str(step["stepId"]),
                checkpoint=checkpoint,
                publisher=publish,
            )
            return {**checkpoint, "__already_published__": True}
        except AttemptFenced:
            raise

    def _build_vectors(
        self,
        *,
        fence: AttemptFence,
        step: Mapping[str, Any],
        frozen: AnalysisInputSnapshot,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        page_records: list[dict[str, Any]] = []
        event_records: list[dict[str, Any]] = []
        for page in frozen.pages:
            analysis = _object(page.get("analysis"))
            summary = str(analysis.get("page_summary", "")).strip()
            if summary:
                page_records.append(
                    {
                        "id": f"page-{page['pageId']}",
                        "document": summary,
                        "metadata": {
                            "book_id": frozen.book_id,
                            "page_id": str(page["pageId"]),
                            "page_number": int(page["pageNumber"]),
                            "type": "page",
                        },
                    }
                )
        event_records.extend(self._layer_zero_event_records(frozen))
        previous = _object(step.get("checkpoint"))
        generation = int(
            previous.get("generation")
            or self.repository.next_vector_generation(frozen.book_id)
        )
        page_count = int(previous.get("pageCount", 0))
        event_count = int(previous.get("eventCount", 0))
        if (
            generation < 1
            or page_count < 0
            or event_count < 0
            or page_count > len(page_records)
            or event_count > len(event_records)
        ):
            raise InsightConflict("vector checkpoint is invalid")
        checkpoint: dict[str, Any] = {
            "generation": generation,
            "pageCount": page_count,
            "eventCount": event_count,
            "pageTotal": len(page_records),
            "eventTotal": len(event_records),
        }

        def checkpoint_batch(kind: str, count: int) -> bool:
            if kind == "pages":
                checkpoint["pageCount"] = count
            else:
                checkpoint["eventCount"] = count
            total = len(page_records) + len(event_records)
            completed = int(checkpoint["pageCount"]) + int(
                checkpoint["eventCount"]
            )
            checkpoint["coverage"] = 1.0 if total == 0 else completed / total

            def publish_partial(connection: Connection) -> None:
                self.repository.checkpoint_vector_generation(
                    connection=connection,
                    frozen=frozen,
                    generation=generation,
                    page_count=int(checkpoint["pageCount"]),
                    event_count=int(checkpoint["eventCount"]),
                )

            status = self.jobs.checkpoint_step(
                fence,
                step_id=str(step["stepId"]),
                checkpoint=checkpoint,
                publisher=publish_partial,
            )
            return status == "running"

        try:
            result = self.vector_store.publish_batches(
                book_id=frozen.book_id,
                generation=generation,
                page_batches=self._embedding_batches(
                    page_records[page_count:],
                    config=config,
                ),
                event_batches=self._embedding_batches(
                    event_records[event_count:],
                    config=config,
                ),
                resume=bool(previous.get("generation")),
                initial_page_count=page_count,
                initial_event_count=event_count,
                expected_page_count=len(page_records),
                expected_event_count=len(event_records),
                on_batch=checkpoint_batch,
            )
        except AttemptFenced:
            raise
        except Exception:
            self.repository.fail_vector_generation(
                book_id=frozen.book_id,
                generation=generation,
            )
            raise
        if isinstance(result, Mapping):
            checkpoint["pageCount"] = int(result.get("pageCount", page_count))
            checkpoint["eventCount"] = int(result.get("eventCount", event_count))
            if not bool(result.get("completed")):
                return {**checkpoint, "__control_drained__": True}
        else:
            # Lightweight test stores may consume the generators without
            # returning counters; a successful return still means full coverage.
            checkpoint["pageCount"] = len(page_records)
            checkpoint["eventCount"] = len(event_records)
        checkpoint["coverage"] = 1.0
        return {
            **checkpoint,
        }

    def _embedding_batches(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
        batch_size: int = 64,
    ) -> Iterable[
        tuple[Sequence[Mapping[str, Any]], Sequence[Sequence[float]]]
    ]:
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            documents = [str(row["document"]) for row in batch]
            embeddings = list(
                self.algorithms.embed_documents(documents, config=config)
            )
            if len(embeddings) != len(batch):
                raise InsightConflict("embedding result count mismatch")
            yield batch, embeddings

    def _layer_zero_event_records(
        self,
        frozen: AnalysisInputSnapshot,
    ) -> list[dict[str, Any]]:
        if not frozen.source_run_id:
            return []
        with self.engine.connect() as connection:
            layers = list(
                connection.execute(
                    select(
                        analysis_layer_results.c.id,
                        analysis_layer_results.c.content_json,
                    )
                    .where(
                        analysis_layer_results.c.run_id
                        == frozen.source_run_id,
                        analysis_layer_results.c.layer_index == 0,
                        analysis_layer_results.c.status.in_(
                            ("staging", "published")
                        ),
                    )
                    .order_by(analysis_layer_results.c.unit_index)
                ).mappings()
            )
            page_rows = list(
                connection.execute(
                    select(
                        analysis_layer_result_pages.c.layer_result_id,
                        analysis_layer_result_pages.c.page_id_snapshot,
                        analysis_layer_result_pages.c.page_number_snapshot,
                    )
                    .where(
                        analysis_layer_result_pages.c.layer_result_id.in_(
                            tuple(str(row["id"]) for row in layers)
                        )
                    )
                    .order_by(
                        analysis_layer_result_pages.c.layer_result_id,
                        analysis_layer_result_pages.c.ordinal,
                    )
                )
            ) if layers else []
        pages_by_layer: dict[str, list[tuple[str, int]]] = {}
        for layer_result_id, page_id, page_number in page_rows:
            pages_by_layer.setdefault(
                str(layer_result_id),
                [],
            ).append((str(page_id), int(page_number)))
        records: list[dict[str, Any]] = []
        for layer in layers:
            layer_id = str(layer["id"])
            content = _object(_load(layer["content_json"], {}))
            page_refs = pages_by_layer.get(layer_id, [])
            for index, event in enumerate(
                content.get("key_events", []),
                start=1,
            ):
                if not isinstance(event, Mapping):
                    continue
                text = str(event.get("summary", "")).strip()
                if not text:
                    continue
                records.append(
                    {
                        "id": f"event-{layer_id}-{index}",
                        "document": text,
                        "metadata": {
                            "book_id": frozen.book_id,
                            "page_id": (
                                page_refs[0][0] if page_refs else ""
                            ),
                            "page_number": (
                                page_refs[0][1] if page_refs else 0
                            ),
                            "page_ids_json": _json(
                                [value[0] for value in page_refs]
                            ),
                            "page_numbers_json": _json(
                                [value[1] for value in page_refs]
                            ),
                            "importance": str(
                                event.get("importance", "normal")
                            ),
                            "type": "event",
                        },
                    }
                )
        return records

    def _with_credentials(
        self,
        config: Mapping[str, Any],
        *,
        section_names: Sequence[str],
    ) -> dict[str, Any]:
        try:
            return self.credentials.resolve_credential_sections(
                config,
                section_names,
            )
        except LookupError as exc:
            raise JobConflict(
                "frozen Insight credential version no longer exists"
            ) from exc


def _chat_credential_sections(config: Mapping[str, Any]) -> tuple[str, ...]:
    return ("chat",) if _object(config.get("chat")).get("provider") else ("vlm",)


def _publication_status(
    frozen: AnalysisInputSnapshot,
    current: AnalysisInputSnapshot,
) -> str:
    if frozen.fingerprint != current.fingerprint:
        return "stale"
    if frozen.source_run_status == "completed_with_errors":
        return "degraded"
    return "ready"


def _analysis_input_fingerprint(
    pages_payload: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the same immutable identity fields before and after publication."""

    canonical = [
        {
            "resultId": str(page["resultId"]),
            "pageId": str(page["pageId"]),
            "pageNumber": int(page["pageNumber"]),
            "sourceChecksum": str(page["sourceChecksum"]),
            "currentSourceChecksum": str(page["currentSourceChecksum"]),
        }
        for page in pages_payload
    ]
    return hashlib.sha256(_json(canonical).encode("utf-8")).hexdigest()


def _layer_prompt_type(
    *,
    layer_index: int,
    layer_count: int,
    align_to_chapter: bool,
) -> str:
    if layer_index == 0:
        return "batch_analysis"
    if layer_index == layer_count - 1:
        return "book_overview"
    if align_to_chapter:
        return "chapter_summary"
    return "segment_summary"


def _page_context(pages: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(
        (
            _page_context_label(page)
            + "\n"
            + _json(page.get("analysis", {}))
        )
        for page in pages
    )


def _page_context_label(page: Mapping[str, Any]) -> str:
    page_ids = [str(value) for value in page.get("pageIds", ()) if value]
    page_numbers = [int(value) for value in page.get("pageNumbers", ())]
    if not page_ids and page.get("pageId"):
        page_ids = [str(page["pageId"])]
    if not page_numbers and page.get("pageNumber"):
        page_numbers = [int(page["pageNumber"])]
    if not page_numbers:
        return "全书汇总"
    page_range = (
        str(page_numbers[0])
        if len(page_numbers) == 1
        else f"{page_numbers[0]}-{page_numbers[-1]}"
    )
    return f"第 {page_range} 页（page_ids={_json(page_ids)}）"
