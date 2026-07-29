"""Versioned Insight overview, timeline, compressed-context, and vector jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol
import uuid

from sqlalchemy import Engine, func, insert, select, update
from sqlalchemy.engine import Connection

from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
    utcnow,
)
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobConflict,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.settings.resolver import SettingsResolver
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
    credential_versions,
    pages,
    page_assets,
    timeline_characters,
    timeline_events,
    timeline_versions,
    vector_generations,
)


DERIVED_KINDS = frozenset(
    {"overview", "compressed_context", "timeline", "vector"}
)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


class LegacyDerivedAlgorithms:
    """Worker-only adapters around the shared chat and embedding transports."""

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
        prompt = (
            "根据以下漫画分析生成增强时间线。输出 JSON："
            '{"content":{},"events":[{"summary":"...","page_ids":["..."]}],'
            '"characters":[{"name":"...","first_page":1,"key_moments":[]}]}。'
            "不要把推断写成事实。\n\n"
            + _page_context(pages)
        )
        try:
            result = self._chat_json(
                prompt,
                config=config,
                prompt_type="book_overview",
            )
            if not isinstance(result, Mapping):
                raise ValueError("timeline response must be an object")
            events = result.get("events")
            characters = result.get("characters")
            if not isinstance(events, list) or not isinstance(characters, list):
                raise ValueError("timeline response is missing events/characters")
            return {
                "mode": "enhanced",
                "content": _object(result.get("content")),
                "events": events,
                "characters": characters,
            }
        except Exception as exc:
            events = []
            for page in pages:
                payload = _object(page.get("analysis"))
                for event in payload.get("key_events", []):
                    if isinstance(event, Mapping):
                        events.append(
                            {
                                "summary": str(event.get("summary", "")),
                                "importance": str(
                                    event.get("importance", "normal")
                                ),
                                "page_ids": [str(page["pageId"])],
                                "page_numbers": [int(page["pageNumber"])],
                            }
                        )
            return {
                "mode": "simple",
                "content": {
                    "degradedReason": str(exc),
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
            "api_key": section.get("api_key", section.get("apiKey", "")),
            "model": section.get("model_name", section.get("modelName", "")),
            "base_url": section.get(
                "custom_base_url",
                section.get("base_url"),
            ),
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
            "api_key": section.get("api_key", section.get("apiKey", "")),
            "model": section.get("model_name", section.get("modelName", "")),
            "base_url": section.get(
                "custom_base_url",
                section.get("base_url"),
            ),
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
        if len(page_records) != len(page_embeddings):
            raise InsightConflict("page embedding result count mismatch")
        if len(event_records) != len(event_embeddings):
            raise InsightConflict("event embedding result count mismatch")
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
        try:
            if page_records:
                pages_collection.add(
                    ids=[str(row["id"]) for row in page_records],
                    embeddings=[list(value) for value in page_embeddings],
                    documents=[str(row["document"]) for row in page_records],
                    metadatas=[dict(row["metadata"]) for row in page_records],
                )
            if event_records:
                events_collection.add(
                    ids=[str(row["id"]) for row in event_records],
                    embeddings=[list(value) for value in event_embeddings],
                    documents=[str(row["document"]) for row in event_records],
                    metadatas=[dict(row["metadata"]) for row in event_records],
                )
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
                source_units = []
                for row in rows:
                    covered = list(
                        connection.execute(
                            select(analysis_layer_result_pages)
                            .where(
                                analysis_layer_result_pages.c.layer_result_id
                                == row["id"]
                            )
                            .order_by(analysis_layer_result_pages.c.ordinal)
                        ).mappings()
                    )
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
        generation_id = str(uuid.uuid4())
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
        has_more_events = len(event_rows) > event_limit
        selected_events = event_rows[:event_limit]
        has_more_characters = len(character_rows) > character_limit
        selected_characters = character_rows[:character_limit]
        return {
            "timelineVersionId": str(row["id"]),
            "bookId": str(row["book_id"]),
            "runId": row["run_id"],
            "mode": str(row["mode"]),
            "status": str(row["status"]),
            "content": _load(row["content_json"], {}),
            "events": [
                {
                    "eventId": str(event_id),
                    **_object(_load(value, {})),
                }
                for event_id, _ordinal, value in selected_events
            ],
            "characters": [
                {
                    "characterId": str(character_id),
                    **_object(_load(value, {})),
                }
                for character_id, _name, value in selected_characters
            ],
            "eventPage": {
                "nextCursor": (
                    int(selected_events[-1][1])
                    if has_more_events and selected_events
                    else None
                )
            },
            "characterPage": {
                "nextCursor": (
                    str(selected_characters[-1][1])
                    if has_more_characters and selected_characters
                    else None
                )
            },
            "dependencyFingerprint": str(row["dependency_fingerprint"]),
        }

    def qa_status(self, *, book_id: str) -> dict[str, Any]:
        try:
            current = self.snapshot(book_id=book_id)
        except InsightConflict:
            return {
                "available": False,
                "reason": "analysis_missing",
                "repairAction": "analyze",
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
        self.engine = engine
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
        self.algorithms = algorithms or LegacyDerivedAlgorithms()
        self.vector_store = vector_store or InsightVectorStore(data_root)

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raw_config = _object(step.get("config"))
        config = self._with_credentials(raw_config)
        book_id = str(config.get("bookId", ""))
        run_id = str(config.get("runId", ""))
        kind = str(step["stepKind"])
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
                    algorithm = getattr(
                        self.algorithms,
                        "build_layer",
                        None,
                    )
                    if algorithm is None:
                        content = {
                            "summary": "\n".join(
                                str(
                                    value.get(
                                        "page_summary",
                                        value.get("summary", ""),
                                    )
                                )
                                for value in unit["inputs"]
                            ).strip()
                        }
                    else:
                        content = algorithm(
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
                payload = self.algorithms.build_overview(
                    frozen.pages,
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
                payload = self.algorithms.build_compressed_context(
                    frozen.pages,
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
                timeline = self.algorithms.build_timeline(
                    frozen.pages,
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
                    frozen=frozen,
                    config=config,
                )
                checkpoint = {}

                def publish(connection: Connection) -> None:
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
        documents = [
            str(row["document"])
            for row in (*page_records, *event_records)
        ]
        embeddings = list(
            self.algorithms.embed_documents(documents, config=config)
        )
        if len(embeddings) != len(documents):
            raise InsightConflict("embedding result count mismatch")
        split = len(page_records)
        generation = self.repository.next_vector_generation(frozen.book_id)
        self.vector_store.publish(
            book_id=frozen.book_id,
            generation=generation,
            page_records=page_records,
            page_embeddings=embeddings[:split],
            event_records=event_records,
            event_embeddings=embeddings[split:],
        )
        return {
            "generation": generation,
            "pageCount": len(page_records),
            "eventCount": len(event_records),
        }

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
                text = str(
                    event.get("summary", event.get("content", ""))
                ).strip()
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
    ) -> dict[str, Any]:
        result = json.loads(json.dumps(config))
        for key in ("vlm", "chat", "embedding", "reranker", "imageGen"):
            section = _object(result.get(key))
            version_id = section.pop("credentialVersionId", None)
            if version_id:
                with self.engine.connect() as connection:
                    value = connection.execute(
                        select(credential_versions.c.secret_json).where(
                            credential_versions.c.id == version_id
                        )
                    ).scalar_one_or_none()
                if value is None:
                    raise JobConflict(
                        f"frozen {key} credential version no longer exists"
                    )
                secret = json.loads(value)
                if isinstance(secret, Mapping):
                    section.update(secret)
            result[key] = section
        return result


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
            f"第 {page['pageNumber']} 页（page_id={page['pageId']}）\n"
            + _json(page.get("analysis", {}))
        )
        for page in pages
    )
