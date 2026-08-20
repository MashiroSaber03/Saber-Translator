from __future__ import annotations

import base64
from datetime import timedelta
from io import BytesIO
import gc
import json
from pathlib import Path
import sys
import threading
from typing import Any, Mapping
import uuid
import zipfile

from PIL import Image
import pytest
from flask import Flask
from sqlalchemy import delete, event, insert, select, update

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.insight.commands import InsightAnalysisCommandService
from src.backend_v2.insight.continuation import (
    ContinuationCommandService,
    DefaultContinuationAlgorithms,
    ContinuationRepository,
    ContinuationWorkerService,
)
from src.backend_v2.insight.derived import (
    InsightDerivedCommandService,
    InsightDerivedRepository,
    InsightDerivedWorkerService,
    InsightVectorStore,
    ProviderDerivedAlgorithms,
)
from src.backend_v2.insight.exports import (
    InsightExportCommandService,
    InsightExportWorkerService,
)
from src.backend_v2.insight.page_schema import (
    InvalidPageAnalysis,
    normalize_page_analysis,
)
from src.backend_v2.insight.qa import (
    DefaultQAApiAlgorithms,
    DefaultQARetrievalAlgorithms,
    InsightQACommandService,
    InsightQAWorkerService,
    QAConflict,
    QAFenced,
    TransientRequestRepository,
    citations_for,
    validate_retrieval_candidates,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
    InsightRepository,
)
from src.backend_v2.insight.routes import create_insight_blueprint
from src.backend_v2.insight.worker import InsightAnalysisWorkerService
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import DEFAULT_TEXT_STYLE
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_layer_results,
    analysis_page_results,
    analysis_run_targets,
    analysis_runs,
    app_settings,
    assets,
    continuation_character_forms,
    continuation_characters,
    continuation_form_image_versions,
    continuation_image_versions,
    continuation_pages,
    continuation_projects,
    job_asset_inputs,
    jobs,
    metadata,
    page_assets,
    provider_settings,
    timeline_versions,
    transient_requests,
    vector_generations,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.timestamps import utcnow


@pytest.fixture()
def isolated_chromadb_modules():
    """Keep the real Chroma probe out of later API import-boundary tests."""

    modules_before = set(sys.modules)
    try:
        yield
    finally:
        gc.collect()
        for module_name in set(sys.modules) - modules_before:
            if module_name == "chromadb" or module_name.startswith("chromadb."):
                sys.modules.pop(module_name, None)


class FakeInsightAlgorithms:
    def __init__(self, *, fail_page: int | None = None) -> None:
        self.fail_page = fail_page
        self.calls: list[int] = []

    def analyze_page(
        self,
        _image_bytes: bytes,
        *,
        page_number: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append(page_number)
        assert config["runId"]
        if page_number == self.fail_page:
            raise RuntimeError("injected page failure")
        return {
            "pages": [
                {
                    "page_number": page_number,
                    "page_summary": f"第 {page_number} 页摘要",
                    "key_events": [
                        {
                            "summary": "关键事件",
                            "importance": "high",
                            "event_type": "turn",
                        }
                    ],
                    "continuity_notes": "与前页连续",
                    "warnings": [],
                }
            ]
        }


class FakeDerivedAlgorithms:
    def build_layer(self, inputs, *, layer, config):
        return {
            "summary": f"{layer['name']}:{len(inputs)}",
            "key_events": [
                dict(event)
                for value in inputs
                for event in value.get("key_events", [])
                if isinstance(event, Mapping)
            ],
        }

    def build_overview(self, pages, *, template, config):
        return {"title": template, "content": f"{len(pages)} pages"}

    def build_compressed_context(self, pages, *, config):
        return {"content": f"compressed:{len(pages)}"}

    def build_timeline(self, pages, *, config):
        return {
            "mode": "enhanced",
            "content": {
                "arc": "main",
                "story_summary": "main story",
                "requested_mode": "enhanced",
                "actual_mode": "enhanced",
                "fallback_reason": None,
                "degraded": False,
            },
            "events": [{"summary": "event", "page_ids": [pages[0]["pageId"]]}],
            "characters": [
                {
                    "name": "Saber",
                    "description": "main character",
                    "first_page": 1,
                    "key_moments": [],
                }
            ],
        }

    def embed_documents(self, documents, *, config):
        return [[float(index + 1), 0.5] for index, _ in enumerate(documents)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.publications: list[dict[str, Any]] = []

    def publish_batches(self, **kwargs) -> dict[str, object]:
        page_batches = list(kwargs.pop("page_batches"))
        event_batches = list(kwargs.pop("event_batches"))
        self.publications.append(
            {
                **kwargs,
                "page_records": [row for records, _ in page_batches for row in records],
                "event_records": [row for records, _ in event_batches for row in records],
            }
        )
        return {
            "pageCount": int(kwargs["expected_page_count"]),
            "eventCount": int(kwargs["expected_event_count"]),
            "completed": True,
        }


class CheckpointingFakeVectorStore:
    def __init__(
        self,
        *,
        queue: JobQueueRepository,
        job_id: str,
        control: str | None = None,
    ) -> None:
        self.queue = queue
        self.job_id = job_id
        self.control = control
        self.calls: list[dict[str, object]] = []

    def publish_batches(self, **kwargs) -> dict[str, object]:
        callback = kwargs["on_batch"]
        page_count = int(kwargs["initial_page_count"])
        event_count = int(kwargs["initial_event_count"])
        self.calls.append(
            {
                "resume": bool(kwargs["resume"]),
                "initialPageCount": page_count,
                "initialEventCount": event_count,
            }
        )
        control_sent = False
        for kind, batches in (
            ("pages", kwargs["page_batches"]),
            ("events", kwargs["event_batches"]),
        ):
            for records, _embeddings in batches:
                if kind == "pages":
                    page_count += len(records)
                    count = page_count
                else:
                    event_count += len(records)
                    count = event_count
                if self.control and not control_sent:
                    control_sent = True
                    if self.control == "pause":
                        self.queue.request_pause(self.job_id)
                    else:
                        self.queue.request_cancel(self.job_id)
                if not callback(kind, count):
                    return {
                        "completed": False,
                        "pageCount": page_count,
                        "eventCount": event_count,
                    }
        return {
            "completed": True,
            "pageCount": page_count,
            "eventCount": event_count,
        }


class FakeContinuationAlgorithms:
    def __init__(self) -> None:
        self.script_contexts: list[Mapping[str, Any]] = []
        self.image_reference_paths: list[tuple[Path, ...]] = []

    def generate_script(self, *, context, config):
        self.script_contexts.append(context)
        return "第1页：新的开始\n第2页：继续前进"

    def generate_page(self, *, ordinal, script, previous, config):
        return {
            "storyText": f"第 {ordinal} 页剧情",
            "continuityText": (
                str(previous.get("storyText", "")) if previous else "原作结尾"
            ),
            "dialogueText": "对白",
            "characters": ["Saber"],
            "finalPrompt": f"page {ordinal}",
            "status": "ready",
        }

    def generate_image(self, *, prompt, reference_paths, config):
        self.image_reference_paths.append(tuple(reference_paths))
        payload = BytesIO()
        with Image.new("RGB", (48, 64), (120, 80, 160)) as image:
            image.save(payload, format="PNG")
        return payload.getvalue()


class FakeQARetrievalAlgorithms:
    def embed_queries(self, queries, *, config):
        return [[float(index + 1), 0.25] for index, _ in enumerate(queries)]


def _insight_openai_options(temperature: float | None) -> dict[str, object]:
    return {
        "request": {
            "force_json_output": False,
            "temperature": temperature,
            "extra_body": {},
        },
        "execution": {
            "use_stream": False,
            "rpm_limit": 0,
            "transport_retries": 1,
            "business_retries": 0,
        },
    }


class FakeQAApiAlgorithms:
    def __init__(self) -> None:
        self.rerank_calls = 0

    def rerank(self, *, question, candidates, top_k, config):
        self.rerank_calls += 1
        return list(candidates)[:top_k]

    def stream_answer(
        self,
        *,
        question,
        candidates,
        config,
        cancelled,
    ):
        yield "答案"
        yield "内容"


def test_qa_reasoning_queries_are_embedded_in_bounded_requests(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_embed(_transport, request):
        calls.append(list(request.inputs))
        return [[float(len(calls)), 0.25]]

    monkeypatch.setattr(
        "src.shared.ai_transport.AsyncOpenAICompatibleTransport.embed",
        fake_embed,
    )
    result = DefaultQARetrievalAlgorithms().embed_queries(
        ["原问题", "推理问题一", "推理问题二"],
        config={
                "embedding": {
                    "provider": "siliconflow",
                    "model_name": "embedding-model",
                    "api_key": "test-key",
                    "custom_base_url": "",
                    "rpm_limit": 0,
                    "transport_retries": 1,
                    "business_retries": 0,
                    "timeout_seconds": 0,
                }
        },
    )

    assert calls == [["原问题"], ["推理问题一"], ["推理问题二"]]
    assert result == [[1.0, 0.25], [2.0, 0.25], [3.0, 0.25]]


@pytest.fixture()
def insight_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    with engine.begin() as connection:
        insight_row = connection.execute(
            select(app_settings.c.payload_json).where(
                app_settings.c.domain == "insight"
            )
        ).scalar_one()
        insight_payload = json.loads(insight_row)
        insight_payload["vlm"] = {"provider": "ollama"}
        insight_payload["chat"] = {
            "provider": "ollama",
            "useSameAsVlm": False,
        }
        insight_payload["embedding"] = {"provider": "ollama"}
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "insight")
            .values(payload_json=json.dumps(insight_payload))
        )
        connection.execute(
            insert(provider_settings),
            [
                {
                    "domain": "insight_vlm",
                    "provider": "ollama",
                    "payload_json": json.dumps(
                        {
                            "modelName": "fake-vlm",
                            "customBaseUrl": "",
                            "openaiOptions": _insight_openai_options(0.3),
                            "imageMaxSize": 0,
                        }
                    ),
                },
                {
                    "domain": "insight_chat",
                    "provider": "ollama",
                    "payload_json": json.dumps(
                        {
                            "modelName": "fake-chat",
                            "customBaseUrl": "",
                            "openaiOptions": _insight_openai_options(None),
                        }
                    ),
                },
                {
                    "domain": "insight_embedding",
                    "provider": "ollama",
                    "payload_json": json.dumps(
                        {
                            "modelName": "fake-embedding",
                            "customBaseUrl": "",
                            "rpmLimit": 0,
                            "transportRetries": 1,
                            "businessRetries": 0,
                            "timeoutSeconds": 0,
                        }
                    ),
                },
            ],
        )
    content = ContentRepository(engine)
    book = content.create_book(title="Insight Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    storage = AssetStorageService(data_root, engine)
    importer = ImageImportService(
        data_root=data_root,
        repository=content,
        storage=storage,
    )
    page_ids: list[str] = []
    for index, color in enumerate(((255, 255, 255), (240, 240, 240)), 1):
        payload = BytesIO()
        with Image.new("RGB", (64, 64), color) as image:
            image.save(payload, format="PNG")
        imported, _ = importer.import_page(
            chapter_id=str(chapter["id"]),
            logical_path=f"page-{index}.png",
            text_style=dict(DEFAULT_TEXT_STYLE),
            upload=BytesIO(payload.getvalue()),
            idempotency_key=f"page-{index}",
        )
        page_ids.append(str(imported["page"]["id"]))
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 555)
    )
    try:
        yield {
            "data_root": data_root,
            "engine": engine,
            "book": book,
            "chapter": chapter,
            "page_ids": page_ids,
            "epoch_id": epoch_id,
        }
    finally:
        engine.dispose()


def _run_job(platform, algorithms: FakeInsightAlgorithms) -> str:
    queue = JobQueueRepository(platform["engine"])
    service = InsightAnalysisWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=algorithms,
    )
    derived = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeDerivedAlgorithms(),
        vector_store=FakeVectorStore(),
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        if (
            str(step["stepKind"]).startswith("insight_build_layer_")
            or str(step["stepKind"]).startswith("insight_stage_")
        ):
            result = derived.handle(fence, step)
        else:
            result = service.handle(fence, step)
        assert result["__already_published__"]
    final = queue.finish_if_complete(fence)
    assert final is not None
    return final


def test_chapter_summaries_aggregate_in_sql_and_keep_empty_chapters(
    insight_platform,
) -> None:
    platform = insight_platform
    empty_chapter = ContentRepository(platform["engine"]).create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Empty Chapter",
    )
    repository = InsightRepository(platform["engine"])

    initial = repository.list_chapters(str(platform["book"]["id"]))["items"]
    assert initial == [
        {
            "chapterId": str(platform["chapter"]["id"]),
            "title": "Chapter",
            "ordinal": 1,
            "pageCount": 2,
            "analysisCounts": {
                "ready": 0,
                "stale": 0,
                "running": 0,
                "failed": 0,
                "not_analyzed": 2,
            },
        },
        {
            "chapterId": str(empty_chapter["id"]),
            "title": "Empty Chapter",
            "ordinal": 2,
            "pageCount": 0,
            "analysisCounts": {
                "ready": 0,
                "stale": 0,
                "running": 0,
                "failed": 0,
                "not_analyzed": 0,
            },
        },
    ]

    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "page",
            "pageIds": [platform["page_ids"][0]],
        },
        idempotency_key="chapter-summary-page-analysis",
    )
    _run_job(platform, FakeInsightAlgorithms())

    updated = repository.list_chapters(str(platform["book"]["id"]))["items"]
    assert updated[0]["analysisCounts"] == {
        "ready": 1,
        "stale": 0,
        "running": 0,
        "failed": 0,
        "not_analyzed": 1,
    }


def test_validation_failure_persists_failed_run_state(
    insight_platform,
    monkeypatch,
) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="validation-failure-run-state",
    )
    queue = JobQueueRepository(platform["engine"])
    service = InsightAnalysisWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeInsightAlgorithms(),
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while True:
        validate_step = queue.next_step(fence)
        assert validate_step is not None
        if validate_step["stepKind"] != "insight_analyze_page":
            break
        service.handle(fence, validate_step)
    assert validate_step["stepKind"] == "insight_validate_run"

    def no_valid_sources(_connection, *, run_id: str) -> dict[str, object]:
        return {"runId": run_id, "successCount": 0}

    monkeypatch.setattr(
        InsightRepository,
        "validate_run_sources",
        staticmethod(no_valid_sources),
    )

    result = service.handle(fence, validate_step)

    assert result == {
        "runId": accepted["runId"],
        "failed": True,
        "__already_published__": True,
    }
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(analysis_runs.c.status).where(
                analysis_runs.c.id == accepted["runId"]
            )
        ).scalar_one() == "failed"


def test_analysis_rejects_incomplete_vlm_settings_before_queue(
    insight_platform,
) -> None:
    platform = insight_platform
    with platform["engine"].begin() as connection:
        connection.execute(
            update(provider_settings)
            .where(
                provider_settings.c.domain == "insight_vlm",
                provider_settings.c.provider == "ollama",
            )
            .values(payload_json="{}")
        )

    with pytest.raises(
        ValueError,
        match=r"provider_settings\.insight_vlm.*missing=.*modelName",
    ):
        InsightAnalysisCommandService(
            platform["engine"]
        ).create_analysis_job(
            command={
                "bookId": str(platform["book"]["id"]),
                "scope": "page",
                "pageIds": [platform["page_ids"][0]],
            },
            idempotency_key="missing-vlm-model",
        )

    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(jobs.c.id).where(jobs.c.kind == "insight_analysis")
        ).scalar_one_or_none() is None


def test_insight_chat_can_reuse_the_frozen_vlm_provider(
    insight_platform,
) -> None:
    platform = insight_platform
    with platform["engine"].begin() as connection:
        raw = connection.execute(
            select(app_settings.c.payload_json).where(
                app_settings.c.domain == "insight"
            )
        ).scalar_one()
        payload = json.loads(raw)
        payload["chat"] = {
            "provider": "gemini",
            "useSameAsVlm": True,
        }
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "insight")
            .values(payload_json=json.dumps(payload))
        )

    config = SettingsResolver(platform["engine"]).resolve_insight(
        book_id=str(platform["book"]["id"]),
        scope="full",
    )

    assert config["chat"] == {
        key: config["vlm"][key]
        for key in (
            "provider",
            "model_name",
            "custom_base_url",
            "openai_options",
        )
    }
    assert "image_max_size" not in config["chat"]


def _run_derived_job(
    platform,
    *,
    algorithms: FakeDerivedAlgorithms,
    vector_store: FakeVectorStore | None = None,
) -> str:
    queue = JobQueueRepository(platform["engine"])
    service = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=algorithms,
        vector_store=vector_store,
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        result = service.handle(fence, step)
        assert result["__already_published__"]
    final = queue.finish_if_complete(fence)
    assert final is not None
    return final


def test_full_analysis_freezes_assets_and_publishes_canonical_results(
    insight_platform,
    monkeypatch,
) -> None:
    platform = insight_platform
    command = InsightAnalysisCommandService(platform["engine"])
    accepted = command.create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="full-1",
    )
    monkeypatch.setattr(
        command,
        "_resolve_targets",
        lambda **_kwargs: pytest.fail(
            "idempotent replay reread current Insight targets"
        ),
    )
    replay = command.create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="full-1",
    )
    assert replay == accepted
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    validation_statements: list[str] = []

    def record_validation_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        validation_statements.append(statement.upper())

    event.listen(
        platform["engine"],
        "before_cursor_execute",
        record_validation_statement,
    )
    try:
        with platform["engine"].begin() as connection:
            InsightRepository.validate_run_sources(
                connection,
                run_id=str(accepted["runId"]),
            )
    finally:
        event.remove(
            platform["engine"],
            "before_cursor_execute",
            record_validation_statement,
        )
    assert sum(
        "FROM PAGE_ASSETS JOIN ASSETS" in statement
        for statement in validation_statements
    ) == 1

    with platform["engine"].connect() as connection:
        run = connection.execute(
            select(analysis_runs).where(
                analysis_runs.c.id == accepted["runId"]
            )
        ).mappings().one()
        payloads = list(
            connection.execute(
                select(analysis_page_results.c.payload_json)
                .where(analysis_page_results.c.run_id == accepted["runId"])
                .order_by(analysis_page_results.c.page_number_snapshot)
            ).scalars()
        )
        head_count = connection.execute(
            select(analysis_heads.c.id).where(
                analysis_heads.c.book_id == platform["book"]["id"]
            )
        ).all()
        frozen_assets = connection.execute(
            select(job_asset_inputs.c.asset_id).where(
                job_asset_inputs.c.job_id == accepted["jobIds"][0]
            )
        ).all()
        layer_indices = set(
            connection.execute(
                select(analysis_layer_results.c.layer_index).where(
                    analysis_layer_results.c.run_id == accepted["runId"],
                    analysis_layer_results.c.status == "published",
                )
            ).scalars()
        )
        active_artifacts = set(
            connection.execute(
                select(
                    analysis_artifacts.c.kind,
                    analysis_artifacts.c.template,
                ).where(
                    analysis_artifacts.c.run_id == accepted["runId"],
                    analysis_artifacts.c.is_active.is_(True),
                )
            )
        )
        active_timeline = connection.execute(
            select(timeline_versions.c.run_id).where(
                timeline_versions.c.is_active.is_(True)
            )
        ).scalar_one()
        active_vector = connection.execute(
            select(vector_generations.c.run_id).where(
                vector_generations.c.is_active.is_(True)
            )
        ).scalar_one()
    assert run["status"] == "completed"
    assert (run["success_count"], run["failed_count"]) == (2, 0)
    assert len(head_count) == 3  # one book head plus two page heads
    assert len(frozen_assets) == 2
    assert layer_indices == {0, 1, 2}
    assert active_artifacts == {
        ("compressed_context", "default"),
        ("overview", "no_spoiler"),
        ("overview", "story_summary"),
    }
    assert active_timeline == accepted["runId"]
    assert active_vector == accepted["runId"]
    assert all('"schema_version":2' in payload for payload in payloads)
    assert all('"scene"' not in payload for payload in payloads)
    assert all('"dialogues"' not in payload for payload in payloads)
    assert all('"characters"' not in payload for payload in payloads)


def test_full_analysis_degraded_publish_keeps_failed_page_missing(
    insight_platform,
) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="degraded-1",
    )
    assert (
        _run_job(platform, FakeInsightAlgorithms(fail_page=2))
        == "completed_with_errors"
    )
    run = InsightRepository(platform["engine"]).get_run(
        str(accepted["runId"])
    )
    assert run["status"] == "completed_with_errors"
    assert run["successCount"] == 1
    assert run["failedCount"] == 1
    assert run["missingPageIds"] == [platform["page_ids"][1]]
    page_one = InsightRepository(platform["engine"]).page_detail(
        page_id=platform["page_ids"][0]
    )
    page_two = InsightRepository(platform["engine"]).page_detail(
        page_id=platform["page_ids"][1]
    )
    assert page_one["analysisState"] == "ready"
    assert page_two["analysisState"] == "failed"
    assert InsightRepository(platform["engine"]).list_chapters(
        str(platform["book"]["id"])
    )["items"][0]["analysisCounts"] == {
        "ready": 1,
        "stale": 0,
        "running": 0,
        "failed": 1,
        "not_analyzed": 0,
    }
    bootstrap = InsightRepository(platform["engine"]).bootstrap()
    book = next(
        item
        for item in bootstrap["books"]
        if item["bookId"] == str(platform["book"]["id"])
    )
    assert book["pageCount"] == 2
    assert book["analyzedPageCount"] == 1
    assert book["activeRun"]["status"] == "completed_with_errors"


def test_failed_run_counts_all_unfinished_targets(insight_platform) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="failed-before-targets-finish",
    )

    with platform["engine"].begin() as connection:
        InsightRepository.mark_run_failed(
            connection,
            run_id=str(accepted["runId"]),
        )
        InsightRepository.mark_run_failed(
            connection,
            run_id=str(accepted["runId"]),
        )

    run = InsightRepository(platform["engine"]).get_run(
        str(accepted["runId"])
    )
    assert run["status"] == "failed"
    assert run["successCount"] == 0
    assert run["failedCount"] == 2
    assert run["missingPageIds"] == platform["page_ids"]


def test_full_analysis_failed_item_retry_refreshes_settings_and_republishes(
    insight_platform,
) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="retry-source",
    )
    assert (
        _run_job(platform, FakeInsightAlgorithms(fail_page=2))
        == "completed_with_errors"
    )

    with platform["engine"].begin() as connection:
        settings_row = connection.execute(
            select(
                app_settings.c.payload_json,
                app_settings.c.revision,
            ).where(app_settings.c.domain == "insight")
        ).mappings().one()
        payload = json.loads(settings_row["payload_json"])
        payload.setdefault("analysis", {}).setdefault("batch", {})[
            "pagesPerBatch"
        ] = 7
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "insight")
            .values(
                payload_json=json.dumps(payload, separators=(",", ":")),
                revision=int(settings_row["revision"]) + 1,
            )
        )

    retried = JobRetryService(platform["engine"]).retry(
        job_id=str(accepted["jobIds"][0]),
        failed_only=True,
        strategy="current",
        idempotency_key="retry-current",
    )
    retry_job_id = str(retried["jobIds"][0])
    with platform["engine"].connect() as connection:
        config = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == retry_job_id)
            ).scalar_one()
        )

    assert retried["retryMode"] == "current"
    assert config["runId"] == retried["runId"]
    assert retried["runId"] != accepted["runId"]
    assert config["analysis"]["pagesPerBatch"] == 7
    retry_detail = JobQueueRepository(platform["engine"]).get_job(
        retry_job_id
    )
    assert retry_detail["counts"]["total"] == 2
    assert retry_detail["target"]["pageCount"] == 1
    assert retry_detail["target"]["retryItemCount"] == 1
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    run = InsightRepository(platform["engine"]).get_run(
        str(retried["runId"])
    )
    assert run["status"] == "completed"
    assert run["successCount"] == 2
    assert run["failedCount"] == 0
    assert run["missingPageIds"] == []


@pytest.mark.parametrize("strategy", ("current", "original"))
def test_partial_analysis_retry_creates_isolated_run_and_rebinds_source(
    insight_platform,
    strategy: str,
) -> None:
    platform = insight_platform
    page_id = str(platform["page_ids"][1])
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "page",
            "pageIds": [page_id],
        },
        idempotency_key=f"partial-retry-source-{strategy}",
    )
    assert (
        _run_job(platform, FakeInsightAlgorithms(fail_page=2))
        == "completed_with_errors"
    )

    with platform["engine"].connect() as connection:
        original_asset_id = str(
            connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == page_id,
                    page_assets.c.role == "source",
                )
            ).scalar_one()
        )
    replacement = BytesIO()
    with Image.new("RGB", (64, 64), (12, 34, 56)) as image:
        image.save(replacement, format="PNG")
    replacement_asset = AssetStorageService(
        platform["data_root"],
        platform["engine"],
    ).publish_bytes(
        replacement.getvalue(),
        extension="png",
        mime_type="image/png",
        width=64,
        height=64,
        bind=lambda connection, asset_id: connection.execute(
            update(page_assets)
            .where(
                page_assets.c.page_id == page_id,
                page_assets.c.role == "source",
            )
            .values(asset_id=asset_id)
        ),
    )

    retried = JobRetryService(platform["engine"]).retry(
        job_id=str(accepted["jobIds"][0]),
        failed_only=True,
        strategy=strategy,
        idempotency_key=f"partial-retry-{strategy}",
    )
    retry_job_id = str(retried["jobIds"][0])
    expected_asset_id = replacement_asset.id
    with platform["engine"].connect() as connection:
        bound_asset_id = str(
            connection.execute(
                select(job_asset_inputs.c.asset_id).where(
                    job_asset_inputs.c.job_id == retry_job_id,
                    job_asset_inputs.c.role == "source",
                )
            ).scalar_one()
        )
        config = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == retry_job_id)
            ).scalar_one()
        )

    assert retried["retryMode"] == strategy
    assert retried["runId"] != accepted["runId"]
    assert config["runId"] == retried["runId"]
    assert config["targetCount"] == 1
    assert bound_asset_id == expected_asset_id
    assert bound_asset_id != original_asset_id
    assert (
        JobQueueRepository(platform["engine"])
        .get_job(retry_job_id)["counts"]["total"]
        == 2
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    run = InsightRepository(platform["engine"]).get_run(
        str(retried["runId"])
    )
    with platform["engine"].connect() as connection:
        result_asset_id = str(
            connection.execute(
                select(analysis_page_results.c.source_asset_id).where(
                    analysis_page_results.c.run_id == str(retried["runId"])
                )
            ).scalar_one()
        )
    assert run["status"] == "completed"
    assert run["successCount"] == 1
    assert result_asset_id == expected_asset_id


def test_page_summaries_include_source_assets_but_only_thumbnail_urls(
    insight_platform,
) -> None:
    repository = InsightRepository(insight_platform["engine"])
    result = repository.list_pages(
        book_id=str(insight_platform["book"]["id"]),
        chapter_id=None,
        after=0,
        limit=100,
    )

    assert [item["pageId"] for item in result["items"]] == (
        insight_platform["page_ids"]
    )
    assert all(item["sourceAssetId"] for item in result["items"])
    assert all(
        item["thumbnailUrl"].startswith("/api/v2/assets/")
        for item in result["items"]
    )
    assert all(
        item["sourceAssetId"] not in item["thumbnailUrl"]
        for item in result["items"]
    )


def test_page_scope_publishes_page_head_without_switching_book_head(
    insight_platform,
) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "page",
            "pageIds": [platform["page_ids"][1]],
        },
        idempotency_key="page-scope-1",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    with platform["engine"].connect() as connection:
        book_head = connection.execute(
            select(analysis_heads.c.id).where(
                analysis_heads.c.book_id == platform["book"]["id"],
                analysis_heads.c.page_id.is_(None),
            )
        ).scalar_one_or_none()
        page_head = connection.execute(
            select(analysis_heads.c.active_run_id).where(
                analysis_heads.c.page_id == platform["page_ids"][1]
            )
        ).scalar_one()
    assert book_head is None
    assert page_head == accepted["runId"]


def test_page_state_distinguishes_local_reanalysis_from_full_run_fallback(
    insight_platform,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    page_id = platform["page_ids"][1]
    commands = InsightAnalysisCommandService(platform["engine"])
    repository = InsightRepository(platform["engine"])

    commands.create_analysis_job(
        command={"bookId": book_id, "scope": "full"},
        idempotency_key="page-state-full-baseline",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    commands.create_analysis_job(
        command={
            "bookId": book_id,
            "scope": "page",
            "pageIds": [page_id],
        },
        idempotency_key="page-state-local-reanalysis",
    )
    pending_detail = repository.page_detail(page_id=page_id)
    assert pending_detail["analysisState"] == "running"
    assert pending_detail["analysis"] is not None
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    local_detail = repository.page_detail(page_id=page_id)
    assert local_detail["analysisState"] == "ready"
    assert local_detail["staleReasons"] == []
    listed = repository.list_pages(
        book_id=book_id,
        chapter_id=None,
        after=0,
        limit=100,
    )
    assert listed["items"][1]["analysisState"] == "ready"
    assert repository.list_chapters(book_id)["items"][0][
        "analysisCounts"
    ] == {
        "ready": 2,
        "stale": 0,
        "running": 0,
        "failed": 0,
        "not_analyzed": 0,
    }
    commands.create_analysis_job(
        command={"bookId": book_id, "scope": "full"},
        idempotency_key="page-state-degraded-full",
    )
    assert (
        _run_job(platform, FakeInsightAlgorithms(fail_page=2))
        == "completed_with_errors"
    )
    fallback_detail = repository.page_detail(page_id=page_id)
    assert fallback_detail["analysisState"] == "stale"
    assert fallback_detail["staleReasons"] == [
        "fallback_from_previous_run"
    ]
    assert repository.list_chapters(book_id)["items"][0][
        "analysisCounts"
    ] == {
        "ready": 1,
        "stale": 1,
        "running": 0,
        "failed": 0,
        "not_analyzed": 0,
    }
    fallback_snapshot = InsightDerivedRepository(
        platform["engine"]
    ).snapshot(book_id=book_id)
    assert [page["pageId"] for page in fallback_snapshot.pages] == [
        platform["page_ids"][0]
    ]
    assert fallback_snapshot.source_run_status == "completed_with_errors"

    commands.create_analysis_job(
        command={
            "bookId": book_id,
            "scope": "page",
            "pageIds": [page_id],
        },
        idempotency_key="page-state-reanalysis-after-degraded-full",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    recovered_detail = repository.page_detail(page_id=page_id)
    assert recovered_detail["analysisState"] == "ready"
    assert recovered_detail["staleReasons"] == []
    assert repository.list_chapters(book_id)["items"][0][
        "analysisCounts"
    ] == {
        "ready": 2,
        "stale": 0,
        "running": 0,
        "failed": 0,
        "not_analyzed": 0,
    }
    recovered_snapshot = InsightDerivedRepository(
        platform["engine"]
    ).snapshot(book_id=book_id)
    assert set(recovered_snapshot.result_ids) == {
        item["activeAnalysisId"]
        for item in repository.list_pages(
            book_id=book_id,
            chapter_id=None,
            after=0,
            limit=100,
        )["items"]
    }
    assert recovered_snapshot.source_run_id is None


def test_page_state_remains_running_until_the_run_is_published(
    insight_platform,
) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="page-state-through-publication",
    )
    queue = JobQueueRepository(platform["engine"])
    worker = InsightAnalysisWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeInsightAlgorithms(),
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    step = queue.next_step(fence)
    assert step is not None
    worker.handle(fence, step)
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(analysis_run_targets.c.status).where(
                analysis_run_targets.c.run_id == accepted["runId"],
                analysis_run_targets.c.page_id_snapshot
                == platform["page_ids"][0],
            )
        ).scalar_one() == "completed"
    listed = InsightRepository(platform["engine"]).list_pages(
        book_id=str(platform["book"]["id"]),
        chapter_id=None,
        after=0,
        limit=100,
    )
    assert listed["items"][0]["analysisState"] == "running"
    assert queue.request_cancel(str(accepted["jobIds"][0]))["status"] in {
        "cancelling",
        "cancelled",
    }


def test_page_reorder_marks_analysis_and_whole_book_derivatives_stale(
    insight_platform,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    commands = InsightAnalysisCommandService(platform["engine"])
    repository = InsightRepository(platform["engine"])
    content = ContentRepository(platform["engine"])
    commands.create_analysis_job(
        command={"bookId": book_id, "scope": "full"},
        idempotency_key="page-order-full-baseline",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    chapter = content.list_chapters(book_id)["chapters"][0]
    content.reorder_pages(
        chapter_id=str(chapter["id"]),
        ordered_ids=list(reversed(platform["page_ids"])),
        base_revision=int(chapter["pageOrderRevision"]),
    )

    detail = repository.page_detail(page_id=platform["page_ids"][0])
    assert detail["displayPageNumber"] == 2
    assert detail["analysis"]["page_number_snapshot"] == 1
    assert detail["analysisState"] == "stale"
    assert detail["staleReasons"] == ["page_order_changed"]
    listed = repository.list_pages(
        book_id=book_id,
        chapter_id=None,
        after=0,
        limit=100,
    )
    assert [item["analysisState"] for item in listed["items"]] == [
        "stale",
        "stale",
    ]
    assert repository.list_chapters(book_id)["items"][0][
        "analysisCounts"
    ] == {
        "ready": 0,
        "stale": 2,
        "running": 0,
        "failed": 0,
        "not_analyzed": 0,
    }
    snapshot = InsightDerivedRepository(platform["engine"]).snapshot(
        book_id=book_id
    )
    assert [page["pageId"] for page in snapshot.pages] == list(
        reversed(platform["page_ids"])
    )
    assert [page["pageNumber"] for page in snapshot.pages] == [1, 2]
    assert [
        page["analysis"]["page_number_snapshot"]
        for page in snapshot.pages
    ] == [2, 1]

    recent = repository.list_recent_page_analyses(book_id=book_id)
    assert {item["displayPageNumber"] for item in recent["items"]} == {1, 2}
    with platform["engine"].connect() as connection:
        assert set(
            connection.execute(
                select(analysis_artifacts.c.status).where(
                    analysis_artifacts.c.book_id == book_id,
                    analysis_artifacts.c.is_active.is_(True),
                )
            ).scalars()
        ) == {"stale"}
        assert connection.execute(
            select(timeline_versions.c.status).where(
                timeline_versions.c.book_id == book_id,
                timeline_versions.c.is_active.is_(True),
            )
        ).scalar_one() == "stale"
        assert connection.execute(
            select(vector_generations.c.status).where(
                vector_generations.c.book_id == book_id,
                vector_generations.c.is_active.is_(True),
            )
        ).scalar_one() == "stale"

    InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=book_id,
        kind="overview",
        template="no_spoiler",
        idempotency_key="page-order-overview-rebuild",
    )
    assert (
        _run_derived_job(platform, algorithms=FakeDerivedAlgorithms())
        == "completed"
    )
    rebuilt_overview = InsightDerivedRepository(
        platform["engine"]
    ).get_artifact(
        book_id=book_id,
        kind="overview",
        template="no_spoiler",
    )
    assert rebuilt_overview is not None
    assert rebuilt_overview["status"] == "ready"


def test_source_replacement_marks_analysis_and_derivatives_stale(
    insight_platform,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    page_id = platform["page_ids"][0]
    commands = InsightAnalysisCommandService(platform["engine"])
    repository = InsightRepository(platform["engine"])
    content = ContentRepository(platform["engine"])
    commands.create_analysis_job(
        command={"bookId": book_id, "scope": "full"},
        idempotency_key="source-replacement-full-baseline",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=book_id,
        kind="overview",
        template="no_spoiler",
        idempotency_key="source-replacement-overview-history",
    )
    assert (
        _run_derived_job(platform, algorithms=FakeDerivedAlgorithms())
        == "completed"
    )
    with platform["engine"].connect() as connection:
        inactive_artifact_id = connection.execute(
            select(analysis_artifacts.c.id).where(
                analysis_artifacts.c.book_id == book_id,
                analysis_artifacts.c.kind == "overview",
                analysis_artifacts.c.template == "no_spoiler",
                analysis_artifacts.c.is_active.is_(False),
            )
        ).scalar_one()

    replacement = BytesIO()
    with Image.new("RGB", (64, 64), (12, 34, 56)) as image:
        image.save(replacement, format="PNG")
    ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(
            platform["data_root"],
            platform["engine"],
        ),
    ).replace_page_source(
        page_id=page_id,
        base_source_revision=int(
            content.get_page_summary(page_id)["sourceRevision"]
        ),
        upload=BytesIO(replacement.getvalue()),
        idempotency_key="source-replacement-stales-insight",
    )

    detail = repository.page_detail(page_id=page_id)
    assert detail["analysisState"] == "stale"
    assert detail["staleReasons"] == ["source_changed"]
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(analysis_artifacts.c.status).where(
                analysis_artifacts.c.id == inactive_artifact_id
            )
        ).scalar_one() == "ready"
        assert set(
            connection.execute(
                select(analysis_artifacts.c.status).where(
                    analysis_artifacts.c.book_id == book_id,
                    analysis_artifacts.c.is_active.is_(True),
                )
            ).scalars()
        ) == {"stale"}
        assert connection.execute(
            select(timeline_versions.c.status).where(
                timeline_versions.c.book_id == book_id,
                timeline_versions.c.is_active.is_(True),
            )
        ).scalar_one() == "stale"
        assert connection.execute(
            select(vector_generations.c.status).where(
                vector_generations.c.book_id == book_id,
                vector_generations.c.is_active.is_(True),
            )
        ).scalar_one() == "stale"


def test_derived_rebuild_requires_new_pages_to_be_analyzed(
    insight_platform,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    analysis_commands = InsightAnalysisCommandService(platform["engine"])
    analysis_commands.create_analysis_job(
        command={"bookId": book_id, "scope": "full"},
        idempotency_key="new-page-derived-full-baseline",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    payload = BytesIO()
    with Image.new("RGB", (64, 64), (78, 90, 12)) as image:
        image.save(payload, format="PNG")
    imported, _ = ImageImportService(
        data_root=platform["data_root"],
        repository=ContentRepository(platform["engine"]),
        storage=AssetStorageService(
            platform["data_root"],
            platform["engine"],
        ),
    ).import_page(
        chapter_id=str(platform["chapter"]["id"]),
        logical_path="new-page.png",
        text_style=dict(DEFAULT_TEXT_STYLE),
        upload=BytesIO(payload.getvalue()),
        idempotency_key="new-page-derived-import",
    )
    new_page_id = str(imported["page"]["id"])

    with pytest.raises(
        InsightConflict,
        match="pages without published analysis",
    ):
        InsightDerivedCommandService(platform["engine"]).create_job(
            book_id=book_id,
            kind="overview",
            template="no_spoiler",
            idempotency_key="new-page-derived-before-analysis",
        )

    analysis_commands.create_analysis_job(
        command={
            "bookId": book_id,
            "scope": "page",
            "pageIds": [new_page_id],
        },
        idempotency_key="new-page-derived-page-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    snapshot = InsightDerivedRepository(platform["engine"]).snapshot(
        book_id=book_id
    )
    assert [page["pageId"] for page in snapshot.pages] == [
        *platform["page_ids"],
        new_page_id,
    ]


def test_note_revision_and_citation_snapshots(insight_platform) -> None:
    platform = insight_platform
    repository = InsightRepository(platform["engine"])
    note = repository.create_note(
        idempotency_key="note-revision-create",
        book_id=str(platform["book"]["id"]),
        title="线索",
        content="内容",
        citations=[{"pageId": platform["page_ids"][1]}],
    )
    assert note["revision"] == 1
    assert note["citations"][0]["pageNumberSnapshot"] == 2
    updated = repository.update_note(
        idempotency_key="note-revision-update",
        note_id=str(note["noteId"]),
        base_revision=1,
        title="线索（更新）",
        content="新内容",
        citations=[{"pageId": platform["page_ids"][0]}],
    )
    assert updated["revision"] == 2
    with pytest.raises(InsightConflict, match="revision"):
        repository.update_note(
            idempotency_key="note-revision-stale-update",
            note_id=str(note["noteId"]),
            base_revision=1,
            title="过期写入",
            content="",
            citations=[],
        )
    page = repository.list_notes(
        book_id=str(platform["book"]["id"]),
        limit=1,
    )
    assert page["items"][0]["content"] is None
    detail_page = repository.list_notes(
        book_id=str(platform["book"]["id"]),
        limit=1,
        include_content=True,
    )
    assert detail_page["items"][0]["content"] == "新内容"
    assert repository.get_note(note_id=note["noteId"])["content"] == "新内容"


def test_note_mutations_replay_idempotently(insight_platform) -> None:
    platform = insight_platform
    repository = InsightRepository(platform["engine"])
    book_id = str(platform["book"]["id"])
    create_input = {
        "idempotency_key": "note-idempotent-create",
        "book_id": book_id,
        "title": "幂等笔记",
        "content": "初始内容",
    }
    created = repository.create_note(**create_input)
    assert repository.create_note(**create_input) == created
    assert len(repository.list_notes(book_id=book_id)["items"]) == 1
    with pytest.raises(InsightConflict, match="Idempotency-Key"):
        repository.create_note(
            **{**create_input, "title": "不同内容"}
        )

    update_input = {
        "idempotency_key": "note-idempotent-update",
        "note_id": str(created["noteId"]),
        "base_revision": 1,
        "title": "幂等笔记",
        "content": "更新内容",
    }
    updated = repository.update_note(**update_input)
    assert repository.update_note(**update_input) == updated
    assert repository.get_note(note_id=str(created["noteId"]))["revision"] == 2

    delete_input = {
        "idempotency_key": "note-idempotent-delete",
        "note_id": str(created["noteId"]),
        "base_revision": 2,
    }
    repository.delete_note(**delete_input)
    repository.delete_note(**delete_input)
    with pytest.raises(InsightConflict, match="Idempotency-Key"):
        repository.delete_note(
            **{**delete_input, "base_revision": 3}
        )


@pytest.mark.parametrize("cursor", ("*", "不是-base64"))
def test_note_list_rejects_malformed_cursor(
    insight_platform,
    cursor: str,
) -> None:
    with pytest.raises(ValueError, match="invalid note cursor"):
        InsightRepository(insight_platform["engine"]).list_notes(
            book_id=str(insight_platform["book"]["id"]),
            cursor=cursor,
        )


def test_note_list_rejects_noncanonical_cursor(insight_platform) -> None:
    padded = base64.urlsafe_b64encode(
        b"2026-08-11T12:00:00|00000000-0000-0000-0000-000000000001"
    ).decode("ascii")
    assert padded.endswith("=")
    with pytest.raises(ValueError, match="invalid note cursor"):
        InsightRepository(insight_platform["engine"]).list_notes(
            book_id=str(insight_platform["book"]["id"]),
            cursor=padded,
        )


def test_note_citations_use_bulk_queries(insight_platform) -> None:
    platform = insight_platform
    repository = InsightRepository(platform["engine"])
    create_statements: list[str] = []

    def record_create_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        create_statements.append(statement.upper())

    event.listen(
        platform["engine"],
        "before_cursor_execute",
        record_create_statement,
    )
    try:
        repository.create_note(
            idempotency_key="note-bulk-citations-create",
            book_id=str(platform["book"]["id"]),
            title="批量引用",
            content="内容",
            citations=[
                {"pageId": page_id} for page_id in platform["page_ids"]
            ],
        )
    finally:
        event.remove(
            platform["engine"],
            "before_cursor_execute",
            record_create_statement,
        )
    assert sum(
        "FROM ANALYSIS_HEADS" in statement
        for statement in create_statements
    ) == 1

    repository.create_note(
        idempotency_key="note-bulk-citations-second",
        book_id=str(platform["book"]["id"]),
        title="第二条",
        content="内容",
        citations=[{"pageId": platform["page_ids"][0]}],
    )
    list_statements: list[str] = []

    def record_list_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        list_statements.append(statement.upper())

    event.listen(
        platform["engine"],
        "before_cursor_execute",
        record_list_statement,
    )
    try:
        listed = repository.list_notes(
            book_id=str(platform["book"]["id"]),
            limit=10,
        )
    finally:
        event.remove(
            platform["engine"],
            "before_cursor_execute",
            record_list_statement,
        )

    assert len(listed["items"]) == 2
    assert sum(
        "FROM NOTE_CITATIONS" in statement
        for statement in list_statements
    ) == 1


def test_insight_bootstrap_identifies_active_job_kinds(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="bootstrap-derived-source",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    accepted = InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=str(platform["book"]["id"]),
        kind="timeline",
        template="default",
        idempotency_key="bootstrap-derived-kind",
    )

    active = next(
        item
        for item in InsightRepository(platform["engine"]).bootstrap()["activeJobs"]
        if item["jobId"] == accepted["jobIds"][0]
    )

    assert active["kind"] == "derived_rebuild"


def test_derived_artifacts_timeline_and_vectors_publish_as_generations(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="derived-source",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    commands = InsightDerivedCommandService(platform["engine"])
    repository = InsightDerivedRepository(platform["engine"])

    commands.create_job(
        book_id=str(platform["book"]["id"]),
        kind="overview",
        template="no_spoiler",
        idempotency_key="overview-1",
    )
    assert (
        _run_derived_job(platform, algorithms=FakeDerivedAlgorithms())
        == "completed"
    )
    overview = repository.get_artifact(
        book_id=str(platform["book"]["id"]),
        kind="overview",
        template="no_spoiler",
    )
    assert overview is not None
    assert overview["status"] == "ready"
    assert overview["payload"]["content"] == "1 pages"
    assert InsightRepository(platform["engine"]).list_overview_templates(
        str(platform["book"]["id"])
    ) == {"items": ["no_spoiler", "story_summary"]}
    recent_pages = InsightRepository(
        platform["engine"]
    ).list_recent_page_analyses(
        book_id=str(platform["book"]["id"]),
        limit=5,
    )
    assert len(recent_pages["items"]) == 2
    assert {
        item["displayPageNumber"] for item in recent_pages["items"]
    } == {1, 2}
    assert all(item["summary"] for item in recent_pages["items"])

    commands.create_job(
        book_id=str(platform["book"]["id"]),
        kind="timeline",
        template="default",
        idempotency_key="timeline-1",
    )
    assert (
        _run_derived_job(platform, algorithms=FakeDerivedAlgorithms())
        == "completed"
    )
    timeline = repository.get_timeline(
        book_id=str(platform["book"]["id"])
    )
    assert timeline is not None
    assert timeline["mode"] == "enhanced"
    assert timeline["events"][0]["summary"] == "event"
    assert timeline["characters"][0]["name"] == "Saber"
    assert timeline["pageThumbnails"]["1"].startswith("/api/v2/assets/")

    global_status = repository.qa_status(
        book_id=str(platform["book"]["id"]),
        mode="global",
    )
    assert global_status == {
        "available": True,
        "reason": None,
    }
    with platform["engine"].begin() as connection:
        connection.execute(
            update(analysis_artifacts)
            .where(
                analysis_artifacts.c.book_id == str(platform["book"]["id"]),
                analysis_artifacts.c.kind == "compressed_context",
                analysis_artifacts.c.template == "default",
                analysis_artifacts.c.is_active.is_(True),
            )
            .values(status="stale")
        )
    assert repository.qa_status(
        book_id=str(platform["book"]["id"]),
        mode="global",
    ) == {
        "available": False,
        "reason": "compressed_context_stale",
        "repairAction": "compressed_context_rebuild",
    }
    commands.create_job(
        book_id=str(platform["book"]["id"]),
        kind="compressed_context",
        template="default",
        idempotency_key="compressed-context-1",
    )
    assert (
        _run_derived_job(platform, algorithms=FakeDerivedAlgorithms())
        == "completed"
    )
    assert repository.qa_status(
        book_id=str(platform["book"]["id"]),
        mode="global",
    ) == {
        "available": True,
        "reason": None,
    }

    vector_store = FakeVectorStore()
    commands.create_job(
        book_id=str(platform["book"]["id"]),
        kind="vector",
        template="default",
        idempotency_key="vector-1",
    )
    assert (
        _run_derived_job(
            platform,
            algorithms=FakeDerivedAlgorithms(),
            vector_store=vector_store,
        )
        == "completed"
    )
    assert len(vector_store.publications) == 1
    status = repository.qa_status(book_id=str(platform["book"]["id"]))
    assert status["available"]
    assert status["coverage"] == {"pages": 2, "events": 2}


def test_local_reanalysis_does_not_reuse_older_full_run_derived_inputs(
    insight_platform,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    commands = InsightAnalysisCommandService(platform["engine"])
    commands.create_analysis_job(
        command={"bookId": book_id, "scope": "full"},
        idempotency_key="derived-current-full",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    repository = InsightDerivedRepository(platform["engine"])
    full_snapshot = repository.snapshot(book_id=book_id)
    assert full_snapshot.source_run_id is not None
    assert repository.compressed_context_input(full_snapshot) is not None

    commands.create_analysis_job(
        command={
            "bookId": book_id,
            "scope": "page",
            "pageIds": [platform["page_ids"][0]],
        },
        idempotency_key="derived-current-local",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    current = repository.snapshot(book_id=book_id)
    assert current.source_run_id is None
    assert current.source_run_status is None
    assert repository.compressed_context_input(current) is None
    summary_inputs = repository.summary_inputs(current)
    assert {item["resultId"] for item in summary_inputs} == set(current.result_ids)
    assert all("page_summary" in item["analysis"] for item in summary_inputs)


def test_qa_status_does_not_report_corrupt_analysis_as_missing(
    insight_platform,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={"bookId": book_id, "scope": "full"},
        idempotency_key="qa-corrupt-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    with platform["engine"].begin() as connection:
        active_result_id = connection.execute(
            select(analysis_heads.c.active_result_id).where(
                analysis_heads.c.book_id == book_id,
                analysis_heads.c.page_id == platform["page_ids"][0],
            )
        ).scalar_one()
        connection.execute(
            update(analysis_page_results)
            .where(analysis_page_results.c.id == active_result_id)
            .values(payload_json="[]")
        )

    with pytest.raises(InsightConflict, match="must be an object"):
        InsightDerivedRepository(platform["engine"]).qa_status(book_id=book_id)


@pytest.mark.parametrize(
    ("kind", "template"),
    [
        ("overview", "legacy_template"),
        ("timeline", "no_spoiler"),
        ("vector", "story_summary"),
        ("compressed_context", "story_summary"),
    ],
)
def test_derived_commands_reject_noncanonical_templates(
    insight_platform,
    kind,
    template,
) -> None:
    platform = insight_platform
    with pytest.raises(ValueError, match="template"):
        InsightDerivedCommandService(platform["engine"]).create_job(
            book_id=str(platform["book"]["id"]),
            kind=kind,
            template=template,
            idempotency_key=f"invalid-template-{kind}",
        )


def test_vector_cancel_keeps_partial_generation_without_switching_active(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={"bookId": str(platform["book"]["id"]), "scope": "full"},
        idempotency_key="vector-cancel-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    accepted = InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=str(platform["book"]["id"]),
        kind="vector",
        template="default",
        idempotency_key="vector-cancel-safe-point",
    )
    job_id = str(accepted["jobIds"][0])
    queue = JobQueueRepository(platform["engine"])
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    step = queue.next_step(fence)
    assert step is not None
    service = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeDerivedAlgorithms(),
        vector_store=CheckpointingFakeVectorStore(
            queue=queue,
            job_id=job_id,
            control="cancel",
        ),
    )

    result = service.handle(fence, step)

    assert result["__control_drained__"]
    assert queue.finalize_control(fence) == "cancelled"
    detail = queue.get_job(job_id)
    checkpoint = detail["items"][0]["steps"][0]["checkpoint"]
    assert checkpoint["pageCount"] == 2
    assert checkpoint["eventCount"] == 0
    assert 0 < checkpoint["coverage"] < 1
    with platform["engine"].connect() as connection:
        generations = list(
            connection.execute(
                select(vector_generations)
                .where(vector_generations.c.book_id == platform["book"]["id"])
                .order_by(vector_generations.c.generation)
            ).mappings()
        )
    assert len(generations) == 2
    assert generations[0]["is_active"]
    assert not generations[1]["is_active"]
    assert generations[1]["status"] == "building"
    assert generations[1]["page_count"] == 2


def test_vector_pause_resumes_from_checkpoint_and_switches_only_on_completion(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={"bookId": str(platform["book"]["id"]), "scope": "full"},
        idempotency_key="vector-pause-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    accepted = InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=str(platform["book"]["id"]),
        kind="vector",
        template="default",
        idempotency_key="vector-pause-safe-point",
    )
    job_id = str(accepted["jobIds"][0])
    queue = JobQueueRepository(platform["engine"])
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    step = queue.next_step(fence)
    assert step is not None
    pausing_store = CheckpointingFakeVectorStore(
        queue=queue,
        job_id=job_id,
        control="pause",
    )
    service = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeDerivedAlgorithms(),
        vector_store=pausing_store,
    )
    assert service.handle(fence, step)["__control_drained__"]
    assert queue.finalize_control(fence) == "paused"

    queue.resume(job_id)
    resumed_fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    if resumed_fence is None:
        resumed_fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert resumed_fence is not None
    resumed_step = queue.next_step(resumed_fence)
    assert resumed_step is not None
    assert resumed_step["checkpoint"]["pageCount"] == 2
    resumed_store = CheckpointingFakeVectorStore(
        queue=queue,
        job_id=job_id,
    )
    resumed_service = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeDerivedAlgorithms(),
        vector_store=resumed_store,
    )

    completed = resumed_service.handle(resumed_fence, resumed_step)

    assert "__control_drained__" not in completed
    assert queue.finish_if_complete(resumed_fence) == "completed"
    assert resumed_store.calls == [
        {
            "resume": True,
            "initialPageCount": 2,
            "initialEventCount": 0,
        }
    ]
    with platform["engine"].connect() as connection:
        active = connection.execute(
            select(vector_generations).where(
                vector_generations.c.book_id == platform["book"]["id"],
                vector_generations.c.is_active.is_(True),
            )
        ).mappings().one()
    assert active["generation"] == 2
    assert active["status"] == "ready"
    assert active["page_count"] == 2
    assert active["event_count"] == 2


def test_provider_timeline_falls_back_through_compressed_context(
    monkeypatch,
) -> None:
    algorithms = ProviderDerivedAlgorithms()
    calls = iter(
        [
            ValueError("enhanced output was invalid"),
            {
                "content": {"story_summary": "压缩时间线"},
                "events": [{"summary": "事件", "page_numbers": [1]}],
                "characters": [],
            },
        ]
    )

    def fake_chat_json(*_args, **_kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(algorithms, "_chat_json", fake_chat_json)
    result = algorithms.build_timeline(
        [
            {
                "pageId": "page-1",
                "pageNumber": 1,
                "analysis": {"key_events": [{"summary": "事件"}]},
            },
            {
                "pageId": "page-1",
                "pageIds": ["page-1"],
                "pageNumber": 1,
                "pageNumbers": [1],
                "analysis": {
                    "compressed_context": {"content": "压缩上下文"},
                },
            },
        ],
        config={},
    )

    assert result["mode"] == "compressed"
    assert result["content"] == {
        "story_summary": "压缩时间线",
        "requested_mode": "enhanced",
        "actual_mode": "compressed",
        "fallback_reason": "enhanced output was invalid",
        "degraded": True,
    }


@pytest.mark.parametrize(
    ("method_name", "result", "message"),
    [
        ("build_layer", {}, "non-empty object"),
        ("build_overview", {}, "title must be a non-empty string"),
        ("build_overview", {"title": "title", "content": ""}, "content must be a non-empty string"),
        ("build_compressed_context", {}, "non-empty object"),
    ],
)
def test_provider_derived_algorithms_reject_empty_or_partial_results(
    monkeypatch,
    method_name,
    result,
    message,
) -> None:
    algorithms = ProviderDerivedAlgorithms()
    monkeypatch.setattr(
        algorithms,
        "_chat_json",
        lambda *_args, **_kwargs: result,
    )
    pages = [
        {
            "pageId": "page-1",
            "pageNumber": 1,
            "analysis": {"page_summary": "summary"},
        }
    ]
    kwargs = {"config": {}}
    if method_name == "build_layer":
        kwargs["layer"] = {
            "name": "汇总",
            "promptType": "segment_summary",
        }
    elif method_name == "build_overview":
        kwargs["template"] = "story_summary"

    with pytest.raises(ValueError, match=message):
        getattr(algorithms, method_name)(pages, **kwargs)


def test_provider_timeline_does_not_fallback_after_memory_failure(
    monkeypatch,
) -> None:
    algorithms = ProviderDerivedAlgorithms()
    calls = 0

    def fail_chat_json(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise MemoryError("native allocation failed")

    monkeypatch.setattr(algorithms, "_chat_json", fail_chat_json)
    with pytest.raises(MemoryError, match="allocation failed"):
        algorithms.build_timeline(
            [
                {
                    "pageId": "page-1",
                    "pageNumber": 1,
                    "analysis": {
                        "compressed_context": {"content": "压缩上下文"},
                    },
                }
            ],
            config={},
        )
    assert calls == 1


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs", "message"),
    [
        (
            "build_layer",
            ([{"pageId": "page-1", "pageNumber": 1, "analysis": {"page_summary": "summary"}}],),
            {
                "layer": {
                    "name": "汇总",
                    "promptType": "segment_summary",
                },
                "config": {},
            },
            "summary layer response must be a non-empty object",
        ),
        (
            "build_overview",
            ([{"pageId": "page-1", "pageNumber": 1, "analysis": {"page_summary": "summary"}}],),
            {"template": "story_summary", "config": {}},
            "overview response must be an object",
        ),
        (
            "build_compressed_context",
            ([{"pageId": "page-1", "pageNumber": 1, "analysis": {"page_summary": "summary"}}],),
            {"config": {}},
            "compressed context response must be a non-empty object",
        ),
    ],
)
def test_provider_derived_algorithms_reject_non_object_model_results(
    monkeypatch,
    method_name,
    args,
    kwargs,
    message,
) -> None:
    algorithms = ProviderDerivedAlgorithms()
    monkeypatch.setattr(algorithms, "_chat_json", lambda *_args, **_kwargs: "bad")

    with pytest.raises(ValueError, match=message):
        getattr(algorithms, method_name)(*args, **kwargs)


def test_vector_worker_rejects_invalid_store_success_result(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={"bookId": str(platform["book"]["id"]), "scope": "full"},
        idempotency_key="invalid-vector-result-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=str(platform["book"]["id"]),
        kind="vector",
        template="default",
        idempotency_key="invalid-vector-result",
    )
    queue = JobQueueRepository(platform["engine"])
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    step = queue.next_step(fence)
    assert step is not None

    class InvalidVectorStore:
        def publish_batches(self, **kwargs):
            assert kwargs["on_batch"](
                "pages",
                kwargs["expected_page_count"],
            )
            assert kwargs["on_batch"](
                "events",
                kwargs["expected_event_count"],
            )
            return None

    service = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeDerivedAlgorithms(),
        vector_store=InvalidVectorStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(InsightConflict, match="invalid result"):
        service.handle(fence, step)
    with platform["engine"].connect() as connection:
        failed = connection.execute(
            select(vector_generations).where(
                vector_generations.c.book_id == platform["book"]["id"],
                vector_generations.c.is_active.is_(False),
            )
        ).mappings().one()
    assert failed["status"] == "failed"


def test_vector_worker_rejects_malformed_resume_checkpoint(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={"bookId": str(platform["book"]["id"]), "scope": "full"},
        idempotency_key="invalid-vector-checkpoint-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=str(platform["book"]["id"]),
        kind="vector",
        template="default",
        idempotency_key="invalid-vector-checkpoint",
    )
    queue = JobQueueRepository(platform["engine"])
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    step = queue.next_step(fence)
    assert step is not None
    step["checkpoint"] = {
        "generation": "2",
        "pageCount": 0,
        "eventCount": 0,
        "pageTotal": 2,
        "eventTotal": 2,
        "coverage": 0.0,
    }

    class UnusedVectorStore:
        def publish_batches(self, **_kwargs):
            pytest.fail("malformed checkpoint reached the vector store")

    service = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeDerivedAlgorithms(),
        vector_store=UnusedVectorStore(),  # type: ignore[arg-type]
    )
    with pytest.raises(InsightConflict, match="generation must be an integer"):
        service.handle(fence, step)


def test_embedding_batches_reject_malformed_vectors(insight_platform) -> None:
    platform = insight_platform

    class InvalidEmbeddingAlgorithms(FakeDerivedAlgorithms):
        def embed_documents(self, documents, *, config):
            return [[1.0], [1.0, 2.0]]

    service = InsightDerivedWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=JobQueueRepository(platform["engine"]),
        algorithms=InvalidEmbeddingAlgorithms(),
        vector_store=FakeVectorStore(),
    )
    records = [
        {"id": "one", "document": "one", "metadata": {}},
        {"id": "two", "document": "two", "metadata": {}},
    ]
    with pytest.raises(InsightConflict, match="dimensions do not match"):
        list(service._embedding_batches(records, config={}))


@pytest.mark.parametrize(
    "result",
    [
        {
            "mode": "enhanced",
            "content": {},
            "events": ["not-an-object"],
            "characters": [],
        },
        {
            "mode": "enhanced",
            "content": {},
            "events": [{"summary": "event"}],
            "characters": [{"name": ""}],
        },
    ],
)
def test_timeline_publication_rejects_malformed_entries(
    insight_platform,
    result,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={"bookId": str(platform["book"]["id"]), "scope": "full"},
        idempotency_key=f"invalid-timeline-{uuid.uuid4()}",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    repository = InsightDerivedRepository(platform["engine"])
    frozen = repository.snapshot(book_id=str(platform["book"]["id"]))

    with platform["engine"].begin() as connection:
        with pytest.raises(InsightConflict, match="timeline"):
            repository.publish_timeline(
                connection=connection,
                frozen=frozen,
                result=result,
            )


def test_timeline_publication_rejects_unstable_plot_identities(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={"bookId": str(platform["book"]["id"]), "scope": "full"},
        idempotency_key=f"invalid-timeline-content-{uuid.uuid4()}",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    repository = InsightDerivedRepository(platform["engine"])
    frozen = repository.snapshot(book_id=str(platform["book"]["id"]))
    invalid_collections = [
        {
            "plot_arcs": [
                {"name": "missing id", "page_range": {"start": 1, "end": 1}}
            ]
        },
        {
            "plot_arcs": [
                {
                    "id": "arc-1",
                    "name": "first",
                    "description": "first description",
                    "page_range": {"start": 1, "end": 1},
                },
                {
                    "id": "arc-1",
                    "name": "second",
                    "description": "second description",
                    "page_range": {"start": 1, "end": 1},
                },
            ]
        },
        {
            "plot_threads": [
                {"id": "thread-1", "name": "thread", "type": "clue"}
            ]
        },
    ]

    for collections in invalid_collections:
        result = {
            "mode": "enhanced",
            "content": {
                "story_summary": "story",
                "requested_mode": "enhanced",
                "actual_mode": "enhanced",
                "fallback_reason": None,
                "degraded": False,
                **collections,
            },
            "events": [
                {
                    "summary": "event",
                    "page_ids": [frozen.pages[0]["pageId"]],
                }
            ],
            "characters": [],
        }
        with platform["engine"].begin() as connection:
            with pytest.raises(InsightConflict, match="timeline plot"):
                repository.publish_timeline(
                    connection=connection,
                    frozen=frozen,
                    result=result,
                )


@pytest.mark.parametrize(
    "event",
    [
        {
            "eventId": "provider-owned-id",
            "summary": "event",
            "page_ids": [],
            "page_numbers": [1],
        },
        {
            "summary": "event",
            "page_ids": ["unknown-page"],
        },
        {
            "summary": "event",
            "page_numbers": [999],
        },
    ],
)
def test_timeline_publication_rejects_reserved_or_unknown_page_references(
    insight_platform,
    event,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key=f"invalid-timeline-reference-{uuid.uuid4()}",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    repository = InsightDerivedRepository(platform["engine"])
    frozen = repository.snapshot(book_id=str(platform["book"]["id"]))
    result = {
        "mode": "enhanced",
        "content": {
            "story_summary": "story",
            "requested_mode": "enhanced",
            "actual_mode": "enhanced",
            "fallback_reason": None,
            "degraded": False,
        },
        "events": [event],
        "characters": [],
    }

    with platform["engine"].begin() as connection:
        with pytest.raises(InsightConflict, match="timeline"):
            repository.publish_timeline(
                connection=connection,
                frozen=frozen,
                result=result,
            )


def test_vector_store_reports_and_removes_only_unowned_collections(
    insight_platform,
    isolated_chromadb_modules,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    store = InsightVectorStore(platform["data_root"])
    assert store.path == platform["data_root"] / "chroma"
    store.publish(
        book_id=book_id,
        generation=1,
        page_records=(),
        page_embeddings=(),
        event_records=(),
        event_embeddings=(),
    )
    store.publish(
        book_id=book_id,
        generation=2,
        page_records=(),
        page_embeddings=(),
        event_records=(),
        event_embeddings=(),
    )
    with platform["engine"].begin() as connection:
        connection.execute(
            vector_generations.insert().values(
                id=str(uuid.uuid4()),
                book_id=book_id,
                generation=1,
                status="ready",
                dependency_fingerprint="a" * 64,
                is_active=True,
            )
        )

    expected = set(store.names(book_id, 1))
    orphaned = set(store.names(book_id, 2))
    inspection = store.inspect_collections(platform["engine"])
    assert set(inspection.expected) == expected
    assert set(inspection.missing) == set()
    assert set(inspection.orphaned) == orphaned

    assert store.collect_orphan_collections(platform["engine"]) == 2
    after = store.inspect_collections(platform["engine"])
    assert set(after.actual) == expected
    assert after.missing == ()
    assert after.orphaned == ()


def test_page_analysis_schema_requires_current_exact_page_result() -> None:
    normalized = normalize_page_analysis(
        {
            "pages": [
                {
                    "page_number": 1,
                    "page_summary": "当前页面",
                    "key_events": [],
                    "continuity_notes": "",
                    "warnings": [],
                }
            ]
        },
        page_id="page",
        source_asset_id="asset",
        source_checksum="0" * 64,
        page_number=1,
    )
    assert normalized["page_number_snapshot"] == 1
    assert normalized["page_summary"] == "当前页面"

    with pytest.raises(InvalidPageAnalysis, match="page_number must be 1"):
        normalize_page_analysis(
            {
                "pages": [
                    {
                        "page_number": 2,
                        "page_summary": "wrong page 2",
                        "key_events": [],
                        "continuity_notes": "",
                        "warnings": [],
                    },
                ]
            },
            page_id="page",
            source_asset_id="asset",
            source_checksum="0" * 64,
            page_number=1,
        )
    with pytest.raises(InvalidPageAnalysis, match="contain only pages"):
        normalize_page_analysis(
            {"page_summary": ""},
            page_id="page",
            source_asset_id="asset",
            source_checksum="0" * 64,
            page_number=1,
        )


def test_page_analysis_schema_rejects_legacy_defaults_and_allows_large_results() -> None:
    base_page = {
        "page_number": 1,
        "page_summary": "摘要",
        "key_events": [],
        "continuity_notes": "",
        "warnings": [],
    }
    for invalid in (
        {"pages": [{**base_page, "scene": "legacy"}]},
        {"pages": [{key: value for key, value in base_page.items() if key != "warnings"}]},
        {
            "pages": [
                {
                    **base_page,
                    "key_events": [{"summary": "事件", "importance": "unexpected"}],
                }
            ]
        },
    ):
        with pytest.raises(InvalidPageAnalysis):
            normalize_page_analysis(
                invalid,
                page_id="page",
                source_asset_id="asset",
                source_checksum="0" * 64,
                page_number=1,
            )

    events = [
        {"summary": f"事件 {index}", "importance": "normal"}
        for index in range(101)
    ]
    result = normalize_page_analysis(
        {
            "pages": [
                {
                    **base_page,
                    "page_summary": "摘要" * 20_001,
                    "key_events": events,
                }
            ]
        },
        page_id="page",
        source_asset_id="asset",
        source_checksum="0" * 64,
        page_number=1,
    )
    assert len(result["key_events"]) == 101
    assert len(result["page_summary"]) > 20_000


def test_browser_cannot_supply_provider_or_prompt_configuration() -> None:
    with pytest.raises(ValueError, match="unknown Insight command fields"):
        from src.backend_v2.insight.commands import normalize_analysis_command

        normalize_analysis_command(
            {
                "bookId": "book",
                "scope": "full",
                "vlm": {"apiKey": "must-not-enter-command"},
            }
        )


@pytest.mark.parametrize(
    ("command", "message"),
    (
        ({"bookId": 7, "scope": "full"}, "bookId is required"),
        ({"bookId": "book", "scope": 7}, "scope must be"),
        (
            {"bookId": "book", "scope": "full", "force": 1},
            "unknown Insight command fields",
        ),
        (
            {"bookId": "book", "scope": "chapter", "chapterIds": [7]},
            "chapterIds must contain non-empty strings",
        ),
    ),
)
def test_analysis_command_rejects_scalar_coercion(
    command: dict[str, object],
    message: str,
) -> None:
    from src.backend_v2.insight.commands import normalize_analysis_command

    with pytest.raises(ValueError, match=message):
        normalize_analysis_command(command)


def test_analysis_replay_requires_current_run_id(
    insight_platform,
    monkeypatch,
) -> None:
    service = InsightAnalysisCommandService(insight_platform["engine"])
    monkeypatch.setattr(
        service.jobs,
        "idempotency_replay",
        lambda **_kwargs: {"jobIds": [str(uuid.uuid4())]},
    )

    with pytest.raises(ValueError, match="Insight replay runId"):
        service.create_analysis_job(
            command={
                "bookId": str(insight_platform["book"]["id"]),
                "scope": "full",
            },
            idempotency_key="retired-replay-shape",
        )


def test_analysis_job_admission_rejects_a_raced_page_order(
    insight_platform,
    monkeypatch,
) -> None:
    platform = insight_platform
    service = InsightAnalysisCommandService(platform["engine"])
    original_create_batch = service.jobs.create_batch

    def create_after_reorder(**kwargs):
        content = ContentRepository(platform["engine"])
        chapter = content.list_chapters(str(platform["book"]["id"]))[
            "chapters"
        ][0]
        content.reorder_pages(
            chapter_id=str(chapter["id"]),
            ordered_ids=list(reversed(platform["page_ids"])),
            base_revision=int(chapter["pageOrderRevision"]),
        )
        return original_create_batch(**kwargs)

    monkeypatch.setattr(service.jobs, "create_batch", create_after_reorder)
    with pytest.raises(InsightConflict, match="targets changed"):
        service.create_analysis_job(
            command={
                "bookId": str(platform["book"]["id"]),
                "scope": "full",
            },
            idempotency_key="analysis-admission-order-race",
        )
    with platform["engine"].connect() as connection:
        assert connection.execute(select(analysis_runs.c.id)).first() is None
        assert connection.execute(
            select(jobs.c.id).where(jobs.c.kind == "insight_analysis")
        ).first() is None


def test_page_run_preview_validates_the_run_and_pending_target(
    insight_platform,
) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="page-preview-pending-run",
    )
    repository = InsightRepository(platform["engine"])
    detail = repository.page_detail(
        page_id=platform["page_ids"][0],
        run_id=str(accepted["runId"]),
    )
    assert detail["preview"] is True
    assert detail["analysisState"] == "running"
    assert detail["analysis"] is None
    with pytest.raises(InsightNotFound, match="analysis run"):
        repository.page_detail(
            page_id=platform["page_ids"][0],
            run_id=str(uuid.uuid4()),
        )


@pytest.mark.parametrize("selector", ("chapterId", "pageId"))
def test_analysis_command_rejects_retired_singular_selectors(
    selector: str,
) -> None:
    from src.backend_v2.insight.commands import normalize_analysis_command

    with pytest.raises(ValueError, match="unknown Insight command fields"):
        normalize_analysis_command(
            {
                "bookId": "book",
                "scope": "chapter" if selector == "chapterId" else "page",
                selector: "target",
            }
        )


def test_queued_full_run_cancel_converges_without_publication(
    insight_platform,
) -> None:
    platform = insight_platform
    accepted = InsightAnalysisCommandService(
        platform["engine"]
    ).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="cancel-before-claim",
    )
    queue = JobQueueRepository(platform["engine"])
    cancelled = queue.request_cancel(str(accepted["jobIds"][0]))
    assert cancelled["status"] == "cancelled"
    run = InsightRepository(platform["engine"]).get_run(
        str(accepted["runId"])
    )
    assert run["status"] == "cancelled"
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(analysis_heads.c.id).where(
                analysis_heads.c.book_id == platform["book"]["id"]
            )
        ).all() == []


def test_continuation_config_is_strict_without_arbitrary_upper_bounds(
    insight_platform,
) -> None:
    platform = insight_platform
    project_id = str(uuid.uuid4())
    with platform["engine"].begin() as connection:
        connection.execute(
            insert(continuation_projects).values(
                id=project_id,
                book_id=platform["book"]["id"],
                revision=1,
                payload_json=json.dumps(
                    {
                        "pageCount": 15,
                        "styleReferencePages": 3,
                        "direction": "",
                        "analysisInputs": [
                            {
                                "resultId": str(uuid.uuid4()),
                                "pageId": platform["page_ids"][0],
                                "pageNumber": 1,
                                "currentSourceChecksum": "0" * 64,
                            }
                        ],
                        "analysisInputFingerprint": "0" * 64,
                    }
                ),
            )
        )
    repository = ContinuationRepository(platform["engine"])

    for invalid in (
        {"pageCount": "2", "styleReferencePages": 3, "direction": ""},
        {"pageCount": True, "styleReferencePages": 3, "direction": ""},
        {"pageCount": 0, "styleReferencePages": 3, "direction": ""},
        {"pageCount": 2, "styleReferencePages": 0, "direction": ""},
        {"pageCount": 2, "styleReferencePages": 3},
        {
            "pageCount": 2,
            "styleReferencePages": 3,
            "direction": "",
            "legacy": True,
        },
    ):
        with pytest.raises(ValueError):
            repository.update_project(
                idempotency_key="invalid-continuation-project-update",
                project_id=project_id,
                base_revision=1,
                config=invalid,
            )

    updated = repository.update_project(
        idempotency_key="valid-continuation-project-update",
        project_id=project_id,
        base_revision=1,
        config={
            "pageCount": 250,
            "styleReferencePages": 50,
            "direction": "继续前进",
        },
    )
    assert repository.update_project(
        idempotency_key="valid-continuation-project-update",
        project_id=project_id,
        base_revision=1,
        config={
            "pageCount": 250,
            "styleReferencePages": 50,
            "direction": "继续前进",
        },
    ) == updated
    with pytest.raises(InsightConflict, match="Idempotency-Key"):
        repository.update_project(
            idempotency_key="valid-continuation-project-update",
            project_id=project_id,
            base_revision=1,
            config={
                "pageCount": 251,
                "styleReferencePages": 50,
                "direction": "继续前进",
            },
        )
    assert updated["config"] == {
        "pageCount": 250,
        "styleReferencePages": 50,
        "direction": "继续前进",
    }
    with platform["engine"].begin() as connection:
        connection.execute(
            insert(continuation_pages),
            [
                {
                    "id": str(uuid.uuid4()),
                    "project_id": project_id,
                    "ordinal": ordinal,
                    "revision": 1,
                    "payload_json": json.dumps(
                        {
                            "storyText": "",
                            "continuityText": "",
                            "dialogueText": "",
                            "characters": [],
                            "finalPrompt": "",
                            "status": "pending",
                        }
                    ),
                }
                for ordinal in (1, 2, 250)
            ],
        )
    shrunk = repository.update_project(
        idempotency_key="shrink-continuation-project",
        project_id=project_id,
        base_revision=updated["revision"],
        config={
            "pageCount": 2,
            "styleReferencePages": 50,
            "direction": "继续前进",
        },
    )
    assert [page["ordinal"] for page in shrunk["pages"]] == [1, 2]
    non_image = AssetStorageService(
        platform["data_root"],
        platform["engine"],
    ).publish_bytes(
        b"not an image",
        extension="bin",
        mime_type="application/octet-stream",
    )
    with pytest.raises(ValueError, match="must be images"):
        repository.set_project_references(
            idempotency_key="invalid-continuation-reference",
            project_id=project_id,
            base_revision=shrunk["revision"],
            asset_ids=[non_image.id],
        )
    with platform["engine"].begin() as connection:
        connection.execute(
            update(continuation_projects)
            .where(continuation_projects.c.id == project_id)
            .values(
                payload_json=json.dumps(
                    {
                        "pageCount": 2,
                        "styleReferencePages": 50,
                        "direction": "旧结构",
                    }
                )
            )
        )
    with pytest.raises(InsightConflict, match="clear the project"):
        repository.bootstrap(book_id=str(platform["book"]["id"]))


def test_continuation_commands_reject_ambiguous_ordinals(
    insight_platform,
) -> None:
    platform = insight_platform
    project_id = str(uuid.uuid4())
    with platform["engine"].begin() as connection:
        connection.execute(
            insert(continuation_projects).values(
                id=project_id,
                book_id=platform["book"]["id"],
                revision=1,
                payload_json=json.dumps(
                    {
                        "pageCount": 2,
                        "styleReferencePages": 3,
                        "direction": "",
                        "analysisInputs": [
                            {
                                "resultId": str(uuid.uuid4()),
                                "pageId": platform["page_ids"][0],
                                "pageNumber": 1,
                                "currentSourceChecksum": "0" * 64,
                            }
                        ],
                        "analysisInputFingerprint": "0" * 64,
                    }
                ),
            )
        )
    commands = ContinuationCommandService(platform["engine"])

    for invalid in ([], [1, 1], [0], [True]):
        with pytest.raises(ValueError, match="ordinals"):
            commands.create_pages_job(
                book_id=str(platform["book"]["id"]),
                ordinals=invalid,
                idempotency_key=f"invalid-pages-{invalid}",
            )
        with pytest.raises(ValueError, match="ordinals"):
            commands.create_images_job(
                book_id=str(platform["book"]["id"]),
                ordinals=invalid,
                idempotency_key=f"invalid-images-{invalid}",
            )

    with pytest.raises(ValueError, match="out of range"):
        commands.create_pages_job(
            book_id=str(platform["book"]["id"]),
            ordinals=[3],
            idempotency_key="invalid-pages-range",
        )
    with pytest.raises(ValueError, match="out of range"):
        commands.create_images_job(
            book_id=str(platform["book"]["id"]),
            ordinals=[1],
            idempotency_key="invalid-images-range",
        )


def test_continuation_http_commands_reject_missing_or_ignored_fields(
    insight_platform,
) -> None:
    platform = insight_platform
    app = Flask("continuation-validation-test")
    app.register_blueprint(
        create_insight_blueprint(
            engine=platform["engine"],
            data_root=platform["data_root"],
        )
    )
    client = app.test_client()
    headers = {"Idempotency-Key": "strict-continuation-command"}
    requests = (
        client.patch(
            "/api/v2/insight/continuation/projects/project-id",
            headers=headers,
            json={
                "baseRevision": "1",
                "config": {
                    "pageCount": 1,
                    "styleReferencePages": 1,
                    "direction": "",
                },
            },
        ),
        client.post(
            "/api/v2/insight/continuation/projects/project-id/characters",
            headers=headers,
            json={"name": "主角", "aliases": [], "payload": {}},
        ),
        client.post(
            f"/api/v2/insight/books/{platform['book']['id']}/continuation/jobs",
            headers=headers,
            json={"kind": "pages", "ordinals": []},
        ),
        client.post(
            f"/api/v2/insight/books/{platform['book']['id']}/continuation/jobs",
            headers=headers,
            json={"kind": "script", "format": "zip"},
        ),
        client.post(
            f"/api/v2/insight/books/{platform['book']['id']}/continuation/jobs",
            headers=headers,
            json={"kind": "export"},
        ),
        client.patch(
            "/api/v2/insight/continuation/forms/form-id",
            headers=headers,
            json={"baseRevision": 1, "name": "常服"},
        ),
    )

    assert [response.status_code for response in requests] == [422] * len(requests)
    assert all(
        response.get_json()["error"]["code"] == "validation_error"
        for response in requests
    )


@pytest.mark.parametrize(
    "response",
    (
        {
            "storyText": "剧情",
            "continuityText": "",
            "dialogueText": "",
            "characters": [],
            "finalPrompt": "",
        },
        {
            "storyText": "剧情",
            "continuityText": "",
            "dialogueText": "",
            "characters": "主角",
            "finalPrompt": "prompt",
        },
    ),
)
def test_continuation_page_generation_rejects_malformed_provider_results(
    monkeypatch,
    response: Mapping[str, Any],
) -> None:
    monkeypatch.setattr(
        ProviderDerivedAlgorithms,
        "_chat_json",
        staticmethod(lambda *_args, **_kwargs: response),
    )

    with pytest.raises(ValueError, match="continuation page response"):
        DefaultContinuationAlgorithms().generate_page(
            ordinal=1,
            script="script",
            previous=None,
            config={},
        )


def test_continuation_dtos_keep_every_generated_image_version(
    insight_platform,
) -> None:
    platform = insight_platform
    project_id = str(uuid.uuid4())
    character_id = str(uuid.uuid4())
    form_id = str(uuid.uuid4())
    continuation_page_id = str(uuid.uuid4())
    storage = AssetStorageService(platform["data_root"], platform["engine"])
    generated_assets = [
        storage.publish_bytes(
            f"version-{version}".encode(),
            extension="bin",
            mime_type="application/octet-stream",
        )
        for version in range(1, 7)
    ]
    with platform["engine"].begin() as connection:
        connection.execute(
            insert(continuation_projects).values(
                id=project_id,
                book_id=platform["book"]["id"],
                revision=1,
                payload_json=json.dumps(
                    {
                        "pageCount": 1,
                        "styleReferencePages": 1,
                        "direction": "",
                        "analysisInputs": [
                            {
                                "resultId": str(uuid.uuid4()),
                                "pageId": platform["page_ids"][0],
                                "pageNumber": 1,
                                "currentSourceChecksum": "0" * 64,
                            }
                        ],
                        "analysisInputFingerprint": "0" * 64,
                    }
                ),
            )
        )
        connection.execute(
            insert(continuation_characters).values(
                id=character_id,
                project_id=project_id,
                name="主角",
                aliases_json="[]",
                payload_json="{}",
                revision=1,
            )
        )
        connection.execute(
            insert(continuation_character_forms).values(
                id=form_id,
                character_id=character_id,
                name="常服",
                payload_json="{}",
                revision=1,
            )
        )
        connection.execute(
            insert(continuation_pages).values(
                id=continuation_page_id,
                project_id=project_id,
                ordinal=1,
                revision=1,
                payload_json=json.dumps(
                    {
                        "storyText": "",
                        "continuityText": "",
                        "dialogueText": "",
                        "characters": [],
                        "finalPrompt": "",
                        "status": "pending",
                    }
                ),
            )
        )
        connection.execute(
            insert(continuation_form_image_versions),
            [
                {
                    "id": str(uuid.uuid4()),
                    "form_id": form_id,
                    "asset_id": asset.id,
                    "thumbnail_asset_id": asset.id,
                    "version": version,
                    "is_adopted": version == 1,
                }
                for version, asset in enumerate(generated_assets, start=1)
            ],
        )
        connection.execute(
            insert(continuation_image_versions),
            [
                {
                    "id": str(uuid.uuid4()),
                    "continuation_page_id": continuation_page_id,
                    "asset_id": asset.id,
                    "thumbnail_asset_id": asset.id,
                    "version": version,
                    "is_active": version == 6,
                }
                for version, asset in enumerate(generated_assets, start=1)
            ],
        )

    repository = ContinuationRepository(platform["engine"])
    forms = repository.list_forms(project_id=project_id)["items"]
    project = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]

    assert [value["version"] for value in forms[0]["imageVersions"]] == [
        6,
        5,
        4,
        3,
        2,
        1,
    ]
    assert [
        value["version"] for value in project["pages"][0]["imageVersions"]
    ] == [6, 5, 4, 3, 2, 1]


def test_continuation_script_and_page_loops_are_worker_owned(
    insight_platform,
    monkeypatch,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="continuation-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    repository = ContinuationRepository(platform["engine"])
    project = repository.sync_latest(
        idempotency_key="continuation-sync-initial",
        book_id=str(platform["book"]["id"])
    )
    assert repository.sync_latest(
        idempotency_key="continuation-sync-initial",
        book_id=str(platform["book"]["id"]),
    ) == project
    original_run_id = project["sourceRunId"]
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="continuation-newer-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    project = repository.update_project(
        idempotency_key="continuation-config-update",
        project_id=str(project["projectId"]),
        base_revision=int(project["revision"]),
        config={
            "pageCount": 2,
            "styleReferencePages": 2,
            "direction": "继续冒险",
        },
    )
    commands = ContinuationCommandService(platform["engine"])
    queue = JobQueueRepository(platform["engine"])
    algorithms = FakeContinuationAlgorithms()
    worker = ContinuationWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=algorithms,
    )
    script_job = commands.create_script_job(
        book_id=str(platform["book"]["id"]),
        idempotency_key="continuation-script",
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    assert project["sourceRunId"] == original_run_id
    assert {
        "overview:story_summary",
        "compressed_context:default",
    }.issubset(algorithms.script_contexts[0]["artifacts"])

    script_before_race = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]["script"]
    original_create_batch = commands.jobs.create_batch

    def create_batch_after_script_edit(**kwargs):
        repository.update_script(
            idempotency_key="continuation-script-race-edit",
            project_id=str(project["projectId"]),
            base_revision=int(script_before_race["revision"]),
            content=str(script_before_race["content"]),
        )
        return original_create_batch(**kwargs)

    monkeypatch.setattr(
        commands.jobs,
        "create_batch",
        create_batch_after_script_edit,
    )
    with pytest.raises(InsightConflict, match="script changed"):
        commands.create_pages_job(
            book_id=str(platform["book"]["id"]),
            ordinals=None,
            idempotency_key="continuation-pages-raced",
        )
    monkeypatch.setattr(
        commands.jobs,
        "create_batch",
        original_create_batch,
    )
    with platform["engine"].connect() as connection:
        assert list(connection.execute(select(continuation_pages.c.id))) == []

    initial_pages_job = commands.create_pages_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-pages",
    )
    pending_pages = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]["pages"]
    assert [page["payload"] for page in pending_pages] == [
        {
            "storyText": "",
            "continuityText": "",
            "dialogueText": "",
            "characters": [],
            "finalPrompt": "",
            "status": "pending",
        }
    ] * 2
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    bootstrap_statements: list[str] = []

    def record_bootstrap_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        bootstrap_statements.append(statement.upper())

    event.listen(
        platform["engine"],
        "before_cursor_execute",
        record_bootstrap_statement,
    )
    try:
        restored = repository.bootstrap(
            book_id=str(platform["book"]["id"])
        )["project"]
    finally:
        event.remove(
            platform["engine"],
            "before_cursor_execute",
            record_bootstrap_statement,
        )
    assert sum(
        "FROM CONTINUATION_IMAGE_VERSIONS" in statement
        for statement in bootstrap_statements
    ) == 1
    assert restored["script"]["content"].startswith("第1页")
    assert [page["payload"]["storyText"] for page in restored["pages"]] == [
        "第 1 页剧情",
        "第 2 页剧情",
    ]
    original_page_revisions = [
        int(page["revision"]) for page in restored["pages"]
    ]
    skip_job = commands.create_pages_job(
        book_id=str(platform["book"]["id"]),
        ordinals=[1],
        idempotency_key="continuation-pages-existing-content",
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    skipped_detail = queue.get_job(str(skip_job["jobIds"][0]))
    assert skipped_detail["counts"]["skipped"] == 1
    assert skipped_detail["counts"]["failed"] == 0
    commands.create_script_job(
        book_id=str(platform["book"]["id"]),
        idempotency_key="continuation-script-regenerate",
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    assert commands.create_pages_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-pages",
    ) == initial_pages_job
    assert commands.create_script_job(
        book_id=str(platform["book"]["id"]),
        idempotency_key="continuation-script",
    ) == script_job
    script_refreshed = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]
    assert [
        page["payload"]["storyText"] for page in script_refreshed["pages"]
    ] == ["第 1 页剧情", "第 2 页剧情"]
    assert [
        page["payload"]["staleReason"]
        for page in script_refreshed["pages"]
    ] == ["script_changed", "script_changed"]
    assert [
        int(page["revision"]) for page in script_refreshed["pages"]
    ] == [revision + 1 for revision in original_page_revisions]

    pages_job = commands.create_pages_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-pages-after-script",
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    assert commands.create_pages_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-pages-after-script",
    ) == pages_job
    editable_page = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]["pages"][0]
    page_update_input = {
        "idempotency_key": "continuation-page-manual-edit",
        "page_id": str(editable_page["continuationPageId"]),
        "base_revision": int(editable_page["revision"]),
        "payload": {
            **editable_page["payload"],
            "finalPrompt": editable_page["payload"]["finalPrompt"] + "，近景",
        },
    }
    updated_page = repository.update_page(**page_update_input)
    assert repository.update_page(**page_update_input) == updated_page
    images_job = commands.create_images_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-images",
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    first_image_step = queue.next_step(fence)
    assert first_image_step is not None
    assert first_image_step["itemOrdinal"] == 1
    assert worker.handle(fence, first_image_step)["__already_published__"]
    second_image_step = queue.next_step(fence)
    assert second_image_step is not None
    assert second_image_step["itemOrdinal"] == 2
    second_target = second_image_step["config"]["targets"][1]
    candidate_ids = worker._reference_window_asset_ids(
        project_id=str(second_image_step["config"]["projectId"]),
        before_ordinal=int(second_target["ordinal"]),
        count=int(
            second_image_step["config"]["projectConfig"][
                "styleReferencePages"
            ]
        ),
        initial_asset_ids=[
            str(value)
            for value in second_image_step["config"][
                "initialReferenceAssetIds"
            ]
        ],
    )
    assert len(candidate_ids) == 2
    reference_roles = {
        f"continuation_reference_{index:03d}": asset_id
        for index, asset_id in enumerate(candidate_ids, start=1)
    }
    frozen = queue.bind_explicit_item_inputs(
        fence,
        item_id=str(second_image_step["itemId"]),
        assets_by_role=reference_roles,
    )
    rebound = queue.bind_explicit_item_inputs(
        fence,
        item_id=str(second_image_step["itemId"]),
        assets_by_role={
            role: candidate_ids[-index]
            for index, role in enumerate(reference_roles, start=1)
        },
    )
    assert {
        role: value["id"] for role, value in rebound.items()
    } == {
        role: value["id"] for role, value in frozen.items()
    }
    assert worker.handle(fence, second_image_step)["__already_published__"]
    assert queue.next_step(fence) is None
    assert queue.finish_if_complete(fence) == "completed"
    assert commands.create_images_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-images",
    ) == images_job
    generated_page = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]["pages"][0]
    activation_input = {
        "idempotency_key": "continuation-image-activate",
        "continuation_page_id": str(generated_page["continuationPageId"]),
        "version": 1,
    }
    activated = repository.switch_image_version(**activation_input)
    assert repository.switch_image_version(**activation_input) == activated

    exported = commands.create_export_job(
        book_id=str(platform["book"]["id"]),
        output_format="zip",
        idempotency_key="continuation-export",
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    assert commands.create_export_job(
        book_id=str(platform["book"]["id"]),
        output_format="zip",
        idempotency_key="continuation-export",
    ) == exported
    detail = queue.get_job(str(exported["jobIds"][0]))
    assert detail["artifacts"][0]["kind"] == "continuation_export"

    before_manual_edit = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]
    script_update_input = {
        "idempotency_key": "continuation-script-manual-edit",
        "project_id": str(before_manual_edit["projectId"]),
        "base_revision": int(before_manual_edit["script"]["revision"]),
        "content": "手工改写后的脚本",
    }
    updated_script = repository.update_script(**script_update_input)
    assert repository.update_script(**script_update_input) == updated_script
    after_manual_edit = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]
    assert [
        page["payload"]["storyText"] for page in after_manual_edit["pages"]
    ] == ["第 1 页剧情", "第 2 页剧情"]
    assert all(
        page["payload"]["staleReason"] == "script_changed"
        for page in after_manual_edit["pages"]
    )
    assert [page["revision"] for page in after_manual_edit["pages"]] == [
        int(page["revision"]) + 1 for page in before_manual_edit["pages"]
    ]


def test_continuation_freezes_a_composed_page_analysis_snapshot(
    insight_platform,
) -> None:
    platform = insight_platform
    book_id = str(platform["book"]["id"])
    for index, page_id in enumerate(platform["page_ids"], start=1):
        InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
            command={
                "bookId": book_id,
                "scope": "page",
                "pageIds": [page_id],
            },
            idempotency_key=f"continuation-page-analysis-{index}",
        )
        assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    derived = InsightDerivedCommandService(platform["engine"])
    for kind, template in (
        ("overview", "story_summary"),
        ("compressed_context", "default"),
        ("timeline", "default"),
    ):
        derived.create_job(
            book_id=book_id,
            kind=kind,
            template=template,
            idempotency_key=f"continuation-page-{kind}-{template}",
        )
        assert (
            _run_derived_job(
                platform,
                algorithms=FakeDerivedAlgorithms(),
            )
            == "completed"
        )

    snapshot = InsightDerivedRepository(platform["engine"]).snapshot(
        book_id=book_id
    )
    assert snapshot.source_run_id is None
    repository = ContinuationRepository(platform["engine"])
    state = repository.bootstrap(book_id=book_id)
    assert state["ready"]
    assert state["missing"] == []
    assert state["activeRunId"] is None

    project = repository.sync_latest(
        idempotency_key="continuation-page-snapshot-sync",
        book_id=book_id,
    )
    assert project["sourceRunId"] is None
    assert project["config"] == {
        "pageCount": 15,
        "styleReferencePages": 3,
        "direction": "",
    }
    with platform["engine"].connect() as connection:
        stored_payload = json.loads(
            connection.execute(
                select(continuation_projects.c.payload_json).where(
                    continuation_projects.c.id == project["projectId"]
                )
            ).scalar_one()
        )
    assert stored_payload["analysisInputFingerprint"] == snapshot.fingerprint
    assert [
        value["resultId"] for value in stored_payload["analysisInputs"]
    ] == list(snapshot.result_ids)

    accepted = ContinuationCommandService(
        platform["engine"]
    ).create_script_job(
        book_id=book_id,
        idempotency_key="continuation-page-script",
    )
    with platform["engine"].connect() as connection:
        job = connection.execute(
            select(jobs).where(jobs.c.id == accepted["jobIds"][0])
        ).mappings().one()
    assert job["analysis_run_id"] is None

    queue = JobQueueRepository(platform["engine"])
    algorithms = FakeContinuationAlgorithms()
    worker = ContinuationWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=algorithms,
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    assert len(algorithms.script_contexts[0]["pages"]) == len(
        platform["page_ids"]
    )
    assert {
        "overview:story_summary",
        "compressed_context:default",
    }.issubset(algorithms.script_contexts[0]["artifacts"])


def test_continuation_character_forms_and_sheet_are_versioned(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="form-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    repository = ContinuationRepository(platform["engine"])
    project = repository.sync_latest(
        idempotency_key="continuation-form-sync",
        book_id=str(platform["book"]["id"])
    )
    assert repository.sync_latest(
        idempotency_key="continuation-form-sync",
        book_id=str(platform["book"]["id"]),
    ) == project
    character = repository.create_character(
        idempotency_key="continuation-character-create",
        project_id=project["projectId"],
        name="Alter",
        aliases=["黑化"],
        enabled=True,
        payload={"description": "盔甲形态"},
    )
    assert repository.create_character(
        idempotency_key="continuation-character-create",
        project_id=project["projectId"],
        name="Alter",
        aliases=["黑化"],
        enabled=True,
        payload={"description": "盔甲形态"},
    ) == character
    form = repository.create_form(
        idempotency_key="continuation-form-create-combat",
        character_id=character["characterId"],
        name="战斗服",
        payload={"colors": ["black", "red"]},
    )
    assert repository.create_form(
        idempotency_key="continuation-form-create-combat",
        character_id=character["characterId"],
        name="战斗服",
        payload={"colors": ["black", "red"]},
    ) == form
    casual_form = repository.create_form(
        idempotency_key="continuation-form-create-casual",
        character_id=character["characterId"],
        name="常服",
        payload={"colors": ["white"]},
    )
    with platform["engine"].connect() as connection:
        reference_asset_id = str(
            connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == platform["page_ids"][0],
                    page_assets.c.role == "source",
                )
            ).scalar_one()
        )
        reference_thumbnail_id = str(
            connection.execute(
                select(page_assets.c.asset_id).where(
                    page_assets.c.page_id == platform["page_ids"][0],
                    page_assets.c.role == "thumbnail_source",
                )
            ).scalar_one()
        )
        reference_checksum = str(
            connection.execute(
                select(assets.c.checksum).where(
                    assets.c.id == reference_asset_id
                )
            ).scalar_one()
        )
    form = repository.bind_form_reference(
        idempotency_key="continuation-form-bind-reference",
        form_id=form["formId"],
        base_revision=form["revision"],
        asset_id=reference_asset_id,
        thumbnail_asset_id=reference_thumbnail_id,
        content_checksum=reference_checksum,
    )
    assert repository.bind_form_reference(
        idempotency_key="continuation-form-bind-reference",
        form_id=form["formId"],
        base_revision=1,
        asset_id=reference_asset_id,
        thumbnail_asset_id=reference_thumbnail_id,
        content_checksum=reference_checksum,
    ) == form
    assert repository.replay_form_reference_upload(
        idempotency_key="continuation-form-bind-reference",
        form_id=form["formId"],
        base_revision=1,
        content_checksum=reference_checksum,
    ) == form
    project = repository.set_project_references(
        idempotency_key="continuation-project-references",
        project_id=project["projectId"],
        base_revision=project["revision"],
        asset_ids=[reference_asset_id],
    )
    assert repository.set_project_references(
        idempotency_key="continuation-project-references",
        project_id=project["projectId"],
        base_revision=1,
        asset_ids=[reference_asset_id],
    ) == project
    assert project["referenceAssets"][0]["thumbnailUrl"].endswith(
        reference_thumbnail_id
    )
    commands = ContinuationCommandService(platform["engine"])
    accepted = commands.create_character_sheet_job(
        book_id=str(platform["book"]["id"]),
        form_id=form["formId"],
        idempotency_key="character-sheet",
    )
    queue = JobQueueRepository(platform["engine"])
    worker = ContinuationWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeContinuationAlgorithms(),
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    form_list_statements: list[str] = []

    def record_form_list_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        form_list_statements.append(statement.upper())

    event.listen(
        platform["engine"],
        "before_cursor_execute",
        record_form_list_statement,
    )
    try:
        listed_forms = repository.list_forms(
            project_id=project["projectId"]
        )["items"]
    finally:
        event.remove(
            platform["engine"],
            "before_cursor_execute",
            record_form_list_statement,
        )
    assert sum(
        "FROM CONTINUATION_FORM_IMAGE_VERSIONS" in statement
        for statement in form_list_statements
    ) == 1
    generated = next(
        item for item in listed_forms if item["formId"] == form["formId"]
    )
    assert generated["imageVersions"][0]["thumbnailUrl"]
    adopted = repository.adopt_form_image(
        idempotency_key="continuation-form-adopt-image",
        form_id=form["formId"],
        version=1,
        base_revision=generated["revision"],
    )
    assert repository.adopt_form_image(
        idempotency_key="continuation-form-adopt-image",
        form_id=form["formId"],
        version=1,
        base_revision=generated["revision"],
    ) == adopted
    assert commands.create_character_sheet_job(
        book_id=str(platform["book"]["id"]),
        form_id=form["formId"],
        idempotency_key="character-sheet",
    ) == accepted
    assert adopted["version"] == 1
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(continuation_form_image_versions.c.is_adopted)
        ).scalar_one()
        job = connection.execute(
            select(jobs).where(jobs.c.id == accepted["jobIds"][0])
        ).mappings().one()
        assert job["continuation_project_id"] == project["projectId"]
        assert job["analysis_run_id"] == project["sourceRunId"]

    character_update_input = {
        "idempotency_key": "continuation-character-update",
        "character_id": character["characterId"],
        "base_revision": 1,
        "name": "Alter",
        "aliases": ["黑化", "另一面"],
        "enabled": False,
        "payload": {"description": "更新后的盔甲形态"},
    }
    updated_character = repository.update_character(**character_update_input)
    assert repository.update_character(**character_update_input) == updated_character

    form_update_input = {
        "idempotency_key": "continuation-form-update",
        "form_id": form["formId"],
        "base_revision": adopted["revision"],
        "name": "决战服",
        "payload": {"colors": ["black", "gold"]},
    }
    updated_form = repository.update_form(**form_update_input)
    assert repository.update_form(**form_update_input) == updated_form

    form_delete_input = {
        "idempotency_key": "continuation-form-delete",
        "form_id": casual_form["formId"],
        "base_revision": casual_form["revision"],
    }
    repository.delete_form(**form_delete_input)
    repository.delete_form(**form_delete_input)

    character_delete_input = {
        "idempotency_key": "continuation-character-delete",
        "character_id": character["characterId"],
        "base_revision": updated_character["revision"],
    }
    repository.delete_character(**character_delete_input)
    repository.delete_character(**character_delete_input)

    clear_input = {
        "idempotency_key": "continuation-project-clear",
        "book_id": str(platform["book"]["id"]),
    }
    repository.clear(**clear_input)
    repository.clear(**clear_input)


def test_insight_export_job_freezes_run_and_builds_backend_zip(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="export-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    content = ContentRepository(platform["engine"])
    chapter = content.list_chapters(str(platform["book"]["id"]))[
        "chapters"
    ][0]
    content.reorder_pages(
        chapter_id=str(chapter["id"]),
        ordered_ids=list(reversed(platform["page_ids"])),
        base_revision=int(chapter["pageOrderRevision"]),
    )
    export_commands = InsightExportCommandService(platform["engine"])
    accepted = export_commands.create_export_job(
        book_id=str(platform["book"]["id"]),
        idempotency_key="insight-export",
    )
    queue = JobQueueRepository(platform["engine"])
    worker = InsightExportWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"
    detail = queue.get_job(str(accepted["jobIds"][0]))
    artifact_id = detail["artifacts"][0]["assetId"]
    with platform["engine"].connect() as connection:
        relative_path = connection.execute(
            select(assets.c.relative_path).where(assets.c.id == artifact_id)
        ).scalar_one()
    path = platform["data_root"] / str(relative_path)
    with zipfile.ZipFile(path) as archive:
        assert {
            "manifest.json",
            "pages.json",
            "layers.json",
            "timeline.json",
            "report.md",
        }.issubset(archive.namelist())
        exported_pages = json.loads(archive.read("pages.json"))
    assert [page["pageId"] for page in exported_pages] == list(
        reversed(platform["page_ids"])
    )
    assert [page["pageNumber"] for page in exported_pages] == [1, 2]
    assert [
        page["analysis"]["page_number_snapshot"]
        for page in exported_pages
    ] == [2, 1]
    with platform["engine"].begin() as connection:
        connection.execute(
            delete(analysis_heads).where(
                analysis_heads.c.book_id == platform["book"]["id"],
                analysis_heads.c.page_id.is_(None),
            )
        )
    assert export_commands.create_export_job(
        book_id=str(platform["book"]["id"]),
        idempotency_key="insight-export",
    ) == accepted
    with pytest.raises(InsightNotFound, match="已发布"):
        export_commands.create_export_job(
            book_id=str(platform["book"]["id"]),
            idempotency_key="insight-export-after-head-removed",
        )


def test_insight_export_admission_rejects_a_raced_analysis_head(
    insight_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="export-race-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    export_commands = InsightExportCommandService(platform["engine"])
    create_batch = export_commands.jobs.create_batch

    def create_batch_after_head_removal(**kwargs):
        with platform["engine"].begin() as connection:
            connection.execute(
                delete(analysis_heads).where(
                    analysis_heads.c.book_id == platform["book"]["id"],
                    analysis_heads.c.page_id.is_(None),
                )
            )
        return create_batch(**kwargs)

    monkeypatch.setattr(
        export_commands.jobs,
        "create_batch",
        create_batch_after_head_removal,
    )

    with pytest.raises(InsightConflict, match="changed before export admission"):
        export_commands.create_export_job(
            book_id=str(platform["book"]["id"]),
            idempotency_key="insight-export-raced-head",
        )

    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(jobs.c.id).where(jobs.c.kind == "insight_export")
        ).scalar_one_or_none() is None


def _transient_qa_payload(book_id: str) -> dict[str, object]:
    return {
        "bookId": book_id,
        "sourceRunId": None,
        "question": "发生了什么？",
        "mode": "global",
        "keywords": [],
        "queryVariants": ["发生了什么"],
        "candidateLimit": 6,
        "useParentChild": False,
        "vectorGeneration": 0,
        "dependencyFingerprint": "fingerprint",
        "config": {},
    }


def test_transient_request_connection_touch_prevents_stale_pruning(
    insight_platform,
) -> None:
    platform = insight_platform
    repository = TransientRequestRepository(platform["engine"])
    book_id = str(platform["book"]["id"])
    request_id, connection_token = repository.create_vector_query(
        book_id=book_id,
        request_payload=_transient_qa_payload(book_id),
    )
    with platform["engine"].begin() as connection:
        connection.execute(
            update(transient_requests)
            .where(transient_requests.c.id == request_id)
            .values(updated_at=utcnow() - timedelta(seconds=600))
        )

    assert repository.touch_connection(
        request_id=request_id,
        connection_token=connection_token,
    )
    assert repository.prune(older_than_seconds=300) == 0
    repository.close(
        request_id=request_id,
        connection_token=connection_token,
    )


def test_transient_request_prunes_abandoned_active_connection(
    insight_platform,
) -> None:
    platform = insight_platform
    repository = TransientRequestRepository(platform["engine"])
    book_id = str(platform["book"]["id"])
    request_id, connection_token = repository.create_vector_query(
        book_id=book_id,
        request_payload=_transient_qa_payload(book_id),
    )
    with platform["engine"].begin() as connection:
        connection.execute(
            update(transient_requests)
            .where(transient_requests.c.id == request_id)
            .values(updated_at=utcnow() - timedelta(seconds=600))
        )

    assert repository.prune(older_than_seconds=300) == 1
    with pytest.raises(QAFenced, match="no longer exists"):
        repository.poll(
            request_id=request_id,
            connection_token=connection_token,
        )


def test_qa_vector_query_is_connection_bound_and_worker_owned(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="qa-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    commands = InsightQACommandService(platform["engine"])
    handle = commands.create(
        book_id=str(platform["book"]["id"]),
        command={
            "question": "主角为什么继续前进？",
            "mode": "exact",
            "topK": 5,
            "useReasoning": True,
        },
    )
    with pytest.raises(QAConflict):
        commands.create(
            book_id=str(platform["book"]["id"]),
            command={"question": "第二个问题", "mode": "exact"},
        )
    requests = TransientRequestRepository(platform["engine"])
    worker = InsightQAWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        worker_epoch_id=platform["epoch_id"],
        repository=requests,
        algorithms=FakeQARetrievalAlgorithms(),
    )
    worker._query_chroma = lambda **_kwargs: [
        {
            "id": "page-result",
            "type": "page",
            "pageId": platform["page_ids"][0],
            "pageNumber": 1,
            "document": "主角决定继续前进",
            "vectorScore": 0.9,
            "hybridScore": 0.92,
        }
    ]
    assert worker.run_one()
    state = requests.poll(
        request_id=handle.request_id,
        connection_token=handle.connection_token,
    )
    assert state["status"] == "completed"
    result = requests.consume(
        request_id=handle.request_id,
        connection_token=handle.connection_token,
    )
    assert result["candidates"][0]["pageId"] == platform["page_ids"][0]
    requests.close(
        request_id=handle.request_id,
        connection_token=handle.connection_token,
    )
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(transient_requests.c.id)
        ).scalar_one_or_none() is None

    cancelled = commands.create(
        book_id=str(platform["book"]["id"]),
        command={"question": "取消这个问题", "mode": "exact"},
    )
    requests.close(
        request_id=cancelled.request_id,
        connection_token=cancelled.connection_token,
    )
    assert not worker.run_one()
    assert requests.poll(
        request_id=cancelled.request_id,
        connection_token=cancelled.connection_token,
    )["status"] == "cancelled"
    requests.close(
        request_id=cancelled.request_id,
        connection_token=cancelled.connection_token,
    )


@pytest.mark.parametrize(
    "command",
    [
        {"question": "问题", "topK": "5"},
        {"question": "问题", "threshold": "0.5"},
        {"question": "问题", "threshold": float("nan")},
        {"question": "问题", "useParentChild": "false"},
        {"question": "问题", "useReasoning": 1},
        {"question": "问题", "useReranker": 0},
    ],
)
def test_qa_command_rejects_coerced_scalar_options(
    insight_platform,
    command,
) -> None:
    with pytest.raises(ValueError):
        InsightQACommandService(insight_platform["engine"]).create(
            book_id=str(insight_platform["book"]["id"]),
            command=command,
        )


def test_qa_command_has_no_arbitrary_question_or_top_k_limit(
    insight_platform,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    service = InsightQACommandService(insight_platform["engine"])

    class Snapshot:
        source_run_id = None
        fingerprint = "fingerprint"

    monkeypatch.setattr(
        service.settings,
        "resolve_insight",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        service.derived,
        "snapshot",
        lambda **_kwargs: Snapshot(),
    )
    monkeypatch.setattr(
        service.derived,
        "qa_status",
        lambda **_kwargs: {"available": True, "generation": 1},
    )

    def create_vector_query(**kwargs):
        captured.update(kwargs)
        return "request-id", "connection-token"

    monkeypatch.setattr(
        service.repository,
        "create_vector_query",
        create_vector_query,
    )
    question = "问" * 4001

    handle = service.create(
        book_id=str(insight_platform["book"]["id"]),
        command={"question": question, "topK": 21},
    )

    assert handle.options["topK"] == 21
    assert captured["request_payload"]["candidateLimit"] == 126
    assert captured["request_payload"]["question"] == question


def test_qa_retrieval_result_rejects_malformed_candidates() -> None:
    with pytest.raises(QAConflict, match="hybridScore"):
        validate_retrieval_candidates(
            {
                "mode": "exact",
                "candidates": [
                    {
                        "id": "page-1",
                        "type": "page",
                        "pageId": "page-1",
                        "pageNumber": 1,
                        "document": "正文",
                        "hybridScore": "0.9",
                    }
                ],
            }
        )

    valid = {
        "id": "page-1",
        "type": "page",
        "pageId": "page-1",
        "pageNumber": 1,
        "document": "正文",
        "hybridScore": 0.9,
    }
    with pytest.raises(QAConflict, match="fields"):
        validate_retrieval_candidates(
            {
                "mode": "exact",
                "candidates": [{**valid, "legacyScore": 0.9}],
            }
        )
    with pytest.raises(QAConflict, match="parent context"):
        validate_retrieval_candidates(
            {
                "mode": "exact",
                "candidates": [
                    {
                        **valid,
                        "parentContext": [
                            {"layerIndex": "1", "content": {"summary": "父级"}}
                        ],
                    }
                ],
            }
        )


def test_qa_citation_keeps_complete_current_excerpt() -> None:
    document = "引用内容" * 300
    citations = citations_for(
        [
            {
                "id": "page-1",
                "type": "page",
                "pageId": "page-1",
                "pageNumber": 1,
                "document": document,
                "hybridScore": 0.9,
            }
        ]
    )

    assert citations[0]["excerpt"] == document


def test_qa_accepts_page_scoped_snapshot_after_vector_rebuild(
    insight_platform,
) -> None:
    platform = insight_platform
    for index, page_id in enumerate(platform["page_ids"], start=1):
        InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
            command={
                "bookId": str(platform["book"]["id"]),
                "scope": "page",
                "pageIds": [page_id],
            },
            idempotency_key=f"qa-page-analysis-{index}",
        )
        assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    snapshot = InsightDerivedRepository(platform["engine"]).snapshot(
        book_id=str(platform["book"]["id"]),
    )
    assert snapshot.source_run_id is None
    InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=str(platform["book"]["id"]),
        kind="vector",
        template="default",
        idempotency_key="qa-page-vector",
    )
    assert (
        _run_derived_job(
            platform,
            algorithms=FakeDerivedAlgorithms(),
            vector_store=FakeVectorStore(),
        )
        == "completed"
    )

    handle = InsightQACommandService(platform["engine"]).create(
        book_id=str(platform["book"]["id"]),
        command={
            "question": "逐页结果能否用于问答？",
            "mode": "exact",
            "useParentChild": True,
        },
    )
    with platform["engine"].connect() as connection:
        request_payload = json.loads(
            connection.execute(
                select(transient_requests.c.request_json).where(
                    transient_requests.c.id == handle.request_id
                )
            ).scalar_one()
        )
    assert request_payload["sourceRunId"] is None
    TransientRequestRepository(platform["engine"]).close(
        request_id=handle.request_id,
        connection_token=handle.connection_token,
    )


def test_global_qa_reads_published_artifacts_as_mapping_rows(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="qa-global-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

    commands = InsightQACommandService(platform["engine"])
    handle = commands.create(
        book_id=str(platform["book"]["id"]),
        command={"question": "全局发生了什么？", "mode": "global"},
    )
    requests = TransientRequestRepository(platform["engine"])
    worker = InsightQAWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        worker_epoch_id=platform["epoch_id"],
        repository=requests,
        algorithms=FakeQARetrievalAlgorithms(),
    )

    assert worker.run_one()
    result = requests.consume(
        request_id=handle.request_id,
        connection_token=handle.connection_token,
    )
    assert result["mode"] == "global"
    assert {
        candidate["id"] for candidate in result["candidates"]
    } == {
        "overview:story_summary",
        "compressed_context:default",
    }
    requests.close(
        request_id=handle.request_id,
        connection_token=handle.connection_token,
    )


def test_partial_analysis_cannot_publish_incomplete_global_artifacts(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "page",
            "pageIds": [platform["page_ids"][0]],
        },
        idempotency_key="qa-global-partial-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    with pytest.raises(
        InsightConflict,
        match="pages without published analysis",
    ):
        InsightDerivedCommandService(platform["engine"]).create_job(
            book_id=str(platform["book"]["id"]),
            kind="overview",
            template="no_spoiler",
            idempotency_key="qa-global-partial-overview",
        )


def test_qa_answer_stream_reuses_wall_clock_async_transport(monkeypatch) -> None:
    seen_requests: list[Any] = []

    class FakeAsyncTransport:
        async def complete(self, request, **_kwargs) -> str:
            seen_requests.append(request)
            callback = request.runtime_options.on_stream_chunk
            assert callback is not None
            callback("第一段", "第一段")
            callback("第二段", "第一段第二段")
            return "第一段第二段"

    monkeypatch.setattr(
        "src.shared.ai_transport.AsyncOpenAICompatibleTransport",
        FakeAsyncTransport,
    )
    chunks = list(
        DefaultQAApiAlgorithms().stream_answer(
            question="发生了什么？",
            candidates=[{"document": "漫画资料", "pageNumber": 1}],
            config={
                "chat": {
                    "provider": "custom",
                    "model_name": "test-model",
                    "api_key": "test-key",
                    "custom_base_url": "https://example.com/v1",
                    "openai_options": {
                        "request": {
                            "force_json_output": False,
                            "temperature": None,
                            "extra_body": {},
                        },
                        "execution": {
                            "use_stream": False,
                            "rpm_limit": 0,
                            "transport_retries": 1,
                            "business_retries": 0,
                        },
                    },
                },
                "prompts": {
                    "qa_response": {"content": "只依据资料回答。"},
                    "analysis_system": {"content": "你是漫画分析助手。"},
                },
            },
            cancelled=threading.Event(),
        )
    )

    assert chunks == ["第一段", "第二段"]
    assert len(seen_requests) == 1
    request = seen_requests[0]
    assert request.runtime_options.timeout is None
    assert request.openai_options.request.temperature is None
    assert request.messages[0] == {
        "role": "system",
        "content": "你是漫画分析助手。",
    }
    assert request.messages[1]["content"].startswith("只依据资料回答。")


def test_qa_reranker_honors_current_retry_and_timeout_config(
    monkeypatch,
) -> None:
    requests: list[Any] = []
    retry_counts: list[int] = []

    class FakeAsyncTransport:
        def __init__(self, *, max_retries: int) -> None:
            retry_counts.append(max_retries)

        async def rerank(self, request):
            requests.append(request)
            if len(requests) == 1:
                return {"results": []}
            return {
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                ]
            }

    async def immediate_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "src.shared.ai_transport.AsyncOpenAICompatibleTransport",
        FakeAsyncTransport,
    )
    monkeypatch.setattr("src.backend_v2.insight.qa.asyncio.sleep", immediate_sleep)

    result = DefaultQAApiAlgorithms().rerank(
        question="发生了什么？",
        candidates=[{"document": "漫画资料", "pageNumber": 1}],
        top_k=1,
        config={
            "reranker": {
                "provider": "qwen",
                "model_name": "qwen-rerank",
                "custom_base_url": "",
                "api_key": "test-key",
                "transport_retries": 1,
                "business_retries": 1,
                "timeout_seconds": 0,
            }
        },
    )

    assert retry_counts == [1]
    assert len(requests) == 2
    assert requests[0].timeout is None
    assert result == [
        {
            "document": "漫画资料",
            "pageNumber": 1,
            "rerankScore": 0.8,
        }
    ]


def test_qa_reranker_rejects_partial_results(monkeypatch) -> None:
    class FakeAsyncTransport:
        def __init__(self, *, max_retries: int) -> None:
            assert max_retries == 1

        async def rerank(self, _request):
            return {
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                ]
            }

    monkeypatch.setattr(
        "src.shared.ai_transport.AsyncOpenAICompatibleTransport",
        FakeAsyncTransport,
    )

    with pytest.raises(QAConflict, match="count does not match"):
        DefaultQAApiAlgorithms().rerank(
            question="发生了什么？",
            candidates=[
                {"document": "第一份资料"},
                {"document": "第二份资料"},
            ],
            top_k=2,
            config={
                "reranker": {
                    "provider": "qwen",
                    "model_name": "qwen-rerank",
                    "custom_base_url": "",
                    "api_key": "test-key",
                    "transport_retries": 1,
                    "business_retries": 0,
                    "timeout_seconds": 0,
                }
            },
        )


def test_qa_http_response_streams_without_creating_job_history(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="qa-stream-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    with platform["engine"].connect() as connection:
        jobs_before = len(list(connection.execute(select(jobs.c.id)).scalars()))
    qa_algorithms = FakeQAApiAlgorithms()
    app = Flask("qa-test")
    app.register_blueprint(
        create_insight_blueprint(
            engine=platform["engine"],
            data_root=platform["data_root"],
            qa_algorithms=qa_algorithms,
        )
    )
    client = app.test_client()
    response = client.post(
        f"/api/v2/insight/books/{platform['book']['id']}/qa",
        json={"question": "发生了什么？", "mode": "exact"},
        buffered=False,
    )
    worker = InsightQAWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        worker_epoch_id=platform["epoch_id"],
        algorithms=FakeQARetrievalAlgorithms(),
    )
    worker._query_chroma = lambda **_kwargs: [
        {
            "id": "page-result",
            "type": "page",
            "pageId": platform["page_ids"][0],
            "pageNumber": 1,
            "document": "关键事件",
            "vectorScore": 0.9,
            "hybridScore": 0.9,
        }
    ]
    assert worker.run_one()
    payload = b"".join(response.response).decode("utf-8")
    assert response.status_code == 200
    assert "event: chunk" in payload
    assert "答案" in payload
    assert "event: done\ndata: {}\n\n" in payload
    assert "suggestedQuestions" not in payload
    assert qa_algorithms.rerank_calls == 0
    with platform["engine"].connect() as connection:
        assert len(
            list(connection.execute(select(jobs.c.id)).scalars())
        ) == jobs_before
        assert connection.execute(
            select(transient_requests.c.id)
        ).scalar_one_or_none() is None

    qa_algorithms.stream_answer = lambda **_kwargs: iter(())
    empty_response = client.post(
        f"/api/v2/insight/books/{platform['book']['id']}/qa",
        json={"question": "不要把空回答标成成功", "mode": "exact"},
        buffered=False,
    )
    assert worker.run_one()
    empty_payload = b"".join(empty_response.response).decode("utf-8")
    assert "event: error" in empty_payload
    assert "QA provider returned an empty answer" in empty_payload
    assert "event: done" not in empty_payload
