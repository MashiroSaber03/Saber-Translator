from __future__ import annotations

from io import BytesIO
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid
import zipfile

from PIL import Image
import pytest
from flask import Flask
from sqlalchemy import event, insert, select, update

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.insight.commands import InsightAnalysisCommandService
from src.backend_v2.insight.continuation import (
    ContinuationCommandService,
    ContinuationRepository,
    ContinuationWorkerService,
)
from src.backend_v2.insight.derived import (
    InsightDerivedCommandService,
    InsightDerivedRepository,
    InsightDerivedWorkerService,
    InsightVectorStore,
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
    DefaultQARetrievalAlgorithms,
    InsightQACommandService,
    InsightQAWorkerService,
    QAConflict,
    TransientFence,
    TransientHeartbeat,
    TransientRequestRepository,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightRepository,
)
from src.backend_v2.insight.routes import create_insight_blueprint
from src.backend_v2.insight.worker import InsightAnalysisWorkerService
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_layer_results,
    analysis_page_results,
    analysis_runs,
    app_settings,
    assets,
    continuation_form_image_versions,
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
                    "scene": "must be discarded",
                    "dialogues": [{"speaker_name": "must not persist"}],
                }
            ],
            "characters": ["must not persist"],
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
            "content": {"arc": "main"},
            "events": [{"summary": "event", "page_ids": [pages[0]["pageId"]]}],
            "characters": [{"name": "Saber", "first_page": 1}],
        }

    def embed_documents(self, documents, *, config):
        return [[float(index + 1), 0.5] for index, _ in enumerate(documents)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.publications: list[dict[str, Any]] = []

    def publish(self, **kwargs) -> None:
        self.publications.append(kwargs)


class FakeContinuationAlgorithms:
    def __init__(self) -> None:
        self.script_contexts: list[Mapping[str, Any]] = []

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
            "characterForms": [],
            "finalPrompt": f"page {ordinal}",
            "status": "ready",
        }

    def generate_image(self, *, prompt, reference_paths, config):
        payload = BytesIO()
        with Image.new("RGB", (48, 64), (120, 80, 160)) as image:
            image.save(payload, format="PNG")
        return payload.getvalue()


class FakeQARetrievalAlgorithms:
    def embed_queries(self, queries, *, config):
        return [[float(index + 1), 0.25] for index, _ in enumerate(queries)]


class FakeQAApiAlgorithms:
    def rerank(self, *, question, candidates, top_k, config):
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
                    "payload_json": json.dumps({"modelName": "fake-vlm"}),
                },
                {
                    "domain": "insight_chat",
                    "provider": "ollama",
                    "payload_json": json.dumps({"modelName": "fake-chat"}),
                },
                {
                    "domain": "insight_embedding",
                    "provider": "ollama",
                    "payload_json": json.dumps(
                        {"modelName": "fake-embedding"}
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
    lease = content.create_import_lease(str(chapter["id"]))
    page_ids: list[str] = []
    try:
        for index, color in enumerate(((255, 255, 255), (240, 240, 240)), 1):
            payload = BytesIO()
            with Image.new("RGB", (64, 64), color) as image:
                image.save(payload, format="PNG")
            imported, _ = importer.import_page(
                chapter_id=str(chapter["id"]),
                logical_path=f"page-{index}.png",
                upload=BytesIO(payload.getvalue()),
                lease_id=lease.id,
                owner_token=lease.owner_token,
                idempotency_key=f"page-{index}",
            )
            page_ids.append(str(imported["page"]["id"]))
    finally:
        content.release_import_lease(
            chapter_id=str(chapter["id"]),
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )
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

    with pytest.raises(ValueError, match="漫画分析 VLM 缺少模型名称"):
        InsightAnalysisCommandService(
            platform["engine"]
        ).create_analysis_job(
            command={
                "bookId": str(platform["book"]["id"]),
                "scope": "page",
                "pageId": platform["page_ids"][0],
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
        command={"scope": "full"},
    )

    assert config["chat"] == config["vlm"]


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
    assert page_two["analysisState"] == "not_analyzed"
    bootstrap = InsightRepository(platform["engine"]).bootstrap()
    book = next(
        item
        for item in bootstrap["books"]
        if item["bookId"] == str(platform["book"]["id"])
    )
    assert book["pageCount"] == 2
    assert book["analyzedPageCount"] == 1
    assert book["activeRun"]["status"] == "completed_with_errors"


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
    assert (
        JobQueueRepository(platform["engine"])
        .get_job(retry_job_id)["counts"]["total"]
        == 2
    )
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
            "pageId": page_id,
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
            "pageId": platform["page_ids"][1],
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


def test_note_revision_and_citation_snapshots(insight_platform) -> None:
    platform = insight_platform
    repository = InsightRepository(platform["engine"])
    note = repository.create_note(
        book_id=str(platform["book"]["id"]),
        title="线索",
        content="内容",
        citations=[{"pageId": platform["page_ids"][1]}],
    )
    assert note["revision"] == 1
    assert note["citations"][0]["pageNumberSnapshot"] == 2
    updated = repository.update_note(
        note_id=str(note["noteId"]),
        base_revision=1,
        title="线索（更新）",
        content="新内容",
        citations=[{"pageId": platform["page_ids"][0]}],
    )
    assert updated["revision"] == 2
    with pytest.raises(InsightConflict, match="revision"):
        repository.update_note(
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
    assert overview["payload"]["content"] == "2 pages"

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


def test_page_analysis_schema_uses_backend_identity_for_single_page() -> None:
    normalized = normalize_page_analysis(
        {
            "pages": [
                {
                    "page_number": 24,
                    "page_summary": "图片内印刷页码不同",
                }
            ]
        },
        page_id="page",
        source_asset_id="asset",
        source_checksum="0" * 64,
        page_number=1,
    )
    assert normalized["page_number_snapshot"] == 1
    assert normalized["page_summary"] == "图片内印刷页码不同"

    with pytest.raises(InvalidPageAnalysis, match="exactly one"):
        normalize_page_analysis(
            {
                "pages": [
                    {
                        "page_number": 2,
                        "page_summary": "wrong page 2",
                    },
                    {
                        "page_number": 3,
                        "page_summary": "wrong page 3",
                    },
                ]
            },
            page_id="page",
            source_asset_id="asset",
            source_checksum="0" * 64,
            page_number=1,
        )
    with pytest.raises(InvalidPageAnalysis, match="page_summary"):
        normalize_page_analysis(
            {"page_summary": ""},
            page_id="page",
            source_asset_id="asset",
            source_checksum="0" * 64,
            page_number=1,
        )


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


def test_continuation_script_and_page_loops_are_worker_owned(
    insight_platform,
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
        book_id=str(platform["book"]["id"])
    )
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
    commands.create_script_job(
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

    commands.create_pages_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-pages",
    )
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
    commands.create_images_job(
        book_id=str(platform["book"]["id"]),
        ordinals=None,
        idempotency_key="continuation-images",
    )
    fence = queue.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := queue.next_step(fence)) is not None:
        assert worker.handle(fence, step)["__already_published__"]
    assert queue.finish_if_complete(fence) == "completed"

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
    detail = queue.get_job(str(exported["jobIds"][0]))
    assert detail["artifacts"][0]["kind"] == "continuation_export"


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
        book_id=str(platform["book"]["id"])
    )
    character = repository.create_character(
        project_id=project["projectId"],
        name="Alter",
        aliases=["黑化"],
        enabled=True,
        payload={"description": "盔甲形态"},
    )
    form = repository.create_form(
        character_id=character["characterId"],
        name="战斗服",
        payload={"colors": ["black", "red"]},
    )
    repository.create_form(
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
    form = repository.bind_form_reference(
        form_id=form["formId"],
        base_revision=form["revision"],
        asset_id=reference_asset_id,
        thumbnail_asset_id=reference_thumbnail_id,
    )
    project = repository.set_project_references(
        project_id=project["projectId"],
        base_revision=project["revision"],
        asset_ids=[reference_asset_id],
    )
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
        form_id=form["formId"],
        version=1,
        base_revision=generated["revision"],
    )
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
    accepted = InsightExportCommandService(
        platform["engine"]
    ).create_export_job(
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


def test_transient_heartbeat_exception_cannot_silently_lose_its_lease() -> None:
    class FailingRepository:
        lease_seconds = 3

        def renew(self, _fence):
            raise RuntimeError("database unavailable")

    heartbeat = TransientHeartbeat(
        FailingRepository(),  # type: ignore[arg-type]
        TransientFence(
            request_id="00000000-0000-0000-0000-000000000001",
            attempt_id="00000000-0000-0000-0000-000000000002",
            lease_token="lease",
            worker_epoch_id="00000000-0000-0000-0000-000000000003",
        ),
        interval_seconds=0.01,
    )

    with heartbeat:
        assert heartbeat.fenced.wait(1)


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


def test_qa_accepts_page_scoped_snapshot_after_vector_rebuild(
    insight_platform,
) -> None:
    platform = insight_platform
    for index, page_id in enumerate(platform["page_ids"], start=1):
        InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
            command={
                "bookId": str(platform["book"]["id"]),
                "scope": "page",
                "pageId": page_id,
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
    assert request_payload["runId"] == ""
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
    }.issuperset(
        {
            "overview:story_summary",
            "compressed_context:default",
        }
    )
    requests.close(
        request_id=handle.request_id,
        connection_token=handle.connection_token,
    )


def test_global_qa_reports_incomplete_artifacts_as_conflict(
    insight_platform,
) -> None:
    platform = insight_platform
    InsightAnalysisCommandService(platform["engine"]).create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "page",
            "pageId": platform["page_ids"][0],
        },
        idempotency_key="qa-global-partial-analysis",
    )
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"
    InsightDerivedCommandService(platform["engine"]).create_job(
        book_id=str(platform["book"]["id"]),
        kind="overview",
        template="no_spoiler",
        idempotency_key="qa-global-partial-overview",
    )
    assert (
        _run_derived_job(platform, algorithms=FakeDerivedAlgorithms())
        == "completed"
    )

    with pytest.raises(QAConflict, match="missing or stale"):
        InsightQACommandService(platform["engine"]).create(
            book_id=str(platform["book"]["id"]),
            command={"question": "全局内容？", "mode": "global"},
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
    app = Flask("qa-test")
    app.register_blueprint(
        create_insight_blueprint(
            engine=platform["engine"],
            data_root=platform["data_root"],
            qa_algorithms=FakeQAApiAlgorithms(),
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
    assert "event: done" in payload
    with platform["engine"].connect() as connection:
        assert len(
            list(connection.execute(select(jobs.c.id)).scalars())
        ) == jobs_before
        assert connection.execute(
            select(transient_requests.c.id)
        ).scalar_one_or_none() is None
