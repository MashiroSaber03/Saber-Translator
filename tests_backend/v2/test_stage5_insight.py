from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping
import uuid
import zipfile

from PIL import Image
import pytest
from flask import Flask
from sqlalchemy import select

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.insight.commands import InsightAnalysisCommandService
from src.backend_v2.insight.continuation import (
    ContinuationAlgorithms,
    ContinuationCommandService,
    ContinuationRepository,
    ContinuationWorkerService,
)
from src.backend_v2.insight.derived import (
    InsightDerivedCommandService,
    InsightDerivedRepository,
    InsightDerivedWorkerService,
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
    InsightQACommandService,
    InsightQAWorkerService,
    QAConflict,
    TransientRequestRepository,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightRepository,
)
from src.backend_v2.insight.routes import create_insight_blueprint
from src.backend_v2.insight.worker import InsightAnalysisWorkerService
from src.backend_v2.jobs.repository import JobQueueRepository
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
    assets,
    continuation_form_image_versions,
    job_asset_inputs,
    jobs,
    metadata,
    page_assets,
    timeline_versions,
    transient_requests,
    vector_generations,
)
from src.backend_v2.storage.seeding import seed_system_records


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
    def generate_script(self, *, context, config):
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


@pytest.fixture()
def insight_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
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
    replay = command.create_analysis_job(
        command={
            "bookId": str(platform["book"]["id"]),
            "scope": "full",
        },
        idempotency_key="full-1",
    )
    assert replay == accepted
    assert _run_job(platform, FakeInsightAlgorithms()) == "completed"

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
        page_ids=[platform["page_ids"][1]],
    )
    assert note["revision"] == 1
    assert note["citations"][0]["pageNumberSnapshot"] == 2
    updated = repository.update_note(
        note_id=str(note["noteId"]),
        base_revision=1,
        title="线索（更新）",
        content="新内容",
        page_ids=[platform["page_ids"][0]],
    )
    assert updated["revision"] == 2
    with pytest.raises(InsightConflict, match="revision"):
        repository.update_note(
            note_id=str(note["noteId"]),
            base_revision=1,
            title="过期写入",
            content="",
            page_ids=[],
        )
    page = repository.list_notes(
        book_id=str(platform["book"]["id"]),
        limit=1,
    )
    assert page["items"][0]["content"] is None
    assert repository.get_note(note_id=note["noteId"])["content"] == "新内容"


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


def test_page_analysis_schema_rejects_missing_or_mismatched_pages() -> None:
    with pytest.raises(InvalidPageAnalysis, match="exactly one"):
        normalize_page_analysis(
            {
                "pages": [
                    {
                        "page_number": 2,
                        "page_summary": "wrong page",
                    }
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
    command = InsightAnalysisCommandService
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
    worker = ContinuationWorkerService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=queue,
        algorithms=FakeContinuationAlgorithms(),
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
    restored = repository.bootstrap(
        book_id=str(platform["book"]["id"])
    )["project"]
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
    generated = repository.list_forms(
        project_id=project["projectId"]
    )["items"][0]
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
