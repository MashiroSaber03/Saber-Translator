from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from PIL import Image
import pytest
from sqlalchemy import select

from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.insight.commands import InsightAnalysisCommandService
from src.backend_v2.insight.derived import (
    InsightDerivedCommandService,
    InsightDerivedRepository,
    InsightDerivedWorkerService,
)
from src.backend_v2.insight.page_schema import (
    InvalidPageAnalysis,
    normalize_page_analysis,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightRepository,
)
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
    job_asset_inputs,
    jobs,
    metadata,
    timeline_versions,
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
            "key_events": [],
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
