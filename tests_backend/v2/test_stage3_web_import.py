from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib
from io import BytesIO
import json
from pathlib import Path
import uuid
import zipfile

from PIL import Image
import pytest
from sqlalchemy import select, update

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.content.image_import import ImportSafetyLimits
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.platform_repositories import (
    CredentialEdit,
    ProviderSettingMutation,
    SettingMutation,
    SettingsRepository,
)
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import DEFAULT_WEB_IMPORT_SETTINGS
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    job_config_snapshots,
    job_credential_snapshots,
    jobs,
    metadata,
    page_assets,
    pages,
    web_import_draft_pages,
    web_import_drafts,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.timestamps import utcnow
from src.backend_v2.transfer.commands import TransferCommandService
from src.backend_v2.web_import.commands import WebImportCommandService
from src.backend_v2.web_import.worker import WebImportWorkerService
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.worker.maintenance import WorkerMaintenance
from src.core.web_import.agent import MangaScraperAgent


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    with Image.new("RGB", (28, 36), color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def test_web_agent_result_parser_accepts_only_the_current_schema() -> None:
    parser = object.__new__(MangaScraperAgent)
    current = parser._parse_result(
        json.dumps(
            {
                "comic_title": "Comic",
                "chapter_title": "Chapter",
                "pages": [
                    {"page_number": 1, "image_url": "https://example.test/1.jpg"}
                ],
                "total_pages": 1,
            }
        ),
        "https://example.test/chapter",
    )
    assert current.success
    assert current.pages == [
        {"pageNumber": 1, "imageUrl": "https://example.test/1.jpg"}
    ]

    retired = parser._parse_result(
        json.dumps(
            {
                "comicTitle": "Comic",
                "chapterTitle": "Chapter",
                "pages": [
                    {"pageNumber": 1, "imageUrl": "https://example.test/1.jpg"}
                ],
                "totalPages": 1,
            }
        ),
        "https://example.test/chapter",
    )
    assert not retired.success


def _run_job(
    repository: JobQueueRepository,
    worker: WebImportWorkerService,
    epoch_id: str,
) -> str:
    fence = repository.claim_next(worker_epoch_id=epoch_id)
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    while (step := repository.next_step(fence)) is not None:
        result = worker.handle(fence, step)
        assert result["__already_published__"]
    assert repository.finish_if_complete(fence) in {
        "completed",
        "completed_with_errors",
    }
    return fence.job_id


def test_web_extract_draft_selection_and_commit_survive_the_browser(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Web Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 901)
    )
    repository = JobQueueRepository(engine)
    commands = WebImportCommandService(
        data_root=data_root,
        engine=engine,
    )
    worker = WebImportWorkerService(
        data_root=data_root,
        engine=engine,
        jobs=repository,
    )
    monkeypatch.setattr(
        worker,
        "_extract_urls",
        lambda *_args: (
            [
                "https://example.test/10.png",
                "https://example.test/20.png",
            ],
            "html",
        ),
    )
    payloads = {
        "https://example.test/10.png": _png((220, 10, 10)),
        "https://example.test/20.png": _png((10, 220, 10)),
    }

    def fake_download(url, target, _options):
        payload = payloads[url]
        target.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(worker, "_download", fake_download)
    accepted = commands.create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/chapter",
        requested_engine="auto",
        idempotency_key="web-extract-1",
    )
    assert _run_job(repository, worker, epoch_id) == accepted["jobIds"][0]
    draft = commands.get_draft(str(accepted["draftId"]))
    assert draft["status"] == "ready"
    assert draft["candidateCount"] == 2
    candidates = commands.list_draft_pages(
        draft_id=str(accepted["draftId"]),
        after_ordinal=0,
        limit=50,
    )["items"]
    with engine.connect() as connection:
        draft_thumbnail_id = connection.execute(
            select(web_import_draft_pages.c.thumbnail_asset_id).where(
                web_import_draft_pages.c.id == candidates[1]["id"]
            )
        ).scalar_one()
    selection = commands.update_selection(
        draft_id=str(accepted["draftId"]),
        selected_page_ids=[str(candidates[1]["id"])],
        base_revision=int(draft["revision"]),
        idempotency_key="web-selection-1",
    )
    assert commands.update_selection(
        draft_id=str(accepted["draftId"]),
        selected_page_ids=[str(candidates[1]["id"])],
        base_revision=int(draft["revision"]),
        idempotency_key="web-selection-1",
    ) == selection
    selected = commands.get_draft(str(accepted["draftId"]))
    commit = commands.commit(
        draft_id=str(accepted["draftId"]),
        base_revision=int(selected["revision"]),
        idempotency_key="web-commit-1",
    )
    assert _run_job(repository, worker, epoch_id) == commit["jobIds"][0]
    assert commands.get_draft(str(accepted["draftId"]))["status"] == "completed"
    with engine.connect() as connection:
        imported = list(
            connection.execute(
                select(pages.c.logical_source_path)
                .where(pages.c.chapter_id == chapter["id"])
                .order_by(pages.c.ordinal)
            ).scalars()
        )
        page_thumbnail_id = connection.execute(
            select(page_assets.c.asset_id)
            .join(pages, pages.c.id == page_assets.c.page_id)
            .where(
                pages.c.chapter_id == chapter["id"],
                page_assets.c.role == "thumbnail_source",
            )
        ).scalar_one()
    assert imported == ["20.png"]
    assert page_thumbnail_id == draft_thumbnail_id
    assert commands.delete_draft(
        str(accepted["draftId"]),
        idempotency_key="delete-web-draft-1",
    ) == {"deleted": True}
    retained_thumbnail = worker.storage.get_record(str(page_thumbnail_id))
    assert retained_thumbnail is not None
    assert worker.storage.resolve_relative_path(
        retained_thumbnail.relative_path
    ).is_file()
    assert commands.delete_draft(
        str(accepted["draftId"]),
        idempotency_key="delete-web-draft-1",
    ) == {"deleted": True}
    engine.dispose()


def test_web_import_ai_agent_config_is_resolved_and_frozen_server_side(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Agent Book")
    chapter = content.create_chapter(book_id=str(book["id"]), title="Chapter")
    settings = SettingsRepository(engine)
    settings.save_transaction(
        settings=(
            SettingMutation(
                domain="web_import",
                payload={
                    "firecrawl": {},
                    "agent": {
                        "provider": "custom",
                        "customBaseUrl": "https://agent.example/v1",
                        "modelName": "agent-model",
                        "useStream": False,
                        "forceJsonOutput": True,
                        "maxRetries": 2,
                        "timeout": 60,
                    },
                    "download": {
                        "concurrency": 4,
                        "timeout": 30,
                        "retries": 2,
                        "delay": 0,
                        "useReferer": True,
                    },
                    "extraction": {"prompt": "extract", "maxIterations": 4},
                    "imagePreprocess": {
                        "enabled": False,
                        "autoRotate": True,
                        "compression": {
                            "enabled": False,
                            "quality": 85,
                            "maxWidth": 0,
                            "maxHeight": 0,
                        },
                        "formatConvert": {
                            "enabled": False,
                            "targetFormat": "original",
                        },
                    },
                    "advanced": {"bypassProxy": False},
                    "ui": {
                        "showAgentLogs": True,
                        "autoImport": False,
                    },
                },
                base_revision=1,
                schema_version=1,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="web_import_agent",
                provider="custom",
                secret={"api_key": "agent-secret"},
                base_revision=0,
                client_ref="agent",
            ),
            CredentialEdit(
                domain="web_import_firecrawl",
                provider="firecrawl",
                secret={"api_key": "firecrawl-secret"},
                base_revision=0,
                client_ref="firecrawl",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="web_import_agent",
                provider="custom",
                payload={
                    "modelName": "agent-model",
                    "customBaseUrl": "https://agent.example/v1",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="agent",
            ),
            ProviderSettingMutation(
                domain="web_import_firecrawl",
                provider="firecrawl",
                payload={},
                base_revision=0,
                schema_version=1,
                credential_edit_ref="firecrawl",
            ),
        ),
    )
    accepted = WebImportCommandService(
        data_root=data_root,
        engine=engine,
    ).create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/chapter",
        requested_engine="ai-agent",
        idempotency_key="agent-draft",
    )
    job_id = str(accepted["jobIds"][0])
    with engine.connect() as connection:
        config_json = connection.execute(
            select(job_config_snapshots.c.payload_json).where(
                job_config_snapshots.c.job_id == job_id
            )
        ).scalar_one()
        credential_count = len(
            connection.execute(
                select(job_credential_snapshots.c.credential_version_id).where(
                    job_credential_snapshots.c.job_id == job_id
                )
            ).scalars().all()
        )
    assert "agent-secret" not in config_json
    assert "firecrawl-secret" not in config_json
    assert '"model_name":"agent-model"' in config_json
    assert '"concurrency":4' in config_json
    assert '"autoImport":false' in config_json
    assert credential_count == 2
    engine.dispose()


def test_web_extract_auto_import_is_created_by_the_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Auto Web Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    settings_payload = deepcopy(DEFAULT_WEB_IMPORT_SETTINGS)
    settings_payload["ui"]["autoImport"] = True
    SettingsRepository(engine).save_transaction(
        settings=(
            SettingMutation(
                domain="web_import",
                payload=settings_payload,
                base_revision=1,
                schema_version=1,
            ),
        ),
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 904)
    )
    repository = JobQueueRepository(engine)
    commands = WebImportCommandService(data_root=data_root, engine=engine)
    worker = WebImportWorkerService(
        data_root=data_root,
        engine=engine,
        jobs=repository,
    )
    monkeypatch.setattr(
        worker,
        "_extract_urls",
        lambda *_args: (["https://example.test/auto.png"], "html"),
    )
    payload = _png((40, 80, 160))

    def fake_download(_url, target, _options):
        target.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(worker, "_download", fake_download)
    accepted = commands.create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/auto",
        requested_engine="auto",
        idempotency_key="auto-web-draft",
    )
    extract_job_id = _run_job(repository, worker, epoch_id)
    assert extract_job_id == accepted["jobIds"][0]
    draft = commands.get_draft(str(accepted["draftId"]))
    assert draft["autoImport"] is True
    assert draft["status"] == "committing"
    commit_jobs = [job for job in draft["jobs"] if job["kind"] == "web_import_commit"]
    assert len(commit_jobs) == 1

    replay = commands.commit(
        draft_id=str(accepted["draftId"]),
        base_revision=int(draft["revision"]),
        idempotency_key="manual-race-loser",
    )
    assert replay["jobIds"] == [commit_jobs[0]["id"]]

    assert _run_job(repository, worker, epoch_id) == commit_jobs[0]["id"]
    assert commands.get_draft(str(accepted["draftId"]))["status"] == "completed"
    engine.dispose()


def test_web_import_preprocess_resizes_and_converts_the_draft_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "page.image"
    with Image.new("RGBA", (120, 80), (20, 40, 60, 120)) as image:
        image.save(target, format="PNG")
    original_checksum = hashlib.sha256(target.read_bytes()).hexdigest()

    checksum = WebImportWorkerService._preprocess_image(
        target,
        {
            "enabled": True,
            "autoRotate": True,
            "compression": {
                "enabled": True,
                "quality": 82,
                "maxWidth": 60,
                "maxHeight": 40,
            },
            "formatConvert": {
                "enabled": True,
                "targetFormat": "webp",
            },
        },
        original_checksum,
    )

    assert checksum != original_checksum
    assert checksum == hashlib.sha256(target.read_bytes()).hexdigest()
    with Image.open(target) as processed:
        assert processed.format == "WEBP"
        assert processed.size == (60, 40)


def test_web_import_download_delay_is_milliseconds_and_proxy_bypass_is_used(
    tmp_path: Path,
    monkeypatch,
) -> None:
    attempts: list[dict[str, object]] = []
    sleeps: list[float] = []
    payload = _png((10, 20, 30))

    class FakeResponse:
        headers = {"content-type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self, _chunk_size):
            yield payload

    class FakeClient:
        def __init__(self, **kwargs):
            attempts.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, _method, _url):
            if len(attempts) == 1:
                raise __import__("httpx").ConnectError("temporary failure")
            return FakeResponse()

    monkeypatch.setattr("src.backend_v2.web_import.worker.httpx.Client", FakeClient)
    monkeypatch.setattr("src.backend_v2.web_import.worker.time.sleep", sleeps.append)
    worker = object.__new__(WebImportWorkerService)
    worker.limits = ImportSafetyLimits()
    target = tmp_path / "download.image"
    checksum = worker._download(
        "https://example.test/page.png",
        target,
        {
            "timeout": 10,
            "retries": 1,
            "delay": 100,
            "referer": None,
            "bypassProxy": True,
        },
    )

    assert checksum == hashlib.sha256(payload).hexdigest()
    assert sleeps == [0.1]
    assert [attempt["trust_env"] for attempt in attempts] == [False, False]


def test_web_import_download_batch_uses_the_frozen_concurrency(
    monkeypatch,
) -> None:
    import threading

    worker = object.__new__(WebImportWorkerService)
    worker.jobs = type(
        "Jobs",
        (),
        {"fail_step": staticmethod(lambda *_args, **_kwargs: None)},
    )()
    barrier = threading.Barrier(4)
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake_handle(_fence, _step):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return {"__already_published__": True}

    monkeypatch.setattr(worker, "handle", fake_handle)
    steps = [
        {"stepId": str(index), "stepKind": "web_extract_page"}
        for index in range(4)
    ]
    worker.handle_download_batch(object(), steps)

    assert maximum == 4
    assert JobWorkerLoop._batch_size(
        "web_extract_page",
        {"options": {"concurrency": 4}},
        step_ordinal=1,
    ) == 4


def test_cancelled_web_extract_draft_is_not_restored_as_active(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Cancelled Web Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    commands = WebImportCommandService(data_root=data_root, engine=engine)
    accepted = commands.create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/cancelled",
        requested_engine="auto",
        idempotency_key="cancelled-web-draft",
    )
    draft_id = str(accepted["draftId"])
    job_id = str(accepted["jobIds"][0])

    assert content.translation_bootstrap(
        book_id=str(book["id"]),
        chapter_id=str(chapter["id"]),
    )["activeWebImportDraft"]["id"] == draft_id
    assert JobQueueRepository(engine).request_cancel(job_id)["status"] == "cancelled"
    assert commands.get_draft(draft_id)["status"] == "cancelled"
    assert content.translation_bootstrap(
        book_id=str(book["id"]),
        chapter_id=str(chapter["id"]),
    )["activeWebImportDraft"] is None

    engine.dispose()


def test_web_extract_retry_creates_a_fresh_durable_draft(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Retry Web Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 902)
    )
    repository = JobQueueRepository(engine)
    commands = WebImportCommandService(data_root=data_root, engine=engine)
    accepted = commands.create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/retry",
        requested_engine="auto",
        idempotency_key="failed-web-draft",
    )
    source_job_id = str(accepted["jobIds"][0])
    source_draft_id = str(accepted["draftId"])
    fence = repository.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    step = repository.next_step(fence)
    assert step is not None
    repository.fail_step(
        fence,
        step_id=str(step["stepId"]),
        code="TEST_FAILURE",
        message="simulated extraction failure",
    )
    assert repository.finish_if_complete(fence) == "completed_with_errors"
    assert commands.get_draft(source_draft_id)["status"] == "failed"

    retried = JobRetryService(engine).retry(
        job_id=source_job_id,
        failed_only=True,
        strategy="current",
        idempotency_key="retry-failed-web-draft",
    )
    retry_job_id = str(retried["jobIds"][0])
    with engine.connect() as connection:
        retry_job = connection.execute(
            select(jobs).where(jobs.c.id == retry_job_id)
        ).mappings().one()
    retry_draft_id = str(retry_job["web_import_draft_id"])
    assert retry_draft_id != source_draft_id
    assert retry_job["retry_of_job_id"] == source_job_id
    assert retry_job["retry_mode"] == "current"
    assert commands.get_draft(retry_draft_id)["status"] == "extracting"
    assert content.translation_bootstrap(
        book_id=str(book["id"]),
        chapter_id=str(chapter["id"]),
    )["activeWebImportDraft"]["id"] == retry_draft_id
    assert repository.request_cancel(retry_job_id)["status"] == "cancelled"
    engine.dispose()


def test_web_import_commit_retry_only_replays_failed_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Retry Commit Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 903)
    )
    repository = JobQueueRepository(engine)
    commands = WebImportCommandService(data_root=data_root, engine=engine)
    worker = WebImportWorkerService(
        data_root=data_root,
        engine=engine,
        jobs=repository,
    )
    monkeypatch.setattr(
        worker,
        "_extract_urls",
        lambda *_args: (
            [
                "https://example.test/first.png",
                "https://example.test/second.png",
            ],
            "html",
        ),
    )
    payloads = {
        "https://example.test/first.png": _png((200, 20, 20)),
        "https://example.test/second.png": _png((20, 200, 20)),
    }

    def fake_download(url, target, _options):
        payload = payloads[url]
        target.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(worker, "_download", fake_download)
    accepted = commands.create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/chapter",
        requested_engine="auto",
        idempotency_key="retry-commit-extract",
    )
    _run_job(repository, worker, epoch_id)
    draft_id = str(accepted["draftId"])
    draft = commands.get_draft(draft_id)
    commit = commands.commit(
        draft_id=draft_id,
        base_revision=int(draft["revision"]),
        idempotency_key="retry-commit-source",
    )
    source_job_id = str(commit["jobIds"][0])
    fence = repository.claim_next(worker_epoch_id=epoch_id)
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    failed_step = repository.next_step(fence)
    assert failed_step is not None
    assert failed_step["stepKind"] == "web_import_commit_page"
    repository.fail_step(
        fence,
        step_id=str(failed_step["stepId"]),
        code="TEST_FAILURE",
        message="simulated first-page commit failure",
    )
    while (step := repository.next_step(fence)) is not None:
        result = worker.handle(fence, step)
        assert result["__already_published__"]
    assert repository.finish_if_complete(fence) == "completed_with_errors"
    assert commands.get_draft(draft_id)["status"] == "completed"

    retried = JobRetryService(engine).retry(
        job_id=source_job_id,
        failed_only=True,
        strategy="current",
        idempotency_key="retry-commit-failed-page",
    )
    retry_job_id = str(retried["jobIds"][0])
    assert commands.get_draft(draft_id)["status"] == "committing"
    assert _run_job(repository, worker, epoch_id) == retry_job_id
    assert commands.get_draft(draft_id)["status"] == "completed"
    with engine.connect() as connection:
        imported = list(
            connection.execute(
                select(pages.c.logical_source_path)
                .where(pages.c.chapter_id == chapter["id"])
                .order_by(pages.c.ordinal)
            ).scalars()
        )
    assert imported == ["second.png", "first.png"]
    engine.dispose()


def test_web_extract_with_no_successful_pages_finishes_the_draft_as_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Failed Web Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 904)
    )
    repository = JobQueueRepository(engine)
    commands = WebImportCommandService(data_root=data_root, engine=engine)
    worker = WebImportWorkerService(
        data_root=data_root,
        engine=engine,
        jobs=repository,
    )
    monkeypatch.setattr(
        worker,
        "_extract_urls",
        lambda *_args: (["https://example.test/broken.png"], "html"),
    )
    monkeypatch.setattr(
        worker,
        "_download",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("download failed")),
    )
    accepted = commands.create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/chapter",
        requested_engine="auto",
        idempotency_key="all-downloads-fail",
    )
    fence = repository.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    scan = repository.next_step(fence)
    assert scan is not None
    worker.handle(fence, scan)
    download = repository.next_step(fence)
    assert download is not None
    with pytest.raises(RuntimeError, match="download failed"):
        worker.handle(fence, download)
    repository.fail_step(
        fence,
        step_id=str(download["stepId"]),
        code="WEB_IMPORT_DOWNLOAD_FAILED",
        message="download failed",
    )
    finalize = repository.next_step(fence)
    assert finalize is not None
    assert finalize["stepKind"] == "web_extract_finalize"
    worker.handle(fence, finalize)
    assert repository.finish_if_complete(fence) == "completed_with_errors"
    assert commands.get_draft(str(accepted["draftId"]))["status"] == "failed"
    engine.dispose()


def test_web_import_routes_validate_numbers_and_report_active_draft_as_locked(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Route Web Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    accepted = WebImportCommandService(
        data_root=data_root,
        engine=engine,
    ).create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/chapter",
        requested_engine="auto",
        idempotency_key="route-web-draft",
    )
    draft_id = str(accepted["draftId"])
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="web-route-api",
                epoch_token="test-token",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    client = app.test_client()

    invalid_cursor = client.get(
        f"/api/v2/web-import/drafts/{draft_id}/pages?cursor=bad"
    )
    assert invalid_cursor.status_code == 422
    locked = client.delete(
        f"/api/v2/web-import/drafts/{draft_id}",
        headers={"Idempotency-Key": "delete-active-web-draft"},
    )
    assert locked.status_code == 423
    assert locked.get_json()["error"]["code"] == "draft_locked"
    engine.dispose()


def test_worker_maintenance_prunes_only_expired_inactive_web_drafts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Expiring Web Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    accepted = WebImportCommandService(
        data_root=data_root,
        engine=engine,
    ).create_draft(
        chapter_id=str(chapter["id"]),
        source_url="https://example.test/chapter",
        requested_engine="auto",
        idempotency_key="expiring-web-draft",
    )
    draft_id = str(accepted["draftId"])
    job_id = str(accepted["jobIds"][0])
    directory = data_root / "temp" / "web-import" / draft_id
    directory.mkdir(parents=True)
    (directory / "source.image").write_bytes(_png((1, 2, 3)))
    expired_at = utcnow() - timedelta(minutes=1)
    with engine.begin() as connection:
        connection.execute(
            update(web_import_drafts)
            .where(web_import_drafts.c.id == draft_id)
            .values(expires_at=expired_at)
        )
    maintenance = WorkerMaintenance(data_root=data_root, engine=engine)

    maintenance._prune_import_temp()
    assert directory.is_dir()
    assert JobQueueRepository(engine).request_cancel(job_id)["status"] == "cancelled"
    with engine.begin() as connection:
        connection.execute(
            update(web_import_drafts)
            .where(web_import_drafts.c.id == draft_id)
            .values(expires_at=expired_at)
        )
    maintenance._prune_import_temp()
    with engine.connect() as connection:
        assert connection.execute(
            select(web_import_drafts.c.id).where(
                web_import_drafts.c.id == draft_id
            )
        ).scalar_one_or_none() is None
    assert not directory.exists()
    engine.dispose()


def test_worker_maintenance_retains_container_input_for_retry_window(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Container Retention Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("page.png", _png((4, 5, 6)))
    archive.seek(0)
    accepted = TransferCommandService(
        data_root=data_root,
        engine=engine,
    ).create_container_import(
        chapter_id=str(chapter["id"]),
        upload=archive,
        filename="chapter.cbz",
        idempotency_key="container-retention",
    )
    job_id = str(accepted["jobIds"][0])
    with engine.connect() as connection:
        config = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == job_id)
            ).scalar_one()
        )
    source = data_root / str(config["containerRelativePath"])
    maintenance = WorkerMaintenance(data_root=data_root, engine=engine)

    maintenance._prune_import_temp()
    assert source.is_file()
    assert JobQueueRepository(engine).request_cancel(job_id)["status"] == "cancelled"
    expired_at = utcnow() - timedelta(hours=25)
    with engine.begin() as connection:
        connection.execute(
            update(jobs)
            .where(jobs.c.id == job_id)
            .values(finished_at=expired_at, updated_at=expired_at)
        )
    maintenance._prune_import_temp()
    assert not source.exists()
    engine.dispose()
