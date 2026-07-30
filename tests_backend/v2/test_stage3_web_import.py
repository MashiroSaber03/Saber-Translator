from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import uuid

from PIL import Image
from sqlalchemy import select, update

from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.storage.platform_repositories import (
    CredentialEdit,
    ProviderSettingMutation,
    SettingMutation,
    SettingsRepository,
)
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    job_config_snapshots,
    job_credential_snapshots,
    jobs,
    metadata,
    pages,
    web_import_drafts,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.web_import.commands import WebImportCommandService
from src.backend_v2.web_import.worker import WebImportWorkerService


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    with Image.new("RGB", (28, 36), color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


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
        config={},
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
    commands.update_selection(
        draft_id=str(accepted["draftId"]),
        selected_page_ids=[str(candidates[1]["id"])],
        base_revision=int(draft["revision"]),
    )
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
    assert imported == ["20.png"]
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
                    "agent": {
                        "provider": "custom",
                        "useStream": False,
                        "forceJsonOutput": True,
                        "maxRetries": 2,
                        "timeout": 60,
                    },
                    "download": {
                        "timeout": 30,
                        "retries": 2,
                        "delay": 0,
                        "useReferer": True,
                    },
                    "extraction": {"prompt": "extract", "maxIterations": 4},
                },
                base_revision=1,
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
                credential_edit_ref="agent",
            ),
            ProviderSettingMutation(
                domain="web_import_firecrawl",
                provider="firecrawl",
                payload={},
                base_revision=0,
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
        config={},
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
    assert credential_count == 2
    engine.dispose()


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
        config={},
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

    # Older databases can contain an extracting draft whose owning job already
    # reached a terminal state. Read paths must remain self-healing.
    with engine.begin() as connection:
        connection.execute(
            update(web_import_drafts)
            .where(web_import_drafts.c.id == draft_id)
            .values(status="extracting")
        )
        assert connection.execute(
            select(jobs.c.status).where(jobs.c.id == job_id)
        ).scalar_one() == "cancelled"
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
        config={},
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
        config={},
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
