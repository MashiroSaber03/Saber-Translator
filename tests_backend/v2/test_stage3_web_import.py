from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import uuid

from PIL import Image
from sqlalchemy import select

from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import metadata, pages
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
