from __future__ import annotations

from io import BytesIO
from pathlib import Path
import uuid
import zipfile

from PIL import Image
from sqlalchemy import insert, select

from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.epochs import (
    EpochRegistration,
    ProcessEpochRepository,
)
from src.backend_v2.storage.schema import (
    job_artifacts,
    metadata,
    page_assets,
    pages,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.transfer.commands import TransferCommandService
from src.backend_v2.transfer.worker import TransferWorkerService


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    with Image.new("RGB", (32, 40), color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def _run_job(
    repository: JobQueueRepository,
    worker: TransferWorkerService,
    epoch_id: str,
) -> str:
    fence = repository.claim_next(worker_epoch_id=epoch_id)
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    while (step := repository.next_step(fence)) is not None:
        result = worker.handler(fence, step)
        assert result["__already_published__"]
    assert repository.finish_if_complete(fence) in {
        "completed",
        "completed_with_errors",
    }
    return fence.job_id


def test_container_import_and_export_are_worker_owned_and_durable(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 900)
    )
    commands = TransferCommandService(data_root=data_root, engine=engine)
    jobs = JobQueueRepository(engine)
    worker = TransferWorkerService(
        data_root=data_root,
        engine=engine,
        jobs_repository=jobs,
    )

    archive_bytes = BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("chapter/001.png", _png((255, 0, 0)))
        archive.writestr("chapter/002.png", _png((0, 255, 0)))
    accepted = commands.create_container_import(
        chapter_id=str(chapter["id"]),
        upload=BytesIO(archive_bytes.getvalue()),
        filename="pages.cbz",
        idempotency_key="container-1",
    )
    import_job_id = _run_job(jobs, worker, epoch_id)
    assert import_job_id == accepted["jobIds"][0]

    with engine.connect() as connection:
        imported_pages = list(
            connection.execute(
                select(pages.c.id, pages.c.logical_source_path)
                .where(pages.c.chapter_id == chapter["id"])
                .order_by(pages.c.ordinal)
            )
        )
        roles = list(
            connection.execute(
                select(page_assets.c.role).where(
                    page_assets.c.page_id.in_(
                        [row.id for row in imported_pages]
                    )
                )
            ).scalars()
        )
    assert [row.logical_source_path for row in imported_pages] == [
        "chapter/001.png",
        "chapter/002.png",
    ]
    assert roles.count("source") == 2
    assert roles.count("thumbnail_source") == 2

    storage = AssetStorageService(data_root, engine)
    clean_assets = []
    for index, page in enumerate(imported_pages):
        clean_assets.append(
            storage.publish_bytes(
                _png((0, 0, 64 + index)),
                extension="png",
                mime_type="image/png",
                width=32,
                height=40,
                bind=lambda connection, asset_id, page_id=page.id: connection.execute(
                    insert(page_assets).values(
                        page_id=page_id,
                        role="clean",
                        asset_id=asset_id,
                        input_source_revision=1,
                    )
                ),
            )
        )
    storage.publish_bytes(
        _png((64, 0, 0)),
        extension="png",
        mime_type="image/png",
        width=32,
        height=40,
        bind=lambda connection, asset_id: connection.execute(
            insert(page_assets).values(
                page_id=imported_pages[0].id,
                role="translated",
                asset_id=asset_id,
                input_source_revision=1,
                input_document_revision=1,
            )
        ),
    )

    export = commands.create_export(
        chapter_id=str(chapter["id"]),
        export_format="cbz",
        page_ids=None,
        idempotency_key="export-1",
    )
    export_job_id = _run_job(jobs, worker, epoch_id)
    assert export_job_id == export["jobIds"][0]
    with engine.connect() as connection:
        artifact_id = connection.execute(
            select(job_artifacts.c.asset_id).where(
                job_artifacts.c.job_id == export_job_id
            )
        ).scalar_one()
    artifact_url = jobs.get_job(export_job_id)["artifacts"][0]["url"]
    assert artifact_url == f"/api/v2/assets/{artifact_id}"

    from src.backend_v2.storage.schema import assets

    with engine.connect() as connection:
        relative = connection.execute(
            select(assets.c.relative_path).where(assets.c.id == artifact_id)
        ).scalar_one()
    with zipfile.ZipFile(
        AssetStorageService(data_root, engine).resolve_relative_path(relative)
    ) as packaged:
        assert packaged.namelist() == [
            "chapter/translated_001.png",
            "chapter/clean_002.png",
        ]
    engine.dispose()
