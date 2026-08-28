from __future__ import annotations

from io import BytesIO
from pathlib import Path
import uuid
import zipfile

from PIL import Image
import pytest
from sqlalchemy import insert, select, update

from src.backend_v2.content.image_import import ImportSafetyLimits
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.runtime_profile import PROFILE_ENV
from src.backend_v2.storage.assets import AssetQuotaExceeded, AssetStorageService
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
    platform_config,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.transfer.commands import TransferCommandService
from src.backend_v2.transfer.worker import TransferWorkerService


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    with Image.new("RGB", (32, 40), color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def test_container_page_config_does_not_rescan_the_complete_entry_list() -> None:
    class FrozenEntries(list[dict[str, object]]):
        def __iter__(self):
            raise AssertionError("page handling must not rescan every frozen entry")

    entries = FrozenEntries([{"kind": "zip"}])
    config = {"entries": entries}

    resolved = TransferWorkerService._config(
        {
            "stepKind": "container_import_page",
            "config": config,
        }
    )

    assert resolved["entries"] is entries


def test_local_archive_scan_has_no_public_size_or_entry_caps(tmp_path: Path) -> None:
    archive_path = tmp_path / "pages.cbz"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("001.png", _png((255, 0, 0)))
        archive.writestr("002.png", _png((0, 255, 0)))
        archive.writestr("__MACOSX/._001.png", b"metadata")

    worker = object.__new__(TransferWorkerService)
    worker.limits = ImportSafetyLimits()

    assert [entry["logicalPath"] for entry in worker._scan_zip(archive_path)] == [
        "001.png",
        "002.png",
    ]


def test_public_archive_scan_enforces_configured_entry_cap(tmp_path: Path) -> None:
    archive_path = tmp_path / "pages.cbz"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("001.png", _png((255, 0, 0)))
        archive.writestr("__MACOSX/._001.png", b"metadata")

    worker = object.__new__(TransferWorkerService)
    worker.limits = ImportSafetyLimits(max_container_entries=1)

    with pytest.raises(ValueError, match="too many members"):
        worker._scan_zip(archive_path)


def _bmp(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    with Image.new("RGB", size, color) as image:
        image.save(output, format="BMP")
    return output.getvalue()


def test_mobi_scan_uses_extraction_root_filters_deduplicates_and_naturally_sorts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mobi

    extraction_root = tmp_path / "mobi-extraction"
    (extraction_root / "part-a").mkdir(parents=True)
    (extraction_root / "part-b").mkdir()
    (extraction_root / "part-c").mkdir()
    (extraction_root / "part-a" / "page10.bmp").write_bytes(
        _bmp((48, 48), (10, 20, 30))
    )
    (extraction_root / "part-a" / "page2.bmp").write_bytes(
        _bmp((48, 48), (20, 30, 40))
    )
    larger_duplicate = _bmp((64, 64), (30, 40, 50))
    (extraction_root / "part-b" / "page2.bmp").write_bytes(larger_duplicate)
    (extraction_root / "part-c" / "page1.bmp").write_bytes(
        _bmp((52, 52), (40, 50, 60))
    )
    (extraction_root / "part-c" / "thumbnail.png").write_bytes(_png((1, 2, 3)))
    extracted_document = extraction_root / "book.html"
    extracted_document.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(
        mobi,
        "extract",
        lambda _path: (str(extraction_root), str(extracted_document)),
    )

    worker = object.__new__(TransferWorkerService)
    worker.data_root = (tmp_path / "data-v2").resolve()
    entries = worker._scan_mobi(tmp_path / "book.azw3", "job-1")

    assert [Path(str(entry["logicalPath"])).name for entry in entries] == [
        "page1.bmp",
        "page2.bmp",
        "page10.bmp",
    ]
    selected_duplicate = entries[1]
    assert selected_duplicate["byteSize"] == len(larger_duplicate)
    assert selected_duplicate["logicalPath"] == "part-b/page2.bmp"
    copied = worker.data_root / str(selected_duplicate["relativePath"])
    assert copied.read_bytes() == larger_duplicate
    assert not extraction_root.exists()


def test_pdf_import_renders_pages_at_two_times_the_source_dimensions(
    tmp_path: Path,
) -> None:
    import fitz

    data_root = (tmp_path / "data-v2").resolve()
    data_root.mkdir()
    pdf_path = data_root / "sample.pdf"
    with fitz.open() as document:
        document.new_page(width=100, height=150)
        document.save(pdf_path)

    imported_sizes: list[tuple[int, int]] = []

    class CapturingImporter:
        @staticmethod
        def publish_standalone_image(source):
            with Image.open(source) as image:
                imported_sizes.append(image.size)
            return object(), object()

    class LocalStorage:
        @staticmethod
        def resolve_relative_path(relative: str) -> Path:
            return data_root / relative

    worker = object.__new__(TransferWorkerService)
    worker.data_root = data_root
    worker.importer = CapturingImporter()
    worker.storage = LocalStorage()

    worker._publish_entry(
        {"containerRelativePath": "sample.pdf"},
        {"kind": "pdf", "pageIndex": 0},
    )

    assert imported_sizes == [(200, 300)]


def test_public_container_upload_stops_at_the_current_asset_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine, profile_name="public")
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(book_id=str(book["id"]), title="Chapter")
    with engine.begin() as connection:
        connection.execute(
            update(platform_config)
            .where(platform_config.c.singleton_id == 1)
            .values(asset_quota_bytes=128)
        )
    monkeypatch.setenv(PROFILE_ENV, "public")
    source = BytesIO(b"x" * 1024)
    commands = TransferCommandService(
        data_root=data_root,
        engine=engine,
        limits=ImportSafetyLimits(stream_chunk_bytes=256),
    )

    try:
        with pytest.raises(AssetQuotaExceeded):
            commands.create_container_import(
                chapter_id=str(chapter["id"]),
                upload=source,
                filename="pages.cbz",
                idempotency_key="container-over-quota",
            )

        assert source.tell() == 256
        assert not list((data_root / "temp" / "container-import").glob("*.cbz"))
    finally:
        engine.dispose()


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
    replayed = commands.create_container_import(
        chapter_id=str(chapter["id"]),
        upload=BytesIO(archive_bytes.getvalue()),
        filename="pages.cbz",
        idempotency_key="container-1",
    )
    assert replayed == accepted
    assert len(list((data_root / "temp" / "container-import").glob("*.cbz"))) == 1
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
        preserve_original_filenames=False,
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

    preserved_export = commands.create_export(
        chapter_id=str(chapter["id"]),
        export_format="zip",
        page_ids=None,
        preserve_original_filenames=True,
        idempotency_key="export-preserved-names",
    )
    preserved_job_id = _run_job(jobs, worker, epoch_id)
    assert preserved_job_id == preserved_export["jobIds"][0]
    with engine.connect() as connection:
        preserved_asset_id = connection.execute(
            select(job_artifacts.c.asset_id).where(
                job_artifacts.c.job_id == preserved_job_id
            )
        ).scalar_one()
        preserved_relative = connection.execute(
            select(assets.c.relative_path).where(
                assets.c.id == preserved_asset_id
            )
        ).scalar_one()
    with zipfile.ZipFile(
        AssetStorageService(data_root, engine).resolve_relative_path(
            preserved_relative
        )
    ) as packaged:
        assert packaged.namelist() == [
            "chapter/001.png",
            "chapter/002.png",
        ]

    broken_export = commands.create_export(
        chapter_id=str(chapter["id"]),
        export_format="cbz",
        page_ids=None,
        preserve_original_filenames=False,
        idempotency_key="export-missing-page",
    )
    AssetStorageService(data_root, engine).resolve_relative_path(
        clean_assets[1].relative_path
    ).unlink()
    broken_fence = jobs.claim_next(worker_epoch_id=epoch_id)
    assert broken_fence is not None
    assert broken_fence.job_id == broken_export["jobIds"][0]
    broken_step = jobs.next_step(broken_fence)
    assert broken_step is not None
    with pytest.raises(FileNotFoundError):
        worker.handler(broken_fence, broken_step)
    with engine.connect() as connection:
        assert connection.execute(
            select(job_artifacts.c.asset_id).where(
                job_artifacts.c.job_id == broken_fence.job_id
            )
        ).scalar_one_or_none() is None
    assert not (
        data_root / "temp" / "exports" / f"{broken_fence.job_id}.cbz"
    ).exists()
    engine.dispose()


def test_export_propagates_memory_failure_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    book = content.create_book(title="Memory failure")
    chapter = content.create_chapter(
        book_id=str(book["id"]),
        title="Chapter",
    )
    page_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            insert(pages).values(
                id=page_id,
                chapter_id=str(chapter["id"]),
                ordinal=1,
                logical_source_path="page.png",
            )
        )
    storage = AssetStorageService(data_root, engine)
    storage.publish_bytes(
        _png((32, 64, 96)),
        extension="png",
        mime_type="image/png",
        width=32,
        height=40,
        bind=lambda connection, asset_id: connection.execute(
            insert(page_assets).values(
                page_id=page_id,
                role="source",
                asset_id=asset_id,
                input_source_revision=1,
                input_document_revision=1,
            )
        ),
    )
    jobs = JobQueueRepository(engine)
    accepted = TransferCommandService(
        data_root=data_root,
        engine=engine,
    ).create_export(
        chapter_id=str(chapter["id"]),
        export_format="cbz",
        page_ids=None,
        preserve_original_filenames=False,
        idempotency_key="memory-failure",
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 901)
    )
    fence = jobs.claim_next(worker_epoch_id=epoch_id)
    assert fence is not None
    assert fence.job_id == accepted["jobIds"][0]
    step = jobs.next_step(fence)
    assert step is not None
    worker = TransferWorkerService(
        data_root=data_root,
        engine=engine,
        jobs_repository=jobs,
    )

    def fail_write(*_args, **_kwargs) -> None:
        raise MemoryError("native allocation failed")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_write)
    with pytest.raises(MemoryError, match="allocation failed"):
        worker.handler(fence, step)
    assert not (
        data_root / "temp" / "exports" / f"{fence.job_id}.cbz"
    ).exists()
    engine.dispose()


def test_multi_book_export_uses_one_durable_zip_job(tmp_path: Path) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    storage = AssetStorageService(data_root, engine)
    book_specs = [
        ("Book/One", "Chapter:A", "001.png", (255, 0, 0)),
        ("Book Two", "Chapter B", "nested/002.png", (0, 255, 0)),
    ]
    book_ids: list[str] = []
    for ordinal, (book_title, chapter_title, logical_path, color) in enumerate(
        book_specs,
        start=1,
    ):
        book = content.create_book(title=book_title)
        chapter = content.create_chapter(
            book_id=str(book["id"]),
            title=chapter_title,
        )
        book_ids.append(str(book["id"]))
        page_id = str(uuid.uuid4())
        with engine.begin() as connection:
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=str(chapter["id"]),
                    ordinal=ordinal,
                    logical_source_path=logical_path,
                )
            )
        storage.publish_bytes(
            _png(color),
            extension="png",
            mime_type="image/png",
            width=32,
            height=40,
            bind=lambda connection, asset_id, page_id=page_id: connection.execute(
                insert(page_assets).values(
                    page_id=page_id,
                    role="source",
                    asset_id=asset_id,
                    input_source_revision=1,
                )
            ),
        )

    jobs = JobQueueRepository(engine)
    accepted = TransferCommandService(
        data_root=data_root,
        engine=engine,
    ).create_books_export(
        book_ids=book_ids,
        preserve_original_filenames=True,
        idempotency_key="books-export",
    )
    assert len(accepted["jobIds"]) == 1
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 902)
    )
    worker = TransferWorkerService(
        data_root=data_root,
        engine=engine,
        jobs_repository=jobs,
    )
    job_id = _run_job(jobs, worker, epoch_id)
    assert job_id == accepted["jobIds"][0]

    from src.backend_v2.storage.schema import assets

    with engine.connect() as connection:
        artifact_id = connection.execute(
            select(job_artifacts.c.asset_id).where(
                job_artifacts.c.job_id == job_id
            )
        ).scalar_one()
        relative = connection.execute(
            select(assets.c.relative_path).where(assets.c.id == artifact_id)
        ).scalar_one()
    with zipfile.ZipFile(storage.resolve_relative_path(relative)) as packaged:
        assert packaged.namelist() == [
            "Book Two/Chapter B/nested/002.png",
            "Book_One/Chapter_A/001.png",
        ]
    engine.dispose()


def test_selected_chapters_export_as_one_durable_zip_job(tmp_path: Path) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    content = ContentRepository(engine)
    storage = AssetStorageService(data_root, engine)
    book = content.create_book(title="Selected Book")
    selected_chapter_ids: list[str] = []
    for ordinal, title in enumerate(
        ("Chapter 1", "Chapter 2", "Chapter 3"),
        start=1,
    ):
        chapter = content.create_chapter(book_id=str(book["id"]), title=title)
        chapter_id = str(chapter["id"])
        if ordinal != 2:
            selected_chapter_ids.append(chapter_id)
        page_id = str(uuid.uuid4())
        with engine.begin() as connection:
            connection.execute(
                insert(pages).values(
                    id=page_id,
                    chapter_id=chapter_id,
                    ordinal=1,
                    logical_source_path=f"{ordinal:03d}.png",
                )
            )
        storage.publish_bytes(
            _png((ordinal * 60, 0, 0)),
            extension="png",
            mime_type="image/png",
            width=32,
            height=40,
            bind=lambda connection, asset_id, page_id=page_id: connection.execute(
                insert(page_assets).values(
                    page_id=page_id,
                    role="source",
                    asset_id=asset_id,
                    input_source_revision=1,
                )
            ),
        )

    jobs = JobQueueRepository(engine)
    accepted = TransferCommandService(
        data_root=data_root,
        engine=engine,
    ).create_chapters_export(
        chapter_ids=selected_chapter_ids,
        preserve_original_filenames=True,
        idempotency_key="chapters-export",
    )
    assert len(accepted["jobIds"]) == 1
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 903)
    )
    worker = TransferWorkerService(
        data_root=data_root,
        engine=engine,
        jobs_repository=jobs,
    )
    job_id = _run_job(jobs, worker, epoch_id)
    assert job_id == accepted["jobIds"][0]

    from src.backend_v2.storage.schema import assets

    with engine.connect() as connection:
        artifact_id = connection.execute(
            select(job_artifacts.c.asset_id).where(
                job_artifacts.c.job_id == job_id
            )
        ).scalar_one()
        relative = connection.execute(
            select(assets.c.relative_path).where(assets.c.id == artifact_id)
        ).scalar_one()
    with zipfile.ZipFile(storage.resolve_relative_path(relative)) as packaged:
        assert packaged.namelist() == [
            "Selected Book/Chapter 1/001.png",
            "Selected Book/Chapter 3/003.png",
        ]
    engine.dispose()
