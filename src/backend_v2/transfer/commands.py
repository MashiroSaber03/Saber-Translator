"""Domain commands that freeze container-import and export job inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, BinaryIO
import uuid

from sqlalchemy import Engine, select

from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.storage.schema import (
    assets,
    books,
    chapters,
    page_assets,
    pages,
)


CONTAINER_SUFFIXES = {".pdf", ".zip", ".cbz", ".mobi", ".azw", ".azw3"}
EXPORT_FORMATS = {"zip", "cbz", "pdf"}


class TransferCommandService:
    def __init__(self, *, data_root: Path, engine: Engine) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = JobQueueRepository(engine)

    def create_container_import(
        self,
        *,
        chapter_id: str,
        upload: BinaryIO,
        filename: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        suffix = Path(filename).suffix.lower()
        if suffix not in CONTAINER_SUFFIXES:
            raise ValueError("unsupported container format")
        relative = (
            Path("temp")
            / "container-import"
            / f"{uuid.uuid4().hex}{suffix}"
        )
        target = self.data_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        byte_size = 0
        try:
            with target.open("xb") as output:
                while True:
                    chunk = upload.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_size += len(chunk)
                    if byte_size > 1024 * 1024 * 1024:
                        raise ValueError("container exceeds 1 GiB")
                    digest.update(chunk)
                    output.write(chunk)
            if byte_size == 0:
                raise ValueError("container is empty")
            chapter = self._chapter(chapter_id)
            config = {
                "containerRelativePath": relative.as_posix(),
                "containerType": suffix[1:],
                "filename": Path(filename).name,
                "checksum": digest.hexdigest(),
                "executionMode": "sequential",
            }
            return self.jobs.create_batch(
                kind="container_import",
                display_name=f"导入 {Path(filename).name}",
                specs=[
                    JobSpec(
                        kind="container_import",
                        book_id=str(chapter["book_id"]),
                        chapter_id=chapter_id,
                        config=config,
                        items=(
                            JobItemSpec(
                                page_id=None,
                                step_kinds=("container_scan",),
                            ),
                        ),
                        target_display={
                            "book": chapter["book_title"],
                            "chapter": chapter["title"],
                            "filename": Path(filename).name,
                        },
                    )
                ],
                idempotency_scope=f"container-import:{chapter_id}",
                idempotency_key=idempotency_key,
                idempotency_payload={
                    "chapterId": chapter_id,
                    "checksum": digest.hexdigest(),
                    "filename": Path(filename).name,
                },
            )
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def create_export(
        self,
        *,
        chapter_id: str,
        export_format: str,
        page_ids: list[str] | None,
        idempotency_key: str,
    ) -> dict[str, object]:
        if export_format not in EXPORT_FORMATS:
            raise ValueError("export format must be zip, cbz, or pdf")
        chapter = self._chapter(chapter_id)
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        pages.c.id,
                        pages.c.logical_source_path,
                        pages.c.ordinal,
                        page_assets.c.role,
                        page_assets.c.asset_id,
                        assets.c.mime_type,
                        assets.c.relative_path,
                    )
                    .join(
                        page_assets,
                        page_assets.c.page_id == pages.c.id,
                    )
                    .join(assets, assets.c.id == page_assets.c.asset_id)
                    .where(
                        pages.c.chapter_id == chapter_id,
                        page_assets.c.role.in_(("source", "translated")),
                    )
                    .order_by(
                        pages.c.ordinal,
                        (page_assets.c.role == "translated").desc(),
                    )
                ).mappings()
            )
        selected = set(page_ids or [])
        if page_ids is not None and (
            not page_ids or len(selected) != len(page_ids)
        ):
            raise ValueError("pageIds must contain unique page IDs")
        entries: list[dict[str, Any]] = []
        seen_pages: set[str] = set()
        for row in rows:
            page_id = str(row["id"])
            if page_id in seen_pages or (selected and page_id not in selected):
                continue
            seen_pages.add(page_id)
            entries.append(
                {
                    "pageId": page_id,
                    "ordinal": int(row["ordinal"]),
                    "logicalPath": str(row["logical_source_path"]),
                    "assetId": str(row["asset_id"]),
                    "assetRole": str(row["role"]),
                    "mimeType": str(row["mime_type"]),
                    "relativePath": str(row["relative_path"]),
                }
            )
        if page_ids is not None and seen_pages != selected:
            raise ValueError("pageIds must all belong to the chapter")
        if not entries:
            raise ValueError("export requires at least one page")
        asset_inputs = {
            f"page:{index:06d}": str(entry["assetId"])
            for index, entry in enumerate(entries, start=1)
        }
        return self.jobs.create_batch(
            kind="export",
            display_name=f"导出 {chapter['book_title']} / {chapter['title']}",
            specs=[
                JobSpec(
                    kind="export",
                    book_id=str(chapter["book_id"]),
                    chapter_id=chapter_id,
                    config={
                        "format": export_format,
                        "entries": entries,
                        "executionMode": "sequential",
                    },
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("export_package",),
                            asset_inputs=asset_inputs,
                        ),
                    ),
                    target_display={
                        "book": chapter["book_title"],
                        "chapter": chapter["title"],
                        "pageCount": len(entries),
                        "format": export_format,
                    },
                )
            ],
            idempotency_scope=f"chapter-export:{chapter_id}",
            idempotency_key=idempotency_key,
            idempotency_payload={
                "chapterId": chapter_id,
                "format": export_format,
                "entries": [
                    {
                        "pageId": entry["pageId"],
                        "assetId": entry["assetId"],
                    }
                    for entry in entries
                ],
            },
        )

    def _chapter(self, chapter_id: str):
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    chapters.c.id,
                    chapters.c.book_id,
                    chapters.c.title,
                    books.c.title.label("book_title"),
                )
                .join(books, books.c.id == chapters.c.book_id)
                .where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
        if row is None:
            raise ValueError("chapter not found")
        return row
