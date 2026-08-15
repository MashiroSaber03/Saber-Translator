"""Domain commands that freeze container-import and export job inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, BinaryIO, Mapping
import uuid

from sqlalchemy import Engine, select

from src.backend_v2.content.image_import import ImportSafetyLimits
from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.storage.schema import (
    books,
    chapters,
    page_assets,
    pages,
)


CONTAINER_SUFFIXES = {".pdf", ".zip", ".cbz", ".mobi", ".azw", ".azw3"}
EXPORT_FORMATS = {"zip", "cbz", "pdf"}


class TransferDataInvalid(RuntimeError):
    pass


def _required_text(value: Mapping[str, Any], field: str) -> str:
    selected = value.get(field)
    if not isinstance(selected, str) or not selected:
        raise TransferDataInvalid(f"transfer config {field} must be a non-empty string")
    return selected


def _required_integer(
    value: Mapping[str, Any],
    field: str,
    *,
    minimum: int = 0,
) -> int:
    selected = value.get(field)
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < minimum:
        raise TransferDataInvalid(f"transfer config {field} is invalid")
    return selected


def _checksum(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TransferDataInvalid(f"transfer config {field} is invalid")
    return value


def validate_container_config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TransferDataInvalid("container import config must be an object")
    config = dict(value)
    base_fields = {
        "containerRelativePath",
        "containerType",
        "filename",
        "checksum",
        "executionMode",
    }
    scanned_fields = base_fields | {"entries", "entryItemOrdinalBase"}
    container_type = _required_text(config, "containerType")
    expected = (
        scanned_fields | {"extractedRelativePath"}
        if container_type in {"mobi", "azw", "azw3"} and "entries" in config
        else scanned_fields
        if "entries" in config
        else base_fields
    )
    if set(config) != expected:
        raise TransferDataInvalid("container import config fields are invalid")
    if container_type not in {"zip", "cbz", "pdf", "mobi", "azw", "azw3"}:
        raise TransferDataInvalid("container import type is invalid")
    _required_text(config, "containerRelativePath")
    _required_text(config, "filename")
    _checksum(config.get("checksum"), "checksum")
    if config.get("executionMode") != "sequential":
        raise TransferDataInvalid("container import execution mode is invalid")
    if "entries" not in config:
        return config
    if container_type in {"mobi", "azw", "azw3"}:
        _required_text(config, "extractedRelativePath")
    _required_integer(config, "entryItemOrdinalBase", minimum=1)
    entries = config["entries"]
    if not isinstance(entries, list) or not entries:
        raise TransferDataInvalid("container import entries must be a non-empty array")
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise TransferDataInvalid(f"container import entry {index} must be an object")
        entry = dict(raw_entry)
        kind = entry.get("kind")
        expected_entry_fields = {
            "zip": {"kind", "member", "logicalPath", "byteSize"},
            "pdf": {"kind", "pageIndex", "logicalPath"},
            "file": {"kind", "relativePath", "logicalPath", "byteSize"},
        }.get(kind)
        if expected_entry_fields is None or set(entry) != expected_entry_fields:
            raise TransferDataInvalid(f"container import entry {index} fields are invalid")
        expected_kind = (
            "zip"
            if container_type in {"zip", "cbz"}
            else "pdf"
            if container_type == "pdf"
            else "file"
        )
        if kind != expected_kind:
            raise TransferDataInvalid(
                f"container import entry {index} kind does not match its container"
            )
        _required_text(entry, "logicalPath")
        if kind == "zip":
            _required_text(entry, "member")
            _required_integer(entry, "byteSize")
        elif kind == "pdf":
            _required_integer(entry, "pageIndex")
        else:
            _required_text(entry, "relativePath")
            _required_integer(entry, "byteSize")
    return config


def validate_export_config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TransferDataInvalid("export config must be an object")
    config = dict(value)
    if set(config) != {"format", "entries", "executionMode"}:
        raise TransferDataInvalid("export config fields are invalid")
    if config.get("format") not in EXPORT_FORMATS:
        raise TransferDataInvalid("export format is invalid")
    if config.get("executionMode") != "sequential":
        raise TransferDataInvalid("export execution mode is invalid")
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        raise TransferDataInvalid("export entries must be a non-empty array")
    page_ids: set[str] = set()
    asset_ids: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
            "pageId",
            "logicalPath",
            "assetId",
            "assetRole",
        }:
            raise TransferDataInvalid(f"export entry {index} fields are invalid")
        entry = dict(raw_entry)
        page_id = _required_text(entry, "pageId")
        asset_id = _required_text(entry, "assetId")
        _required_text(entry, "logicalPath")
        if entry.get("assetRole") not in {"source", "clean", "translated"}:
            raise TransferDataInvalid(f"export entry {index} asset role is invalid")
        if page_id in page_ids or asset_id in asset_ids:
            raise TransferDataInvalid("export entries contain duplicate page or asset IDs")
        page_ids.add(page_id)
        asset_ids.add(asset_id)
    return config


class TransferCommandService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        limits: ImportSafetyLimits = ImportSafetyLimits(),
    ) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.limits = limits

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
        chapter = self._chapter(chapter_id)
        safe_filename = Path(filename).name
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
                    chunk = upload.read(self.limits.stream_chunk_bytes)
                    if not chunk:
                        break
                    byte_size += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
            if byte_size == 0:
                raise ValueError("container is empty")
            _validate_container_signature(target, suffix)
            checksum = digest.hexdigest()
            idempotency_scope = f"container-import:{chapter_id}"
            idempotency_payload = {
                "chapterId": chapter_id,
                "checksum": checksum,
                "filename": safe_filename,
            }
            replay = self.jobs.idempotency_replay(
                scope=idempotency_scope,
                key=idempotency_key,
                payload=idempotency_payload,
            )
            if replay is not None:
                target.unlink(missing_ok=True)
                return replay
            config = {
                "containerRelativePath": relative.as_posix(),
                "containerType": suffix[1:],
                "filename": safe_filename,
                "checksum": checksum,
                "executionMode": "sequential",
            }
            validate_container_config(config)
            return self.jobs.create_batch(
                kind="container_import",
                display_name=f"导入 {safe_filename}",
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
                            "filename": safe_filename,
                        },
                    )
                ],
                idempotency_scope=idempotency_scope,
                idempotency_key=idempotency_key,
                idempotency_payload=idempotency_payload,
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
                        page_assets.c.role,
                        page_assets.c.asset_id,
                    )
                    .join(
                        page_assets,
                        page_assets.c.page_id == pages.c.id,
                    )
                    .where(
                        pages.c.chapter_id == chapter_id,
                        page_assets.c.role.in_(("source", "clean", "translated")),
                    )
                    .order_by(
                        pages.c.ordinal,
                        (page_assets.c.role == "translated").desc(),
                        (page_assets.c.role == "clean").desc(),
                    )
                ).mappings()
            )
        selected = set(page_ids or [])
        if page_ids is not None and (
            not page_ids or len(selected) != len(page_ids)
        ):
            raise ValueError("pageIds must contain unique page IDs")
        entries: list[dict[str, object]] = []
        seen_pages: set[str] = set()
        for row in rows:
            page_id = str(row["id"])
            if page_id in seen_pages or (selected and page_id not in selected):
                continue
            seen_pages.add(page_id)
            entries.append(
                {
                    "pageId": page_id,
                    "logicalPath": str(row["logical_source_path"]),
                    "assetId": str(row["asset_id"]),
                    "assetRole": str(row["role"]),
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
        config = validate_export_config(
            {
                "format": export_format,
                "entries": entries,
                "executionMode": "sequential",
            }
        )
        return self.jobs.create_batch(
            kind="export",
            display_name=f"导出 {chapter['book_title']} / {chapter['title']}",
            specs=[
                JobSpec(
                    kind="export",
                    book_id=str(chapter["book_id"]),
                    chapter_id=chapter_id,
                    config=config,
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


def _validate_container_signature(path: Path, suffix: str) -> None:
    if suffix in {".zip", ".cbz"}:
        import zipfile

        if not zipfile.is_zipfile(path):
            raise ValueError("uploaded container is not a valid ZIP archive")
        return
    with path.open("rb") as source:
        header = source.read(68)
    if suffix == ".pdf" and not header.startswith(b"%PDF-"):
        raise ValueError("uploaded container is not a PDF file")
    if suffix in {".mobi", ".azw", ".azw3"} and header[60:68] != b"BOOKMOBI":
        raise ValueError("uploaded container is not a MOBI/AZW file")
