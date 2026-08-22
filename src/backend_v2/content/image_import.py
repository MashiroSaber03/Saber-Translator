"""One-page-at-a-time image validation, thumbnailing, and publication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from typing import BinaryIO, Callable
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError

from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.storage.assets import AssetStorageService


FORMAT_DETAILS: dict[str, tuple[str, str]] = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
    "WEBP": ("webp", "image/webp"),
    "GIF": ("gif", "image/gif"),
    "BMP": ("bmp", "image/bmp"),
    "TIFF": ("tiff", "image/tiff"),
}

# Local mode keeps the original unrestricted artwork workflow. Public-facing
# profiles retain a generous decompression-bomb ceiling to protect the host.
Image.MAX_IMAGE_PIXELS = (
    50_000_000
    if os.environ.get("SABER_V2_PROFILE", "local") == "public"
    else None
)


@dataclass(frozen=True, slots=True)
class ImportSafetyLimits:
    max_compression_ratio: float = 1000.0
    stream_chunk_bytes: int = 1024 * 1024
    max_container_entries: int = 10_000
    max_archive_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024


class UnsupportedImage(ValueError):
    pass


class ImageImportService:
    def __init__(
        self,
        *,
        data_root: Path,
        repository: ContentRepository,
        storage: AssetStorageService,
        limits: ImportSafetyLimits = ImportSafetyLimits(),
    ) -> None:
        self.data_root = data_root
        self.repository = repository
        self.storage = storage
        self.limits = limits

    def import_page(
        self,
        *,
        chapter_id: str,
        logical_path: str,
        text_style: dict[str, object],
        upload: BinaryIO,
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("Idempotency-Key is required and must be at most 200 characters")
        temporary = (
            self.data_root
            / "temp"
            / "imports"
            / f"page-{uuid.uuid4().hex}.upload"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            checksum, byte_size = self._copy_upload(upload, temporary)
            normalized_request = {
                "chapterId": chapter_id,
                "logicalPath": logical_path.replace("\\", "/"),
                "checksum": checksum,
                "byteSize": byte_size,
                "textStyle": text_style,
            }
            request_hash = hashlib.sha256(
                json.dumps(
                    normalized_request,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            scope = (
                "POST:createChapterPage:"
                f"{chapter_id}"
            )
            replay = self.repository.replay_idempotency(
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay, True

            source_asset, thumbnail_asset = self._publish_temporary(temporary)
            return self.repository.append_page(
                chapter_id=chapter_id,
                requested_logical_path=logical_path,
                text_style=text_style,
                source=source_asset,
                thumbnail=thumbnail_asset,
                idempotency_scope=scope,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def replace_page_source(
        self,
        *,
        page_id: str,
        base_source_revision: int,
        upload: BinaryIO,
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError(
                "Idempotency-Key is required and must be at most 200 characters"
            )
        temporary = (
            self.data_root
            / "temp"
            / "imports"
            / f"replacement-{uuid.uuid4().hex}.upload"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            checksum, byte_size = self._copy_upload(upload, temporary)
            request_hash = hashlib.sha256(
                json.dumps(
                    {
                        "pageId": page_id,
                        "baseSourceRevision": base_source_revision,
                        "checksum": checksum,
                        "byteSize": byte_size,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            scope = f"PUT:replacePageSource:{page_id}"
            replay = self.repository.replay_idempotency(
                scope=scope,
                key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay, True
            source, thumbnail = self._publish_temporary(temporary)
            return self.repository.replace_page_source(
                page_id=page_id,
                base_source_revision=base_source_revision,
                source=source,
                thumbnail=thumbnail,
                idempotency_scope=scope,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def publish_cover(self, upload: BinaryIO):
        temporary = (
            self.data_root
            / "temp"
            / "imports"
            / f"cover-{uuid.uuid4().hex}.upload"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._copy_upload(upload, temporary)
            try:
                with Image.open(temporary) as probe:
                    probe.verify()
                with Image.open(temporary) as decoded:
                    decoded.seek(0)
                    oriented = ImageOps.exif_transpose(decoded)
                    try:
                        oriented.load()
                        cover = oriented.copy()
                    finally:
                        if oriented is not decoded:
                            oriented.close()
                try:
                    cover.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    if cover.mode not in ("RGB", "RGBA"):
                        converted = cover.convert("RGBA")
                        cover.close()
                        cover = converted
                    width, height = cover.size
                    with BytesIO() as output:
                        cover.save(output, format="WEBP", quality=85, method=4)
                        payload = output.getvalue()
                finally:
                    cover.close()
            except (UnidentifiedImageError, OSError) as exc:
                raise UnsupportedImage(
                    "uploaded cover is not a decodable image"
                ) from exc
            return self.storage.publish_bytes(
                payload,
                extension="webp",
                mime_type="image/webp",
                width=width,
                height=height,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def publish_draft_thumbnail(self, path: Path):
        (
            _extension,
            _mime_type,
            _width,
            _height,
            thumbnail_width,
            thumbnail_height,
            thumbnail,
        ) = self._decode_and_thumbnail(path)
        return self.storage.publish_bytes(
            thumbnail,
            extension="webp",
            mime_type="image/webp",
            width=thumbnail_width,
            height=thumbnail_height,
        )

    def publish_standalone_image(self, upload: BinaryIO):
        """Publish one validated source image and its source thumbnail."""

        temporary = (
            self.data_root
            / "temp"
            / "imports"
            / f"asset-{uuid.uuid4().hex}.upload"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._copy_upload(upload, temporary)
            return self._publish_temporary(temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def publish_standalone_source(
        self,
        upload: BinaryIO,
        *,
        bind: Callable[[object, str], None] | None = None,
    ):
        """Publish one validated image when no thumbnail relation exists."""

        temporary = (
            self.data_root
            / "temp"
            / "imports"
            / f"standalone-{uuid.uuid4().hex}.upload"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._copy_upload(upload, temporary)
            return self.publish_standalone_path(temporary, bind=bind)
        finally:
            temporary.unlink(missing_ok=True)

    def publish_standalone_path(
        self,
        path: Path,
        *,
        bind: Callable[[object, str], None] | None = None,
    ):
        """Validate and publish an already-spooled image without another RAM copy."""

        extension, mime_type, width, height = self._decode_metadata(path)
        with path.open("rb") as source_stream:
            return self.storage.publish_stream(
                source_stream,
                extension=extension,
                mime_type=mime_type,
                width=width,
                height=height,
                bind=bind,
            )

    def _publish_temporary(self, temporary: Path):
        (
            extension,
            mime_type,
            width,
            height,
            thumbnail_width,
            thumbnail_height,
            thumbnail,
        ) = self._decode_and_thumbnail(temporary)
        with temporary.open("rb") as source_stream:
            source_asset = self.storage.publish_stream(
                source_stream,
                extension=extension,
                mime_type=mime_type,
                width=width,
                height=height,
            )
        thumbnail_asset = self.storage.publish_bytes(
            thumbnail,
            extension="webp",
            mime_type="image/webp",
            width=thumbnail_width,
            height=thumbnail_height,
        )
        return source_asset, thumbnail_asset

    def _copy_upload(self, upload: BinaryIO, destination: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        byte_size = 0
        with destination.open("xb") as output:
            while True:
                chunk = upload.read(self.limits.stream_chunk_bytes)
                if not chunk:
                    break
                byte_size += len(chunk)
                digest.update(chunk)
                output.write(chunk)
        if byte_size == 0:
            raise ValueError("uploaded image is empty")
        return digest.hexdigest(), byte_size

    @staticmethod
    def _decode_metadata(path: Path) -> tuple[str, str, int, int]:
        try:
            with Image.open(path) as probe:
                image_format = (probe.format or "").upper()
                if image_format not in FORMAT_DETAILS:
                    raise UnsupportedImage(
                        f"unsupported image format: {image_format or 'unknown'}"
                    )
                probe.verify()
            with Image.open(path) as decoded:
                decoded.seek(0)
                oriented = ImageOps.exif_transpose(decoded)
                try:
                    oriented.load()
                    width, height = oriented.size
                finally:
                    if oriented is not decoded:
                        oriented.close()
        except (UnidentifiedImageError, OSError) as exc:
            raise UnsupportedImage(
                "uploaded file is not a decodable supported image"
            ) from exc
        extension, mime_type = FORMAT_DETAILS[image_format]
        return extension, mime_type, width, height

    @staticmethod
    def _decode_and_thumbnail(
        path: Path,
    ) -> tuple[str, str, int, int, int, int, bytes]:
        try:
            with Image.open(path) as probe:
                image_format = (probe.format or "").upper()
                if image_format not in FORMAT_DETAILS:
                    raise UnsupportedImage(
                        f"unsupported image format: {image_format or 'unknown'}"
                    )
                probe.verify()

            with Image.open(path) as decoded:
                decoded.seek(0)
                oriented = ImageOps.exif_transpose(decoded)
                try:
                    oriented.load()
                    width, height = oriented.size
                    thumbnail_image = oriented.copy()
                finally:
                    if oriented is not decoded:
                        oriented.close()
            try:
                if height / max(width, 1) > 4:
                    if thumbnail_image.width > 320:
                        target_height = max(
                            1,
                            round(thumbnail_image.height * 320 / thumbnail_image.width),
                        )
                        resized = thumbnail_image.resize(
                            (320, target_height),
                            Image.Resampling.LANCZOS,
                        )
                        thumbnail_image.close()
                        thumbnail_image = resized
                    if thumbnail_image.height > 1280:
                        cropped = thumbnail_image.crop(
                            (0, 0, thumbnail_image.width, 1280)
                        )
                        thumbnail_image.close()
                        thumbnail_image = cropped
                else:
                    thumbnail_image.thumbnail(
                        (320, 320),
                        Image.Resampling.LANCZOS,
                    )
                if thumbnail_image.mode not in ("RGB", "RGBA"):
                    converted = thumbnail_image.convert("RGBA")
                    thumbnail_image.close()
                    thumbnail_image = converted
                thumbnail_width, thumbnail_height = thumbnail_image.size
                with BytesIO() as output:
                    thumbnail_image.save(output, format="WEBP", quality=80, method=4)
                    thumbnail = output.getvalue()
            finally:
                thumbnail_image.close()
        except (UnidentifiedImageError, OSError) as exc:
            raise UnsupportedImage("uploaded file is not a decodable supported image") from exc
        extension, mime_type = FORMAT_DETAILS[image_format]
        return (
            extension,
            mime_type,
            width,
            height,
            thumbnail_width,
            thumbnail_height,
            thumbnail,
        )
