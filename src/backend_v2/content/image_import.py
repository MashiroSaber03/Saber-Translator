"""One-page-at-a-time image validation, thumbnailing, and publication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import BinaryIO
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


@dataclass(frozen=True, slots=True)
class ImportSafetyLimits:
    max_image_bytes: int = 128 * 1024 * 1024
    stream_chunk_bytes: int = 1024 * 1024


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
        upload: BinaryIO,
        lease_id: str,
        owner_token: str,
        idempotency_key: str,
    ) -> tuple[dict[str, object], bool]:
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("Idempotency-Key is required and must be at most 200 characters")
        if not owner_token:
            raise ValueError("Import-Lease-Token is required")
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

            self.repository.validate_and_renew_import_lease(
                chapter_id=chapter_id,
                lease_id=lease_id,
                owner_token=owner_token,
            )
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
            return self.repository.append_page(
                chapter_id=chapter_id,
                requested_logical_path=logical_path,
                source=source_asset,
                thumbnail=thumbnail_asset,
                idempotency_scope=scope,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                lease_id=lease_id,
                owner_token=owner_token,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def _copy_upload(self, upload: BinaryIO, destination: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        byte_size = 0
        with destination.open("xb") as output:
            while True:
                chunk = upload.read(self.limits.stream_chunk_bytes)
                if not chunk:
                    break
                byte_size += len(chunk)
                if byte_size > self.limits.max_image_bytes:
                    raise ValueError("image exceeds the configured single-file byte limit")
                digest.update(chunk)
                output.write(chunk)
        if byte_size == 0:
            raise ValueError("uploaded image is empty")
        return digest.hexdigest(), byte_size

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
                oriented.load()
                width, height = oriented.size
                thumbnail_image = oriented.copy()
                if height / max(width, 1) > 4:
                    if thumbnail_image.width > 320:
                        target_height = max(
                            1,
                            round(thumbnail_image.height * 320 / thumbnail_image.width),
                        )
                        thumbnail_image = thumbnail_image.resize(
                            (320, target_height),
                            Image.Resampling.LANCZOS,
                        )
                    if thumbnail_image.height > 1280:
                        thumbnail_image = thumbnail_image.crop(
                            (0, 0, thumbnail_image.width, 1280)
                        )
                else:
                    thumbnail_image.thumbnail(
                        (320, 320),
                        Image.Resampling.LANCZOS,
                    )
                if thumbnail_image.mode not in ("RGB", "RGBA"):
                    thumbnail_image = thumbnail_image.convert("RGBA")
                thumbnail_width, thumbnail_height = thumbnail_image.size
                output = BytesIO()
                thumbnail_image.save(output, format="WEBP", quality=80, method=4)
                thumbnail_image.close()
                if oriented is not decoded:
                    oriented.close()
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
            output.getvalue(),
        )
