"""Fixed vertical font specimens and direction-preserving reference views."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

from fontTools.ttLib import TTFont, TTLibError
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError, __version__ as pillow_version

SAMPLE_TEXT = "永明日月山川大小天地人心春雨星空"
SAMPLE_LAYOUT = (448, 43, 50, 8, 0)  # Canvas, font size, cell size, rows, font face
SAMPLE_VERSION = 1
REFERENCE_VERSION = 2
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 20_000_000


def cache_key(*parts: object) -> str:
    serialized = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def mark_cache_used(path: Path) -> None:
    with suppress(OSError):
        os.utime(path)


def png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def read_reference(data: bytes, *, background: str = "white") -> Image.Image:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("请选择不超过 12 MiB 的文字图片")
    try:
        with Image.open(BytesIO(data), formats=["PNG", "JPEG", "WEBP", "BMP", "GIF"]) as image:
            if image.width < 2 or image.height < 2 or image.width * image.height > MAX_PIXELS:
                raise ValueError("文字选区尺寸无效，或超过 2000 万像素")
            if getattr(image, "is_animated", False):
                raise ValueError("请选择静态文字图片")
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, background)
            canvas.alpha_composite(rgba)
            return canvas.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("无法读取参考图片，请重新选取") from exc


def trim_reference(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int]]:
    width, height = image.size
    corners = [image.getpixel((x, y)) for x in (0, width - 1) for y in (0, height - 1)]
    background = tuple(int(median(values)) for values in zip(*corners))
    delta = ImageChops.difference(image, Image.new("RGB", image.size, background))
    red, green, blue = delta.split()
    contrast = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    threshold = max(1, min(20, contrast.getextrema()[1] // 4))
    bounds = contrast.point(lambda value: 255 if value >= threshold else 0).getbbox()
    if bounds is None:
        raise ValueError("参考选区中没有可比较的文字，请重新框选")
    return image.crop(bounds), background


def reference_views(image: Image.Image) -> tuple[bytes, ...]:
    image, background = trim_reference(image)
    width, height = image.size
    horizontal = width >= height
    length = width if horizontal else height
    window = max(1, (length + 1) // 2)
    offsets = (0, (length - window) // 2, length - window)
    crops = [
        image.crop((start, 0, start + window, height) if horizontal
                   else (0, start, width, start + window))
        for start in offsets
    ]
    return tuple(
        png(ImageOps.pad(view, (size, size), method=Image.Resampling.LANCZOS, color=background))
        for view, size in zip((image, *crops), (448, 256, 256, 256))
    )


class UnsupportedSample(ValueError):
    """The first face cannot produce the complete fixed specimen."""


def sample_for_font(path: Path, cache: Path) -> tuple[str, Path]:
    info = path.stat()
    state = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    sample = _sample_for_font(path, cache, state)
    if not sample[1].is_file():
        _sample_for_font.cache_clear()
        return _sample_for_font(path, cache, state)
    mark_cache_used(sample[1])
    return sample


@lru_cache(maxsize=256)
def _sample_for_font(path: Path, cache: Path, _file_state: tuple[int, ...]) -> tuple[str, Path]:
    canvas, size, cell, rows, face = SAMPLE_LAYOUT
    # Read a single snapshot so the hash, cmap and rendered pixels all have the
    # same origin even when a font file is replaced manually.
    data = path.read_bytes()
    fingerprint = hashlib.sha256(data).hexdigest()
    try:
        with TTFont(BytesIO(data), fontNumber=face, lazy=True) as definition:
            cmap = definition.getBestCmap() or {}
            if any(cmap.get(ord(char)) in (None, ".notdef") for char in SAMPLE_TEXT):
                raise UnsupportedSample("不支持完整的固定样本文字")
            axes = {axis.axisTag: axis.defaultValue for axis in definition["fvar"].axes} if "fvar" in definition else {}
            key = cache_key(
                fingerprint, SAMPLE_TEXT, SAMPLE_LAYOUT, SAMPLE_VERSION,
                pillow_version, ImageFont.core.freetype2_version, axes,
            )
            target = cache / (key + ".png")
            if target.is_file():
                return key, target
            if definition.flavor is not None:
                # Pillow's FreeType loader expects an SFNT, not a web wrapper.
                definition.flavor = None
                unpacked = BytesIO()
                definition.save(unpacked)
                data = unpacked.getvalue()
        font = ImageFont.truetype(BytesIO(data), size=size, index=face)
    except (TTLibError, OSError, ImportError) as exc:
        raise UnsupportedSample("无法用该字体渲染固定样张") from exc

    image = Image.new("RGB", (canvas, canvas), "white")
    draw = ImageDraw.Draw(image)
    for index, char in enumerate(SAMPLE_TEXT):
        if font.getmask(char).getbbox() is None:
            raise UnsupportedSample("固定样本文字包含空字形")
        column, row = divmod(index, rows)
        x = canvas // 2 + cell // 2 - column * cell
        y = (canvas - rows * cell) // 2 + cell // 2 + row * cell
        draw.text((x, y), char, fill="black", font=font, anchor="mm")
    if image.getextrema()[0] == (255, 255):
        raise UnsupportedSample("固定样张没有可见文字")
    cache.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as pending:
        path_to_publish = Path(pending.name)
        try:
            pending.write(png(image))
        except BaseException:
            pending.close()
            path_to_publish.unlink(missing_ok=True)
            raise
    try:
        path_to_publish.replace(target)
    finally:
        path_to_publish.unlink(missing_ok=True)
    return key, target
