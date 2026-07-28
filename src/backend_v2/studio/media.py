"""Character-card PNG metadata codec owned by the v2 boundary."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageOps, PngImagePlugin


def read_card_png(payload: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(payload)) as image:
        encoded = image.info.get("chara") or image.info.get("ccv3") or ""
    if not encoded:
        return {}
    for decoder in (
        lambda value: base64.b64decode(value, validate=True).decode("utf-8"),
        lambda value: value,
    ):
        try:
            decoded = json.loads(decoder(str(encoded)))
        except (ValueError, TypeError, UnicodeDecodeError):
            continue
        if isinstance(decoded, dict):
            return decoded
    return {}


def write_card_png(
    card: Mapping[str, Any],
    *,
    base_image_path: Path | None,
) -> bytes:
    image = _base_image(card, base_image_path=base_image_path)
    encoded = base64.b64encode(
        json.dumps(
            dict(card),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("chara", encoded)
    metadata.add_text("ccv3", encoded)
    output = BytesIO()
    try:
        image.save(output, format="PNG", pnginfo=metadata)
    finally:
        image.close()
    return output.getvalue()


def _base_image(
    card: Mapping[str, Any],
    *,
    base_image_path: Path | None,
) -> Image.Image:
    if base_image_path is not None and base_image_path.is_file():
        try:
            with Image.open(base_image_path) as source:
                oriented = ImageOps.exif_transpose(source)
                result = oriented.convert("RGB")
                if oriented is not source:
                    oriented.close()
                return result
        except OSError:
            pass
    data = card.get("data")
    name = (
        str(data.get("name") or "Character")
        if isinstance(data, Mapping)
        else "Character"
    )
    image = Image.new("RGB", (768, 1152), color=(247, 249, 252))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (28, 28, 740, 1124),
        radius=32,
        outline=(44, 82, 130),
        width=4,
    )
    draw.text(
        (60, 70),
        f"Character Studio\n{name}",
        fill=(21, 45, 74),
    )
    return image
