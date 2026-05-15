"""
PNG codec for Character Studio exports.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, PngImagePlugin


class CharacterStudioPngCodec:
    """Encode/decode chara_card payloads in PNG metadata."""

    @staticmethod
    def _make_placeholder(card: Dict[str, Any]) -> Image.Image:
        name = card.get("data", {}).get("name") or "Character"
        image = Image.new("RGB", (768, 1152), color=(247, 249, 252))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((28, 28, 740, 1124), radius=32, outline=(44, 82, 130), width=4)
        draw.text((60, 70), f"Character Studio\n{name}", fill=(21, 45, 74))
        return image

    @staticmethod
    def _load_base_image(base_image_path: Optional[str], card: Dict[str, Any]) -> Image.Image:
        if base_image_path:
            try:
                with Image.open(base_image_path) as src:
                    img = src.convert("RGB") if src.mode != "RGB" else src.copy()
                return img
            except Exception:
                pass
        return CharacterStudioPngCodec._make_placeholder(card)

    @staticmethod
    def _encode_card_payload(card: Dict[str, Any]) -> str:
        text = json.dumps(card, ensure_ascii=False, separators=(",", ":"))
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")

    @staticmethod
    def _decode_card_payload(payload: str) -> Dict[str, Any]:
        if not payload:
            return {}
        try:
            decoded = base64.b64decode(payload).decode("utf-8")
            return json.loads(decoded)
        except Exception:
            pass
        try:
            return json.loads(payload)
        except Exception:
            return {}

    @classmethod
    def write_card_png(
        cls,
        card: Dict[str, Any],
        *,
        base_image_path: Optional[str] = None,
        mirror_ccv3: bool = True,
    ) -> bytes:
        image = cls._load_base_image(base_image_path, card)
        payload = cls._encode_card_payload(card)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("chara", payload)
        if mirror_ccv3:
            png_info.add_text("ccv3", payload)

        output = io.BytesIO()
        image.save(output, format="PNG", pnginfo=png_info)
        return output.getvalue()

    @classmethod
    def read_card_png(cls, png_bytes: bytes) -> Dict[str, Any]:
        with Image.open(io.BytesIO(png_bytes)) as img:
            payload = img.info.get("chara") or img.info.get("ccv3") or ""
        return cls._decode_card_payload(payload)
