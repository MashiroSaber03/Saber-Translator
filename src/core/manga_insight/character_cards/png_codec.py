"""
角色卡 PNG 编解码。

SillyTavern 常见做法：在 PNG 文本块中写入 `chara`，内容为 base64(JSON)。
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, PngImagePlugin


class CharacterCardPngCodec:
    """角色卡 PNG 编解码器。"""

    @staticmethod
    def _make_placeholder(card: Dict[str, Any]) -> Image.Image:
        name = card.get("data", {}).get("name", "Character")
        image = Image.new("RGB", (512, 768), color=(245, 247, 252))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(24, 24), (488, 744)], outline=(93, 103, 204), width=3)
        draw.text((40, 56), f"Character Card\n{name}", fill=(20, 25, 45))
        return image

    @staticmethod
    def _load_base_image(base_image_path: Optional[str], card: Dict[str, Any]) -> Image.Image:
        if base_image_path:
            try:
                # 使用 with 确保源文件句柄及时关闭
                with Image.open(base_image_path) as src:
                    img = src.convert("RGB") if src.mode != "RGB" else src.copy()
                return img
            except Exception:
                pass
        return CharacterCardPngCodec._make_placeholder(card)

    @staticmethod
    def _encode_card_payload(card: Dict[str, Any]) -> str:
        json_text = json.dumps(card, ensure_ascii=False, separators=(",", ":"))
        return base64.b64encode(json_text.encode("utf-8")).decode("utf-8")

    @staticmethod
    def _decode_card_payload(payload: str) -> Dict[str, Any]:
        if not payload:
            return {}

        # 优先按 base64(JSON) 解码
        try:
            decoded = base64.b64decode(payload).decode("utf-8")
            return json.loads(decoded)
        except Exception:
            pass

        # 兜底：直接当 JSON
        try:
            return json.loads(payload)
        except Exception:
            return {}

    @classmethod
    def write_card_png(
        cls,
        card: Dict[str, Any],
        base_image_path: Optional[str] = None,
        mirror_ccv3: bool = True,
    ) -> bytes:
        """
        将角色卡写入 PNG。
        """
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
        """
        从 PNG 读取角色卡 JSON。
        """
        with Image.open(io.BytesIO(png_bytes)) as img:
            payload = img.info.get("chara") or img.info.get("ccv3") or ""
        return cls._decode_card_payload(payload)

    @classmethod
    def validate_roundtrip(
        cls,
        card: Dict[str, Any],
        png_bytes: bytes,
    ) -> tuple[bool, str]:
        """
        校验 PNG 写入后可回读同一对象结构。
        """
        decoded = cls.read_card_png(png_bytes)
        if not decoded:
            return False, "PNG 回读失败：未读取到角色卡数据"

        original = json.dumps(card, ensure_ascii=False, sort_keys=True)
        restored = json.dumps(decoded, ensure_ascii=False, sort_keys=True)
        if original != restored:
            return False, "PNG 回读校验失败：写入内容与读取内容不一致"
        return True, ""
