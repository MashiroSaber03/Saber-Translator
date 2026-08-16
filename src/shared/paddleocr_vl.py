"""PaddleOCR-VL language-aware OCR prompt contract."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


PADDLEOCR_VL_DEFAULT_LANGUAGE: Final = "japanese"
PADDLEOCR_VL_LANGUAGE_NAMES: Final[Mapping[str, str]] = MappingProxyType({
    "japanese": "日语",
    "chinese": "简体中文",
    "chinese_cht": "繁体中文",
    "korean": "韩语",
    "english": "英语",
    "french": "法语",
    "german": "德语",
    "spanish": "西班牙语",
    "italian": "意大利语",
    "portuguese": "葡萄牙语",
    "dutch": "荷兰语",
    "polish": "波兰语",
    "thai": "泰语",
    "vietnamese": "越南语",
    "indonesian": "印尼语",
    "malay": "马来语",
    "russian": "俄语",
    "arabic": "阿拉伯语",
    "hindi": "印地语",
    "turkish": "土耳其语",
    "greek": "希腊语",
    "hebrew": "希伯来语",
})


def build_paddleocr_vl_prompt(source_language: str) -> str:
    """Build the language-specific instruction accepted by PaddleOCR-VL."""

    if not isinstance(source_language, str):
        raise ValueError("PaddleOCR-VL 源语言必须是字符串")
    try:
        language_name = PADDLEOCR_VL_LANGUAGE_NAMES[source_language]
    except KeyError as error:
        raise ValueError(
            f"PaddleOCR-VL 不支持源语言: {source_language}"
        ) from error
    return f"对图中的{language_name}进行OCR:"
