import copy
import re

from src.plugins.base import PluginBase


class OcrTextNormalizerPlugin(PluginBase):
    plugin_id = "ocr_text_normalizer"
    display_name = "OCR Text Normalizer"
    plugin_version = "1.0.0"
    plugin_author = "AI Agent"
    plugin_description = "OCR后处理：去除首尾空格，将各类省略号统一为三个点。"
    default_enabled = False
    supported_steps = ("ocr",)
    supported_modes = ("standard", "hq", "proofread", "remove_text")
    priority = 100
    failure_policy = "continue"

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ellipsis(text: str) -> str:
        """将各种省略号形式统一为三个英文句点 '...'。

        覆盖的情况：
        - 中文省略号：……
        - Unicode 省略号：…
        - 连续句号（3个及以上）：。。。  ....
        - 连续中点（3个及以上）：・・・  ・・・
        """
        if not text:
            return text

        # 1. 先把中文省略号 …… 替换成占位，避免后续被重复处理
        text = text.replace("……", "...")

        # 2. 单个 Unicode 省略号 …
        text = text.replace("…", "...")

        # 3. 3个及以上连续句号（中/英）统一为 ...
        text = re.sub(r"[。.]{3,}", "...", text)

        # 4. 3个及以上连续中点 ・　替换为 ...
        text = re.sub(r"[・・]{3,}", "...", text)

        return text

    @staticmethod
    def _normalize_text(text: str) -> str:
        """依次执行：去首尾空格 → 省略号统一。"""
        if not text:
            return text
        text = text.strip()
        text = OcrTextNormalizerPlugin._normalize_ellipsis(text)
        return text

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def after_ocr(self, context, result):
        updated = copy.deepcopy(result)

        original_texts = updated.get("original_texts", [])
        normalized_texts = [self._normalize_text(t) for t in original_texts]
        updated["original_texts"] = normalized_texts

        ocr_results = updated.get("ocr_results", [])
        for index, item in enumerate(ocr_results):
            if isinstance(item, dict) and index < len(normalized_texts):
                item["text"] = normalized_texts[index]

        return updated
