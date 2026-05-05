import copy
from typing import Any, Dict

from src.plugins.base import PluginBase


class TextMutationPlugin(PluginBase):
    plugin_id = "text_mutation_plugin"
    display_name = "文本改写测试插件"
    plugin_version = "1.0.0"
    plugin_author = "Codex"
    plugin_description = "用于验证 OCR / 翻译 / AI翻译 hook 是否真的能改写文本数据。"
    default_enabled = False
    supported_steps = (
        "ocr",
        "translate",
        "ai_translate",
    )
    supported_modes = (
        "standard",
        "hq",
        "proofread",
    )
    priority = 30

    def get_config_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            "ocr_suffix": {
                "type": "text",
                "label": "OCR 后缀",
                "default": "【OCR插件】",
                "description": "after_ocr 会把这个后缀追加到识别结果里。",
            },
            "source_prefix": {
                "type": "text",
                "label": "翻译前源文前缀",
                "default": "[PRE]",
                "description": "before_translate 会把这个前缀加到待翻译文本前。",
            },
            "translate_suffix": {
                "type": "text",
                "label": "普通翻译后缀",
                "default": "【TR插件】",
                "description": "after_translate 会把这个后缀追加到普通翻译结果。",
            },
            "textbox_suffix": {
                "type": "text",
                "label": "文本框翻译后缀",
                "default": "【TB插件】",
                "description": "after_translate 会把这个后缀追加到 textboxTexts。",
            },
            "ai_suffix": {
                "type": "text",
                "label": "AI翻译后缀",
                "default": "【AI插件】",
                "description": "after_ai_translate 会把这个后缀追加到 AI 翻译结果。",
            },
        }

    def after_ocr(self, context, result):
        updated = copy.deepcopy(result)
        suffix = str(self.config.get("ocr_suffix", "【OCR插件】") or "")
        updated["original_texts"] = [
            f"{text}{suffix}" for text in (updated.get("original_texts") or [])
        ]
        for index, item in enumerate(updated.get("ocr_results") or []):
            if isinstance(item, dict) and index < len(updated["original_texts"]):
                item["text"] = updated["original_texts"][index]
        return updated

    def before_translate(self, context, payload):
        updated = copy.deepcopy(payload)
        prefix = str(self.config.get("source_prefix", "[PRE]") or "")
        if prefix:
            updated["original_texts"] = [
                f"{prefix}{text}" for text in (updated.get("original_texts") or [])
            ]
        return updated

    def after_translate(self, context, result):
        updated = copy.deepcopy(result)
        translate_suffix = str(self.config.get("translate_suffix", "【TR插件】") or "")
        textbox_suffix = str(self.config.get("textbox_suffix", "【TB插件】") or "")
        updated["translated_texts"] = [
            f"{text}{translate_suffix}" for text in (updated.get("translated_texts") or [])
        ]
        updated["textbox_texts"] = [
            f"{text}{textbox_suffix}" if text else text
            for text in (updated.get("textbox_texts") or [])
        ]
        return updated

    def after_ai_translate(self, context, result):
        updated = copy.deepcopy(result)
        suffix = str(self.config.get("ai_suffix", "【AI插件】") or "")
        for image_item in updated.get("results") or []:
            if not isinstance(image_item, dict):
                continue
            for bubble in image_item.get("bubbles") or []:
                if isinstance(bubble, dict) and bubble.get("translated"):
                    bubble["translated"] = f"{bubble['translated']}{suffix}"
        return updated

