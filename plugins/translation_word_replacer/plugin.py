import copy
import json
from src.plugins.base import PluginBase


class TranslationWordReplacerPlugin(PluginBase):
    plugin_id = "translation_word_replacer"
    display_name = "Translation Word Replacer"
    plugin_version = "1.0.0"
    plugin_author = "AI Agent"
    plugin_description = "Replace specific words in translated texts based on a configurable mapping."
    default_enabled = False
    supported_steps = ("translate",)
    supported_modes = ("standard", "hq", "proofread")
    priority = 100
    failure_policy = "continue"

    def get_config_schema(self):
        return {
            "replacement_map": {
                "type": "text",
                "label": "替换词典",
                "default": "{\"原词\": \"替换词\"}",
                "description": "JSON 格式的词汇替换映射表，例如：{\"hello\": \"你好\", \"world\": \"世界\"}"
            },
            "case_sensitive": {
                "type": "boolean",
                "label": "区分大小写",
                "default": True,
                "description": "替换时是否区分英文字母大小写"
            }
        }

    def _get_replacement_map(self):
        raw_map = self.config.get("replacement_map", "{}")
        try:
            mapping = json.loads(raw_map)
            if isinstance(mapping, dict):
                return mapping
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    def after_translate(self, context, result):
        replacement_map = self._get_replacement_map()
        if not replacement_map:
            return None

        case_sensitive = self.config.get("case_sensitive", True)
        updated = copy.deepcopy(result)

        translated_texts = updated.get("translated_texts", [])
        new_texts = []

        for text in translated_texts:
            if not isinstance(text, str):
                new_texts.append(text)
                continue
            for old_word, new_word in replacement_map.items():
                if case_sensitive:
                    text = text.replace(old_word, new_word)
                else:
                    text = text.replace(old_word.lower(), new_word.lower())
                    text = text.replace(old_word.upper(), new_word.upper())
                    text = text.replace(old_word.capitalize(), new_word.capitalize())
            new_texts.append(text)

        updated["translated_texts"] = new_texts

        if "textbox_texts" in updated and isinstance(updated["textbox_texts"], list):
            new_tb_texts = []
            for text in updated["textbox_texts"]:
                if not isinstance(text, str):
                    new_tb_texts.append(text)
                    continue
                for old_word, new_word in replacement_map.items():
                    if case_sensitive:
                        text = text.replace(old_word, new_word)
                    else:
                        text = text.replace(old_word.lower(), new_word.lower())
                        text = text.replace(old_word.upper(), new_word.upper())
                        text = text.replace(old_word.capitalize(), new_word.capitalize())
                new_tb_texts.append(text)
            updated["textbox_texts"] = new_tb_texts

        return updated
