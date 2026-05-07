from src.plugins.base import PluginBase
import copy


class TranslateWordReplacerPlugin(PluginBase):
    plugin_id = "translate_word_replacer"
    display_name = "译文名词替换"
    plugin_version = "1.0.0"
    plugin_author = "AI Agent"
    plugin_description = "普通翻译后处理插件，将译文中的指定词替换为目标词（区分大小写，包含匹配）。"
    default_enabled = False
    supported_steps = ("translate",)
    supported_modes = ("standard", "hq", "proofread", "remove_text")
    priority = 100
    failure_policy = "continue"

    def get_config_schema(self):
        return {
            "replacements": {
                "type": "dict",
                "label": "替换词表",
                "default": {
                    "Slime": "史莱姆",
                    "Dragon": "巨龙"
                },
                "description": "键为要替换的原词，值为替换后的词。区分大小写，包含即替换。"
            }
        }

    def after_translate(self, context, result):
        replacements = self.config.get("replacements", {})
        if not replacements:
            return None

        translated_texts = result.get("translated_texts", [])
        if not translated_texts:
            return None

        updated_texts = []
        for text in translated_texts:
            new_text = text
            for src_word, dst_word in replacements.items():
                if src_word and src_word in new_text:
                    new_text = new_text.replace(src_word, dst_word)
            updated_texts.append(new_text)

        updated = copy.deepcopy(result)
        updated["translated_texts"] = updated_texts
        return updated
