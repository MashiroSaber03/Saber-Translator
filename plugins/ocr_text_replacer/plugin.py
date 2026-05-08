from src.plugins.base import PluginBase
import copy


class OcrTextReplacerPlugin(PluginBase):
    plugin_id = "ocr_text_replacer"
    display_name = "OCR文本替换器"
    plugin_version = "1.0.0"
    plugin_author = "AI Agent"
    plugin_description = "在OCR后处理阶段，根据自定义映射表将特定文本精确替换为目标文本。"
    default_enabled = False
    supported_steps = ("ocr",)
    supported_modes = ("standard", "hq", "proofread")
    priority = 100
    failure_policy = "continue"

    def get_config_schema(self):
        return {
            "replace_rules": {
                "type": "text",
                "label": "替换规则",
                "default": "旧文本=新文本\n错误词=正确词",
                "description": "每行一条规则，格式为'旧文本=新文本'，仅支持精确替换"
            }
        }

    def _parse_rules(self):
        rules = {}
        rules_text = self.config.get("replace_rules", "")
        for line in rules_text.splitlines():
            line = line.strip()
            if "=" in line:
                parts = line.split("=", 1)
                old_text = parts[0].strip()
                new_text = parts[1].strip()
                if old_text:
                    rules[old_text] = new_text
        return rules

    def after_ocr(self, context, result):
        rules = self._parse_rules()
        if not rules:
            return None

        updated = copy.deepcopy(result)
        original_texts = updated.get("original_texts", [])
        ocr_results = updated.get("ocr_results", [])

        for i, text in enumerate(original_texts):
            for old_text, new_text in rules.items():
                if old_text in text:
                    text = text.replace(old_text, new_text)
            original_texts[i] = text

        for index, item in enumerate(ocr_results):
            if isinstance(item, dict) and index < len(original_texts):
                item["text"] = original_texts[index]

        updated["original_texts"] = original_texts
        updated["ocr_results"] = ocr_results

        return updated
