from copy import deepcopy


class Plugin:
    def after_ocr(self, context, data):
        result = deepcopy(dict(data))
        suffix = str(
            context.config.get("ocr_suffix", "【OCR v3】")
        )
        texts = result.get("originalTexts")
        if isinstance(texts, list):
            result["originalTexts"] = [
                f"{text}{suffix}" for text in texts
            ]
        return result

    def before_translate(self, context, data):
        result = deepcopy(dict(data))
        prefix = str(
            context.config.get("source_prefix", "[v3 源文]")
        )
        texts = result.get("originalTexts")
        if isinstance(texts, list):
            result["originalTexts"] = [
                f"{prefix}{text}" for text in texts
            ]
        return result

    def after_translate(self, context, data):
        result = deepcopy(dict(data))
        suffix = str(
            context.config.get(
                "translate_suffix",
                "【普通翻译 v3】",
            )
        )
        translations = result.get("translations")
        if isinstance(translations, list):
            result["translations"] = [
                f"{text}{suffix}" for text in translations
            ]
        return result

    def after_ai_translate(self, context, data):
        result = deepcopy(dict(data))
        suffix = str(
            context.config.get("ai_suffix", "【AI 翻译 v3】")
        )
        translations = result.get("translations")
        if isinstance(translations, list):
            result["translations"] = [
                f"{text}{suffix}" for text in translations
            ]
        return result
