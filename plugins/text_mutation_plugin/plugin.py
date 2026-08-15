from copy import deepcopy


class Plugin:
    def after_ocr(self, context, data):
        result = deepcopy(data)
        suffix = context.config["ocr_suffix"]
        result["originalTexts"] = [
            f"{text}{suffix}" for text in result["originalTexts"]
        ]
        return result

    def before_translate(self, context, data):
        result = deepcopy(data)
        prefix = context.config["source_prefix"]
        result["originalTexts"] = [
            f"{prefix}{text}" for text in result["originalTexts"]
        ]
        return result

    def after_translate(self, context, data):
        result = deepcopy(data)
        suffix = context.config["translate_suffix"]
        result["translations"] = [
            f"{text}{suffix}" for text in result["translations"]
        ]
        return result

    def after_ai_translate(self, context, data):
        result = deepcopy(data)
        suffix = context.config["ai_suffix"]
        result["translations"] = [
            f"{text}{suffix}" for text in result["translations"]
        ]
        return result
