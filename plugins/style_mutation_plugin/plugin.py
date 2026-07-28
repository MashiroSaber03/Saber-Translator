from copy import deepcopy


class Plugin:
    def before_detect(self, context, data):
        result = deepcopy(dict(data))
        detector = str(
            context.config.get("force_detector_type", "")
        ).strip()
        if detector:
            result["detectorType"] = detector
        return result

    def after_color(self, context, data):
        result = deepcopy(dict(data))
        if not context.config.get("override_palette", True):
            return result
        text_color = str(
            context.config.get("override_text_color", "#ff0055")
        )
        for color in result.get("colors", []):
            if isinstance(color, dict):
                color["textColor"] = text_color
        return result

    def before_inpaint(self, context, data):
        result = deepcopy(dict(data))
        result["fillColor"] = str(
            context.config.get(
                "override_fill_color",
                "#c8ffb0",
            )
        )
        return result

    def before_render(self, context, data):
        result = deepcopy(dict(data))
        text_color = str(
            context.config.get("override_text_color", "#ff0055")
        )
        stroke_color = str(
            context.config.get("override_stroke_color", "#00aaff")
        )
        result["textColor"] = text_color
        result["strokeEnabled"] = True
        result["strokeColor"] = stroke_color
        for bubble in result.get("bubbles", []):
            if isinstance(bubble, dict):
                bubble["textColor"] = text_color
                bubble["strokeEnabled"] = True
                bubble["strokeColor"] = stroke_color
        return result
