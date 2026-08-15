from copy import deepcopy


class Plugin:
    def before_detect(self, context, data):
        result = deepcopy(data)
        detector = context.config["force_detector_type"]
        if detector:
            detector_config = result["detectorConfig"]
            detector_config["detector_type"] = detector
        return result

    def after_color(self, context, data):
        result = deepcopy(data)
        if not context.config["override_palette"]:
            return result
        text_color = context.config["override_text_color"]
        for color in result["colors"]:
            color["fgColor"] = [
                int(text_color[index:index + 2], 16)
                for index in (1, 3, 5)
            ]
        return result

    def before_inpaint(self, context, data):
        result = deepcopy(data)
        if result["method"] == "solid":
            result["fillColor"] = context.config["override_fill_color"]
        return result

    def before_render(self, context, data):
        result = deepcopy(data)
        text_color = context.config["override_text_color"]
        stroke_color = context.config["override_stroke_color"]
        for bubble in result["bubbles"]:
            bubble["textColor"] = text_color
            bubble["strokeEnabled"] = True
            bubble["strokeColor"] = stroke_color
        return result
