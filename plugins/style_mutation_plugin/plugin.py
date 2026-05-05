import copy
from typing import Any, Dict

from src.plugins.base import PluginBase


class StyleMutationPlugin(PluginBase):
    plugin_id = "style_mutation_plugin"
    display_name = "样式调试插件"
    plugin_version = "1.0.0"
    plugin_author = "Codex"
    plugin_description = "用于验证颜色、修复和渲染相关 hook 的改写能力。"
    default_enabled = False
    supported_steps = (
        "detect",
        "color",
        "inpaint",
        "render",
    )
    supported_modes = (
        "standard",
        "hq",
        "proofread",
        "remove_text",
    )
    priority = 40

    def get_config_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            "force_detector_type": {
                "type": "text",
                "label": "强制检测器",
                "default": "",
                "description": "before_detect 可选地覆盖 detector_type；留空表示不改。",
            },
            "override_text_color": {
                "type": "text",
                "label": "渲染文字颜色",
                "default": "#ff0055",
                "description": "before_render 会把文字颜色改成这个值。",
            },
            "override_stroke_color": {
                "type": "text",
                "label": "描边颜色",
                "default": "#00aaff",
                "description": "before_render 会把描边颜色改成这个值。",
            },
            "override_fill_color": {
                "type": "text",
                "label": "修复填充色",
                "default": "#c8ffb0",
                "description": "before_inpaint 会把 solid 修复的填充色改成这个值。",
            },
            "override_palette": {
                "type": "boolean",
                "label": "覆盖颜色提取",
                "default": True,
                "description": "after_color 是否把颜色提取结果改成插件设定值。",
            },
            "inject_debug_bubble": {
                "type": "boolean",
                "label": "注入测试气泡",
                "default": False,
                "description": "after_detect 是否额外插入一个偏移测试框，用于直观看到检测结果被插件改写。",
            },
        }

    def before_detect(self, context, payload):
        detector = str(self.config.get("force_detector_type", "") or "").strip()
        if not detector:
            return None
        updated = copy.deepcopy(payload)
        updated["detector_type"] = detector
        return updated

    def after_detect(self, context, result):
        if not self.config.get("inject_debug_bubble", False):
            return None

        updated = copy.deepcopy(result)
        bubble_coords = updated.get("bubble_coords") or []
        if not bubble_coords:
            return None

        first = bubble_coords[0]
        if not isinstance(first, list) or len(first) < 4:
            return None

        x1, y1, x2, y2 = [int(value) for value in first[:4]]
        debug_coords = [x1 + 12, y1 + 12, x2 + 12, y2 + 12]
        bubble_coords.append(debug_coords)

        if isinstance(updated.get("bubble_angles"), list):
            updated["bubble_angles"].append(0)
        if isinstance(updated.get("bubble_polygons"), list):
            updated["bubble_polygons"].append([])
        if isinstance(updated.get("auto_directions"), list):
            directions = updated["auto_directions"]
            directions.append(directions[0] if directions else "v")
        if isinstance(updated.get("textlines_per_bubble"), list):
            updated["textlines_per_bubble"].append([])

        return updated

    def after_color(self, context, result):
        if not self.config.get("override_palette", True):
            return None
        updated = copy.deepcopy(result)
        for color_info in updated.get("colors") or []:
            if isinstance(color_info, dict):
                color_info["textColor"] = "#ff0055"
                color_info["bgColor"] = "#fff2a8"
        return updated

    def before_inpaint(self, context, payload):
        updated = copy.deepcopy(payload)
        override_fill = str(self.config.get("override_fill_color", "#c8ffb0") or "").strip()
        if override_fill:
            updated["fill_color"] = override_fill
        return updated

    def after_inpaint(self, context, result):
        return None

    def before_render(self, context, payload):
        updated = copy.deepcopy(payload)
        text_color = str(self.config.get("override_text_color", "#ff0055") or "").strip()
        stroke_color = str(self.config.get("override_stroke_color", "#00aaff") or "").strip()

        if text_color:
            updated["textColor"] = text_color
        if stroke_color:
            updated["strokeColor"] = stroke_color
            updated["strokeEnabled"] = True

        for bubble_state in updated.get("bubble_states") or []:
            if not isinstance(bubble_state, dict):
                continue
            if text_color:
                bubble_state["textColor"] = text_color
            if stroke_color:
                bubble_state["strokeEnabled"] = True
                bubble_state["strokeColor"] = stroke_color
                bubble_state["strokeWidth"] = max(int(bubble_state.get("strokeWidth", 2) or 2), 2)
        return updated

    def after_render(self, context, result):
        return None
