"""
配置数据模型

使用 dataclass 定义配置对象，减少函数参数传递的复杂度。
包含统一的气泡状态模型 BubbleState，用于前后端统一管理气泡渲染参数。

注意：以下类和函数已删除，因为它们是为已废弃的 process_image_translation 设计的：
- create_bubble_states_from_response (未被任何地方导入使用)
- bubble_states_from_api_request (未被任何地方导入使用)
- OCRConfig (只在 TranslationRequest 中使用)
- TranslationConfig (只在 TranslationRequest 中使用)
- InpaintingConfig (只在 TranslationRequest 中使用)
- RenderConfig (只在 TranslationRequest 和 create_bubble_states_from_response 中使用)
- TranslationRequest (只在已删除的 processing.py 中导入)

新的原子步骤架构使用 parallel_routes.py 中的 API 端点，
每个步骤独立处理，不需要这些整合配置类。
"""

from dataclasses import dataclass, field, fields as dataclass_fields, asdict
from typing import Optional, List, Tuple, Dict, Any
from src.shared import constants


# ============================================================
# BubbleState: 统一的气泡状态模型
# ============================================================

@dataclass
class BubbleState:
    """
    统一的单个气泡状态模型。
    
    所有翻译方法、编辑模式、渲染操作都只操作这个状态。
    前后端共用，通过 to_dict() 和 from_dict() 进行序列化。
    
    命名约定:
    - Python后端使用下划线命名 (snake_case)
    - 前端使用驼峰命名 (camelCase)
    - from_dict() 支持自动转换
    """
    # === 文本内容 ===
    original_text: str = ""           # 原文
    translated_text: str = ""         # 译文
    textbox_text: str = ""            # 文本框解释文本
    
    # === 坐标信息 ===
    coords: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    polygon: List[List[int]] = field(default_factory=list)  # 多边形顶点
    
    # === 渲染参数 ===
    font_size: int = constants.DEFAULT_FONT_SIZE
    font_family: str = constants.DEFAULT_FONT_RELATIVE_PATH
    text_direction: str = constants.DEFAULT_TEXT_DIRECTION  # "vertical" | "horizontal"
    auto_text_direction: str = constants.DEFAULT_TEXT_DIRECTION  # 自动检测的排版方向（始终在检测时计算，不受用户选择影响）
    text_color: str = constants.DEFAULT_TEXT_COLOR
    fill_color: str = constants.DEFAULT_FILL_COLOR       # 单个气泡的填充色
    rotation_angle: float = constants.DEFAULT_ROTATION_ANGLE  # 旋转角度（度）
    position_offset: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    
    # === 描边参数 ===
    stroke_enabled: bool = constants.DEFAULT_STROKE_ENABLED
    stroke_color: str = constants.DEFAULT_STROKE_COLOR
    stroke_width: int = constants.DEFAULT_STROKE_WIDTH
    
    # === 修复参数 ===
    inpaint_method: str = "solid"  # "solid" | "lama"
    
    # === 自动颜色提取（48px OCR 模型） ===
    auto_fg_color: Optional[Tuple[int, int, int]] = None  # 自动提取的前景色 RGB (0-255)
    auto_bg_color: Optional[Tuple[int, int, int]] = None  # 自动提取的背景色 RGB (0-255)
    color_confidence: float = 0.0  # 颜色提取置信度 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（用于JSON序列化，发送到前端）。
        使用驼峰命名以便前端直接使用。
        """
        return {
            # 文本内容
            "originalText": self.original_text,
            "translatedText": self.translated_text,
            "textboxText": self.textbox_text,
            # 坐标信息
            "coords": list(self.coords),
            "polygon": self.polygon,
            # 渲染参数
            "fontSize": self.font_size,
            "fontFamily": self.font_family,
            "textDirection": self.text_direction,
            "autoTextDirection": self.auto_text_direction,  # 自动检测的排版方向
            "textColor": self.text_color,
            "fillColor": self.fill_color,
            "rotationAngle": self.rotation_angle,
            "position": self.position_offset,
            # 描边参数
            "strokeEnabled": self.stroke_enabled,
            "strokeColor": self.stroke_color,
            "strokeWidth": self.stroke_width,
            # 修复参数
            "inpaintMethod": self.inpaint_method,
            # 自动颜色提取
            "autoFgColor": list(self.auto_fg_color) if self.auto_fg_color else None,
            "autoBgColor": list(self.auto_bg_color) if self.auto_bg_color else None,
            "colorConfidence": self.color_confidence,
        }
    
    def to_render_dict(self) -> Dict[str, Any]:
        """
        转换为后端渲染函数需要的字典格式（使用下划线命名）。
        用于兼容现有的 render_all_bubbles 函数。
        """
        return {
            "fontSize": self.font_size,
            "fontFamily": self.font_family,
            "text_direction": self.text_direction,
            "position_offset": self.position_offset,
            "text_color": self.text_color,
            "rotation_angle": self.rotation_angle,
            "stroke_enabled": self.stroke_enabled,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BubbleState":
        """
        从字典创建 BubbleState（支持前端驼峰命名自动转换）。
        
        Args:
            data: 来自前端的字典数据（可能使用驼峰命名）
            
        Returns:
            BubbleState 实例
        """
        # 驼峰命名 -> 下划线命名 映射
        camel_to_snake = {
            # 文本内容
            "originalText": "original_text",
            "translatedText": "translated_text",
            "textboxText": "textbox_text",
            "text": "translated_text",  # 兼容旧的 'text' 字段
            # 坐标信息
            "coords": "coords",
            "polygon": "polygon",
            # 渲染参数
            "fontSize": "font_size",
            "fontFamily": "font_family",
            "textDirection": "text_direction",
            "text_direction": "text_direction",  # 兼容后端命名
            "autoTextDirection": "auto_text_direction",  # 自动检测的排版方向
            "auto_text_direction": "auto_text_direction",  # 兼容后端命名
            "textColor": "text_color",
            "text_color": "text_color",  # 兼容后端命名
            "fillColor": "fill_color",
            "rotationAngle": "rotation_angle",
            "rotation_angle": "rotation_angle",  # 兼容后端命名
            "position": "position_offset",
            "position_offset": "position_offset",  # 兼容后端命名
            # 描边参数
            "strokeEnabled": "stroke_enabled",
            "stroke_enabled": "stroke_enabled",
            "strokeColor": "stroke_color",
            "stroke_color": "stroke_color",
            "strokeWidth": "stroke_width",
            "stroke_width": "stroke_width",
            # 修复参数
            "inpaintMethod": "inpaint_method",
            # 自动颜色提取
            "autoFgColor": "auto_fg_color",
            "auto_fg_color": "auto_fg_color",
            "autoBgColor": "auto_bg_color",
            "auto_bg_color": "auto_bg_color",
            "colorConfidence": "color_confidence",
            "color_confidence": "color_confidence",
        }
        
        # 转换字典键名
        converted = {}
        for key, value in data.items():
            snake_key = camel_to_snake.get(key, key)
            converted[snake_key] = value
        
        # 只保留 BubbleState 定义的字段
        valid_field_names = {f.name for f in dataclass_fields(cls)}
        filtered = {}
        for k, v in converted.items():
            if k in valid_field_names:
                filtered[k] = v
        
        # 处理 coords 可能是列表的情况
        if "coords" in filtered and isinstance(filtered["coords"], list):
            filtered["coords"] = tuple(filtered["coords"])
        
        # 处理颜色字段（列表转元组）
        if "auto_fg_color" in filtered and isinstance(filtered["auto_fg_color"], list):
            filtered["auto_fg_color"] = tuple(filtered["auto_fg_color"])
        if "auto_bg_color" in filtered and isinstance(filtered["auto_bg_color"], list):
            filtered["auto_bg_color"] = tuple(filtered["auto_bg_color"])
        
        return cls(**filtered)
    
    def update_from_dict(self, data: Dict[str, Any]) -> "BubbleState":
        """
        使用字典中的值更新当前状态（部分更新）。
        
        Args:
            data: 要更新的字段字典
            
        Returns:
            更新后的 self（支持链式调用）
        """
        temp = BubbleState.from_dict(data)
        for field_info in dataclass_fields(self):
            new_value = getattr(temp, field_info.name)
            # 只更新非默认值的字段
            default_instance = BubbleState()
            if new_value != getattr(default_instance, field_info.name):
                setattr(self, field_info.name, new_value)
        return self


# ============================================================
# 工具函数
# ============================================================

def bubble_states_to_api_response(states: List[BubbleState]) -> List[Dict]:
    """
    将 BubbleState 列表转换为 API 响应格式。
    
    Args:
        states: BubbleState 列表
        
    Returns:
        用于 JSON 响应的字典列表
    """
    return [state.to_dict() for state in states]
