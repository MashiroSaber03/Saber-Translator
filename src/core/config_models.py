"""
配置数据模型

包含后端渲染边界使用的统一气泡状态模型。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from src.shared import constants
from src.core.ocr_types import OcrResult


# ============================================================
# BubbleTextline: 最小文本行模型
# ============================================================

@dataclass
class BubbleTextline:
    polygon: List[List[int]] = field(default_factory=list)
    direction: str = "h"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polygon": self.polygon,
            "direction": self.direction,
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BubbleTextline":
        if not isinstance(data, dict):
            return cls()

        polygon = data.get("polygon", [])
        if not isinstance(polygon, list):
            polygon = []

        normalized_polygon = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                normalized_polygon.append([int(point[0]), int(point[1])])

        direction = str(data.get("direction", "h") or "h")
        if direction not in ("h", "v"):
            direction = "h"

        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        return cls(
            polygon=normalized_polygon,
            direction=direction,
            confidence=confidence,
        )


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
    - from_dict() 只接收当前 API 的驼峰字段
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

    # === 排版参数 ===
    line_spacing: float = constants.DEFAULT_LINE_SPACING  # 行间距倍数
    text_align: str = constants.DEFAULT_TEXT_ALIGN  # 'start' | 'center' | 'end'
    
    # === 修复参数 ===
    inpaint_method: str = constants.DEFAULT_INPAINT_METHOD  # "solid" | "lama"
    
    # === 自动颜色提取（48px OCR 模型） ===
    auto_fg_color: Optional[Tuple[int, int, int]] = None  # 自动提取的前景色 RGB (0-255)
    auto_bg_color: Optional[Tuple[int, int, int]] = None  # 自动提取的背景色 RGB (0-255)
    color_confidence: float = 0.0  # 颜色提取置信度 0-1

    # === 文本行信息 ===
    textlines: List[BubbleTextline] = field(default_factory=list)

    # === OCR 元数据 ===
    ocr_result: Optional[OcrResult] = None
    
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
            # 排版参数
            "lineSpacing": self.line_spacing,
            "textAlign": self.text_align,
            # 修复参数
            "inpaintMethod": self.inpaint_method,
            # 自动颜色提取
            "autoFgColor": list(self.auto_fg_color) if self.auto_fg_color else None,
            "autoBgColor": list(self.auto_bg_color) if self.auto_bg_color else None,
            "colorConfidence": self.color_confidence,
            "textlines": [textline.to_dict() for textline in self.textlines],
            "ocrResult": self.ocr_result.to_dict() if self.ocr_result else None,
        }
    
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BubbleState":
        """
        从当前 API 的驼峰字段创建 BubbleState。
        
        Args:
            data: 后端渲染服务生成的标准气泡字典
            
        Returns:
            BubbleState 实例
        """
        # 驼峰命名 -> 下划线命名 映射
        camel_to_snake = {
            # 文本内容
            "originalText": "original_text",
            "translatedText": "translated_text",
            "textboxText": "textbox_text",
            # 坐标信息
            "coords": "coords",
            "polygon": "polygon",
            # 渲染参数
            "fontSize": "font_size",
            "fontFamily": "font_family",
            "textDirection": "text_direction",
            "autoTextDirection": "auto_text_direction",  # 自动检测的排版方向
            "textColor": "text_color",
            "fillColor": "fill_color",
            "rotationAngle": "rotation_angle",
            "position": "position_offset",
            # 描边参数
            "strokeEnabled": "stroke_enabled",
            "strokeColor": "stroke_color",
            "strokeWidth": "stroke_width",
            # 排版参数
            "lineSpacing": "line_spacing",
            "textAlign": "text_align",
            # 修复参数
            "inpaintMethod": "inpaint_method",
            # 自动颜色提取
            "autoFgColor": "auto_fg_color",
            "autoBgColor": "auto_bg_color",
            "colorConfidence": "color_confidence",
            "textlines": "textlines",
            "ocrResult": "ocr_result",
        }
        
        # 转换字典键名
        converted = {}
        for key, value in data.items():
            snake_key = camel_to_snake.get(key)
            if snake_key is not None:
                converted[snake_key] = value
        
        filtered = converted
        
        # 处理 coords 可能是列表的情况
        if "coords" in filtered and isinstance(filtered["coords"], list):
            filtered["coords"] = tuple(filtered["coords"])
        
        # 处理颜色字段（列表转元组）
        if "auto_fg_color" in filtered and isinstance(filtered["auto_fg_color"], list):
            filtered["auto_fg_color"] = tuple(filtered["auto_fg_color"])
        if "auto_bg_color" in filtered and isinstance(filtered["auto_bg_color"], list):
            filtered["auto_bg_color"] = tuple(filtered["auto_bg_color"])
        if "textlines" in filtered and isinstance(filtered["textlines"], list):
            filtered["textlines"] = [BubbleTextline.from_dict(item) for item in filtered["textlines"]]
        if "ocr_result" in filtered and isinstance(filtered["ocr_result"], dict):
            filtered["ocr_result"] = OcrResult.from_dict(filtered["ocr_result"])
        
        return cls(**filtered)
    


# ============================================================
# 工具函数
# ============================================================
