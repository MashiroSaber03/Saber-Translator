"""
颜色提取器模块

基于 48px OCR 模型的颜色预测能力，实现智能颜色识别功能。

设计理念：
- **强制提取**：翻译时自动提取所有气泡的文字和背景颜色
- **灵活使用**：用户可选择使用自动颜色、默认颜色或自定义颜色
- **完整数据**：始终保留完整的颜色信息，编辑时可随时切换

使用方法:
    from src.core.color_extractor import get_color_extractor, ColorExtractionResult
    
    extractor = get_color_extractor()
    if extractor.initialize('cuda'):
        colors = extractor.extract_colors(image, bubble_coords, textlines_per_bubble)
        for i, color in enumerate(colors):
            print(f"气泡 {i}: fg={color.fg_color}, bg={color.bg_color}")
"""

import logging
from typing import List, Tuple, Dict, Optional
from PIL import Image

logger = logging.getLogger("ColorExtractor")

# 单例实例
_color_extractor_instance = None


class ColorExtractionResult:
    """颜色提取结果"""
    
    def __init__(
        self,
        fg_color: Optional[Tuple[int, int, int]] = None,
        bg_color: Optional[Tuple[int, int, int]] = None,
        confidence: float = 0.0
    ):
        """
        Args:
            fg_color: 前景色 RGB (0-255)，例如 (15, 20, 25)
            bg_color: 背景色 RGB (0-255)，例如 (248, 250, 252)
            confidence: 置信度 0-1
        """
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        """转换为字典（用于 API 响应）"""
        return {
            'fg_color': list(self.fg_color) if self.fg_color else None,
            'bg_color': list(self.bg_color) if self.bg_color else None,
            'confidence': self.confidence
        }
    
    @property
    def has_fg_color(self) -> bool:
        """是否有有效的前景色"""
        return self.fg_color is not None
    
    @property
    def has_bg_color(self) -> bool:
        """是否有有效的背景色"""
        return self.bg_color is not None
    
    def __repr__(self):
        return f"ColorExtractionResult(fg={self.fg_color}, bg={self.bg_color}, conf={self.confidence:.2f})"


class ColorExtractor:
    """
    48px 模型颜色提取器
    
    复用 48px OCR 模型的颜色预测能力，为每个气泡提取文字颜色和背景颜色。
    """
    
    def __init__(self):
        self._ocr_handler = None
        self._initialized = False
        self._device = 'cpu'
    
    def initialize(self, device: str = 'cpu') -> bool:
        """
        初始化颜色提取器
        
        Args:
            device: 计算设备 ('cpu', 'cuda', 'mps')
        
        Returns:
            是否初始化成功
        """
        if self._initialized:
            return True
        
        try:
            from src.interfaces.ocr_48px import get_48px_ocr_handler
            
            self._ocr_handler = get_48px_ocr_handler()
            if not self._ocr_handler.initialize(device):
                logger.error("48px OCR 模型初始化失败，颜色提取不可用")
                return False
            
            self._device = device
            self._initialized = True
            logger.info(f"✅ 颜色提取器已初始化 (设备: {device})")
            return True
            
        except Exception as e:
            logger.error(f"颜色提取器初始化失败: {e}", exc_info=True)
            return False
    
    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized
    
    def extract_colors(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: Optional[List[List[Dict]]] = None,
        extract_fg: bool = True,
        extract_bg: bool = True
    ) -> List[ColorExtractionResult]:
        """
        提取每个气泡的颜色
        
        这是设计文档中"强制提取"功能的入口。
        无论 extract_fg/extract_bg 参数如何设置，都会尝试提取所有颜色，
        参数只控制最终返回的颜色是否为 None。
        
        Args:
            image: PIL 图像
            bubble_coords: 气泡坐标列表 [(x1, y1, x2, y2), ...]
            textlines_per_bubble: 每个气泡对应的原始文本行列表（可选）
                格式: [[{'polygon': [[x,y], ...], 'direction': 'h'}, ...], ...]
            extract_fg: 是否返回前景色
            extract_bg: 是否返回背景色
        
        Returns:
            List[ColorExtractionResult]: 每个气泡的颜色提取结果
        """
        if not self._initialized or self._ocr_handler is None:
            logger.warning("颜色提取器未初始化，返回空结果")
            return [ColorExtractionResult() for _ in bubble_coords]
        
        if not bubble_coords:
            return []
        
        try:
            # 调用 48px OCR 的颜色提取方法
            raw_results = self._ocr_handler.extract_colors_for_bubbles(
                image, bubble_coords, textlines_per_bubble
            )
            
            # 根据参数过滤结果
            results = []
            for raw in raw_results:
                fg = raw.fg_color if extract_fg else None
                bg = raw.bg_color if extract_bg else None
                results.append(ColorExtractionResult(fg, bg, raw.confidence))
            
            return results
            
        except Exception as e:
            logger.error(f"颜色提取失败: {e}", exc_info=True)
            return [ColorExtractionResult() for _ in bubble_coords]
    
    def extract_colors_simple(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]]
    ) -> List[ColorExtractionResult]:
        """
        简化版颜色提取（不使用原始文本行）
        
        适用于没有文本行信息的场景。
        
        Args:
            image: PIL 图像
            bubble_coords: 气泡坐标列表
        
        Returns:
            颜色提取结果列表
        """
        return self.extract_colors(image, bubble_coords, None)


def get_color_extractor() -> ColorExtractor:
    """
    获取颜色提取器单例
    
    Returns:
        ColorExtractor 实例
    """
    global _color_extractor_instance
    if _color_extractor_instance is None:
        _color_extractor_instance = ColorExtractor()
    return _color_extractor_instance


def extract_bubble_colors(
    image: Image.Image,
    bubble_coords: List[Tuple[int, int, int, int]],
    textlines_per_bubble: Optional[List[List[Dict]]] = None,
    device: str = 'cpu'
) -> List[Dict]:
    """
    便捷函数：提取气泡颜色并返回字典格式
    
    这是供外部直接调用的便捷方法，自动处理初始化。
    
    Args:
        image: PIL 图像
        bubble_coords: 气泡坐标列表
        textlines_per_bubble: 每个气泡的文本行信息
        device: 计算设备
    
    Returns:
        [
            {
                'fg_color': [r, g, b] or None,
                'bg_color': [r, g, b] or None,
                'confidence': float
            },
            ...
        ]
    """
    extractor = get_color_extractor()
    
    if not extractor.is_initialized:
        if not extractor.initialize(device):
            logger.error("颜色提取器初始化失败")
            return [{'fg_color': None, 'bg_color': None, 'confidence': 0.0} for _ in bubble_coords]
    
    results = extractor.extract_colors(image, bubble_coords, textlines_per_bubble)
    return [r.to_dict() for r in results]
