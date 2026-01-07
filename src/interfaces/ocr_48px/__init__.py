"""
48px OCR 模块

支持日中英韩等多语言识别，并提供文本颜色提取功能

使用方法:
    from src.interfaces.ocr_48px import get_48px_ocr_handler, Model48pxOCR, ColorExtractionResult
    
    handler = get_48px_ocr_handler()
    handler.initialize('cuda')
    
    # OCR 识别
    texts = handler.recognize_text(image, bubble_coords, textlines_per_bubble)
    
    # 颜色提取
    colors = handler.extract_colors_for_bubbles(image, bubble_coords, textlines_per_bubble)
    for color in colors:
        print(f"前景色: {color.fg_color}, 背景色: {color.bg_color}, 置信度: {color.confidence}")
"""

from .interface import get_48px_ocr_handler, Model48pxOCR, ColorExtractionResult
from .core import OCR

__all__ = ['get_48px_ocr_handler', 'Model48pxOCR', 'OCR', 'ColorExtractionResult']
