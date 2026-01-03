"""
48px OCR 模块

基于 manga-image-translator 的 48px OCR 模型
支持日中英韩等多语言识别

使用方法:
    from src.interfaces.ocr_48px import get_48px_ocr_handler, Model48pxOCR
    
    handler = get_48px_ocr_handler()
    handler.initialize('cuda')
    texts = handler.recognize_text(image, bubble_coords, textlines_per_bubble)
"""

from .interface import get_48px_ocr_handler, Model48pxOCR
from .core import OCR

__all__ = ['get_48px_ocr_handler', 'Model48pxOCR', 'OCR']
