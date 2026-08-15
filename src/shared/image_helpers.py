"""
图像处理辅助模块，提供图像处理相关的通用函数
"""

import base64
import io

from PIL import Image


def image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    """
    将PIL图像对象转换为base64编码字符串
    
    Args:
        image: PIL图像对象
        image_format: 图像格式，默认为 PNG
        
    Returns:
        base64编码的图像字符串
    """
    if not isinstance(image, Image.Image):
        raise TypeError("待编码图片必须是 PIL Image")
    if not isinstance(image_format, str) or not image_format.strip():
        raise ValueError("图片编码格式不能为空")
    with io.BytesIO() as buffered:
        image.save(buffered, format=image_format.strip())
        return base64.b64encode(buffered.getvalue()).decode("ascii")


def image_to_rgb_array(image: Image.Image):
    """Copy a PIL image into a three-channel NumPy array."""

    if not isinstance(image, Image.Image):
        raise TypeError("待转换图片必须是 PIL Image")
    import numpy as np

    converted = image.convert("RGB")
    try:
        return np.array(converted)
    finally:
        if converted is not image:
            converted.close()
