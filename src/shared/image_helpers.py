"""
图像处理辅助模块，提供图像处理相关的通用函数
"""

import base64
import io


def image_to_base64(image, format="PNG"):
    """
    将PIL图像对象转换为base64编码字符串
    
    Args:
        image: PIL图像对象
        format: 图像格式，默认为PNG
        
    Returns:
        base64编码的图像字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
