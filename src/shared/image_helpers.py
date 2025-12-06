"""
图像处理辅助模块，提供图像处理相关的通用函数
"""

import base64
import io
from PIL import Image, ImageDraw


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


def base64_to_image(base64_string):
    """
    将base64编码字符串转换为PIL图像对象
    
    Args:
        base64_string: base64编码的图像字符串
        
    Returns:
        PIL图像对象
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def create_mask(image_size, bubble_coords, padding=0):
    """
    根据气泡坐标创建掩码图像
    
    Args:
        image_size: 原始图像大小，(width, height)
        bubble_coords: 气泡坐标列表，每个元素是 [x1, y1, x2, y2]
        padding: 可选的内边距，用于扩大掩码范围
        
    Returns:
        与原图同大小的掩码图像，气泡区域为白色(255)，其余区域为黑色(0)
    """
    mask = Image.new("L", image_size, 0)  # 创建黑色背景掩码
    draw = ImageDraw.Draw(mask)
    
    # 绘制白色矩形作为掩码区域
    for x1, y1, x2, y2 in bubble_coords:
        # 应用内边距
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_size[0], x2 + padding)
        y2 = min(image_size[1], y2 + padding)
        
        draw.rectangle([x1, y1, x2, y2], fill=255)
    
    return mask


def blend_images(base_image, overlay_image, mask):
    """
    根据掩码图像混合两个图像
    
    Args:
        base_image: 基础图像
        overlay_image: 覆盖图像
        mask: 掩码图像，白色区域使用覆盖图像，黑色区域使用基础图像
        
    Returns:
        混合后的图像
    """
    return Image.composite(overlay_image, base_image, mask)


def resize_image_to_fit(image, max_width, max_height):
    """
    调整图像大小以适应指定的最大宽度和高度，保持纵横比
    
    Args:
        image: PIL图像对象
        max_width: 最大宽度
        max_height: 最大高度
        
    Returns:
        调整大小后的图像
    """
    width, height = image.size
    
    # 计算宽高比
    ratio = min(max_width / width, max_height / height)
    
    if ratio < 1:  # 仅当需要缩小时才调整大小
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    return image  # 如果不需要缩小，则返回原图