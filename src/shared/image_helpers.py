"""
图像处理辅助模块，提供图像处理相关的通用函数
"""

import base64
import io

from PIL import Image


VISION_IMAGE_JPEG_QUALITY = 95


def encode_vision_image(
    image: Image.Image,
    *,
    compress: bool,
) -> tuple[str, str]:
    """Encode a PIL image for an OpenAI-compatible vision request."""

    if not isinstance(image, Image.Image):
        raise TypeError("待编码图片必须是 PIL Image")
    if not isinstance(compress, bool):
        raise TypeError("视觉模型图片压缩开关必须是布尔值")

    converted = image.convert("RGB") if compress and image.mode != "RGB" else image
    try:
        with io.BytesIO() as buffered:
            if compress:
                converted.save(
                    buffered,
                    format="JPEG",
                    quality=VISION_IMAGE_JPEG_QUALITY,
                    optimize=True,
                    subsampling=0,
                )
                media_type = "image/jpeg"
            else:
                converted.save(buffered, format="PNG")
                media_type = "image/png"
            encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    finally:
        if converted is not image:
            converted.close()
    return encoded, media_type


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
