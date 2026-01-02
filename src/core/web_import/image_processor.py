"""
网页漫画导入 - 图片预处理器

提供图片预处理功能：
- EXIF 自动旋转
- 压缩和缩放
- 格式转换
"""

import logging
import io
import base64
from typing import Optional, Tuple
from PIL import Image, ExifTags

logger = logging.getLogger("WebImport.ImageProcessor")


class ImageProcessor:
    """图片预处理器"""
    
    def __init__(self, config: dict = None):
        """
        初始化图片预处理器
        
        Args:
            config: 预处理配置
                - enabled: 是否启用预处理
                - autoRotate: 是否根据 EXIF 自动旋转
                - compression.enabled: 是否启用压缩
                - compression.quality: 压缩质量 (0-100)
                - compression.maxWidth: 最大宽度
                - compression.maxHeight: 最大高度
                - formatConvert.enabled: 是否转换格式
                - formatConvert.targetFormat: 目标格式
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
    
    def process(self, image_data: bytes, filename: str = None) -> Tuple[bytes, str]:
        """
        处理图片
        
        Args:
            image_data: 原始图片数据
            filename: 原始文件名（用于判断格式）
        
        Returns:
            (处理后的图片数据, 输出格式)
        """
        if not self.enabled:
            # 不处理，直接返回原始数据
            ext = self._get_extension(filename) if filename else 'png'
            return image_data, ext
        
        try:
            # 打开图片
            image = Image.open(io.BytesIO(image_data))
            original_format = image.format or 'PNG'
            
            # 1. EXIF 自动旋转
            if self.config.get('autoRotate', False):
                image = self._auto_rotate(image)
            
            # 2. 压缩和缩放
            compression = self.config.get('compression', {})
            if compression.get('enabled', False):
                image = self._resize_image(
                    image,
                    max_width=compression.get('maxWidth', 0),
                    max_height=compression.get('maxHeight', 0)
                )
            
            # 3. 确定输出格式
            format_config = self.config.get('formatConvert', {})
            if format_config.get('enabled', False):
                target_format = format_config.get('targetFormat', 'original')
                if target_format != 'original':
                    output_format = target_format.upper()
                else:
                    output_format = original_format
            else:
                output_format = original_format
            
            # 4. 转换为字节
            output_data, ext = self._image_to_bytes(
                image,
                output_format,
                quality=compression.get('quality', 85)
            )
            
            return output_data, ext
            
        except Exception as e:
            logger.error(f"图片处理失败: {e}")
            # 处理失败，返回原始数据
            ext = self._get_extension(filename) if filename else 'png'
            return image_data, ext
    
    def _auto_rotate(self, image: Image.Image) -> Image.Image:
        """
        根据 EXIF 信息自动旋转图片
        
        Args:
            image: PIL Image 对象
        
        Returns:
            旋转后的图片
        """
        try:
            # 获取 EXIF 信息
            exif = image._getexif()
            if exif is None:
                return image
            
            # 查找 Orientation 标签
            orientation_key = None
            for key, val in ExifTags.TAGS.items():
                if val == 'Orientation':
                    orientation_key = key
                    break
            
            if orientation_key is None or orientation_key not in exif:
                return image
            
            orientation = exif[orientation_key]
            
            # 根据方向旋转
            if orientation == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                image = image.rotate(180)
            elif orientation == 4:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
            elif orientation == 6:
                image = image.rotate(-90, expand=True)
            elif orientation == 7:
                image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            
            return image
            
        except Exception as e:
            logger.debug(f"EXIF 旋转处理失败: {e}")
            return image
    
    def _resize_image(
        self,
        image: Image.Image,
        max_width: int = 0,
        max_height: int = 0
    ) -> Image.Image:
        """
        缩放图片
        
        Args:
            image: PIL Image 对象
            max_width: 最大宽度 (0 = 不限制)
            max_height: 最大高度 (0 = 不限制)
        
        Returns:
            缩放后的图片
        """
        if max_width <= 0 and max_height <= 0:
            return image
        
        original_width, original_height = image.size
        
        # 计算缩放比例
        width_ratio = max_width / original_width if max_width > 0 else float('inf')
        height_ratio = max_height / original_height if max_height > 0 else float('inf')
        ratio = min(width_ratio, height_ratio, 1.0)  # 不放大，只缩小
        
        if ratio >= 1.0:
            return image
        
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _image_to_bytes(
        self,
        image: Image.Image,
        format: str,
        quality: int = 85
    ) -> Tuple[bytes, str]:
        """
        将 PIL Image 转换为字节
        
        Args:
            image: PIL Image 对象
            format: 输出格式
            quality: 压缩质量
        
        Returns:
            (图片字节, 文件扩展名)
        """
        buffer = io.BytesIO()
        
        # 处理格式
        format = format.upper()
        if format == 'JPG':
            format = 'JPEG'
        
        # JPEG 不支持透明度，需要转换
        if format == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        
        # 保存
        save_kwargs = {}
        if format in ('JPEG', 'WEBP'):
            save_kwargs['quality'] = quality
        if format == 'PNG':
            save_kwargs['compress_level'] = 6
        
        image.save(buffer, format=format, **save_kwargs)
        
        # 扩展名映射
        ext_map = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'GIF': 'gif'}
        ext = ext_map.get(format, 'png')
        
        return buffer.getvalue(), ext
    
    def _get_extension(self, filename: str) -> str:
        """从文件名获取扩展名"""
        if not filename:
            return 'png'
        ext = filename.rsplit('.', 1)[-1].lower()
        if ext in ('jpg', 'jpeg'):
            return 'jpg'
        elif ext in ('png', 'webp', 'gif'):
            return ext
        return 'png'
    
    @staticmethod
    def to_data_url(image_data: bytes, ext: str) -> str:
        """
        将图片数据转换为 Data URL
        
        Args:
            image_data: 图片字节数据
            ext: 文件扩展名
        
        Returns:
            Data URL 字符串
        """
        mime_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'webp': 'image/webp',
            'gif': 'image/gif'
        }
        mime = mime_map.get(ext.lower(), 'image/png')
        b64 = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime};base64,{b64}"
