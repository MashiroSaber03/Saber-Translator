"""
配置验证模块

提供各种参数和配置的验证功能
"""

import re
import os
from typing import Any, Optional, List, Tuple
from src.shared.exceptions import ValidationException
from src.shared import constants


class Validator:
    """参数验证器基类"""
    
    @staticmethod
    def validate_language_code(lang_code: str, supported_langs: List[str] = None) -> bool:
        """
        验证语言代码
        
        Args:
            lang_code: 语言代码
            supported_langs: 支持的语言列表（可选）
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: 语言代码无效
        """
        if not lang_code or not isinstance(lang_code, str):
            raise ValidationException("语言代码不能为空")
        
        if supported_langs and lang_code not in supported_langs:
            raise ValidationException(
                f"不支持的语言代码: {lang_code}",
                {'supported': supported_langs}
            )
        
        return True
    
    @staticmethod
    def validate_color(color: str) -> bool:
        """
        验证颜色值（支持#RRGGBB格式）
        
        Args:
            color: 颜色值
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: 颜色格式无效
        """
        if not color or not isinstance(color, str):
            raise ValidationException("颜色值不能为空")
        
        # 验证十六进制颜色格式
        if not re.match(r'^#[0-9A-Fa-f]{6}$', color):
            raise ValidationException(
                f"无效的颜色格式: {color}，应为 #RRGGBB 格式"
            )
        
        return True
    
    @staticmethod
    def validate_font_size(font_size: Any) -> bool:
        """
        验证字体大小
        
        Args:
            font_size: 字体大小（整数或'auto'）
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: 字体大小无效
        """
        if font_size == 'auto':
            return True
        
        if not isinstance(font_size, (int, float)):
            raise ValidationException(
                f"字体大小必须是数字或'auto'，收到: {type(font_size).__name__}"
            )
        
        if font_size <= 0:
            raise ValidationException(
                f"字体大小必须大于0，收到: {font_size}"
            )
        
        if font_size > 500:
            raise ValidationException(
                f"字体大小过大（最大500），收到: {font_size}"
            )
        
        return True
    
    @staticmethod
    def validate_coordinates(coords: List[Tuple[int, int, int, int]]) -> bool:
        """
        验证气泡坐标列表
        
        Args:
            coords: 坐标列表 [(x1, y1, x2, y2), ...]
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: 坐标无效
        """
        if not isinstance(coords, list):
            raise ValidationException("坐标必须是列表")
        
        for i, coord in enumerate(coords):
            if not isinstance(coord, (list, tuple)) or len(coord) != 4:
                raise ValidationException(
                    f"坐标 {i} 格式错误，应为 (x1, y1, x2, y2)"
                )
            
            x1, y1, x2, y2 = coord
            
            if not all(isinstance(c, (int, float)) for c in coord):
                raise ValidationException(
                    f"坐标 {i} 必须包含数字"
                )
            
            if x1 >= x2 or y1 >= y2:
                raise ValidationException(
                    f"坐标 {i} 无效: x1 必须小于 x2，y1 必须小于 y2"
                )
            
            if any(c < 0 for c in coord):
                raise ValidationException(
                    f"坐标 {i} 包含负数"
                )
        
        return True
    
    @staticmethod
    def validate_api_key(api_key: Optional[str], required: bool = True) -> bool:
        """
        验证API密钥
        
        Args:
            api_key: API密钥
            required: 是否必需
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: API密钥无效
        """
        if required and not api_key:
            raise ValidationException("API密钥不能为空")
        
        if api_key and not isinstance(api_key, str):
            raise ValidationException("API密钥必须是字符串")
        
        if api_key and len(api_key) < 8:
            raise ValidationException("API密钥长度过短（至少8个字符）")
        
        return True
    
    @staticmethod
    def validate_file_path(path: str, must_exist: bool = False) -> bool:
        """
        验证文件路径
        
        Args:
            path: 文件路径
            must_exist: 文件是否必须存在
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: 路径无效
        """
        if not path or not isinstance(path, str):
            raise ValidationException("文件路径不能为空")
        
        if must_exist and not os.path.exists(path):
            raise ValidationException(
                f"文件不存在: {path}"
            )
        
        return True
    
    @staticmethod
    def validate_confidence_threshold(threshold: float) -> bool:
        """
        验证置信度阈值
        
        Args:
            threshold: 置信度阈值（0-1之间）
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: 阈值无效
        """
        if not isinstance(threshold, (int, float)):
            raise ValidationException("置信度阈值必须是数字")
        
        if not 0 <= threshold <= 1:
            raise ValidationException(
                f"置信度阈值必须在0-1之间，收到: {threshold}"
            )
        
        return True
    
    @staticmethod
    def validate_rpm_limit(rpm: int) -> bool:
        """
        验证RPM限制
        
        Args:
            rpm: 每分钟请求数（0表示无限制）
            
        Returns:
            是否有效
            
        Raises:
            ValidationException: RPM值无效
        """
        if not isinstance(rpm, int):
            raise ValidationException("RPM限制必须是整数")
        
        if rpm < 0:
            raise ValidationException("RPM限制不能为负数")
        
        if rpm > 1000:
            raise ValidationException(
                f"RPM限制过高（最大1000），收到: {rpm}"
            )
        
        return True


class TranslationRequestValidator:
    """翻译请求验证器"""
    
    @staticmethod
    def validate(data: dict) -> Tuple[bool, Optional[str]]:
        """
        验证翻译请求数据
        
        Args:
            data: 请求数据字典
            
        Returns:
            (是否有效, 错误消息)
        """
        try:
            # 验证必需字段
            required_fields = ['image', 'target_language', 'model_provider']
            for field in required_fields:
                if field not in data or not data[field]:
                    return False, f"缺少必需字段: {field}"
            
            # 验证语言代码
            target_lang = data.get('target_language')
            if target_lang:
                Validator.validate_language_code(target_lang)
            
            # 验证字体大小
            font_size = data.get('fontSize')
            if font_size:
                Validator.validate_font_size(font_size)
            
            # 验证颜色
            if 'text_color' in data:
                Validator.validate_color(data['text_color'])
            
            if 'fill_color' in data:
                Validator.validate_color(data['fill_color'])
            
            # 验证坐标
            if 'bubble_coords' in data and data['bubble_coords']:
                Validator.validate_coordinates(data['bubble_coords'])
            
            # 验证RPM限制
            if 'rpm_limit_translation' in data:
                Validator.validate_rpm_limit(int(data['rpm_limit_translation']))
            
            if 'rpm_limit_ai_vision_ocr' in data:
                Validator.validate_rpm_limit(int(data['rpm_limit_ai_vision_ocr']))
            
            return True, None
            
        except ValidationException as e:
            return False, str(e)
        except Exception as e:
            return False, f"验证时发生错误: {str(e)}"


# 便捷函数
def validate_or_raise(validator_func, *args, **kwargs):
    """
    执行验证，如果失败则抛出异常
    
    使用示例:
        validate_or_raise(Validator.validate_color, color_value)
    """
    return validator_func(*args, **kwargs)
