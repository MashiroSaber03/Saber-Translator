"""
路径处理辅助模块，提供文件路径处理相关的通用函数
"""

import os
import sys
import logging

from src.shared.constants import DEFAULT_FONT_RELATIVE_PATH

logger = logging.getLogger("PathHelpers")


def get_project_root():
    """
    获取源码项目根目录。

    Returns:
        源码项目根目录绝对路径
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_resource_root():
    """
    获取只读资源根目录。

    开发环境下返回源码项目根目录；PyInstaller 打包环境下返回 bundle 内部资源目录。
    """
    try:
        # PyInstaller创建的临时/内部资源目录
        base_path = sys._MEIPASS
        logger.debug(f"打包环境中，资源基础路径: {base_path}")
        return base_path
    except Exception:
        base_path = get_project_root()
        logger.debug(f"开发环境中，资源基础路径: {base_path}")
        return base_path


def resource_path(relative_path):
    """
    获取资源的绝对路径，适用于开发环境和PyInstaller打包环境
    
    Args:
        relative_path: 相对路径
    
    Returns:
        资源的绝对路径
    """
    abs_path = os.path.join(get_resource_root(), relative_path)
    logger.debug(f"资源路径解析: '{relative_path}' -> '{abs_path}'")
    return abs_path


def get_font_path(font_path):
    """
    获取字体的绝对路径
    
    Args:
        font_path: 字体路径，可能是相对路径也可能是绝对路径
        
    Returns:
        字体的绝对路径
    """
    if not font_path:
        # 如果未提供字体，使用默认字体
        return resource_path(DEFAULT_FONT_RELATIVE_PATH)
    
    builtin_fonts = os.path.join('src', 'backend_v2', 'resources', 'fonts')

    # 兼容旧文档中的展示路径，实际统一解析到 v2 只读字体资源。
    if font_path.startswith('static/fonts/'):
        return resource_path(os.path.join(builtin_fonts, os.path.basename(font_path)))
    elif font_path.startswith('static/'):
        return resource_path(os.path.join(builtin_fonts, os.path.basename(font_path)))
    elif font_path.startswith('fonts/'):
        return resource_path(os.path.join(builtin_fonts, os.path.basename(font_path)))
    elif os.path.exists(font_path):
        # 如果路径存在，直接返回
        return font_path
    else:
        # 尝试在当前路径下查找
        app_dir_path = resource_path(os.path.basename(font_path))
        if os.path.exists(app_dir_path):
            return app_dir_path
            
        # 尝试在fonts目录下查找
        fonts_dir_path = resource_path(
            os.path.join(builtin_fonts, os.path.basename(font_path))
        )
        if os.path.exists(fonts_dir_path):
            return fonts_dir_path
    
    # 如果所有尝试都失败，返回默认字体
    logger.warning(f"未找到字体 {font_path}，使用默认字体")
    return resource_path(DEFAULT_FONT_RELATIVE_PATH)
