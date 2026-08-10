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
    except AttributeError:
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
    """Resolve one canonical bundled-relative or uploaded absolute font path."""
    requested = font_path or DEFAULT_FONT_RELATIVE_PATH
    resolved = requested if os.path.isabs(requested) else resource_path(requested)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"字体文件不存在: {requested}")
    return resolved
