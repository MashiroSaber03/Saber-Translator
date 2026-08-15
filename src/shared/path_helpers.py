"""
路径处理辅助模块，提供文件路径处理相关的通用函数
"""

import logging
import os
from pathlib import Path
import sys

from src.shared.constants import DEFAULT_FONT_RELATIVE_PATH

logger = logging.getLogger("PathHelpers")


def get_project_root() -> str:
    """
    获取源码项目根目录。

    Returns:
        源码项目根目录绝对路径
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_resource_root() -> str:
    """
    获取只读资源根目录。

    开发环境下返回源码项目根目录；PyInstaller 打包环境下返回 bundle 内部资源目录。
    """
    base_path = getattr(sys, "_MEIPASS", get_project_root())
    logger.debug("资源基础路径: %s", base_path)
    return base_path


def resource_path(relative_path: str | os.PathLike[str]) -> str:
    """
    获取资源的绝对路径，适用于开发环境和PyInstaller打包环境
    
    Args:
        relative_path: 相对路径
    
    Returns:
        资源的绝对路径
    """
    if not isinstance(relative_path, (str, os.PathLike)):
        raise TypeError("relative_path 必须是路径")
    relative = os.fspath(relative_path)
    if not relative or os.path.isabs(relative):
        raise ValueError("relative_path 必须是非空相对路径")
    root = Path(get_resource_root()).resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("relative_path 不能超出资源根目录") from exc
    logger.debug("资源路径解析: '%s' -> '%s'", relative, resolved)
    return str(resolved)


def get_font_path(font_path: str | os.PathLike[str] | None) -> str:
    """Resolve one canonical bundled-relative or uploaded absolute font path."""
    if font_path is not None and not isinstance(font_path, (str, os.PathLike)):
        raise TypeError("font_path 必须是路径或 null")
    requested = font_path or DEFAULT_FONT_RELATIVE_PATH
    resolved = requested if os.path.isabs(requested) else resource_path(requested)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"字体文件不存在: {requested}")
    return resolved
