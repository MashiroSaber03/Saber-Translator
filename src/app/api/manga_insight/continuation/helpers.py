"""
续写功能 - 通用辅助函数

提供路由模块共享的辅助函数。
"""

import os
import logging
from typing import List

from src.shared.path_helpers import resource_path

logger = logging.getLogger("MangaInsight.API.Continuation")


def is_subpath(path: str, parent: str) -> bool:
    """
    安全检查 path 是否在 parent 目录下

    使用 commonpath 避免前缀碰撞攻击。
    """
    try:
        real_parent = os.path.realpath(parent)
        real_child = os.path.realpath(path)
        return os.path.commonpath([real_child, real_parent]) == real_parent
    except (ValueError, OSError):
        return False


def check_path_safety(file_path: str, allowed_dirs: List[str] = None) -> bool:
    """
    检查文件路径是否在允许的目录内

    Args:
        file_path: 要检查的文件路径
        allowed_dirs: 允许的目录列表，默认为数据目录

    Returns:
        bool: 是否安全
    """
    if allowed_dirs is None:
        allowed_dirs = [
            resource_path("data/bookshelf"),
            resource_path("data/manga_insight"),
            resource_path("data/sessions"),
        ]

    real_path = os.path.realpath(os.path.normpath(file_path))

    return any(
        is_subpath(real_path, allowed_dir)
        for allowed_dir in allowed_dirs if os.path.exists(allowed_dir)
    )


def get_mimetype(file_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = os.path.splitext(file_path)[1].lower()
    mimetype_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
    }
    return mimetype_map.get(ext, 'application/octet-stream')
