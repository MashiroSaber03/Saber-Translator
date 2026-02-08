"""
Manga Insight 存储模块

提供统一的存储层抽象和路径管理。
"""

from .path_manager import (
    InsightPathManager,
    get_insight_storage_path,
    get_old_insight_path,
    BOOKSHELF_BASE,
    OLD_INSIGHT_BASE
)

__all__ = [
    "InsightPathManager",
    "get_insight_storage_path",
    "get_old_insight_path",
    "BOOKSHELF_BASE",
    "OLD_INSIGHT_BASE"
]
