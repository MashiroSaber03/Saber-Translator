"""
Manga Insight 路径管理器

集中管理所有 Insight 相关的路径解析，避免路径逻辑分散。
"""

import os
from typing import Optional

from src.shared.path_helpers import resource_path

# 基础目录常量
BOOKSHELF_BASE = "data/bookshelf"
OLD_INSIGHT_BASE = "data/manga_insight"  # 旧路径，用于迁移


class InsightPathManager:
    """
    Insight 存储路径管理器

    统一管理所有 Insight 相关的路径解析，支持新旧路径兼容。
    """

    def __init__(self, book_id: str):
        """
        初始化路径管理器

        Args:
            book_id: 书籍 ID
        """
        self.book_id = book_id
        self._insight_dir: Optional[str] = None

    @property
    def insight_dir(self) -> str:
        """获取 Insight 存储目录（带缓存）"""
        if self._insight_dir is None:
            self._insight_dir = resource_path(
                os.path.join(BOOKSHELF_BASE, self.book_id, "insight")
            )
        return self._insight_dir

    @property
    def old_insight_dir(self) -> str:
        """获取旧版 Insight 目录（用于迁移检查）"""
        return resource_path(os.path.join(OLD_INSIGHT_BASE, self.book_id))

    def needs_migration(self) -> bool:
        """检查是否需要从旧路径迁移"""
        return os.path.exists(self.old_insight_dir) and not os.path.exists(self.insight_dir)

    # ==================== 文件路径方法 ====================

    def get_metadata_path(self) -> str:
        """元数据文件路径"""
        return os.path.join(self.insight_dir, "metadata.json")

    def get_status_path(self) -> str:
        """分析状态文件路径"""
        return os.path.join(self.insight_dir, "analysis_status.json")

    def get_overview_path(self, template_key: Optional[str] = None) -> str:
        """概览文件路径"""
        if template_key:
            return os.path.join(self.insight_dir, f"overview_{template_key}.json")
        return os.path.join(self.insight_dir, "overview.json")

    def get_timeline_path(self) -> str:
        """时间线文件路径"""
        return os.path.join(self.insight_dir, "timeline.json")

    def get_compressed_context_path(self) -> str:
        """压缩上下文文件路径"""
        return os.path.join(self.insight_dir, "compressed_context.json")

    # ==================== 目录路径方法 ====================

    def get_pages_dir(self) -> str:
        """页面分析目录"""
        return os.path.join(self.insight_dir, "pages")

    def get_batches_dir(self) -> str:
        """批量分析目录"""
        return os.path.join(self.insight_dir, "batches")

    def get_segments_dir(self) -> str:
        """小总结目录"""
        return os.path.join(self.insight_dir, "segments")

    def get_chapters_dir(self) -> str:
        """章节分析目录"""
        return os.path.join(self.insight_dir, "chapters")

    def get_embeddings_dir(self) -> str:
        """向量嵌入目录（ChromaDB）"""
        return os.path.join(self.insight_dir, "embeddings")

    def get_continuation_dir(self) -> str:
        """续写数据目录"""
        return os.path.join(self.insight_dir, "continuation")

    # ==================== 具体文件路径方法 ====================

    def get_page_path(self, page_num: int) -> str:
        """获取单页分析文件路径"""
        return os.path.join(self.get_pages_dir(), f"page_{page_num:03d}.json")

    def get_batch_path(self, start_page: int, end_page: int) -> str:
        """获取批量分析文件路径"""
        return os.path.join(
            self.get_batches_dir(),
            f"batch_{start_page:03d}_{end_page:03d}.json"
        )

    def get_segment_path(self, segment_id: str) -> str:
        """获取小总结文件路径"""
        return os.path.join(self.get_segments_dir(), f"{segment_id}.json")

    def get_chapter_path(self, chapter_id: str) -> str:
        """获取章节分析文件路径"""
        return os.path.join(self.get_chapters_dir(), f"{chapter_id}.json")

    # ==================== 目录创建方法 ====================

    def ensure_dirs(self) -> None:
        """确保所有必要目录存在"""
        dirs = [
            self.insight_dir,
            self.get_pages_dir(),
            self.get_batches_dir(),
            self.get_segments_dir(),
            self.get_chapters_dir(),
            self.get_continuation_dir(),
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)


# ==================== 便捷函数 ====================

def get_insight_storage_path(book_id: str) -> str:
    """
    获取 Insight 存储路径（向后兼容函数）

    Args:
        book_id: 书籍 ID

    Returns:
        str: Insight 存储目录路径
    """
    return InsightPathManager(book_id).insight_dir


def get_old_insight_path(book_id: str) -> str:
    """
    获取旧版 Insight 路径（用于迁移）

    Args:
        book_id: 书籍 ID

    Returns:
        str: 旧版 Insight 目录路径
    """
    return InsightPathManager(book_id).old_insight_dir
