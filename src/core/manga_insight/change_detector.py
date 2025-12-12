"""
Manga Insight 内容变更检测

检测书籍内容变化，支持增量分析。
"""

import hashlib
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("MangaInsight.ChangeDetector")


class ChangeType(Enum):
    """变更类型"""
    NO_CHANGE = "no_change"
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class ContentChange:
    """内容变更"""
    change_type: ChangeType
    chapter_id: Optional[str] = None
    page_numbers: Optional[List[int]] = None


class ContentChangeDetector:
    """检测书籍内容变更"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
    
    async def detect_changes(
        self,
        current_content: Dict,
        previous_snapshot: Optional[Dict] = None
    ) -> List[ContentChange]:
        """
        对比当前内容与上次分析时的快照，识别变更
        
        Args:
            current_content: 当前内容快照
                {
                    "chapters": [
                        {
                            "chapter_id": "ch_001",
                            "title": "第一章",
                            "pages": [
                                {"page_num": 1, "hash": "abc123"},
                                ...
                            ]
                        }
                    ]
                }
            previous_snapshot: 上次分析时的快照
        
        Returns:
            List[ContentChange]: 变更列表
        """
        if previous_snapshot is None:
            # 首次分析，所有内容都是新增
            return [
                ContentChange(
                    change_type=ChangeType.ADDED,
                    chapter_id=ch.get("chapter_id"),
                    page_numbers=[p.get("page_num") for p in ch.get("pages", [])]
                )
                for ch in current_content.get("chapters", [])
            ]
        
        changes = []
        
        prev_chapters = {
            ch.get("chapter_id"): ch 
            for ch in previous_snapshot.get("chapters", [])
        }
        curr_chapters = {
            ch.get("chapter_id"): ch 
            for ch in current_content.get("chapters", [])
        }
        
        # 检测新增和修改
        for ch_id, curr_ch in curr_chapters.items():
            if ch_id not in prev_chapters:
                # 新增章节
                changes.append(ContentChange(
                    change_type=ChangeType.ADDED,
                    chapter_id=ch_id,
                    page_numbers=[p.get("page_num") for p in curr_ch.get("pages", [])]
                ))
            else:
                # 检测页面级变更
                modified_pages = self._detect_page_changes(
                    curr_ch.get("pages", []),
                    prev_chapters[ch_id].get("pages", [])
                )
                if modified_pages:
                    changes.append(ContentChange(
                        change_type=ChangeType.MODIFIED,
                        chapter_id=ch_id,
                        page_numbers=modified_pages
                    ))
        
        # 检测删除
        for ch_id in prev_chapters:
            if ch_id not in curr_chapters:
                changes.append(ContentChange(
                    change_type=ChangeType.DELETED,
                    chapter_id=ch_id
                ))
        
        return changes
    
    def _detect_page_changes(
        self,
        current: List[Dict],
        previous: List[Dict]
    ) -> List[int]:
        """
        检测页面级变更（通过哈希对比）
        
        Args:
            current: 当前页面列表
            previous: 之前页面列表
        
        Returns:
            List[int]: 变更的页码列表
        """
        prev_hashes = {
            p.get("page_num"): p.get("hash") 
            for p in previous
        }
        
        changed = []
        for page in current:
            page_num = page.get("page_num")
            page_hash = page.get("hash")
            
            # 新增或修改
            if page_num not in prev_hashes or page_hash != prev_hashes[page_num]:
                changed.append(page_num)
        
        return changed
    
    @staticmethod
    def compute_page_hash(image_data: bytes) -> str:
        """
        计算页面内容哈希
        
        Args:
            image_data: 图片二进制数据
        
        Returns:
            str: 16位哈希值
        """
        return hashlib.sha256(image_data).hexdigest()[:16]
    
    async def build_content_snapshot(self) -> Dict:
        """
        构建当前内容快照
        
        Returns:
            Dict: 内容快照
        """
        try:
            from src.core import bookshelf_manager
            import os
            
            book = bookshelf_manager.get_book(self.book_id)
            
            if not book:
                return {"chapters": []}
            
            chapters = []
            for ch in book.get("chapters", []):
                chapter_id = ch.get("id") or ch.get("chapter_id")
                chapter_data = {
                    "chapter_id": chapter_id,
                    "title": ch.get("title", ""),
                    "pages": []
                }
                
                # 获取章节详情和图片
                chapter_detail = bookshelf_manager.get_chapter(self.book_id, chapter_id)
                if chapter_detail:
                    images = chapter_detail.get("images", [])
                    for idx, img in enumerate(images):
                        image_path = img.get("original_path") or img.get("path")
                        if image_path and os.path.exists(image_path):
                            with open(image_path, "rb") as f:
                                page_hash = self.compute_page_hash(f.read())
                            
                            chapter_data["pages"].append({
                                "page_num": img.get("index", idx + 1),
                                "hash": page_hash
                            })
                
                chapters.append(chapter_data)
            
            return {"chapters": chapters}
            
        except Exception as e:
            logger.error(f"构建内容快照失败: {e}")
            return {"chapters": []}


def get_pages_to_analyze(changes: List[ContentChange]) -> List[int]:
    """
    从变更列表中提取需要分析的页面
    
    Args:
        changes: 变更列表
    
    Returns:
        List[int]: 需要分析的页码列表
    """
    pages = []
    for change in changes:
        if change.change_type in [ChangeType.ADDED, ChangeType.MODIFIED]:
            pages.extend(change.page_numbers or [])
    return sorted(list(set(pages)))
