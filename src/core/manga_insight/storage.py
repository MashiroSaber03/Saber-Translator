"""
Manga Insight 存储模块

管理分析结果的存储和读取。

重构说明：
- 使用 asyncio.to_thread() 包装同步 I/O，避免阻塞事件循环
- 使用原子写入模式（临时文件 + rename），防止断电损坏数据
- 添加文件锁防止并发写入冲突
"""

import os
import json
import logging
import asyncio
import tempfile
import threading
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from src.shared.path_helpers import resource_path

logger = logging.getLogger("MangaInsight.Storage")

# 文件锁字典，防止并发写入同一文件
_file_locks: Dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_file_lock(filepath: str) -> threading.Lock:
    """获取文件锁（线程安全）"""
    with _locks_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]

# 新路径：统一存储在书架目录下
BOOKSHELF_DIR = "data/bookshelf"
# 旧路径：用于兼容检查
OLD_STORAGE_BASE_DIR = "data/manga_insight"


def get_insight_storage_path(book_id: str) -> str:
    """
    获取 Insight 存储路径（新格式：在书架目录下）

    新路径: data/bookshelf/{book_id}/insight/
    旧路径: data/manga_insight/{book_id}/ (已弃用，自动迁移)
    """
    return resource_path(os.path.join(BOOKSHELF_DIR, book_id, "insight"))


class AnalysisStorage:
    """分析结果存储管理器"""

    def __init__(self, book_id: str):
        self.book_id = book_id
        # 使用新路径
        self.base_path = get_insight_storage_path(book_id)
        # 检查旧路径是否存在数据（兼容迁移）
        self._check_and_migrate_old_data()
        self._ensure_directories()

    def _check_and_migrate_old_data(self):
        """检查并迁移旧路径的数据"""
        old_path = resource_path(os.path.join(OLD_STORAGE_BASE_DIR, self.book_id))

        # 如果旧路径存在且新路径不存在或为空，则迁移
        if os.path.exists(old_path) and os.path.isdir(old_path):
            if not os.path.exists(self.base_path) or not os.listdir(self.base_path):
                try:
                    # 确保父目录存在
                    os.makedirs(os.path.dirname(self.base_path), exist_ok=True)
                    # 移动数据
                    import shutil
                    shutil.move(old_path, self.base_path)
                    logger.info(f"自动迁移 Insight 数据: {old_path} -> {self.base_path}")
                except Exception as e:
                    logger.warning(f"迁移 Insight 数据失败（将使用旧路径）: {e}")
                    # 迁移失败时回退到旧路径
                    self.base_path = old_path
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        dirs = [
            self.base_path,
            os.path.join(self.base_path, "pages"),
            os.path.join(self.base_path, "chapters"),
            os.path.join(self.base_path, "batches"),
            os.path.join(self.base_path, "segments"),
            os.path.join(self.base_path, "embeddings")
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _load_json_sync(self, filename: str, default: Any = None) -> Any:
        """同步加载 JSON 文件（内部方法）"""
        filepath = os.path.join(self.base_path, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return default if default is not None else {}
        except Exception as e:
            logger.error(f"加载 JSON 失败: {filepath} - {e}")
            return default if default is not None else {}

    def _save_json_sync(self, filename: str, data: Any) -> bool:
        """
        同步保存 JSON 文件（内部方法）

        使用原子写入模式：
        1. 写入临时文件
        2. 成功后 rename 替换原文件
        3. 使用文件锁防止并发写入
        """
        filepath = os.path.join(self.base_path, filename)
        lock = _get_file_lock(filepath)

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with lock:
                # 在同一目录创建临时文件
                dir_path = os.path.dirname(filepath)
                fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=dir_path)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                    # 原子替换：Windows 需要先删除目标文件
                    if os.path.exists(filepath):
                        os.replace(tmp_path, filepath)
                    else:
                        os.rename(tmp_path, filepath)
                    return True
                except Exception:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
        except Exception as e:
            logger.error(f"保存 JSON 失败: {filepath} - {e}")
            return False

    async def _load_json(self, filename: str, default: Any = None) -> Any:
        """异步加载 JSON 文件"""
        return await asyncio.to_thread(self._load_json_sync, filename, default)

    async def _save_json(self, filename: str, data: Any) -> bool:
        """异步保存 JSON 文件"""
        return await asyncio.to_thread(self._save_json_sync, filename, data)

    async def _delete_file_if_exists(self, filepath: str) -> bool:
        """异步删除文件，文件不存在时返回 False。"""
        def _delete() -> bool:
            if not os.path.exists(filepath):
                return False
            os.remove(filepath)
            return True

        try:
            return await asyncio.to_thread(_delete)
        except Exception as e:
            logger.error(f"删除文件失败: {filepath} - {e}")
            return False
    
    async def load_metadata(self) -> Dict:
        return await self._load_json("metadata.json")

    async def save_metadata(self, metadata: Dict) -> bool:
        metadata["updated_at"] = datetime.now().isoformat()
        return await self._save_json("metadata.json", metadata)

    async def load_analysis_status(self) -> Dict:
        return await self._load_json("analysis_status.json")

    async def save_analysis_status(self, status: Dict) -> bool:
        status["updated_at"] = datetime.now().isoformat()
        return await self._save_json("analysis_status.json", status)

    async def load_content_snapshot(self) -> Optional[Dict]:
        return await self._load_json("content_snapshot.json", None)

    async def save_content_snapshot(self, snapshot: Dict) -> bool:
        snapshot["created_at"] = datetime.now().isoformat()
        return await self._save_json("content_snapshot.json", snapshot)

    async def load_page_analysis(self, page_num: int) -> Optional[Dict]:
        filename = f"pages/page_{page_num:03d}.json"
        return await self._load_json(filename, None)

    async def save_page_analysis(self, page_num: int, analysis: Dict) -> bool:
        filename = f"pages/page_{page_num:03d}.json"
        analysis["saved_at"] = datetime.now().isoformat()
        return await self._save_json(filename, analysis)

    async def delete_page_analysis(self, page_num: int) -> bool:
        """删除单页分析结果。"""
        filepath = os.path.join(self.base_path, f"pages/page_{page_num:03d}.json")
        return await self._delete_file_if_exists(filepath)

    async def load_chapter_analysis(self, chapter_id: str) -> Optional[Dict]:
        filename = f"chapters/{chapter_id}.json"
        return await self._load_json(filename, None)

    async def save_chapter_analysis(self, chapter_id: str, analysis: Dict) -> bool:
        filename = f"chapters/{chapter_id}.json"
        analysis["saved_at"] = datetime.now().isoformat()
        return await self._save_json(filename, analysis)

    async def delete_chapter_analysis(self, chapter_id: str) -> bool:
        """删除章节分析结果。"""
        filepath = os.path.join(self.base_path, f"chapters/{chapter_id}.json")
        return await self._delete_file_if_exists(filepath)

    async def load_timeline(self) -> Optional[Dict]:
        """加载时间线缓存"""
        return await self._load_json("timeline.json", None)

    async def save_timeline(self, timeline_data: Dict) -> bool:
        """保存时间线缓存"""
        timeline_data["saved_at"] = datetime.now().isoformat()
        return await self._save_json("timeline.json", timeline_data)

    async def load_notes(self) -> Optional[List]:
        """加载笔记列表"""
        return await self._load_json("notes.json", [])

    async def save_notes(self, notes: List) -> bool:
        """保存笔记列表"""
        return await self._save_json("notes.json", notes)
    
    async def has_timeline_cache(self) -> bool:
        """检查是否存在时间线缓存"""
        filepath = os.path.join(self.base_path, "timeline.json")
        return os.path.exists(filepath)
    
    async def delete_timeline_cache(self) -> bool:
        """删除时间线缓存"""
        filepath = os.path.join(self.base_path, "timeline.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"删除时间线缓存失败: {e}")
            return False
    
    async def load_overview(self) -> Dict:
        return await self._load_json("overview.json")

    async def save_overview(self, overview: Dict) -> bool:
        overview["updated_at"] = datetime.now().isoformat()
        return await self._save_json("overview.json", overview)
    
    # ============================================================
    # 压缩摘要存储方法（供问答全局模式使用）
    # ============================================================
    
    async def load_compressed_context(self) -> Optional[Dict]:
        """
        加载压缩后的全文摘要

        Returns:
            Dict: {
                "context": str,      # 压缩后的全文摘要
                "source": str,       # 数据来源
                "group_count": int,  # 分组数量
                "char_count": int,   # 字符数
                "generated_at": str  # 生成时间
            }
        """
        return await self._load_json("compressed_context.json", None)

    async def save_compressed_context(self, data: Dict) -> bool:
        """保存压缩后的全文摘要"""
        data["saved_at"] = datetime.now().isoformat()
        return await self._save_json("compressed_context.json", data)
    
    async def has_compressed_context(self) -> bool:
        """检查是否存在压缩摘要"""
        filepath = os.path.join(self.base_path, "compressed_context.json")
        return os.path.exists(filepath)
    
    async def delete_compressed_context(self) -> bool:
        """删除压缩摘要"""
        filepath = os.path.join(self.base_path, "compressed_context.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"删除压缩摘要失败: {e}")
            return False
    
    # ============================================================
    # 模板概要存储方法（多种概要类型）
    # ============================================================
    
    async def load_template_overview(self, template_key: str) -> Optional[Dict]:
        """
        加载指定模板的概要

        Args:
            template_key: 模板键名（如 story_summary, recap, no_spoiler 等）

        Returns:
            Dict: 模板概要数据
        """
        filename = f"overview_{template_key}.json"
        return await self._load_json(filename, None)

    async def save_template_overview(self, template_key: str, data: Dict) -> bool:
        """
        保存指定模板的概要

        Args:
            template_key: 模板键名
            data: 概要数据
        """
        filename = f"overview_{template_key}.json"
        data["saved_at"] = datetime.now().isoformat()
        data["template_key"] = template_key
        return await self._save_json(filename, data)
    
    async def delete_template_overview(self, template_key: str) -> bool:
        """删除指定模板的概要缓存"""
        filepath = os.path.join(self.base_path, f"overview_{template_key}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"删除模板概要失败: {e}")
            return False
    
    async def list_template_overviews(self) -> List[Dict]:
        """列出所有已生成的模板概要"""
        overviews = []
        for filename in os.listdir(self.base_path):
            if filename.startswith("overview_") and filename.endswith(".json"):
                template_key = filename[9:-5]  # 去掉 "overview_" 和 ".json"
                data = await self.load_template_overview(template_key)
                if data:
                    overviews.append({
                        "template_key": template_key,
                        "template_name": data.get("template_name", template_key),
                        "template_icon": data.get("template_icon", "📄"),
                        "generated_at": data.get("generated_at"),
                        "has_content": bool(data.get("content"))
                    })
        return overviews
    
    async def clear_all_template_overviews(self) -> bool:
        """清除所有模板概要缓存"""
        try:
            for filename in os.listdir(self.base_path):
                if filename.startswith("overview_") and filename.endswith(".json"):
                    filepath = os.path.join(self.base_path, filename)
                    os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"清除模板概要失败: {e}")
            return False
    
    # ============================================================
    # 批量分析存储方法
    # ============================================================
    
    async def load_batch_analysis(self, start_page: int, end_page: int) -> Optional[Dict]:
        """加载批量分析结果"""
        filename = f"batches/batch_{start_page:03d}_{end_page:03d}.json"
        return await self._load_json(filename, None)

    async def save_batch_analysis(self, start_page: int, end_page: int, analysis: Dict) -> bool:
        """保存批量分析结果"""
        filename = f"batches/batch_{start_page:03d}_{end_page:03d}.json"
        analysis["saved_at"] = datetime.now().isoformat()
        analysis["page_range"] = {"start": start_page, "end": end_page}
        return await self._save_json(filename, analysis)
    
    async def list_batches(self) -> List[Dict]:
        """列出所有批量分析结果"""
        batches_dir = os.path.join(self.base_path, "batches")
        if not os.path.exists(batches_dir):
            return []
        
        batches = []
        for filename in os.listdir(batches_dir):
            if filename.startswith("batch_") and filename.endswith(".json"):
                try:
                    parts = filename[6:-5].split("_")
                    start_page = int(parts[0])
                    end_page = int(parts[1])
                    batches.append({
                        "start_page": start_page,
                        "end_page": end_page,
                        "filename": filename
                    })
                except (ValueError, IndexError):
                    pass
        return sorted(batches, key=lambda x: x["start_page"])
    
    async def find_batch_for_page(self, page_num: int) -> Optional[Dict]:
        """根据页码找到对应的批次（用于父子块检索）"""
        batches = await self.list_batches()
        for batch in batches:
            if batch["start_page"] <= page_num <= batch["end_page"]:
                return await self.load_batch_analysis(batch["start_page"], batch["end_page"])
        return None
    
    async def delete_batch_analysis(self, start_page: int, end_page: int) -> bool:
        """删除批量分析结果"""
        filepath = os.path.join(self.base_path, f"batches/batch_{start_page:03d}_{end_page:03d}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"删除批量分析失败: {e}")
            return False
    
    # ============================================================
    # 小总结 (Segment) 存储方法
    # ============================================================
    
    async def load_segment_summary(self, segment_id: str) -> Optional[Dict]:
        """加载小总结"""
        filename = f"segments/{segment_id}.json"
        return await self._load_json(filename, None)

    async def save_segment_summary(self, segment_id: str, summary: Dict) -> bool:
        """保存小总结"""
        filename = f"segments/{segment_id}.json"
        summary["saved_at"] = datetime.now().isoformat()
        summary["segment_id"] = segment_id
        return await self._save_json(filename, summary)
    
    async def list_segments(self) -> List[Dict]:
        """列出所有小总结"""
        segments_dir = os.path.join(self.base_path, "segments")
        if not os.path.exists(segments_dir):
            return []
        
        segments = []
        for filename in os.listdir(segments_dir):
            if filename.endswith(".json"):
                segment_id = filename[:-5]
                data = await self.load_segment_summary(segment_id)
                if data:
                    segments.append({
                        "segment_id": segment_id,
                        "page_range": data.get("page_range", {}),
                        "summary": data.get("summary", "")
                    })
        return sorted(segments, key=lambda x: x.get("page_range", {}).get("start", 0))
    
    async def delete_segment_summary(self, segment_id: str) -> bool:
        """删除小总结"""
        filepath = os.path.join(self.base_path, f"segments/{segment_id}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"删除小总结失败: {e}")
            return False
    
    async def get_segments_for_chapter(self, chapter_id: str, start_page: int, end_page: int) -> List[Dict]:
        """获取某章节范围内的所有小总结"""
        all_segments = await self.list_segments()
        chapter_segments = []
        for seg in all_segments:
            seg_range = seg.get("page_range", {})
            seg_start = seg_range.get("start", 0)
            seg_end = seg_range.get("end", 0)
            # 检查小总结是否在章节范围内
            if seg_start >= start_page and seg_end <= end_page:
                full_data = await self.load_segment_summary(seg["segment_id"])
                if full_data:
                    chapter_segments.append(full_data)
        return chapter_segments
    
    async def clear_batches_and_segments(self) -> bool:
        """清除所有批量分析和小总结"""
        import shutil
        try:
            batches_dir = os.path.join(self.base_path, "batches")
            segments_dir = os.path.join(self.base_path, "segments")
            if os.path.exists(batches_dir):
                shutil.rmtree(batches_dir)
            if os.path.exists(segments_dir):
                shutil.rmtree(segments_dir)
            self._ensure_directories()
            return True
        except Exception as e:
            logger.error(f"清除批量分析和小总结失败: {e}")
            return False

    @staticmethod
    def _range_contains_any_page(start_page: int, end_page: int, target_pages: Set[int]) -> bool:
        """判断页码区间是否与目标页集合有交集。"""
        if start_page <= 0 or end_page <= 0 or start_page > end_page:
            return False
        for page_num in target_pages:
            if start_page <= page_num <= end_page:
                return True
        return False

    async def _clear_global_derived_caches(self) -> int:
        """清理受页面变更影响的全局衍生缓存。"""
        deleted_count = 0

        cache_files = [
            os.path.join(self.base_path, "overview.json"),
            os.path.join(self.base_path, "timeline.json"),
            os.path.join(self.base_path, "compressed_context.json")
        ]
        for cache_file in cache_files:
            if await self._delete_file_if_exists(cache_file):
                deleted_count += 1

        # 多模板概览缓存
        for filename in os.listdir(self.base_path):
            if filename.startswith("overview_") and filename.endswith(".json"):
                if await self._delete_file_if_exists(os.path.join(self.base_path, filename)):
                    deleted_count += 1

        return deleted_count

    async def clear_cache_for_pages(self, page_nums: List[int], chapter_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """
        按页清理相关缓存（page/batch/segment/chapter + 全局衍生缓存）。

        Args:
            page_nums: 受影响页码
            chapter_ids: 可选，明确受影响章节 ID
        """
        result = {
            "pages_deleted": 0,
            "batches_deleted": 0,
            "segments_deleted": 0,
            "chapters_deleted": 0,
            "global_caches_deleted": 0
        }

        normalized_pages = sorted({
            p for p in page_nums
            if isinstance(p, int) and not isinstance(p, bool) and p > 0
        })
        target_pages = set(normalized_pages)

        if chapter_ids is None:
            chapter_ids = []
        chapter_id_set = {cid for cid in chapter_ids if isinstance(cid, str) and cid}

        # 1) 删除页面结果
        for page_num in normalized_pages:
            if await self.delete_page_analysis(page_num):
                result["pages_deleted"] += 1

        # 2) 删除命中的批量结果
        batches = await self.list_batches()
        for batch in batches:
            start_page = batch.get("start_page", 0)
            end_page = batch.get("end_page", 0)
            if self._range_contains_any_page(start_page, end_page, target_pages):
                deleted = await self.delete_batch_analysis(start_page, end_page)
                if deleted:
                    result["batches_deleted"] += 1

        # 3) 删除命中的小总结
        segments = await self.list_segments()
        for segment in segments:
            page_range = segment.get("page_range", {})
            start_page = page_range.get("start", 0)
            end_page = page_range.get("end", 0)
            if self._range_contains_any_page(start_page, end_page, target_pages):
                segment_id = segment.get("segment_id")
                if segment_id and await self.delete_segment_summary(segment_id):
                    result["segments_deleted"] += 1

        # 4) 删除命中的章节总结（按章节ID或页码范围）
        chapters = await self.list_chapters()
        deleted_chapter_ids = set()
        for chapter in chapters:
            chapter_id = chapter.get("id")
            if not chapter_id:
                continue

            should_delete = chapter_id in chapter_id_set
            if not should_delete:
                start_page = chapter.get("start_page", 0)
                end_page = chapter.get("end_page", 0)
                should_delete = self._range_contains_any_page(start_page, end_page, target_pages)

            if should_delete and chapter_id not in deleted_chapter_ids:
                if await self.delete_chapter_analysis(chapter_id):
                    result["chapters_deleted"] += 1
                deleted_chapter_ids.add(chapter_id)

        # 5) 删除明确指定但未在 list_chapters 中出现的章节文件
        for chapter_id in chapter_id_set:
            if chapter_id in deleted_chapter_ids:
                continue
            if await self.delete_chapter_analysis(chapter_id):
                result["chapters_deleted"] += 1

        # 6) 清理全局衍生缓存
        result["global_caches_deleted"] = await self._clear_global_derived_caches()

        logger.info(
            "按页清理缓存完成: pages=%s, batches=%s, segments=%s, chapters=%s, globals=%s",
            result["pages_deleted"],
            result["batches_deleted"],
            result["segments_deleted"],
            result["chapters_deleted"],
            result["global_caches_deleted"]
        )
        return result

    async def clear_cache_for_chapters(self, chapter_ids: List[str], page_nums: Optional[List[int]] = None) -> Dict[str, int]:
        """
        按章节清理相关缓存。

        Args:
            chapter_ids: 章节 ID 列表
            page_nums: 可选，章节关联页码（用于精确删除 batch/segment）
        """
        normalized_chapters = [cid for cid in chapter_ids if isinstance(cid, str) and cid]
        derived_pages: List[int] = list(page_nums or [])

        if not derived_pages and normalized_chapters:
            existing_chapters = await self.list_chapters()
            for chapter in existing_chapters:
                chapter_id = chapter.get("id")
                if chapter_id in normalized_chapters:
                    start_page = chapter.get("start_page", 0)
                    end_page = chapter.get("end_page", 0)
                    if start_page > 0 and end_page >= start_page:
                        derived_pages.extend(list(range(start_page, end_page + 1)))

        return await self.clear_cache_for_pages(derived_pages, chapter_ids=normalized_chapters)
    
    async def list_pages(self) -> List[int]:
        """列出已分析的页面"""
        pages_dir = os.path.join(self.base_path, "pages")
        if not os.path.exists(pages_dir):
            return []
        
        pages = []
        for filename in os.listdir(pages_dir):
            if filename.startswith("page_") and filename.endswith(".json"):
                try:
                    page_num = int(filename[5:-5])
                    pages.append(page_num)
                except ValueError:
                    pass
        return sorted(pages)
    
    async def list_chapters(self) -> List[Dict]:
        """列出已分析的章节"""
        chapters_dir = os.path.join(self.base_path, "chapters")
        if not os.path.exists(chapters_dir):
            return []

        chapters = []
        for filename in os.listdir(chapters_dir):
            if filename.endswith(".json"):
                chapter_id = filename[:-5]
                analysis = await self.load_chapter_analysis(chapter_id)
                if analysis:
                    # 获取页面范围
                    page_range = analysis.get("page_range", {})
                    start_page = page_range.get("start", 0)
                    end_page = page_range.get("end", 0)
                    chapters.append({
                        "id": chapter_id,
                        "title": analysis.get("title", chapter_id),
                        "start_page": start_page,
                        "end_page": end_page
                    })
        return chapters
    
    async def clear_all(self) -> bool:
        """清除所有分析结果"""
        import shutil
        try:
            if os.path.exists(self.base_path):
                shutil.rmtree(self.base_path)
            self._ensure_directories()
            return True
        except Exception as e:
            logger.error(f"清除分析结果失败: {e}")
            return False
    
    async def export_all(self) -> Dict:
        """导出所有分析数据"""
        # 加载批量分析
        batches = []
        for batch_info in await self.list_batches():
            batch_data = await self.load_batch_analysis(
                batch_info["start_page"], 
                batch_info["end_page"]
            )
            if batch_data:
                batches.append(batch_data)
        
        # 加载小总结
        segments = []
        for seg_info in await self.list_segments():
            seg_data = await self.load_segment_summary(seg_info["segment_id"])
            if seg_data:
                segments.append(seg_data)
        
        return {
            "book_id": self.book_id,
            "metadata": await self.load_metadata(),
            "overview": await self.load_overview(),
            "timeline": await self.load_timeline(),
            "pages": [await self.load_page_analysis(p) for p in await self.list_pages()],
            "batches": batches,
            "segments": segments,
            "exported_at": datetime.now().isoformat()
        }
    
    # ============================================================
    # 续写功能数据存储
    # ============================================================
    
    async def load_continuation_script(self) -> Optional[Dict]:
        """加载续写脚本"""
        return await self._load_json("continuation/script.json", None)

    async def save_continuation_script(self, script: Dict) -> bool:
        """保存续写脚本"""
        script["saved_at"] = datetime.now().isoformat()
        return await self._save_json("continuation/script.json", script)

    async def load_continuation_pages(self) -> Optional[List]:
        """加载续写页面详情列表"""
        return await self._load_json("continuation/pages.json", None)

    async def save_continuation_pages(self, pages: List) -> bool:
        """保存续写页面详情列表"""
        data = {
            "pages": pages,
            "saved_at": datetime.now().isoformat()
        }
        return await self._save_json("continuation/pages.json", data)

    async def load_continuation_config(self) -> Optional[Dict]:
        """加载续写配置"""
        return await self._load_json("continuation/config.json", None)

    async def save_continuation_config(self, config: Dict) -> bool:
        """保存续写配置"""
        config["saved_at"] = datetime.now().isoformat()
        return await self._save_json("continuation/config.json", config)
    
    async def load_continuation_all(self) -> Dict:
        """加载所有续写数据"""
        script = await self.load_continuation_script()
        pages_data = await self.load_continuation_pages()
        config = await self.load_continuation_config()
        
        return {
            "script": script,
            "pages": pages_data.get("pages", []) if pages_data else [],
            "config": config,
            "has_data": script is not None or (pages_data is not None and len(pages_data.get("pages", [])) > 0)
        }
    
    async def clear_continuation_data(self) -> bool:
        """清除所有续写数据"""
        import shutil
        continuation_path = os.path.join(self.base_path, "continuation")
        try:
            if os.path.exists(continuation_path):
                # 只删除脚本和页面数据，保留生成的图片
                for filename in ["script.json", "pages.json", "config.json"]:
                    filepath = os.path.join(continuation_path, filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"清除续写数据失败: {e}")
            return False
