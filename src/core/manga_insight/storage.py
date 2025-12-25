"""
Manga Insight 存储模块

管理分析结果的存储和读取。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.shared.path_helpers import resource_path

logger = logging.getLogger("MangaInsight.Storage")

STORAGE_BASE_DIR = "data/manga_insight"


class AnalysisStorage:
    """分析结果存储管理器"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.base_path = resource_path(os.path.join(STORAGE_BASE_DIR, book_id))
        self._ensure_directories()
    
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
    
    def _load_json(self, filename: str, default: Any = None) -> Any:
        """加载 JSON 文件"""
        filepath = os.path.join(self.base_path, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return default if default is not None else {}
        except Exception as e:
            logger.error(f"加载 JSON 失败: {filepath} - {e}")
            return default if default is not None else {}
    
    def _save_json(self, filename: str, data: Any) -> bool:
        """保存 JSON 文件"""
        filepath = os.path.join(self.base_path, filename)
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存 JSON 失败: {filepath} - {e}")
            return False
    
    async def load_metadata(self) -> Dict:
        return self._load_json("metadata.json")
    
    async def save_metadata(self, metadata: Dict) -> bool:
        metadata["updated_at"] = datetime.now().isoformat()
        return self._save_json("metadata.json", metadata)
    
    async def load_analysis_status(self) -> Dict:
        return self._load_json("analysis_status.json")
    
    async def save_analysis_status(self, status: Dict) -> bool:
        status["updated_at"] = datetime.now().isoformat()
        return self._save_json("analysis_status.json", status)
    
    async def load_content_snapshot(self) -> Optional[Dict]:
        return self._load_json("content_snapshot.json", None)
    
    async def save_content_snapshot(self, snapshot: Dict) -> bool:
        snapshot["created_at"] = datetime.now().isoformat()
        return self._save_json("content_snapshot.json", snapshot)
    
    async def load_page_analysis(self, page_num: int) -> Optional[Dict]:
        filename = f"pages/page_{page_num:03d}.json"
        return self._load_json(filename, None)
    
    async def save_page_analysis(self, page_num: int, analysis: Dict) -> bool:
        filename = f"pages/page_{page_num:03d}.json"
        analysis["saved_at"] = datetime.now().isoformat()
        return self._save_json(filename, analysis)
    
    async def load_chapter_analysis(self, chapter_id: str) -> Optional[Dict]:
        filename = f"chapters/{chapter_id}.json"
        return self._load_json(filename, None)
    
    async def save_chapter_analysis(self, chapter_id: str, analysis: Dict) -> bool:
        filename = f"chapters/{chapter_id}.json"
        analysis["saved_at"] = datetime.now().isoformat()
        return self._save_json(filename, analysis)
    
    async def load_timeline(self) -> Optional[Dict]:
        """加载时间线缓存"""
        return self._load_json("timeline.json", None)
    
    async def save_timeline(self, timeline_data: Dict) -> bool:
        """保存时间线缓存"""
        timeline_data["saved_at"] = datetime.now().isoformat()
        return self._save_json("timeline.json", timeline_data)
    
    async def load_notes(self) -> Optional[List]:
        """加载笔记列表"""
        return self._load_json("notes.json", [])
    
    async def save_notes(self, notes: List) -> bool:
        """保存笔记列表"""
        return self._save_json("notes.json", notes)
    
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
        return self._load_json("overview.json")
    
    async def save_overview(self, overview: Dict) -> bool:
        overview["updated_at"] = datetime.now().isoformat()
        return self._save_json("overview.json", overview)
    
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
        return self._load_json("compressed_context.json", None)
    
    async def save_compressed_context(self, data: Dict) -> bool:
        """保存压缩后的全文摘要"""
        data["saved_at"] = datetime.now().isoformat()
        return self._save_json("compressed_context.json", data)
    
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
        return self._load_json(filename, None)
    
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
        return self._save_json(filename, data)
    
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
        return self._load_json(filename, None)
    
    async def save_batch_analysis(self, start_page: int, end_page: int, analysis: Dict) -> bool:
        """保存批量分析结果"""
        filename = f"batches/batch_{start_page:03d}_{end_page:03d}.json"
        analysis["saved_at"] = datetime.now().isoformat()
        analysis["page_range"] = {"start": start_page, "end": end_page}
        return self._save_json(filename, analysis)
    
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
        return self._load_json(filename, None)
    
    async def save_segment_summary(self, segment_id: str, summary: Dict) -> bool:
        """保存小总结"""
        filename = f"segments/{segment_id}.json"
        summary["saved_at"] = datetime.now().isoformat()
        summary["segment_id"] = segment_id
        return self._save_json(filename, summary)
    
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
    
    async def list_pages(self) -> List[int]:
        """列出已分析的页面"""
        pages_dir = os.path.join(self.base_path, "pages")
        if not os.path.exists(pages_dir):
            return []
        
        pages = []
        for filename in os.listdir(pages_dir):
            if filename.startswith("page_") and filename.endswith(".json"):
                try:
                    page_num = int(filename[5:8])
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
                    chapters.append({
                        "id": chapter_id,
                        "title": analysis.get("title", chapter_id)
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
