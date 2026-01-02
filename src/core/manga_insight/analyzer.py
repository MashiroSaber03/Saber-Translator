"""
Manga Insight 主分析器

协调各模块完成漫画分析。
"""

import logging
import os
import json
import re
from typing import Dict, List
from datetime import datetime

from tqdm import tqdm

from .config_models import (
    MangaInsightConfig, 
    DEFAULT_SEGMENT_SUMMARY_PROMPT,
    DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT,
    DEFAULT_ANALYSIS_SYSTEM_PROMPT
)
from .storage import AnalysisStorage
from .vlm_client import VLMClient
from .embedding_client import EmbeddingClient
from .vector_store import MangaVectorStore

logger = logging.getLogger("MangaInsight.Analyzer")


class MangaAnalyzer:
    """
    漫画分析器
    
    使用漫画原图进行分析，所有输出为中文。
    """
    
    def __init__(self, book_id: str, config: MangaInsightConfig):
        self.book_id = book_id
        self.config = config
        self.storage = AnalysisStorage(book_id)
        self.vlm = VLMClient(config.vlm, config.prompts)
        self.embedding = EmbeddingClient(config.embedding) if config.embedding.api_key else None
        self.vector_store = MangaVectorStore(book_id)
        self._book_info_cache = None
    
    async def get_book_info(self) -> Dict:
        """获取书籍信息（从书架系统）"""
        if self._book_info_cache:
            return self._book_info_cache
        
        try:
            from src.core import bookshelf_manager
            from src.shared.path_helpers import resource_path
            
            book = bookshelf_manager.get_book(self.book_id)
            
            if book:
                # 计算总页数：遍历所有章节的 session 数据
                total_pages = 0
                chapters = book.get("chapters", [])
                all_images = []
                
                sessions_base = resource_path("data/sessions/bookshelf")
                
                for chapter in chapters:
                    chapter_id = chapter.get("id")
                    if chapter_id:
                        # 从 session_meta.json 获取图片信息
                        session_dir = os.path.join(sessions_base, self.book_id, chapter_id)
                        session_meta_path = os.path.join(session_dir, "session_meta.json")
                        
                        if os.path.exists(session_meta_path):
                            try:
                                with open(session_meta_path, "r", encoding="utf-8") as f:
                                    session_data = json.load(f)
                                
                                images_meta = session_data.get("images_meta", [])
                                image_count = len(images_meta)
                                
                                for i in range(image_count):
                                    image_path = os.path.join(session_dir, f"image_{i}_original.png")
                                    if os.path.exists(image_path):
                                        all_images.append({
                                            "chapter_id": chapter_id,
                                            "index": i,
                                            "path": image_path,
                                            "meta": images_meta[i] if i < len(images_meta) else {}
                                        })
                                        total_pages += 1
                            except Exception as e:
                                logger.error(f"读取 session 数据失败: {chapter_id} - {e}")
                
                self._book_info_cache = {
                    "book_id": self.book_id,
                    "title": book.get("title", ""),
                    "total_pages": total_pages,
                    "chapters": chapters,
                    "cover": book.get("cover", ""),
                    "all_images": all_images
                }
                
                logger.info(f"书籍 {self.book_id} 共有 {total_pages} 页")
            else:
                self._book_info_cache = {
                    "book_id": self.book_id,
                    "total_pages": 0,
                    "all_images": []
                }
            
            return self._book_info_cache
        except Exception as e:
            logger.error(f"获取书籍信息失败: {e}")
            return {"book_id": self.book_id, "total_pages": 0, "all_images": []}
    
    async def get_original_image(self, page_num: int) -> bytes:
        """
        获取原图（非翻译图）用于分析
        
        从书架系统获取原图路径
        """
        try:
            from src.core import bookshelf_manager
            
            # 获取书籍信息
            book = bookshelf_manager.get_book(self.book_id)
            if not book:
                raise ValueError(f"未找到书籍: {self.book_id}")
            
            # 查找包含该页面的章节
            chapters = book.get("chapters", [])
            for chapter in chapters:
                chapter_detail = bookshelf_manager.get_chapter(self.book_id, chapter.get("id"))
                if not chapter_detail:
                    continue
                
                images = chapter_detail.get("images", [])
                for img in images:
                    if img.get("index") == page_num or img.get("page_num") == page_num:
                        image_path = img.get("original_path") or img.get("path")
                        if image_path and os.path.exists(image_path):
                            with open(image_path, "rb") as f:
                                return f.read()
            
            raise ValueError(f"未找到页面图片: 第{page_num}页")
        except Exception as e:
            logger.error(f"获取原图失败: 第{page_num}页 - {e}")
            raise
    
    async def _get_image_from_info(self, page_num: int, image_info: Dict = None) -> bytes:
        """
        从图片信息获取图片数据
        
        Args:
            page_num: 页码
            image_info: 图片信息字典
        
        Returns:
            bytes: 图片数据
        """
        if image_info:
            image_path = image_info.get("path")
            if image_path and os.path.exists(image_path):
                logger.debug(f"读取图片: {image_path}")
                with open(image_path, "rb") as f:
                    return f.read()
        
        raise ValueError(f"无法获取第 {page_num} 页的图片")
    
    # ============================================================
    # 批量分析模式（四层级架构）
    # ============================================================
    
    async def analyze_batch(
        self,
        page_nums: List[int],
        images: List[bytes] = None,
        image_infos: List[Dict] = None,
        force: bool = False,
        previous_results: List[Dict] = None
    ) -> Dict:
        """
        批量分析多页（第一层级）
        
        Args:
            page_nums: 页码列表
            images: 图片数据列表
            image_infos: 图片信息列表
            force: 是否强制重新分析
            previous_results: 前 N 批的分析结果列表，用于上下文连贯
        
        Returns:
            Dict: 批量分析结果
        """
        if not page_nums:
            return {}
        
        start_page = min(page_nums)
        end_page = max(page_nums)
        
        # 检查缓存
        if not force:
            cached = await self.storage.load_batch_analysis(start_page, end_page)
            if cached and not cached.get("parse_error"):
                logger.debug(f"使用批量缓存: 第{start_page}-{end_page}页")
                return cached
        
        # 获取图片
        if images is None:
            images = []
            for i, page_num in enumerate(page_nums):
                img_info = image_infos[i] if image_infos and i < len(image_infos) else None
                img_data = await self._get_image_from_info(page_num, img_info)
                images.append(img_data)
        
        # 构建上下文（使用传入的前 N 批结果）
        context = await self._build_batch_context(start_page, previous_results)
        
        # 调用 VLM 批量分析
        context_count = len(previous_results) if previous_results else 0
        has_context = bool(context.get("previous_summary"))
        logger.info(f"批量分析: 第{start_page}-{end_page}页 ({len(images)}张图片) [上文{context_count}批]")
        logger.debug(f"批量分析上下文: {context}")
        result = await self.vlm.analyze_batch(images, start_page, context)
        logger.info(f"批量分析完成: 第{start_page}-{end_page}页, 结果包含 {len(result.get('pages', []))} 个页面")
        
        # 添加元数据
        result["analyzed_at"] = datetime.now().isoformat()
        result["analysis_mode"] = "batch"
        
        # 保存批量结果
        await self.storage.save_batch_analysis(start_page, end_page, result)
        
        # 同时保存单页结果（便于其他功能使用）
        if result.get("pages"):
            for page_data in result["pages"]:
                page_num = page_data.get("page_number")
                if page_num:
                    page_data["from_batch"] = True
                    page_data["batch_range"] = {"start": start_page, "end": end_page}
                    page_data["analyzed_at"] = result["analyzed_at"]
                    await self.storage.save_page_analysis(page_num, page_data)
        
        return result
    
    async def _build_batch_context(self, start_page: int, previous_results: List[Dict] = None) -> Dict:
        """构建批量分析上下文（支持多批上文）"""
        context = {}
        
        # 使用传入的前 N 批结果
        if previous_results and len(previous_results) > 0:
            # 格式化多批结果为上文
            context["previous_summary"] = self._format_previous_results(previous_results)
            context["context_batch_count"] = len(previous_results)
        elif start_page > 1:
            # 回退：从存储中查找前一个批次（单批）
            batches = await self.storage.list_batches()
            for batch in reversed(batches):
                if batch["end_page"] < start_page:
                    batch_data = await self.storage.load_batch_analysis(
                        batch["start_page"], batch["end_page"]
                    )
                    if batch_data:
                        context["previous_summary"] = self._format_previous_results([batch_data])
                        context["context_batch_count"] = 1
                    break
        
        return context
    
    def _format_previous_results(self, results: List[Dict]) -> str:
        """格式化多批结果为易读文本"""
        if not results:
            return ""
        
        all_parts = []
        
        for idx, result in enumerate(results):
            formatted = self._format_single_result(result, idx + 1, len(results))
            if formatted:
                all_parts.append(formatted)
        
        return "\n\n".join(all_parts)
    
    def _format_single_result(self, result: Dict, batch_num: int, total_batches: int) -> str:
        """格式化单批结果"""
        if not result or result.get("parse_error"):
            return ""
        
        parts = []
        
        # 页面范围
        page_range = result.get("page_range", {})
        if page_range:
            if total_batches > 1:
                parts.append(f"【前第{total_batches - batch_num + 1}批：第{page_range.get('start', '?')}-{page_range.get('end', '?')}页】")
            else:
                parts.append(f"【第{page_range.get('start', '?')}-{page_range.get('end', '?')}页】")
        
        # 批次摘要
        batch_summary = result.get("batch_summary", "")
        if batch_summary:
            if len(batch_summary) > 600:
                batch_summary = batch_summary[:600] + "..."
            parts.append(f"剧情: {batch_summary}")
        
        # 关键事件
        key_events = result.get("key_events", [])
        if key_events:
            valid_events = [str(e) for e in key_events[:3] if e]
            if valid_events:
                parts.append(f"事件: {'; '.join(valid_events)}")
        
        return "\n".join(parts)
    
    async def generate_segment_summary(
        self,
        segment_id: str,
        batch_results: List[Dict],
        force: bool = False
    ) -> Dict:
        """
        生成小总结（第二层级）
        
        Args:
            segment_id: 小总结ID
            batch_results: 批量分析结果列表
            force: 是否强制重新生成
        
        Returns:
            Dict: 小总结
        """
        if not batch_results:
            return {}
        
        # 检查缓存
        if not force:
            cached = await self.storage.load_segment_summary(segment_id)
            if cached:
                logger.debug(f"使用小总结缓存: {segment_id}")
                return cached
        
        # 收集批量摘要
        batch_summaries = []
        all_key_events = []
        start_page = float('inf')
        end_page = 0
        
        for batch in batch_results:
            page_range = batch.get("page_range", {})
            batch_start = page_range.get("start", 0)
            batch_end = page_range.get("end", 0)
            
            start_page = min(start_page, batch_start)
            end_page = max(end_page, batch_end)
            
            summary = batch.get("batch_summary", "")
            if summary:
                batch_summaries.append(f"第{batch_start}-{batch_end}页: {summary}")
            
            events = batch.get("key_events", [])
            all_key_events.extend(events)
        
        # 使用 LLM 生成小总结
        from .embedding_client import ChatClient
        
        segment_data = {
            "segment_id": segment_id,
            "page_range": {"start": int(start_page), "end": int(end_page)},
            "summary": "",
            "key_events": all_key_events[:10],
            "plot_progression": "",
            "themes": [],
            "batch_count": len(batch_results)
        }
        
        try:
            if self.config.chat_llm.use_same_as_vlm:
                llm = ChatClient(self.config.vlm)
            else:
                llm = ChatClient(self.config.chat_llm)
            
            # 构建提示词（优先使用用户自定义，否则使用默认）
            batch_summaries_text = "\n".join(batch_summaries)
            base_prompt = self.config.prompts.segment_summary if self.config.prompts.segment_summary else DEFAULT_SEGMENT_SUMMARY_PROMPT
            # 使用 replace 而不是 format，避免 JSON 示例中的 {} 被误解析
            prompt = base_prompt.replace("{batch_count}", str(len(batch_results)))
            prompt = prompt.replace("{start_page}", str(int(start_page)))
            prompt = prompt.replace("{end_page}", str(int(end_page)))
            prompt = prompt.replace("{batch_summaries}", batch_summaries_text)
            prompt = prompt.replace("{segment_id}", str(segment_id))
            
            # 优先使用用户自定义提示词，否则使用默认
            system_prompt = self.config.prompts.analysis_system if self.config.prompts.analysis_system else DEFAULT_ANALYSIS_SYSTEM_PROMPT
            response = await llm.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3
            )
            
            # 解析 JSON 响应
            try:
                text = response.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                
                parsed = json.loads(text.strip())
                segment_data.update(parsed)
            except json.JSONDecodeError:
                # 解析失败，使用简单合并
                segment_data["summary"] = "\n".join(batch_summaries[:3])
            
            await llm.close()
            
        except Exception as e:
            logger.warning(f"小总结生成失败: {e}")
            segment_data["summary"] = "\n".join(batch_summaries[:3])
        
        # 保存小总结
        segment_data["generated_at"] = datetime.now().isoformat()
        await self.storage.save_segment_summary(segment_id, segment_data)
        
        return segment_data
    
    async def analyze_chapter_with_segments(
        self,
        chapter_id: str,
        progress_callback=None
    ) -> Dict:
        """
        使用动态层级模式分析章节
        
        根据配置的架构执行相应层级的分析
        
        Args:
            chapter_id: 章节ID
            progress_callback: 进度回调
        
        Returns:
            Dict: 章节分析结果
        """
        batch_settings = self.config.analysis.batch
        
        book_info = await self.get_book_info()
        chapters = book_info.get("chapters", [])
        all_images = book_info.get("all_images", [])
        
        # 找到章节信息
        chapter = None
        for ch in chapters:
            if ch.get("id") == chapter_id or ch.get("chapter_id") == chapter_id:
                chapter = ch
                break
        
        if not chapter:
            raise ValueError(f"未找到章节: {chapter_id}")
        
        # 获取该章节的所有图片
        chapter_images = [
            (idx + 1, img) for idx, img in enumerate(all_images)
            if img.get("chapter_id") == chapter_id
        ]
        
        if not chapter_images:
            raise ValueError(f"章节 {chapter_id} 没有图片")
        
        total_pages = len(chapter_images)
        pages_per_batch = batch_settings.pages_per_batch
        context_batch_count = batch_settings.context_batch_count
        
        # 获取层级配置
        layers = batch_settings.get_layers()
        layer_names = [l["name"] for l in layers]
        
        logger.info(f"动态层级分析章节 {chapter_id}: {total_pages}页, "
                    f"架构: {' → '.join(layer_names)}, "
                    f"上文参考{context_batch_count}批")
        
        # 第一层: 批量分析
        batch_results = []
        batch_idx = 0
        total_batches = (total_pages + pages_per_batch - 1) // pages_per_batch
        
        for i in range(0, total_pages, pages_per_batch):
            batch_pages = chapter_images[i:i + pages_per_batch]
            page_nums = [p[0] for p in batch_pages]
            image_infos = [p[1] for p in batch_pages]
            
            if progress_callback:
                progress_callback(
                    phase="batch",
                    current=batch_idx + 1,
                    total=total_batches,
                    message=f"批量分析第{page_nums[0]}-{page_nums[-1]}页"
                )
            
            # 获取前 N 批的结果作为上下文
            previous_results = []
            if context_batch_count > 0:
                valid_results = [r for r in batch_results if not r.get("parse_error")]
                previous_results = valid_results[-context_batch_count:]
            
            batch_result = await self.analyze_batch(page_nums, image_infos=image_infos, previous_results=previous_results)
            batch_results.append(batch_result)
            batch_idx += 1
        
        # 中间层: 按架构执行汇总（如果有）
        current_results = batch_results
        
        for layer_idx in range(1, len(layers) - 1):
            layer = layers[layer_idx]
            layer_name = layer.get("name", f"层级{layer_idx}")
            units_per_group = layer.get("units_per_group", 5)
            
            if units_per_group <= 0:
                units_per_group = max(len(current_results), 1)  # 0表示全部汇总，防止除零
            
            if not current_results:
                continue  # 跳过空结果
            
            total_groups = (len(current_results) + units_per_group - 1) // units_per_group
            segment_results = []
            
            for group_idx in range(total_groups):
                start = group_idx * units_per_group
                end = min(start + units_per_group, len(current_results))
                group_items = current_results[start:end]
                
                segment_id = f"{chapter_id}_L{layer_idx}_{group_idx:03d}"
                
                if progress_callback:
                    progress_callback(
                        phase=layer_name,
                        current=group_idx + 1,
                        total=total_groups,
                        message=f"生成{layer_name} {segment_id}"
                    )
                
                segment_result = await self.generate_segment_summary(segment_id, group_items)
                segment_results.append(segment_result)
            
            current_results = segment_results
        
        # 最后层: 生成章节总结
        if progress_callback:
            progress_callback(
                phase="chapter",
                current=1,
                total=1,
                message=f"生成章节总结"
            )
        
        chapter_analysis = await self._generate_chapter_from_segments(
            chapter_id, chapter, current_results
        )
        
        # 保存章节分析
        await self.storage.save_chapter_analysis(chapter_id, chapter_analysis)
        
        return chapter_analysis
    
    async def _generate_chapter_from_segments(
        self,
        chapter_id: str,
        chapter_info: Dict,
        segment_results: List[Dict]
    ) -> Dict:
        """从小总结生成章节总结"""
        from .embedding_client import ChatClient
        
        if not segment_results:
            return {
                "chapter_id": chapter_id,
                "title": chapter_info.get("title", f"第{chapter_id}章"),
                "summary": "",
                "analyzed_at": datetime.now().isoformat()
            }
        
        # 收集小总结信息
        segment_summaries = []
        all_events = []
        start_page = float('inf')
        end_page = 0
        
        for seg in segment_results:
            page_range = seg.get("page_range", {})
            start_page = min(start_page, page_range.get("start", 0))
            end_page = max(end_page, page_range.get("end", 0))
            
            summary = seg.get("summary", "")
            if summary:
                seg_range = f"第{page_range.get('start')}-{page_range.get('end')}页"
                segment_summaries.append(f"{seg_range}: {summary}")
            
            all_events.extend(seg.get("key_events", []))
        
        # 使用 LLM 生成章节总结
        chapter_summary = ""
        try:
            if self.config.chat_llm.use_same_as_vlm:
                llm = ChatClient(self.config.vlm)
            else:
                llm = ChatClient(self.config.chat_llm)
            
            segment_summaries_text = "\n".join(segment_summaries)
            # 优先使用用户自定义提示词，否则使用默认
            base_prompt = self.config.prompts.chapter_summary if self.config.prompts.chapter_summary else DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT
            # 使用 replace 而不是 format，避免 JSON 示例中的 {} 被误解析
            prompt = base_prompt.replace("{chapter_title}", chapter_info.get("title", f"第{chapter_id}章"))
            prompt = prompt.replace("{chapter_id}", str(chapter_id))
            prompt = prompt.replace("{start_page}", str(int(start_page)))
            prompt = prompt.replace("{end_page}", str(int(end_page)))
            prompt = prompt.replace("{segment_summaries}", segment_summaries_text)
            
            # 优先使用用户自定义提示词，否则使用默认
            system_prompt = self.config.prompts.analysis_system if self.config.prompts.analysis_system else DEFAULT_ANALYSIS_SYSTEM_PROMPT
            response = await llm.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3
            )
            
            # 解析 JSON
            try:
                text = response.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                
                parsed = json.loads(text.strip())
                chapter_summary = parsed.get("summary", "")
                
                return {
                    "chapter_id": chapter_id,
                    "title": chapter_info.get("title", f"第{chapter_id}章"),
                    "page_range": {"start": int(start_page), "end": int(end_page)},
                    "analyzed_at": datetime.now().isoformat(),
                    "analysis_mode": "four_tier",
                    "summary": chapter_summary,
                    "main_plot": parsed.get("main_plot", ""),
                    "plot_events": parsed.get("key_events", all_events[:10]),
                    "themes": parsed.get("themes", []),
                    "atmosphere": parsed.get("atmosphere", ""),
                    "connections": parsed.get("connections", {}),
                    "segment_count": len(segment_results)
                }
            except json.JSONDecodeError:
                chapter_summary = response.strip()
            
            await llm.close()
            
        except Exception as e:
            logger.warning(f"章节总结生成失败: {e}")
            chapter_summary = "\n".join(segment_summaries[:3])
        
        return {
            "chapter_id": chapter_id,
            "title": chapter_info.get("title", f"第{chapter_id}章"),
            "page_range": {"start": int(start_page), "end": int(end_page)},
            "analyzed_at": datetime.now().isoformat(),
            "analysis_mode": "four_tier",
            "summary": chapter_summary,
            "plot_events": all_events[:10],
            "segment_count": len(segment_results)
        }
    
    async def generate_chapter_summary_from_batches(
        self,
        chapter_id: str,
        batch_results: List[Dict],
        all_images: List[Dict]
    ) -> Dict:
        """
        从已有的批量分析结果生成章节摘要（不重新分析页面）
        
        Args:
            chapter_id: 章节ID
            batch_results: 全书的批量分析结果列表
            all_images: 全书图片列表
        
        Returns:
            Dict: 章节分析结果
        """
        # 获取章节信息
        book_info = await self.get_book_info()
        chapters = book_info.get("chapters", [])
        
        chapter = None
        for ch in chapters:
            if ch.get("id") == chapter_id or ch.get("chapter_id") == chapter_id:
                chapter = ch
                break
        
        if not chapter:
            logger.warning(f"未找到章节: {chapter_id}")
            return {}
        
        # 找出属于该章节的页码范围
        chapter_page_nums = set()
        for idx, img in enumerate(all_images):
            if img.get("chapter_id") == chapter_id:
                chapter_page_nums.add(idx + 1)  # 页码从1开始
        
        if not chapter_page_nums:
            logger.warning(f"章节 {chapter_id} 没有图片")
            return {}
        
        chapter_start = min(chapter_page_nums)
        chapter_end = max(chapter_page_nums)
        
        # 筛选属于该章节的批量分析结果
        chapter_batches = []
        for batch in batch_results:
            page_range = batch.get("page_range", {})
            batch_start = page_range.get("start", 0)
            batch_end = page_range.get("end", 0)
            
            # 如果批次与章节有重叠，则包含此批次
            if batch_start <= chapter_end and batch_end >= chapter_start:
                chapter_batches.append(batch)
        
        logger.info(f"章节 {chapter_id}: 页面 {chapter_start}-{chapter_end}, 包含 {len(chapter_batches)} 个批次")
        
        if not chapter_batches:
            logger.warning(f"章节 {chapter_id} 没有对应的批量分析结果")
            return {}
        
        # 收集信息生成摘要
        all_summaries = []
        all_events = []
        
        for batch in chapter_batches:
            batch_summary = batch.get("batch_summary", "")
            if batch_summary:
                page_range = batch.get("page_range", {})
                all_summaries.append(f"第{page_range.get('start')}-{page_range.get('end')}页: {batch_summary}")
            
            for event in batch.get("key_events", []):
                if event:
                    all_events.append(event)
        
        # 生成章节摘要（简单合并，或调用 LLM 生成）
        chapter_summary = " ".join([s.split(": ", 1)[-1] for s in all_summaries[:5]])
        
        result = {
            "chapter_id": chapter_id,
            "title": chapter.get("title", f"第{chapter_id}章"),
            "page_range": {"start": chapter_start, "end": chapter_end},
            "analyzed_at": datetime.now().isoformat(),
            "analysis_mode": "batch_summary",
            "summary": chapter_summary[:1000] if chapter_summary else "",
            "plot_events": all_events[:10],
            "batch_count": len(chapter_batches)
        }
        
        # 保存章节分析
        await self.storage.save_chapter_analysis(chapter_id, result)
        
        return result
    
    async def reanalyze_batch(self, start_page: int, end_page: int) -> Dict:
        """重新分析指定批次"""
        # 删除旧的批量分析
        await self.storage.delete_batch_analysis(start_page, end_page)
        
        # 构建页码列表
        page_nums = list(range(start_page, end_page + 1))
        
        # 重新分析
        return await self.analyze_batch(page_nums, force=True)
    
    async def reanalyze_segment(self, segment_id: str) -> Dict:
        """重新生成指定小总结"""
        # 加载现有小总结获取页面范围
        existing = await self.storage.load_segment_summary(segment_id)
        if not existing:
            raise ValueError(f"未找到小总结: {segment_id}")
        
        page_range = existing.get("page_range", {})
        start_page = page_range.get("start", 0)
        end_page = page_range.get("end", 0)
        
        # 获取该范围内的批量分析结果
        batches = await self.storage.list_batches()
        batch_results = []
        for batch in batches:
            if batch["start_page"] >= start_page and batch["end_page"] <= end_page:
                batch_data = await self.storage.load_batch_analysis(
                    batch["start_page"], batch["end_page"]
                )
                if batch_data:
                    batch_results.append(batch_data)
        
        # 重新生成
        return await self.generate_segment_summary(segment_id, batch_results, force=True)
    
    async def build_embeddings(self) -> Dict:
        """
        构建向量嵌入（页面 + 事件）
        
        基于批量分析结果构建两层索引：
        1. 页面级向量 (page_summary)
        2. 事件级向量 (key_events) - 细粒度检索
        
        Returns:
            Dict: 构建结果统计
        """
        if not self.embedding or not self.vector_store.is_available():
            logger.warning("向量功能不可用，跳过构建嵌入")
            return {"success": False, "error": "向量功能不可用"}
        
        # 获取所有批次
        batches = await self.storage.list_batches()
        
        if not batches:
            # 降级：从页面分析构建
            logger.info("无批次数据，从页面分析构建向量")
            return await self._build_embeddings_from_pages()
        
        logger.info(f"开始构建向量嵌入: 共 {len(batches)} 个批次")
        
        # 清除现有向量
        await self.vector_store.delete_all_pages()
        await self.vector_store.delete_all_events()
        
        pages_count = 0
        events_count = 0
        skip_count = 0
        
        for batch_info in tqdm(batches, desc="构建向量嵌入", unit="批次"):
            start_page = batch_info["start_page"]
            end_page = batch_info["end_page"]
            batch_id = f"batch_{start_page}_{end_page}"
            
            batch_data = await self.storage.load_batch_analysis(start_page, end_page)
            if not batch_data:
                skip_count += 1
                continue
            
            # ========== 1. 页面级向量 ==========
            for page in batch_data.get("pages", []):
                page_num = page.get("page_number")
                page_summary = page.get("page_summary", "")
                
                if page_num and page_summary:
                    try:
                        embedding = await self.embedding.embed(page_summary)
                        await self.vector_store.add_page_embedding(
                            page_num=page_num,
                            embedding=embedding,
                            metadata={
                                "page_summary": page_summary,
                                "type": "page",
                                "parent_batch": batch_id
                            }
                        )
                        pages_count += 1
                    except Exception as e:
                        logger.warning(f"页面 {page_num} 向量化失败: {e}")
            
            # ========== 2. 事件级向量 (key_events) ==========
            key_events = batch_data.get("key_events", [])
            for event_idx, event in enumerate(key_events):
                # 过滤无效事件
                if not event or not isinstance(event, str):
                    continue
                event = event.strip()
                if len(event) < 5:  # 太短的事件跳过
                    continue
                
                try:
                    embedding = await self.embedding.embed(event)
                    event_id = f"event_{start_page}_{end_page}_{event_idx}"
                    
                    await self.vector_store.add_event_embedding(
                        event_id=event_id,
                        embedding=embedding,
                        metadata={
                            "content": event,
                            "type": "event",
                            "parent_batch": batch_id,
                            "start_page": start_page,
                            "end_page": end_page
                        }
                    )
                    events_count += 1
                except Exception as e:
                    logger.warning(f"事件向量化失败 ({batch_id}): {e}")
        
        result = {
            "success": True,
            "pages_count": pages_count,
            "events_count": events_count,
            "total_count": pages_count + events_count,
            "batches_processed": len(batches) - skip_count,
            "batches_skipped": skip_count
        }
        
        logger.info(
            f"向量嵌入构建完成: {pages_count} 页面, {events_count} 事件, "
            f"共 {pages_count + events_count} 条向量"
        )
        return result
    
    async def _build_embeddings_from_pages(self) -> Dict:
        """从页面分析构建向量（降级方案）"""
        page_nums = await self.storage.list_pages()
        
        if not page_nums:
            logger.warning("没有已分析的页面，跳过构建嵌入")
            return {"success": False, "error": "没有已分析的页面"}
        
        logger.info(f"从页面分析构建向量: 共 {len(page_nums)} 页")
        
        # 清除现有向量
        await self.vector_store.delete_all_pages()
        
        pages_count = 0
        skip_count = 0
        
        for page_num in tqdm(page_nums, desc="构建向量嵌入", unit="页"):
            analysis = await self.storage.load_page_analysis(page_num)
            if not analysis:
                skip_count += 1
                continue
            
            summary = analysis.get("page_summary", "")
            if summary:
                try:
                    embedding = await self.embedding.embed(summary)
                    await self.vector_store.add_page_embedding(
                        page_num, embedding, {
                            "page_summary": summary,
                            "type": "page"
                        }
                    )
                    pages_count += 1
                except Exception as e:
                    logger.warning(f"第 {page_num} 页向量化失败: {e}")
                    skip_count += 1
            else:
                skip_count += 1
        
        result = {
            "success": pages_count > 0,
            "pages_count": pages_count,
            "events_count": 0,
            "total_count": pages_count,
            "skip_count": skip_count
        }
        
        logger.info(f"向量嵌入构建完成: 成功 {pages_count} 页, 跳过 {skip_count} 页")
        return result
    
    async def generate_overview(self) -> Dict:
        """生成全书概述（层级式摘要）
        
        优先级：
        1. 从章节总结生成（如果有 chapter 分析结果）
        2. 从段落总结生成（如果有 segment 分析结果）
        3. 从批量分析生成（如果有 batch 分析结果）
        4. 从页面摘要生成（降级方案）
        
        同时会自动生成默认的「故事概要」模板。
        """
        from .features.hierarchical_summary import HierarchicalSummaryGenerator
        from .embedding_client import ChatClient
        
        book_info = await self.get_book_info()
        chapters = await self.storage.list_chapters()
        
        # 创建 LLM 客户端用于层级摘要
        llm_client = None
        try:
            if self.config.chat_llm.use_same_as_vlm:
                llm_client = ChatClient(self.config.vlm)
            else:
                llm_client = ChatClient(self.config.chat_llm)
        except Exception as e:
            logger.warning(f"LLM 客户端初始化失败，将使用简单合并: {e}")
        
        # 使用层级式摘要生成器（传入提示词配置）
        summary_generator = HierarchicalSummaryGenerator(
            book_id=self.book_id, 
            storage=self.storage,
            llm_client=llm_client,
            book_info=book_info,
            prompts_config=self.config.prompts  # 传入提示词配置
        )
        
        summary_source = "none"
        book_summary = ""
        section_summaries = []
        
        try:
            # 先生成层级概述（会构建压缩摘要）
            hierarchical_result = await summary_generator.generate_hierarchical_overview()
            book_summary = hierarchical_result.get("book_summary", "")
            section_summaries = hierarchical_result.get("section_summaries", [])
            summary_source = hierarchical_result.get("source", "unknown")
            logger.info(f"概要生成完成，数据来源: {summary_source}")
            
            # 自动生成默认的「无剧透简介」模板
            try:
                template_result = await summary_generator.generate_with_template("no_spoiler")  # 【修复】默认生成无剧透简介
                logger.info(f"无剧透简介模板生成完成")
            except Exception as e:
                logger.warning(f"无剧透简介模板生成失败: {e}")
                
        except Exception as e:
            logger.error(f"层级摘要生成失败: {e}", exc_info=True)
            book_summary = "概要生成失败，请重试。"
            section_summaries = []
        finally:
            # 关闭 LLM 客户端，释放资源
            if llm_client:
                try:
                    await llm_client.close()
                except Exception:
                    pass
        
        overview = {
            "book_id": self.book_id,
            "title": book_info.get("title", ""),
            "total_pages": book_info.get("total_pages", 0),
            "total_chapters": len(chapters),
            "summary": book_summary,
            "section_summaries": section_summaries,
            "summary_source": summary_source,  # 记录数据来源
            "themes": [],
            "generated_at": datetime.now().isoformat()
        }
        
        await self.storage.save_overview(overview)
        return overview
