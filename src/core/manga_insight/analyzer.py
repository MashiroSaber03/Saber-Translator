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

from .config_models import MangaInsightConfig
from .storage import AnalysisStorage
from .vlm_client import VLMClient
from .embedding_client import EmbeddingClient
from .vector_store import MangaVectorStore

# 导入拆分的模块
from .batch_analyzer import BatchAnalyzer
from .summary_generator import SummaryGenerator
from .embedding_builder import EmbeddingBuilder
from .overview_generator import OverviewGenerator

logger = logging.getLogger("MangaInsight.Analyzer")


class MangaAnalyzer:
    """
    漫画分析器

    使用漫画原图进行分析，所有输出为中文。
    协调 BatchAnalyzer、SummaryGenerator、EmbeddingBuilder 完成分析任务。
    """

    def __init__(self, book_id: str, config: MangaInsightConfig):
        self.book_id = book_id
        self.config = config
        self.storage = AnalysisStorage(book_id)
        self.vlm = VLMClient(config.vlm, config.prompts)
        self.embedding = EmbeddingClient(config.embedding) if config.embedding.api_key else None
        self.vector_store = MangaVectorStore(book_id)
        self._book_info_cache = None

        # 初始化子模块
        self._batch_analyzer = BatchAnalyzer(book_id, self.storage, self.vlm)
        self._summary_generator = SummaryGenerator(book_id, config, self.storage)
        self._embedding_builder = EmbeddingBuilder(
            book_id, self.storage, self.embedding, self.vector_store
        )
        self._overview_generator = OverviewGenerator(
            book_id, config, self.storage, self.get_book_info
        )

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

                for chapter in chapters:
                    chapter_id = chapter.get("id")
                    if chapter_id:
                        # 从 session_meta.json 获取图片信息（使用新路径格式）
                        session_dir = resource_path(f"data/bookshelf/{self.book_id}/chapters/{chapter_id}/session")
                        session_meta_path = os.path.join(session_dir, "session_meta.json")

                        if os.path.exists(session_meta_path):
                            try:
                                with open(session_meta_path, "r", encoding="utf-8") as f:
                                    session_data = json.load(f)

                                # 支持两种格式：新格式使用 total_pages，旧格式使用 images_meta
                                if "total_pages" in session_data:
                                    # 新格式: images/{idx}/original.png
                                    image_count = session_data.get("total_pages", 0)
                                    for i in range(image_count):
                                        # 新格式路径
                                        image_path = os.path.join(session_dir, "images", str(i), "original.png")
                                        page_meta_path = os.path.join(session_dir, "images", str(i), "meta.json")

                                        # 加载页面元数据（包含 fileName, relativePath 等）
                                        page_meta = {}
                                        if os.path.exists(page_meta_path):
                                            try:
                                                with open(page_meta_path, "r", encoding="utf-8") as mf:
                                                    page_meta = json.load(mf)
                                            except Exception:
                                                pass

                                        if os.path.exists(image_path):
                                            all_images.append({
                                                "chapter_id": chapter_id,
                                                "index": i,
                                                "path": image_path,
                                                "meta": page_meta
                                            })
                                            total_pages += 1
                                else:
                                    # 旧格式: image_{idx}_original.png
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

                # 【多文件夹支持】按 relativePath/fileName 进行自然排序
                def natural_sort_key(item):
                    path = item.get("meta", {}).get("relativePath") or item.get("meta", {}).get("fileName", "")
                    return [int(text) if text.isdigit() else text.lower()
                            for text in re.split(r'(\d+)', path)]

                all_images = sorted(all_images, key=natural_sort_key)

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
        """从图片信息获取图片数据，无信息时回退到原图获取"""
        if image_info:
            image_path = image_info.get("path")
            if image_path and os.path.exists(image_path):
                logger.debug(f"读取图片: {image_path}")
                with open(image_path, "rb") as f:
                    return f.read()

        # 回退到原图获取逻辑
        return await self.get_original_image(page_num)

    # ============================================================
    # 批量分析模式（委托给 BatchAnalyzer）
    # ============================================================

    async def analyze_batch(
        self,
        page_nums: List[int],
        images: List[bytes] = None,
        image_infos: List[Dict] = None,
        force: bool = False,
        previous_results: List[Dict] = None
    ) -> Dict:
        """批量分析多页（第一层级）- 委托给 BatchAnalyzer"""
        return await self._batch_analyzer.analyze_batch(
            page_nums=page_nums,
            images=images,
            image_infos=image_infos,
            force=force,
            previous_results=previous_results,
            get_image_func=self._get_image_from_info
        )

    # ============================================================
    # 摘要生成（委托给 SummaryGenerator）
    # ============================================================

    async def generate_segment_summary(
        self,
        segment_id: str,
        batch_results: List[Dict],
        force: bool = False
    ) -> Dict:
        """生成小总结（第二层级）- 委托给 SummaryGenerator"""
        return await self._summary_generator.generate_segment_summary(
            segment_id, batch_results, force
        )

    async def generate_chapter_summary_from_batches(
        self,
        chapter_id: str,
        batch_results: List[Dict],
        all_images: List[Dict]
    ) -> Dict:
        """从已有的批量分析结果生成章节摘要"""
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
                chapter_page_nums.add(idx + 1)

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

        # 生成章节摘要（简单合并）
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

    # ============================================================
    # 章节分析（使用动态层级模式）
    # ============================================================

    async def analyze_chapter_with_segments(
        self,
        chapter_id: str,
        progress_callback=None
    ) -> Dict:
        """使用动态层级模式分析章节"""
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

        # 中间层: 按架构执行汇总
        current_results = batch_results

        for layer_idx in range(1, len(layers) - 1):
            layer = layers[layer_idx]
            layer_name = layer.get("name", f"层级{layer_idx}")
            units_per_group = layer.get("units_per_group", 5)

            if units_per_group <= 0:
                units_per_group = max(len(current_results), 1)

            if not current_results:
                continue

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

        chapter_analysis = await self._summary_generator.generate_chapter_from_segments(
            chapter_id, chapter, current_results
        )

        # 保存章节分析
        await self.storage.save_chapter_analysis(chapter_id, chapter_analysis)

        return chapter_analysis

    # ============================================================
    # 重新分析
    # ============================================================

    async def reanalyze_batch(self, start_page: int, end_page: int) -> Dict:
        """重新分析指定批次"""
        await self.storage.delete_batch_analysis(start_page, end_page)
        page_nums = list(range(start_page, end_page + 1))
        return await self.analyze_batch(page_nums, force=True)

    async def reanalyze_segment(self, segment_id: str) -> Dict:
        """重新生成指定小总结"""
        existing = await self.storage.load_segment_summary(segment_id)
        if not existing:
            raise ValueError(f"未找到小总结: {segment_id}")

        page_range = existing.get("page_range", {})
        start_page = page_range.get("start", 0)
        end_page = page_range.get("end", 0)

        batches = await self.storage.list_batches()
        batch_results = []
        for batch in batches:
            if batch["start_page"] >= start_page and batch["end_page"] <= end_page:
                batch_data = await self.storage.load_batch_analysis(
                    batch["start_page"], batch["end_page"]
                )
                if batch_data:
                    batch_results.append(batch_data)

        return await self.generate_segment_summary(segment_id, batch_results, force=True)

    # ============================================================
    # 向量嵌入（委托给 EmbeddingBuilder）
    # ============================================================

    async def build_embeddings(self) -> Dict:
        """构建向量嵌入（页面 + 事件）- 委托给 EmbeddingBuilder"""
        return await self._embedding_builder.build_embeddings()

    # ============================================================
    # 全书概览（委托给 OverviewGenerator）
    # ============================================================

    async def generate_overview(self) -> Dict:
        """生成全书概述（层级式摘要）- 委托给 OverviewGenerator"""
        return await self._overview_generator.generate_overview()
