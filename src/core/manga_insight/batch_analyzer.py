# src/core/manga_insight/batch_analyzer.py
"""
批量分析器模块

从 analyzer.py 拆分，负责批量分析逻辑。
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from .storage import AnalysisStorage
from .vlm_client import VLMClient

logger = logging.getLogger("MangaInsight.BatchAnalyzer")


class BatchAnalyzer:
    """
    批量分析器

    负责多页批量分析，构建上下文，格式化结果。
    """

    def __init__(self, book_id: str, storage: AnalysisStorage, vlm: VLMClient):
        self.book_id = book_id
        self.storage = storage
        self.vlm = vlm

    async def analyze_batch(
        self,
        page_nums: List[int],
        images: List[bytes] = None,
        image_infos: List[Dict] = None,
        force: bool = False,
        previous_results: List[Dict] = None,
        get_image_func=None
    ) -> Dict:
        """
        批量分析多页（第一层级）

        Args:
            page_nums: 页码列表
            images: 图片数据列表
            image_infos: 图片信息列表
            force: 是否强制重新分析
            previous_results: 前 N 批的分析结果列表，用于上下文连贯
            get_image_func: 获取图片的回调函数

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
        if images is None and get_image_func:
            images = []
            for i, page_num in enumerate(page_nums):
                img_info = image_infos[i] if image_infos and i < len(image_infos) else None
                img_data = await get_image_func(page_num, img_info)
                images.append(img_data)

        # 构建上下文
        context = self._build_batch_context(start_page, previous_results)

        # 调用 VLM 批量分析
        context_count = len(previous_results) if previous_results else 0
        logger.info(f"批量分析: 第{start_page}-{end_page}页 ({len(images)}张图片) [上文{context_count}批]")
        result = await self.vlm.analyze_batch(images, start_page, context)
        logger.info(f"批量分析完成: 第{start_page}-{end_page}页, 结果包含 {len(result.get('pages', []))} 个页面")

        # 添加元数据
        result["analyzed_at"] = datetime.now().isoformat()
        result["analysis_mode"] = "batch"

        # 保存批量结果
        await self.storage.save_batch_analysis(start_page, end_page, result)

        # 同时保存单页结果
        if result.get("pages"):
            for page_data in result["pages"]:
                page_num = page_data.get("page_number")
                if page_num:
                    page_data["from_batch"] = True
                    page_data["batch_range"] = {"start": start_page, "end": end_page}
                    page_data["analyzed_at"] = result["analyzed_at"]
                    await self.storage.save_page_analysis(page_num, page_data)

        return result

    def _build_batch_context(self, start_page: int, previous_results: List[Dict] = None) -> Dict:
        """构建批量分析上下文"""
        context = {}

        if previous_results and len(previous_results) > 0:
            context["previous_summary"] = self._format_previous_results(previous_results)
            context["context_batch_count"] = len(previous_results)

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
            start = page_range.get("start", "?")
            end = page_range.get("end", "?")
            if total_batches > 1:
                parts.append(f"【前第{total_batches - batch_num + 1}批：第{start}-{end}页】")
            else:
                parts.append(f"【第{start}-{end}页】")

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
