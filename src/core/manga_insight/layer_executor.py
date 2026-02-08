# src/core/manga_insight/layer_executor.py
"""
层级执行器模块

从 task_manager.py 拆分，负责动态层级分析的执行。
"""

import logging
from typing import Dict, List, Optional, Callable

logger = logging.getLogger("MangaInsight.LayerExecutor")


class LayerExecutor:
    """
    层级执行器

    负责执行动态层级架构的分析流程：
    批次分析 → 段落汇总 → 章节汇总 → 全书概览
    """

    def __init__(self, analyzer, check_pause_cancel_func: Callable[[], bool]):
        """
        Args:
            analyzer: MangaAnalyzer 实例
            check_pause_cancel_func: 检查暂停/取消状态的回调
        """
        self.analyzer = analyzer
        self._check_pause_and_cancel = check_pause_cancel_func

    async def execute_batch_layer(
        self,
        all_images: List[Dict],
        pages_per_batch: int,
        context_batch_count: int,
        align_to_chapter: bool,
        chapter_page_map: Dict[str, List[int]],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        执行批量分析层（第一层）

        Args:
            all_images: 全书图片列表
            pages_per_batch: 每批页数
            context_batch_count: 上文参考批次数
            align_to_chapter: 是否按章节边界对齐
            chapter_page_map: 章节页码映射
            progress_callback: 进度回调

        Returns:
            List[Dict]: 批量分析结果列表
        """
        total_pages = len(all_images)
        batch_results = []

        if align_to_chapter and chapter_page_map:
            # 按章节边界分批
            batch_groups = []
            for ch_id, page_nums in chapter_page_map.items():
                for i in range(0, len(page_nums), pages_per_batch):
                    batch_pages = page_nums[i:i + pages_per_batch]
                    batch_groups.append((ch_id, batch_pages))

            total_batches = len(batch_groups)
            for batch_idx, (ch_id, page_nums) in enumerate(batch_groups):
                if not self._check_pause_and_cancel():
                    return batch_results

                result = await self._analyze_single_batch(
                    all_images, page_nums, batch_idx, total_batches,
                    batch_results, context_batch_count, progress_callback
                )
                if result:
                    result["chapter_id"] = ch_id
                    batch_results.append(result)
        else:
            # 不考虑章节边界，按页数分批
            total_batches = (total_pages + pages_per_batch - 1) // pages_per_batch

            for batch_idx in range(total_batches):
                if not self._check_pause_and_cancel():
                    return batch_results

                start_idx = batch_idx * pages_per_batch
                end_idx = min(start_idx + pages_per_batch, total_pages)
                page_nums = [i + 1 for i in range(start_idx, end_idx)]

                result = await self._analyze_single_batch(
                    all_images, page_nums, batch_idx, total_batches,
                    batch_results, context_batch_count, progress_callback
                )
                if result:
                    batch_results.append(result)

        return batch_results

    async def _analyze_single_batch(
        self,
        all_images: List[Dict],
        page_nums: List[int],
        batch_idx: int,
        total_batches: int,
        batch_results: List[Dict],
        context_batch_count: int,
        progress_callback: Optional[Callable] = None
    ) -> Optional[Dict]:
        """分析单个批次"""
        image_infos = [all_images[p - 1] for p in page_nums]

        # 获取上下文
        previous_results = []
        if context_batch_count > 0:
            valid_results = [r for r in batch_results if not r.get("parse_error")]
            previous_results = valid_results[-context_batch_count:]

        try:
            logger.info(f"批量分析: 第{page_nums[0]}-{page_nums[-1]}页 ({batch_idx+1}/{total_batches}) [上文{len(previous_results)}批]")
            result = await self.analyzer.analyze_batch(
                page_nums, image_infos=image_infos, force=True, previous_results=previous_results
            )

            if progress_callback:
                progress_callback(batch_idx + 1, total_batches, page_nums[-1])

            return result
        except Exception as e:
            logger.error(f"批量分析失败: 第{page_nums[0]}-{page_nums[-1]}页 - {e}")
            return None

    async def execute_summary_layer(
        self,
        input_results: List[Dict],
        units_per_group: int,
        layer_name: str,
        layer_idx: int,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        执行汇总层（按数量分组）

        Args:
            input_results: 输入结果列表
            units_per_group: 每组单元数
            layer_name: 层级名称
            layer_idx: 层级索引
            progress_callback: 进度回调

        Returns:
            List[Dict]: 汇总结果列表
        """
        if not input_results:
            logger.warning(f"{layer_name}: 输入结果为空，跳过")
            return []

        if units_per_group <= 0:
            units_per_group = len(input_results)

        total_groups = (len(input_results) + units_per_group - 1) // units_per_group
        summary_results = []

        for group_idx in range(total_groups):
            if not self._check_pause_and_cancel():
                return summary_results

            start = group_idx * units_per_group
            end = min(start + units_per_group, len(input_results))
            group_items = input_results[start:end]

            segment_id = f"layer{layer_idx}_seg_{group_idx:03d}"

            try:
                logger.info(f"生成{layer_name}: {segment_id} ({group_idx+1}/{total_groups})")
                result = await self.analyzer.generate_segment_summary(segment_id, group_items, force=True)
                summary_results.append(result)
            except Exception as e:
                logger.error(f"{layer_name}生成失败: {segment_id} - {e}")

            if progress_callback:
                progress_callback(group_idx + 1, total_groups)

        return summary_results

    async def execute_chapter_summary_layer(
        self,
        batch_results: List[Dict],
        all_images: List[Dict],
        chapters: List[Dict],
        layer_name: str,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        执行章节汇总层（按章节边界分组）

        Args:
            batch_results: 批量分析结果
            all_images: 全书图片列表
            chapters: 章节列表
            layer_name: 层级名称
            progress_callback: 进度回调

        Returns:
            List[Dict]: 章节汇总结果列表
        """
        chapter_summaries = []

        for ch in chapters:
            if not self._check_pause_and_cancel():
                return chapter_summaries

            ch_id = ch.get("id") or ch.get("chapter_id")
            if not ch_id:
                continue

            try:
                logger.info(f"生成{layer_name}: {ch_id}")
                result = await self.analyzer.generate_chapter_summary_from_batches(
                    ch_id, batch_results, all_images
                )
                if result:
                    chapter_summaries.append(result)
            except Exception as e:
                logger.error(f"{layer_name}生成失败: {ch_id} - {e}", exc_info=True)

            if progress_callback:
                progress_callback(len(chapter_summaries), len(chapters))

        return chapter_summaries
