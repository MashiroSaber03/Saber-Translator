# src/core/manga_insight/task_executor.py
"""
任务执行器模块

从 task_manager.py 拆分，负责具体的分析任务执行逻辑。
"""

import logging
from typing import Dict, List, Callable, Optional

from .task_models import AnalysisTask, TaskType
from .config_utils import load_insight_config
from .layer_executor import LayerExecutor

logger = logging.getLogger("MangaInsight.TaskExecutor")


class TaskExecutor:
    """
    任务执行器

    负责执行具体的分析任务：
    - 全书分析
    - 章节分析
    - 增量分析
    - 重新分析
    """

    def __init__(
        self,
        check_pause_cancel_func: Callable[[str], bool],
        notify_progress_func: Callable[[str, Dict], None]
    ):
        """
        Args:
            check_pause_cancel_func: 检查暂停/取消状态的回调 (task_id) -> bool
            notify_progress_func: 通知进度更新的回调 (task_id, progress) -> None
        """
        self._check_pause_and_cancel = check_pause_cancel_func
        self._notify_progress = notify_progress_func

    async def execute(self, task: AnalysisTask, analyzer) -> List[str]:
        """
        执行分析任务

        Args:
            task: 任务对象
            analyzer: MangaAnalyzer 实例
        """
        if task.task_type == TaskType.FULL_BOOK:
            return await self.execute_full_book_analysis(task, analyzer)
        elif task.task_type == TaskType.CHAPTER:
            return await self.execute_chapter_analysis(task, analyzer)
        elif task.task_type == TaskType.INCREMENTAL:
            return await self.execute_incremental_analysis(task, analyzer)
        elif task.task_type == TaskType.REANALYZE:
            return await self.execute_reanalysis(task, analyzer)
        return []

    async def execute_full_book_analysis(self, task: AnalysisTask, analyzer) -> List[str]:
        """执行全书分析（支持四层级批量模式）"""
        warnings: List[str] = []

        # 检查 VLM 是否已配置
        if not analyzer.vlm.is_configured():
            raise ValueError("VLM 未配置，请先在设置中配置 VLM 服务商和 API Key")

        if task.force_reanalyze:
            logger.info("全书任务启用强制重算：清理现有分析缓存")
            cleared = await analyzer.storage.clear_all()
            if not cleared:
                warning_msg = "全书重分析前清理缓存失败，继续执行重算"
                logger.warning(warning_msg)
                warnings.append(warning_msg)

        # 获取书籍信息
        book_info = await analyzer.get_book_info()
        all_images = book_info.get("all_images", [])
        total_pages = len(all_images)
        task.progress.total_pages = total_pages

        if total_pages == 0:
            raise ValueError("书籍没有可分析的图片，请先添加章节和图片")

        # 获取配置
        config = load_insight_config()
        pages_per_batch = config.analysis.batch.pages_per_batch

        force_json = config.vlm.force_json
        use_stream = config.vlm.use_stream
        logger.info(f"批量分析: 每批 {pages_per_batch} 页, 强制JSON: {'是' if force_json else '否'}, 流式请求: {'是' if use_stream else '否'}")

        batch_warnings, had_batch_failures = await self._execute_full_book_batch_analysis(task, analyzer, book_info)
        warnings.extend(batch_warnings)

        if had_batch_failures:
            warning_msg = "全书分析存在失败批次，本次不更新内容快照（避免后续增量漏检）"
            logger.warning(warning_msg)
            warnings.append(warning_msg)
        elif self._check_pause_and_cancel(task.task_id):
            snapshot_warning = await self._refresh_content_snapshot(task.book_id, analyzer)
            if snapshot_warning:
                warnings.append(snapshot_warning)
        return warnings

    async def _execute_full_book_batch_analysis(self, task: AnalysisTask, analyzer, book_info: dict) -> tuple[List[str], bool]:
        """执行全书动态层级批量分析"""
        warnings: List[str] = []
        config = load_insight_config()
        batch_settings = config.analysis.batch
        pages_per_batch = max(1, batch_settings.pages_per_batch)
        context_batch_count = batch_settings.context_batch_count

        # 获取层级配置
        layers = batch_settings.get_layers()
        layer_names = [l["name"] for l in layers]

        all_images = book_info.get("all_images", [])
        total_pages = len(all_images)
        chapters = book_info.get("chapters", [])

        # 构建章节页面映射
        chapter_page_map = self._build_chapter_page_map(all_images)

        logger.info(f"开始动态层级分析: {total_pages}页, {len(layers)}层架构: {' → '.join(layer_names)}")
        logger.info(f"每批{pages_per_batch}页, 上文参考{context_batch_count}批")

        # ========== 第一层: 批量分析 ==========
        first_layer = layers[0] if layers else {"name": "批量分析", "units_per_group": 5, "align_to_chapter": False}
        align_to_chapter = first_layer.get("align_to_chapter", False)
        expected_batches = self._estimate_expected_batches(
            all_images=all_images,
            chapter_page_map=chapter_page_map,
            pages_per_batch=pages_per_batch,
            align_to_chapter=align_to_chapter
        )

        task.progress.current_phase = "batch_analysis"
        batch_results = await self._execute_batch_layer(
            task, analyzer, all_images, pages_per_batch, context_batch_count,
            align_to_chapter, chapter_page_map
        )
        parse_error_batches = [result for result in batch_results if result.get("parse_error")]
        dropped_batches = max(0, expected_batches - len(batch_results))
        had_batch_failures = (
            self._check_pause_and_cancel(task.task_id)
            and (dropped_batches > 0 or len(parse_error_batches) > 0)
        )
        if had_batch_failures:
            warning_msg = (
                f"全书批量分析存在失败批次: "
                f"丢失{dropped_batches}批, 解析失败{len(parse_error_batches)}批, 预期{expected_batches}批"
            )
            logger.warning(warning_msg)
            warnings.append(warning_msg)

        valid_batch_results = [result for result in batch_results if not result.get("parse_error")]

        if not valid_batch_results:
            logger.warning("批量分析无结果，跳过后续层级")
            warnings.extend(await self._post_analysis_processing(task, analyzer))
            return warnings, had_batch_failures

        # ========== 中间层: 汇总层级 ==========
        current_results = valid_batch_results

        for layer_idx in range(1, len(layers) - 1):
            if not self._check_pause_and_cancel(task.task_id):
                return warnings, had_batch_failures

            layer = layers[layer_idx]
            layer_name = layer.get("name", f"层级{layer_idx}")
            units_per_group = layer.get("units_per_group", 5)
            align_to_chapter = layer.get("align_to_chapter", False)

            task.progress.current_phase = layer_name
            logger.info(f"开始 {layer_name}...")

            if align_to_chapter and units_per_group == 0:
                current_results = await self._execute_chapter_summary_layer(
                    task, analyzer, current_results, all_images, chapters, layer_name
                )
            else:
                current_results = await self._execute_summary_layer(
                    task, analyzer, current_results, units_per_group, layer_name, layer_idx
                )

        # ========== 最后一层: 全书总结 ==========
        if len(layers) > 1:
            last_layer = layers[-1]
            task.progress.current_phase = last_layer.get("name", "全书总结")
            logger.info(f"开始 {last_layer.get('name', '全书总结')}...")

        warnings.extend(await self._post_analysis_processing(task, analyzer))
        return warnings, had_batch_failures

    @staticmethod
    def _estimate_expected_batches(
        all_images: List[Dict],
        chapter_page_map: Dict[str, List[int]],
        pages_per_batch: int,
        align_to_chapter: bool
    ) -> int:
        """估算第一层批量分析的预期批次数。"""
        if align_to_chapter and chapter_page_map:
            total = 0
            for page_nums in chapter_page_map.values():
                if not page_nums:
                    continue
                total += (len(page_nums) + pages_per_batch - 1) // pages_per_batch
            return total

        total_pages = len(all_images)
        if total_pages <= 0:
            return 0
        return (total_pages + pages_per_batch - 1) // pages_per_batch

    def _build_chapter_page_map(self, all_images: List[Dict]) -> Dict[str, List[int]]:
        """构建章节到页码的映射"""
        chapter_page_map = {}
        for idx, img in enumerate(all_images):
            ch_id = img.get("chapter_id")
            if ch_id:
                if ch_id not in chapter_page_map:
                    chapter_page_map[ch_id] = []
                chapter_page_map[ch_id].append(idx + 1)
        return chapter_page_map

    async def _execute_batch_layer(
        self, task: AnalysisTask, analyzer, all_images: List[Dict],
        pages_per_batch: int, context_batch_count: int,
        align_to_chapter: bool, chapter_page_map: Dict[str, List[int]]
    ) -> List[Dict]:
        """执行批量分析层"""
        def check_func():
            return self._check_pause_and_cancel(task.task_id)

        def progress_cb(batch_idx, total_batches, last_page):
            task.progress.analyzed_pages = last_page
            self._notify_progress(task.task_id, task.progress.to_dict())

        executor = LayerExecutor(analyzer, check_func)
        return await executor.execute_batch_layer(
            all_images, pages_per_batch, context_batch_count,
            align_to_chapter, chapter_page_map, progress_cb
        )

    async def _execute_summary_layer(
        self, task: AnalysisTask, analyzer, input_results: List[Dict],
        units_per_group: int, layer_name: str, layer_idx: int
    ) -> List[Dict]:
        """执行汇总层"""
        def check_func():
            return self._check_pause_and_cancel(task.task_id)

        def progress_cb(current, total):
            self._notify_progress(task.task_id, task.progress.to_dict())

        executor = LayerExecutor(analyzer, check_func)
        return await executor.execute_summary_layer(
            input_results, units_per_group, layer_name, layer_idx, progress_cb
        )

    async def _execute_chapter_summary_layer(
        self, task: AnalysisTask, analyzer, batch_results: List[Dict],
        all_images: List[Dict], chapters: List[Dict], layer_name: str
    ) -> List[Dict]:
        """执行章节汇总层"""
        def check_func():
            return self._check_pause_and_cancel(task.task_id)

        def progress_cb(current, total):
            self._notify_progress(task.task_id, task.progress.to_dict())

        executor = LayerExecutor(analyzer, check_func)
        return await executor.execute_chapter_summary_layer(
            batch_results, all_images, chapters, layer_name, progress_cb
        )

    async def execute_chapter_analysis(self, task: AnalysisTask, analyzer) -> List[str]:
        """执行章节分析（使用动态层级模式）"""
        warnings: List[str] = []
        if not analyzer.vlm.is_configured():
            raise ValueError("VLM 未配置，请先在设置中配置 VLM 服务商和 API Key")

        chapters = task.target_chapters or []
        task.progress.total_pages = len(chapters)
        book_info = await analyzer.get_book_info()
        all_images = book_info.get("all_images", [])
        chapter_page_map = self._build_chapter_page_map(all_images)

        for i, chapter_id in enumerate(chapters):
            if not self._check_pause_and_cancel(task.task_id):
                return warnings

            task.progress.current_phase = f"分析章节: {chapter_id}"

            try:
                if task.force_reanalyze:
                    chapter_pages = chapter_page_map.get(chapter_id, [])
                    await analyzer.storage.clear_cache_for_chapters([chapter_id], chapter_pages)

                def progress_cb(phase, current, total, message):
                    task.progress.current_phase = f"{chapter_id}: {message}"
                    self._notify_progress(task.task_id, task.progress.to_dict())

                await analyzer.analyze_chapter_with_segments(
                    chapter_id,
                    progress_callback=progress_cb,
                    force=task.force_reanalyze
                )

                task.progress.analyzed_pages = i + 1
                logger.info(f"完成章节分析: {chapter_id}")
            except Exception as e:
                logger.error(f"章节分析失败: {chapter_id} - {e}")
                warnings.append(f"章节 {chapter_id} 分析失败: {e}")

            self._notify_progress(task.task_id, task.progress.to_dict())

        warnings.extend(await self._post_analysis_processing(task, analyzer))
        return warnings

    async def execute_incremental_analysis(self, task: AnalysisTask, analyzer) -> List[str]:
        """执行增量分析"""
        warnings: List[str] = []
        from .incremental_analyzer import IncrementalAnalyzer

        incremental = IncrementalAnalyzer(task.book_id, load_insight_config())

        def should_stop():
            return not self._check_pause_and_cancel(task.task_id)

        def on_progress(analyzed, total):
            task.progress.analyzed_pages = analyzed
            task.progress.total_pages = total
            self._notify_progress(task.task_id, task.progress.to_dict())

        result = await incremental.analyze_new_content(
            on_progress=on_progress,
            should_stop=should_stop
        )

        if result.get("status") == "cancelled":
            logger.info(f"增量分析已取消: {task.task_id}")
            return warnings

        if result.get("status") == "no_pages":
            logger.warning(f"增量分析: {result.get('message', '没有可分析的图片')}")
            warnings.append(result.get("message", "没有可分析的图片"))
            return warnings

        if result.get("status") == "no_changes":
            logger.info(f"增量分析: {result.get('message', '无需分析')}")
            cache_cleanup = result.get("cache_cleanup") or {}
            had_cleanup = any(
                isinstance(value, int) and value > 0
                for value in cache_cleanup.values()
            )
            if not had_cleanup:
                return warnings
            logger.info("增量分析无重跑页面，但检测到缓存清理，继续执行后处理")
        else:
            pages_analyzed = result.get("pages_analyzed", 0)
            previously_analyzed = result.get("previously_analyzed", 0)
            total_pages = result.get("total_pages", 0)
            pages_failed = result.get("pages_failed", 0)
            logger.info(f"增量分析完成: 本次分析 {pages_analyzed} 页, 之前已分析 {previously_analyzed} 页, 共 {total_pages} 页")
            if pages_failed:
                warning_msg = f"增量分析存在失败页面: {pages_failed} 页（快照未更新，将在下次重试）"
                logger.warning(warning_msg)
                warnings.append(warning_msg)

        warnings.extend(await self._post_analysis_processing(task, analyzer))
        return warnings

    async def execute_reanalysis(self, task: AnalysisTask, analyzer) -> List[str]:
        """执行重新分析（使用批量分析模式）"""
        warnings: List[str] = []
        pages = task.target_pages or []
        if not pages:
            return warnings

        pages = sorted({
            page for page in pages
            if isinstance(page, int) and not isinstance(page, bool) and page > 0
        })
        if not pages:
            return warnings

        task.progress.total_pages = len(pages)

        book_info = await analyzer.get_book_info()
        all_images = book_info.get("all_images", [])
        batch_settings = analyzer.config.analysis.batch
        pages_per_batch = max(1, batch_settings.pages_per_batch)

        await analyzer.storage.clear_cache_for_pages(pages)

        contiguous_ranges = self._split_contiguous_ranges(pages)
        contiguous_batches: List[List[int]] = []
        for page_range in contiguous_ranges:
            contiguous_batches.extend(
                [page_range[i:i + pages_per_batch] for i in range(0, len(page_range), pages_per_batch)]
            )

        total_batches = len(contiguous_batches)
        analyzed_pages_count = 0

        for batch_idx, batch_pages in enumerate(contiguous_batches):
            if not self._check_pause_and_cancel(task.task_id):
                return warnings

            batch_image_infos = []

            for page_num in batch_pages:
                image_info = all_images[page_num - 1] if page_num <= len(all_images) else None
                batch_image_infos.append(image_info)

            task.progress.current_page = batch_pages[0]

            try:
                result = await analyzer.analyze_batch(
                    page_nums=batch_pages,
                    image_infos=batch_image_infos,
                    force=True
                )
                if not result or result.get("parse_error"):
                    raise RuntimeError("批量结果解析失败")
                analyzed_pages_count += len(batch_pages)
                task.progress.analyzed_pages = analyzed_pages_count
                logger.info(f"完成重新分析批次 {batch_idx + 1}/{total_batches}: 第{batch_pages[0]}-{batch_pages[-1]}页")
            except Exception as e:
                logger.error(f"重新分析批次失败: 第{batch_pages[0]}-{batch_pages[-1]}页 - {e}")
                task.failed_pages.extend(batch_pages)
                warnings.append(f"第{batch_pages[0]}-{batch_pages[-1]}页重分析失败: {e}")

            self._notify_progress(task.task_id, task.progress.to_dict())

        warnings.extend(await self._post_analysis_processing(task, analyzer))
        return warnings

    @staticmethod
    def _split_contiguous_ranges(page_nums: List[int]) -> List[List[int]]:
        """将页码按连续段切分。"""
        if not page_nums:
            return []

        ranges: List[List[int]] = []
        current_range: List[int] = [page_nums[0]]

        for page_num in page_nums[1:]:
            if page_num == current_range[-1] + 1:
                current_range.append(page_num)
            else:
                ranges.append(current_range)
                current_range = [page_num]

        ranges.append(current_range)
        return ranges

    async def _post_analysis_processing(self, task: AnalysisTask, analyzer) -> List[str]:
        """分析完成后的后续处理（嵌入、概述等）"""
        warnings: List[str] = []
        # 生成向量嵌入
        logger.info("开始构建向量嵌入...")
        task.progress.current_phase = "embedding"
        try:
            await analyzer.build_embeddings()
            logger.info("向量嵌入完成")
        except Exception as e:
            logger.error(f"向量嵌入失败: {e}", exc_info=True)
            warnings.append(f"向量嵌入失败: {e}")

        # 生成概述
        logger.info("开始生成概述...")
        task.progress.current_phase = "overview"
        try:
            await analyzer.generate_overview()
            logger.info("概述生成完成")
        except Exception as e:
            logger.error(f"概述生成失败: {e}", exc_info=True)
            warnings.append(f"概述生成失败: {e}")

        return warnings

    async def _refresh_content_snapshot(self, book_id: str, analyzer) -> Optional[str]:
        """重建并保存内容快照，供后续增量分析使用。"""
        try:
            from .change_detector import ContentChangeDetector

            detector = ContentChangeDetector(book_id)
            snapshot = await detector.build_content_snapshot()
            saved = await analyzer.storage.save_content_snapshot(snapshot)
            if not saved:
                warning = "内容快照保存失败，后续增量分析可能退化为全量扫描"
                logger.warning(warning)
                return warning

            logger.info("全书分析后已更新内容快照")
        except Exception as e:
            warning = f"内容快照更新失败: {e}"
            logger.warning(warning)
            return warning
        return None

    async def build_timeline_on_complete(self, book_id: str) -> Optional[str]:
        """
        分析完成后自动构建并保存时间线（增强模式）

        Args:
            book_id: 书籍 ID
        """
        try:
            from .features.timeline_enhanced import EnhancedTimelineBuilder
            from .storage import AnalysisStorage

            logger.info(f"开始构建增强时间线: {book_id}")

            config = load_insight_config()
            builder = EnhancedTimelineBuilder(book_id, config)
            storage = AnalysisStorage(book_id)

            timeline_data = await builder.build(mode="enhanced")
            await storage.save_timeline(timeline_data)

            stats = timeline_data.get("stats", {})
            mode = timeline_data.get("mode", "unknown")

            if mode == "enhanced":
                logger.info(
                    f"增强时间线构建完成: {stats.get('total_events', 0)} 个事件, "
                    f"{stats.get('total_arcs', 0)} 个剧情弧, "
                    f"{stats.get('total_characters', 0)} 个角色"
                )
            else:
                logger.info(f"时间线构建完成（简单模式）: {stats.get('total_events', 0)} 个事件")

        except Exception as e:
            logger.error(f"构建时间线失败: {e}", exc_info=True)
            return f"时间线构建失败: {e}"
        return None
