"""
Manga Insight 增量分析器

仅分析新增或变更的内容。
"""

import logging
from typing import Dict, List, Optional, Callable

from tqdm import tqdm

from .config_models import MangaInsightConfig
from .storage import AnalysisStorage
from .change_detector import ContentChangeDetector, ChangeType, get_pages_to_analyze

logger = logging.getLogger("MangaInsight.IncrementalAnalyzer")


class IncrementalAnalyzer:
    """增量分析执行器"""
    
    def __init__(self, book_id: str, config: MangaInsightConfig):
        self.book_id = book_id
        self.config = config
        self.change_detector = ContentChangeDetector(book_id)
        self.storage = AnalysisStorage(book_id)
    
    async def analyze_new_content(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None
    ) -> Dict:
        """
        分析新增内容（增量分析）
        
        流程:
        1. 加载上次内容快照
        2. 构建当前内容快照并检测变化
        3. 对删除内容先清理缓存
        4. 仅分析新增/修改页面
        5. 成功完成后保存新快照（取消时不覆盖）
        
        Args:
            on_progress: 进度回调函数 (analyzed_count, total_count)
            should_stop: 停止检查回调
        
        Returns:
            Dict: 分析结果摘要
        """
        from .analyzer import MangaAnalyzer
        analyzer = MangaAnalyzer(self.book_id, self.config)

        try:
            # 1) 获取书籍页面
            book_info = await analyzer.get_book_info()
            all_images = book_info.get("all_images", [])
            total_pages = len(all_images)

            if total_pages == 0:
                return {
                    "status": "no_pages",
                    "message": "书籍没有可分析的图片"
                }

            # 2) 快照对比识别变更
            previous_snapshot = await self.storage.load_content_snapshot()
            current_snapshot = await self.change_detector.build_content_snapshot()
            changes = await self.change_detector.detect_changes(current_snapshot, previous_snapshot)

            pages_to_analyze = get_pages_to_analyze(changes)
            deleted_pages, deleted_chapter_ids = self._extract_deleted_changes(changes)

            if should_stop and should_stop():
                logger.info("增量分析在缓存清理前已取消")
                return {
                    "status": "cancelled",
                    "total_pages": total_pages,
                    "pages_analyzed": 0,
                    "pages_remaining": len(pages_to_analyze),
                    "pages_failed": 0,
                    "failed_pages": []
                }

            cache_cleanup: Dict[str, int] = {}
            if deleted_pages or deleted_chapter_ids:
                if should_stop and should_stop():
                    logger.info("增量分析在删除内容清理前已取消")
                    return {
                        "status": "cancelled",
                        "total_pages": total_pages,
                        "pages_analyzed": 0,
                        "pages_remaining": len(pages_to_analyze),
                        "pages_failed": 0,
                        "failed_pages": []
                    }
                deleted_cleanup = await self.storage.clear_cache_for_pages(
                    deleted_pages,
                    chapter_ids=deleted_chapter_ids
                )
                self._merge_cleanup_stats(cache_cleanup, deleted_cleanup)
                logger.info(
                    "增量分析检测到删除内容: pages=%s, chapters=%s",
                    len(deleted_pages),
                    len(deleted_chapter_ids)
                )

            if pages_to_analyze:
                if should_stop and should_stop():
                    logger.info("增量分析在变更页面清理前已取消")
                    return {
                        "status": "cancelled",
                        "total_pages": total_pages,
                        "pages_analyzed": 0,
                        "pages_remaining": len(pages_to_analyze),
                        "pages_failed": 0,
                        "failed_pages": []
                    }
                changed_cleanup = await self.storage.clear_cache_for_pages(pages_to_analyze)
                self._merge_cleanup_stats(cache_cleanup, changed_cleanup)
                logger.info(
                    "增量分析已清理待重跑页面相关缓存: pages=%s",
                    len(pages_to_analyze)
                )

            previously_analyzed = await self.storage.list_pages()
            previously_analyzed_count = len(previously_analyzed)

            if not pages_to_analyze:
                snapshot_saved = await self.storage.save_content_snapshot(current_snapshot)
                if not snapshot_saved:
                    logger.warning("增量分析完成但快照保存失败")

                message = "无新增或修改页面，已是最新状态"
                if deleted_pages or deleted_chapter_ids:
                    message = "无新增/修改页面，已清理删除内容缓存"

                return {
                    "status": "no_changes",
                    "message": message,
                    "total_pages": total_pages,
                    "analyzed_pages": previously_analyzed_count,
                    "deleted_pages": len(deleted_pages),
                    "deleted_chapters": len(deleted_chapter_ids),
                    "cache_cleanup": cache_cleanup
                }

            logger.info(
                "增量分析: 共%s页, 变更待分析%s页, 删除页%s, 删除章节%s",
                total_pages,
                len(pages_to_analyze),
                len(deleted_pages),
                len(deleted_chapter_ids)
            )

            # 3) 获取已有批次结果作为上下文
            existing_batches = await self.storage.list_batches()
            existing_batch_results = []
            for batch in existing_batches:
                batch_data = await self.storage.load_batch_analysis(batch["start_page"], batch["end_page"])
                if batch_data and not batch_data.get("parse_error"):
                    existing_batch_results.append(batch_data)
            existing_batch_results.sort(key=lambda x: x.get("page_range", {}).get("start", 0))

            # 4) 连续段分批分析
            pages_per_batch = max(1, self.config.analysis.batch.pages_per_batch)
            context_batch_count = max(0, self.config.analysis.batch.context_batch_count)
            contiguous_batches = self._build_contiguous_batches(pages_to_analyze, pages_per_batch)

            analyzed_count = 0
            failed_pages = []
            new_batch_results = []
            total_targets = len(pages_to_analyze)

            if on_progress:
                on_progress(0, total_targets)

            for batch_pages in contiguous_batches:
                if should_stop and should_stop():
                    logger.info("增量分析已取消")
                    return {
                        "status": "cancelled",
                        "total_pages": total_pages,
                        "pages_analyzed": analyzed_count,
                        "pages_remaining": total_targets - analyzed_count - len(failed_pages),
                        "pages_failed": len(failed_pages),
                        "failed_pages": failed_pages
                    }

                batch_image_infos = []
                for page_num in batch_pages:
                    image_info = all_images[page_num - 1] if page_num <= len(all_images) else None
                    batch_image_infos.append(image_info)

                all_results = existing_batch_results + new_batch_results
                previous_results = []
                if context_batch_count > 0 and all_results:
                    current_start = batch_pages[0]
                    relevant_results = [
                        r for r in all_results
                        if r.get("page_range", {}).get("end", 0) < current_start and not r.get("parse_error")
                    ]
                    if relevant_results:
                        previous_results = relevant_results[-context_batch_count:]

                try:
                    logger.info(
                        "增量分析: 第%s-%s页 [上文%s批]",
                        batch_pages[0],
                        batch_pages[-1],
                        len(previous_results)
                    )
                    result = await analyzer.analyze_batch(
                        page_nums=batch_pages,
                        image_infos=batch_image_infos,
                        force=True,
                        previous_results=previous_results
                    )
                    if not result or result.get("parse_error"):
                        logger.warning(
                            "增量分析批次解析失败: 第%s-%s页",
                            batch_pages[0],
                            batch_pages[-1]
                        )
                        failed_pages.extend(batch_pages)
                        continue
                    new_batch_results.append(result)
                    analyzed_count += len(batch_pages)

                    if on_progress:
                        on_progress(analyzed_count, total_targets)

                except Exception as e:
                    logger.error(f"增量分析批次失败: 第{batch_pages[0]}-{batch_pages[-1]}页 - {e}")
                    failed_pages.extend(batch_pages)

            # 5) 成功完成后写入新快照
            snapshot_saved = False
            if failed_pages:
                logger.warning(
                    "增量分析存在失败页（%s页），为避免漏检，本次不更新内容快照",
                    len(failed_pages)
                )
            else:
                snapshot_saved = await self.storage.save_content_snapshot(current_snapshot)
                if not snapshot_saved:
                    logger.warning("增量分析完成但快照保存失败")

            return {
                "status": "completed",
                "total_pages": total_pages,
                "previously_analyzed": previously_analyzed_count,
                "pages_analyzed": analyzed_count,
                "pages_failed": len(failed_pages),
                "failed_pages": failed_pages,
                "pages_targeted": len(pages_to_analyze),
                "deleted_pages": len(deleted_pages),
                "deleted_chapters": len(deleted_chapter_ids),
                "cache_cleanup": cache_cleanup,
                "snapshot_saved": snapshot_saved
            }
        finally:
            try:
                await analyzer.close()
            except Exception as e:
                logger.warning(f"关闭增量分析器失败: {e}")

    def _build_contiguous_batches(self, page_nums: List[int], pages_per_batch: int) -> List[List[int]]:
        """将页码按连续段分批。"""
        normalized_pages = sorted({
            page_num for page_num in page_nums
            if isinstance(page_num, int) and not isinstance(page_num, bool) and page_num > 0
        })
        if not normalized_pages:
            return []

        contiguous_ranges: List[List[int]] = []
        current_range = [normalized_pages[0]]
        for page_num in normalized_pages[1:]:
            if page_num == current_range[-1] + 1:
                current_range.append(page_num)
            else:
                contiguous_ranges.append(current_range)
                current_range = [page_num]
        contiguous_ranges.append(current_range)

        batches: List[List[int]] = []
        for page_range in contiguous_ranges:
            for i in range(0, len(page_range), pages_per_batch):
                batches.append(page_range[i:i + pages_per_batch])
        return batches

    def _extract_deleted_changes(self, changes: List) -> tuple[List[int], List[str]]:
        """提取删除变更的页码与章节。"""
        deleted_pages = set()
        deleted_chapter_ids = set()

        for change in changes:
            if change.change_type != ChangeType.DELETED:
                continue

            if change.chapter_id:
                deleted_chapter_ids.add(change.chapter_id)

            for page_num in change.page_numbers or []:
                if isinstance(page_num, int) and page_num > 0:
                    deleted_pages.add(page_num)

        return sorted(deleted_pages), sorted(deleted_chapter_ids)

    @staticmethod
    def _merge_cleanup_stats(target: Dict[str, int], extra: Dict[str, int]) -> None:
        """合并缓存清理统计。"""
        for key, value in (extra or {}).items():
            if isinstance(value, int):
                target[key] = target.get(key, 0) + value
    
    async def _get_analyzed_pages(self) -> set:
        """
        从 batches 目录获取已分析的页面集合
        
        Returns:
            set: 已分析的页码集合
        """
        analyzed = set()
        batches = await self.storage.list_batches()
        for batch in batches:
            start = batch.get("start_page", 0)
            end = batch.get("end_page", 0)
            for page in range(start, end + 1):
                analyzed.add(page)
        return analyzed
    
    async def _update_cross_page_relations(self, changes: List):
        """
        更新跨页关联（剧情连贯性等）
        
        Args:
            changes: 变更列表
        """
        # 获取受影响的章节
        affected_chapters = [
            c.chapter_id for c in changes 
            if c.chapter_id and c.change_type != ChangeType.DELETED
        ]
        
        if not affected_chapters:
            return
        
        # 重建受影响页面的向量索引
        affected_pages = get_pages_to_analyze(changes)
        await self._rebuild_embeddings_for_pages(affected_pages)
        
        logger.info(f"更新了 {len(affected_chapters)} 个章节的跨页关联")
    
    async def _rebuild_embeddings_for_pages(self, page_nums: List[int]):
        """重建指定页面的向量嵌入"""
        from .vector_store import MangaVectorStore
        from .embedding_client import EmbeddingClient
        
        if not self.config.embedding.api_key:
            return
        
        if not page_nums:
            return
        
        vector_store = MangaVectorStore(self.book_id)
        if not vector_store.is_available():
            return
        
        embedding_client = EmbeddingClient(self.config.embedding)
        
        logger.info(f"开始重建向量嵌入: 共 {len(page_nums)} 页")
        success_count = 0
        try:
            # 删除旧向量
            await vector_store.delete_page_embeddings(page_nums)

            # 添加新向量
            for page_num in tqdm(page_nums, desc="重建向量嵌入", unit="页"):
                analysis = await self.storage.load_page_analysis(page_num)
                if not analysis:
                    continue

                summary = analysis.get("page_summary", "")
                if summary:
                    try:
                        embedding = await embedding_client.embed(summary)
                        await vector_store.add_page_embedding(
                            page_num, embedding, {
                                "page_summary": summary,
                                "type": "page"  # 添加类型标识
                            }
                        )
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"第 {page_num} 页向量化失败: {e}")
        finally:
            await embedding_client.close()

        logger.info(f"向量嵌入重建完成: 成功 {success_count}/{len(page_nums)} 页")


class ReanalyzeManager:
    """模块化重新分析管理器"""
    
    def __init__(self, book_id: str, config: MangaInsightConfig = None):
        self.book_id = book_id
        self.config = config
        
        if config is None:
            from .config_utils import load_insight_config
            self.config = load_insight_config()
    
    async def reanalyze_pages(self, page_nums: List[int]):
        """重新分析指定页面（使用批量分析模式）"""
        from .task_manager import get_task_manager
        from .task_models import TaskType
        
        normalized_pages = sorted({
            page_num for page_num in page_nums
            if isinstance(page_num, int) and not isinstance(page_num, bool) and page_num > 0
        })

        task_manager = get_task_manager()
        task = await task_manager.create_task(
            book_id=self.book_id,
            task_type=TaskType.REANALYZE,
            target_pages=normalized_pages,
            force_reanalyze=True
        )
        start_result = await task_manager.start_task(task.task_id)
        if not start_result.task_id:
            start_result.task_id = task.task_id
        return start_result
    
    async def reanalyze_chapter(self, chapter_id: str):
        """重新分析整个章节"""
        from .task_manager import get_task_manager
        from .task_models import TaskType
        
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            book_id=self.book_id,
            task_type=TaskType.CHAPTER,
            target_chapters=[chapter_id],
            force_reanalyze=True
        )
        start_result = await task_manager.start_task(task.task_id)
        if not start_result.task_id:
            start_result.task_id = task.task_id
        return start_result
    
    async def reanalyze_book(self):
        """重新分析全书"""
        from .task_manager import get_task_manager
        from .task_models import TaskType

        task_manager = get_task_manager()
        task = await task_manager.create_task(
            book_id=self.book_id,
            task_type=TaskType.FULL_BOOK,
            force_reanalyze=True
        )
        start_result = await task_manager.start_task(task.task_id)
        if not start_result.task_id:
            start_result.task_id = task.task_id
        return start_result
    
    # ============================================================
    # 四层级模式重新分析方法
    # ============================================================
    
    async def reanalyze_batch(self, start_page: int, end_page: int) -> Dict:
        """
        重新分析指定批次
        
        Args:
            start_page: 起始页码
            end_page: 结束页码
        
        Returns:
            Dict: 重新分析结果
        """
        from .analyzer import MangaAnalyzer
        
        analyzer = MangaAnalyzer(self.book_id, self.config)
        try:
            result = await analyzer.reanalyze_batch(start_page, end_page)
            logger.info(f"重新分析批次完成: 第{start_page}-{end_page}页")
            return result
        finally:
            try:
                await analyzer.close()
            except Exception as e:
                logger.warning(f"关闭重新分析批次分析器失败: {e}")
    
    async def reanalyze_segment(self, segment_id: str) -> Dict:
        """
        重新生成指定小总结
        
        Args:
            segment_id: 小总结ID
        
        Returns:
            Dict: 重新生成的小总结
        """
        from .analyzer import MangaAnalyzer
        
        analyzer = MangaAnalyzer(self.book_id, self.config)
        try:
            result = await analyzer.reanalyze_segment(segment_id)
            logger.info(f"重新生成小总结完成: {segment_id}")
            return result
        finally:
            try:
                await analyzer.close()
            except Exception as e:
                logger.warning(f"关闭小总结重分析器失败: {e}")
    
    async def reanalyze_chapter_summary(self, chapter_id: str) -> Dict:
        """
        重新生成章节总结（基于现有小总结）
        
        Args:
            chapter_id: 章节ID
        
        Returns:
            Dict: 重新生成的章节总结
        """
        from .analyzer import MangaAnalyzer
        
        storage = AnalysisStorage(self.book_id)
        analyzer = MangaAnalyzer(self.book_id, self.config)
        try:
            # 加载现有章节分析获取页面范围
            chapter_analysis = await storage.load_chapter_analysis(chapter_id)
            if not chapter_analysis:
                raise ValueError(f"未找到章节分析: {chapter_id}")
            
            page_range = chapter_analysis.get("page_range", {})
            start_page = page_range.get("start", 1)
            end_page = page_range.get("end", 1)
            
            # 获取该章节范围内的小总结
            segments = await storage.get_segments_for_chapter(chapter_id, start_page, end_page)
            
            if not segments:
                logger.warning(f"章节 {chapter_id} 没有小总结，使用动态层级模式重新分析")
                return await analyzer.analyze_chapter_with_segments(chapter_id, force=True)
            
            # 获取章节信息
            book_info = await analyzer.get_book_info()
            chapters = book_info.get("chapters", [])
            chapter_info = None
            for ch in chapters:
                if ch.get("id") == chapter_id or ch.get("chapter_id") == chapter_id:
                    chapter_info = ch
                    break
            
            if not chapter_info:
                chapter_info = {"id": chapter_id, "title": f"第{chapter_id}章"}

            # 重新生成章节总结 - 通过 _summary_generator 调用
            result = await analyzer._summary_generator.generate_chapter_from_segments(chapter_id, chapter_info, segments)
            await storage.save_chapter_analysis(chapter_id, result)
            
            logger.info(f"重新生成章节总结完成: {chapter_id}")
            return result
        finally:
            try:
                await analyzer.close()
            except Exception as e:
                logger.warning(f"关闭章节总结重分析器失败: {e}")
    
    async def list_batches(self) -> List[Dict]:
        """列出所有批量分析"""
        storage = AnalysisStorage(self.book_id)
        return await storage.list_batches()
    
    async def list_segments(self) -> List[Dict]:
        """列出所有小总结"""
        storage = AnalysisStorage(self.book_id)
        return await storage.list_segments()
    
    async def get_batch_analysis(self, start_page: int, end_page: int) -> Optional[Dict]:
        """获取指定批次的分析结果"""
        storage = AnalysisStorage(self.book_id)
        return await storage.load_batch_analysis(start_page, end_page)
    
    async def get_segment_summary(self, segment_id: str) -> Optional[Dict]:
        """获取指定小总结"""
        storage = AnalysisStorage(self.book_id)
        return await storage.load_segment_summary(segment_id)
