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
        1. 获取书籍所有页面
        2. 检查已分析的页面（从 batches 目录）
        3. 只分析未完成的页面
        4. 更新跨页关联
        
        Args:
            on_progress: 进度回调函数 (analyzed_count, total_count)
            should_stop: 停止检查回调
        
        Returns:
            Dict: 分析结果摘要
        """
        from .analyzer import MangaAnalyzer
        analyzer = MangaAnalyzer(self.book_id, self.config)
        
        # 1. 获取书籍信息和所有页面
        book_info = await analyzer.get_book_info()
        all_images = book_info.get("all_images", [])
        total_pages = len(all_images)
        
        if total_pages == 0:
            return {
                "status": "no_pages",
                "message": "书籍没有可分析的图片"
            }
        
        all_page_nums = set(range(1, total_pages + 1))
        
        # 2. 获取已分析的页面（从 batches 目录读取）
        analyzed_pages = await self._get_analyzed_pages()
        
        # 3. 计算待分析页面
        pages_to_analyze = sorted(all_page_nums - analyzed_pages)
        
        if not pages_to_analyze:
            logger.info(f"增量分析: 全部 {total_pages} 页已分析完成，无需继续")
            return {
                "status": "no_changes",
                "message": f"所有 {total_pages} 页已分析完成",
                "total_pages": total_pages,
                "analyzed_pages": len(analyzed_pages)
            }
        
        logger.info(f"增量分析: 共 {total_pages} 页, 已分析 {len(analyzed_pages)} 页, 待分析 {len(pages_to_analyze)} 页")
        
        # 4. 获取已有的批量分析结果用于上下文
        existing_batches = await self.storage.list_batches()
        existing_batch_results = []
        for batch in existing_batches:
            batch_data = await self.storage.load_batch_analysis(batch["start_page"], batch["end_page"])
            if batch_data and not batch_data.get("parse_error"):
                existing_batch_results.append(batch_data)
        
        # 按页码排序已有结果
        existing_batch_results.sort(key=lambda x: x.get("page_range", {}).get("start", 0))
        
        # 5. 按批次处理待分析页面
        pages_per_batch = self.config.analysis.batch.pages_per_batch
        context_batch_count = self.config.analysis.batch.context_batch_count
        
        analyzed_count = 0
        failed_pages = []
        new_batch_results = []
        
        for i in range(0, len(pages_to_analyze), pages_per_batch):
            # 检查是否应该停止
            if should_stop and should_stop():
                logger.info("增量分析已取消")
                return {
                    "status": "cancelled",
                    "total_pages": total_pages,
                    "pages_analyzed": analyzed_count,
                    "pages_remaining": len(pages_to_analyze) - analyzed_count - len(failed_pages),
                    "pages_failed": len(failed_pages),
                    "failed_pages": failed_pages
                }
            
            batch_pages = pages_to_analyze[i:i + pages_per_batch]
            batch_image_infos = []
            for page_num in batch_pages:
                image_info = all_images[page_num - 1] if page_num <= len(all_images) else None
                batch_image_infos.append(image_info)
            
            # 构建上下文：合并已有结果和新结果
            all_results = existing_batch_results + new_batch_results
            previous_results = []
            if context_batch_count > 0 and all_results:
                # 找到当前批次之前的结果作为上下文
                current_start = batch_pages[0]
                relevant_results = [
                    r for r in all_results 
                    if r.get("page_range", {}).get("end", 0) < current_start
                ]
                previous_results = relevant_results[-context_batch_count:] if relevant_results else []
            
            try:
                logger.info(f"增量分析: 第{batch_pages[0]}-{batch_pages[-1]}页 [上文{len(previous_results)}批]")
                result = await analyzer.analyze_batch(
                    page_nums=batch_pages,
                    image_infos=batch_image_infos,
                    force=True,
                    previous_results=previous_results
                )
                new_batch_results.append(result)
                analyzed_count += len(batch_pages)
                
                if on_progress:
                    on_progress(analyzed_count, len(pages_to_analyze))
                    
            except Exception as e:
                logger.error(f"增量分析批次失败: 第{batch_pages[0]}-{batch_pages[-1]}页 - {e}")
                failed_pages.extend(batch_pages)
        
        # 注意: 向量嵌入在 _post_analysis_processing 中统一构建，这里不重复构建
        
        return {
            "status": "completed",
            "total_pages": total_pages,
            "previously_analyzed": len(analyzed_pages),
            "pages_analyzed": analyzed_count,
            "pages_failed": len(failed_pages),
            "failed_pages": failed_pages
        }
    
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
        
        # 删除旧向量
        await vector_store.delete_page_embeddings(page_nums)
        
        # 添加新向量
        success_count = 0
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
    
    async def reanalyze_pages(self, page_nums: List[int]) -> str:
        """重新分析指定页面（使用批量分析模式）"""
        from .task_manager import get_task_manager
        from .task_models import TaskType
        
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            book_id=self.book_id,
            task_type=TaskType.REANALYZE,
            target_pages=page_nums
        )
        await task_manager.start_task(task.task_id)
        return task.task_id
    
    async def reanalyze_chapter(self, chapter_id: str) -> str:
        """重新分析整个章节"""
        from .task_manager import get_task_manager
        from .task_models import TaskType
        
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            book_id=self.book_id,
            task_type=TaskType.CHAPTER,
            target_chapters=[chapter_id]
        )
        await task_manager.start_task(task.task_id)
        return task.task_id
    
    async def reanalyze_book(self) -> str:
        """重新分析全书"""
        from .task_manager import get_task_manager
        from .task_models import TaskType
        
        # 清除现有分析结果
        storage = AnalysisStorage(self.book_id)
        await storage.clear_all()
        
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            book_id=self.book_id,
            task_type=TaskType.FULL_BOOK
        )
        await task_manager.start_task(task.task_id)
        return task.task_id
    
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
        result = await analyzer.reanalyze_batch(start_page, end_page)
        
        logger.info(f"重新分析批次完成: 第{start_page}-{end_page}页")
        return result
    
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
        result = await analyzer.reanalyze_segment(segment_id)
        
        logger.info(f"重新生成小总结完成: {segment_id}")
        return result
    
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
            return await analyzer.analyze_chapter_with_segments(chapter_id)
        
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
        
        # 重新生成章节总结
        result = await analyzer._generate_chapter_from_segments(chapter_id, chapter_info, segments)
        await storage.save_chapter_analysis(chapter_id, result)
        
        logger.info(f"重新生成章节总结完成: {chapter_id}")
        return result
    
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
