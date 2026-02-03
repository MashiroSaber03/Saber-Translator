"""
Manga Insight 任务管理器

管理分析任务的创建、执行、暂停、恢复和取消。
"""

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime

from .task_models import AnalysisTask, TaskStatus, TaskType, AnalysisProgress
from .config_utils import load_insight_config

logger = logging.getLogger("MangaInsight.TaskManager")


class AnalysisTaskManager:
    """
    分析任务管理器 - 单例模式
    
    负责管理所有分析任务的生命周期。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.tasks: Dict[str, AnalysisTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.book_tasks: Dict[str, List[str]] = {}  # book_id -> [task_ids]
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        self._pause_events: Dict[str, threading.Event] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._initialized = True
        
        logger.info("任务管理器已初始化")
    
    async def create_task(
        self,
        book_id: str,
        task_type: TaskType,
        target_chapters: Optional[List[str]] = None,
        target_pages: Optional[List[int]] = None,
        is_incremental: bool = False
    ) -> AnalysisTask:
        """
        创建分析任务
        
        Args:
            book_id: 书籍ID
            task_type: 任务类型
            target_chapters: 目标章节列表
            target_pages: 目标页面列表
            is_incremental: 是否为增量分析
        
        Returns:
            AnalysisTask: 创建的任务
        """
        task = AnalysisTask(
            book_id=book_id,
            task_type=task_type,
            target_chapters=target_chapters,
            target_pages=target_pages,
            is_incremental=is_incremental
        )
        
        self.tasks[task.task_id] = task
        
        # 记录书籍关联的任务
        if book_id not in self.book_tasks:
            self.book_tasks[book_id] = []
        self.book_tasks[book_id].append(task.task_id)
        
        # 初始化暂停事件和取消标志（使用线程安全的 Event）
        self._pause_events[task.task_id] = threading.Event()
        self._pause_events[task.task_id].set()  # 初始为非暂停状态
        self._cancel_flags[task.task_id] = False
        
        logger.info(f"创建任务: {task.task_id} (类型: {task_type.value}, 书籍: {book_id})")
        return task
    
    async def start_task(self, task_id: str) -> bool:
        """
        启动任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 是否成功启动
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return False
        
        if task.status == TaskStatus.RUNNING:
            logger.warning(f"任务已在运行中: {task_id}")
            return False
        
        # 检查该书籍是否已有运行中的任务
        book_id = task.book_id
        for tid in self.book_tasks.get(book_id, []):
            other_task = self.tasks.get(tid)
            if other_task and other_task.task_id != task_id and other_task.status == TaskStatus.RUNNING:
                logger.warning(f"书籍 {book_id} 已有运行中的任务: {tid}")
                return False
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # 在后台线程中执行异步任务
        import threading
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._execute_task(task))
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        self.running_tasks[task_id] = thread
        
        logger.info(f"启动任务: {task_id}")
        return True
    
    async def pause_task(self, task_id: str) -> bool:
        """
        暂停任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 是否成功暂停
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status != TaskStatus.RUNNING:
            return False
        
        task.status = TaskStatus.PAUSED
        self._pause_events[task_id].clear()  # 触发暂停
        
        logger.info(f"暂停任务: {task_id}")
        return True
    
    async def resume_task(self, task_id: str) -> bool:
        """
        恢复任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 是否成功恢复
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status != TaskStatus.PAUSED:
            return False
        
        task.status = TaskStatus.RUNNING
        self._pause_events[task_id].set()  # 解除暂停
        
        logger.info(f"恢复任务: {task_id}")
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 是否成功取消
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        self._cancel_flags[task_id] = True
        
        # 如果任务在暂停中，先恢复它让它检测到取消标志
        if task_id in self._pause_events:
            self._pause_events[task_id].set()
        
        # 线程会通过 _cancel_flags 自动停止
        # 不需要显式取消线程
        
        logger.info(f"取消任务: {task_id}")
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            Dict: 任务状态信息
        """
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return task.to_dict()
    
    async def get_book_tasks(self, book_id: str) -> List[Dict]:
        """
        获取书籍的所有任务
        
        Args:
            book_id: 书籍ID
        
        Returns:
            List[Dict]: 任务列表
        """
        task_ids = self.book_tasks.get(book_id, [])
        tasks = []
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task:
                tasks.append(task.to_dict())
        
        # 按创建时间倒序排列
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks
    
    async def get_latest_book_task(self, book_id: str) -> Optional[Dict]:
        """
        获取书籍最新的任务
        
        Args:
            book_id: 书籍ID
        
        Returns:
            Dict: 最新任务信息
        """
        tasks = await self.get_book_tasks(book_id)
        return tasks[0] if tasks else None
    
    def register_progress_callback(self, task_id: str, callback: Callable):
        """注册进度回调"""
        if task_id not in self.progress_callbacks:
            self.progress_callbacks[task_id] = []
        self.progress_callbacks[task_id].append(callback)
    
    def _notify_progress(self, task_id: str, progress: Dict):
        """通知进度更新"""
        callbacks = self.progress_callbacks.get(task_id, [])
        for callback in callbacks:
            try:
                callback(task_id, progress)
            except Exception as e:
                logger.error(f"进度回调执行失败: {e}")
    
    async def _execute_task(self, task: AnalysisTask):
        """
        执行分析任务
        
        Args:
            task: 任务对象
        """
        try:
            logger.info(f"开始执行任务: {task.task_id}")
            
            # 延迟导入以避免循环引用
            from .analyzer import MangaAnalyzer
            
            config = load_insight_config()
            analyzer = MangaAnalyzer(task.book_id, config)
            
            # 根据任务类型执行不同的分析
            if task.task_type == TaskType.FULL_BOOK:
                await self._execute_full_book_analysis(task, analyzer)
            elif task.task_type == TaskType.CHAPTER:
                await self._execute_chapter_analysis(task, analyzer)
            elif task.task_type == TaskType.INCREMENTAL:
                await self._execute_incremental_analysis(task, analyzer)
            elif task.task_type == TaskType.REANALYZE:
                await self._execute_reanalysis(task, analyzer)
            
            # 检查是否被取消
            if self._cancel_flags.get(task.task_id):
                task.status = TaskStatus.CANCELLED
            else:
                # 分析完成后自动构建时间线（在设置完成状态之前，确保前端感知到时数据已就绪）
                await self._build_timeline_on_complete(task.book_id)
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
            
            logger.info(f"任务完成: {task.task_id}, 状态: {task.status.value}")
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            logger.info(f"任务被取消: {task.task_id}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            logger.error(f"任务执行失败: {task.task_id} - {e}", exc_info=True)
        finally:
            # 清理
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    async def _build_timeline_on_complete(self, book_id: str):
        """
        分析完成后自动构建并保存时间线（增强模式）
        
        Args:
            book_id: 书籍 ID
        """
        try:
            from .features.timeline_enhanced import EnhancedTimelineBuilder
            from .storage import AnalysisStorage
            from .config_utils import load_insight_config
            
            logger.info(f"开始构建增强时间线: {book_id}")
            
            config = load_insight_config()
            builder = EnhancedTimelineBuilder(book_id, config)
            storage = AnalysisStorage(book_id)
            
            # 构建增强时间线（会自动降级到简单模式如果 LLM 不可用）
            timeline_data = await builder.build(mode="enhanced")
            
            # 保存到缓存
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
            # 时间线构建失败不影响主任务状态
    
    def _check_pause_and_cancel(self, task_id: str) -> bool:
        """
        检查暂停和取消状态（线程安全）
        
        Returns:
            bool: True 表示应该继续，False 表示应该停止
        """
        # 检查取消
        if self._cancel_flags.get(task_id):
            return False
        
        # 等待暂停恢复（使用线程安全的方式）
        pause_event = self._pause_events.get(task_id)
        if pause_event:
            pause_event.wait()  # 线程安全，阻塞直到事件被设置
        
        # 再次检查取消（可能在暂停期间被取消）
        if self._cancel_flags.get(task_id):
            return False
        
        return True
    
    async def _execute_full_book_analysis(self, task: AnalysisTask, analyzer):
        """执行全书分析（支持四层级批量模式）"""
        # 检查 VLM 是否已配置
        if not analyzer.vlm.is_configured():
            raise ValueError("VLM 未配置，请先在设置中配置 VLM 服务商和 API Key")
        
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
        
        await self._execute_full_book_batch_analysis(task, analyzer, book_info)
    
    async def _execute_full_book_batch_analysis(self, task: AnalysisTask, analyzer, book_info: dict):
        """执行全书动态层级批量分析"""
        config = load_insight_config()
        batch_settings = config.analysis.batch
        pages_per_batch = batch_settings.pages_per_batch
        context_batch_count = batch_settings.context_batch_count
        
        # 获取层级配置
        layers = batch_settings.get_layers()
        layer_names = [l["name"] for l in layers]
        
        all_images = book_info.get("all_images", [])
        total_pages = len(all_images)
        chapters = book_info.get("chapters", [])
        
        # 构建章节页面映射
        chapter_page_map = self._build_chapter_page_map(all_images, chapters)
        
        logger.info(f"开始动态层级分析: {total_pages}页, {len(layers)}层架构: {' → '.join(layer_names)}")
        logger.info(f"每批{pages_per_batch}页, 上文参考{context_batch_count}批")
        
        # ========== 第一层: 批量分析（永远是第一层）==========
        first_layer = layers[0] if layers else {"name": "批量分析", "units_per_group": 5, "align_to_chapter": False}
        align_to_chapter = first_layer.get("align_to_chapter", False)
        
        task.progress.current_phase = "batch_analysis"
        batch_results = await self._execute_batch_analysis_layer(
            task, analyzer, all_images, pages_per_batch, context_batch_count,
            align_to_chapter, chapter_page_map
        )
        
        if not batch_results:
            logger.warning("批量分析无结果，跳过后续层级")
            await self._post_analysis_processing(task, analyzer)
            return
        
        # ========== 中间层: 汇总层级（第2层到倒数第2层）==========
        current_results = batch_results
        
        for layer_idx in range(1, len(layers) - 1):
            if not self._check_pause_and_cancel(task.task_id):
                return
            
            layer = layers[layer_idx]
            layer_name = layer.get("name", f"层级{layer_idx}")
            units_per_group = layer.get("units_per_group", 5)
            align_to_chapter = layer.get("align_to_chapter", False)
            
            task.progress.current_phase = layer_name
            logger.info(f"开始 {layer_name}...")
            
            if align_to_chapter and units_per_group == 0:
                # 按章节汇总
                current_results = await self._execute_chapter_summary_layer(
                    task, analyzer, current_results, all_images, chapters, layer_name
                )
            else:
                # 按数量分组汇总
                current_results = await self._execute_summary_layer(
                    task, analyzer, current_results, units_per_group, layer_name, layer_idx
                )
        
        # ========== 最后一层: 全书总结 ==========
        if len(layers) > 1:
            last_layer = layers[-1]
            task.progress.current_phase = last_layer.get("name", "全书总结")
            logger.info(f"开始 {last_layer.get('name', '全书总结')}...")
            # 全书总结在 _post_analysis_processing 中生成
        
        # 后续处理（嵌入构建、全书概览）
        await self._post_analysis_processing(task, analyzer)
    
    def _build_chapter_page_map(self, all_images: List[Dict], chapters: List[Dict]) -> Dict[str, List[int]]:
        """构建章节到页码的映射"""
        chapter_page_map = {}
        for idx, img in enumerate(all_images):
            ch_id = img.get("chapter_id")
            if ch_id:
                if ch_id not in chapter_page_map:
                    chapter_page_map[ch_id] = []
                chapter_page_map[ch_id].append(idx + 1)  # 页码从1开始
        return chapter_page_map
    
    async def _execute_batch_analysis_layer(
        self, task: AnalysisTask, analyzer, all_images: List[Dict],
        pages_per_batch: int, context_batch_count: int,
        align_to_chapter: bool, chapter_page_map: Dict[str, List[int]]
    ) -> List[Dict]:
        """执行批量分析层（第一层）"""
        total_pages = len(all_images)
        batch_results = []
        
        if align_to_chapter and chapter_page_map:
            # 按章节边界分批
            batch_groups = []
            for ch_id, page_nums in chapter_page_map.items():
                # 每个章节内部再按 pages_per_batch 分批
                for i in range(0, len(page_nums), pages_per_batch):
                    batch_pages = page_nums[i:i + pages_per_batch]
                    batch_groups.append((ch_id, batch_pages))
            
            total_batches = len(batch_groups)
            for batch_idx, (ch_id, page_nums) in enumerate(batch_groups):
                if not self._check_pause_and_cancel(task.task_id):
                    return batch_results
                
                result = await self._analyze_single_batch(
                    task, analyzer, all_images, page_nums, batch_idx, total_batches,
                    batch_results, context_batch_count
                )
                if result:
                    result["chapter_id"] = ch_id
                    batch_results.append(result)
        else:
            # 不考虑章节边界，按页数分批
            total_batches = (total_pages + pages_per_batch - 1) // pages_per_batch
            
            for batch_idx in range(total_batches):
                if not self._check_pause_and_cancel(task.task_id):
                    return batch_results
                
                start_idx = batch_idx * pages_per_batch
                end_idx = min(start_idx + pages_per_batch, total_pages)
                page_nums = [i + 1 for i in range(start_idx, end_idx)]
                
                result = await self._analyze_single_batch(
                    task, analyzer, all_images, page_nums, batch_idx, total_batches,
                    batch_results, context_batch_count
                )
                if result:
                    batch_results.append(result)
        
        return batch_results
    
    async def _analyze_single_batch(
        self, task: AnalysisTask, analyzer, all_images: List[Dict],
        page_nums: List[int], batch_idx: int, total_batches: int,
        batch_results: List[Dict], context_batch_count: int
    ) -> Optional[Dict]:
        """分析单个批次"""
        image_infos = [all_images[p - 1] for p in page_nums]
        task.progress.current_page = page_nums[0]
        
        # 获取上下文
        previous_results = []
        if context_batch_count > 0:
            valid_results = [r for r in batch_results if not r.get("parse_error")]
            previous_results = valid_results[-context_batch_count:]
        
        try:
            logger.info(f"批量分析: 第{page_nums[0]}-{page_nums[-1]}页 ({batch_idx+1}/{total_batches}) [上文{len(previous_results)}批]")
            result = await analyzer.analyze_batch(
                page_nums, image_infos=image_infos, force=True, previous_results=previous_results
            )
            task.progress.analyzed_pages = page_nums[-1]
            self._notify_progress(task.task_id, task.progress.to_dict())
            return result
        except Exception as e:
            logger.error(f"批量分析失败: 第{page_nums[0]}-{page_nums[-1]}页 - {e}")
            task.failed_pages.extend(page_nums)
            task.progress.analyzed_pages = page_nums[-1]
            self._notify_progress(task.task_id, task.progress.to_dict())
            return None
    
    async def _execute_summary_layer(
        self, task: AnalysisTask, analyzer, input_results: List[Dict],
        units_per_group: int, layer_name: str, layer_idx: int
    ) -> List[Dict]:
        """执行汇总层（按数量分组）"""
        if not input_results:
            logger.warning(f"{layer_name}: 输入结果为空，跳过")
            return []
        
        if units_per_group <= 0:
            units_per_group = len(input_results)  # 0表示全部汇总为一个
        
        total_groups = (len(input_results) + units_per_group - 1) // units_per_group
        summary_results = []
        
        for group_idx in range(total_groups):
            if not self._check_pause_and_cancel(task.task_id):
                return summary_results
            
            start = group_idx * units_per_group
            end = min(start + units_per_group, len(input_results))
            group_items = input_results[start:end]
            
            segment_id = f"layer{layer_idx}_seg_{group_idx:03d}"
            
            try:
                logger.info(f"生成{layer_name}: {segment_id} ({group_idx+1}/{total_groups})")
                result = await analyzer.generate_segment_summary(segment_id, group_items, force=True)
                summary_results.append(result)
            except Exception as e:
                logger.error(f"{layer_name}生成失败: {segment_id} - {e}")
            
            self._notify_progress(task.task_id, task.progress.to_dict())
        
        return summary_results
    
    async def _execute_chapter_summary_layer(
        self, task: AnalysisTask, analyzer, batch_results: List[Dict],
        all_images: List[Dict], chapters: List[Dict], layer_name: str
    ) -> List[Dict]:
        """执行章节汇总层（按章节边界分组）"""
        chapter_summaries = []
        
        for ch in chapters:
            if not self._check_pause_and_cancel(task.task_id):
                return chapter_summaries
            
            ch_id = ch.get("id") or ch.get("chapter_id")
            if not ch_id:
                continue
            
            try:
                logger.info(f"生成{layer_name}: {ch_id}")
                result = await analyzer.generate_chapter_summary_from_batches(
                    ch_id, batch_results, all_images
                )
                if result:
                    chapter_summaries.append(result)
            except Exception as e:
                logger.error(f"{layer_name}生成失败: {ch_id} - {e}", exc_info=True)
            
            self._notify_progress(task.task_id, task.progress.to_dict())
        
        return chapter_summaries
    
    async def _execute_chapter_analysis(self, task: AnalysisTask, analyzer):
        """执行章节分析（使用动态层级模式）"""
        # 检查 VLM 是否已配置
        if not analyzer.vlm.is_configured():
            raise ValueError("VLM 未配置，请先在设置中配置 VLM 服务商和 API Key")
        
        chapters = task.target_chapters or []
        task.progress.total_pages = len(chapters)
        
        for i, chapter_id in enumerate(chapters):
            if not self._check_pause_and_cancel(task.task_id):
                return
            
            task.progress.current_phase = f"分析章节: {chapter_id}"
            
            try:
                # 使用动态层级模式
                def progress_cb(phase, current, total, message):
                    task.progress.current_phase = f"{chapter_id}: {message}"
                    self._notify_progress(task.task_id, task.progress.to_dict())
                
                await analyzer.analyze_chapter_with_segments(chapter_id, progress_callback=progress_cb)
                
                task.progress.analyzed_pages = i + 1
                logger.info(f"完成章节分析: {chapter_id}")
            except Exception as e:
                logger.error(f"章节分析失败: {chapter_id} - {e}")
            
            self._notify_progress(task.task_id, task.progress.to_dict())
        
        # 后续处理（重建向量、概述等）
        await self._post_analysis_processing(task, analyzer)
    
    async def _execute_incremental_analysis(self, task: AnalysisTask, analyzer):
        """执行增量分析"""
        from .incremental_analyzer import IncrementalAnalyzer
        
        incremental = IncrementalAnalyzer(task.book_id, load_insight_config())
        
        # 定义停止检查回调
        def should_stop():
            return not self._check_pause_and_cancel(task.task_id)
        
        result = await incremental.analyze_new_content(
            on_progress=lambda analyzed, total: self._update_progress(task, analyzed, total),
            should_stop=should_stop
        )
        
        # 如果被取消，直接返回
        if result.get("status") == "cancelled":
            logger.info(f"增量分析已取消: {task.task_id}")
            return
        
        # 如果书籍没有图片，直接返回
        if result.get("status") == "no_pages":
            logger.warning(f"增量分析: {result.get('message', '没有可分析的图片')}")
            return
        
        # 如果没有需要分析的页面（全部已完成）
        if result.get("status") == "no_changes":
            logger.info(f"增量分析: {result.get('message', '无需分析')}")
            # 仍然执行后续处理（生成概述等）
        
        # 记录分析结果
        pages_analyzed = result.get("pages_analyzed", 0)
        previously_analyzed = result.get("previously_analyzed", 0)
        total_pages = result.get("total_pages", 0)
        logger.info(f"增量分析完成: 本次分析 {pages_analyzed} 页, 之前已分析 {previously_analyzed} 页, 共 {total_pages} 页")
        
        # 后续处理（与全书分析相同）
        await self._post_analysis_processing(task, analyzer)
    
    async def _execute_reanalysis(self, task: AnalysisTask, analyzer):
        """执行重新分析（使用批量分析模式）"""
        pages = task.target_pages or []
        if not pages:
            return
        
        # 排序页码
        pages = sorted(pages)
        task.progress.total_pages = len(pages)
        
        # 获取书籍信息和配置
        book_info = await analyzer.get_book_info()
        all_images = book_info.get("all_images", [])
        batch_settings = analyzer.config.analysis.batch
        pages_per_batch = batch_settings.pages_per_batch
        
        # 按批次处理
        batch_idx = 0
        total_batches = (len(pages) + pages_per_batch - 1) // pages_per_batch
        
        for i in range(0, len(pages), pages_per_batch):
            if not self._check_pause_and_cancel(task.task_id):
                return
            
            batch_pages = pages[i:i + pages_per_batch]
            batch_image_infos = []
            
            for page_num in batch_pages:
                image_info = all_images[page_num - 1] if page_num <= len(all_images) else None
                batch_image_infos.append(image_info)
            
            task.progress.current_page = batch_pages[0]
            
            try:
                await analyzer.analyze_batch(
                    page_nums=batch_pages,
                    image_infos=batch_image_infos,
                    force=True
                )
                task.progress.analyzed_pages = min(i + pages_per_batch, len(pages))
                logger.info(f"完成重新分析批次 {batch_idx + 1}/{total_batches}: 第{batch_pages[0]}-{batch_pages[-1]}页")
            except Exception as e:
                logger.error(f"重新分析批次失败: 第{batch_pages[0]}-{batch_pages[-1]}页 - {e}")
                task.failed_pages.extend(batch_pages)
            
            batch_idx += 1
            self._notify_progress(task.task_id, task.progress.to_dict())
        
        # 后续处理（重建向量、概述等）
        await self._post_analysis_processing(task, analyzer)
    
    def _update_progress(self, task: AnalysisTask, analyzed: int, total: int):
        """更新任务进度"""
        task.progress.analyzed_pages = analyzed
        task.progress.total_pages = total
        self._notify_progress(task.task_id, task.progress.to_dict())
    
    async def _post_analysis_processing(self, task: AnalysisTask, analyzer):
        """分析完成后的后续处理（嵌入、概述等）"""
        # 生成向量嵌入
        logger.info("开始构建向量嵌入...")
        task.progress.current_phase = "embedding"
        try:
            await analyzer.build_embeddings()
            logger.info("向量嵌入完成")
        except Exception as e:
            logger.error(f"向量嵌入失败: {e}", exc_info=True)
        
        # 生成概述
        logger.info("开始生成概述...")
        task.progress.current_phase = "overview"
        try:
            await analyzer.generate_overview()
            logger.info("概述生成完成")
        except Exception as e:
            logger.error(f"概述生成失败: {e}", exc_info=True)


# 获取任务管理器单例
def get_task_manager() -> AnalysisTaskManager:
    """获取任务管理器单例实例"""
    return AnalysisTaskManager()
