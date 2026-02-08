"""
Manga Insight 任务管理器

管理分析任务的创建、执行、暂停、恢复和取消。
"""

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime

from .task_models import AnalysisTask, TaskStatus, TaskType
from .config_utils import load_insight_config

logger = logging.getLogger("MangaInsight.TaskManager")


class AnalysisTaskManager:
    """
    分析任务管理器 - 单例模式

    负责管理所有分析任务的生命周期。
    具体执行逻辑委托给 TaskExecutor。
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

        # 初始化暂停事件和取消标志
        self._pause_events[task.task_id] = threading.Event()
        self._pause_events[task.task_id].set()  # 初始为非暂停状态
        self._cancel_flags[task.task_id] = False

        logger.info(f"创建任务: {task.task_id} (类型: {task_type.value}, 书籍: {book_id})")
        return task

    async def start_task(self, task_id: str) -> bool:
        """启动任务"""
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
        """暂停任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.RUNNING:
            return False

        task.status = TaskStatus.PAUSED
        self._pause_events[task_id].clear()

        logger.info(f"暂停任务: {task_id}")
        return True

    async def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.PAUSED:
            return False

        task.status = TaskStatus.RUNNING
        self._pause_events[task_id].set()

        logger.info(f"恢复任务: {task_id}")
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
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

        logger.info(f"取消任务: {task_id}")
        return True

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        return task.to_dict()

    async def get_book_tasks(self, book_id: str) -> List[Dict]:
        """获取书籍的所有任务"""
        task_ids = self.book_tasks.get(book_id, [])
        tasks = []
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task:
                tasks.append(task.to_dict())

        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks

    async def get_latest_book_task(self, book_id: str) -> Optional[Dict]:
        """获取书籍最新的任务"""
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

    def _check_pause_and_cancel(self, task_id: str) -> bool:
        """
        检查暂停和取消状态（线程安全）

        Returns:
            bool: True 表示应该继续，False 表示应该停止
        """
        # 检查取消
        if self._cancel_flags.get(task_id):
            return False

        # 等待暂停恢复
        pause_event = self._pause_events.get(task_id)
        if pause_event:
            pause_event.wait()

        # 再次检查取消
        if self._cancel_flags.get(task_id):
            return False

        return True

    async def _execute_task(self, task: AnalysisTask):
        """执行分析任务 - 委托给 TaskExecutor"""
        try:
            logger.info(f"开始执行任务: {task.task_id}")

            from .analyzer import MangaAnalyzer
            from .task_executor import TaskExecutor

            config = load_insight_config()
            analyzer = MangaAnalyzer(task.book_id, config)

            # 创建执行器并执行任务
            executor = TaskExecutor(
                check_pause_cancel_func=self._check_pause_and_cancel,
                notify_progress_func=self._notify_progress
            )
            await executor.execute(task, analyzer)

            # 检查是否被取消
            if self._cancel_flags.get(task.task_id):
                task.status = TaskStatus.CANCELLED
            else:
                # 构建时间线
                await executor.build_timeline_on_complete(task.book_id)

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
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]


# 获取任务管理器单例
def get_task_manager() -> AnalysisTaskManager:
    """获取任务管理器单例实例"""
    return AnalysisTaskManager()
