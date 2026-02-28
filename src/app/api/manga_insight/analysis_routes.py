"""
Manga Insight 分析任务 API

处理分析任务的创建、控制和状态查询。
"""

import logging
from flask import request

from . import manga_insight_bp
from .async_helpers import run_async
from .response_builder import success_response, error_response, task_response, analysis_status_response
from src.core.manga_insight.task_manager import get_task_manager
from src.core.manga_insight.task_models import TaskType
from src.core.manga_insight.storage import AnalysisStorage
from src.core.manga_insight.config_utils import load_insight_config
from src.core.manga_insight.book_pages import build_book_pages_manifest

logger = logging.getLogger("MangaInsight.API.Analysis")


def _validate_pages_input(pages, total_pages: int, field_name: str = "pages") -> list[int]:
    """校验并标准化页码列表。"""
    if total_pages <= 0:
        raise ValueError("书籍没有可分析的图片")

    if not isinstance(pages, list) or not pages:
        raise ValueError(f"{field_name} 不能为空")

    normalized_pages: list[int] = []
    for page_num in pages:
        if not isinstance(page_num, int) or isinstance(page_num, bool):
            raise ValueError(f"页码必须是整数: {page_num}")
        if page_num <= 0:
            raise ValueError(f"页码必须大于 0: {page_num}")
        if total_pages > 0 and page_num > total_pages:
            raise ValueError(f"页码越界: {page_num} (总页数 {total_pages})")
        normalized_pages.append(page_num)

    return sorted(set(normalized_pages))


def _validate_chapters_input(chapters, valid_chapter_ids: set[str]) -> list[str]:
    """校验并标准化章节列表。"""
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("chapters 不能为空")

    normalized: list[str] = []
    for chapter_id in chapters:
        if not isinstance(chapter_id, str) or not chapter_id.strip():
            raise ValueError(f"章节 ID 无效: {chapter_id}")
        if chapter_id not in valid_chapter_ids:
            raise ValueError(f"章节不存在: {chapter_id}")
        normalized.append(chapter_id)

    return list(dict.fromkeys(normalized))


@manga_insight_bp.route('/<book_id>/analyze/start', methods=['POST'])
def start_analysis(book_id: str):
    """
    启动分析任务

    Request Body:
        {
            "mode": "full" | "incremental" | "chapters" | "pages",
            "chapters": ["ch_001", ...],  // 可选
            "pages": [1, 2, 3, ...],       // 可选
            "force": false                 // 是否强制重新分析
        }
    """
    try:
        data = request.json or {}
        mode = data.get("mode", "full")
        chapters = data.get("chapters")
        pages = data.get("pages")
        force = data.get("force", False)

        manifest = build_book_pages_manifest(book_id)
        total_pages = manifest.get("total_pages", 0)
        valid_chapter_ids = {
            (chapter.get("id") or chapter.get("chapter_id"))
            for chapter in manifest.get("chapters", [])
            if (chapter.get("id") or chapter.get("chapter_id"))
        }

        task_manager = get_task_manager()
        target_chapters = None
        target_pages = None

        # 根据模式确定任务类型
        if mode == "full":
            task_type = TaskType.FULL_BOOK
        elif mode == "incremental":
            task_type = TaskType.INCREMENTAL
        elif mode == "chapters":
            task_type = TaskType.CHAPTER
            target_chapters = _validate_chapters_input(chapters, valid_chapter_ids)
        elif mode == "pages":
            task_type = TaskType.REANALYZE  # 使用批量分析模式
            target_pages = _validate_pages_input(pages, total_pages)
        else:
            return error_response(f"无效的分析模式: {mode}", 400)

        # 创建任务
        task = run_async(task_manager.create_task(
            book_id=book_id,
            task_type=task_type,
            target_chapters=target_chapters,
            target_pages=target_pages,
            is_incremental=(mode == "incremental"),
            force_reanalyze=bool(force) or mode == "pages"
        ))

        # 启动任务
        start_result = run_async(task_manager.start_task(task.task_id))
        if not start_result.success:
            return error_response(
                start_result.reason or "任务启动失败",
                start_result.status_code or 409,
                error_code=start_result.error_code or "TASK_START_REJECTED",
                task_id=start_result.task_id,
                running_task_id=start_result.running_task_id
            )

        return task_response(task.task_id, "started", message="分析任务已启动")

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"启动分析失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/analyze/pause', methods=['POST'])
def pause_analysis(book_id: str):
    """暂停分析任务"""
    try:
        data = request.json or {}
        task_id = data.get("task_id")

        if not task_id:
            # 获取最新任务
            task_manager = get_task_manager()
            latest = run_async(task_manager.get_latest_book_task(book_id))
            if latest:
                task_id = latest.get("task_id")

        if not task_id:
            return error_response("未找到运行中的任务", 404)

        task_manager = get_task_manager()
        success = run_async(task_manager.pause_task(task_id))

        if success:
            return success_response(message="任务已暂停")
        else:
            return error_response("暂停失败，任务可能不在运行中", 400)

    except Exception as e:
        logger.error(f"暂停分析失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/analyze/resume', methods=['POST'])
def resume_analysis(book_id: str):
    """恢复分析任务"""
    try:
        data = request.json or {}
        task_id = data.get("task_id")

        if not task_id:
            task_manager = get_task_manager()
            latest = run_async(task_manager.get_latest_book_task(book_id))
            if latest:
                task_id = latest.get("task_id")

        if not task_id:
            return error_response("未找到已暂停的任务", 404)

        task_manager = get_task_manager()
        success = run_async(task_manager.resume_task(task_id))

        if success:
            return success_response(message="任务已恢复")
        else:
            return error_response("恢复失败，任务可能不在暂停状态", 400)

    except Exception as e:
        logger.error(f"恢复分析失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/analyze/cancel', methods=['POST'])
def cancel_analysis(book_id: str):
    """取消分析任务"""
    try:
        data = request.json or {}
        task_id = data.get("task_id")

        if not task_id:
            task_manager = get_task_manager()
            latest = run_async(task_manager.get_latest_book_task(book_id))
            if latest:
                task_id = latest.get("task_id")

        if not task_id:
            return error_response("未找到任务", 404)

        task_manager = get_task_manager()
        success = run_async(task_manager.cancel_task(task_id))

        if success:
            return success_response(message="任务已取消")
        else:
            return error_response("取消失败", 400)

    except Exception as e:
        logger.error(f"取消分析失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/analyze/status', methods=['GET'])
def get_analysis_status(book_id: str):
    """获取分析状态"""
    try:
        task_manager = get_task_manager()

        # 获取最新任务
        latest_task = run_async(task_manager.get_latest_book_task(book_id))

        # 获取存储状态
        storage = AnalysisStorage(book_id)
        analyzed_pages = run_async(storage.list_pages())
        overview = run_async(storage.load_overview())
        manifest = build_book_pages_manifest(book_id)
        total_pages = manifest.get("total_pages", 0)

        return analysis_status_response(
            book_id=book_id,
            analyzed=len(analyzed_pages) > 0,
            analyzed_pages_count=len(analyzed_pages),
            total_pages=total_pages,
            has_overview=bool(overview),
            current_task=latest_task
        )

    except Exception as e:
        logger.error(f"获取分析状态失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/analyze/tasks', methods=['GET'])
def get_analysis_tasks(book_id: str):
    """获取书籍的所有任务"""
    try:
        task_manager = get_task_manager()
        tasks = run_async(task_manager.get_book_tasks(book_id))

        return success_response(data=tasks)

    except Exception as e:
        logger.error(f"获取任务列表失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/preview', methods=['POST'])
def preview_analysis(book_id: str):
    """预览分析效果（使用批量分析模式）"""
    try:
        manifest = build_book_pages_manifest(book_id)
        total_pages = manifest.get("total_pages", 0)
        if total_pages == 0:
            return error_response("书籍没有可分析的图片", 400)

        data = request.json or {}
        if "pages" in data:
            raw_pages = data.get("pages")
        else:
            raw_pages = list(range(1, min(total_pages, 5) + 1))
        preview_pages = raw_pages[:5] if isinstance(raw_pages, list) else raw_pages
        pages = _validate_pages_input(preview_pages, total_pages, field_name="pages")

        config = load_insight_config()

        from src.core.manga_insight.analyzer import MangaAnalyzer
        analyzer = MangaAnalyzer(book_id, config)

        # 使用批量分析
        result = run_async(analyzer.analyze_batch(
            page_nums=pages,
            force=True
        ))

        return success_response(
            data={"preview": result},
            message=f"已预览分析 {len(pages)} 页"
        )

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"预览分析失败: {e}", exc_info=True)
        return error_response(str(e), 500)
