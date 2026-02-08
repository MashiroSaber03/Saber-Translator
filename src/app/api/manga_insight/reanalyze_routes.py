"""
Manga Insight 重新分析 API

处理单页、章节、全书的重新分析请求。
"""

import logging
from flask import request

from . import manga_insight_bp
from .async_helpers import run_async
from .response_builder import success_response, error_response, task_response
from src.core.manga_insight.incremental_analyzer import ReanalyzeManager

logger = logging.getLogger("MangaInsight.API.Reanalyze")


@manga_insight_bp.route('/<book_id>/reanalyze/page/<int:page_num>', methods=['POST'])
def reanalyze_page(book_id: str, page_num: int):
    """重新分析单页（使用批量分析模式）"""
    try:
        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_pages([page_num]))

        return task_response(task_id, "started", message=f"已开始重新分析第 {page_num} 页")

    except Exception as e:
        logger.error(f"重新分析页面失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/reanalyze/pages', methods=['POST'])
def reanalyze_pages(book_id: str):
    """
    重新分析多个页面
    
    Request Body:
        {
            "pages": [1, 2, 3, ...]
        }
    """
    try:
        data = request.json or {}
        pages = data.get("pages", [])
        
        if not pages:
            return error_response("未指定页面", 400)

        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_pages(pages))

        return task_response(task_id, "started", message=f"已开始重新分析 {len(pages)} 个页面")

    except Exception as e:
        logger.error(f"重新分析页面失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/reanalyze/chapter/<chapter_id>', methods=['POST'])
def reanalyze_chapter(book_id: str, chapter_id: str):
    """重新分析章节"""
    try:
        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_chapter(chapter_id))

        return task_response(task_id, "started", message=f"已开始重新分析章节 {chapter_id}")

    except Exception as e:
        logger.error(f"重新分析章节失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/reanalyze/book', methods=['POST'])
def reanalyze_book(book_id: str):
    """重新分析全书"""
    try:
        data = request.json or {}
        confirm = data.get("confirm", False)
        
        if not confirm:
            return error_response(
                "重新分析全书将清除现有分析结果，请确认操作",
                400,
                require_confirm=True
            )

        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_book())

        return task_response(task_id, "started", message="已开始重新分析全书")

    except Exception as e:
        logger.error(f"重新分析全书失败: {e}", exc_info=True)
        return error_response(str(e), 500)
