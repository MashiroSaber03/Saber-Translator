"""
Manga Insight 重新分析 API

处理单页、章节、全书的重新分析请求。
"""

import logging
import asyncio
from flask import request, jsonify

from . import manga_insight_bp
from src.core.manga_insight.incremental_analyzer import ReanalyzeManager

logger = logging.getLogger("MangaInsight.API.Reanalyze")


def run_async(coro):
    """在同步上下文中运行异步函数"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@manga_insight_bp.route('/<book_id>/reanalyze/page/<int:page_num>', methods=['POST'])
def reanalyze_page(book_id: str, page_num: int):
    """重新分析单页（使用批量分析模式）"""
    try:
        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_pages([page_num]))
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "message": f"已开始重新分析第 {page_num} 页"
        })
        
    except Exception as e:
        logger.error(f"重新分析页面失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
            return jsonify({
                "success": False,
                "error": "未指定页面"
            }), 400
        
        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_pages(pages))
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "message": f"已开始重新分析 {len(pages)} 个页面"
        })
        
    except Exception as e:
        logger.error(f"重新分析页面失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/reanalyze/chapter/<chapter_id>', methods=['POST'])
def reanalyze_chapter(book_id: str, chapter_id: str):
    """重新分析章节"""
    try:
        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_chapter(chapter_id))
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "message": f"已开始重新分析章节 {chapter_id}"
        })
        
    except Exception as e:
        logger.error(f"重新分析章节失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/reanalyze/book', methods=['POST'])
def reanalyze_book(book_id: str):
    """重新分析全书"""
    try:
        data = request.json or {}
        confirm = data.get("confirm", False)
        
        if not confirm:
            return jsonify({
                "success": False,
                "error": "重新分析全书将清除现有分析结果，请确认操作",
                "require_confirm": True
            }), 400
        
        manager = ReanalyzeManager(book_id)
        task_id = run_async(manager.reanalyze_book())
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "message": "已开始重新分析全书"
        })
        
    except Exception as e:
        logger.error(f"重新分析全书失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
