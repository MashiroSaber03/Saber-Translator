"""
续写功能 - 剧情生成路由

处理脚本生成与页面剧情生成请求。
"""

import logging
from flask import request

from .. import manga_insight_bp
from ..async_helpers import run_async
from ..response_builder import success_response, error_response
from src.core.manga_insight.continuation import (
    StoryGenerator,
    ChapterScript,
    PageContent,
)
from src.core.manga_insight.storage import AnalysisStorage

logger = logging.getLogger("MangaInsight.API.Continuation.Story")


@manga_insight_bp.route('/<book_id>/continuation/script', methods=['POST'])
def generate_script(book_id: str):
    """
    生成全话脚本（第一层）

    Request Body:
        {
            "direction": "用户指定的续写方向",
            "page_count": 15,
            "reference_image_count": 5,
            "reference_tokens": ["original:77", "original:78", ...]  // 可选，自定义参考图 token 列表
        }

    Returns:
        {
            "success": true,
            "script": {
                "chapter_title": "标题",
                "page_count": 15,
                "script_text": "脚本内容",
                "generated_at": "时间戳"
            }
        }
    """
    try:
        data = request.get_json() or {}
        direction = data.get("direction", "")
        page_count = max(1, int(data.get("page_count", 15) or 15))
        reference_image_count = max(1, int(data.get("reference_image_count", 5) or 5))
        reference_tokens_raw = data.get("reference_tokens", None)
        reference_tokens = reference_tokens_raw if isinstance(reference_tokens_raw, list) else None

        story_gen = StoryGenerator(book_id)
        script = run_async(story_gen.generate_chapter_script(
            user_direction=direction,
            page_count=page_count,
            custom_reference_tokens=reference_tokens,
            reference_image_count=reference_image_count,
        ))

        # 自动保存脚本到持久化存储
        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_script(script.to_dict()))
        logger.info(f"脚本已自动保存")

        return success_response(data={"script": script.to_dict()})

    except Exception as e:
        logger.error(f"生成脚本失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/save-script', methods=['POST'])
def save_script(book_id: str):
    """保存编辑后的续写脚本"""
    try:
        data = request.get_json() or {}
        script_data = data.get("script", {})

        if not script_data or not isinstance(script_data, dict):
            return error_response("缺少脚本数据", 400)

        script = ChapterScript.from_dict(script_data)
        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_script(script.to_dict()))

        return success_response(data={"script": script.to_dict()})
    except Exception as e:
        logger.error(f"保存脚本失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/pages/<int:page_number>', methods=['POST'])
def generate_single_page_details(book_id: str, page_number: int):
    """
    生成单页页面剧情（推荐使用，避免超时）

    Request Body:
        {
            "script": {
                "chapter_title": "...",
                "page_count": 15,
                "script_text": "..."
            }
        }

    Returns:
        {
            "success": true,
            "page": {
                "page_number": 1,
                "continuity_text": "...",
                "story_text": "...",
                "dialogue_text": "...",
                "characters": ["角色1"],
                "character_forms": []
            }
        }
    """
    try:
        data = request.get_json() or {}
        script_data = data.get("script", {})

        if not script_data:
            return error_response("缺少脚本数据", 400)

        script = ChapterScript.from_dict(script_data)

        if page_number < 1 or page_number > script.page_count:
            return error_response(f"页码 {page_number} 超出范围 (1-{script.page_count})", 400)

        story_gen = StoryGenerator(book_id)
        page = run_async(story_gen.generate_page_details(script, page_number))

        return success_response(data={"page": page.to_dict()})

    except ValueError as e:
        logger.warning(f"生成第 {page_number} 页页面剧情失败，模型返回无效结果: {e}")
        return error_response(str(e), 502, error_code="INVALID_PAGE_DETAILS_RESPONSE")
    except Exception as e:
        logger.error(f"生成第 {page_number} 页页面剧情失败: {e}")
        return error_response(str(e), 500)
