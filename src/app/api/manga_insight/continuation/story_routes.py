"""
续写功能 - 剧情生成路由

处理脚本生成、页面详情生成、提示词生成等请求。
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
            "reference_images": ["路径1", "路径2", ...]  // 可选，自定义参考图路径列表
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
        page_count = data.get("page_count", 15)
        reference_images = data.get("reference_images", None)

        story_gen = StoryGenerator(book_id)
        script = run_async(story_gen.generate_chapter_script(
            user_direction=direction,
            page_count=page_count,
            custom_reference_images=reference_images
        ))

        # 自动保存脚本到持久化存储
        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_script(script.to_dict()))
        logger.info(f"脚本已自动保存")

        return success_response(data={"script": script.to_dict()})

    except Exception as e:
        logger.error(f"生成脚本失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/pages/<int:page_number>', methods=['POST'])
def generate_single_page_details(book_id: str, page_number: int):
    """
    生成单页剧情细化（推荐使用，避免超时）

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
                "characters": ["角色1"],
                "description": "描述",
                "dialogues": []
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

    except Exception as e:
        logger.error(f"生成第 {page_number} 页详情失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/prompts', methods=['POST'])
def generate_image_prompts(book_id: str):
    """
    生成生图提示词（第三层）

    Request Body:
        {
            "pages": [...页面数据...]
        }

    Returns:
        {
            "success": true,
            "pages": [...带提示词的页面数据...]
        }
    """
    try:
        data = request.get_json() or {}
        pages_data = data.get("pages", [])

        if not pages_data:
            return error_response("缺少页面数据", 400)

        pages = [PageContent.from_dict(p) for p in pages_data]

        story_gen = StoryGenerator(book_id)
        updated_pages = run_async(story_gen.generate_all_image_prompts(pages))

        return success_response(data={"pages": [p.to_dict() for p in updated_pages]})

    except Exception as e:
        logger.error(f"生成提示词失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/prompts/<int:page_number>', methods=['POST'])
def generate_single_image_prompt(book_id: str, page_number: int):
    """
    生成单页生图提示词（推荐使用，避免超时）

    Request Body:
        {
            "page": {...页面数据...}
        }

    Returns:
        {
            "success": true,
            "page": {...带提示词的页面数据...}
        }
    """
    try:
        data = request.get_json() or {}
        page_data = data.get("page", {})

        if not page_data:
            return error_response("缺少页面数据", 400)

        page = PageContent.from_dict(page_data)

        story_gen = StoryGenerator(book_id)
        image_prompt = run_async(story_gen.generate_image_prompt(page))
        page.image_prompt = image_prompt

        return success_response(data={"page": page.to_dict()})

    except Exception as e:
        logger.error(f"生成第 {page_number} 页提示词失败: {e}")
        return error_response(str(e), 500)
