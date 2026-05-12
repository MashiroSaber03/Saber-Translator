"""
续写功能 - 图片生成和参考图管理路由

处理图片生成、参考图获取、三视图生成等请求。
"""

import os
import re
import logging
from urllib.parse import unquote
from datetime import datetime
from flask import request, send_file

from .. import manga_insight_bp
from ..async_helpers import run_async
from ..response_builder import success_response, error_response
from .helpers import check_path_safety, get_mimetype
from src.core.manga_insight.continuation import (
    ImageGenerator,
    CharacterManager,
    PageContent,
)
from src.core.manga_insight.continuation.reference_tokens import (
    build_character_reference_candidates,
    build_continuation_reference_candidates,
    build_original_reference_candidates,
    list_original_manga_page_paths,
)
from src.core.manga_insight.storage import AnalysisStorage
from src.shared.path_helpers import resource_path

logger = logging.getLogger("MangaInsight.API.Continuation.Image")


# ==================== 参考图获取 ====================

@manga_insight_bp.route('/<book_id>/continuation/generated-image', methods=['GET'])
def get_generated_image(book_id: str):
    """
    获取生成的图片（通过路径参数）

    Query:
        path: 图片的绝对路径
    """
    try:
        image_path = request.args.get('path', '')

        if not image_path:
            return error_response("缺少图片路径参数", 400)

        # 规范化路径
        image_path = os.path.normpath(image_path)

        # 路径转换（旧路径 -> 新路径）
        if not os.path.exists(image_path):
            normalized = image_path.replace("\\", "/")
            if "data/manga_insight/" in normalized or "data\\manga_insight\\" in image_path:
                match = re.search(r'data[/\\]manga_insight[/\\]([^/\\]+)[/\\](.+)$', image_path)
                if match:
                    old_book_id = match.group(1)
                    remaining_path = match.group(2)
                    new_path = resource_path(f"data/bookshelf/{old_book_id}/insight/{remaining_path}")
                    if os.path.exists(new_path):
                        image_path = new_path
                        logger.info(f"路径转换成功: 旧路径 -> {new_path}")

        # 安全校验
        allowed_dirs = [
            resource_path(f"data/bookshelf/{book_id}/insight"),
            resource_path(f"data/manga_insight/{book_id}"),
        ]

        if not check_path_safety(image_path, allowed_dirs):
            logger.warning(f"路径安全检查失败: {image_path}")
            return error_response("无效的图片路径", 403)

        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            return error_response("图片文件不存在", 404)

        return send_file(image_path, mimetype=get_mimetype(image_path))

    except Exception as e:
        logger.error(f"获取图片失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/style-references', methods=['GET'])
def get_style_references(book_id: str):
    """
    获取最近可用的画风参考图 token

    Query:
        count: 需要的页数（默认3）

    Returns:
        {
            "success": true,
            "tokens": ["original:77", "original:78", ...]
        }
    """
    try:
        count = request.args.get('count', 3, type=int)

        image_gen = ImageGenerator(book_id)
        try:
            style_tokens = run_async(image_gen.get_recent_style_reference_tokens(count))
        finally:
            run_async(image_gen.close())

        return success_response(data={"tokens": style_tokens})

    except Exception as e:
        logger.error(f"获取画风参考图失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/available-images', methods=['GET'])
def get_available_images(book_id: str):
    """
    获取可用于参考图选择的所有图片列表

    Query:
        mode: "script" 或 "image"

    Returns:
        原作图片、续写图片、角色档案信息
    """
    try:
        mode = request.args.get('mode', 'script')

        # 获取原作图片列表
        original_pages = list_original_manga_page_paths(book_id)
        original_images = build_original_reference_candidates(original_pages)

        total_original_pages = len(original_images)

        result = {
            "original_images": original_images,
            "continuation_images": [],
            "character_forms": [],
            "total_original_pages": total_original_pages
        }

        # 如果是生图场景，获取续写图片和角色档案
        if mode == "image":
            storage = AnalysisStorage(book_id)
            pages_data = run_async(storage.load_continuation_pages())

            if pages_data and "pages" in pages_data:
                result["continuation_images"] = build_continuation_reference_candidates(
                    total_original_pages=total_original_pages,
                    pages=pages_data["pages"],
                )

            # 获取角色档案
            char_manager = CharacterManager(book_id)
            characters = char_manager.load_characters()
            result["character_forms"] = build_character_reference_candidates(characters)

        return success_response(data=result)

    except Exception as e:
        logger.error(f"获取可用图片列表失败: {e}", exc_info=True)
        return error_response(str(e), 500)


# ==================== 三视图生成 ====================

@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/orthographic', methods=['POST'])
def generate_form_orthographic(book_id: str, character_name: str, form_id: str):
    """
    生成指定形态的三视图

    接收用户上传的图片作为源图片，生成该形态的三视图。
    """
    try:
        char_name = unquote(character_name)
        form_id = unquote(form_id)

        if 'images' not in request.files:
            return error_response("请上传角色图片", 400)

        uploaded_files = request.files.getlist('images')
        if not uploaded_files or len(uploaded_files) == 0:
            return error_response("请至少上传一张图片", 400)

        # 保存上传的图片
        char_manager = CharacterManager(book_id)
        saved_paths = []

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for i, file in enumerate(uploaded_files):
            if file.filename:
                ext = os.path.splitext(file.filename)[1] or '.png'
                save_path = os.path.join(
                    char_manager.characters_dir,
                    char_name,
                    f"{char_name}_{form_id}_source_{timestamp}_{i}{ext}"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                file.save(save_path)
                saved_paths.append(save_path)
                logger.info(f"源图片已保存: {save_path}")

        if not saved_paths:
            return error_response("图片保存失败", 500)

        # 生成三视图
        image_gen = ImageGenerator(book_id)
        try:
            ortho_path = run_async(image_gen.generate_character_orthographic(
                character_name=char_name,
                source_image_paths=saved_paths,
                form_id=form_id
            ))
        finally:
            run_async(image_gen.close())

        # 清理临时源图片
        for temp_path in saved_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"已清理临时源图片: {temp_path}")
            except Exception as e:
                logger.warning(f"清理源图片失败: {e}")

        logger.info(f"形态三视图生成成功: {char_name}/{form_id} -> {ortho_path}")

        return success_response(data={"image_path": ortho_path})

    except Exception as e:
        logger.error(f"生成形态三视图失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/set-reference', methods=['POST'])
def set_form_reference(book_id: str, character_name: str, form_id: str):
    """将三视图设置为指定形态的参考图"""
    try:
        char_name = unquote(character_name)
        form_id = unquote(form_id)

        data = request.get_json() or {}
        image_path = data.get('image_path', '')

        if not image_path:
            return error_response("缺少图片路径参数", 400)

        if not os.path.exists(image_path):
            return error_response("图片文件不存在", 404)

        char_manager = CharacterManager(book_id)
        success = char_manager.update_character_reference(char_name, image_path, form_id)

        if success:
            logger.info(f"已将三视图设置为形态参考图: {char_name}/{form_id} -> {image_path}")
            return success_response()
        else:
            return error_response("更新失败", 500)

    except Exception as e:
        logger.error(f"设置形态参考图失败: {e}")
        return error_response(str(e), 500)


# ==================== 图片生成 ====================

@manga_insight_bp.route('/<book_id>/continuation/generate/<int:page_number>', methods=['POST'])
def generate_page_image(book_id: str, page_number: int):
    """
    生成单页图片

    Request Body:
        {
            "page": {...页面数据...},
            "style_reference_tokens": ["original:78", "continuation:1"],
            "session_id": "会话ID",
            "style_ref_count": 3,
            "custom_style_reference_tokens": ["original:78", "continuation:2"]
        }
    """
    try:
        data = request.get_json() or {}
        page_data = data.get("page", {})
        style_reference_tokens = data.get("style_reference_tokens", [])
        session_id = data.get("session_id", "")
        style_ref_count = data.get("style_ref_count", 3)
        custom_style_reference_tokens = data.get("custom_style_reference_tokens", None)

        if not page_data:
            return error_response("缺少页面数据", 400)

        page = PageContent.from_dict(page_data)

        # 获取角色配置
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()

        final_style_tokens = (
            custom_style_reference_tokens
            if isinstance(custom_style_reference_tokens, list) and custom_style_reference_tokens
            else style_reference_tokens
        )

        image_gen = ImageGenerator(book_id)
        try:
            if not page.final_prompt:
                page.final_prompt = image_gen.compose_final_prompt(page)
            image_path = run_async(image_gen.generate_page_image(
                page_content=page,
                characters=characters,
                style_reference_tokens=final_style_tokens,
                session_id=session_id,
                style_ref_count=style_ref_count
            ))
        finally:
            run_async(image_gen.close())

        return success_response(data={
            "image_path": image_path,
            "page": page.to_dict()
        })

    except Exception as e:
        logger.error(f"生成图片失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/regenerate/<int:page_number>', methods=['POST'])
def regenerate_page_image(book_id: str, page_number: int):
    """
    重新生成页面图片（保留上一版本）

    Request Body:
        {
            "page": {...页面数据...},
            "style_reference_tokens": ["original:78", "continuation:1"],
            "session_id": "会话ID",
            "style_ref_count": 3,
            "custom_style_reference_tokens": ["original:78", "continuation:2"]
        }
    """
    try:
        data = request.get_json() or {}
        page_data = data.get("page", {})
        style_reference_tokens = data.get("style_reference_tokens", [])
        session_id = data.get("session_id", "")
        style_ref_count = data.get("style_ref_count", 3)
        custom_style_reference_tokens = data.get("custom_style_reference_tokens", None)

        if not page_data:
            return error_response("缺少页面数据", 400)

        page = PageContent.from_dict(page_data)
        previous_path = page.image_url

        # 获取角色配置
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()

        final_style_tokens = (
            custom_style_reference_tokens
            if isinstance(custom_style_reference_tokens, list) and custom_style_reference_tokens
            else style_reference_tokens
        )

        image_gen = ImageGenerator(book_id)
        try:
            if not page.final_prompt:
                page.final_prompt = image_gen.compose_final_prompt(page)
            image_path = run_async(image_gen.regenerate_page_image(
                page_content=page,
                characters=characters,
                style_reference_tokens=final_style_tokens,
                session_id=session_id,
                style_ref_count=style_ref_count
            ))
        finally:
            run_async(image_gen.close())

        return success_response(data={
            "image_path": image_path,
            "previous_path": previous_path,
            "page": page.to_dict()
        })

    except Exception as e:
        logger.error(f"重新生成图片失败: {e}")
        return error_response(str(e), 500)
