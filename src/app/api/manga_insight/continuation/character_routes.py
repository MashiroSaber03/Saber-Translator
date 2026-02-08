"""
续写功能 - 角色和形态管理路由

处理角色 CRUD、形态 CRUD、参考图上传等请求。
"""

import os
import uuid
import logging
from urllib.parse import unquote
from flask import request, send_file

from .. import manga_insight_bp
from ..async_helpers import run_async
from ..response_builder import success_response, error_response
from src.core.manga_insight.continuation import (
    CharacterManager,
    CharacterProfile,
)
from src.core.manga_insight.storage import AnalysisStorage

logger = logging.getLogger("MangaInsight.API.Continuation.Character")


# ==================== 续写准备 ====================

@manga_insight_bp.route('/<book_id>/continuation/prepare', methods=['GET'])
def prepare_continuation(book_id: str):
    """
    准备续写所需的数据

    检查故事概要和时间线是否存在，如果不存在则生成。
    同时返回已保存的续写数据（脚本、页面等）。
    """
    try:
        from src.core.manga_insight.continuation import StoryGenerator

        story_gen = StoryGenerator(book_id)
        result = run_async(story_gen.prepare_continuation_data())

        storage = AnalysisStorage(book_id)
        saved_data = run_async(storage.load_continuation_all())

        return success_response(data={**result, "saved_data": saved_data})

    except Exception as e:
        logger.error(f"准备续写数据失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/save-pages', methods=['POST'])
def save_continuation_pages(book_id: str):
    """保存页面详情和提示词"""
    try:
        data = request.get_json() or {}
        pages_data = data.get("pages", [])

        if not pages_data:
            return error_response("缺少页面数据", 400)

        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_pages(pages_data))
        logger.info(f"已保存 {len(pages_data)} 页续写数据")

        return success_response()

    except Exception as e:
        logger.error(f"保存页面数据失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/save-config', methods=['POST'])
def save_continuation_config(book_id: str):
    """保存续写配置"""
    try:
        config = request.get_json() or {}

        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_config(config))

        return success_response()

    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/clear', methods=['DELETE'])
def clear_continuation_data(book_id: str):
    """清除续写数据（重新开始）"""
    try:
        storage = AnalysisStorage(book_id)
        run_async(storage.clear_continuation_data())

        return success_response(message="续写数据已清除")

    except Exception as e:
        logger.error(f"清除续写数据失败: {e}")
        return error_response(str(e), 500)


# ==================== 角色管理 ====================

@manga_insight_bp.route('/<book_id>/continuation/characters', methods=['GET'])
def get_characters(book_id: str):
    """获取角色列表（从时间线加载）"""
    try:
        char_manager = CharacterManager(book_id)
        characters = run_async(char_manager.initialize_characters())

        return success_response(data={
            "characters": [c.to_dict() for c in characters.characters]
        })

    except Exception as e:
        logger.error(f"获取角色列表失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters', methods=['POST'])
def add_character(book_id: str):
    """新增角色"""
    try:
        data = request.get_json() or {}
        name = data.get('name', '').strip()

        if not name:
            return error_response("角色名不能为空", 400)

        aliases = data.get('aliases', [])
        description = data.get('description', '')

        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()

        # 检查角色是否已存在
        for char in characters.characters:
            if char.name == name or name in char.aliases:
                return error_response(f"角色 '{name}' 已存在", 400)

        new_char = CharacterProfile(
            name=name,
            aliases=aliases if isinstance(aliases, list) else [],
            description=description
        )

        characters.characters.append(new_char)
        char_manager.save_characters(characters)

        logger.info(f"新增角色: {name}")

        return success_response(data={"character": new_char.to_dict()})

    except Exception as e:
        logger.error(f"新增角色失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>', methods=['DELETE'])
def delete_character(book_id: str, character_name: str):
    """删除角色"""
    try:
        char_name = unquote(character_name)

        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()

        found = False
        for i, char in enumerate(characters.characters):
            if char.name == char_name or char_name in char.aliases:
                # 删除所有形态的参考图
                for form in char.forms:
                    if form.reference_image and os.path.exists(form.reference_image):
                        try:
                            os.remove(form.reference_image)
                            logger.info(f"删除角色形态参考图: {form.reference_image}")
                        except Exception as e:
                            logger.warning(f"删除参考图失败: {e}")

                characters.characters.pop(i)
                found = True
                break

        if not found:
            return error_response(f"未找到角色: {char_name}", 404)

        char_manager.save_characters(characters)
        logger.info(f"删除角色: {char_name}")

        return success_response(message=f"角色 '{char_name}' 已删除")

    except Exception as e:
        logger.error(f"删除角色失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>', methods=['PUT'])
def update_character_info(book_id: str, character_name: str):
    """更新角色信息（名称、别名和enabled状态）"""
    try:
        data = request.get_json() or {}
        new_name = data.get('name', '').strip()
        new_aliases = data.get('aliases', None)
        new_enabled = data.get('enabled', None)

        char_name = unquote(character_name)

        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()

        found_char = None
        for char in characters.characters:
            if char.name == char_name or char_name in char.aliases:
                found_char = char
                break

        if not found_char:
            return error_response(f"未找到角色: {char_name}", 404)

        if new_name and new_name != found_char.name:
            found_char.name = new_name

        if new_aliases is not None:
            found_char.aliases = [a.strip() for a in new_aliases if a.strip()]

        if new_enabled is not None:
            found_char.enabled = bool(new_enabled)
            logger.info(f"角色 {found_char.name} enabled状态更新为: {found_char.enabled}")

        if char_manager.save_characters(characters):
            logger.info(f"角色信息已更新: {found_char.name}")
            return success_response(data={"character": found_char.to_dict()})
        else:
            return error_response("保存失败", 500)

    except Exception as e:
        logger.error(f"更新角色信息失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/toggle', methods=['POST'])
def toggle_character_enabled(book_id: str, character_name: str):
    """切换角色的启用状态"""
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', True)

        char_manager = CharacterManager(book_id)
        if char_manager.toggle_character_enabled(character_name, enabled):
            return success_response(data={"enabled": enabled})
        else:
            return error_response("角色不存在", 404)

    except Exception as e:
        logger.error(f"切换角色启用状态失败: {e}")
        return error_response(str(e), 500)


# ==================== 形态管理 ====================

@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms', methods=['POST'])
def add_character_form(book_id: str, character_name: str):
    """为角色添加新形态"""
    try:
        char_name = unquote(character_name)

        data = request.get_json() or {}
        form_id = data.get('form_id', '').strip()
        form_name = data.get('form_name', '').strip()
        description = data.get('description', '')

        if not form_id:
            return error_response("形态ID不能为空", 400)

        if not form_name:
            return error_response("形态名称不能为空", 400)

        char_manager = CharacterManager(book_id)
        new_form = char_manager.add_form(
            character_name=char_name,
            form_id=form_id,
            form_name=form_name,
            description=description
        )

        if new_form:
            logger.info(f"新增形态: {char_name}/{form_id}")
            return success_response(data={"form": new_form.to_dict()})
        else:
            return error_response("添加形态失败，角色不存在或形态ID已存在", 400)

    except Exception as e:
        logger.error(f"添加形态失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>', methods=['PUT'])
def update_character_form(book_id: str, character_name: str, form_id: str):
    """更新角色形态信息"""
    try:
        char_name = unquote(character_name)
        form_id = unquote(form_id)

        data = request.get_json() or {}
        form_name = data.get('form_name')
        description = data.get('description')

        char_manager = CharacterManager(book_id)
        success = char_manager.update_form(
            character_name=char_name,
            form_id=form_id,
            form_name=form_name,
            description=description
        )

        if success:
            logger.info(f"更新形态: {char_name}/{form_id}")
            return success_response()
        else:
            return error_response("更新失败，角色或形态不存在", 404)

    except Exception as e:
        logger.error(f"更新形态失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>', methods=['DELETE'])
def delete_character_form(book_id: str, character_name: str, form_id: str):
    """删除角色形态"""
    try:
        char_name = unquote(character_name)
        form_id = unquote(form_id)

        char_manager = CharacterManager(book_id)
        success = char_manager.delete_form(
            character_name=char_name,
            form_id=form_id
        )

        if success:
            logger.info(f"删除形态: {char_name}/{form_id}")
            return success_response(message=f"形态 '{form_id}' 已删除")
        else:
            return error_response("删除失败，角色或形态不存在", 404)

    except Exception as e:
        logger.error(f"删除形态失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/toggle', methods=['POST'])
def toggle_form_enabled(book_id: str, character_name: str, form_id: str):
    """切换角色形态的启用状态"""
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', True)

        char_manager = CharacterManager(book_id)
        if char_manager.toggle_form_enabled(character_name, form_id, enabled):
            return success_response(data={"enabled": enabled})
        else:
            return error_response("角色或形态不存在", 404)

    except Exception as e:
        logger.error(f"切换形态启用状态失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/image', methods=['POST'])
def upload_form_image(book_id: str, character_name: str, form_id: str):
    """为指定形态上传参考图"""
    try:
        char_name = unquote(character_name)
        form_id = unquote(form_id)

        if 'image' not in request.files:
            return error_response("未上传图片", 400)

        file = request.files['image']
        if file.filename == '':
            return error_response("未选择文件", 400)

        # 保存临时文件
        import tempfile
        temp_dir = tempfile.gettempdir()
        ext = os.path.splitext(file.filename)[1].lower() or '.png'
        temp_path = os.path.join(temp_dir, f"form_upload_{uuid.uuid4().hex}{ext}")
        file.save(temp_path)

        # 上传到指定形态
        char_manager = CharacterManager(book_id)
        saved_path = char_manager.upload_form_image(
            character_name=char_name,
            form_id=form_id,
            image_path=temp_path,
            is_temp=True
        )

        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if saved_path:
            logger.info(f"形态参考图已上传: {char_name}/{form_id}")
            return success_response(data={"image_path": saved_path})
        else:
            return error_response("上传失败，角色或形态不存在", 404)

    except Exception as e:
        logger.error(f"上传形态参考图失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/image', methods=['DELETE'])
def delete_form_image(book_id: str, character_name: str, form_id: str):
    """删除指定形态的参考图"""
    try:
        char_name = unquote(character_name)
        form_id = unquote(form_id)

        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()

        char = characters.get_character(char_name)
        if not char:
            return error_response("角色不存在", 404)

        form = char.get_form(form_id)
        if not form:
            return error_response("形态不存在", 404)

        # 删除图片文件
        if form.reference_image and os.path.exists(form.reference_image):
            try:
                os.remove(form.reference_image)
                logger.info(f"删除形态参考图: {form.reference_image}")
            except Exception as e:
                logger.warning(f"删除图片文件失败: {e}")

        form.reference_image = ""
        char_manager.save_characters(characters)

        return success_response()

    except Exception as e:
        logger.error(f"删除形态参考图失败: {e}")
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/image', methods=['GET'])
def get_character_image(book_id: str, character_name: str):
    """获取角色参考图"""
    try:
        char_manager = CharacterManager(book_id)
        characters = run_async(char_manager.initialize_characters())

        char = next((c for c in characters.characters if c.name == character_name), None)
        reference_image = char.get_any_reference_image() if char else None

        if not char or not reference_image:
            return error_response("角色图片不存在", 404)

        if not os.path.exists(reference_image):
            return error_response("图片文件不存在", 404)

        return send_file(reference_image, mimetype='image/png')

    except Exception as e:
        logger.error(f"获取角色图片失败: {e}")
        return error_response(str(e), 500)
