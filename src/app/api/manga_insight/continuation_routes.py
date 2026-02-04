"""
Manga Insight 续写功能 API

处理漫画续写相关的请求。
"""

import logging
import asyncio
import os
import uuid
from flask import request, jsonify, send_file

from . import manga_insight_bp
from src.core.manga_insight.continuation import (
    StoryGenerator,
    ImageGenerator,
    CharacterManager,
    CharacterProfile,
    ChapterScript,
    PageContent
)
from src.core.manga_insight.storage import AnalysisStorage
from src.core.manga_insight.config_utils import load_insight_config

logger = logging.getLogger("MangaInsight.API.Continuation")


# ==================== 通用文件服务 ====================

@manga_insight_bp.route('/file', methods=['GET'])
def serve_file():
    """
    通用文件服务接口
    
    Query:
        path: 文件的绝对路径
    """
    try:
        file_path = request.args.get('path', '')
        
        if not file_path:
            return jsonify({
                "success": False,
                "error": "缺少文件路径参数"
            }), 400
        
        # 规范化路径（处理混合斜杠问题）
        file_path = os.path.normpath(file_path)
        
        if not os.path.exists(file_path):
            logger.warning(f"请求的文件不存在: {file_path}")
            return jsonify({
                "success": False,
                "error": f"文件不存在: {file_path}"
            }), 404
        
        ext = os.path.splitext(file_path)[1].lower()
        mimetype_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
        }
        mimetype = mimetype_map.get(ext, 'application/octet-stream')
        
        return send_file(file_path, mimetype=mimetype)
        
    except Exception as e:
        logger.error(f"获取文件失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def run_async(coro):
    """在同步上下文中运行异步函数
    
    使用 asyncio.run() 确保每次调用都在干净的事件循环中执行，
    避免长时间运行的任务导致 'Task was destroyed but it is pending' 错误
    """
    import asyncio
    
    # Python 3.7+ 推荐使用 asyncio.run()，它会创建新的事件循环并正确清理
    try:
        # 检查是否已有运行中的事件循环
        try:
            loop = asyncio.get_running_loop()
            # 如果在已有事件循环中，使用 nest_asyncio 或创建新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # 没有运行中的事件循环，可以直接使用 asyncio.run()
            return asyncio.run(coro)
    except Exception as e:
        logger.error(f"异步执行失败: {e}")
        raise


# ============================================================
# 续写准备和角色管理
# ============================================================

@manga_insight_bp.route('/<book_id>/continuation/prepare', methods=['GET'])
def prepare_continuation(book_id: str):
    """
    准备续写所需的数据
    
    检查故事概要和时间线是否存在，如果不存在则生成。
    同时返回已保存的续写数据（脚本、页面等）。
    
    Returns:
        {
            "success": true,
            "ready": true/false,
            "generating": true/false,
            "message": "状态消息",
            "saved_data": {
                "script": {...},
                "pages": [...],
                "config": {...},
                "has_data": true/false
            }
        }
    """
    try:
        story_gen = StoryGenerator(book_id)
        result = run_async(story_gen.prepare_continuation_data())
        
        # 加载已保存的续写数据
        storage = AnalysisStorage(book_id)
        saved_data = run_async(storage.load_continuation_all())
        
        return jsonify({
            "success": True,
            **result,
            "saved_data": saved_data
        })
        
    except Exception as e:
        logger.error(f"准备续写数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/save-pages', methods=['POST'])
def save_continuation_pages(book_id: str):
    """
    保存页面详情和提示词
    
    Request Body:
        {
            "pages": [...页面数据...]
        }
        
    Returns:
        {
            "success": true
        }
    """
    try:
        data = request.get_json() or {}
        pages_data = data.get("pages", [])
        
        if not pages_data:
            return jsonify({
                "success": False,
                "error": "缺少页面数据"
            }), 400
        
        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_pages(pages_data))
        logger.info(f"已保存 {len(pages_data)} 页续写数据")
        
        return jsonify({
            "success": True
        })
        
    except Exception as e:
        logger.error(f"保存页面数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/save-config', methods=['POST'])
def save_continuation_config(book_id: str):
    """
    保存续写配置
    
    Request Body:
        {
            "page_count": 15,
            "style_reference_pages": 3,
            "continuation_direction": "..."
        }
    """
    try:
        config = request.get_json() or {}
        
        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_config(config))
        
        return jsonify({
            "success": True
        })
        
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/clear', methods=['DELETE'])
def clear_continuation_data(book_id: str):
    """
    清除续写数据（重新开始）
    """
    try:
        storage = AnalysisStorage(book_id)
        run_async(storage.clear_continuation_data())
        
        return jsonify({
            "success": True,
            "message": "续写数据已清除"
        })
        
    except Exception as e:
        logger.error(f"清除续写数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters', methods=['GET'])
def get_characters(book_id: str):
    """
    获取角色列表（从时间线加载）
    
    Returns:
        {
            "success": true,
            "characters": [
                {
                    "name": "角色名",
                    "aliases": ["别名1"],
                    "description": "描述",
                    "reference_image": "图片路径或空"
                }
            ]
        }
    """
    try:
        char_manager = CharacterManager(book_id)
        characters = run_async(char_manager.initialize_characters())
        
        return jsonify({
            "success": True,
            "characters": [c.to_dict() for c in characters.characters]
        })
        
    except Exception as e:
        logger.error(f"获取角色列表失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters', methods=['POST'])
def add_character(book_id: str):
    """
    新增角色
    
    Body:
        name: 角色名（必填）
        aliases: 别名列表（可选）
        description: 角色描述（可选）
        
    Returns:
        {
            "success": true,
            "character": { ... }
        }
    """
    try:
        data = request.get_json() or {}
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({
                "success": False,
                "error": "角色名不能为空"
            }), 400
        
        aliases = data.get('aliases', [])
        description = data.get('description', '')
        
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()
        
        # 检查角色是否已存在
        for char in characters.characters:
            if char.name == name or name in char.aliases:
                return jsonify({
                    "success": False,
                    "error": f"角色 '{name}' 已存在"
                }), 400
        
        # 创建新角色（forms 默认为空列表，用户可后续添加形态）
        new_char = CharacterProfile(
            name=name,
            aliases=aliases if isinstance(aliases, list) else [],
            description=description
        )
        
        characters.characters.append(new_char)
        char_manager.save_characters(characters)
        
        logger.info(f"新增角色: {name}")
        
        return jsonify({
            "success": True,
            "character": new_char.to_dict()
        })
        
    except Exception as e:
        logger.error(f"新增角色失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>', methods=['DELETE'])
def delete_character(book_id: str, character_name: str):
    """
    删除角色
    
    Returns:
        {
            "success": true,
            "message": "角色已删除"
        }
    """
    try:
        from urllib.parse import unquote
        char_name = unquote(character_name)
        
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()
        
        # 查找并删除角色
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
                
                # 从列表中移除
                characters.characters.pop(i)
                found = True
                break
        
        if not found:
            return jsonify({
                "success": False,
                "error": f"未找到角色: {char_name}"
            }), 404
        
        char_manager.save_characters(characters)
        
        logger.info(f"删除角色: {char_name}")
        
        return jsonify({
            "success": True,
            "message": f"角色 '{char_name}' 已删除"
        })
        
    except Exception as e:
        logger.error(f"删除角色失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>', methods=['PUT'])
def update_character_info(book_id: str, character_name: str):
    """
    更新角色信息（名称和别名）
    
    Body:
        name: 新的角色名（可选）
        aliases: 别名列表（可选）
    """
    try:
        data = request.get_json() or {}
        new_name = data.get('name', '').strip()
        new_aliases = data.get('aliases', None)
        
        from urllib.parse import unquote
        char_name = unquote(character_name)
        
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()
        
        # 查找角色
        found_char = None
        for char in characters.characters:
            if char.name == char_name or char_name in char.aliases:
                found_char = char
                break
        
        if not found_char:
            return jsonify({
                "success": False,
                "error": f"未找到角色: {char_name}"
            }), 404
        
        # 更新名称
        if new_name and new_name != found_char.name:
            found_char.name = new_name
        
        # 更新别名
        if new_aliases is not None:
            found_char.aliases = [a.strip() for a in new_aliases if a.strip()]
        
        # 保存
        if char_manager.save_characters(characters):
            logger.info(f"角色信息已更新: {found_char.name}, 别名: {found_char.aliases}")
            return jsonify({
                "success": True,
                "character": found_char.to_dict()
            })
        else:
            return jsonify({
                "success": False,
                "error": "保存失败"
            }), 500
        
    except Exception as e:
        logger.error(f"更新角色信息失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ===== 形态管理 API =====

@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms', methods=['POST'])
def add_character_form(book_id: str, character_name: str):
    """
    为角色添加新形态
    
    Body:
        form_id: 形态ID（必填，如 "battle", "dark"）
        form_name: 形态显示名（必填，如 "战斗服", "黑化"）
        description: 形态描述（可选）
        
    Returns:
        {
            "success": true,
            "form": { ... }
        }
    """
    try:
        from urllib.parse import unquote
        char_name = unquote(character_name)
        
        data = request.get_json() or {}
        form_id = data.get('form_id', '').strip()
        form_name = data.get('form_name', '').strip()
        description = data.get('description', '')
        
        if not form_id:
            return jsonify({
                "success": False,
                "error": "形态ID不能为空"
            }), 400
        
        if not form_name:
            return jsonify({
                "success": False,
                "error": "形态名称不能为空"
            }), 400
        
        char_manager = CharacterManager(book_id)
        new_form = char_manager.add_form(
            character_name=char_name,
            form_id=form_id,
            form_name=form_name,
            description=description
        )
        
        if new_form:
            logger.info(f"新增形态: {char_name}/{form_id}")
            return jsonify({
                "success": True,
                "form": new_form.to_dict()
            })
        else:
            return jsonify({
                "success": False,
                "error": "添加形态失败，角色不存在或形态ID已存在"
            }), 400
        
    except Exception as e:
        logger.error(f"添加形态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>', methods=['PUT'])
def update_character_form(book_id: str, character_name: str, form_id: str):
    """
    更新角色形态信息
    
    Body:
        form_name: 新的形态名（可选）
        description: 新的描述（可选）
        
    Returns:
        {
            "success": true
        }
    """
    try:
        from urllib.parse import unquote
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
            return jsonify({
                "success": True
            })
        else:
            return jsonify({
                "success": False,
                "error": "更新失败，角色或形态不存在"
            }), 404
        
    except Exception as e:
        logger.error(f"更新形态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>', methods=['DELETE'])
def delete_character_form(book_id: str, character_name: str, form_id: str):
    """
    删除角色形态
    
    Returns:
        {
            "success": true,
            "message": "形态已删除"
        }
    """
    try:
        from urllib.parse import unquote
        char_name = unquote(character_name)
        form_id = unquote(form_id)
        
        char_manager = CharacterManager(book_id)
        success = char_manager.delete_form(
            character_name=char_name,
            form_id=form_id
        )
        
        if success:
            logger.info(f"删除形态: {char_name}/{form_id}")
            return jsonify({
                "success": True,
                "message": f"形态 '{form_id}' 已删除"
            })
        else:
            return jsonify({
                "success": False,
                "error": "删除失败，角色或形态不存在"
            }), 404
        
    except Exception as e:
        logger.error(f"删除形态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/toggle', methods=['POST'])
def toggle_character_enabled(book_id: str, character_name: str):
    """
    切换角色的启用状态
    
    Body:
        enabled: bool - 是否启用
        
    Returns:
        {
            "success": true
        }
    """
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', True)
        
        char_manager = CharacterManager(book_id)
        if char_manager.toggle_character_enabled(character_name, enabled):
            return jsonify({
                "success": True,
                "enabled": enabled
            })
        else:
            return jsonify({
                "success": False,
                "error": "角色不存在"
            }), 404
        
    except Exception as e:
        logger.error(f"切换角色启用状态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/toggle', methods=['POST'])
def toggle_form_enabled(book_id: str, character_name: str, form_id: str):
    """
    切换角色形态的启用状态
    
    Body:
        enabled: bool - 是否启用
        
    Returns:
        {
            "success": true
        }
    """
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', True)
        
        char_manager = CharacterManager(book_id)
        if char_manager.toggle_form_enabled(character_name, form_id, enabled):
            return jsonify({
                "success": True,
                "enabled": enabled
            })
        else:
            return jsonify({
                "success": False,
                "error": "角色或形态不存在"
            }), 404
        
    except Exception as e:
        logger.error(f"切换形态启用状态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/image', methods=['POST'])
def upload_form_image(book_id: str, character_name: str, form_id: str):
    """
    为指定形态上传参考图
    
    Form Data:
        - image: 图片文件
        
    Returns:
        {
            "success": true,
            "image_path": "保存的图片路径"
        }
    """
    try:
        from urllib.parse import unquote
        char_name = unquote(character_name)
        form_id = unquote(form_id)
        
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "未上传图片"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "未选择文件"
            }), 400
        
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
            return jsonify({
                "success": True,
                "image_path": saved_path
            })
        else:
            return jsonify({
                "success": False,
                "error": "上传失败，角色或形态不存在"
            }), 404
        
    except Exception as e:
        logger.error(f"上传形态参考图失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/image', methods=['DELETE'])
def delete_form_image(book_id: str, character_name: str, form_id: str):
    """
    删除指定形态的参考图
    
    Returns:
        {
            "success": true
        }
    """
    try:
        from urllib.parse import unquote
        char_name = unquote(character_name)
        form_id = unquote(form_id)
        
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()
        
        char = characters.get_character(char_name)
        if not char:
            return jsonify({
                "success": False,
                "error": "角色不存在"
            }), 404
        
        form = char.get_form(form_id)
        if not form:
            return jsonify({
                "success": False,
                "error": "形态不存在"
            }), 404
        
        # 删除图片文件
        if form.reference_image and os.path.exists(form.reference_image):
            try:
                os.remove(form.reference_image)
                logger.info(f"删除形态参考图: {form.reference_image}")
            except Exception as e:
                logger.warning(f"删除图片文件失败: {e}")
        
        form.reference_image = ""
        char_manager.save_characters(characters)
        
        return jsonify({
            "success": True
        })
        
    except Exception as e:
        logger.error(f"删除形态参考图失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/image', methods=['GET'])
def get_character_image(book_id: str, character_name: str):
    """获取角色参考图"""
    try:
        char_manager = CharacterManager(book_id)
        
        # 初始化角色（如果还没有）
        characters = run_async(char_manager.initialize_characters())
        
        # 找到对应角色
        char = next((c for c in characters.characters if c.name == character_name), None)
        reference_image = char.get_any_reference_image() if char else None
        
        if not char or not reference_image:
            return jsonify({
                "success": False,
                "error": "角色图片不存在"
            }), 404
        
        # 检查文件是否存在
        if not os.path.exists(reference_image):
            return jsonify({
                "success": False,
                "error": "图片文件不存在"
            }), 404
        
        return send_file(
            reference_image,
            mimetype='image/png'
        )
        
    except Exception as e:
        logger.error(f"获取角色图片失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
            return jsonify({
                "success": False,
                "error": "缺少图片路径参数"
            }), 400
        
        if not os.path.exists(image_path):
            return jsonify({
                "success": False,
                "error": "图片文件不存在"
            }), 404
        
        ext = os.path.splitext(image_path)[1].lower()
        mimetype = 'image/png'
        if ext in ['.jpg', '.jpeg']:
            mimetype = 'image/jpeg'
        elif ext == '.webp':
            mimetype = 'image/webp'
        
        return send_file(image_path, mimetype=mimetype)
        
    except Exception as e:
        logger.error(f"获取图片失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/orthographic', methods=['POST'])
def generate_form_orthographic(book_id: str, character_name: str, form_id: str):
    """
    生成指定形态的三视图
    
    接收用户上传的图片作为源图片，生成该形态的三视图。
    
    Returns:
        {
            "success": true,
            "image_path": "三视图图片路径"
        }
    """
    try:
        from urllib.parse import unquote
        char_name = unquote(character_name)
        form_id = unquote(form_id)
        
        # 获取上传的图片
        if 'images' not in request.files:
            return jsonify({
                "success": False,
                "error": "请上传角色图片"
            }), 400
        
        uploaded_files = request.files.getlist('images')
        if not uploaded_files or len(uploaded_files) == 0:
            return jsonify({
                "success": False,
                "error": "请至少上传一张图片"
            }), 400
        
        # 保存上传的图片
        char_manager = CharacterManager(book_id)
        saved_paths = []
        
        from datetime import datetime
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
            return jsonify({
                "success": False,
                "error": "图片保存失败"
            }), 500
        
        # 使用所有上传的图片作为参考生成三视图
        image_gen = ImageGenerator(book_id)
        
        ortho_path = run_async(image_gen.generate_character_orthographic(
            character_name=char_name,
            source_image_paths=saved_paths,
            form_id=form_id
        ))
        
        # 清理临时源图片
        for temp_path in saved_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"已清理临时源图片: {temp_path}")
            except Exception as e:
                logger.warning(f"清理源图片失败: {e}")
        
        logger.info(f"形态三视图生成成功: {char_name}/{form_id} -> {ortho_path}")
        
        return jsonify({
            "success": True,
            "image_path": ortho_path
        })
        
    except Exception as e:
        logger.error(f"生成形态三视图失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/characters/<character_name>/forms/<form_id>/set-reference', methods=['POST'])
def set_form_reference(book_id: str, character_name: str, form_id: str):
    """
    将三视图设置为指定形态的参考图
    
    Body:
        image_path: 要设置的图片路径
    """
    try:
        from urllib.parse import unquote
        char_name = unquote(character_name)
        form_id = unquote(form_id)
        
        data = request.get_json() or {}
        image_path = data.get('image_path', '')
        
        if not image_path:
            return jsonify({
                "success": False,
                "error": "缺少图片路径参数"
            }), 400
        
        if not os.path.exists(image_path):
            return jsonify({
                "success": False,
                "error": "图片文件不存在"
            }), 404
        
        # 更新形态的参考图
        char_manager = CharacterManager(book_id)
        success = char_manager.update_character_reference(char_name, image_path, form_id)
        
        if success:
            logger.info(f"已将三视图设置为形态参考图: {char_name}/{form_id} -> {image_path}")
            return jsonify({"success": True})
        else:
            return jsonify({
                "success": False,
                "error": "更新失败"
            }), 500
        
    except Exception as e:
        logger.error(f"设置形态参考图失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@manga_insight_bp.route('/<book_id>/continuation/style-references', methods=['GET'])
def get_style_references(book_id: str):
    """
    获取画风参考图路径（原漫画最后N页）
    
    Query:
        count: 需要的页数（默认3）
        
    Returns:
        {
            "success": true,
            "images": ["路径1", "路径2", ...]
        }
    """
    try:
        count = request.args.get('count', 3, type=int)
        
        # 获取原漫画页面路径
        image_gen = ImageGenerator(book_id)
        style_refs = image_gen.get_style_reference_images(count)
        
        return jsonify({
            "success": True,
            "images": style_refs
        })
        
    except Exception as e:
        logger.error(f"获取画风参考图失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# 剧情生成
# ============================================================

@manga_insight_bp.route('/<book_id>/continuation/script', methods=['POST'])
def generate_script(book_id: str):
    """
    生成全话脚本（第一层）
    
    Request Body:
        {
            "direction": "用户指定的续写方向",
            "page_count": 15
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
        
        story_gen = StoryGenerator(book_id)
        script = run_async(story_gen.generate_chapter_script(
            user_direction=direction,
            page_count=page_count
        ))
        
        # 自动保存脚本到持久化存储
        storage = AnalysisStorage(book_id)
        run_async(storage.save_continuation_script(script.to_dict()))
        logger.info(f"脚本已自动保存")
        
        return jsonify({
            "success": True,
            "script": script.to_dict()
        })
        
    except Exception as e:
        logger.error(f"生成脚本失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
            return jsonify({
                "success": False,
                "error": "缺少脚本数据"
            }), 400
        
        script = ChapterScript.from_dict(script_data)
        
        if page_number < 1 or page_number > script.page_count:
            return jsonify({
                "success": False,
                "error": f"页码 {page_number} 超出范围 (1-{script.page_count})"
            }), 400
        
        story_gen = StoryGenerator(book_id)
        page = run_async(story_gen.generate_page_details(script, page_number))
        
        return jsonify({
            "success": True,
            "page": page.to_dict()
        })
        
    except Exception as e:
        logger.error(f"生成第 {page_number} 页详情失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
            return jsonify({
                "success": False,
                "error": "缺少页面数据"
            }), 400
        
        pages = [PageContent.from_dict(p) for p in pages_data]
        
        story_gen = StoryGenerator(book_id)
        updated_pages = run_async(story_gen.generate_all_image_prompts(pages))
        
        return jsonify({
            "success": True,
            "pages": [p.to_dict() for p in updated_pages]
        })
        
    except Exception as e:
        logger.error(f"生成提示词失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
            return jsonify({
                "success": False,
                "error": "缺少页面数据"
            }), 400
        
        page = PageContent.from_dict(page_data)
        
        story_gen = StoryGenerator(book_id)
        image_prompt = run_async(story_gen.generate_image_prompt(page))
        page.image_prompt = image_prompt
        
        return jsonify({
            "success": True,
            "page": page.to_dict()
        })
        
    except Exception as e:
        logger.error(f"生成第 {page_number} 页提示词失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# 图片生成
# ============================================================

@manga_insight_bp.route('/<book_id>/continuation/generate/<int:page_number>', methods=['POST'])
def generate_page_image(book_id: str, page_number: int):
    """
    生成单页图片
    
    Request Body:
        {
            "page": {...页面数据...},
            "style_reference_images": ["路径1", "路径2"],
            "session_id": "会话ID"
        }
        
    Returns:
        {
            "success": true,
            "image_path": "生成的图片路径"
        }
    """
    try:
        data = request.get_json() or {}
        page_data = data.get("page", {})
        style_refs = data.get("style_reference_images", [])
        session_id = data.get("session_id", "")
        style_ref_count = data.get("style_ref_count", 3)  # 用户设置的画风参考图数量
        
        if not page_data:
            return jsonify({
                "success": False,
                "error": "缺少页面数据"
            }), 400
        
        page = PageContent.from_dict(page_data)
        
        # 获取角色配置
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()
        
        image_gen = ImageGenerator(book_id)
        image_path = run_async(image_gen.generate_page_image(
            page_content=page,
            characters=characters,
            style_reference_images=style_refs,
            session_id=session_id,
            style_ref_count=style_ref_count  # 传递滑动窗口大小
        ))
        
        return jsonify({
            "success": True,
            "image_path": image_path
        })
        
    except Exception as e:
        logger.error(f"生成图片失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/regenerate/<int:page_number>', methods=['POST'])
def regenerate_page_image(book_id: str, page_number: int):
    """
    重新生成页面图片（保留上一版本）
    
    Request Body:
        {
            "page": {...页面数据...},
            "style_reference_images": ["路径1", "路径2"],
            "session_id": "会话ID"
        }
        
    Returns:
        {
            "success": true,
            "image_path": "新图片路径",
            "previous_path": "上一版本路径"
        }
    """
    try:
        data = request.get_json() or {}
        page_data = data.get("page", {})
        style_refs = data.get("style_reference_images", [])
        session_id = data.get("session_id", "")
        style_ref_count = data.get("style_ref_count", 3)  # 用户设置的画风参考图数量
        
        if not page_data:
            return jsonify({
                "success": False,
                "error": "缺少页面数据"
            }), 400
        
        page = PageContent.from_dict(page_data)
        previous_path = page.image_url
        
        # 获取角色配置
        char_manager = CharacterManager(book_id)
        characters = char_manager.load_characters()
        
        image_gen = ImageGenerator(book_id)
        image_path = run_async(image_gen.regenerate_page_image(
            page_content=page,
            characters=characters,
            style_reference_images=style_refs,
            session_id=session_id,
            style_ref_count=style_ref_count  # 传递滑动窗口大小
        ))
        
        return jsonify({
            "success": True,
            "image_path": image_path,
            "previous_path": previous_path
        })
        
    except Exception as e:
        logger.error(f"重新生成图片失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# 导出功能
# ============================================================

@manga_insight_bp.route('/<book_id>/continuation/export/images', methods=['POST'])
def export_as_images(book_id: str):
    """
    导出为图片 ZIP
    
    Request Body:
        {
            "pages": [...页面数据，包含 image_url...]
        }
        
    如果不传 pages，则自动从持久化存储加载
        
    Returns:
        ZIP 文件
    """
    try:
        import zipfile
        import io
        
        data = request.get_json() or {}
        pages_data = data.get("pages", [])
        
        # 如果没有传入 pages，则从持久化存储加载
        if not pages_data:
            storage = AnalysisStorage(book_id)
            loaded = run_async(storage.load_continuation_pages())
            if loaded:
                pages_data = loaded.get("pages", [])
        
        if not pages_data:
            return jsonify({
                "success": False,
                "error": "没有可导出的页面数据"
            }), 400
        
        # 创建 ZIP 文件
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for page in pages_data:
                image_url = page.get("image_url", "")
                page_number = page.get("page_number", 0)
                if image_url and os.path.exists(image_url):
                    filename = f"page_{page_number:03d}.png"
                    zf.write(image_url, filename)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{book_id}_continuation.zip'
        )
        
    except Exception as e:
        logger.error(f"导出 ZIP 失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/<book_id>/continuation/export/pdf', methods=['POST'])
def export_as_pdf(book_id: str):
    """
    导出为 PDF
    
    Request Body:
        {
            "pages": [...页面数据，包含 image_url...]
        }
        
    如果不传 pages，则自动从持久化存储加载
        
    Returns:
        PDF 文件
    """
    try:
        from PIL import Image
        import io
        
        data = request.get_json() or {}
        pages_data = data.get("pages", [])
        
        # 如果没有传入 pages，则从持久化存储加载
        if not pages_data:
            storage = AnalysisStorage(book_id)
            loaded = run_async(storage.load_continuation_pages())
            if loaded:
                pages_data = loaded.get("pages", [])
        
        if not pages_data:
            return jsonify({
                "success": False,
                "error": "没有可导出的页面数据"
            }), 400
        
        # 收集所有图片
        images = []
        for page in pages_data:
            image_url = page.get("image_url", "")
            if image_url and os.path.exists(image_url):
                img = Image.open(image_url)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                images.append(img)
        
        if not images:
            return jsonify({
                "success": False,
                "error": "没有可导出的图片"
            }), 400
        
        # 创建 PDF
        pdf_buffer = io.BytesIO()
        images[0].save(
            pdf_buffer,
            'PDF',
            save_all=True,
            append_images=images[1:] if len(images) > 1 else []
        )
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'{book_id}_continuation.pdf'
        )
        
    except Exception as e:
        logger.error(f"导出 PDF 失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
