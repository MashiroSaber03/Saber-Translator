# src/app/api/page_storage_api.py
"""
单页存储 API 路由

提供单页独立保存/加载的 RESTful API
"""

from flask import Blueprint, request, jsonify, send_file
import io
import logging

from src.core.page_storage import (
    save_session_meta,
    load_session_meta,
    save_page_image,
    load_page_image,
    save_page_meta,
    load_page_meta,
    presave_all_pages,
    save_translated_page,
    load_session
)

logger = logging.getLogger("PageStorageAPI")

page_storage_bp = Blueprint('page_storage', __name__, url_prefix='/api/sessions')


# ============================================================
# 会话元数据 API
# ============================================================

@page_storage_bp.route('/meta/<path:session_path>', methods=['POST'])
def api_save_session_meta(session_path):
    """保存会话元数据"""
    try:
        data = request.get_json()
        result = save_session_meta(session_path, data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"保存会话元数据失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@page_storage_bp.route('/meta/<path:session_path>', methods=['GET'])
def api_load_session_meta(session_path):
    """加载会话元数据"""
    try:
        data = load_session_meta(session_path)
        if data:
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"success": False, "error": "会话不存在"}), 404
    except Exception as e:
        logger.error(f"加载会话元数据失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# 单页图片 API
# ============================================================

@page_storage_bp.route('/page/<path:session_path>/<int:page_index>/<image_type>', methods=['POST'])
def api_save_page_image(session_path, page_index, image_type):
    """保存单页图片"""
    try:
        data = request.get_json()
        base64_data = data.get("data", "")
        result = save_page_image(session_path, page_index, image_type, base64_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"保存页面图片失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@page_storage_bp.route('/page/<path:session_path>/<int:page_index>/<image_type>', methods=['GET'])
def api_load_page_image(session_path, page_index, image_type):
    """加载单页图片"""
    try:
        image_bytes = load_page_image(session_path, page_index, image_type)
        if image_bytes:
            return send_file(
                io.BytesIO(image_bytes),
                mimetype='image/png',
                as_attachment=False
            )
        else:
            return jsonify({"success": False, "error": "图片不存在"}), 404
    except Exception as e:
        logger.error(f"加载页面图片失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# 单页元数据 API
# ============================================================

@page_storage_bp.route('/page/<path:session_path>/<int:page_index>/meta', methods=['POST'])
def api_save_page_meta(session_path, page_index):
    """保存单页元数据"""
    try:
        data = request.get_json()
        result = save_page_meta(session_path, page_index, data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"保存页面元数据失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@page_storage_bp.route('/page/<path:session_path>/<int:page_index>/meta', methods=['GET'])
def api_load_page_meta(session_path, page_index):
    """加载单页元数据"""
    try:
        data = load_page_meta(session_path, page_index)
        if data:
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"success": False, "error": "元数据不存在"}), 404
    except Exception as e:
        logger.error(f"加载页面元数据失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# 批量操作 API
# ============================================================

@page_storage_bp.route('/presave/<path:session_path>', methods=['POST'])
def api_presave_all_pages(session_path):
    """预保存所有页面"""
    try:
        data = request.get_json()
        images = data.get("images", [])
        ui_settings = data.get("ui_settings", {})
        
        result = presave_all_pages(session_path, images, ui_settings)
        
        # 更新书架章节的图片数量
        if result.get("success"):
            try:
                from src.core import bookshelf_manager
                # session_path 格式: bookshelf/{book_id}/{chapter_id}
                path_parts = session_path.replace('\\', '/').split('/')
                if 'bookshelf' in path_parts:
                    bookshelf_idx = path_parts.index('bookshelf')
                    if len(path_parts) > bookshelf_idx + 2:
                        book_id = path_parts[bookshelf_idx + 1]
                        chapter_id = path_parts[bookshelf_idx + 2]
                        image_count = len(images)
                        bookshelf_manager.update_chapter_image_count(book_id, chapter_id, image_count)
                        logger.info(f"已更新章节 {book_id}/{chapter_id} 图片数量为 {image_count}")
            except Exception as e:
                logger.warning(f"更新章节图片数量时出错（非致命）: {e}")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"预保存失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@page_storage_bp.route('/save_translated/<path:session_path>/<int:page_index>', methods=['POST'])
def api_save_translated_page(session_path, page_index):
    """保存翻译完成的页面"""
    try:
        data = request.get_json()
        translated_data = data.get("translated")
        clean_data = data.get("clean")
        page_meta = data.get("meta")
        
        result = save_translated_page(
            session_path, 
            page_index,
            translated_data=translated_data,
            clean_data=clean_data,
            page_meta=page_meta
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"保存翻译页面失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@page_storage_bp.route('/load/<path:session_path>', methods=['GET'])
def api_load_session(session_path):
    """加载完整会话"""
    try:
        data = load_session(session_path)
        if data:
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"success": False, "error": "会话不存在"}), 404
    except Exception as e:
        logger.error(f"加载会话失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
