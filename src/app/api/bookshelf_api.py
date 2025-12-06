# src/app/api/bookshelf_api.py
"""
书架API - 提供书籍和章节的RESTful API接口
"""

from flask import Blueprint, request, jsonify, Response
import logging
import base64

from src.core import bookshelf_manager

logger = logging.getLogger("BookshelfAPI")

bookshelf_bp = Blueprint('bookshelf', __name__, url_prefix='/api/bookshelf')


# ==================== 书籍 API ====================

@bookshelf_bp.route('/books', methods=['GET'])
def list_books():
    """获取所有书籍列表，支持搜索和标签筛选"""
    try:
        # 获取查询参数
        search = request.args.get('search', '').strip()
        tags_param = request.args.get('tags', '')
        tags = [t.strip() for t in tags_param.split(',') if t.strip()] if tags_param else None
        
        books = bookshelf_manager.get_all_books(search=search or None, tags=tags)
        return jsonify({
            "success": True,
            "books": books
        })
    except Exception as e:
        logger.error(f"获取书籍列表失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books', methods=['POST'])
def create_book():
    """创建新书籍"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        title = data.get('title', '').strip()
        if not title:
            return jsonify({"success": False, "error": "书籍标题不能为空"}), 400
        
        cover_data = data.get('cover')  # Base64封面数据（可选）
        tags = data.get('tags', [])  # 标签列表（可选）
        
        book = bookshelf_manager.create_book(title, cover_data, tags)
        if book:
            return jsonify({
                "success": True,
                "book": book
            })
        else:
            return jsonify({"success": False, "error": "创建书籍失败"}), 500
    except Exception as e:
        logger.error(f"创建书籍失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>', methods=['GET'])
def get_book(book_id):
    """获取书籍详情"""
    try:
        book = bookshelf_manager.get_book(book_id)
        if book:
            return jsonify({
                "success": True,
                "book": book
            })
        else:
            return jsonify({"success": False, "error": "书籍不存在"}), 404
    except Exception as e:
        logger.error(f"获取书籍详情失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>', methods=['PUT'])
def update_book(book_id):
    """更新书籍信息"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        title = data.get('title')
        cover_data = data.get('cover')
        tags = data.get('tags')  # 可能是列表或None
        
        book = bookshelf_manager.update_book(book_id, title, cover_data, tags)
        if book:
            return jsonify({
                "success": True,
                "book": book
            })
        else:
            return jsonify({"success": False, "error": "更新书籍失败"}), 500
    except Exception as e:
        logger.error(f"更新书籍失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>', methods=['DELETE'])
def delete_book(book_id):
    """删除书籍"""
    try:
        if bookshelf_manager.delete_book(book_id):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "删除书籍失败"}), 500
    except Exception as e:
        logger.error(f"删除书籍失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>/cover', methods=['GET'])
def get_book_cover(book_id):
    """获取书籍封面图片"""
    try:
        cover_data = bookshelf_manager.get_cover(book_id)
        if cover_data:
            # 解码Base64并返回图片
            try:
                image_data = base64.b64decode(cover_data)
                # 尝试检测图片格式
                content_type = 'image/png'
                if image_data[:3] == b'\xff\xd8\xff':
                    content_type = 'image/jpeg'
                elif image_data[:4] == b'\x89PNG':
                    content_type = 'image/png'
                elif image_data[:4] == b'RIFF':
                    content_type = 'image/webp'
                
                return Response(image_data, mimetype=content_type)
            except Exception as e:
                logger.error(f"解码封面失败: {e}")
                return jsonify({"success": False, "error": "封面数据损坏"}), 500
        else:
            # 返回默认封面或404
            return jsonify({"success": False, "error": "封面不存在"}), 404
    except Exception as e:
        logger.error(f"获取封面失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 章节 API ====================

@bookshelf_bp.route('/books/<book_id>/chapters', methods=['GET'])
def list_chapters(book_id):
    """获取书籍的所有章节"""
    try:
        chapters = bookshelf_manager.get_chapters(book_id)
        return jsonify({
            "success": True,
            "chapters": chapters
        })
    except Exception as e:
        logger.error(f"获取章节列表失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>/chapters', methods=['POST'])
def create_chapter(book_id):
    """创建新章节"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        title = data.get('title', '').strip()
        if not title:
            return jsonify({"success": False, "error": "章节标题不能为空"}), 400
        
        order = data.get('order')  # 可选的排序位置
        
        chapter = bookshelf_manager.create_chapter(book_id, title, order)
        if chapter:
            return jsonify({
                "success": True,
                "chapter": chapter
            })
        else:
            return jsonify({"success": False, "error": "创建章节失败"}), 500
    except Exception as e:
        logger.error(f"创建章节失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>/chapters/<chapter_id>', methods=['GET'])
def get_chapter(book_id, chapter_id):
    """获取章节详情"""
    try:
        chapter = bookshelf_manager.get_chapter(book_id, chapter_id)
        if chapter:
            return jsonify({
                "success": True,
                "chapter": chapter
            })
        else:
            return jsonify({"success": False, "error": "章节不存在"}), 404
    except Exception as e:
        logger.error(f"获取章节详情失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>/chapters/<chapter_id>', methods=['PUT'])
def update_chapter(book_id, chapter_id):
    """更新章节信息"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        title = data.get('title')
        
        chapter = bookshelf_manager.update_chapter(book_id, chapter_id, title)
        if chapter:
            return jsonify({
                "success": True,
                "chapter": chapter
            })
        else:
            return jsonify({"success": False, "error": "更新章节失败"}), 500
    except Exception as e:
        logger.error(f"更新章节失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>/chapters/<chapter_id>', methods=['DELETE'])
def delete_chapter(book_id, chapter_id):
    """删除章节"""
    try:
        if bookshelf_manager.delete_chapter(book_id, chapter_id):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "删除章节失败"}), 500
    except Exception as e:
        logger.error(f"删除章节失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>/chapters/reorder', methods=['POST'])
def reorder_chapters(book_id):
    """重新排序章节"""
    try:
        data = request.get_json()
        if not data or 'chapter_ids' not in data:
            return jsonify({"success": False, "error": "请求数据无效"}), 400
        
        chapter_ids = data.get('chapter_ids', [])
        
        if bookshelf_manager.reorder_chapters(book_id, chapter_ids):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "重新排序失败"}), 500
    except Exception as e:
        logger.error(f"重新排序章节失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/<book_id>/chapters/<chapter_id>/image-count', methods=['PUT'])
def update_chapter_image_count(book_id, chapter_id):
    """更新章节的图片数量"""
    try:
        data = request.get_json()
        if not data or 'count' not in data:
            return jsonify({"success": False, "error": "请求数据无效"}), 400
        
        count = data.get('count', 0)
        
        if bookshelf_manager.update_chapter_image_count(book_id, chapter_id, count):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "更新失败"}), 500
    except Exception as e:
        logger.error(f"更新章节图片数量失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 章节图片获取 API（用于阅读页面） ====================

@bookshelf_bp.route('/books/<book_id>/chapters/<chapter_id>/images', methods=['GET'])
def get_chapter_images(book_id, chapter_id):
    """
    获取章节的所有图片（用于阅读页面）
    返回原图和翻译后的图片
    
    支持两种返回格式：
    - format=url (默认): 返回图片 URL，前端通过 <img src="url"> 加载
    - format=base64: 返回 Base64 数据（向后兼容）
    """
    from src.core import session_manager
    
    # 获取返回格式参数
    return_format = request.args.get('format', 'url')  # 默认返回 URL
    
    try:
        # 获取章节的会话路径
        session_path = bookshelf_manager.get_chapter_session_path(book_id, chapter_id)
        if not session_path:
            return jsonify({
                "success": True,
                "images": [],
                "chapter_name": "",
                "prev_chapter_id": None,
                "next_chapter_id": None
            })
        
        # 加载会话数据（包含元数据）
        session_data = session_manager.load_session_by_path(session_path)
        
        images = []
        if session_data and session_data.get('images'):
            for idx, img_data in enumerate(session_data['images']):
                if return_format == 'url':
                    # 返回图片 URL
                    original_url = f"/api/sessions/image_by_path/{session_path}/image_{idx}_original.png" if img_data.get('originalDataURL') else None
                    translated_url = f"/api/sessions/image_by_path/{session_path}/image_{idx}_translated.png" if img_data.get('translatedDataURL') else None
                    images.append({
                        "index": idx,
                        "original": original_url,
                        "translated": translated_url,
                        "fileName": img_data.get('fileName', f'image_{idx}.png')
                    })
                else:
                    # 返回 Base64 数据（向后兼容）
                    images.append({
                        "index": idx,
                        "original": img_data.get('originalDataURL'),
                        "translated": img_data.get('translatedDataURL'),
                        "fileName": img_data.get('fileName', f'image_{idx}.png')
                    })
        
        # 获取书籍信息以确定上下章
        book = bookshelf_manager.get_book(book_id)
        chapters = book.get('chapters', []) if book else []
        
        current_idx = -1
        for i, ch in enumerate(chapters):
            if ch.get('id') == chapter_id:
                current_idx = i
                break
        
        prev_chapter_id = chapters[current_idx - 1]['id'] if current_idx > 0 else None
        next_chapter_id = chapters[current_idx + 1]['id'] if current_idx < len(chapters) - 1 else None
        
        # 获取章节名称
        chapter_info = next((ch for ch in chapters if ch.get('id') == chapter_id), None)
        chapter_name = chapter_info.get('title', '') if chapter_info else ''
        
        return jsonify({
            "success": True,
            "images": images,
            "chapter_name": chapter_name,
            "prev_chapter_id": prev_chapter_id,
            "next_chapter_id": next_chapter_id
        })
        
    except Exception as e:
        logger.error(f"获取章节图片失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 章节会话保存 API ====================

@bookshelf_bp.route('/save_chapter_session', methods=['POST'])
def save_chapter_session():
    """保存章节的会话数据"""
    from src.core import session_manager
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        book_id = data.get('book_id')
        chapter_id = data.get('chapter_id')
        session_data = data.get('session_data')
        
        if not book_id or not chapter_id:
            return jsonify({"success": False, "error": "缺少书籍或章节ID"}), 400
        
        if not session_data:
            return jsonify({"success": False, "error": "缺少会话数据"}), 400
        
        # 获取章节的会话路径
        session_path = bookshelf_manager.get_chapter_session_path(book_id, chapter_id)
        if not session_path:
            return jsonify({"success": False, "error": "无法获取章节会话路径"}), 500
        
        # 保存会话数据
        if session_manager.save_session_by_path(session_path, session_data):
            # 更新章节的图片数量
            image_count = len(session_data.get('images', []))
            bookshelf_manager.update_chapter_image_count(book_id, chapter_id, image_count)
            
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "保存会话数据失败"}), 500
            
    except Exception as e:
        logger.error(f"保存章节会话失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 标签 API ====================

@bookshelf_bp.route('/tags', methods=['GET'])
def list_tags():
    """获取所有标签"""
    try:
        tags = bookshelf_manager.get_all_tags()
        return jsonify({
            "success": True,
            "tags": tags
        })
    except Exception as e:
        logger.error(f"获取标签列表失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/tags', methods=['POST'])
def create_tag():
    """创建新标签"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        name = data.get('name', '').strip()
        if not name:
            return jsonify({"success": False, "error": "标签名称不能为空"}), 400
        
        color = data.get('color')
        
        tag = bookshelf_manager.create_tag(name, color)
        if tag:
            return jsonify({
                "success": True,
                "tag": tag
            })
        else:
            return jsonify({"success": False, "error": "标签已存在或创建失败"}), 400
    except Exception as e:
        logger.error(f"创建标签失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/tags/<tag_name>', methods=['PUT'])
def update_tag(tag_name):
    """更新标签"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        new_name = data.get('name')
        color = data.get('color')
        
        tag = bookshelf_manager.update_tag(tag_name, new_name, color)
        if tag:
            return jsonify({
                "success": True,
                "tag": tag
            })
        else:
            return jsonify({"success": False, "error": "更新标签失败"}), 500
    except Exception as e:
        logger.error(f"更新标签失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/tags/<tag_name>', methods=['DELETE'])
def delete_tag(tag_name):
    """删除标签"""
    try:
        if bookshelf_manager.delete_tag(tag_name):
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "删除标签失败"}), 500
    except Exception as e:
        logger.error(f"删除标签失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 批量操作 API ====================

@bookshelf_bp.route('/books/batch/delete', methods=['POST'])
def batch_delete_books():
    """批量删除书籍"""
    try:
        data = request.get_json()
        if not data or 'book_ids' not in data:
            return jsonify({"success": False, "error": "请求数据无效"}), 400
        
        book_ids = data.get('book_ids', [])
        if not book_ids:
            return jsonify({"success": False, "error": "未选择书籍"}), 400
        
        result = bookshelf_manager.delete_books_batch(book_ids)
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        logger.error(f"批量删除书籍失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/batch/add-tags', methods=['POST'])
def batch_add_tags():
    """批量为书籍添加标签"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        book_ids = data.get('book_ids', [])
        tags = data.get('tags', [])
        
        if not book_ids:
            return jsonify({"success": False, "error": "未选择书籍"}), 400
        if not tags:
            return jsonify({"success": False, "error": "未选择标签"}), 400
        
        result = bookshelf_manager.add_tags_to_books_batch(book_ids, tags)
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        logger.error(f"批量添加标签失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@bookshelf_bp.route('/books/batch/remove-tags', methods=['POST'])
def batch_remove_tags():
    """批量从书籍移除标签"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        book_ids = data.get('book_ids', [])
        tags = data.get('tags', [])
        
        if not book_ids:
            return jsonify({"success": False, "error": "未选择书籍"}), 400
        if not tags:
            return jsonify({"success": False, "error": "未选择标签"}), 400
        
        result = bookshelf_manager.remove_tags_from_books_batch(book_ids, tags)
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        logger.error(f"批量移除标签失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
