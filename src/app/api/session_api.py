# src/app/api/session_api.py

from flask import Blueprint, request, jsonify, send_file, Response
import logging
import os
import base64
from src.core import session_manager # 导入我们创建的会话管理器模块
from src.core import bookshelf_manager # 导入书架管理器，用于更新章节图片数量

# 获取 logger
logger = logging.getLogger("SessionAPI")

# 定义蓝图实例，URL 前缀统一为 /api
session_bp = Blueprint('session_api', __name__, url_prefix='/api/sessions')

# --- API 路由 ---

@session_bp.route('/save', methods=['POST'])
def save_session_api():
    """
    接收前端发送的会话数据并保存。
    请求体需要是 JSON 格式，包含 "session_name" 和 "session_data" 键。
    """
    logger.info("收到保存会话请求...")
    data = request.get_json()

    if not data:
        logger.warning("保存请求失败：请求体为空或非 JSON。")
        return jsonify({'success': False, 'error': '请求体必须是 JSON 格式'}), 400

    session_name = data.get('session_name')
    session_data = data.get('session_data')

    if not session_name:
        logger.warning("保存请求失败：缺少 session_name。")
        return jsonify({'success': False, 'error': '缺少会话名称 (session_name)'}), 400

    if not session_data or not isinstance(session_data, dict):
        logger.warning("保存请求失败：缺少或无效的 session_data。")
        return jsonify({'success': False, 'error': '缺少有效的会话数据 (session_data)'}), 400

    # 基本验证 session_data 结构 (可选，但推荐)
    if "images" not in session_data or not isinstance(session_data["images"], list) or \
       "currentImageIndex" not in session_data or not isinstance(session_data["currentImageIndex"], int):
        logger.warning("保存请求失败：session_data 结构不完整。")
        return jsonify({'success': False, 'error': '会话数据结构不完整，需要 images (列表) 和 currentImageIndex (整数)'}), 400

    try:
        # 调用核心保存逻辑
        success = session_manager.save_session(session_name, session_data)

        if success:
            logger.info(f"API: 会话 '{session_name}' 保存成功。")
            return jsonify({'success': True, 'message': f'会话 "{session_name}" 保存成功！'})
        else:
            # save_session 内部已经记录了详细错误，这里返回通用错误
            logger.error(f"API: 会话 '{session_name}' 保存失败 (由 session_manager 返回)。")
            return jsonify({'success': False, 'error': f'保存会话 "{session_name}" 失败，请查看后端日志获取详细信息。'}), 500

    except Exception as e:
        # 捕获 session_manager 可能抛出的未预料错误
        logger.error(f"调用 session_manager.save_session 时发生意外错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '保存会话时发生服务器内部错误。'}), 500

# === 新增：列出所有会话 API ===
@session_bp.route('/list', methods=['GET'])
def list_sessions_api():
    """列出所有已保存的会话元数据。"""
    logger.info("收到列出所有会话请求...")
    try:
        sessions = session_manager.list_sessions()
        logger.info(f"API: 成功获取到 {len(sessions)} 个会话。")
        # 返回包含会话列表的 JSON
        # sessions 已经是 list of dicts 了
        return jsonify({'success': True, 'sessions': sessions})
    except Exception as e:
        logger.error(f"调用 session_manager.list_sessions 时发生错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '获取会话列表时发生服务器错误。'}), 500
# === 结束新增 ===

# === 新增：加载指定会话 API ===
@session_bp.route('/load', methods=['GET'])
def load_session_api():
    """加载指定名称的会话数据。"""
    session_name = request.args.get('name') # 从 URL 查询参数获取会话名称
    logger.info(f"收到加载会话请求: name='{session_name}'")

    if not session_name:
        logger.warning("加载请求失败：缺少 'name' 查询参数。")
        return jsonify({'success': False, 'error': "缺少会话名称参数 ('name')"}), 400

    try:
        loaded_data = session_manager.load_session(session_name)

        if loaded_data is not None:
            logger.info(f"API: 会话 '{session_name}' 加载成功。")
            # 返回包含完整会话数据的 JSON
            return jsonify({'success': True, 'session_data': loaded_data})
        else:
            # load_session 内部会记录错误，这里返回 404 或通用错误
            logger.warning(f"API: 未找到或无法加载会话 '{session_name}'。")
            return jsonify({'success': False, 'error': f'无法找到或加载会话 "{session_name}"。'}), 404
    except Exception as e:
        logger.error(f"调用 session_manager.load_session 时发生意外错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '加载会话时发生服务器内部错误。'}), 500
# === 结束新增 ===

# === 新增：删除指定会话 API ===
@session_bp.route('/delete', methods=['POST'])
def delete_session_api():
    """删除指定名称的会话。"""
    data = request.get_json()
    logger.info(f"收到删除会话请求: {data}")

    if not data or 'session_name' not in data:
        logger.warning("删除请求失败：请求体无效或缺少 session_name。")
        return jsonify({'success': False, 'error': '请求体必须是包含 "session_name" 的 JSON 对象'}), 400

    session_name = data.get('session_name')
    if not session_name: # 再次检查以防万一
         logger.warning("删除请求失败：session_name 为空。")
         return jsonify({'success': False, 'error': '会话名称不能为空'}), 400

    try:
        success = session_manager.delete_session(session_name)
        if success:
            logger.info(f"API: 会话 '{session_name}' 删除成功。")
            return jsonify({'success': True, 'message': f'会话 "{session_name}" 已删除。'})
        else:
            # delete_session 内部会记录错误
            logger.error(f"API: 会话 '{session_name}' 删除失败 (由 session_manager 返回)。")
            # 可能是文件未找到或权限问题
            return jsonify({'success': False, 'error': f'删除会话 "{session_name}" 失败，可能文件不存在或权限不足。'}), 500
    except Exception as e:
        logger.error(f"调用 session_manager.delete_session 时发生意外错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '删除会话时发生服务器内部错误。'}), 500
# === 结束新增 ===

# === 新增：重命名指定会话 API ===
@session_bp.route('/rename', methods=['POST'])
def rename_session_api():
    """重命名指定的会话。"""
    data = request.get_json()
    logger.info(f"收到重命名会话请求: {data}")

    if not data or 'old_name' not in data or 'new_name' not in data:
        logger.warning("重命名请求失败：请求体无效或缺少 old_name 或 new_name。")
        return jsonify({'success': False, 'error': '请求体必须是包含 "old_name" 和 "new_name" 的 JSON 对象'}), 400

    old_name = data.get('old_name')
    new_name = data.get('new_name')

    if not old_name or not new_name:
        logger.warning("重命名请求失败：旧名称或新名称不能为空。")
        return jsonify({'success': False, 'error': '旧名称和新名称都不能为空'}), 400

    if old_name == new_name:
        logger.info(f"重命名请求：新旧名称相同 '{old_name}'，无需操作。")
        return jsonify({'success': True, 'message': '新旧名称相同，无需重命名。'}) # 可以认为是成功的

    try:
        success = session_manager.rename_session(old_name, new_name)
        if success:
            logger.info(f"API: 会话从 '{old_name}' 重命名为 '{new_name}' 成功。")
            return jsonify({'success': True, 'message': f'会话已成功重命名为 "{new_name}"。'})
        else:
            # rename_session 内部会记录错误
            logger.error(f"API: 重命名会话失败 (由 session_manager 返回)。")
            # 错误原因可能是旧名称不存在、新名称已存在或文件系统错误
            return jsonify({'success': False, 'error': f'重命名会话失败，请检查名称是否有效、"{new_name}" 是否已存在或查看后端日志。'}), 500 # 返回 500 更合适
    except Exception as e:
        logger.error(f"调用 session_manager.rename_session 时发生意外错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '重命名会话时发生服务器内部错误。'}), 500
# === 结束新增 ===

# === 新增：按路径加载会话 API ===
@session_bp.route('/load_by_path', methods=['POST'])
def load_session_by_path_api():
    """按路径加载会话数据（用于书籍/章节）"""
    data = request.get_json()
    session_path = data.get('path') if data else None
    
    logger.info(f"收到按路径加载会话请求: path='{session_path}'")
    
    if not session_path:
        logger.warning("加载请求失败：缺少 'path' 参数。")
        return jsonify({'success': False, 'error': "缺少会话路径参数 ('path')"}), 400
    
    try:
        loaded_data = session_manager.load_session_by_path(session_path)
        
        if loaded_data is not None:
            logger.info(f"API: 会话路径 '{session_path}' 加载成功。")
            return jsonify({'success': True, 'session_data': loaded_data})
        else:
            logger.warning(f"API: 未找到或无法加载会话路径 '{session_path}'。")
            return jsonify({'success': False, 'error': f'无法找到或加载会话路径 "{session_path}"。'}), 404
    except Exception as e:
        logger.error(f"按路径加载会话时发生意外错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '加载会话时发生服务器内部错误。'}), 500
# === 结束新增 ===


# === 新增：分批保存 API ===
@session_bp.route('/batch_save/start', methods=['POST'])
def batch_save_start_api():
    """
    分批保存 - 第一步：开始保存，创建目录和元数据。
    请求体: { "session_path": "xxx", "metadata": { ui_settings, images_meta, currentImageIndex } }
    """
    logger.info("收到分批保存开始请求...")
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'error': '请求体为空'}), 400
    
    session_path = data.get('session_path')
    metadata = data.get('metadata', {})
    
    if not session_path:
        return jsonify({'success': False, 'error': '缺少 session_path'}), 400
    
    result = session_manager.start_batch_save(session_path, metadata)
    
    if result.get('success'):
        return jsonify({
            'success': True,
            'session_folder': result.get('session_folder')
        })
    else:
        return jsonify({
            'success': False,
            'error': result.get('error', '未知错误')
        }), 500


@session_bp.route('/batch_save/image', methods=['POST'])
def batch_save_image_api():
    """
    分批保存 - 第二步：保存单张图片。
    请求体: { "session_folder": "xxx", "image_index": 0, "image_type": "original/translated/clean", "image_data": "base64..." }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'error': '请求体为空'}), 400
    
    session_folder = data.get('session_folder')
    image_index = data.get('image_index')
    image_type = data.get('image_type')
    image_data = data.get('image_data')
    
    if not session_folder or image_index is None or not image_type:
        return jsonify({'success': False, 'error': '缺少必要参数'}), 400
    
    result = session_manager.save_single_image(session_folder, image_index, image_type, image_data)
    
    if result.get('success'):
        return jsonify({'success': True})
    else:
        return jsonify({
            'success': False,
            'error': result.get('error', '保存图片失败')
        }), 500


@session_bp.route('/batch_save/complete', methods=['POST'])
def batch_save_complete_api():
    """
    分批保存 - 第三步：完成保存，更新元数据。
    请求体: { "session_folder": "xxx", "images_meta": [...] }
    """
    logger.info("收到分批保存完成请求...")
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'error': '请求体为空'}), 400
    
    session_folder = data.get('session_folder')
    images_meta = data.get('images_meta', [])
    
    if not session_folder:
        return jsonify({'success': False, 'error': '缺少 session_folder'}), 400
    
    result = session_manager.complete_batch_save(session_folder, images_meta)
    
    if result.get('success'):
        # 检查是否是书架章节保存，如果是则更新章节图片数量
        # session_folder 格式: .../data/sessions/bookshelf/{book_id}/{chapter_id}
        try:
            folder_parts = session_folder.replace('\\', '/').split('/')
            if 'bookshelf' in folder_parts:
                bookshelf_idx = folder_parts.index('bookshelf')
                if len(folder_parts) > bookshelf_idx + 2:
                    book_id = folder_parts[bookshelf_idx + 1]
                    chapter_id = folder_parts[bookshelf_idx + 2]
                    image_count = len(images_meta)
                    bookshelf_manager.update_chapter_image_count(book_id, chapter_id, image_count)
                    logger.info(f"已更新章节 {book_id}/{chapter_id} 图片数量为 {image_count}")
        except Exception as e:
            logger.warning(f"更新章节图片数量时出错（非致命）: {e}")
        
        return jsonify({'success': True, 'message': '保存完成'})
    else:
        return jsonify({
            'success': False,
            'error': result.get('error', '完成保存失败')
        }), 500
# === 结束分批保存 API ===


# === 新增：图片文件访问 API ===
@session_bp.route('/image/<session_name>/<filename>', methods=['GET'])
def get_session_image(session_name, filename):
    """
    获取会话中的图片文件。
    支持 PNG 图片文件和旧版 .b64 文件（自动转换为图片返回）。
    
    Args:
        session_name: 会话名称
        filename: 图片文件名 (如 image_0_original.png)
    
    Returns:
        图片文件的二进制内容
    """
    try:
        # 获取会话文件夹路径
        session_folder = session_manager._get_session_path(session_name)
        if not session_folder or not os.path.isdir(session_folder):
            logger.warning(f"图片请求失败：会话文件夹不存在 '{session_name}'")
            return jsonify({'error': '会话不存在'}), 404
        
        # 构建完整文件路径
        filepath = os.path.join(session_folder, filename)
        
        # 安全检查：确保路径在会话目录内
        real_session = os.path.realpath(session_folder)
        real_file = os.path.realpath(filepath)
        if not real_file.startswith(real_session):
            logger.error(f"安全检查失败：文件路径 {filename} 超出会话目录范围")
            return jsonify({'error': '无效的文件路径'}), 403
        
        # 如果请求的是 PNG 文件
        if filename.endswith('.png'):
            if os.path.exists(filepath):
                return send_file(filepath, mimetype='image/png')
            else:
                # 尝试查找对应的 .b64 文件
                legacy_filepath = filepath.replace('.png', '.b64')
                if os.path.exists(legacy_filepath):
                    # 读取 .b64 文件并转换为图片返回
                    with open(legacy_filepath, 'r', encoding='utf-8') as f:
                        base64_data = f.read()
                    image_bytes = base64.b64decode(base64_data)
                    return Response(image_bytes, mimetype='image/png')
        
        # 如果请求的是 .b64 文件（向后兼容）
        elif filename.endswith('.b64'):
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    base64_data = f.read()
                image_bytes = base64.b64decode(base64_data)
                return Response(image_bytes, mimetype='image/png')
        
        logger.warning(f"图片文件不存在: {filepath}")
        return jsonify({'error': '图片不存在'}), 404
        
    except Exception as e:
        logger.error(f"获取会话图片时发生错误: {e}", exc_info=True)
        return jsonify({'error': '获取图片失败'}), 500


@session_bp.route('/image_by_path/<path:session_path>/<filename>', methods=['GET'])
def get_session_image_by_path(session_path, filename):
    """
    按路径获取会话中的图片文件（用于书籍/章节）。
    
    Args:
        session_path: 会话路径（相对于 data/sessions）
        filename: 图片文件名
    
    Returns:
        图片文件的二进制内容
    """
    try:
        # 获取基础目录
        base_dir = session_manager._get_session_base_dir()
        full_path = os.path.join(base_dir, session_path)
        
        # 安全检查
        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(full_path)
        if not real_path.startswith(real_base):
            logger.error(f"安全检查失败：路径 {session_path} 超出会话目录范围")
            return jsonify({'error': '无效的路径'}), 403
        
        if not os.path.isdir(full_path):
            return jsonify({'error': '会话路径不存在'}), 404
        
        filepath = os.path.join(full_path, filename)
        
        # 再次安全检查
        real_file = os.path.realpath(filepath)
        if not real_file.startswith(real_path):
            return jsonify({'error': '无效的文件路径'}), 403
        
        # PNG 文件
        if filename.endswith('.png'):
            if os.path.exists(filepath):
                return send_file(filepath, mimetype='image/png')
            else:
                legacy_filepath = filepath.replace('.png', '.b64')
                if os.path.exists(legacy_filepath):
                    with open(legacy_filepath, 'r', encoding='utf-8') as f:
                        base64_data = f.read()
                    image_bytes = base64.b64decode(base64_data)
                    return Response(image_bytes, mimetype='image/png')
        
        # .b64 文件
        elif filename.endswith('.b64'):
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    base64_data = f.read()
                image_bytes = base64.b64decode(base64_data)
                return Response(image_bytes, mimetype='image/png')
        
        return jsonify({'error': '图片不存在'}), 404
        
    except Exception as e:
        logger.error(f"按路径获取图片时发生错误: {e}", exc_info=True)
        return jsonify({'error': '获取图片失败'}), 500
# === 结束新增 ===