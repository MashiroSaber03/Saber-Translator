"""
MOBI/AZW 电子书解析处理模块
提取 MOBI 文件中的图片资源，支持分批发送避免内存溢出
"""

import os
import io
import re
import base64
import tempfile
import uuid
import logging
import shutil
import time
from threading import Lock
from flask import request, jsonify
from PIL import Image

from . import system_bp

logger = logging.getLogger("MobiHandler")

# MOBI 解析会话存储 (session_id -> session_data)
_mobi_sessions = {}
_sessions_lock = Lock()

# 会话过期时间（秒）
SESSION_EXPIRE_TIME = 600  # 10分钟


def _cleanup_expired_sessions():
    """清理过期的会话"""
    current_time = time.time()
    expired_sessions = []
    
    with _sessions_lock:
        for session_id, session_data in _mobi_sessions.items():
            if current_time - session_data.get('created_at', 0) > SESSION_EXPIRE_TIME:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            session_data = _mobi_sessions.pop(session_id, None)
            if session_data and session_data.get('temp_dir'):
                try:
                    shutil.rmtree(session_data['temp_dir'], ignore_errors=True)
                except Exception as e:
                    logger.warning(f"清理过期会话目录失败: {e}")
            logger.info(f"已清理过期 MOBI 会话: {session_id}")


def _extract_mobi_to_temp(mobi_file_path: str) -> tuple:
    """
    从 MOBI 文件中提取内容到临时目录
    
    Returns:
        tuple: (temp_dir, image_files_sorted)
    """
    try:
        import mobi
    except ImportError:
        raise ImportError("请安装 mobi 库: pip install mobi")
    
    # mobi 库会解包到临时目录
    temp_dir, extracted_path = mobi.extract(mobi_file_path)
    
    logger.info(f"MOBI 解包到: {temp_dir}")
    
    # 列出解包后的目录结构（调试用）
    subdirs = []
    for item in os.listdir(temp_dir):
        item_path = os.path.join(temp_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    logger.info(f"解包目录包含子目录: {subdirs}")
    
    # 查找所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    all_images = []  # 存储 (文件名, 完整路径, 文件大小)
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(full_path)
                    if file_size > 5000:  # 过滤小于 5KB 的图片
                        all_images.append((file, full_path, file_size))
                except:
                    pass
    
    logger.info(f"找到 {len(all_images)} 张图片（去重前）")
    
    # 按文件名去重，保留文件最大的版本（通常质量更好）
    seen_names = {}  # 文件名 -> (完整路径, 文件大小)
    for filename, full_path, file_size in all_images:
        if filename not in seen_names or file_size > seen_names[filename][1]:
            seen_names[filename] = (full_path, file_size)
    
    # 提取去重后的文件路径
    image_files = [path for path, size in seen_names.values()]
    
    # 自然排序函数（正确处理数字，如 1, 2, 10 而不是 1, 10, 2）
    def natural_sort_key(path):
        filename = os.path.basename(path)
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', filename)]
    
    # 按文件名自然排序（保持页面顺序）
    image_files.sort(key=natural_sort_key)
    
    logger.info(f"去重后保留 {len(image_files)} 张有效图片")
    
    return temp_dir, image_files


def _load_single_image(img_path: str) -> dict:
    """
    加载单张图片并转换为 DataURL
    """
    try:
        with Image.open(img_path) as img:
            # 转换为 RGB（处理所有非RGB模式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转为 base64
            buffered = io.BytesIO()
            img.save(buffered, format='JPEG', quality=92)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return {
                'success': True,
                'data_url': f'data:image/jpeg;base64,{img_base64}',
                'width': img.width,
                'height': img.height,
                'filename': os.path.basename(img_path)
            }
    except Exception as e:
        logger.warning(f"处理图片 {img_path} 失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@system_bp.route('/parse_mobi_start', methods=['POST'])
def parse_mobi_start():
    """
    开始解析 MOBI 文件
    返回会话ID和总页数，不返回图片数据
    """
    # 清理过期会话
    _cleanup_expired_sessions()
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未提供文件'}), 400
    
    file = request.files['file']
    
    if not file.filename:
        return jsonify({'success': False, 'error': '文件名为空'}), 400
    
    # 检查文件扩展名
    allowed_extensions = {'.mobi', '.azw', '.azw3'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'success': False, 'error': f'不支持的文件格式: {ext}'}), 400
    
    temp_upload = None
    try:
        # 保存上传文件到临时位置
        temp_upload = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        file.save(temp_upload.name)
        temp_upload.close()
        
        logger.info(f"开始解析 MOBI 文件: {file.filename}")
        
        # 提取 MOBI 内容
        temp_dir, image_files = _extract_mobi_to_temp(temp_upload.name)
        
        if not image_files:
            # 清理
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({
                'success': False, 
                'error': 'MOBI 文件中未找到图片，可能不是漫画格式'
            }), 400
        
        # 创建会话
        session_id = str(uuid.uuid4())
        
        with _sessions_lock:
            _mobi_sessions[session_id] = {
                'temp_dir': temp_dir,
                'image_files': image_files,
                'filename': file.filename,
                'created_at': time.time()
            }
        
        logger.info(f"创建 MOBI 解析会话: {session_id}, 共 {len(image_files)} 页")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': file.filename,
            'total_pages': len(image_files)
        })
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logger.error(f"解析 MOBI 失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'解析失败: {str(e)}'}), 500
    finally:
        # 清理上传的临时文件
        if temp_upload and os.path.exists(temp_upload.name):
            try:
                os.unlink(temp_upload.name)
            except:
                pass


@system_bp.route('/parse_mobi_page/<session_id>/<int:page_index>', methods=['GET'])
def parse_mobi_page(session_id: str, page_index: int):
    """
    获取指定页的图片数据
    
    Args:
        session_id: 会话ID
        page_index: 页码索引（从0开始）
    """
    with _sessions_lock:
        session_data = _mobi_sessions.get(session_id)
    
    if not session_data:
        return jsonify({'success': False, 'error': '会话不存在或已过期'}), 404
    
    image_files = session_data.get('image_files', [])
    
    if page_index < 0 or page_index >= len(image_files):
        return jsonify({
            'success': False, 
            'error': f'页码超出范围 (0-{len(image_files)-1})'
        }), 400
    
    # 加载指定页的图片
    img_path = image_files[page_index]
    result = _load_single_image(img_path)
    
    if result['success']:
        result['page_index'] = page_index
        result['total_pages'] = len(image_files)
        return jsonify(result)
    else:
        return jsonify({
            'success': False,
            'error': result.get('error', '加载图片失败'),
            'page_index': page_index
        }), 500


@system_bp.route('/parse_mobi_batch', methods=['POST'])
def parse_mobi_batch():
    """
    批量获取多页图片数据
    
    请求体:
    {
        "session_id": "xxx",
        "start_index": 0,
        "count": 5
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '请求体为空'}), 400
    
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'error': '缺少 session_id'}), 400
    
    start_index = data.get('start_index', 0)
    count = data.get('count', 5)  # 默认每批5页
    
    # 限制每批最多10页
    count = min(count, 10)
    
    with _sessions_lock:
        session_data = _mobi_sessions.get(session_id)
    
    if not session_data:
        return jsonify({'success': False, 'error': '会话不存在或已过期'}), 404
    
    image_files = session_data.get('image_files', [])
    total_pages = len(image_files)
    
    if start_index < 0 or start_index >= total_pages:
        return jsonify({
            'success': False, 
            'error': f'起始索引超出范围 (0-{total_pages-1})'
        }), 400
    
    # 计算实际获取的页数
    end_index = min(start_index + count, total_pages)
    
    images = []
    for i in range(start_index, end_index):
        result = _load_single_image(image_files[i])
        if result['success']:
            result['page_index'] = i
            images.append(result)
        else:
            # 记录失败但继续处理
            logger.warning(f"加载第 {i} 页失败: {result.get('error')}")
    
    return jsonify({
        'success': True,
        'images': images,
        'start_index': start_index,
        'end_index': end_index,
        'total_pages': total_pages,
        'has_more': end_index < total_pages
    })


@system_bp.route('/parse_mobi_cleanup/<session_id>', methods=['POST'])
def parse_mobi_cleanup(session_id: str):
    """
    清理 MOBI 解析会话
    """
    with _sessions_lock:
        session_data = _mobi_sessions.pop(session_id, None)
    
    if session_data:
        temp_dir = session_data.get('temp_dir')
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"已清理 MOBI 会话: {session_id}")
            except Exception as e:
                logger.warning(f"清理会话目录失败: {e}")
        
        return jsonify({'success': True, 'message': '会话已清理'})
    else:
        return jsonify({'success': True, 'message': '会话不存在或已被清理'})
