"""
PDF 文件解析处理模块
使用 PyMuPDF (fitz) 将 PDF 页面渲染为图片，支持分批发送避免内存溢出
"""

import os
import io
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

logger = logging.getLogger("PdfHandler")

# PDF 解析会话存储 (session_id -> session_data)
_pdf_sessions = {}
_sessions_lock = Lock()

# 会话过期时间（秒）
SESSION_EXPIRE_TIME = 600  # 10分钟


def _cleanup_expired_sessions():
    """清理过期的会话"""
    current_time = time.time()
    expired_sessions = []
    
    with _sessions_lock:
        for session_id, session_data in _pdf_sessions.items():
            if current_time - session_data.get('created_at', 0) > SESSION_EXPIRE_TIME:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            session_data = _pdf_sessions.pop(session_id, None)
            if session_data and session_data.get('temp_file'):
                try:
                    os.unlink(session_data['temp_file'])
                except Exception as e:
                    logger.warning(f"清理过期会话临时文件失败: {e}")
            logger.info(f"已清理过期 PDF 会话: {session_id}")


def _render_pdf_page(pdf_doc, page_index: int, scale: float = 2.0) -> dict:
    """
    渲染单个 PDF 页面为图片
    
    Args:
        pdf_doc: PyMuPDF 文档对象
        page_index: 页码索引（从0开始）
        scale: 渲染缩放比例，默认2.0（获得较高清晰度）
    
    Returns:
        dict: 包含 success, data_url, width, height 等信息
    """
    try:
        # 确保 fitz 已导入
        _ensure_fitz()
        
        page = pdf_doc[page_index]
        
        # 设置渲染矩阵（缩放）
        mat = fitz.Matrix(scale, scale)
        
        # 渲染页面为 pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # 转换为 PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 转为 base64
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG', quality=92)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            'success': True,
            'data_url': f'data:image/jpeg;base64,{img_base64}',
            'width': pix.width,
            'height': pix.height,
            'page_index': page_index
        }
    except Exception as e:
        logger.warning(f"渲染 PDF 页面 {page_index} 失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'page_index': page_index
        }


# 延迟导入 fitz，避免未安装时导入失败
fitz = None

def _ensure_fitz():
    """确保 fitz (PyMuPDF) 已导入"""
    global fitz
    if fitz is None:
        try:
            import fitz as _fitz
            fitz = _fitz
        except ImportError:
            raise ImportError("请安装 PyMuPDF 库: pip install PyMuPDF")
    return fitz


@system_bp.route('/parse_pdf_start', methods=['POST'])
def parse_pdf_start():
    """
    开始解析 PDF 文件
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
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != '.pdf':
        return jsonify({'success': False, 'error': f'不支持的文件格式: {ext}'}), 400
    
    temp_file = None
    try:
        # 确保 fitz 已导入
        _ensure_fitz()
        
        # 保存上传文件到临时位置
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        file.save(temp_path)
        temp_file = temp_path
        
        logger.info(f"开始解析 PDF 文件: {file.filename}")
        
        # 打开 PDF 文件
        pdf_doc = fitz.open(temp_path)
        total_pages = len(pdf_doc)
        
        if total_pages == 0:
            pdf_doc.close()
            os.unlink(temp_path)
            return jsonify({
                'success': False, 
                'error': 'PDF 文件没有页面'
            }), 400
        
        # 创建会话
        session_id = str(uuid.uuid4())
        
        with _sessions_lock:
            _pdf_sessions[session_id] = {
                'temp_file': temp_path,
                'pdf_doc': pdf_doc,
                'filename': file.filename,
                'total_pages': total_pages,
                'created_at': time.time()
            }
        
        logger.info(f"创建 PDF 解析会话: {session_id}, 共 {total_pages} 页")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': file.filename,
            'total_pages': total_pages
        })
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logger.error(f"解析 PDF 失败: {e}", exc_info=True)
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        return jsonify({'success': False, 'error': f'解析失败: {str(e)}'}), 500


@system_bp.route('/parse_pdf_page/<session_id>/<int:page_index>', methods=['GET'])
def parse_pdf_page(session_id: str, page_index: int):
    """
    获取指定页的图片数据
    
    Args:
        session_id: 会话ID
        page_index: 页码索引（从0开始）
    """
    with _sessions_lock:
        session_data = _pdf_sessions.get(session_id)
    
    if not session_data:
        return jsonify({'success': False, 'error': '会话不存在或已过期'}), 404
    
    pdf_doc = session_data.get('pdf_doc')
    total_pages = session_data.get('total_pages', 0)
    
    if page_index < 0 or page_index >= total_pages:
        return jsonify({
            'success': False, 
            'error': f'页码超出范围 (0-{total_pages-1})'
        }), 400
    
    # 渲染指定页
    result = _render_pdf_page(pdf_doc, page_index)
    
    if result['success']:
        result['total_pages'] = total_pages
        return jsonify(result)
    else:
        return jsonify({
            'success': False,
            'error': result.get('error', '渲染页面失败'),
            'page_index': page_index
        }), 500


@system_bp.route('/parse_pdf_batch', methods=['POST'])
def parse_pdf_batch():
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
        session_data = _pdf_sessions.get(session_id)
    
    if not session_data:
        return jsonify({'success': False, 'error': '会话不存在或已过期'}), 404
    
    pdf_doc = session_data.get('pdf_doc')
    total_pages = session_data.get('total_pages', 0)
    
    if start_index < 0 or start_index >= total_pages:
        return jsonify({
            'success': False, 
            'error': f'起始索引超出范围 (0-{total_pages-1})'
        }), 400
    
    # 计算实际获取的页数
    end_index = min(start_index + count, total_pages)
    
    images = []
    for i in range(start_index, end_index):
        result = _render_pdf_page(pdf_doc, i)
        if result['success']:
            images.append(result)
        else:
            # 记录失败但继续处理
            logger.warning(f"渲染第 {i} 页失败: {result.get('error')}")
    
    return jsonify({
        'success': True,
        'images': images,
        'start_index': start_index,
        'end_index': end_index,
        'total_pages': total_pages,
        'has_more': end_index < total_pages
    })


@system_bp.route('/parse_pdf_cleanup/<session_id>', methods=['POST'])
def parse_pdf_cleanup(session_id: str):
    """
    清理 PDF 解析会话
    """
    with _sessions_lock:
        session_data = _pdf_sessions.pop(session_id, None)
    
    if session_data:
        # 关闭 PDF 文档
        pdf_doc = session_data.get('pdf_doc')
        if pdf_doc:
            try:
                pdf_doc.close()
            except:
                pass
        
        # 删除临时文件
        temp_file = session_data.get('temp_file')
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.info(f"已清理 PDF 会话: {session_id}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")
        
        return jsonify({'success': True, 'message': '会话已清理'})
    else:
        return jsonify({'success': True, 'message': '会话不存在或已被清理'})
