"""
续写功能 - 文件服务路由

处理通用文件服务请求。
"""

import os
import logging
from flask import request, send_file

from .. import manga_insight_bp
from ..response_builder import error_response
from src.shared.path_helpers import resource_path

logger = logging.getLogger("MangaInsight.API.Continuation.File")


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
            return error_response("缺少文件路径参数", 400)

        # 规范化路径（处理混合斜杠问题）
        file_path = os.path.normpath(file_path)
        real_path = os.path.realpath(file_path)

        # 路径安全校验 - 只允许访问特定目录
        allowed_dirs = [
            resource_path("data/bookshelf"),
            resource_path("data/manga_insight"),
            resource_path("data/sessions"),
        ]

        def is_subpath(path: str, parent: str) -> bool:
            """安全检查 path 是否在 parent 目录下"""
            try:
                real_parent = os.path.realpath(parent)
                return os.path.commonpath([path, real_parent]) == real_parent
            except (ValueError, OSError):
                return False

        is_safe = any(
            is_subpath(real_path, allowed_dir)
            for allowed_dir in allowed_dirs if os.path.exists(allowed_dir)
        )

        if not is_safe:
            logger.warning(f"路径安全拒绝: {file_path}")
            return error_response("无效的文件路径", 403)

        if not os.path.exists(file_path):
            logger.warning(f"请求的文件不存在: {file_path}")
            return error_response(f"文件不存在: {file_path}", 404)

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
        return error_response(str(e), 500)
