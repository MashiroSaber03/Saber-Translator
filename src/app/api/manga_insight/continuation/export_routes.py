"""
续写功能 - 导出路由

处理图片 ZIP 和 PDF 导出请求。
"""

import os
import io
import zipfile
import logging
from flask import request, send_file

from .. import manga_insight_bp
from ..async_helpers import run_async
from ..response_builder import error_response
from src.core.manga_insight.storage import AnalysisStorage

logger = logging.getLogger("MangaInsight.API.Continuation.Export")


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
        data = request.get_json() or {}
        pages_data = data.get("pages", [])

        # 如果没有传入 pages，则从持久化存储加载
        if not pages_data:
            storage = AnalysisStorage(book_id)
            loaded = run_async(storage.load_continuation_pages())
            if loaded:
                pages_data = loaded.get("pages", [])

        if not pages_data:
            return error_response("没有可导出的页面数据", 400)

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
        return error_response(str(e), 500)


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

        data = request.get_json() or {}
        pages_data = data.get("pages", [])

        # 如果没有传入 pages，则从持久化存储加载
        if not pages_data:
            storage = AnalysisStorage(book_id)
            loaded = run_async(storage.load_continuation_pages())
            if loaded:
                pages_data = loaded.get("pages", [])

        if not pages_data:
            return error_response("没有可导出的页面数据", 400)

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
            return error_response("没有可导出的图片", 400)

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
        return error_response(str(e), 500)
