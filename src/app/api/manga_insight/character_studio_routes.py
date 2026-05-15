"""
Character Studio 2.0 API routes.
"""

from __future__ import annotations

import io
import json
import logging
import os
from flask import request, send_file

from . import manga_insight_bp
from .async_helpers import run_async
from .response_builder import error_response, success_response
from src.core.manga_insight.character_studio import CharacterStudioService

logger = logging.getLogger("MangaInsight.API.CharacterStudio")


def _service(book_id: str) -> CharacterStudioService:
    return CharacterStudioService(book_id)


@manga_insight_bp.route("/<book_id>/character-studio/index", methods=["GET"])
def get_character_studio_index(book_id: str):
    try:
        payload = run_async(_service(book_id).get_index_payload())
        return success_response(data=payload)
    except Exception as exc:
        logger.error("获取角色工作台索引失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/candidates", methods=["GET"])
def get_character_studio_candidates(book_id: str):
    try:
        payload = run_async(_service(book_id).get_candidates())
        return success_response(data=payload)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("获取角色工作台候选失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents", methods=["POST"])
def create_character_studio_document(book_id: str):
    try:
        data = request.get_json() or {}
        document = run_async(_service(book_id).create_document(
            candidate_name=data.get("candidate_name"),
            title=data.get("title"),
        ))
        return success_response(data={"document": document}, message="角色文档已创建")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("创建角色文档失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>", methods=["GET"])
def get_character_studio_document(book_id: str, doc_id: str):
    try:
        service = _service(book_id)
        document = run_async(service.store.load_document(doc_id))
        if not document:
            return error_response("文档不存在", 404)
        preview_session = run_async(service.store.load_preview_session(doc_id))
        return success_response(data={"document": document, "preview_session": preview_session})
    except Exception as exc:
        logger.error("读取角色文档失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>", methods=["PUT"])
def save_character_studio_document(book_id: str, doc_id: str):
    try:
        payload = request.get_json() or {}
        document = run_async(_service(book_id).save_document(doc_id, payload))
        return success_response(data={"document": document}, message="角色文档已保存")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("保存角色文档失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>", methods=["DELETE"])
def delete_character_studio_document(book_id: str, doc_id: str):
    try:
        deleted = run_async(_service(book_id).store.delete_document(doc_id))
        if not deleted:
            return error_response("文档不存在", 404)
        return success_response(message="角色文档已删除")
    except Exception as exc:
        logger.error("删除角色文档失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/generate/<section>", methods=["POST"])
def generate_character_studio_section(book_id: str, doc_id: str, section: str):
    try:
        document = run_async(_service(book_id).generate_section(doc_id, section))
        return success_response(data={"document": document}, message=f"{section} 已生成")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("生成角色 section 失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/validate", methods=["POST"])
def validate_character_studio_document(book_id: str, doc_id: str):
    try:
        report = run_async(_service(book_id).validate_document(doc_id))
        return success_response(data=report, message="诊断完成")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("校验角色文档失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/preview/chat", methods=["POST"])
def preview_character_studio_chat(book_id: str, doc_id: str):
    try:
        data = request.get_json() or {}
        session = run_async(_service(book_id).preview_chat(doc_id, data.get("message", "")))
        return success_response(data=session)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("预览聊天失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/preview/reset", methods=["POST"])
def reset_character_studio_preview(book_id: str, doc_id: str):
    try:
        session = run_async(_service(book_id).preview_reset(doc_id))
        return success_response(data=session, message="预览会话已重置")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("重置预览会话失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/agent", methods=["POST"])
def run_character_studio_agent(book_id: str, doc_id: str):
    try:
        data = request.get_json() or {}
        result = run_async(_service(book_id).run_agent(doc_id, data.get("message", "")))
        return success_response(data=result)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("角色 Agent 调用失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/imports", methods=["POST"])
def import_character_studio_document(book_id: str):
    try:
        service = _service(book_id)
        if request.files:
            file = next(iter(request.files.values()))
            filename = file.filename or "imported"
            payload_bytes = file.read()
            lower_name = filename.lower()
            if lower_name.endswith(".png"):
                document = run_async(service.import_png(payload_bytes))
            elif lower_name.endswith(".json"):
                document = run_async(service.import_payload(json.loads(payload_bytes.decode("utf-8"))))
            elif any(lower_name.endswith(ext) for ext in (".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                _, ext = os.path.splitext(filename)
                document = run_async(service.import_image(title=os.path.splitext(filename)[0], extension=ext or ".png", image_bytes=payload_bytes))
            else:
                return error_response("不支持的导入文件类型", 400)
        else:
            data = request.get_json() or {}
            document = run_async(service.import_payload(data))
        return success_response(data={"document": document}, message="导入成功")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("导入角色文档失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/export", methods=["GET"])
def export_character_studio_document(book_id: str, doc_id: str):
    try:
        format_name = (request.args.get("format") or "v3").strip().lower()
        data, mimetype, filename = run_async(_service(book_id).export_document(doc_id, format_name))
        return send_file(
            io.BytesIO(data),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename,
        )
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("导出角色文档失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/worldbook/import", methods=["POST"])
def import_worldbook_into_character_studio_document(book_id: str, doc_id: str):
    try:
        if request.files:
            file = next(iter(request.files.values()))
            payload = json.loads(file.read().decode("utf-8"))
        else:
            payload = request.get_json() or {}
        document = run_async(_service(book_id).import_worldbook_into_document(doc_id, payload))
        return success_response(data={"document": document}, message="世界书已导入")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("导入世界书失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/worldbook/export", methods=["GET"])
def export_worldbook_from_character_studio_document(book_id: str, doc_id: str):
    try:
        data, mimetype, filename = run_async(_service(book_id).export_document(doc_id, "worldbook"))
        return send_file(
            io.BytesIO(data),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename,
        )
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("导出世界书失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/avatar", methods=["GET"])
def get_character_studio_avatar(book_id: str, doc_id: str):
    try:
        document = run_async(_service(book_id).store.load_document(doc_id))
        if not document:
            return error_response("文档不存在", 404)
        asset_path = document.get("avatar", {}).get("asset_path")
        if not asset_path or not os.path.exists(asset_path):
            return error_response("头像不存在", 404)
        ext = os.path.splitext(asset_path)[1].lower()
        mimetype = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(ext, "image/png")
        return send_file(asset_path, mimetype=mimetype, as_attachment=False)
    except Exception as exc:
        logger.error("获取角色头像失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)
