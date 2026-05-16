"""
Character Studio 2.0 API routes.
"""

from __future__ import annotations

import io
import json
import logging
import os
from queue import Empty, Queue
from threading import Thread
from flask import Response, request, send_file, stream_with_context

from . import manga_insight_bp
from .async_helpers import run_async
from .response_builder import error_response, success_response
from src.core.manga_insight.character_studio import CharacterStudioService

logger = logging.getLogger("MangaInsight.API.CharacterStudio")


def _service(book_id: str) -> CharacterStudioService:
    return CharacterStudioService(book_id)


def _uploaded_chat_attachments() -> list[dict]:
    attachments: list[dict] = []
    for storage in request.files.values():
        filename = storage.filename or "attachment.bin"
        attachments.append({
            "filename": filename,
            "mime_type": storage.mimetype or "application/octet-stream",
            "bytes": storage.read(),
        })
    return attachments


def _sse_event(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _safe_chat_attachment_path(book_id: str, doc_id: str, asset_path: str) -> str | None:
    if not asset_path:
        return None
    service = _service(book_id)
    chat_assets_root = os.path.abspath(os.path.join(
        service.store.base_path,
        "character_studio",
        "assets",
        "chat",
        doc_id,
    ))
    candidate = os.path.abspath(asset_path)
    if not candidate.startswith(chat_assets_root + os.sep):
        return None
    return candidate


def _stream_chat_worker(
    *,
    logger_message: str,
    work,
) -> Response:
    events: Queue = Queue()

    def emit(event_payload: dict) -> None:
        events.put(event_payload)

    def worker() -> None:
        try:
            result = work(emit)
            events.put({"type": "state", "session": result["session"]})
        except Exception as exc:
            logger.error(logger_message, exc, exc_info=True)
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(None)

    def generate():
        thread = Thread(target=worker, daemon=True)
        thread.start()
        while True:
            try:
                item = events.get(timeout=30)
            except Empty:
                yield _sse_event("heartbeat", {"ok": True})
                continue
            if item is None:
                break
            event_type = item.get("type", "message")
            payload = {key: value for key, value in item.items() if key != "type"}
            yield _sse_event(event_type, payload)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


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
        return success_response(data={"document": document})
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


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat", methods=["GET"])
def get_character_studio_chat_state(book_id: str, doc_id: str):
    try:
        state = run_async(_service(book_id).get_chat_state(doc_id))
        return success_response(data=state)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("获取角色聊天状态失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/sessions", methods=["POST"])
def create_character_studio_chat_session(book_id: str, doc_id: str):
    try:
        data = request.get_json() or {}
        state = run_async(
            _service(book_id).create_new_chat_session(
                doc_id,
                greeting_id=data.get("greeting_id"),
            )
        )
        return success_response(data=state, message="聊天会话已创建")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("创建角色聊天会话失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/sessions/<session_id>/activate", methods=["POST"])
def switch_character_studio_chat_session(book_id: str, doc_id: str, session_id: str):
    try:
        state = run_async(_service(book_id).switch_chat_session(doc_id, session_id))
        return success_response(data=state, message="聊天会话已切换")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("切换角色聊天会话失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/messages/stream", methods=["POST"])
def stream_character_studio_chat_message(book_id: str, doc_id: str):
    data = request.form.to_dict() if request.form else (request.get_json(silent=True) or {})
    content = str(data.get("content", "") or "")
    session_id = str(data.get("session_id", "") or "")
    attachments = _uploaded_chat_attachments()
    service = _service(book_id)
    return _stream_chat_worker(
        logger_message="角色聊天消息流式处理失败: %s",
        work=lambda emit: run_async(
            service.send_chat_message(
                doc_id,
                session_id=session_id,
                content=content,
                attachments=attachments,
                on_event=emit,
            )
        ),
    )


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/messages/<message_id>", methods=["PUT"])
def edit_character_studio_chat_message(book_id: str, doc_id: str, message_id: str):
    try:
        data = request.get_json() or {}
        result = run_async(
            _service(book_id).edit_chat_message(
                doc_id,
                session_id=data.get("session_id", ""),
                message_id=message_id,
                new_content=data.get("content", ""),
            )
        )
        return success_response(data=result, message="消息已编辑")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("编辑角色聊天消息失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/messages/<message_id>", methods=["DELETE"])
def delete_character_studio_chat_message(book_id: str, doc_id: str, message_id: str):
    try:
        session_id = request.args.get("session_id", "")
        result = run_async(
            _service(book_id).delete_chat_message(
                doc_id,
                session_id=session_id,
                message_id=message_id,
            )
        )
        return success_response(data=result, message="消息已删除")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("删除角色聊天消息失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/messages/<message_id>/regenerate/stream", methods=["POST"])
def regenerate_character_studio_chat_message(book_id: str, doc_id: str, message_id: str):
    data = request.get_json() or {}
    session_id = str(data.get("session_id", "") or "")
    service = _service(book_id)
    return _stream_chat_worker(
        logger_message="重生角色聊天消息失败: %s",
        work=lambda emit: run_async(
            service.regenerate_chat_message(
                doc_id,
                session_id=session_id,
                anchor_message_id=message_id,
                on_event=emit,
            )
        ),
    )


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/summary", methods=["POST"])
def summarize_character_studio_chat(book_id: str, doc_id: str):
    try:
        data = request.get_json() or {}
        result = run_async(
            _service(book_id).summarize_chat_session(
                doc_id,
                session_id=data.get("session_id", ""),
                cutoff_message_id=data.get("cutoff_message_id"),
            )
        )
        return success_response(data=result, message="聊天摘要已生成")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("总结角色聊天失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/export", methods=["GET"])
def export_character_studio_chat(book_id: str, doc_id: str):
    try:
        session_id = request.args.get("session_id", "")
        data = run_async(_service(book_id).export_chat_session(doc_id, session_id=session_id))
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        return send_file(
            io.BytesIO(payload),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"{doc_id}.{session_id or 'chat'}.json",
        )
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("导出角色聊天失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/import", methods=["POST"])
def import_character_studio_chat(book_id: str, doc_id: str):
    try:
        if request.files:
            file = next(iter(request.files.values()))
            payload = json.loads(file.read().decode("utf-8"))
        else:
            payload = request.get_json() or {}
        state = run_async(_service(book_id).import_chat_session(doc_id, payload))
        return success_response(data=state, message="聊天记录已导入")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("导入角色聊天失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/prompt-preview", methods=["GET"])
def get_character_studio_chat_prompt_preview(book_id: str, doc_id: str):
    try:
        session_id = request.args.get("session_id", "")
        payload = run_async(_service(book_id).get_chat_prompt_preview(doc_id, session_id=session_id))
        return success_response(data=payload)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        logger.error("获取角色聊天提示词预览失败: %s", exc, exc_info=True)
        return error_response(str(exc), 500)


@manga_insight_bp.route("/<book_id>/character-studio/documents/<doc_id>/chat/attachment", methods=["GET"])
def get_character_studio_chat_attachment(book_id: str, doc_id: str):
    try:
        asset_path = _safe_chat_attachment_path(book_id, doc_id, request.args.get("asset_path", ""))
        if not asset_path or not os.path.exists(asset_path):
            return error_response("聊天附件不存在", 404)
        ext = os.path.splitext(asset_path)[1].lower()
        mimetype = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".png": "image/png",
        }.get(ext, "application/octet-stream")
        return send_file(asset_path, mimetype=mimetype, as_attachment=False)
    except Exception as exc:
        logger.error("获取角色聊天附件失败: %s", exc, exc_info=True)
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
