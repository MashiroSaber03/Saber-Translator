"""
Manga Insight 角色卡工坊 API
"""

import io
import logging
from flask import request, send_file

from . import manga_insight_bp
from .async_helpers import run_async
from .response_builder import error_response, success_response
from src.core.manga_insight.character_cards import CharacterCardGenerator
from src.core.manga_insight.storage import AnalysisStorage

logger = logging.getLogger("MangaInsight.API.CharacterCards")


@manga_insight_bp.route('/<book_id>/character-cards/candidates', methods=['GET'])
def get_character_card_candidates(book_id: str):
    """获取角色卡候选角色。"""
    try:
        generator = CharacterCardGenerator(book_id)
        result = run_async(generator.get_candidates())
        return success_response(data=result)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"获取角色卡候选失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/character-cards/generate', methods=['POST'])
def generate_character_cards(book_id: str):
    """生成角色卡草稿。"""
    try:
        data = request.get_json() or {}
        character_names = data.get("character_names", [])
        style = data.get("style", "balanced")

        if not isinstance(character_names, list) or len(character_names) == 0:
            return error_response("character_names 不能为空", 400)
        invalid = [name for name in character_names if not isinstance(name, str) or not name.strip()]
        if invalid:
            return error_response("character_names 必须为非空字符串数组", 400)
        character_names = [name.strip() for name in character_names]

        generator = CharacterCardGenerator(book_id)
        draft = run_async(generator.generate_drafts(character_names, style=style))
        return success_response(
            data=draft,
            message=f"已生成 {len(draft.get('cards', []))} 个角色卡草稿"
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"生成角色卡草稿失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/character-cards/draft', methods=['GET'])
def get_character_card_draft(book_id: str):
    """获取角色卡草稿。"""
    try:
        storage = AnalysisStorage(book_id)
        draft = run_async(storage.load_character_card_draft())
        if not draft:
            return success_response(
                data={"draft": None, "has_data": False},
                message="暂无角色卡草稿"
            )
        return success_response(data={"draft": draft, "has_data": True})
    except Exception as e:
        logger.error(f"获取角色卡草稿失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/character-cards/draft', methods=['PUT'])
def save_character_card_draft(book_id: str):
    """保存角色卡草稿。"""
    try:
        data = request.get_json() or {}
        draft = data.get("draft") if "draft" in data else data
        if not isinstance(draft, dict):
            return error_response("draft 格式错误", 400)

        storage = AnalysisStorage(book_id)
        ok = run_async(storage.save_character_card_draft(draft))
        if not ok:
            return error_response("保存失败", 500)
        return success_response(message="草稿已保存")
    except Exception as e:
        logger.error(f"保存角色卡草稿失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/character-cards/compile', methods=['POST'])
def compile_character_cards(book_id: str):
    """编译并校验角色卡。"""
    try:
        data = request.get_json() or {}
        character_names = data.get("character_names")
        draft = data.get("draft")

        if character_names is not None:
            if not isinstance(character_names, list):
                return error_response("character_names 必须为数组", 400)
            invalid = [name for name in character_names if not isinstance(name, str) or not name.strip()]
            if invalid:
                return error_response("character_names 必须为非空字符串数组", 400)
            character_names = [name.strip() for name in character_names]

        if draft is None:
            storage = AnalysisStorage(book_id)
            draft = run_async(storage.load_character_card_draft())
        elif not isinstance(draft, dict):
            return error_response("draft 必须为对象", 400)
        if not draft:
            return error_response("未找到可编译草稿，请先生成草稿", 400)

        generator = CharacterCardGenerator(book_id)
        result = run_async(generator.compile_cards(draft, character_names))
        message = "编译完成" if result.get("valid") else "编译完成，但存在错误"
        return success_response(data=result, message=message)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"编译角色卡失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/character-cards/export/png', methods=['GET'])
def export_character_card_png(book_id: str):
    """导出单角色 PNG 卡。"""
    try:
        character = (request.args.get("character") or "").strip()
        if not character:
            return error_response("缺少 character 参数", 400)

        generator = CharacterCardGenerator(book_id)
        result = run_async(generator.export_png(character))
        filename = f"{AnalysisStorage.safe_card_filename(character)}.png"

        return send_file(
            io.BytesIO(result["png_bytes"]),
            mimetype="image/png",
            as_attachment=True,
            download_name=filename,
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"导出角色卡 PNG 失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/character-cards/export/batch', methods=['POST'])
def export_character_cards_batch(book_id: str):
    """批量导出角色卡 PNG（ZIP）。"""
    try:
        data = request.get_json() or {}
        character_names = data.get("character_names", [])
        if not isinstance(character_names, list) or len(character_names) == 0:
            return error_response("character_names 不能为空", 400)
        invalid = [name for name in character_names if not isinstance(name, str) or not name.strip()]
        if invalid:
            return error_response("character_names 必须为非空字符串数组", 400)
        character_names = [name.strip() for name in character_names]

        generator = CharacterCardGenerator(book_id)
        result = run_async(generator.export_batch_zip(character_names))
        filename = f"{book_id}_character_cards.zip"

        return send_file(
            io.BytesIO(result["zip_bytes"]),
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
        )
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"批量导出角色卡失败: {e}", exc_info=True)
        return error_response(str(e), 500)


@manga_insight_bp.route('/<book_id>/character-cards/compat', methods=['GET'])
def get_character_card_compat_report(book_id: str):
    """获取单角色兼容性诊断报告。"""
    try:
        character = (request.args.get("character") or "").strip()
        if not character:
            return error_response("缺少 character 参数", 400)

        generator = CharacterCardGenerator(book_id)
        report = run_async(generator.get_compat_report(character))
        return success_response(data=report)
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"获取兼容性报告失败: {e}", exc_info=True)
        return error_response(str(e), 500)
