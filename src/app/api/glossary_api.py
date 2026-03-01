"""
术语库与翻译记忆 API
"""

import logging
from flask import Blueprint, jsonify, request

from src.core import glossary_manager
from src.shared.security import validate_safe_id

logger = logging.getLogger("GlossaryAPI")

glossary_bp = Blueprint("glossary_api", __name__, url_prefix="/api/glossary")


def _to_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


@glossary_bp.route("", methods=["GET"])
def get_glossary_entries():
    """查询术语/记忆条目。"""
    try:
        book_id = request.args.get("book_id")
        include_global = _to_bool(request.args.get("include_global"), True)
        query = request.args.get("query")
        entry_type = request.args.get("entry_type")

        entries = glossary_manager.list_entries(
            book_id=book_id,
            include_global=include_global,
            query=query,
            entry_type=entry_type,
        )
        return jsonify({"success": True, "entries": entries, "total": len(entries)})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"查询术语库失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@glossary_bp.route("", methods=["POST"])
def create_glossary_entry():
    """新增条目。"""
    try:
        payload = request.get_json(silent=True) or {}
        entry = glossary_manager.create_entry(payload)
        return jsonify({"success": True, "entry": entry})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"创建术语条目失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@glossary_bp.route("/<entry_id>", methods=["PUT"])
def update_glossary_entry(entry_id: str):
    """更新条目。"""
    try:
        if not validate_safe_id(entry_id):
            return jsonify({"success": False, "error": "无效的条目 ID"}), 400
        payload = request.get_json(silent=True) or {}
        updated = glossary_manager.update_entry(entry_id, payload)
        if not updated:
            return jsonify({"success": False, "error": "条目不存在"}), 404
        return jsonify({"success": True, "entry": updated})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"更新术语条目失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@glossary_bp.route("/<entry_id>", methods=["DELETE"])
def delete_glossary_entry(entry_id: str):
    """删除条目。"""
    try:
        if not validate_safe_id(entry_id):
            return jsonify({"success": False, "error": "无效的条目 ID"}), 400
        ok = glossary_manager.delete_entry(entry_id)
        if not ok:
            return jsonify({"success": False, "error": "条目不存在"}), 404
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"删除术语条目失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

