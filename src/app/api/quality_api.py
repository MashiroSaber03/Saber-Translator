"""
翻译质量分析 API
"""

import logging
from flask import Blueprint, jsonify, request

from src.core import job_manager, quality_manager
from src.shared.security import validate_relative_path

logger = logging.getLogger("QualityAPI")

quality_bp = Blueprint("quality_api", __name__, url_prefix="/api/quality")


@quality_bp.route("/analyze", methods=["POST"])
def analyze_quality():
    """
    触发会话质量分析。

    Body:
    {
      "session": "bookshelf/book_x/chapters/ch_1/session"
    }
    """
    payload = request.get_json(silent=True) or {}
    session_path = str(payload.get("session", "")).strip()

    ok, normalized_or_error = validate_relative_path(session_path, allow_unicode=False)
    if not ok:
        return jsonify({"success": False, "error": f"无效的 session: {normalized_or_error}"}), 400
    session_path = normalized_or_error

    job = job_manager.create_job(
        job_type="quality",
        title=f"质量分析: {session_path}",
        payload={"session": session_path},
        status="running",
    )

    try:
        report = quality_manager.analyze_session_quality(session_path)
        updated_job = job_manager.mark_completed(
            job["id"],
            {
                "session": session_path,
                "issue_count": report.get("issue_count", 0),
            },
        )
        return jsonify({"success": True, "job": updated_job or job, "report": report})
    except FileNotFoundError as e:
        job_manager.mark_failed(job["id"], str(e))
        return jsonify({"success": False, "error": str(e), "job_id": job["id"]}), 404
    except Exception as e:
        logger.error(f"质量分析失败: {e}", exc_info=True)
        job_manager.mark_failed(job["id"], str(e))
        return jsonify({"success": False, "error": str(e), "job_id": job["id"]}), 500


@quality_bp.route("/report/<path:session_path>", methods=["GET"])
def get_quality_report(session_path: str):
    """
    获取会话最新质量报告。
    """
    try:
        report = quality_manager.get_session_report(session_path)
        if not report:
            return jsonify({"success": False, "error": "报告不存在"}), 404
        return jsonify({"success": True, "report": report})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"读取质量报告失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

