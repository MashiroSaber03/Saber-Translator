"""
译文版本对比 API
"""

import logging
from flask import Blueprint, jsonify, request

from src.core import compare_manager, job_manager
from src.shared.security import validate_safe_id

logger = logging.getLogger("CompareAPI")

compare_bp = Blueprint("compare_api", __name__, url_prefix="/api/compare")


@compare_bp.route("/run", methods=["POST"])
def run_compare():
    """
    执行译文版本对比。
    """
    payload = request.get_json(silent=True) or {}

    job = job_manager.create_job(
        job_type="compare",
        title="译文版本对比",
        payload=payload,
        status="running",
    )

    try:
        run = compare_manager.run_compare(payload)
        updated_job = job_manager.mark_completed(
            job["id"],
            {"run_id": run.get("id"), "candidate_count": len(run.get("candidates", []))},
        )
        return jsonify({"success": True, "job": updated_job or job, "run": run})
    except ValueError as e:
        job_manager.mark_failed(job["id"], str(e))
        return jsonify({"success": False, "error": str(e), "job_id": job["id"]}), 400
    except Exception as e:
        logger.error(f"执行译文对比失败: {e}", exc_info=True)
        job_manager.mark_failed(job["id"], str(e))
        return jsonify({"success": False, "error": str(e), "job_id": job["id"]}), 500


@compare_bp.route("/<run_id>", methods=["GET"])
def get_compare_run(run_id: str):
    """
    获取对比运行结果。
    """
    if not validate_safe_id(run_id):
        return jsonify({"success": False, "error": "无效的 run_id"}), 400

    try:
        run = compare_manager.get_run(run_id)
        if not run:
            return jsonify({"success": False, "error": "对比结果不存在"}), 404
        return jsonify({"success": True, "run": run})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"读取对比结果失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

