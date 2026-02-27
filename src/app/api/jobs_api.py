"""
任务中心 API
"""

import logging
from flask import Blueprint, jsonify, request

from src.core import compare_manager, job_manager, quality_manager
from src.shared.security import validate_safe_id

logger = logging.getLogger("JobsAPI")

jobs_bp = Blueprint("jobs_api", __name__, url_prefix="/api/jobs")


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@jobs_bp.route("", methods=["GET"])
def list_jobs():
    """列出任务。"""
    try:
        status = request.args.get("status")
        job_type = request.args.get("type")
        limit = _to_int(request.args.get("limit"), 200)
        jobs = job_manager.list_jobs(status=status, job_type=job_type, limit=limit)
        return jsonify({"success": True, "jobs": jobs, "total": len(jobs)})
    except Exception as e:
        logger.error(f"查询任务列表失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@jobs_bp.route("/<job_id>/retry", methods=["POST"])
def retry_job(job_id: str):
    """重试失败/已取消任务。"""
    if not validate_safe_id(job_id):
        return jsonify({"success": False, "error": "无效的 job_id"}), 400

    try:
        job = job_manager.get_job(job_id)
        if not job:
            return jsonify({"success": False, "error": "任务不存在"}), 404
        if job.get("status") == "running":
            return jsonify({"success": False, "error": "任务运行中，无法重试"}), 400

        retry_job_item = job_manager.prepare_retry(job_id)
        if not retry_job_item:
            return jsonify({"success": False, "error": "任务不存在"}), 404

        job_type = str(job.get("type", ""))
        payload = job.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        result_payload = None
        extra = {}
        if job_type == "quality":
            session_path = str(payload.get("session", "")).strip()
            report = quality_manager.analyze_session_quality(session_path)
            result_payload = {"session": session_path, "issue_count": report.get("issue_count", 0)}
            extra["report"] = report
        elif job_type == "compare":
            run = compare_manager.run_compare(payload)
            result_payload = {"run_id": run.get("id"), "candidate_count": len(run.get("candidates", []))}
            extra["run"] = run
        else:
            raise ValueError(f"不支持重试的任务类型: {job_type}")

        completed = job_manager.mark_completed(job_id, result=result_payload)
        return jsonify({"success": True, "job": completed or retry_job_item, **extra})
    except ValueError as e:
        job_manager.mark_failed(job_id, str(e))
        return jsonify({"success": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        job_manager.mark_failed(job_id, str(e))
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        logger.error(f"重试任务失败: {e}", exc_info=True)
        job_manager.mark_failed(job_id, str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@jobs_bp.route("/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    """取消任务。"""
    if not validate_safe_id(job_id):
        return jsonify({"success": False, "error": "无效的 job_id"}), 400

    try:
        reason = str((request.get_json(silent=True) or {}).get("reason", "")).strip()
        cancelled = job_manager.mark_cancelled(job_id, reason=reason)
        if not cancelled:
            return jsonify({"success": False, "error": "任务不存在"}), 404
        return jsonify({"success": True, "job": cancelled})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"取消任务失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

