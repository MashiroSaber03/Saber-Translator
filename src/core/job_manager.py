"""
任务中心管理器

用于质量分析、对比运行等异步/可重试任务的统一记录。
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.shared.path_helpers import resource_path
from src.shared.security import validate_safe_id

logger = logging.getLogger("JobManager")

_LOCK = threading.Lock()
_JOBS_FILE = resource_path(os.path.join("data", "jobs", "jobs.json"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_store() -> None:
    os.makedirs(os.path.dirname(_JOBS_FILE), exist_ok=True)


def _load_store() -> Dict[str, Any]:
    _ensure_store()
    if not os.path.exists(_JOBS_FILE):
        return {"jobs": []}
    try:
        with open(_JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"jobs": []}
        jobs = data.get("jobs", [])
        if not isinstance(jobs, list):
            jobs = []
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"加载任务中心数据失败: {e}", exc_info=True)
        return {"jobs": []}


def _save_store(store: Dict[str, Any]) -> bool:
    try:
        _ensure_store()
        tmp_file = _JOBS_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, _JOBS_FILE)
        return True
    except Exception as e:
        logger.error(f"保存任务中心数据失败: {e}", exc_info=True)
        return False


def create_job(
    job_type: str,
    title: str,
    payload: Optional[Dict[str, Any]] = None,
    status: str = "pending",
) -> Dict[str, Any]:
    """创建任务。"""
    now = _now_iso()
    job = {
        "id": f"job_{uuid.uuid4().hex[:12]}",
        "type": str(job_type or "generic"),
        "title": str(title or "任务"),
        "status": status,
        "created_at": now,
        "updated_at": now,
        "started_at": now if status == "running" else None,
        "finished_at": None,
        "retry_count": 0,
        "payload": payload or {},
        "result": None,
        "error": None,
    }
    with _LOCK:
        store = _load_store()
        jobs = store.get("jobs", [])
        jobs.append(job)
        store["jobs"] = jobs
        if not _save_store(store):
            raise RuntimeError("保存任务失败")
    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """查询任务。"""
    if not validate_safe_id(job_id):
        raise ValueError("无效的 job_id")
    with _LOCK:
        store = _load_store()
        for item in store.get("jobs", []):
            if isinstance(item, dict) and item.get("id") == job_id:
                return item
    return None


def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """列出任务。"""
    if limit <= 0:
        limit = 200
    limit = min(limit, 1000)

    with _LOCK:
        store = _load_store()
        jobs = list(store.get("jobs", []))

    filtered: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if status and job.get("status") != status:
            continue
        if job_type and job.get("type") != job_type:
            continue
        filtered.append(job)

    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return filtered[:limit]


def _update_job(job_id: str, updater) -> Optional[Dict[str, Any]]:
    if not validate_safe_id(job_id):
        raise ValueError("无效的 job_id")

    with _LOCK:
        store = _load_store()
        jobs = store.get("jobs", [])
        for i, job in enumerate(jobs):
            if isinstance(job, dict) and job.get("id") == job_id:
                updated = updater(dict(job))
                updated["updated_at"] = _now_iso()
                jobs[i] = updated
                store["jobs"] = jobs
                if not _save_store(store):
                    raise RuntimeError("保存任务失败")
                return updated
    return None


def mark_running(job_id: str) -> Optional[Dict[str, Any]]:
    """标记任务为运行中。"""
    def updater(job: Dict[str, Any]) -> Dict[str, Any]:
        job["status"] = "running"
        if not job.get("started_at"):
            job["started_at"] = _now_iso()
        job["error"] = None
        return job

    return _update_job(job_id, updater)


def mark_completed(job_id: str, result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """标记任务完成。"""
    def updater(job: Dict[str, Any]) -> Dict[str, Any]:
        job["status"] = "completed"
        job["result"] = result or {}
        job["error"] = None
        job["finished_at"] = _now_iso()
        return job

    return _update_job(job_id, updater)


def mark_failed(job_id: str, error: str) -> Optional[Dict[str, Any]]:
    """标记任务失败。"""
    def updater(job: Dict[str, Any]) -> Dict[str, Any]:
        job["status"] = "failed"
        job["error"] = error
        job["finished_at"] = _now_iso()
        return job

    return _update_job(job_id, updater)


def mark_cancelled(job_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
    """标记任务取消。"""
    def updater(job: Dict[str, Any]) -> Dict[str, Any]:
        if job.get("status") == "completed":
            raise ValueError("已完成任务不可取消")
        job["status"] = "cancelled"
        job["error"] = reason or "任务已取消"
        job["finished_at"] = _now_iso()
        return job

    return _update_job(job_id, updater)


def prepare_retry(job_id: str) -> Optional[Dict[str, Any]]:
    """准备重试：状态置为 running，重置错误并增加 retry_count。"""
    def updater(job: Dict[str, Any]) -> Dict[str, Any]:
        job["status"] = "running"
        job["retry_count"] = int(job.get("retry_count") or 0) + 1
        job["started_at"] = _now_iso()
        job["finished_at"] = None
        job["error"] = None
        return job

    return _update_job(job_id, updater)

