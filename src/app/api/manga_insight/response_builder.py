"""
Manga Insight API 响应构建器

统一的 API 响应格式，确保所有端点返回一致的结构。
"""

from typing import Any, Optional, Dict, List
from flask import jsonify


def success_response(
    data: Any = None,
    message: Optional[str] = None,
    **extras
):
    """
    构建成功响应

    Args:
        data: 响应数据（字典会直接展开到根级别，保持 API 兼容性）
        message: 可选消息
        **extras: 额外字段

    Returns:
        Flask Response 对象

    Example:
        return success_response(data={"id": 1}, message="创建成功")
        # {"success": true, "id": 1, "message": "创建成功"}
    """
    response = {"success": True}

    # 字典类型直接展开到根级别（保持原有 API 契约）
    if isinstance(data, dict):
        response.update(data)
    elif data is not None:
        response["data"] = data

    if message:
        response["message"] = message

    response.update(extras)

    return jsonify(response)


def error_response(
    error: str,
    status_code: int = 500,
    error_code: Optional[str] = None,
    **extras
):
    """
    构建错误响应

    Args:
        error: 错误消息
        status_code: HTTP 状态码
        error_code: 可选的错误代码（用于前端识别）
        **extras: 额外字段

    Returns:
        Tuple[Flask Response, int]

    Example:
        return error_response("未找到书籍", 404, error_code="BOOK_NOT_FOUND")
    """
    response = {
        "success": False,
        "error": error
    }

    if error_code:
        response["error_code"] = error_code

    response.update(extras)

    return jsonify(response), status_code


def paginated_response(
    items: List[Any],
    total: int,
    page: int,
    per_page: int,
    **extras
):
    """
    构建分页响应

    Args:
        items: 当前页的数据项
        total: 总数量
        page: 当前页码（1-based）
        per_page: 每页数量
        **extras: 额外字段

    Returns:
        Flask Response 对象

    Example:
        return paginated_response(
            items=[{"id": 1}, {"id": 2}],
            total=100,
            page=1,
            per_page=20
        )
    """
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0

    response = {
        "success": True,
        "data": items,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

    response.update(extras)

    return jsonify(response)


def task_response(
    task_id: str,
    status: str,
    progress: Optional[Dict] = None,
    message: Optional[str] = None,
    **extras
):
    """
    构建任务状态响应

    Args:
        task_id: 任务 ID
        status: 任务状态
        progress: 进度信息
        message: 可选消息
        **extras: 额外字段

    Returns:
        Flask Response 对象

    Example:
        return task_response(
            task_id="task_123",
            status="running",
            progress={"current": 50, "total": 100}
        )
    """
    response = {
        "success": True,
        "task_id": task_id,
        "status": status
    }

    if progress:
        response["progress"] = progress

    if message:
        response["message"] = message

    response.update(extras)

    return jsonify(response)


def analysis_status_response(
    book_id: str,
    analyzed: bool,
    analyzed_pages_count: int,
    total_pages: int,
    has_overview: bool = False,
    current_task: Optional[Dict] = None,
    **extras
):
    """
    构建分析状态响应（专用）

    Args:
        book_id: 书籍 ID
        analyzed: 是否已分析
        analyzed_pages_count: 已分析页数
        total_pages: 总页数
        has_overview: 是否有概览
        current_task: 当前任务信息
        **extras: 额外字段

    Returns:
        Flask Response 对象
    """
    response = {
        "success": True,
        "book_id": book_id,
        "analyzed": analyzed,
        "analyzed_pages_count": analyzed_pages_count,
        "analyzed_pages": analyzed_pages_count,  # 别名，保持兼容
        "total_pages": total_pages,
        "has_overview": has_overview
    }

    if current_task:
        response["current_task"] = current_task
        response["task"] = current_task  # 别名，保持兼容

    response.update(extras)

    return jsonify(response)
