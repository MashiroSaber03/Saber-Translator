"""
Manga Insight 异步辅助工具

统一的异步操作辅助函数，避免在多个路由文件中重复定义。
"""

import asyncio
import threading
from functools import wraps
from typing import Coroutine, Any, TypeVar

T = TypeVar('T')


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    在同步上下文中运行异步协程

    Args:
        coro: 异步协程

    Returns:
        协程返回值
    """
    try:
        # 当前线程存在“正在运行”的事件循环时，不能直接 run_until_complete。
        asyncio.get_running_loop()
    except RuntimeError:
        # 同步上下文：直接创建并运行临时事件循环。
        return asyncio.run(coro)

    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def _run_in_thread() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - 异常透传
            error["value"] = exc

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def async_route(f):
    """
    Flask 路由装饰器，将异步函数转换为同步

    使用方式:
        @bp.route('/api/example')
        @async_route
        async def example_handler():
            result = await some_async_operation()
            return jsonify(result)
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        return run_async(f(*args, **kwargs))
    return wrapper
