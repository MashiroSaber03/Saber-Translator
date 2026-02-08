"""
Manga Insight 异步辅助工具

统一的异步操作辅助函数，避免在多个路由文件中重复定义。
"""

import asyncio
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
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


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
