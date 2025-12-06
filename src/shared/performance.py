"""
性能监控模块

提供性能追踪、计时、统计等功能
"""

import time
import logging
import functools
from typing import Callable, Any, Dict
from contextlib import contextmanager

logger = logging.getLogger("Performance")


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self.metrics: Dict[str, list] = {}
        self.enabled = True
    
    def record(self, operation: str, duration: float):
        """
        记录操作耗时
        
        Args:
            operation: 操作名称
            duration: 耗时（秒）
        """
        if not self.enabled:
            return
            
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        logger.debug(f"操作 '{operation}' 耗时: {duration:.3f}s")
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        获取操作的统计信息
        
        Args:
            operation: 操作名称
            
        Returns:
            包含 min, max, avg, count 的字典
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = self.metrics[operation]
        return {
            'count': len(durations),
            'min': min(durations),
            'max': max(durations),
            'avg': sum(durations) / len(durations),
            'total': sum(durations)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有操作的统计信息
        
        Returns:
            {operation: stats} 字典
        """
        return {op: self.get_stats(op) for op in self.metrics.keys()}
    
    def reset(self):
        """重置所有统计数据"""
        self.metrics.clear()
        logger.info("性能统计数据已重置")
    
    def log_summary(self):
        """记录性能摘要到日志"""
        if not self.metrics:
            logger.info("无性能数据")
            return
        
        logger.info("="*50)
        logger.info("性能统计摘要")
        logger.info("="*50)
        
        for operation, stats in self.get_all_stats().items():
            logger.info(
                f"{operation}: "
                f"次数={stats['count']}, "
                f"平均={stats['avg']:.3f}s, "
                f"最小={stats['min']:.3f}s, "
                f"最大={stats['max']:.3f}s, "
                f"总计={stats['total']:.3f}s"
            )
        
        logger.info("="*50)


# 全局性能监控器实例
_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    return _monitor


@contextmanager
def track_time(operation: str):
    """
    性能追踪上下文管理器
    
    使用示例:
        with track_time("image_translation"):
            translate_image(...)
    
    Args:
        operation: 操作名称
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        _monitor.record(operation, duration)


def timeit(operation: str = None):
    """
    性能追踪装饰器
    
    使用示例:
        @timeit("bubble_detection")
        def detect_bubbles(...):
            ...
    
    Args:
        operation: 操作名称，默认使用函数名
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                _monitor.record(op_name, duration)
        
        return wrapper
    return decorator


class RequestTimer:
    """Flask请求计时器"""
    
    @staticmethod
    def init_app(app):
        """
        初始化Flask应用的请求计时
        
        Args:
            app: Flask应用实例
        """
        @app.before_request
        def before_request():
            from flask import g
            g.start_time = time.time()
        
        @app.after_request
        def after_request(response):
            from flask import g, request
            
            if hasattr(g, 'start_time'):
                duration = time.time() - g.start_time
                
                # 记录慢请求（超过1秒）
                if duration > 1.0:
                    logger.warning(
                        f"慢请求: {request.method} {request.path} "
                        f"耗时 {duration:.3f}s"
                    )
                else:
                    logger.debug(
                        f"请求: {request.method} {request.path} "
                        f"耗时 {duration:.3f}s"
                    )
                
                # 添加响应头
                response.headers['X-Response-Time'] = f"{duration:.3f}s"
            
            return response
        
        logger.info("Flask请求计时器已初始化")
