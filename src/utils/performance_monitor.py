"""
性能监控工具

用于监控关键操作的执行时间和性能指标。
"""

import time
import logging
import functools
from typing import Callable, Any, Dict
from contextlib import contextmanager


logger = logging.getLogger("PerformanceMonitor")


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record(self, operation: str, duration: float, metadata: dict = None):
        """
        记录一次操作的性能数据
        
        Args:
            operation: 操作名称
            duration: 执行时长（秒）
            metadata: 额外的元数据
        """
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        record = {
            'duration': duration,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.metrics[operation].append(record)
        
        # 记录日志
        logger.debug(f"性能记录: {operation} - {duration:.3f}s")
    
    def get_stats(self, operation: str) -> dict:
        """
        获取某个操作的统计信息
        
        Args:
            operation: 操作名称
            
        Returns:
            统计信息字典（平均值、最小值、最大值等）
        """
        if operation not in self.metrics or len(self.metrics[operation]) == 0:
            return {}
        
        durations = [r['duration'] for r in self.metrics[operation]]
        
        return {
            'count': len(durations),
            'total': sum(durations),
            'average': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
        }
    
    def get_all_stats(self) -> dict:
        """获取所有操作的统计信息"""
        return {
            operation: self.get_stats(operation)
            for operation in self.metrics.keys()
        }
    
    def clear(self):
        """清除所有性能数据"""
        self.metrics.clear()
        logger.info("性能监控数据已清除")


# 全局性能监控器实例
_global_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    return _global_monitor


@contextmanager
def monitor_performance(operation: str, metadata: dict = None):
    """
    上下文管理器：监控代码块的执行时间
    
    使用示例:
        with monitor_performance('detection'):
            result = detect_bubbles(image)
    
    Args:
        operation: 操作名称
        metadata: 额外的元数据
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        _global_monitor.record(operation, duration, metadata)


def performance_tracker(operation: str = None):
    """
    装饰器：自动监控函数的执行时间
    
    使用示例:
        @performance_tracker('translation')
        def translate_text(text):
            ...
    
    Args:
        operation: 操作名称（如果为None，使用函数名）
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
                _global_monitor.record(op_name, duration)
        
        return wrapper
    return decorator


def log_performance_stats():
    """记录所有性能统计信息到日志"""
    stats = _global_monitor.get_all_stats()
    
    if not stats:
        logger.info("暂无性能数据")
        return
    
    logger.info("=== 性能统计报告 ===")
    for operation, stat in stats.items():
        logger.info(
            f"{operation}: "
            f"调用{stat['count']}次, "
            f"平均{stat['average']:.3f}s, "
            f"最小{stat['min']:.3f}s, "
            f"最大{stat['max']:.3f}s, "
            f"总计{stat['total']:.3f}s"
        )
    logger.info("=" * 50)


# 使用示例
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 示例1：使用上下文管理器
    with monitor_performance('test_operation'):
        time.sleep(0.1)
    
    # 示例2：使用装饰器
    @performance_tracker('sample_func')
    def sample_function():
        time.sleep(0.05)
    
    sample_function()
    sample_function()
    
    # 查看统计
    log_performance_stats()
