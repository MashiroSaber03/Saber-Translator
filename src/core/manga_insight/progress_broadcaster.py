"""
Manga Insight 进度广播

通过 WebSocket 推送分析进度。
"""

import logging
from typing import Dict, Callable, List

logger = logging.getLogger("MangaInsight.ProgressBroadcaster")


class ProgressBroadcaster:
    """
    进度广播器
    
    用于向客户端推送分析进度更新。
    支持 Flask-SocketIO 或其他 WebSocket 实现。
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.socketio = None
        self.callbacks: Dict[str, List[Callable]] = {}
        self._initialized = True
    
    def set_socketio(self, socketio):
        """
        设置 Flask-SocketIO 实例
        
        Args:
            socketio: Flask-SocketIO 实例
        """
        self.socketio = socketio
        logger.info("SocketIO 已配置")
    
    def broadcast_progress(self, task_id: str, progress: Dict):
        """
        广播进度更新
        
        Args:
            task_id: 任务ID
            progress: 进度数据
        """
        event_data = {
            "task_id": task_id,
            "progress": progress
        }
        
        # 通过 SocketIO 广播
        if self.socketio:
            try:
                self.socketio.emit('analysis_progress', event_data, namespace='/insight')
            except Exception as e:
                logger.error(f"SocketIO 广播失败: {e}")
        
        # 调用注册的回调
        for callback in self.callbacks.get(task_id, []):
            try:
                callback(task_id, progress)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def broadcast_status_change(self, task_id: str, status: str, message: str = ""):
        """
        广播状态变化
        
        Args:
            task_id: 任务ID
            status: 新状态
            message: 可选消息
        """
        event_data = {
            "task_id": task_id,
            "status": status,
            "message": message
        }
        
        if self.socketio:
            try:
                self.socketio.emit('analysis_status', event_data, namespace='/insight')
            except Exception as e:
                logger.error(f"SocketIO 广播失败: {e}")
    
    def broadcast_error(self, task_id: str, error: str):
        """
        广播错误
        
        Args:
            task_id: 任务ID
            error: 错误信息
        """
        event_data = {
            "task_id": task_id,
            "error": error
        }
        
        if self.socketio:
            try:
                self.socketio.emit('analysis_error', event_data, namespace='/insight')
            except Exception as e:
                logger.error(f"SocketIO 广播失败: {e}")
    
    def register_callback(self, task_id: str, callback: Callable):
        """
        注册进度回调
        
        Args:
            task_id: 任务ID
            callback: 回调函数 (task_id, progress) -> None
        """
        if task_id not in self.callbacks:
            self.callbacks[task_id] = []
        self.callbacks[task_id].append(callback)
    
    def unregister_callback(self, task_id: str, callback: Callable = None):
        """
        取消注册回调
        
        Args:
            task_id: 任务ID
            callback: 回调函数，如为 None 则取消所有
        """
        if callback is None:
            self.callbacks.pop(task_id, None)
        elif task_id in self.callbacks:
            self.callbacks[task_id] = [
                cb for cb in self.callbacks[task_id] if cb != callback
            ]


def get_broadcaster() -> ProgressBroadcaster:
    """获取广播器单例"""
    return ProgressBroadcaster()
