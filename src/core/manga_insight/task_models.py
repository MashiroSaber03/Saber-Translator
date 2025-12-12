"""
Manga Insight 任务数据模型

定义分析任务的状态、类型和进度模型。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"        # 等待中
    RUNNING = "running"        # 运行中
    PAUSED = "paused"          # 已暂停
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 已取消


class TaskType(Enum):
    """任务类型枚举"""
    FULL_BOOK = "full_book"           # 全书分析
    CHAPTER = "chapter"               # 章节分析
    INCREMENTAL = "incremental"       # 增量分析
    REANALYZE = "reanalyze"          # 重新分析（批量）


@dataclass
class AnalysisProgress:
    """分析进度"""
    total_pages: int = 0
    analyzed_pages: int = 0
    current_page: int = 0
    current_phase: str = ""  # "page_analysis" / "character_tracking" / "embedding"
    phase_progress: float = 0.0  # 0-100
    estimated_time_remaining: Optional[int] = None  # 秒
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "analyzed_pages": self.analyzed_pages,
            "current_page": self.current_page,
            "current_phase": self.current_phase,
            "phase_progress": self.phase_progress,
            "estimated_time_remaining": self.estimated_time_remaining,
            "percentage": (self.analyzed_pages / self.total_pages * 100) if self.total_pages > 0 else 0
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisProgress":
        return cls(
            total_pages=data.get("total_pages", 0),
            analyzed_pages=data.get("analyzed_pages", 0),
            current_page=data.get("current_page", 0),
            current_phase=data.get("current_phase", ""),
            phase_progress=data.get("phase_progress", 0.0),
            estimated_time_remaining=data.get("estimated_time_remaining")
        )


@dataclass
class AnalysisTask:
    """分析任务"""
    task_id: str = ""
    book_id: str = ""
    task_type: TaskType = TaskType.FULL_BOOK
    status: TaskStatus = TaskStatus.PENDING
    progress: AnalysisProgress = field(default_factory=AnalysisProgress)
    
    # 范围控制
    target_chapters: Optional[List[str]] = None  # None = 全部
    target_pages: Optional[List[int]] = None     # None = 全部
    
    # 时间记录
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 错误信息
    error_message: Optional[str] = None
    failed_pages: List[int] = field(default_factory=list)
    
    # 增量分析相关
    is_incremental: bool = False
    base_analysis_version: Optional[str] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:12]}"
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "book_id": self.book_id,
            "task_type": self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "progress": self.progress.to_dict() if self.progress else {},
            "target_chapters": self.target_chapters,
            "target_pages": self.target_pages,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "failed_pages": self.failed_pages,
            "is_incremental": self.is_incremental,
            "base_analysis_version": self.base_analysis_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisTask":
        # 解析时间字段
        created_at = None
        started_at = None
        completed_at = None
        
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except:
                pass
        
        if data.get("started_at"):
            try:
                started_at = datetime.fromisoformat(data["started_at"])
            except:
                pass
        
        if data.get("completed_at"):
            try:
                completed_at = datetime.fromisoformat(data["completed_at"])
            except:
                pass
        
        # 解析枚举
        task_type = data.get("task_type", "full_book")
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = TaskStatus(status)
        
        return cls(
            task_id=data.get("task_id", ""),
            book_id=data.get("book_id", ""),
            task_type=task_type,
            status=status,
            progress=AnalysisProgress.from_dict(data.get("progress", {})),
            target_chapters=data.get("target_chapters"),
            target_pages=data.get("target_pages"),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            error_message=data.get("error_message"),
            failed_pages=data.get("failed_pages", []),
            is_incremental=data.get("is_incremental", False),
            base_analysis_version=data.get("base_analysis_version")
        )


@dataclass
class BookAnalysisStatus:
    """书籍分析状态"""
    book_id: str
    analyzed: bool = False
    analysis_version: Optional[str] = None
    total_pages: int = 0
    analyzed_pages: int = 0
    last_analyzed_at: Optional[datetime] = None
    current_task_id: Optional[str] = None
    current_task_status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "analyzed": self.analyzed,
            "analysis_version": self.analysis_version,
            "total_pages": self.total_pages,
            "analyzed_pages": self.analyzed_pages,
            "last_analyzed_at": self.last_analyzed_at.isoformat() if self.last_analyzed_at else None,
            "current_task_id": self.current_task_id,
            "current_task_status": self.current_task_status,
            "percentage": (self.analyzed_pages / self.total_pages * 100) if self.total_pages > 0 else 0
        }
