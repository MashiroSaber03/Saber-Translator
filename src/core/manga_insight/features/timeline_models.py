"""
Manga Insight 增强时间线数据模型

定义时间线相关的数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class StoryArc:
    """
    剧情弧（故事发展阶段）
    
    用于划分故事的起承转合结构，如"序章"、"冲突升级"、"高潮"、"结局"等。
    """
    id: str
    name: str                              # 阶段名称
    description: str                       # 该阶段概述
    page_range: Dict[str, int]             # {"start": 1, "end": 25}
    mood: str = ""                         # 基调：紧张、温馨、悲伤等
    event_ids: List[str] = field(default_factory=list)  # 该阶段的事件ID列表


@dataclass
class TimelineEvent:
    """
    增强版时间线事件
    
    不仅包含事件描述，还包含关联信息如涉及角色、因果关系等。
    """
    id: str
    order: int                             # 事件顺序
    event: str                             # 事件描述
    page_range: Dict[str, int]             # {"start": 1, "end": 5}
    
    # 增强字段
    arc_id: str = ""                       # 所属剧情弧
    importance: str = "normal"             # high/medium/normal
    event_type: str = ""                   # 对话/动作/转折/揭示/冲突/情感
    involved_characters: List[str] = field(default_factory=list)  # 涉及角色
    causes: List[str] = field(default_factory=list)    # 前因事件ID
    effects: List[str] = field(default_factory=list)   # 后果事件ID
    related_threads: List[str] = field(default_factory=list)  # 相关线索ID
    context: str = ""                      # 上下文背景


@dataclass
class CharacterMoment:
    """角色关键时刻"""
    page: int
    event: str
    significance: str = ""


@dataclass
class CharacterTrack:
    """
    角色轨迹
    
    追踪主要角色在故事中的发展和行为。
    """
    name: str                              # 角色名
    aliases: List[str] = field(default_factory=list)   # 别名/称呼
    first_appearance: int = 0              # 首次出现页码
    description: str = ""                  # 角色简介
    arc: str = ""                          # 角色发展弧（如"从怀疑到信任"）
    key_moments: List[Dict] = field(default_factory=list)  # 关键时刻
    relationships: List[Dict] = field(default_factory=list)  # 与其他角色关系


@dataclass
class PlotThread:
    """
    剧情线索
    
    追踪伏笔、悬念、冲突等剧情线索的发展状态。
    """
    id: str
    name: str                              # 线索名称
    type: str = ""                         # 伏笔/悬念/冲突/主题
    status: str = "进行中"                 # 未解决/进行中/已解决
    introduced_at: int = 0                 # 引入页码
    resolved_at: Optional[int] = None      # 解决页码
    description: str = ""
    related_events: List[str] = field(default_factory=list)  # 相关事件ID


@dataclass
class TimelineSummary:
    """时间线摘要"""
    one_sentence: str = ""                 # 一句话概括
    main_conflict: str = ""                # 核心冲突
    turning_points: List[str] = field(default_factory=list)  # 重要转折点
    themes: List[str] = field(default_factory=list)          # 主题标签


@dataclass
class TimelineStats:
    """时间线统计信息"""
    total_arcs: int = 0
    total_events: int = 0
    total_characters: int = 0
    total_threads: int = 0
    total_pages: int = 0
    unresolved_threads: int = 0


@dataclass
class EnhancedTimeline:
    """
    增强版时间线
    
    包含剧情弧、事件流、角色轨迹、线索追踪等完整信息。
    """
    book_id: str
    mode: str = "enhanced"                 # enhanced 或 simple
    
    # 剧情弧
    story_arcs: List[StoryArc] = field(default_factory=list)
    
    # 事件流
    events: List[TimelineEvent] = field(default_factory=list)
    
    # 角色轨迹
    characters: List[CharacterTrack] = field(default_factory=list)
    
    # 线索追踪
    plot_threads: List[PlotThread] = field(default_factory=list)
    
    # 摘要
    summary: Optional[TimelineSummary] = None
    
    # 统计
    stats: Optional[TimelineStats] = None
    
    # 元信息
    generated_at: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "book_id": self.book_id,
            "mode": self.mode,
            "story_arcs": [
                {
                    "id": arc.id,
                    "name": arc.name,
                    "description": arc.description,
                    "page_range": arc.page_range,
                    "mood": arc.mood,
                    "event_ids": arc.event_ids
                }
                for arc in self.story_arcs
            ],
            "events": [
                {
                    "id": e.id,
                    "order": e.order,
                    "event": e.event,
                    "page_range": e.page_range,
                    "arc_id": e.arc_id,
                    "importance": e.importance,
                    "event_type": e.event_type,
                    "involved_characters": e.involved_characters,
                    "causes": e.causes,
                    "effects": e.effects,
                    "related_threads": e.related_threads,
                    "context": e.context
                }
                for e in self.events
            ],
            "characters": [
                {
                    "name": c.name,
                    "aliases": c.aliases,
                    "first_appearance": c.first_appearance,
                    "description": c.description,
                    "arc": c.arc,
                    "key_moments": c.key_moments,
                    "relationships": c.relationships
                }
                for c in self.characters
            ],
            "plot_threads": [
                {
                    "id": t.id,
                    "name": t.name,
                    "type": t.type,
                    "status": t.status,
                    "introduced_at": t.introduced_at,
                    "resolved_at": t.resolved_at,
                    "description": t.description,
                    "related_events": t.related_events
                }
                for t in self.plot_threads
            ],
            "summary": {
                "one_sentence": self.summary.one_sentence if self.summary else "",
                "main_conflict": self.summary.main_conflict if self.summary else "",
                "turning_points": self.summary.turning_points if self.summary else [],
                "themes": self.summary.themes if self.summary else []
            } if self.summary else {},
            "stats": {
                "total_arcs": self.stats.total_arcs if self.stats else 0,
                "total_events": self.stats.total_events if self.stats else 0,
                "total_characters": self.stats.total_characters if self.stats else 0,
                "total_threads": self.stats.total_threads if self.stats else 0,
                "total_pages": self.stats.total_pages if self.stats else 0,
                "unresolved_threads": self.stats.unresolved_threads if self.stats else 0
            } if self.stats else {},
            "generated_at": self.generated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EnhancedTimeline":
        """从字典创建实例"""
        timeline = cls(
            book_id=data.get("book_id", ""),
            mode=data.get("mode", "enhanced"),
            generated_at=data.get("generated_at", "")
        )
        
        # 解析剧情弧
        for arc_data in data.get("story_arcs", []):
            timeline.story_arcs.append(StoryArc(
                id=arc_data.get("id", ""),
                name=arc_data.get("name", ""),
                description=arc_data.get("description", ""),
                page_range=arc_data.get("page_range", {}),
                mood=arc_data.get("mood", ""),
                event_ids=arc_data.get("event_ids", [])
            ))
        
        # 解析事件
        for e_data in data.get("events", []):
            timeline.events.append(TimelineEvent(
                id=e_data.get("id", ""),
                order=e_data.get("order", 0),
                event=e_data.get("event", ""),
                page_range=e_data.get("page_range", {}),
                arc_id=e_data.get("arc_id", ""),
                importance=e_data.get("importance", "normal"),
                event_type=e_data.get("event_type", ""),
                involved_characters=e_data.get("involved_characters", []),
                causes=e_data.get("causes", []),
                effects=e_data.get("effects", []),
                related_threads=e_data.get("related_threads", []),
                context=e_data.get("context", "")
            ))
        
        # 解析角色
        for c_data in data.get("characters", []):
            timeline.characters.append(CharacterTrack(
                name=c_data.get("name", ""),
                aliases=c_data.get("aliases", []),
                first_appearance=c_data.get("first_appearance", 0),
                description=c_data.get("description", ""),
                arc=c_data.get("arc", ""),
                key_moments=c_data.get("key_moments", []),
                relationships=c_data.get("relationships", [])
            ))
        
        # 解析线索
        for t_data in data.get("plot_threads", []):
            timeline.plot_threads.append(PlotThread(
                id=t_data.get("id", ""),
                name=t_data.get("name", ""),
                type=t_data.get("type", ""),
                status=t_data.get("status", "进行中"),
                introduced_at=t_data.get("introduced_at", 0),
                resolved_at=t_data.get("resolved_at"),
                description=t_data.get("description", ""),
                related_events=t_data.get("related_events", [])
            ))
        
        # 解析摘要
        summary_data = data.get("summary", {})
        if summary_data:
            timeline.summary = TimelineSummary(
                one_sentence=summary_data.get("one_sentence", ""),
                main_conflict=summary_data.get("main_conflict", ""),
                turning_points=summary_data.get("turning_points", []),
                themes=summary_data.get("themes", [])
            )
        
        # 解析统计
        stats_data = data.get("stats", {})
        if stats_data:
            timeline.stats = TimelineStats(
                total_arcs=stats_data.get("total_arcs", 0),
                total_events=stats_data.get("total_events", 0),
                total_characters=stats_data.get("total_characters", 0),
                total_threads=stats_data.get("total_threads", 0),
                total_pages=stats_data.get("total_pages", 0),
                unresolved_threads=stats_data.get("unresolved_threads", 0)
            )
        
        return timeline
