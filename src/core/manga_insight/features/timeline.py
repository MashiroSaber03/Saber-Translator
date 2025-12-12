"""
Manga Insight 剧情时间线

从批量分析结果构建整本漫画的剧情时间线。
完全脱离章节概念，直接基于 key_events 构建事件流。
"""

import logging
from typing import Dict
from datetime import datetime

from ..storage import AnalysisStorage

logger = logging.getLogger("MangaInsight.Timeline")


class TimelineBuilder:
    """
    剧情时间线构建器
    
    从批量分析结果中提取 key_events，构建整本书的事件时间线。
    完全脱离章节概念，按页码顺序组织剧情事件。
    """
    
    def __init__(self, book_id: str = None):
        self.book_id = book_id
        self.storage = None
        if book_id:
            self.storage = AnalysisStorage(book_id)
    
    async def build_timeline(self, book_id: str = None) -> Dict:
        """
        构建剧情时间线
        
        从批量分析的 key_events 中提取事件，按页码顺序组织成时间线。
        
        Args:
            book_id: 书籍ID（如果初始化时未指定）
        
        Returns:
            Dict: {
                "events": [...],  # 事件列表
                "stats": {...},   # 统计信息
                "generated_at": "..."
            }
        """
        if book_id:
            self.book_id = book_id
            self.storage = AnalysisStorage(book_id)
        
        if not self.storage:
            return self._empty_result()
        
        # 获取所有批量分析结果
        batches = await self.storage.list_batches()
        
        if not batches:
            return self._empty_result()
        
        # 从批量分析中提取事件
        timeline_events = []
        total_pages = 0
        event_order = 0
        
        for batch_info in batches:
            start_page = batch_info.get("start_page", 0)
            end_page = batch_info.get("end_page", 0)
            
            batch_data = await self.storage.load_batch_analysis(start_page, end_page)
            if not batch_data:
                continue
            
            total_pages = max(total_pages, end_page)
            
            # 提取关键事件
            key_events = batch_data.get("key_events", [])
            batch_summary = batch_data.get("batch_summary", "")
            
            # 为每个事件创建时间线条目
            for idx, event in enumerate(key_events):
                if not event or not isinstance(event, str):
                    continue
                
                event_text = event.strip()
                if len(event_text) < 3:
                    continue
                
                event_order += 1
                event_entry = {
                    "id": f"event_{start_page}_{end_page}_{idx}",
                    "order": event_order,
                    "event": event_text,
                    "page_range": {
                        "start": start_page,
                        "end": end_page
                    },
                    "context": self._truncate(batch_summary, 200),
                    "thumbnail_page": start_page,  # 用于显示缩略图
                    "importance": self._calculate_importance(event_text)
                }
                timeline_events.append(event_entry)
        
        # 构建结果
        result = {
            "events": timeline_events,
            "groups": [],  # events 模式下为空
            "stats": {
                "total_events": len(timeline_events),
                "total_groups": 0,
                "total_batches": len(batches),
                "total_pages": total_pages
            },
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"时间线构建完成: {len(timeline_events)} 个事件, {total_pages} 页")
        return result
    
    async def build_timeline_grouped(self, book_id: str = None) -> Dict:
        """
        构建分组时间线（按批次分组）
        
        每个批次作为一个时间段，包含该批次的所有事件。
        适合展示更紧凑的时间线视图。
        
        Returns:
            Dict: {
                "groups": [...],  # 分组列表
                "stats": {...},
                "generated_at": "..."
            }
        """
        if book_id:
            self.book_id = book_id
            self.storage = AnalysisStorage(book_id)
        
        if not self.storage:
            return self._empty_result()
        
        batches = await self.storage.list_batches()
        
        if not batches:
            return self._empty_result()
        
        groups = []
        total_events = 0
        total_pages = 0
        
        for batch_info in batches:
            start_page = batch_info.get("start_page", 0)
            end_page = batch_info.get("end_page", 0)
            
            batch_data = await self.storage.load_batch_analysis(start_page, end_page)
            if not batch_data:
                continue
            
            total_pages = max(total_pages, end_page)
            
            key_events = batch_data.get("key_events", [])
            batch_summary = batch_data.get("batch_summary", "")
            
            # 过滤有效事件
            valid_events = [
                e.strip() for e in key_events 
                if e and isinstance(e, str) and len(e.strip()) >= 3
            ]
            
            if not valid_events and not batch_summary:
                continue
            
            total_events += len(valid_events)
            
            group = {
                "id": f"group_{start_page}_{end_page}",
                "page_range": {
                    "start": start_page,
                    "end": end_page
                },
                "summary": self._truncate(batch_summary, 300),
                "events": valid_events,
                "event_count": len(valid_events),
                "thumbnail_page": start_page
            }
            groups.append(group)
        
        result = {
            "events": [],  # grouped 模式下为空
            "groups": groups,
            "stats": {
                "total_events": total_events,
                "total_groups": len(groups),
                "total_batches": len(batches),
                "total_pages": total_pages
            },
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"分组时间线构建完成: {len(groups)} 组, {total_events} 事件")
        return result
    
    def _truncate(self, text: str, max_len: int) -> str:
        """截断文本"""
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
    
    def _calculate_importance(self, event_text: str) -> str:
        """
        计算事件重要性
        
        根据关键词判断事件重要性等级
        """
        high_keywords = [
            "死", "杀", "战", "决", "败", "胜", "婚", "爱", "恨",
            "重大", "关键", "转折", "真相", "秘密", "背叛",
            "觉醒", "爆发", "突破", "终于", "最终"
        ]
        
        medium_keywords = [
            "发现", "遇到", "决定", "开始", "结束", "变化",
            "承诺", "约定", "危机", "冲突", "争吵"
        ]
        
        text_lower = event_text.lower()
        
        for kw in high_keywords:
            if kw in text_lower:
                return "high"
        
        for kw in medium_keywords:
            if kw in text_lower:
                return "medium"
        
        return "normal"
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            "events": [],
            "groups": [],
            "stats": {
                "total_events": 0,
                "total_groups": 0,
                "total_batches": 0,
                "total_pages": 0
            },
            "generated_at": datetime.now().isoformat()
        }
