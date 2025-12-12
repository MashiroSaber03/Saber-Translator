"""
Manga Insight 增强版时间线构建器

基于已有分析数据，通过 LLM 进行智能整合，生成包含：
1. 剧情弧（故事发展阶段）
2. 增强事件（含因果关系、角色关联）
3. 角色轨迹
4. 线索追踪（伏笔/悬念/冲突）

不修改现有的批量分析流程和提示词，仅在构建时间线时进行二次处理。
"""

import logging
import json
import re
from typing import Dict, List, Optional
from datetime import datetime

from ..storage import AnalysisStorage
from ..embedding_client import ChatClient
from .timeline import TimelineBuilder
# timeline_models 可用于类型提示，当前使用 Dict 返回

logger = logging.getLogger("MangaInsight.EnhancedTimeline")


# 时间线智能整合提示词
TIMELINE_SYNTHESIS_PROMPT = """你是一位专业的漫画剧情分析师。基于以下漫画分析数据，构建一个完整的剧情时间线。

【分析数据】
{analysis_data}

请分析并输出以下结构化信息（JSON格式）：

{{
    "story_arcs": [
        {{
            "id": "arc_1",
            "name": "<阶段名称，如'序章-日常'、'冲突爆发'、'真相揭露'、'最终决战'>",
            "description": "<该阶段50-100字概述>",
            "page_range": {{"start": 1, "end": 25}},
            "mood": "<基调：日常/紧张/温馨/悲伤/激烈/神秘/搞笑等>",
            "event_ids": ["event_1", "event_2"]
        }}
    ],
    
    "events": [
        {{
            "id": "event_1",
            "order": 1,
            "event": "<事件描述，一句话概括>",
            "page_range": {{"start": 1, "end": 5}},
            "importance": "high/medium/normal",
            "event_type": "<转折/揭示/冲突/对话/动作/情感/日常>",
            "involved_characters": ["角色A", "角色B"],
            "causes": [],
            "effects": ["event_2"],
            "related_threads": ["thread_1"],
            "context": "<简短的上下文背景>"
        }}
    ],
    
    "characters": [
        {{
            "name": "<角色名>",
            "aliases": ["<别名或称呼>"],
            "first_appearance": 1,
            "description": "<一句话描述角色特点>",
            "arc": "<角色发展线，如'从敌对到和解'、'逐渐觉醒'>",
            "key_moments": [
                {{"page": 10, "event": "<关键行为>", "significance": "<意义>"}}
            ],
            "relationships": [
                {{"character": "<其他角色名>", "relation": "<关系描述>"}}
            ]
        }}
    ],
    
    "plot_threads": [
        {{
            "id": "thread_1",
            "name": "<线索名>",
            "type": "<伏笔/悬念/冲突/主题>",
            "status": "<未解决/进行中/已解决>",
            "introduced_at": 5,
            "resolved_at": null,
            "description": "<线索说明>",
            "related_events": ["event_1"]
        }}
    ],
    
    "summary": {{
        "one_sentence": "<一句话概括整个故事>",
        "main_conflict": "<核心冲突是什么>",
        "turning_points": ["<重要转折点1>", "<重要转折点2>"],
        "themes": ["<主题1>", "<主题2>"]
    }}
}}

要求：
1. 剧情弧应反映故事的起承转合结构，通常2-5个阶段
2. 事件之间要建立因果关系（causes/effects 使用事件ID关联）
3. 识别3-5个主要角色并追踪其发展
4. 识别未解决的悬念和伏笔
5. 所有内容使用中文输出
6. 确保 page_range 的页码与原始数据一致
7. event_ids 和 related_events 要与 events 中的 id 对应

【重要】请直接输出JSON，不要包含任何解释、markdown代码块或其他文字。"""


class EnhancedTimelineBuilder:
    """
    增强版时间线构建器
    
    在构建时间线时调用 LLM 对已有分析数据进行二次整合，
    生成包含剧情弧、角色轨迹、线索追踪等丰富信息的时间线。
    """
    
    def __init__(self, book_id: str, config=None):
        """
        初始化构建器
        
        Args:
            book_id: 书籍ID
            config: MangaInsightConfig 配置对象
        """
        self.book_id = book_id
        self.config = config
        self.storage = AnalysisStorage(book_id)
        self.llm = None
        
        # 初始化 LLM 客户端
        if config:
            try:
                # 检查是否使用独立的对话模型
                if config.chat_llm and not config.chat_llm.use_same_as_vlm:
                    # 检查 api_key 是否配置
                    if config.chat_llm.api_key:
                        self.llm = ChatClient(config.chat_llm)
                    else:
                        logger.warning("ChatLLM 未配置 API Key")
                elif config.vlm and config.vlm.api_key:
                    # 使用 VLM 配置作为对话模型
                    self.llm = ChatClient(config.vlm)
                else:
                    logger.warning("VLM 未配置 API Key，增强时间线不可用")
            except Exception as e:
                logger.warning(f"初始化 LLM 客户端失败: {e}")
    
    async def build(self, mode: str = "enhanced") -> Dict:
        """
        构建时间线
        
        Args:
            mode: 
                - "simple": 简单模式（使用原有 TimelineBuilder 逻辑）
                - "enhanced": 增强模式（LLM 智能整合）
        
        Returns:
            Dict: 时间线数据
        """
        logger.info(f"开始构建时间线: book_id={self.book_id}, mode={mode}")
        
        # 简单模式：使用原有逻辑
        if mode == "simple":
            simple_builder = TimelineBuilder(self.book_id)
            result = await simple_builder.build_timeline_grouped()
            result["mode"] = "simple"
            return result
        
        # 增强模式
        try:
            # 1. 收集分析数据
            analysis_data = await self._collect_analysis_data()
            
            if not analysis_data:
                logger.warning("没有可用的分析数据，降级到简单模式")
                simple_builder = TimelineBuilder(self.book_id)
                result = await simple_builder.build_timeline_grouped()
                result["mode"] = "simple"
                result["fallback_reason"] = "no_analysis_data"
                return result
            
            # 2. 检查 LLM 是否可用
            if not self.llm:
                logger.warning("LLM 不可用，降级到简单模式")
                simple_builder = TimelineBuilder(self.book_id)
                result = await simple_builder.build_timeline_grouped()
                result["mode"] = "simple"
                result["fallback_reason"] = "llm_unavailable"
                return result
            
            # 3. LLM 智能整合
            enhanced_data = await self._synthesize_timeline(analysis_data)
            
            if not enhanced_data:
                logger.warning("LLM 整合失败，降级到简单模式")
                simple_builder = TimelineBuilder(self.book_id)
                result = await simple_builder.build_timeline_grouped()
                result["mode"] = "simple"
                result["fallback_reason"] = "llm_synthesis_failed"
                return result
            
            # 4. 后处理
            result = self._post_process(enhanced_data)
            
            logger.info(f"增强时间线构建完成: {result.get('stats', {})}")
            return result
            
        except Exception as e:
            logger.error(f"增强时间线构建失败: {e}", exc_info=True)
            # 降级到简单模式
            simple_builder = TimelineBuilder(self.book_id)
            result = await simple_builder.build_timeline_grouped()
            result["mode"] = "simple"
            result["fallback_reason"] = str(e)
            return result
    
    async def _collect_analysis_data(self) -> str:
        """
        收集所有分析数据，格式化为 LLM 输入
        
        Returns:
            str: 格式化的分析数据文本
        """
        parts = []
        total_pages = 0
        
        # 加载全书概述（如有）
        overview = await self.storage.load_overview()
        if overview:
            book_summary = overview.get("book_summary", "")
            if not book_summary:
                book_summary = overview.get("summary", "")
            if book_summary:
                parts.append(f"【全书概述】\n{book_summary}\n")
        
        # 加载批量分析
        batches = await self.storage.list_batches()
        
        if not batches:
            return ""
        
        for batch_info in batches:
            start_page = batch_info.get("start_page", 0)
            end_page = batch_info.get("end_page", 0)
            total_pages = max(total_pages, end_page)
            
            batch = await self.storage.load_batch_analysis(start_page, end_page)
            if not batch:
                continue
            
            batch_parts = [f"【第{start_page}-{end_page}页】"]
            
            # 批次摘要
            batch_summary = batch.get("batch_summary", "")
            if batch_summary:
                batch_parts.append(f"剧情概述: {batch_summary}")
            
            # 关键事件
            events = batch.get("key_events", [])
            if events:
                valid_events = [str(e) for e in events if e]
                if valid_events:
                    batch_parts.append(f"关键事件: {'; '.join(valid_events)}")
            
            # 衔接说明
            notes = batch.get("continuity_notes", "")
            if notes:
                batch_parts.append(f"衔接说明: {notes}")
            
            parts.append("\n".join(batch_parts))
        
        # 加载段落总结（如有）
        segments = await self.storage.list_segments()
        if segments:
            segment_parts = ["\n【段落总结】"]
            for seg_info in segments:
                seg_id = seg_info.get("segment_id", "")
                seg_data = await self.storage.load_segment_summary(seg_id)
                if seg_data:
                    page_range = seg_data.get("page_range", {})
                    summary = seg_data.get("summary", "")
                    if summary:
                        segment_parts.append(
                            f"第{page_range.get('start', 0)}-{page_range.get('end', 0)}页: {summary}"
                        )
            if len(segment_parts) > 1:
                parts.append("\n".join(segment_parts))
        
        # 添加页数信息
        parts.insert(0, f"【基本信息】\n总页数: {total_pages}\n")
        
        return "\n\n".join(parts)
    
    async def _synthesize_timeline(self, analysis_data: str) -> Optional[Dict]:
        """
        调用 LLM 进行智能整合
        
        Args:
            analysis_data: 格式化的分析数据
        
        Returns:
            Dict: 解析后的时间线数据，失败返回 None
        """
        if not self.llm:
            return None
        
        # 构建提示词
        prompt = TIMELINE_SYNTHESIS_PROMPT.format(analysis_data=analysis_data)
        
        try:
            # 调用 LLM
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3
            )
            
            if not response:
                logger.warning("LLM 返回空响应")
                return None
            
            # 解析 JSON
            result = self._parse_json_response(response)
            
            if result:
                logger.info(f"LLM 整合成功: {len(result.get('events', []))} 个事件")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None
        finally:
            # 关闭 LLM 客户端，释放资源
            if self.llm:
                try:
                    await self.llm.close()
                except Exception:
                    pass
                self.llm = None
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 返回的 JSON
        
        Args:
            response: LLM 响应文本
        
        Returns:
            Dict: 解析后的数据，失败返回 None
        """
        if not response:
            return None
        
        # 清理响应
        text = response.strip()
        
        # 移除可能的 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            # 移除首行的 ```json 或 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 移除末行的 ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 对象
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 尝试修复常见问题
        try:
            # 修复尾部逗号
            fixed = re.sub(r',(\s*[}\]])', r'\1', text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        logger.warning(f"无法解析 JSON 响应: {text[:500]}...")
        return None
    
    def _post_process(self, data: Dict) -> Dict:
        """
        后处理：验证、补充、统计
        
        Args:
            data: LLM 返回的原始数据
        
        Returns:
            Dict: 处理后的时间线数据
        """
        # 确保必要字段存在
        data.setdefault("story_arcs", [])
        data.setdefault("events", [])
        data.setdefault("characters", [])
        data.setdefault("plot_threads", [])
        data.setdefault("summary", {})
        
        # 构建事件映射
        events = data.get("events", [])
        event_map = {e.get("id", ""): e for e in events if e.get("id")}
        
        # 建立双向关联
        for event in events:
            event_id = event.get("id", "")
            
            # 确保 causes 中的事件有对应的 effects
            for cause_id in event.get("causes", []):
                if cause_id in event_map:
                    cause = event_map[cause_id]
                    effects = cause.setdefault("effects", [])
                    if event_id and event_id not in effects:
                        effects.append(event_id)
            
            # 确保 effects 中的事件有对应的 causes
            for effect_id in event.get("effects", []):
                if effect_id in event_map:
                    effect = event_map[effect_id]
                    causes = effect.setdefault("causes", [])
                    if event_id and event_id not in causes:
                        causes.append(event_id)
            
            # 确保必要字段存在
            event.setdefault("importance", "normal")
            event.setdefault("event_type", "")
            event.setdefault("involved_characters", [])
            event.setdefault("context", "")
            event.setdefault("arc_id", "")
            event.setdefault("related_threads", [])
        
        # 验证剧情弧的事件关联
        for arc in data.get("story_arcs", []):
            arc.setdefault("event_ids", [])
            arc.setdefault("mood", "")
            arc.setdefault("description", "")
            
            # 确保 arc 的 event_ids 都存在
            arc["event_ids"] = [
                eid for eid in arc.get("event_ids", [])
                if eid in event_map
            ]
        
        # 验证角色数据
        for char in data.get("characters", []):
            char.setdefault("aliases", [])
            char.setdefault("first_appearance", 0)
            char.setdefault("description", "")
            char.setdefault("arc", "")
            char.setdefault("key_moments", [])
            char.setdefault("relationships", [])
        
        # 验证线索数据
        for thread in data.get("plot_threads", []):
            thread.setdefault("type", "悬念")
            thread.setdefault("status", "进行中")
            thread.setdefault("introduced_at", 0)
            thread.setdefault("description", "")
            thread.setdefault("related_events", [])
            
            # 确保 related_events 都存在
            thread["related_events"] = [
                eid for eid in thread.get("related_events", [])
                if eid in event_map
            ]
        
        # 计算总页数
        total_pages = 0
        for event in events:
            page_range = event.get("page_range", {})
            end_page = page_range.get("end", 0)
            total_pages = max(total_pages, end_page)
        
        for arc in data.get("story_arcs", []):
            page_range = arc.get("page_range", {})
            end_page = page_range.get("end", 0)
            total_pages = max(total_pages, end_page)
        
        # 计算统计信息
        unresolved_count = sum(
            1 for t in data.get("plot_threads", [])
            if t.get("status") != "已解决"
        )
        
        data["stats"] = {
            "total_arcs": len(data.get("story_arcs", [])),
            "total_events": len(events),
            "total_characters": len(data.get("characters", [])),
            "total_threads": len(data.get("plot_threads", [])),
            "total_pages": total_pages,
            "unresolved_threads": unresolved_count
        }
        
        # 添加元信息
        data["book_id"] = self.book_id
        data["mode"] = "enhanced"
        data["generated_at"] = datetime.now().isoformat()
        
        # 确保 summary 字段完整
        summary = data.get("summary", {})
        summary.setdefault("one_sentence", "")
        summary.setdefault("main_conflict", "")
        summary.setdefault("turning_points", [])
        summary.setdefault("themes", [])
        data["summary"] = summary
        
        return data
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            "book_id": self.book_id,
            "mode": "enhanced",
            "story_arcs": [],
            "events": [],
            "characters": [],
            "plot_threads": [],
            "summary": {
                "one_sentence": "",
                "main_conflict": "",
                "turning_points": [],
                "themes": []
            },
            "stats": {
                "total_arcs": 0,
                "total_events": 0,
                "total_characters": 0,
                "total_threads": 0,
                "total_pages": 0,
                "unresolved_threads": 0
            },
            "generated_at": datetime.now().isoformat()
        }
