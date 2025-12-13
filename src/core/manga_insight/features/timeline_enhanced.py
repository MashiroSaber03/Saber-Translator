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
3. 识别主要角色并追踪其发展
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
        
        降级策略（三级）：
        1. 增强模式：从章节/段落/批量分析收集数据，LLM 智能整合
        2. 压缩摘要模式：使用生成概述时保存的压缩摘要，LLM 智能整合（数据量更小）
        3. 简单模式：直接从批量分析提取事件，不调用 LLM
        
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
                logger.warning("没有可用的分析数据，尝试使用压缩摘要")
                return await self._build_with_fallback("no_analysis_data")
            
            # 2. 检查 LLM 是否可用
            if not self.llm:
                logger.warning("LLM 不可用，降级到简单模式")
                return await self._fallback_to_simple("llm_unavailable")
            
            # 3. LLM 智能整合
            enhanced_data = await self._synthesize_timeline(analysis_data)
            
            if not enhanced_data:
                logger.warning("LLM 整合失败（可能是数据过多），尝试使用压缩摘要")
                return await self._build_with_fallback("llm_synthesis_failed")
            
            # 4. 后处理
            result = self._post_process(enhanced_data)
            
            logger.info(f"增强时间线构建完成: {result.get('stats', {})}")
            return result
            
        except Exception as e:
            logger.error(f"增强时间线构建失败: {e}", exc_info=True)
            # 尝试使用压缩摘要降级
            return await self._build_with_fallback(str(e))
    
    async def _build_with_fallback(self, original_error: str) -> Dict:
        """
        中间降级：使用压缩摘要构建时间线
        
        当原始数据过多导致 LLM 调用失败时，尝试使用生成概述时保存的压缩摘要。
        压缩摘要通常在 2000-5000 字，更容易被 LLM 处理。
        
        Args:
            original_error: 原始错误原因
        
        Returns:
            Dict: 时间线数据
        """
        logger.info("尝试使用压缩摘要构建时间线（中间降级）")
        
        try:
            # 1. 加载压缩摘要
            compressed_data = await self.storage.load_compressed_context()
            
            if not compressed_data or not compressed_data.get("context"):
                logger.warning("没有压缩摘要，降级到简单模式")
                return await self._fallback_to_simple(f"{original_error} -> no_compressed_context")
            
            context = compressed_data.get("context", "")
            char_count = compressed_data.get("char_count", len(context))
            source = compressed_data.get("source", "unknown")
            
            logger.info(f"使用压缩摘要: {char_count} 字, 来源: {source}")
            
            # 2. 检查 LLM 是否可用（可能需要重新初始化）
            if not self.llm and self.config:
                try:
                    if self.config.chat_llm and not self.config.chat_llm.use_same_as_vlm:
                        if self.config.chat_llm.api_key:
                            self.llm = ChatClient(self.config.chat_llm)
                    elif self.config.vlm and self.config.vlm.api_key:
                        self.llm = ChatClient(self.config.vlm)
                except Exception as e:
                    logger.warning(f"重新初始化 LLM 失败: {e}")
            
            if not self.llm:
                logger.warning("LLM 不可用，降级到简单模式")
                return await self._fallback_to_simple(f"{original_error} -> llm_unavailable")
            
            # 3. 构建简化的分析数据（基于压缩摘要）
            analysis_data = f"【基本信息】\n来源: {source}\n字数: {char_count}\n\n【全书剧情摘要】\n{context}"
            
            # 4. LLM 智能整合
            enhanced_data = await self._synthesize_timeline(analysis_data)
            
            if not enhanced_data:
                logger.warning("压缩摘要模式 LLM 整合也失败，降级到简单模式")
                return await self._fallback_to_simple(f"{original_error} -> compressed_synthesis_failed")
            
            # 5. 后处理
            result = self._post_process(enhanced_data)
            result["mode"] = "compressed"  # 标记为压缩摘要模式
            result["fallback_reason"] = original_error
            
            logger.info(f"压缩摘要模式时间线构建完成: {result.get('stats', {})}")
            return result
            
        except Exception as e:
            logger.error(f"压缩摘要模式构建失败: {e}", exc_info=True)
            return await self._fallback_to_simple(f"{original_error} -> {str(e)}")
    
    async def _fallback_to_simple(self, reason: str) -> Dict:
        """
        最终降级：使用简单模式
        
        Args:
            reason: 降级原因
        
        Returns:
            Dict: 简单模式时间线数据
        """
        logger.info(f"降级到简单模式，原因: {reason}")
        simple_builder = TimelineBuilder(self.book_id)
        result = await simple_builder.build_timeline_grouped()
        result["mode"] = "simple"
        result["fallback_reason"] = reason
        return result
    
    # 配置常量
    MAX_SECTIONS_BEFORE_MERGE = 15  # 超过此数量时进行合并
    MAX_EVENTS_PER_BATCH = 10       # 每批次最多保留的关键事件数
    MAX_TOTAL_EVENTS = 80           # 全书最多保留的关键事件数
    MERGE_GROUP_SIZE = 3            # 合并时每组包含的段落数
    
    async def _collect_analysis_data(self) -> str:
        """
        收集所有分析数据，格式化为 LLM 输入
        
        优先级（仿照概述生成逻辑）：
        1. 优先使用章节总结（最精炼）
        2. 次优先使用段落总结
        3. 降级使用批量分析（会压缩）
        
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
        
        # 方案1：优先使用章节总结
        chapters = await self.storage.list_chapters()
        if chapters:
            logger.info(f"时间线数据收集：使用 {len(chapters)} 个章节总结")
            chapter_data = await self._collect_from_chapters(chapters)
            if chapter_data:
                parts.append(chapter_data["content"])
                total_pages = chapter_data["total_pages"]
                parts.insert(0, f"【基本信息】\n总页数: {total_pages}\n")
                return "\n\n".join(parts)
        
        # 方案2：使用段落总结
        segments = await self.storage.list_segments()
        if segments:
            logger.info(f"时间线数据收集：使用 {len(segments)} 个段落总结")
            segment_data = await self._collect_from_segments(segments)
            if segment_data:
                parts.append(segment_data["content"])
                total_pages = segment_data["total_pages"]
                parts.insert(0, f"【基本信息】\n总页数: {total_pages}\n")
                return "\n\n".join(parts)
        
        # 方案3：使用批量分析（压缩处理）
        batches = await self.storage.list_batches()
        if batches:
            logger.info(f"时间线数据收集：使用 {len(batches)} 个批量分析")
            batch_data = await self._collect_from_batches(batches)
            if batch_data:
                parts.append(batch_data["content"])
                total_pages = batch_data["total_pages"]
                parts.insert(0, f"【基本信息】\n总页数: {total_pages}\n")
                return "\n\n".join(parts)
        
        return ""
    
    async def _collect_from_chapters(self, chapters: List[Dict]) -> Optional[Dict]:
        """从章节总结收集数据"""
        chapter_parts = ["【章节剧情】"]
        total_pages = 0
        all_events = []
        
        for ch in chapters:
            ch_id = ch.get("id", "")
            analysis = await self.storage.load_chapter_analysis(ch_id)
            if not analysis:
                continue
            
            page_range = analysis.get("page_range", {})
            start = page_range.get("start", 0)
            end = page_range.get("end", 0)
            total_pages = max(total_pages, end)
            
            title = analysis.get("title", f"章节{ch_id}")
            summary = analysis.get("summary", "")
            events = analysis.get("plot_events", [])[:self.MAX_EVENTS_PER_BATCH]
            
            if summary:
                chapter_parts.append(f"\n{title}（第{start}-{end}页）:\n{summary}")
                if events:
                    all_events.extend([(start, e) for e in events if e])
        
        if len(chapter_parts) <= 1:
            return None
        
        # 添加关键事件汇总
        if all_events:
            chapter_parts.append("\n【关键事件汇总】")
            # 按页码排序，限制总数
            all_events.sort(key=lambda x: x[0])
            for page, event in all_events[:self.MAX_TOTAL_EVENTS]:
                chapter_parts.append(f"- 第{page}页: {event}")
        
        return {
            "content": "\n".join(chapter_parts),
            "total_pages": total_pages
        }
    
    async def _collect_from_segments(self, segments: List[Dict]) -> Optional[Dict]:
        """从段落总结收集数据"""
        segment_items = []
        total_pages = 0
        
        for seg_info in segments:
            seg_id = seg_info.get("segment_id", "")
            seg_data = await self.storage.load_segment_summary(seg_id)
            if not seg_data:
                continue
            
            page_range = seg_data.get("page_range", {})
            start = page_range.get("start", 0)
            end = page_range.get("end", 0)
            total_pages = max(total_pages, end)
            
            summary = seg_data.get("summary", "")
            events = seg_data.get("key_events", [])[:self.MAX_EVENTS_PER_BATCH]
            
            if summary:
                segment_items.append({
                    "start": start,
                    "end": end,
                    "summary": summary,
                    "events": events
                })
        
        if not segment_items:
            return None
        
        # 如果段落过多，进行合并
        if len(segment_items) > self.MAX_SECTIONS_BEFORE_MERGE:
            segment_items = self._merge_items(segment_items)
            logger.info(f"段落过多，合并后剩余 {len(segment_items)} 个")
        
        # 构建输出
        segment_parts = ["【段落剧情】"]
        all_events = []
        
        for item in segment_items:
            segment_parts.append(
                f"\n第{item['start']}-{item['end']}页:\n{item['summary']}"
            )
            all_events.extend([(item['start'], e) for e in item.get('events', []) if e])
        
        # 添加关键事件汇总
        if all_events:
            segment_parts.append("\n【关键事件汇总】")
            all_events.sort(key=lambda x: x[0])
            for page, event in all_events[:self.MAX_TOTAL_EVENTS]:
                segment_parts.append(f"- 第{page}页: {event}")
        
        return {
            "content": "\n".join(segment_parts),
            "total_pages": total_pages
        }
    
    async def _collect_from_batches(self, batches: List[Dict]) -> Optional[Dict]:
        """从批量分析收集数据（会压缩）"""
        batch_items = []
        total_pages = 0
        
        for batch_info in batches:
            start = batch_info.get("start_page", 0)
            end = batch_info.get("end_page", 0)
            total_pages = max(total_pages, end)
            
            batch = await self.storage.load_batch_analysis(start, end)
            if not batch:
                continue
            
            summary = batch.get("batch_summary", "")
            events = batch.get("key_events", [])[:self.MAX_EVENTS_PER_BATCH]
            
            if summary:
                batch_items.append({
                    "start": start,
                    "end": end,
                    "summary": summary,
                    "events": [str(e) for e in events if e]
                })
        
        if not batch_items:
            return None
        
        # 如果批次过多，进行合并
        if len(batch_items) > self.MAX_SECTIONS_BEFORE_MERGE:
            batch_items = self._merge_items(batch_items)
            logger.info(f"批次过多，合并后剩余 {len(batch_items)} 个")
        
        # 构建输出
        batch_parts = ["【剧情发展】"]
        all_events = []
        
        for item in batch_items:
            batch_parts.append(
                f"\n第{item['start']}-{item['end']}页:\n{item['summary']}"
            )
            all_events.extend([(item['start'], e) for e in item.get('events', []) if e])
        
        # 添加关键事件汇总（限制数量）
        if all_events:
            batch_parts.append("\n【关键事件汇总】")
            all_events.sort(key=lambda x: x[0])
            for page, event in all_events[:self.MAX_TOTAL_EVENTS]:
                batch_parts.append(f"- 第{page}页: {event}")
        
        return {
            "content": "\n".join(batch_parts),
            "total_pages": total_pages
        }
    
    def _merge_items(self, items: List[Dict]) -> List[Dict]:
        """
        合并过多的段落/批次
        
        每 MERGE_GROUP_SIZE 个合并为一个
        """
        merged = []
        
        for i in range(0, len(items), self.MERGE_GROUP_SIZE):
            group = items[i:i + self.MERGE_GROUP_SIZE]
            if not group:
                continue
            
            # 合并页码范围
            start = group[0]["start"]
            end = group[-1]["end"]
            
            # 合并摘要
            summaries = [item["summary"] for item in group if item.get("summary")]
            merged_summary = " ".join(summaries)
            
            # 合并事件（每组只取前几个）
            merged_events = []
            for item in group:
                merged_events.extend(item.get("events", [])[:2])
            
            merged.append({
                "start": start,
                "end": end,
                "summary": merged_summary,
                "events": merged_events[:self.MAX_EVENTS_PER_BATCH]
            })
        
        return merged
    
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
